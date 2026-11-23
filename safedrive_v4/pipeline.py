"""v4 streaming pipeline:
detections + tracks (v3 perception) -> scene graph (incremental) -> KG reasoning -> window features (v3 + SG + KG)
-> knowledge-guided risk prediction (research bundle) -> uncertainty / reliability -> warning policy -> evidence trace.

Replay mode runs on a v3 feature cache (detections, tracks, camera transforms, pixel channels already extracted); every step only
uses data up to that step. Memory is bounded: ring buffers of `buffer_steps` steps. Detailed explanations (SHAP is cheap and runs
per warning; counterfactuals are expensive and run on request) never block the warning decision; per-stage latency is measured.
"""
from __future__ import annotations

import json
import logging
import pickle
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np

from safedrive_v4 import FEATURE_SCHEMA, __version__
from safedrive_v4.config import config_hash, load_config, resolve
from safedrive_v4.event_memory.trace import EvidenceTrace, new_event_id, unknowns_for
from safedrive_v4.explainability import graph_explain, shap_tree
from safedrive_v4.features.graph_features import KG_FEATURES, SG_FEATURES, kg_vector, sg_vector, slog
from safedrive_v4.reasoning.uncertain_engine import UncertainReasoningEngine
from safedrive_v4.scene_graph.builder import SceneGraphBuilder, video_steps
from safedrive_v4.uncertainty.reliability import perception_reliability
from safedrive_v4.warning.policy import WarningPolicy

log = logging.getLogger("safedrive_v4")
MODEL_DIR = resolve("models/safedrive_v4_research")


class ResearchModel:
    """Read-only loader of the v4 research bundle (written by the ablation study)."""

    def __init__(self, model_dir: str | Path = MODEL_DIR):
        d = Path(model_dir)
        self.meta = json.loads((d / "bundle.json").read_text(encoding="utf-8"))
        with open(d / "model.pkl", "rb") as fh:   # locally produced artefact
            pl = pickle.load(fh)
        self.members, self.cal, self.median = pl["members"], pl["calibrator"], pl["col_median"]
        self.groups, self.names = self.meta["groups"], self.meta["feature_names"]
        if self.meta.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError(f"feature schema {self.meta.get('feature_schema')} != code {FEATURE_SCHEMA}")

    def predict(self, X: np.ndarray) -> dict:
        from src.prediction.light_models import summarize_probs
        ps = np.stack([m.predict_proba(X)[:, 1] for m in self.members])
        return summarize_probs(self.cal(ps))


class V3ReadOnlyModel(ResearchModel):
    """A3 configuration: the FROZEN v3 model (models/safedrive_v3, read-only) predicts; v4 only explains.
    Predictions are identical to v3's TabularPredictor (tested). Thresholds: v3 bundle (WARNING @10 FA/h, CAUTION @20 FA/h)."""

    def __init__(self, model_dir: str | Path = "models/safedrive_v3"):
        from src.features.window_features import channel_names
        d = resolve(model_dir)
        meta = json.loads((d / "bundle.json").read_text(encoding="utf-8"))
        with open(d / "model.pkl", "rb") as fh:   # read-only use of the frozen v3 artefact
            pl = pickle.load(fh)
        self.members, self.cal, self.median = pl["members"], pl["calibrator"], pl["col_median"]
        self.groups = list(meta["groups"])
        self.names = channel_names(self.groups)
        r = meta["risk"]
        self.meta = {"version": "v3 (frozen) + v4 explanations (A3)", "window_s": meta["window_s"], "trained": meta.get("trained"),
                     "thresholds_from_dev_oof": {"@20FA": r["caution"], "@10FA": r["warning"], "@5FA": r["high"]},
                     "mi_threshold": r.get("mi_threshold"), "source": str(d)}


class V4Pipeline:
    def __init__(self, model: ResearchModel | None = None, cfg: dict | None = None, buffer_steps: int = 30):
        self.cfg = cfg or load_config()
        self.model = model or ResearchModel()
        self.W = float(self.model.meta.get("window_s", 1.0))
        thr = self.model.meta.get("thresholds_from_dev_oof", {})
        wc = self.cfg["warning"]
        self.policy = WarningPolicy(thr.get("@20FA", 0.5), thr.get("@10FA", 0.7), wc["consecutive_steps"], wc["reliability_min"])
        self.buffer_steps = buffer_steps
        self.reset()

    def reset(self) -> None:
        self.builder = SceneGraphBuilder(self.cfg)
        self.engine = UncertainReasoningEngine(self.cfg)
        self.policy.reset()
        self.buf: deque = deque(maxlen=self.buffer_steps)
        self.latency: dict[str, list[float]] = {k: [] for k in ("graph", "reasoning", "features", "predict", "shap", "total")}

    # v3 window features on a bounded buffer (same code as v3; relational cues are re-derived on the buffer)
    def _v3_window(self, feat, step: int) -> np.ndarray:
        from src.features.window_features import video_window_matrix
        from src.preprocessing.feature_cache import VideoFeatures
        v3g = [g for g in self.model.groups if not g.startswith(("sg", "kg"))]
        lo = max(0, step - self.buffer_steps + 1)
        tr = feat.tracks
        m = (tr[:, 0] >= lo) & (tr[:, 0] <= step)
        sub = tr[m].copy()
        sub[:, 0] -= lo
        mini = VideoFeatures(feat.video_id, feat.times[lo:step + 1], feat.visual[lo:step + 1], feat.motion[lo:step + 1], sub,
                             feat.meta, None if feat.diff is None else feat.diff[lo:step + 1], None if feat.ego is None else feat.ego[lo:step + 1])
        return video_window_matrix(mini, np.array([feat.times[step]]), self.W, v3g)

    def _v4_window(self) -> np.ndarray:
        from src.features.window_features import _aggregate
        n = int(round(self.W * self.cfg["fps"]))
        rows = list(self.buf)[-n:]
        rows = [rows[0]] * (n - len(rows)) + rows   # left-pad like v3 window_indices
        out = []
        col = {"sg": 0, "kg": 1, "kgu": 4}
        for g in self.model.groups:
            name = g.partition("@")[0]
            if name in col:
                C = np.stack([r[col[name]] for r in rows])[None]
                out.append(_aggregate(slog(C), self.cfg["fps"]))
        return np.concatenate(out, 1) if out else np.zeros((1, 0), np.float32)

    def step(self, s: int, t: float, dets, M, scene: dict, v3_window, video_id: str, quality=None, explain: bool = True,
             prev_level: str | None = None, p_external: tuple[float, float] | None = None):
        """One 10 Hz step. v3_window: callable(step) -> (1, d) v3 window features (or None when p_external is given, e.g. live
        mode where the v3 engine already predicted). Returns (row, trace | None, snapshot, uncertain result)."""
        t0 = time.perf_counter()
        snap = self.builder.update(s, t, dets, M, scene)
        t1 = time.perf_counter()
        ures = self.engine.evaluate(snap)
        res = ures.base
        t2 = time.perf_counter()
        sgv, kgv = sg_vector(snap), kg_vector(res)
        from safedrive_v4.reasoning.uncertain_engine import kgu_vector
        self.buf.append((sgv, kgv, snap, res, kgu_vector(ures)))
        X = None
        if p_external is None:
            X = np.concatenate([v3_window(s), self._v4_window()], 1).astype(np.float32)
        t3 = time.perf_counter()
        if p_external is None:
            pr = self.model.predict(X)
            p, mi = float(pr["p_mean"][0]), float(pr["mutual_info"][0])
        else:
            p, mi = p_external
            pr = {"p_mean": np.array([p]), "mutual_info": np.array([mi]), "p_std": np.array([0.0]), "entropy": np.array([0.0])}
        t4 = time.perf_counter()
        rel = {"score": float(ures.reliability.expectation), "opinion": ures.reliability.to_dict(),
               "heuristic": perception_reliability(res.reliability, scene.get("mean_det_conf"), snap.scene.get("n_objects", 0),
                                                   float(sgv[SG_FEATURES.index("sg_mean_unc_corridor")]), snap.scene.get("n_new", 0))["score"]}
        dec = self.policy.decide(p, mi, rel["score"], None, bool(quality and quality.degraded), quality.reasons if quality else None,
                                 float(ures.temporal_risk.expectation))
        t5 = time.perf_counter()
        row = {"step": s, "t": t, "p": p, "mi": mi, "reliability": rel["score"], "level": dec.level, "cautious": dec.cautious,
               "risk_evidence": res.risk_evidence, "kg_risk_expectation": float(ures.temporal_risk.expectation),
               "kg_risk_uncertainty": float(ures.temporal_risk.u), "n_findings": len(res.findings),
               "input_reasons": quality.reasons if quality else []}
        trace = None
        if dec.level in ("MODERATE", "HIGH") and explain and prev_level != dec.level:
            ex = shap_tree.explain(self.model.members, X) if X is not None else None
            self.latency["shap"].append(1000 * (time.perf_counter() - t5))
            trace = self.make_trace(video_id, snap, res, X, pr, ex, rel, dec, ures)
            row["event_id"] = trace.event_id
        for k, a, b in (("graph", t0, t1), ("reasoning", t1, t2), ("features", t2, t3), ("predict", t3, t4), ("total", t0, t5)):
            self.latency[k].append(1000 * (b - a))
        return row, trace, snap, ures

    def process_video(self, feat, explain: bool = True, trace_dir: str | Path | None = None, every: int = 1) -> dict:
        from safedrive_v4.uncertainty.input_quality import replay_quality
        from src.features.motion_features import MOTION_FEATURES
        self.reset()
        timeline, traces = [], []
        bi = MOTION_FEATURES.index("brightness")
        frozen = 0
        for s, t, dets, M, scene in video_steps(feat):
            dm = float(np.abs(feat.diff[s]).mean()) if feat.diff is not None else None
            frozen = frozen + 1 if (dm is not None and s > 0 and dm < 1e-6) else 0
            q = replay_quality(float(feat.motion[s, bi]), dm, frozen)
            row, tr, _, _ = self.step(s, t, dets, M, scene, lambda st_: self._v3_window(feat, st_), feat.video_id, q, explain,
                                      timeline[-1]["level"] if timeline else None)
            if tr is not None:
                if trace_dir:
                    tr.save(trace_dir)
                traces.append(tr)
            timeline.append(row)
        lat = {k: {"mean_ms": float(np.mean(v)), "p95_ms": float(np.percentile(v, 95))} for k, v in self.latency.items() if v}
        return {"timeline": timeline, "traces": traces, "latency": lat}

    def make_trace(self, video_id, snap, res, X, pr, ex, rel, dec, ures=None) -> EvidenceTrace:
        ch = graph_explain.chains(snap, res)
        used_edges = {sd["edge_id"] for c in ch for sd in c["graph_steps"]}
        nodes_used = {n for c in ch for sd in c["graph_steps"] for n in (sd["node"], sd["target"])}
        edges = [e.to_dict() for e in snap.edges if e.edge_id in used_edges]
        nodes = {k: v.to_dict() for k, v in snap.nodes.items() if k in nodes_used}
        top = [] if ex is None else (
            [{"level": "group (stable across ensemble members)", **g} for g in shap_tree.group_contributions(ex["phi"][0], self.model.names)]
            + [{"level": "feature (ranking unstable across members; indicative only)", **c}
               for c in shap_tree.top_contributions(ex["phi"][0], self.model.names, X[0])])
        fop = {}
        if ures is not None:
            for i, op in ures.finding_opinions.items():
                fop[f"{res.findings[i].rule_id}:{res.findings[i].subjects[0]}"] = op.to_dict()
        return EvidenceTrace(
            event_id=new_event_id(), video_id=video_id, timestamp=snap.timestamp, step=snap.step,
            frame_refs=list(range(max(0, snap.step - int(self.W * 10) + 1), snap.step + 1)),
            track_ids=sorted({int(n.split(":")[1]) for n in nodes_used if n.startswith("obj:")}),
            nodes=nodes, edges=edges,
            findings=[{**asdict(f)} for f in res.findings if f.kind == "risk" and set(f.edge_ids) & used_edges],
            kg_paths=[c["kg_path"] for c in ch],
            prediction={"risk_probability": float(pr["p_mean"][0]), "member_std": float(pr["p_std"][0]),
                        "raw_logit_mean": None if ex is None else float(ex["raw_logit"][0]), "model": self.model.meta.get("version")},
            shap=top, uncertainty={"mutual_info": float(pr["mutual_info"][0]), "entropy": float(pr["entropy"][0]),
                                   "perception_reliability": rel, "shap_local_accuracy_error": None if ex is None else ex["local_accuracy_error"],
                                   "kg_risk_opinion_temporal": None if ures is None else ures.temporal_risk.to_dict(),
                                   "kg_risk_opinion_step": None if ures is None else ures.step_risk.to_dict(),
                                   "finding_opinions": fop,
                                   "edge_opinions": {e.edge_id: e.opinion for e in snap.edges if e.edge_id in used_edges}},
            warning=asdict(dec), unknowns=unknowns_for(snap, res),
            provenance={"v4_version": __version__, "feature_schema": FEATURE_SCHEMA, "v4_config_hash": config_hash(self.cfg)[:16],
                        "model_trained": self.model.meta.get("trained")}).seal()


def stage_summary(feat, explain: bool = False) -> dict:
    p = V4Pipeline()
    t0 = time.perf_counter()
    out = p.process_video(feat, explain=explain)
    return {"steps": len(out["timeline"]), "wall_s": time.perf_counter() - t0, "latency": out["latency"]}


def channels_for_video(feat, cfg: dict | None = None):
    """Factual SG/KG channels of a whole video (used by counterfactuals)."""
    from safedrive_v4.features.graph_features import video_channels
    return video_channels(feat, cfg or load_config())


__all__ = ["V4Pipeline", "ResearchModel", "channels_for_video", "SG_FEATURES", "KG_FEATURES"]
