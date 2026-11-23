"""XAI evaluation of the v4 research model on held-out warning events (no human study available: stated as pending).

Metrics
  shap_local_accuracy       max |sum(phi) + phi0 - logit| (exactness of Tree SHAP)
  attribution_stability     mean Spearman rank correlation of |SHAP| between the 5 ensemble members
  deletion_fidelity         logit drop when the top-10 positive SHAP features are set to the dev median vs 10 random ones
  graph_evidence_validity   share of reasoning-chain steps whose node / edge / timestamp exist in the snapshot
  trace_integrity           share of sealed traces whose hash and references verify
  explanation_completeness  share of warning events with >= 1 graph chain AND >= 1 SHAP factor AND a reliability statement
  counterfactual_validity   share of counterfactuals passing identity reproduction, box validity and step-change checks
  counterfactual_effect     mean change of risk when closing speed is halved / TTC doubled / cut-in removed (direction is
                            reported, not assumed)
Usage: python -m safedrive_v4.evaluation.xai_eval [--max-videos 60]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from safedrive_v4.config import load_config, resolve  # noqa: E402
from safedrive_v4.explainability import counterfactual as cf  # noqa: E402
from safedrive_v4.explainability import graph_explain, shap_tree  # noqa: E402
from safedrive_v4.features.graph_features import video_channels, window_matrix  # noqa: E402
from safedrive_v4.pipeline import ResearchModel, V3ReadOnlyModel, V4Pipeline  # noqa: E402
from src.features.window_features import video_window_matrix  # noqa: E402
from src.preprocessing.feature_cache import load_features  # noqa: E402

_MODEL = None
_KIND = "a3"


def model() -> ResearchModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = V3ReadOnlyModel() if _KIND == "a3" else ResearchModel()
    return _MODEL


def full_matrix(feat, t_end: float, cfg: dict, ch=None) -> tuple[np.ndarray, dict]:
    m = model()
    v3g = [g for g in m.groups if not g.startswith(("sg", "kg"))]
    if ch is None:
        SG, KG, _, snaps, res = video_channels(feat, cfg, keep=True)
    else:
        SG, KG, snaps, res = ch
    t = np.array([t_end])
    parts = [video_window_matrix(feat, t, m.meta["window_s"], v3g)]
    for g in m.groups:
        if g.startswith(("sg", "kg")):
            parts.append(window_matrix(feat.times, SG if g.startswith("sg") else KG, t, m.meta["window_s"], feat.fps))
    step = int(np.searchsorted(feat.times, t_end + 1e-4) - 1)
    r = res[step]
    summary = {rid: round(v, 3) for rid, v in r.per_rule.items() if v > 0}
    summary["risk_evidence"] = round(r.risk_evidence, 3)
    return np.concatenate(parts, 1).astype(np.float32), summary


def predict_fn(cfg):
    def f(feat, t_end):
        X, summ = full_matrix(feat, t_end, cfg)
        return float(model().predict(X)["p_mean"][0]), summ
    return f


def counterfactual_suite(video_id: str, t_end: float, track_ids: list[int], cfg: dict | None = None) -> list[dict]:
    cfg = cfg or load_config()
    feat = load_features(resolve(f"data/processed/features/clean/{video_id}.npz"))
    pf = predict_fn(cfg)
    rows = []
    for tid in track_ids:
        for kind, a in (("closing_speed", 0.5), ("higher_ttc", 2.0), ("no_cut_in", 0.0)):
            r = cf.run(pf, feat, tid, t_end, kind, 2.0, a)
            rows.append({"track_id": tid, "what_if": {"closing_speed": "closing speed halved", "higher_ttc": "image TTC doubled",
                                                      "no_cut_in": "no lateral move (stays in its lane)"}[kind],
                         "p_factual": round(r.p_factual, 3), "p_counterfactual": round(r.p_counterfactual, 3), "delta": round(r.delta, 3),
                         "valid": r.validity["valid"], "rule_changes": str(r.changed_findings)[:200],
                         "clipped_rows": r.validity.get("clipped_rows", 0)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-videos", type=int, default=60)
    ap.add_argument("--model", default="a3", choices=["a3", "a6"], help="a3 = frozen v3 model + v4 explanations; a6 = v4 research model")
    a = ap.parse_args()
    global _KIND, _MODEL
    _KIND, _MODEL = a.model, None
    from src.utils.io import load_json
    cfg = load_config()
    split = load_json(resolve("data/processed/splits_full.json"))
    held = sorted(set(split["test"]) | set(split.get("test_ext", [])))[: a.max_videos]
    pipe = V4Pipeline(model(), cfg)
    med = model().median
    acc, stab, fid_top, fid_rnd, fid_frac, gv, gtot, integ, compl, cfs = [], [], [], [], [], 0, 0, [], [], []
    lat = []
    n_events = 0
    from collections import Counter
    from src.data.dataset_loader import load_dataset
    from src.utils.config import load_config as v3cfg
    c3 = v3cfg()
    recs = {r.video_id: r for r in load_dataset(resolve(c3["paths"]["annotations"]), resolve(c3["paths"]["raw_dir"]))}
    lv_pos, lv_neg, high_lowrel = Counter(), Counter(), 0
    t0 = time.perf_counter()
    for vid in held:
        feat = load_features(resolve(f"data/processed/features/clean/{vid}.npz"))
        out = pipe.process_video(feat, explain=True)
        lat.append(out["latency"])
        r_ = recs[vid]
        for row in out["timeline"]:
            if r_.label == 1 and r_.time_of_event is not None:
                if row["t"] < r_.time_of_event:
                    lv_pos[row["level"]] += 1
            elif r_.label == 0:
                lv_neg[row["level"]] += 1
            high_lowrel += int(row["level"] == "HIGH" and row["reliability"] < pipe.policy.rel_min)
        if not out["traces"]:
            continue
        SG, KG, _, snaps, res = video_channels(feat, cfg, keep=True)
        for tr in out["traces"][:3]:
            n_events += 1
            integ.append(tr.verify()["ok"])
            X, _ = full_matrix(feat, tr.timestamp, cfg, (SG, KG, snaps, res))
            ex = shap_tree.explain(model().members, X)
            acc.append(ex["local_accuracy_error"])
            stab.append(shap_tree.member_rank_stability(ex["phi_members"]))
            fd = shap_tree.deletion_fidelity(model().members, X, ex["phi"], med, 10)
            fid_top.append(fd["mean_logit_drop_top"]), fid_rnd.append(fd["mean_logit_drop_random"]), fid_frac.append(fd["fraction_top_beats_random"])
            snap, r = snaps[tr.step], res[tr.step]
            chs = graph_explain.chains(snap, r)
            for c in chs:
                v = graph_explain.validate_chain(c, snap)
                gtot += len(c["graph_steps"])
                gv += len(c["graph_steps"]) - len(v["invalid_steps"])
            compl.append(bool(chs) and bool(tr.shap) and bool(tr.uncertainty.get("perception_reliability")))
            if tr.track_ids and len(cfs) < 60:
                for row in counterfactual_suite(vid, tr.timestamp, tr.track_ids[:1], cfg):
                    cfs.append(row)
    agg = lambda xs: float(np.mean(xs)) if xs else None  # noqa: E731
    lat_total = [l_["total"]["mean_ms"] for l_ in lat if "total" in l_]
    res_ = {"videos": len(held), "warning_events_explained": n_events, "wall_s": round(time.perf_counter() - t0, 1),
            "shap_local_accuracy_max_error": float(np.max(acc)) if acc else None,
            "attribution_stability_spearman": agg(stab),
            "deletion_fidelity": {"top10_logit_drop": agg(fid_top), "random10_logit_drop": agg(fid_rnd), "fraction_top_beats_random": agg(fid_frac)},
            "graph_evidence_validity": gv / gtot if gtot else None, "graph_steps_checked": gtot,
            "trace_integrity": agg(integ), "explanation_completeness": agg(compl),
            "counterfactual_validity": agg([r["valid"] for r in cfs]), "counterfactuals": len(cfs),
            "counterfactual_mean_delta": {k: agg([r["delta"] for r in cfs if r["what_if"] == k]) for k in
                                          ("closing speed halved", "image TTC doubled", "no lateral move (stays in its lane)")},
            "counterfactual_share_lowering_risk": {k: agg([r["delta"] < 0 for r in cfs if r["what_if"] == k]) for k in
                                                   ("closing speed halved", "image TTC doubled", "no lateral move (stays in its lane)")},
            "pipeline_latency_ms_mean": {k: agg([l_[k]["mean_ms"] for l_ in lat if k in l_]) for k in ("graph", "reasoning", "features", "predict", "shap", "total")},
            "pipeline_latency_total_p95_ms": float(np.percentile(lat_total, 95)) if lat_total else None,
            "human_evaluation": "pending: needs the trained annotators (no human rating was collected)",
            "model": model().meta.get("version"),
            "warning_levels_positive_videos_before_event": {k: v / max(sum(lv_pos.values()), 1) for k, v in lv_pos.items()},
            "warning_levels_normal_videos": {k: v / max(sum(lv_neg.values()), 1) for k, v in lv_neg.items()},
            "high_with_low_reliability_steps": high_lowrel}
    out = resolve(f"experiments/v4/xai_eval_{a.model}.json")
    out.write_text(json.dumps({"summary": res_, "counterfactual_rows": cfs}, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res_, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
