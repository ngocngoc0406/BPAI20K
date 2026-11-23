"""v4 ablation study on the v3 protocol (read-only reuse of scripts/run_ew_study.Study).

Protocol (unchanged from v3, declared before running):
  * dev = 1,276 videos, 5 video-level folds (seed 13), OOF windows + OOF per-video streams; held-out = 224 videos (test 60 +
    test_ext 164) evaluated ONCE per reported configuration, never used for any choice;
  * labels L7 (positives end 0.3-2.0 s before the event; 40 dense negatives / video), W = 1 s, k = 3, no EMA;
  * primary metric S = mean recall at 5 / 10 / 20 false alarms per hour (OOF streams); secondary: window mAP (Nexar);
  * a configuration is called BETTER than A0 only if the paired video-bootstrap 90 % CI of dS > 0, or S is non-inferior
    (CI lower bound > -0.02) AND the paired dmAP 90 % CI > 0 (v3 acceptance rule). Otherwise: 'no evidence of improvement'.
Usage: python -m safedrive_v4.evaluation.ablation [--quick] [--configs A0,A1,...] [--heldout]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from run_ew_study import Study  # noqa: E402  (v3 harness, imported read-only)
from safedrive_v4 import FEATURE_SCHEMA, __version__  # noqa: E402
from safedrive_v4.config import config_hash, load_config, resolve  # noqa: E402
from safedrive_v4.evaluation.precompute import load_channels  # noqa: E402
from safedrive_v4.features.graph_features import KG_FEATURES, SG_FEATURES, channel_names as v4_names, window_matrix  # noqa: E402
from src.evaluation.metrics import best_f1_threshold, classification_metrics, nexar_map  # noqa: E402
from src.evaluation.streaming import evaluate_at, operating_points  # noqa: E402
from src.features.window_features import build_matrix, channel_names as v3_names, video_window_matrix  # noqa: E402
from src.prediction.light_models import BaggedEnsemble, Calibrator, make_model, summarize_probs  # noqa: E402
from src.prediction.windows import eval_samples, train_samples  # noqa: E402
from src.uncertainty.calibration import brier, ece  # noqa: E402

V3_GROUPS = ["motion", "diff", "rel_none"]
LABELS = "L7 dense negatives (40/video)"
CONFIGS = {
    "A0 v3 baseline": (V3_GROUPS, "lgbm"),
    "A1 v3 + scene graph": (V3_GROUPS + ["sg"], "lgbm"),
    "A2 v3 + knowledge graph": (V3_GROUPS + ["kg"], "lgbm"),
    "A4 scene graph + KG only": (["sg", "kg"], "lgbm"),
    "A5 KG reasoning only": (["kg"], "lgbm"),
    "A6 full (v3 + SG + KG)": (V3_GROUPS + ["sg", "kg"], "lgbm"),
    "B-logreg (v3 features)": (V3_GROUPS, "logreg"),
    "B-rf (v3 features)": (V3_GROUPS, "rf"),
    "B-xgb (v3 features)": (V3_GROUPS, "xgb"),
    "B-motion-only": (["motion"], "lgbm"),
    "W0.5 full, graph window 0.5 s": (V3_GROUPS + ["sg@0.5", "kg@0.5"], "lgbm"),
    "W2.0 full, graph window 2.0 s": (V3_GROUPS + ["sg@2.0", "kg@2.0"], "lgbm"),
}
SHORT = {k: k.split(" ")[0] for k in CONFIGS}


def model(kind: str, seed: int):
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features="sqrt", n_jobs=4, random_state=seed)
    return make_model(kind, seed)


class V4Study(Study):
    def __init__(self, quick: bool = False):
        super().__init__("full", quick)
        self.v4cfg = load_config()
        self.ch = {v: load_channels(v, self.v4cfg) for v in self.feats}

    def _split(self, groups: list[str]):
        v3 = [g for g in groups if not g.startswith(("sg", "kg"))]
        v4 = []
        for g in groups:
            if g.startswith(("sg", "kg")):
                name, _, w = g.partition("@")
                v4.append((name, float(w) if w else None))
        return v3, v4

    def matrix_video(self, vid: str, t_ends: np.ndarray, W: float, groups: list[str]) -> np.ndarray:
        v3, v4 = self._split(groups)
        f = self.feats[vid]
        parts = [video_window_matrix(f, t_ends, W, v3)] if v3 else []
        for name, w in v4:
            c = self.ch[vid]
            parts.append(window_matrix(f.times, c[name], t_ends, w or W, f.fps))
        return np.concatenate(parts, 1).astype(np.float32)

    def matrix(self, samples, W: float, groups: list[str]) -> tuple[np.ndarray, np.ndarray]:
        by = {}
        for i, s in enumerate(samples):
            by.setdefault(s.video_id, []).append(i)
        X = None
        for vid, ii in by.items():
            Xi = self.matrix_video(vid, np.array([samples[i].t_end for i in ii]), W, groups)
            if X is None:
                X = np.zeros((len(samples), Xi.shape[1]), np.float32)
            X[ii] = Xi
        return X, np.array([s.label for s in samples], np.float32)

    def names(self, groups: list[str]) -> list[str]:
        v3, v4 = self._split(groups)
        out = v3_names(v3) if v3 else []
        for name, w in v4:
            out += [f"{n}{'@' + str(w) if w else ''}" for n in v4_names(SG_FEATURES if name == "sg" else KG_FEATURES)]
        return out

    def run(self, groups: list[str], W: float, labels: str, model_kind: str = "lgbm") -> dict:
        key = (tuple(groups), W, labels, model_kind)
        if key in self.cache:
            return self.cache[key]
        cfg = self.cfg_for(labels)
        y, g, p, vids = [], [], [], []
        streams = {}
        t0 = time.perf_counter()
        for k in range(5):
            tr_recs = [self.recs[v] for v in self.dev if self.fold[v] != k]
            r2 = np.random.default_rng(600 + k)
            tr = sum((train_samples(tr_recs, self.feats, cfg, r2) for _ in range(3 if not self.quick else 1)), [])
            X, yt = self.matrix(tr, W, groups)
            m = model(model_kind, 42).fit(X, yt)
            ev = self.oof_ev[k]
            p.append(m.predict_proba(self.matrix(ev, W, groups)[0])[:, 1])
            y.append([s.label for s in ev])
            g.append([s.group for s in ev])
            vids += [s.video_id for s in ev]
            for v in [v for v in self.dev if self.fold[v] == k]:
                f = self.feats[v]
                idx = np.arange(0, len(f.times), self.stride)
                streams[v] = {"times": f.times[idx], "score": m.predict_proba(self.matrix_video(v, f.times[idx], W, groups))[:, 1]}
        y, g, p = np.concatenate(y).astype(float), np.concatenate(g), np.concatenate(p)
        res = {"y": y, "g": g, "p": p, "vids": np.array(vids), "streams": streams, "seconds": time.perf_counter() - t0,
               "oof_mAP": nexar_map(y, p, g, self.cfg["windows"]["eval_horizons_s"])["mAP"],
               "roc_auc": classification_metrics(y, p, 0.5)["roc_auc"], "n_features": int(X.shape[1])}
        self.cache[key] = res
        return res


def crossfit_platt(p: np.ndarray, y: np.ndarray, vids: np.ndarray, folds: int = 5) -> np.ndarray:
    ug = np.unique(vids)
    rng = np.random.default_rng(0)
    rng.shuffle(ug)
    fo = {v: i % folds for i, v in enumerate(ug)}
    f = np.array([fo[v] for v in vids])
    out = np.zeros_like(p)
    for k in range(folds):
        out[f == k] = Calibrator("platt").fit(p[f != k], y[f != k])(p[f == k])
    return out


def window_metrics(y, p, g, horizons) -> dict:
    cm = classification_metrics(y, p, best_f1_threshold(y, p))
    return {"mAP": nexar_map(y, p, g, horizons)["mAP"], "roc_auc": cm["roc_auc"], "pr_auc": cm["pr_auc"], "precision": cm["precision"],
            "recall": cm["recall"], "f1": cm["f1"], "brier": brier(p, y), "ece": ece(p, y)}


def heldout(st: V4Study, groups, kind, oof) -> dict:
    """Retrain on all dev (bagged x5, Platt from OOF, thresholds from OOF streams) and evaluate the 224 held-out videos once."""
    cfg = st.cfg_for(LABELS)
    cal = Calibrator("platt").fit(oof["p"], oof["y"])
    ops = operating_points({v: {"times": s["times"], "score": cal(s["score"])} for v, s in oof["streams"].items()}, st.recs, (5, 10, 20), 3, 1.0, 1.0)
    tr = sum((train_samples([st.recs[v] for v in st.dev], st.feats, cfg, np.random.default_rng(900)) for _ in range(3)), [])
    X, yt = st.matrix(tr, 1.0, groups)
    if kind == "lgbm":
        ens = BaggedEnsemble("lgbm", 5, seed=300).fit(X, yt, np.array([s.video_id for s in tr]))
    else:
        ens = BaggedEnsemble(kind, 1, seed=300)
        ens.members = [model(kind, 300).fit(X, yt)]
    ids = st.test + st.test_ext
    # exactly as v3 'test_all' (run_ew_study.py): one deterministic draw over all 224 held-out videos with seed 7001
    ev = eval_samples([st.recs[v] for v in ids], st.feats, st.cfg, seed=7001)
    Xh = st.matrix(ev, 1.0, groups)[0]
    yh, gh = np.array([s.label for s in ev], float), np.array([s.group for s in ev])
    ph = summarize_probs(cal(ens.predict_samples(Xh)))["p_mean"]
    streams = {}
    for v in ids:
        f = st.feats[v]
        idx = np.arange(0, len(f.times), st.stride)
        streams[v] = {"times": f.times[idx], "score": summarize_probs(cal(ens.predict_samples(st.matrix_video(v, f.times[idx], 1.0, groups))))["p_mean"]}
    sr = {f"@{b}FA": evaluate_at(streams, st.recs, ops[f"@{b}FA"]["threshold"], 3, 1.0, 1.0) for b in (5, 10, 20)}
    return {"window": window_metrics(yh, ph, gh, st.cfg["windows"]["eval_horizons_s"]), "streaming": sr,
            "thresholds_from_dev": {b: ops[b]["threshold"] for b in ("@5FA", "@10FA", "@20FA")},
            "_ev": ev, "_p": ph, "_streams": streams, "_ens": ens, "_cal": cal, "_X": X, "n_videos": len(ids)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--configs", default="")
    ap.add_argument("--heldout", default="A0,A1,A2,A4,A6")
    a = ap.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = resolve("experiments/v4") / f"ablation_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    st = V4Study(a.quick)
    hz = st.cfg["windows"]["eval_horizons_s"]
    sel = [k for k in CONFIGS if not a.configs or SHORT[k] in a.configs.split(",")]
    rows, results = [], {}
    ref = None
    for name in sel:
        groups, kind = CONFIGS[name]
        t0 = time.perf_counter()
        r = st.run(groups, 1.0, LABELS, kind)
        sc = st.score(r, 1.0)
        pc = crossfit_platt(r["p"], r["y"], r["vids"])
        row = {"config": name, "groups": groups, "model": kind, "n_features": r["n_features"],
               "oof_raw": window_metrics(r["y"], r["p"], r["g"], hz), "oof_calibrated": window_metrics(r["y"], pc, r["g"], hz),
               "S": sc["S"], **{k: sc[k] for k in sc if k.startswith(("recall@", "edr@"))},
               "median_lead_s@10FA": sc["op"]["@10FA"]["median_lead_s"], "fit_eval_s": round(time.perf_counter() - t0, 1)}
        if name.startswith("A0"):
            ref = r
        elif ref is not None:
            dS = st.paired(r, ref, 1.0, 1.0)
            dM = st.paired_map(r, ref)
            better = dS["ci90"][0] > 0 or (dS["ci90"][0] > -0.02 and dM["ci95"][0] > 0)
            row.update(dS_vs_A0=dS, dmAP_vs_A0=dM, verdict="better than A0" if better else "no evidence of improvement over A0")
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("config", "S", "recall@5", "recall@10", "recall@20")} |
                         {"oof_mAP": row["oof_raw"]["mAP"], "verdict": row.get("verdict", "reference")}), flush=True)
        (out / "ablation_oof.json").write_text(json.dumps(rows, indent=1, default=float), encoding="utf-8")
    results["oof"] = rows
    held = {}
    for name in sel:
        if SHORT[name] not in a.heldout.split(","):
            continue
        groups, kind = CONFIGS[name]
        h = heldout(st, groups, kind, st.run(groups, 1.0, LABELS, kind))
        held[name] = h
        print("HELDOUT", name, json.dumps({"mAP": h["window"]["mAP"], "roc": h["window"]["roc_auc"], "ece": h["window"]["ece"],
                                          **{b: (h["streaming"][b]["recall"], h["streaming"][b]["fa_per_hour"]) for b in h["streaming"]}}), flush=True)
    if held:
        import pandas as pd

        from src.evaluation.bootstrap import bootstrap_map
        base = next((k for k in held if k.startswith("A0")), None)
        comp = {}
        for k, h in held.items():
            if k == base or base is None:
                continue
            df = pd.DataFrame({"video_id": [s.video_id for s in held[base]["_ev"]], "y": [s.label for s in held[base]["_ev"]],
                               "group": [s.group for s in held[base]["_ev"]], "a0": held[base]["_p"], "cand": h["_p"]})
            comp[k] = bootstrap_map(df, ["a0", "cand"], hz, n=1000, pairs=[("cand", "a0")])
        results["heldout"] = {k: {kk: vv for kk, vv in h.items() if not kk.startswith("_")} for k, h in held.items()}
        results["heldout_paired_mAP"] = comp
        # keep the full-v4 model (research artefact, never used by the product)
        full = next((k for k in held if k.startswith("A6")), None)
        if full:
            mdir = resolve("models/safedrive_v4_research")
            mdir.mkdir(parents=True, exist_ok=True)
            h = held[full]
            with open(mdir / "model.pkl", "wb") as fh:
                pickle.dump({"members": h["_ens"].members, "calibrator": h["_cal"], "col_median": np.median(h["_X"], 0).astype(np.float32)}, fh)
            (mdir / "bundle.json").write_text(json.dumps({
                "version": __version__, "feature_schema": FEATURE_SCHEMA, "groups": CONFIGS[full][0], "window_s": 1.0, "labels": LABELS,
                "feature_names": st.names(CONFIGS[full][0]), "v4_config_hash": config_hash(st.v4cfg), "trained": stamp,
                "thresholds_from_dev_oof": h["thresholds_from_dev"], "experiment_dir": str(out),
                "note": "research model; NOT the product model; v3 stays the deployed model"}, indent=1), encoding="utf-8")
    results["protocol"] = __doc__
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    (resolve("experiments/v4") / "latest_ablation.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
