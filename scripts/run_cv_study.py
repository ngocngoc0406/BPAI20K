"""Corroborating analysis (NOT used to choose the deployed model): 5-fold grouped, label-stratified cross-validation
over all 400 videos, Nexar-protocol windows per held-out fold (~1,200 out-of-fold windows in total).
Purpose: the 60-video validation set is too small to rank models reliably; CV gives a lower-variance comparison of the
selected lightweight model against heavier alternatives.
Output: experiments/latest/cv_study.json
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.dataset_loader import load_dataset  # noqa: E402
from src.evaluation.bootstrap import bootstrap_map  # noqa: E402
from src.evaluation.metrics import best_f1_threshold, classification_metrics, nexar_map  # noqa: E402
from src.features.frame_diff import FRAME_DIFF_FEATURES  # noqa: E402
from src.features.motion_features import MOTION_FEATURES  # noqa: E402
from src.features.window_features import build_matrix, fit_vabs_pca  # noqa: E402
from src.prediction.light_models import make_model  # noqa: E402
from src.prediction.trainer import train_model  # noqa: E402
from src.prediction.windows import Standardizer, eval_samples, materialize, train_samples  # noqa: E402
from src.preprocessing.feature_cache import VideoFeatures, cache_path, load_features  # noqa: E402
from src.uncertainty.mc_dropout import sample_logits, summarize  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.io import load_json, save_json  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

log = get_logger("cv_study", resolve("experiments/logs/cv_study.log"))
CONFIGS = [  # (name, W, groups, model)
    ("SELECTED logreg|motion|W3", 3.0, ["motion"], "logreg"),
    ("logreg|motion+diff|W3", 3.0, ["motion", "diff"], "logreg"),
    ("lgbm|motion|W2", 2.0, ["motion"], "lgbm"),
    ("lgbm|motion+diff|W2", 2.0, ["motion", "diff"], "lgbm"),
    ("xgb|motion|W2", 2.0, ["motion"], "xgb"),
    ("xgb|motion+diff|W2", 2.0, ["motion", "diff"], "xgb"),
    ("catboost|motion+diff|W3", 3.0, ["motion", "diff"], "catboost"),
    ("catboost|motion+diff+vdelta|W3 [CNN]", 3.0, ["motion", "diff", "vdelta"], "catboost"),
    ("xgb|motion+vabs|W2 [CNN, v1-style]", 2.0, ["motion", "vabs"], "xgb"),
]


def main() -> int:
    cfg = load_config()
    dev = get_device()
    horizons = cfg["windows"]["eval_horizons_s"]
    splits = load_json(resolve(cfg["paths"]["splits"]))
    recs = {r.video_id: r for r in load_dataset(resolve(cfg["paths"]["annotations"]), resolve(cfg["paths"]["raw_dir"]))}
    fdir = resolve(cfg["paths"]["features_dir"])
    ids = sorted(v for sp in splits.values() for v in sp if cache_path(fdir, v).exists())
    feats = {v: load_features(cache_path(fdir, v)) for v in ids}
    rng = np.random.default_rng(7)
    fold = {}
    for lab in (0, 1):
        vs = [v for v in ids if recs[v].label == lab]
        rng.shuffle(vs)
        for i, v in enumerate(vs):
            fold[v] = i % 5
    preds = {c[0]: [] for c in CONFIGS}
    preds["gru|motion+diff|W2 (light NN)"] = []
    meta_rows = []
    per_fold = {k: [] for k in preds}
    t0 = time.perf_counter()
    for k in range(5):
        tr_recs = [recs[v] for v in ids if fold[v] != k]
        te_recs = [recs[v] for v in ids if fold[v] == k]
        ev = eval_samples(te_recs, feats, cfg, seed=2000 + k)
        y = np.array([s.label for s in ev], np.float32)
        g = np.array([s.group for s in ev])
        meta_rows += [{"video_id": s.video_id, "t_end": s.t_end, "y": s.label, "group": s.group, "fold": k} for s in ev]
        r2 = np.random.default_rng(100 + k)
        tr = sum((train_samples(tr_recs, feats, cfg, r2) for _ in range(3)), [])
        pca = fit_vabs_pca([feats[r.video_id] for r in tr_recs])
        for name, W, groups, model in CONFIGS:
            Xtr, ytr = build_matrix(tr, feats, W, groups, pca)
            Xte = build_matrix(ev, feats, W, groups, pca)[0]
            p = make_model(model, 42).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
            preds[name].append(p)
            per_fold[name].append(nexar_map(y, p, g, horizons)["mAP"])
        # light temporal NN (no CNN): early stopping needs a validation set -> 15 % of the training videos of this fold
        fn = {v: VideoFeatures(f.video_id, f.times, f.diff, f.motion, f.tracks, f.meta) for v, f in feats.items()}
        inner_val = tr_recs[::7]
        inner_tr = [r for r in tr_recs if r not in inner_val]
        sc = Standardizer().fit([fn[r.video_id] for r in inner_tr])
        evv = eval_samples(inner_val, fn, cfg, seed=3000 + k)
        Vv, Mv, Yv = materialize(evv, fn, 2.0, sc)
        Vt, Mt, _ = materialize(ev, fn, 2.0, sc)
        hp = dict(kind="gru", use_visual=True, use_motion=True, d_visual=len(FRAME_DIFF_FEATURES), d_motion=len(MOTION_FEATURES),
                  visual_proj=16, motion_proj=32, hidden=32, dropout=0.3, in_dropout=0.3, visual_mode="abs")
        m, _ = train_model(inner_tr, (Vv, Mv, Yv), fn, sc, cfg, hp, 42, 2.0, dev)
        p = summarize(sample_logits([m], Vt, Mt, dev, 1))["p_mean"]
        preds["gru|motion+diff|W2 (light NN)"].append(p)
        per_fold["gru|motion+diff|W2 (light NN)"].append(nexar_map(y, p, g, horizons)["mAP"])
        log.info("fold %d done (%.1f min): %s", k, (time.perf_counter() - t0) / 60,
                 {n: round(v[-1], 3) for n, v in per_fold.items()})
    df = pd.DataFrame(meta_rows)
    for n in preds:
        df[n] = np.concatenate(preds[n])
    df.to_csv(resolve("experiments/latest/cv_oof_predictions.csv"), index=False)
    y, g = df.y.values, df.group.values
    summary = {}
    for n in preds:
        p = df[n].values
        thr = best_f1_threshold(y, p)  # pooled OOF, reported only (not used for any choice)
        summary[n] = {"oof_mAP": nexar_map(y, p, g, horizons)["mAP"], "fold_mAP_mean": float(np.mean(per_fold[n])),
                      "fold_mAP_std": float(np.std(per_fold[n])), **{k2: v for k2, v in classification_metrics(y, p, thr).items()
                                                                       if k2 in ("roc_auc", "pr_auc", "brier", "ece")}}
    sel = CONFIGS[0][0]
    boot = bootstrap_map(df.assign(video_id=df.video_id.astype(str)), list(preds), horizons, n=500,
                         pairs=[(n, sel) for n in preds if n != sel])
    out = {"n_videos": len(ids), "n_oof_windows": len(df), "folds": 5, "summary": summary, "bootstrap": boot,
           "note": "Corroborating analysis only; the deployed model was chosen on the fixed validation split before this run."}
    save_json(out, resolve("experiments/latest/cv_study.json"))
    for n, v in sorted(summary.items(), key=lambda kv: -kv[1]["oof_mAP"]):
        d = boot["paired"].get(f"{n} - {sel}")
        log.info("%-40s OOF mAP=%.3f folds=%.3f±%.3f AUC=%.3f %s", n, v["oof_mAP"], v["fold_mAP_mean"], v["fold_mAP_std"], v["roc_auc"],
                 "" if d is None else f"vs selected {d['diff']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}] P={d['p_a_better']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
