"""Product calibration of the CAUTION level (dev data only; test untouched; WARNING/alarm threshold unchanged).

Rule: CAUTION = the risk-score quantile at which CAUTION-or-above covers at most `max_fraction` (default 20 %) of
normal-driving time on out-of-fold streams of the 340 train+val videos, kept strictly below WARNING.
Reason: the window-recall-based CAUTION threshold put ~50-75 % of all timeline steps into CAUTION, which is unusable.
Updates models/safedrive_v2_light/bundle.json (risk.caution) and records the change in experiments/latest/results_light.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.data.dataset_loader import load_dataset  # noqa: E402
from src.features.motion_features import MOTION_FEATURES  # noqa: E402
from src.features.window_features import build_matrix, video_window_matrix  # noqa: E402
from src.prediction.light_models import BaggedEnsemble, Calibrator, summarize_probs  # noqa: E402
from src.prediction.tabular_predictor import TabularPredictor  # noqa: E402
from src.prediction.windows import train_samples  # noqa: E402
from src.preprocessing.feature_cache import cache_path, load_features  # noqa: E402
from src.risk.risk_engine import RiskConfig, RiskEngine  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.io import load_json, save_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-fraction", type=float, default=0.20)
    a = ap.parse_args()
    cfg = load_config()
    bdir = resolve("models/safedrive_v2_light")
    pred = TabularPredictor.load(bdir)
    meta = load_json(bdir / "bundle.json")
    W, groups, clf = pred.window_s, pred.groups, meta["classifier"]
    splits = load_json(resolve(cfg["paths"]["splits"]))
    recs = {r.video_id: r for r in load_dataset(resolve(cfg["paths"]["annotations"]), resolve(cfg["paths"]["raw_dir"]))}
    fdir = resolve(cfg["paths"]["features_dir"])
    dev = sorted(v for v in splits["train"] + splits["val"] if cache_path(fdir, v).exists())
    feats = {v: load_features(cache_path(fdir, v)) for v in dev}
    rng = np.random.default_rng(11)  # same fold assignment as run_light_cv_select.py
    fold = {}
    for lab in (0, 1):
        vs = [v for v in dev if recs[v].label == lab]
        rng.shuffle(vs)
        fold.update({v: i % 5 for i, v in enumerate(vs)})
    rc = RiskConfig.from_dict(meta["risk"])
    eng = RiskEngine(rc)
    li = MOTION_FEATURES.index("looming_center")
    neg_scores = []
    for k in range(5):
        tr = sum((train_samples([recs[v] for v in dev if fold[v] != k], feats, cfg, np.random.default_rng(500 + k)) for _ in range(3)), [])
        X, y = build_matrix(tr, feats, W, groups)
        ens = BaggedEnsemble(clf, len(pred.members), seed=200 + k).fit(X, y, np.array([s.video_id for s in tr]))
        for v in [v for v in dev if fold[v] == k and recs[v].label == 0]:
            f = feats[v]
            idx = np.arange(0, len(f.times), cfg["windows"]["inference_stride_steps"])
            s = summarize_probs(pred.calibrator(ens.predict_samples(video_window_matrix(f, f.times[idx], W, groups))))
            score, _ = eng.risk_scores(f.times[idx], s["p_mean"], f.motion[idx, li])
            neg_scores.append(score[f.times[idx] >= W - 1e-6])
    neg = np.concatenate(neg_scores)
    old = rc.caution
    new = float(min(np.quantile(neg, 1 - a.max_fraction), 0.95 * rc.warning))
    meta["risk"]["caution"] = new
    meta["risk"]["caution_rule"] = f"CAUTION-or-above <= {a.max_fraction:.0%} of normal-driving time on OOF dev streams"
    save_json(meta, bdir / "bundle.json")
    rp = resolve("experiments/latest/results_light.json")
    r = load_json(rp)
    r["caution_recalibration"] = {"old": old, "new": new, "max_fraction": a.max_fraction,
                                  "fraction_of_normal_time_at_or_above_old": float((neg >= old).mean()),
                                  "fraction_of_normal_time_at_or_above_new": float((neg >= new).mean()),
                                  "normal_driving_hours_oof": float(len(neg) * 0.2 / 3600)}
    r["risk_thresholds_val"]["caution"] = new
    save_json(r, rp)
    exp_dir = Path(meta["experiment_dir"])
    if exp_dir.exists():
        save_json(r, exp_dir / "results_light.json")
    print(r["caution_recalibration"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
