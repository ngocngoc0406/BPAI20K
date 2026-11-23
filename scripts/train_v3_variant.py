"""Train a deployable bundle for a given v3 feature/label config exactly like run_ew_study.py's final stage
(OOF dev streams -> Platt + FA-budgeted thresholds; bagged LightGBM x5 on all 1,276 dev videos).
Used for the optional camera-shake-robust variant 'v3-stabilized' (ego(H) + homography-compensated interactions + dense negatives),
which passed the primary dev rule in Stage A and was favoured under camera shake in the dev probe (shake_dev_probe.json).
Usage: python scripts/train_v3_variant.py --groups motion,diff,ego_h,rel_h --labels "L7 dense negatives (40/video)" --out models/safedrive_v3_stabilized
"""
from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from run_ew_study import Study  # noqa: E402

from src.evaluation.streaming import operating_points  # noqa: E402
from src.features.motion_features import MOTION_FEATURES  # noqa: E402
from src.features.window_features import build_matrix, channel_names  # noqa: E402
from src.prediction.light_models import BaggedEnsemble, Calibrator  # noqa: E402
from src.prediction.tabular_predictor import TabularPredictor  # noqa: E402
from src.prediction.windows import train_samples  # noqa: E402
from src.risk.risk_engine import RiskConfig, cue_reference  # noqa: E402
from src.utils.config import resolve  # noqa: E402
from src.utils.io import save_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--W", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--description", default="")
    a = ap.parse_args()
    groups = a.groups.split(",")
    st = Study("full", False)
    oof = st.run(groups, a.W, a.labels)
    cal = Calibrator("platt").fit(oof["p"], oof["y"])
    streams = {v: {"times": s["times"], "score": cal(s["score"])} for v, s in oof["streams"].items()}
    ops = operating_points(streams, st.recs, (5, 10, 20), 3, a.W, 1.0)
    cfg = st.cfg_for(a.labels)
    tr = sum((train_samples([st.recs[v] for v in st.dev], st.feats, cfg, np.random.default_rng(900)) for _ in range(3)), [])
    X, y = build_matrix(tr, st.feats, a.W, groups)
    ens = BaggedEnsemble("lgbm", 5, seed=300).fit(X, y, np.array([s.video_id for s in tr]))
    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "model.pkl", "wb") as fh:
        pickle.dump({"members": ens.members, "calibrator": cal, "pca": None, "col_median": np.median(X, 0).astype(np.float32)}, fh)
    rc = RiskConfig(consecutive_steps=3, level_signal="probability", cue_reference=cue_reference(np.concatenate([st.feats[v].motion for v in st.dev])))
    rc.warning, rc.high, rc.caution = (float(ops[f"@{b}FA"]["threshold"]) for b in (10, 5, 20))
    rc.high = max(rc.high, rc.warning)
    rc.caution = min(rc.caution, rc.warning)
    meta = {"type": "tabular", "feature_version": 1, "description": a.description or f"SafeDrive v3 variant: LightGBM x5 on {'+'.join(groups)}", "window_s": a.W,
            "warmup_s": a.W, "groups": groups, "classifier": "lgbm", "n_members": 5, "calibration": "platt", "threshold_f1_val": 0.5, "fps": 10.0,
            "uses_cnn": False, "feature_names": channel_names(groups), "motion_features": MOTION_FEATURES, "trained": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "labels": a.labels, "oof_dev": {"mAP": oof["oof_mAP"], **{f"recall@{b}": ops[f"@{b}FA"]["recall"] for b in (5, 10, 20)}},
            "risk": {**rc.to_dict(), "level_rule": "WARNING @10 FA/h, HIGH @5 FA/h, CAUTION @20 FA/h on OOF dev streams"}}
    save_json(meta, out / "bundle.json")
    p = TabularPredictor.load(out)
    rng = np.random.default_rng(0)
    mis = [p.predict_video(st.feats[v], 2)["mutual_info"] for v in rng.choice(st.dev, 200, replace=False)]
    meta["risk"]["mi_threshold"] = float(np.quantile(np.concatenate(mis), 0.9))
    save_json(meta, out / "bundle.json")
    print("saved", out, meta["oof_dev"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
