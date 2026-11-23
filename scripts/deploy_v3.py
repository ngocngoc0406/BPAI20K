"""Deploy the v3 candidate selected by run_ew_study.py (dev-based rule) as models/safedrive_v3.

- copies models/safedrive_v3_candidate_full -> models/safedrive_v3
- sets the UNCERTAIN threshold to the 90th percentile of ensemble mutual information on dev streams (200 dev videos)
- risk levels are the FA-budgeted thresholds from OOF dev streams: CAUTION @20 FA/h, WARNING (alarm) @10 FA/h, HIGH @5 FA/h
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.prediction.tabular_predictor import TabularPredictor  # noqa: E402
from src.preprocessing.feature_cache import cache_path, load_features  # noqa: E402
from src.utils.config import resolve  # noqa: E402
from src.utils.io import load_json, save_json  # noqa: E402


def main() -> int:
    src, dst = resolve("models/safedrive_v3_candidate_full"), resolve("models/safedrive_v3")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    p = TabularPredictor.load(dst)
    sp = load_json(resolve("data/processed/splits_full.json"))
    dev = sorted(sp["train"] + sp["val"] + sp["dev_new"])
    rng = np.random.default_rng(0)
    mis = []
    for v in rng.choice(dev, 200, replace=False):
        mis.append(p.predict_video(load_features(cache_path(resolve("data/processed/features"), v)), 2)["mutual_info"])
    meta = load_json(dst / "bundle.json")
    meta["risk"]["mi_threshold"] = float(np.quantile(np.concatenate(mis), 0.90))
    meta["risk"]["mi_rule"] = "90th percentile of ensemble mutual information on 200 dev videos"
    res = load_json(resolve("experiments/latest/results_ew_full.json"))
    meta["selection"] = {"rule": res["rule"], "selected": res["selected"], "experiment": meta["experiment_dir"]}
    meta["description"] = "SafeDrive v3: LightGBM x5 on motion + frame-difference + vehicle-interaction cues (W=1 s, dense-negative training)"
    save_json(meta, dst / "bundle.json")
    print("deployed", dst, "risk:", {k: round(v, 3) if isinstance(v, float) else v for k, v in meta["risk"].items() if k != "cue_reference"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
