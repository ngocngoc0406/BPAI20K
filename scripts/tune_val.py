"""Validation-only design study (the test split is NOT loaded).

Compares training-set design choices for the GRU visual+motion model on the validation videos:
window-level Nexar mAP and streaming early-warning metrics (detection rate, TTA, FA/h).
Output: experiments/latest/val_design_study.json

Usage:  python scripts/tune_val.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.data.dataset_loader import load_dataset  # noqa: E402
from src.evaluation.early_warning import evaluate_early_warning  # noqa: E402
from src.evaluation.metrics import best_f1_threshold, nexar_map  # noqa: E402
from src.features.motion_features import MOTION_FEATURES  # noqa: E402
from src.prediction.predictor import EnsemblePredictor  # noqa: E402
from src.prediction.trainer import train_model  # noqa: E402
from src.prediction.windows import Standardizer, eval_samples, materialize  # noqa: E402
from src.preprocessing.feature_cache import cache_path, load_features  # noqa: E402
from src.uncertainty.mc_dropout import sample_logits, summarize  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.io import load_json, save_json  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

log = get_logger("tune_val", resolve("experiments/logs/tune_val.log"))

NOHN = {"windows": {"hard_negative_margin_s": None}}
REG = {"model": {"lr": 3e-4, "weight_decay": 1e-2, "in_dropout": 0.3}}
VARIANTS = {
    "base_no_hardneg": NOHN,
    "hardneg3": {},
    "reg_no_hardneg": {**NOHN, **REG},
    "reg_hardneg3": REG,
    "reg_hardneg3_small": {"model": {**REG["model"], "visual_proj": 16, "motion_proj": 16, "hidden": 32}},
    "reg_hardneg3_indrop0.5": {"model": {**REG["model"], "in_dropout": 0.5}},
}


def merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def main() -> int:
    base = load_config()
    dev = get_device()
    splits = load_json(resolve(base["paths"]["splits"]))
    recs = {r.video_id: r for r in load_dataset(resolve(base["paths"]["annotations"]), resolve(base["paths"]["raw_dir"]))}
    fdir = resolve(base["paths"]["features_dir"])
    R = {sp: [recs[v] for v in splits[sp] if v in recs] for sp in ("train", "val")}  # test deliberately not loaded
    feats = {r.video_id: load_features(cache_path(fdir, r.video_id)) for sp in R for r in R[sp]}
    scaler = Standardizer().fit([feats[r.video_id] for r in R["train"]])
    ev = eval_samples(R["val"], feats, base, seed=1000)
    out = {}
    for W in (2.0, 3.0):
        V, M, Y = materialize(ev, feats, W, scaler)
        G = np.array([s.group for s in ev])
        for name, ov in VARIANTS.items():
            cfg = merge(base, ov)
            mc = cfg["model"]
            hp = dict(kind="gru", use_visual=True, use_motion=True, d_visual=512, d_motion=len(MOTION_FEATURES),
                      visual_proj=mc["visual_proj"], motion_proj=mc["motion_proj"], hidden=mc["hidden"], dropout=mc["dropout"], in_dropout=mc.get("in_dropout", 0.0))
            rows = []
            for seed in (42, 43):
                m, info = train_model(R["train"], (V, M, Y), feats, scaler, cfg, hp, seed, W, dev)
                p = summarize(sample_logits([m], V, M, dev, 1))["p_mean"]
                thr = best_f1_threshold(Y, p)
                pred = EnsemblePredictor([m], scaler, {"window_s": W, "temperature": 1.0, "mc_samples": 1}, dev)
                tl = {r.video_id: pred.predict_video(feats[r.video_id], 2) for r in R["val"]}
                ew = evaluate_early_warning({v: {"times": t["times"], "score": t["p_mean"]} for v, t in tl.items()},
                                            {r.video_id: r for r in R["val"]}, thr, 3)
                rows.append({"mAP": nexar_map(Y, p, G, base["windows"]["eval_horizons_s"])["mAP"], "det": ew["detection_rate"],
                             "tta": ew["mean_tta_s"], "fa_h": ew["false_alarms_per_hour"], "best_epoch": info["best_epoch"]})
            agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            out[f"W{W}_{name}"] = {"mean": agg, "rows": rows}
            log.info("W=%.0f %-24s val mAP=%.3f det=%.2f TTA=%.1fs FA/h=%.0f", W, name, agg["mAP"], agg["det"], agg["tta"], agg["fa_h"])
    save_json(out, resolve("experiments/latest/val_design_study.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
