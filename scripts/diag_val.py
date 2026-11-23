"""Validation-only streaming diagnostics for training-design choices (test split not loaded).

temporal_auc : AUC separating p in the last 2 s before the event from p earlier than 3 s before the annotated alert,
               within positive videos (does the score rise *when* it should?)
video_auc    : AUC of per-video peak score: positives max p in [event-2, event-0.3] vs negatives max p over the video
window_mAP   : Nexar-protocol mAP on validation windows
Usage: python scripts/diag_val.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.data.dataset_loader import load_dataset  # noqa: E402
from src.evaluation.metrics import nexar_map  # noqa: E402
from src.features.motion_features import MOTION_FEATURES  # noqa: E402
from src.prediction.predictor import EnsemblePredictor  # noqa: E402
from src.prediction.trainer import train_model  # noqa: E402
from src.prediction.windows import Standardizer, eval_samples, materialize  # noqa: E402
from src.preprocessing.feature_cache import cache_path, load_features  # noqa: E402
from src.uncertainty.mc_dropout import sample_logits, summarize  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.io import load_json, save_json  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

log = get_logger("diag_val", resolve("experiments/logs/diag_val.log"))


def streaming_diag(tl: dict, recs: dict, W: float) -> dict:
    near, far, vpos, vneg = [], [], [], []
    for vid, t in tl.items():
        r, ts, p = recs[vid], t["times"], t["p_mean"]
        ok = ts >= W - 1e-6
        ts, p = ts[ok], p[ok]
        if r.label == 1:
            m_near = (ts >= r.time_of_event - 2.0) & (ts < r.time_of_event - 0.3)
            m_far = ts < r.time_of_alert - 3.0
            near += p[m_near].tolist()
            far += p[m_far].tolist()
            if m_near.any():
                vpos.append(p[m_near].max())
        else:
            vneg.append(p.max())
    y = [1] * len(near) + [0] * len(far)
    yv = [1] * len(vpos) + [0] * len(vneg)
    return {"temporal_auc": float(roc_auc_score(y, near + far)), "video_auc": float(roc_auc_score(yv, vpos + vneg))}


def main() -> int:
    base = load_config()
    base["model"].update(lr=3e-4, weight_decay=1e-2, in_dropout=0.3)
    sp = load_json(resolve(base["paths"]["splits"]))
    recs = {r.video_id: r for r in load_dataset(resolve(base["paths"]["annotations"]), resolve(base["paths"]["raw_dir"]))}
    R = {k: [recs[v] for v in sp[k]] for k in ("train", "val")}
    F = {r.video_id: load_features(cache_path(resolve(base["paths"]["features_dir"]), r.video_id)) for k in R for r in R[k]}
    sc = Standardizer().fit([F[r.video_id] for r in R["train"]])
    out = {}
    for W in (2.0, 3.0):
        ev = eval_samples(R["val"], F, base, 1000)
        V, M, Y = materialize(ev, F, W, sc)
        G = np.array([s.group for s in ev])
        for hn in (None, 0.0, 1.5, 3.0):
            for vis, mot, tag in ((True, True, "vis+mot"), (False, True, "mot")):
                cfg = copy.deepcopy(base)
                cfg["windows"]["hard_negative_margin_s"] = hn
                mc = cfg["model"]
                hp = dict(kind="gru", use_visual=vis, use_motion=mot, d_visual=512, d_motion=len(MOTION_FEATURES),
                          visual_proj=mc["visual_proj"], motion_proj=mc["motion_proj"], hidden=mc["hidden"],
                          dropout=mc["dropout"], in_dropout=mc["in_dropout"])
                rows = []
                for seed in (42, 43):
                    m, _ = train_model(R["train"], (V, M, Y), F, sc, cfg, hp, seed, W, "cuda")
                    p = summarize(sample_logits([m], V, M, "cuda", 1))["p_mean"]
                    pr = EnsemblePredictor([m], sc, {"window_s": W}, "cuda")
                    tl = {r.video_id: pr.predict_video(F[r.video_id], 2) for r in R["val"]}
                    rows.append({"window_mAP": nexar_map(Y, p, G, base["windows"]["eval_horizons_s"])["mAP"],
                                 **streaming_diag(tl, recs, W)})
                agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
                key = f"W{W}_hn{hn}_{tag}"
                out[key] = agg
                log.info("%-22s mAP=%.3f temporalAUC=%.3f videoAUC=%.3f", key, agg["window_mAP"], agg["temporal_auc"], agg["video_auc"])
    save_json(out, resolve("experiments/latest/val_streaming_diagnostics.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
