"""Robustness (RQ4): re-run the FULL perception pipeline on corrupted test clips and evaluate the trained bundle.

For each test video, the clip covering its evaluation windows (plus a 2 s tracker warm-up) is decoded,
corrupted with a controlled transform (see src/preprocessing/video_transform.py), re-processed
(detection, tracking, flow, CNN) and scored. 'clean' is re-extracted the same way for a fair reference.

Usage:  python scripts/run_robustness.py [--limit N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.data.dataset_loader import load_dataset  # noqa: E402
from src.evaluation.metrics import classification_metrics, nexar_map  # noqa: E402
from src.features.extractor import FeatureExtractor  # noqa: E402
from src.pipeline import load_predictor  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.prediction.windows import eval_samples  # noqa: E402
from src.preprocessing.feature_cache import cache_path, load_features, save_features  # noqa: E402
from src.preprocessing.video_transform import CORRUPTIONS, STRESS_V33, make_transform  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.io import load_json, save_json  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

log = get_logger("robustness", resolve("experiments/logs/robustness.log"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model-dir", default="models/safedrive_v1")
    ap.add_argument("--out", default="robustness.json")
    ap.add_argument("--conditions", default=None, help="comma list, e.g. clean,light_change (default: clean + all CORRUPTIONS)")
    ap.add_argument("--keep-awake", action="store_true", help="prevent system sleep while this process runs (process-scoped)")
    a = ap.parse_args()
    if a.keep_awake and sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    cfg = load_config(a.config)
    pred = load_predictor(resolve(a.model_dir), get_device())
    uses_cnn = getattr(pred, "uses_cnn", True)
    fx = FeatureExtractor(cfg, pred.device if uses_cnn else get_device(), use_cnn=uses_cnn)
    prefix = "robust" if uses_cnn else "robust_v3"  # v3 caches: no ResNet; frame-difference + ego-motion (serve v2 and v3 models)
    splits = load_json(resolve(cfg["paths"]["splits"]))
    recs = [r for r in load_dataset(resolve(cfg["paths"]["annotations"]), resolve(cfg["paths"]["raw_dir"]))
            if r.video_id in set(splits["test"])]
    fdir = resolve(cfg["paths"]["features_dir"])
    clean_full = {r.video_id: load_features(cache_path(fdir, r.video_id)) for r in recs if cache_path(fdir, r.video_id).exists()}
    recs = [r for r in recs if r.video_id in clean_full][: a.limit]
    samples = eval_samples(recs, clean_full, cfg, seed=1001)  # identical windows to run_experiments test set
    by_vid: dict[str, list] = {}
    for s in samples:
        by_vid.setdefault(s.video_id, []).append(s)
    conditions = [("clean", 0)] + [(n, sev) for n in CORRUPTIONS for sev in sorted(CORRUPTIONS[n])]
    if a.conditions:
        allc = {**CORRUPTIONS, **STRESS_V33}
        names = a.conditions.split(",")
        conditions = [(n, 0) for n in names if n == "clean"] + [(n, sev) for n in names if n != "clean" for sev in sorted(allc[n])]
    horizons = cfg["windows"]["eval_horizons_s"]
    thr = float(pred.meta["threshold_f1_val"])
    results = {"threshold_from_val": thr, "window_s": pred.window_s, "n_videos": len(by_vid), "n_windows": len(samples),
               "conditions": {}}
    for name, sev in conditions:
        tag = f"{prefix}_{name}_{sev}"
        t0 = time.perf_counter()
        ys, ps, mis, groups, fps = [], [], [], [], []
        for vid, ss in by_vid.items():
            r = next(x for x in recs if x.video_id == vid)
            t_hi = max(s.t_end for s in ss)
            t_lo = max(min(s.t_end for s in ss) - pred.window_s - 2.0, 0.0)
            cp = cache_path(fdir, vid, tag)
            if cp.exists():
                f = load_features(cp)
            else:
                seed = int(vid) if vid.isdigit() else 0
                f, _ = fx.extract(r.path, vid, start_s=t_lo, end_s=t_hi + 0.05, transform=make_transform(name, sev, seed))
                save_features(f, cp)
            fps.append(f.meta.get("extract_fps", np.nan))
            out = pred.predict_at(f, np.array([s.t_end for s in ss]))
            ys += [s.label for s in ss]
            groups += [s.group for s in ss]
            ps.append(out["p_mean"])
            mis.append(out["mutual_info"])
        y, p, mi, g = np.array(ys, float), np.concatenate(ps), np.concatenate(mis), np.array(groups)
        res = {**classification_metrics(y, p, thr), **nexar_map(y, p, g, horizons),
               "mean_mutual_info": float(mi.mean()), "mean_entropy": float(np.mean(-(p * np.log(p + 1e-7) + (1 - p) * np.log(1 - p + 1e-7)))),
               "seconds": time.perf_counter() - t0, "extract_fps_mean": float(np.nanmean(fps)) if fps else None}
        results["conditions"][f"{name}_{sev}"] = res
        log.info("%-14s sev%d mAP=%.4f F1=%.4f ECE=%.4f MI=%.4f", name, sev, res["mAP"], res["f1"], res["ece"],
                 res["mean_mutual_info"])
    results["model_dir"] = a.model_dir
    save_json(results, resolve(cfg["paths"]["experiments_dir"]) / "latest" / a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
