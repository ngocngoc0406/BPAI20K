"""Extract and cache per-video features for all split videos (resumable: skips cached videos).

Usage:  python scripts/extract_features.py [--limit N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset_loader import load_dataset  # noqa: E402
from src.features.extractor import FeatureExtractor  # noqa: E402
from src.preprocessing.feature_cache import cache_path, save_features  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.io import load_json, save_json  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

log = get_logger("extract", resolve("experiments/logs/extract.log"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--splits", default=None, help="splits json (default: config paths.splits)")
    ap.add_argument("--no-cnn", action="store_true", help="light mode: skip ResNet-18 (v2/v3 models do not use it)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    cfg = load_config(a.config)
    splits = load_json(resolve(a.splits or cfg["paths"]["splits"]))
    ids = set(sum(splits.values(), []))
    recs = [r for r in load_dataset(resolve(cfg["paths"]["annotations"]), resolve(cfg["paths"]["raw_dir"]))
            if r.video_id in ids]
    recs = sorted(recs, key=lambda r: r.video_id)[: a.limit][a.shard:: a.nshards]
    fx = FeatureExtractor(cfg, use_cnn=not a.no_cnn)
    fdir = resolve(cfg["paths"]["features_dir"])
    failures, speeds = {}, []
    t0 = time.perf_counter()
    for i, r in enumerate(recs, 1):
        out = cache_path(fdir, r.video_id)
        if out.exists():
            continue
        try:
            feats, _ = fx.extract(r.path, r.video_id)
            save_features(feats, out)
            speeds.append(feats.meta["extract_fps"])
        except Exception as e:  # noqa: BLE001 — log and continue with the next video
            failures[r.video_id] = repr(e)
            log.error("%s failed: %s", r.video_id, e)
        if i % 10 == 0:
            log.info("%d/%d videos, %.1f min elapsed, last %.1f fps", i, len(recs), (time.perf_counter() - t0) / 60,
                     speeds[-1] if speeds else 0)
    save_json({"device": fx.device, "n": len(recs), "failures": failures,
               "mean_extract_fps": sum(speeds) / len(speeds) if speeds else None},
              resolve(f"data/processed/extract_summary{'' if a.nshards == 1 else f'_shard{a.shard}'}.json"))
    log.info("done: %d failures", len(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
