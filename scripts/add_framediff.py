"""Add the frame-difference descriptor to existing v1 feature caches (decode-only pass, same 10 Hz grid and
640-px resize as the extractor, so values are identical to a fresh light-mode extraction).
Usage: python scripts/add_framediff.py [--workers 4]
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.data.dataset_loader import index_videos  # noqa: E402
from src.features.frame_diff import FrameDiff  # noqa: E402
from src.preprocessing.feature_cache import load_features, save_features  # noqa: E402
from src.preprocessing.frame_extractor import resize_width  # noqa: E402
from src.preprocessing.frame_sampler import sample_on_grid  # noqa: E402
from src.preprocessing.video_reader import iter_frames  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402


def work(args):
    cache, video = args
    f = load_features(cache)
    if f.diff is not None:
        return f.video_id, "skip"
    fd = FrameDiff((160, 90))
    d = [fd(resize_width(fr, 640)) for _, _, _, fr in sample_on_grid(iter_frames(video), f.fps)]
    d = np.stack(d)
    if len(d) < len(f.times):
        return f.video_id, f"length mismatch {len(d)} vs {len(f.times)}"
    f.diff = d[: len(f.times)]
    save_features(f, cache)
    return f.video_id, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cfg = load_config()
    vids = index_videos(resolve(cfg["paths"]["raw_dir"]))
    caches = sorted((resolve(cfg["paths"]["features_dir"]) / "clean").glob("*.npz"))
    jobs = [(c, vids[c.stem]) for c in caches if c.stem in vids]
    bad = []
    with Pool(a.workers) as p:
        for i, (vid, st) in enumerate(p.imap_unordered(work, jobs), 1):
            if st not in ("ok", "skip"):
                bad.append((vid, st))
            if i % 50 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    print("done", len(jobs), "problems:", bad)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
