"""Back-fill camera ego-motion (cache v3 'ego' array) into existing feature caches: decode-only pass on the same 10 Hz grid
and 640-px resize as the extractor, using the CACHED detection boxes for background masking (identical to a fresh extraction
except for the detector's run-to-run nondeterminism, which is not involved here).
Usage: python scripts/add_ego_features.py [--workers 6] [--tag clean]
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.data.dataset_loader import index_videos  # noqa: E402
from src.features.ego_motion import EgoMotionEstimator  # noqa: E402
from src.features.optical_flow import FlowComputer  # noqa: E402
from src.preprocessing.feature_cache import load_features, save_features  # noqa: E402
from src.preprocessing.frame_extractor import resize_width  # noqa: E402
from src.preprocessing.frame_sampler import sample_on_grid  # noqa: E402
from src.preprocessing.video_reader import iter_frames  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402


def work(args):
    cache, video = args
    f = load_features(cache)
    if f.ego is not None:
        return f.video_id, "skip"
    fl, eg = FlowComputer((160, 90)), EgoMotionEstimator()
    boxes = {int(s): f.tracks[f.tracks[:, 0] == s, 4:8] for s in np.unique(f.tracks[:, 0])} if len(f.tracks) else {}
    out = []
    start = float(f.meta.get("start_s") or 0.0)
    end = f.meta.get("end_s")
    for step, _t, _i, fr in sample_on_grid(iter_frames(video, start, end), f.fps, start):
        if step >= len(f.times):
            break
        fr = resize_width(fr, 640)
        fl(fr, 1.0 / f.fps)
        out.append(eg(fr, boxes.get(step, np.zeros((0, 4))), 1.0 / f.fps, fl.last_field, (160, 90)))
    if len(out) < len(f.times):
        return f.video_id, f"length mismatch {len(out)} vs {len(f.times)}"
    f.ego = np.stack(out)
    save_features(f, cache)
    return f.video_id, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="clean")
    a = ap.parse_args()
    cfg = load_config()
    vids = index_videos(resolve(cfg["paths"]["raw_dir"]))
    caches = sorted((resolve(cfg["paths"]["features_dir"]) / a.tag).glob("*.npz"))
    jobs = [(c, vids[c.stem]) for c in caches if c.stem in vids]
    bad = []
    with Pool(a.workers) as p:
        for i, (vid, st) in enumerate(p.imap_unordered(work, jobs), 1):
            if st not in ("ok", "skip"):
                bad.append((vid, st))
            if i % 50 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    print("done", len(jobs), "problems:", bad, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
