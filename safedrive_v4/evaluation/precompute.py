"""Precompute per-step SG / KG channels for every cached video (read-only on v3 caches).
Output: data/processed/v4_graph/<schema>_<cfghash8>/<video_id>.npz (+ summary.json with timing and config).
Usage: python -m safedrive_v4.evaluation.precompute [--workers 4] [--limit N] [--config path]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from safedrive_v4 import FEATURE_SCHEMA, SCENE_GRAPH_SCHEMA  # noqa: E402
from safedrive_v4.config import config_hash, load_config, resolve  # noqa: E402

CACHE_DIR = resolve("data/processed/features/clean")


def out_dir(cfg: dict) -> Path:
    return resolve("data/processed/v4_graph") / f"{SCENE_GRAPH_SCHEMA}_{config_hash(cfg)[:8]}"


def _one(args):
    vid, cfg_path, overrides, dest = args
    from safedrive_v4.features.graph_features import video_channels
    from src.preprocessing.feature_cache import load_features
    cfg = load_config(cfg_path, overrides)
    p = Path(dest) / f"{vid}.npz"
    if p.exists():
        return vid, None
    f = load_features(CACHE_DIR / f"{vid}.npz")
    SG, KG, tm = video_channels(f, cfg)
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, times=f.times.astype(np.float32), sg=SG, kg=KG)
    tmp.replace(p)
    return vid, tm


def load_channels(vid: str, cfg: dict) -> dict:
    with np.load(out_dir(cfg) / f"{vid}.npz") as z:
        return {"times": z["times"], "sg": z["sg"], "kg": z["kg"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    dest = out_dir(cfg)
    dest.mkdir(parents=True, exist_ok=True)
    vids = sorted(p.stem for p in CACHE_DIR.glob("*.npz"))
    if a.limit:
        vids = vids[: a.limit]
    t0 = time.perf_counter()
    timings = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (vid, tm) in enumerate(ex.map(_one, [(v, a.config, None, str(dest)) for v in vids], chunksize=8)):
            if tm:
                timings.append(tm)
            if i % 100 == 0:
                print(f"{i}/{len(vids)} {time.perf_counter() - t0:.0f}s", flush=True)
    summ = {"schema": SCENE_GRAPH_SCHEMA, "feature_schema": FEATURE_SCHEMA, "config_hash": config_hash(cfg), "config": cfg,
            "videos": len(vids), "computed_now": len(timings), "wall_s": round(time.perf_counter() - t0, 1),
            "graph_ms_per_step_mean": float(np.mean([t["graph_ms_per_step"] for t in timings])) if timings else None,
            "reasoning_ms_per_step_mean": float(np.mean([t["reasoning_ms_per_step"] for t in timings])) if timings else None}
    (dest / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k != "config"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
