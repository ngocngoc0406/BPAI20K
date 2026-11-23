"""Precompute (a) uncertainty-aware KG channels (kgu) on clean tracks and (b) a perception-error version of every video for H2.

Perception-error model (track level, deterministic per video: seed = hash(video_id) + NOISE_SEED), applied to the TRACKER output:
  dropout      each detection removed with p = 0.15                      (missed detections)
  jitter       box corners + N(0, (0.03 * box size)^2)                    (localisation noise)
  id switch    each track switches to a fresh id with p = 0.02 per step   (association errors)
  false pos.   with p = 0.05 per step a spurious box (1-2 steps, conf 0.3-0.5, random place) (ghost detections)
Under noise, v3 object-motion features are RECOMPUTED from the noisy tracks (and relational cues re-derived); pixel channels
(optical flow, frame difference, brightness) are unchanged because perception errors do not change the image.
Output: <v4_graph dir>/u1/clean/<vid>.npz (kgu) and <v4_graph dir>/u1/noisy/<vid>.npz (tracks, motion, sg, kg, kgu).
Usage: python -m safedrive_v4.evaluation.precompute_u [--workers 4]
"""
from __future__ import annotations

import argparse
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

NOISE = {"dropout": 0.15, "jitter": 0.03, "id_switch": 0.02, "false_pos": 0.05}
NOISE_SEED = 20261123
TAG = "u1"


def perturb(feat, noise: dict = NOISE, seed: int | None = None):
    import copy
    rng = np.random.default_rng(seed if seed is not None else (zlib.crc32(feat.video_id.encode()) + NOISE_SEED))
    tr = feat.tracks.copy()
    keep = rng.random(len(tr)) >= noise["dropout"]
    tr = tr[keep]
    wh = np.c_[tr[:, 6] - tr[:, 4], tr[:, 7] - tr[:, 5]]
    tr[:, 4:8] += rng.normal(0, noise["jitter"], (len(tr), 4)) * np.c_[wh, wh]
    tr[:, 4:8] = np.clip(tr[:, 4:8], 0, 1)
    bad = (tr[:, 6] - tr[:, 4] < 1e-3) | (tr[:, 7] - tr[:, 5] < 1e-3)
    tr = tr[~bad]
    next_id = int(tr[:, 1].max()) + 1000 if len(tr) else 1000
    remap: dict[int, int] = {}
    for i in np.argsort(tr[:, 0], kind="stable"):
        tid = int(tr[i, 1])
        if tid < 0:
            continue
        if rng.random() < noise["id_switch"]:
            remap[tid] = next_id
            next_id += 1
        tr[i, 1] = remap.get(tid, tid)
    ghosts = []
    for s in range(len(feat.times)):
        if rng.random() < noise["false_pos"]:
            cx, cy, sz = rng.uniform(0.15, 0.85), rng.uniform(0.4, 0.8), rng.uniform(0.03, 0.12)
            gid = next_id
            next_id += 1
            for k in range(int(rng.integers(1, 3))):
                if s + k < len(feat.times):
                    ghosts.append([s + k, gid, 2, rng.uniform(0.3, 0.5), cx - sz / 2, cy - sz / 2, cx + sz / 2, cy + sz / 2])
    if ghosts:
        tr = np.concatenate([tr, np.array(ghosts, np.float32)])
    tr = tr[np.lexsort((tr[:, 1], tr[:, 0]))].astype(np.float32)
    f2 = copy.copy(feat)
    f2.tracks = tr
    f2.derived = {}
    from safedrive_v4.explainability.counterfactual import recompute_motion
    f2.motion = recompute_motion(f2)
    return f2


def kgu_channels(feat, cfg):
    from safedrive_v4.features.graph_features import kg_vector, sg_vector
    from safedrive_v4.reasoning.uncertain_engine import KGU_FEATURES, UncertainReasoningEngine, kgu_vector
    from safedrive_v4.scene_graph.builder import SceneGraphBuilder, video_steps
    b, e = SceneGraphBuilder(cfg, feat.fps), UncertainReasoningEngine(cfg)
    T = len(feat.times)
    SG, KG, KGU = [], [], np.zeros((T, len(KGU_FEATURES)), np.float32)
    for s, t, d, M, sc in video_steps(feat):
        snap = b.update(s, t, d, M, sc)
        r = e.evaluate(snap)
        SG.append(sg_vector(snap)), KG.append(kg_vector(r.base))
        KGU[s] = kgu_vector(r)
    return np.stack(SG), np.stack(KG), KGU


def base_dir():
    from safedrive_v4.config import load_config
    from safedrive_v4.evaluation.precompute import out_dir
    return out_dir(load_config()) / TAG


def _one(vid: str):
    from safedrive_v4.config import load_config
    from src.preprocessing.feature_cache import load_features
    cfg = load_config()
    d = base_dir()
    pc, pn = d / "clean" / f"{vid}.npz", d / "noisy" / f"{vid}.npz"
    f = load_features(ROOT / "data/processed/features/clean" / f"{vid}.npz")
    if not pc.exists():
        _, _, KGU = kgu_channels(f, cfg)
        np.savez_compressed(pc, kgu=KGU)
    if not pn.exists():
        fn = perturb(f)
        SG, KG, KGU = kgu_channels(fn, cfg)
        np.savez_compressed(pn, tracks=fn.tracks, motion=fn.motion.astype(np.float32), sg=SG, kg=KG, kgu=KGU)
    return vid


def load_u(vid: str, which: str = "clean") -> dict:
    with np.load(base_dir() / which / f"{vid}.npz") as z:
        return {k: z[k] for k in z.files}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    d = base_dir()
    (d / "clean").mkdir(parents=True, exist_ok=True)
    (d / "noisy").mkdir(parents=True, exist_ok=True)
    vids = sorted(p.stem for p in (ROOT / "data/processed/features/clean").glob("*.npz"))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(_one, vids, chunksize=8)):
            if i % 250 == 0:
                print(i, f"{time.perf_counter() - t0:.0f}s", flush=True)
    (d / "noise_model.txt").write_text(f"{NOISE} seed {NOISE_SEED}\n", encoding="utf-8")
    print("done", len(vids), f"{time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
