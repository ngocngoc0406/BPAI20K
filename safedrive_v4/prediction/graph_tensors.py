"""Per-step graph tensors for the temporal GNN (direction B).
For each step: ego scene vector (5), up to N_OBJ object nodes ordered by area (21 features each, zero-padded with a mask) and a
relation-typed weighted adjacency (R, 1 + N_OBJ, 1 + N_OBJ) over the scene-graph edges (weight = edge confidence; node 0 = ego).
Output: data/processed/v4_graph/<tag>/tensors/<video>.npz
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from safedrive_v4.scene_graph.schema import EDGE_TYPES, REGION_IDS  # noqa: E402

N_OBJ = 8
RELS = [r for r in EDGE_TYPES if r != "LOCATED_IN"]
CLS = ["Pedestrian", "Bicycle", "Car", "Motorcycle", "Bus", "Truck"]
F_OBJ = 21
F_EGO = 5


def snapshot_tensors(s) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    objs = sorted([n for n in s.nodes.values() if n.object_id is not None], key=lambda n: -n.attrs.get("area", 0))[:N_OBJ]
    X = np.zeros((N_OBJ, F_OBJ), np.float32)
    mask = np.zeros(N_OBJ, np.float32)
    idx = {"ego": 0}
    for i, n in enumerate(objs):
        idx[n.node_id] = i + 1
        mask[i] = 1
        x1, y1, x2, y2 = n.bounding_box
        v = n.velocity_estimate
        X[i, CLS.index(n.node_type)] = 1
        X[i, 6:11] = [n.confidence or 0, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
        X[i, 11:14] = [v[0], v[1], 1.0] if v is not None else [0, 0, 0]
        X[i, 14:18] = [n.scale_growth or 0.0, (n.tau_s or 10.0) / 10.0, min(n.track_age_steps, 20) / 20.0, n.uncertainty]
        if n.region in REGION_IDS:
            X[i, 18 + REGION_IDS.index(n.region)] = 1
    A = np.zeros((len(RELS), N_OBJ + 1, N_OBJ + 1), np.float32)
    for e in s.edges:
        if e.relation_type in RELS and e.source_node in idx and e.target_node in idx:
            A[RELS.index(e.relation_type), idx[e.source_node], idx[e.target_node]] = e.confidence
    sc = s.scene
    ego = np.array([sc.get("flow_mag_mean", 0), sc.get("flow_dx_median", 0), sc.get("brightness", 0), sc.get("mean_det_conf", 0),
                    sc.get("n_objects", 0) / 10.0], np.float32)
    return ego, X, mask, A


def _one(args):
    vid, dest = args
    from safedrive_v4.config import load_config
    from safedrive_v4.scene_graph.builder import build_video
    from src.preprocessing.feature_cache import load_features
    p = Path(dest) / f"{vid}.npz"
    if p.exists():
        return vid
    f = load_features(ROOT / "data/processed/features/clean" / f"{vid}.npz")
    E, X, M, A = [], [], [], []
    for s in build_video(f, load_config()):
        e, x, m, a = snapshot_tensors(s)
        E.append(e), X.append(x), M.append(m), A.append(a)
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, ego=np.stack(E), x=np.stack(X).astype(np.float16), mask=np.stack(M).astype(np.uint8),
                        adj=np.stack(A).astype(np.float16))
    tmp.replace(p)
    return vid


def tensor_dir() -> Path:
    from safedrive_v4.config import load_config
    from safedrive_v4.evaluation.precompute import out_dir
    return out_dir(load_config()) / "tensors"


def main() -> int:
    dest = tensor_dir()
    dest.mkdir(parents=True, exist_ok=True)
    vids = sorted(p.stem for p in (ROOT / "data/processed/features/clean").glob("*.npz"))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(3) as ex:
        for i, _ in enumerate(ex.map(_one, [(v, str(dest)) for v in vids], chunksize=8)):
            if i % 250 == 0:
                print(i, f"{time.perf_counter() - t0:.0f}s", flush=True)
    print("done", len(vids), f"{time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
