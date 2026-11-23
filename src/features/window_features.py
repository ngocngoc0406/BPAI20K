"""Vectorised window-level (tabular) features for lightweight classifiers.

For a causal window of W s ending at t_end (same indexing as src/prediction/windows.py), each per-step channel
is summarised by 8 temporal aggregates:
    last, mean, max, min, std, slope (least squares, 1/s), delta (last - first), trend (mean of last third - mean of first third)
Channel groups:
    motion  28 image-plane motion cues (signed-log transformed first)
    diff    20 frame-difference descriptors (no CNN)
    vdelta  2 CNN appearance-change channels: ||v_t - v_{t-1}|| and 1 - cos(v_t, v_{t-1})       (needs ResNet)
    vabs    PCA-32 of the window-mean absolute ResNet embedding (fitted on train)                (needs ResNet)
"""
from __future__ import annotations

import numpy as np

from src.features.ego_motion import EGO_F_STATS, EGO_H_STATS, EGO_LAYOUT
from src.features.frame_diff import FRAME_DIFF_FEATURES
from src.features.relational import REL_FEATURES, derive_relational
from src.features.motion_features import MOTION_FEATURES
from src.prediction.windows import window_indices
from src.preprocessing.feature_cache import VideoFeatures

# Version of every per-step feature definition used by tabular bundles (motion, frame diff, ego, relational) and of the
# window aggregates. Bump it whenever any definition changes; bundles record the version they were trained with and
# TabularPredictor refuses to load a mismatching bundle (prevents silently changing a deployed model's inputs).
FEATURE_VERSION = 1

AGGS = ["last", "mean", "max", "min", "std", "slope", "delta", "trend"]
NEEDS_CNN = {"vdelta", "vabs"}
REL_MODES = {"rel_h": "h", "rel_f": "f", "rel_none": "none"}  # ego-compensated (homography / flow) or raw relational cues


def ego_channels(f: VideoFeatures, group: str) -> np.ndarray:
    if f.ego is None:
        raise ValueError(f"{f.video_id}: cache has no ego-motion features (run scripts/add_ego_features.py)")
    return f.ego[:, EGO_LAYOUT["h_stats"] if group == "ego_h" else EGO_LAYOUT["f_stats"]]


def rel_channels(f: VideoFeatures, group: str) -> np.ndarray:
    """Derived on the fly from cached tracks + ego transforms; memoised per video."""
    key = f"rel:{group}"
    if key not in f.derived:
        mode = REL_MODES[group]
        if mode != "none" and f.ego is None:
            raise ValueError(f"{f.video_id}: cache has no ego-motion features (run scripts/add_ego_features.py)")
        f.derived[key] = derive_relational(f.times, f.tracks, f.ego, f.fps, mode=mode)
    return f.derived[key]


def slog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _aggregate(C: np.ndarray, fps: float) -> np.ndarray:
    """C: (N, n, d) windows -> (N, 8*d)."""
    n = C.shape[1]
    t = (np.arange(n) - (n - 1) / 2) / fps
    denom = float((t ** 2).sum()) or 1.0
    k = max(n // 3, 1)
    parts = [C[:, -1], C.mean(1), C.max(1), C.min(1), C.std(1),
             np.einsum("i,nid->nd", t, C - C.mean(1, keepdims=True)) / denom,
             C[:, -1] - C[:, 0], C[:, -k:].mean(1) - C[:, :k].mean(1)]
    return np.concatenate(parts, 1)


def channel_names(groups: list[str]) -> list[str]:
    base = {"motion": MOTION_FEATURES, "diff": FRAME_DIFF_FEATURES, "vdelta": ["vis_change_norm", "vis_change_cos"],
            "ego_h": EGO_H_STATS, "ego_f": EGO_F_STATS,
            **{g: [f"{n}@{m}" for n in REL_FEATURES] for g, m in REL_MODES.items()}}
    names = []
    for g in groups:
        if g == "vabs":
            continue
        for a in AGGS:
            names += [f"{c}__{a}" for c in base[g]]
    if "vabs" in groups:
        names += [f"vis_pca{i}" for i in range(32)]
    return names


def _vdelta_channels(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    d = np.zeros((len(v), 2), np.float32)
    if len(v) > 1:
        diff = v[1:] - v[:-1]
        d[1:, 0] = np.linalg.norm(diff, axis=1) / np.sqrt(v.shape[1])
        nv = np.linalg.norm(v, axis=1) + 1e-6
        d[1:, 1] = 1 - (v[1:] * v[:-1]).sum(1) / (nv[1:] * nv[:-1])
    return d


def video_window_matrix(f: VideoFeatures, t_ends: np.ndarray, W: float, groups: list[str], pca=None) -> np.ndarray:
    n = int(round(W * f.fps))
    idx = np.stack([window_indices(f.times, float(t), n) for t in t_ends])  # (N, n)
    out = []
    for g in groups:
        if g == "motion":
            out.append(_aggregate(slog(f.motion)[idx], f.fps))
        elif g == "diff":
            if f.diff is None:
                raise ValueError(f"{f.video_id}: cache has no frame-difference features (run scripts/add_framediff.py)")
            out.append(_aggregate(f.diff[idx], f.fps))
        elif g == "vdelta":
            out.append(_aggregate(_vdelta_channels(f.visual)[idx], f.fps))
        elif g in ("ego_h", "ego_f"):
            out.append(_aggregate(ego_channels(f, g)[idx], f.fps))
        elif g in REL_MODES:
            out.append(_aggregate(slog(rel_channels(f, g))[idx], f.fps))
    if "vabs" in groups:
        out.append(pca.transform(f.visual[idx].astype(np.float32).mean(1)))
    return np.concatenate(out, 1).astype(np.float32)


def build_matrix(samples, feats: dict[str, VideoFeatures], W: float, groups: list[str], pca=None) -> tuple[np.ndarray, np.ndarray]:
    """samples: list of WindowSample. Rows keep the order of `samples`."""
    order: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        order.setdefault(s.video_id, []).append(i)
    X = None
    for vid, ii in order.items():
        m = video_window_matrix(feats[vid], np.array([samples[i].t_end for i in ii]), W, groups, pca)
        if X is None:
            X = np.zeros((len(samples), m.shape[1]), np.float32)
        X[ii] = m
    return X, np.array([s.label for s in samples], np.float32)


def fit_vabs_pca(feats: list[VideoFeatures], seed: int = 0):
    from sklearn.decomposition import PCA
    V = np.concatenate([f.visual[::5].astype(np.float32) for f in feats])
    return PCA(32, random_state=seed).fit(V)
