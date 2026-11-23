"""On-disk cache for per-video extracted features (compressed .npz + JSON metadata)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CACHE_VERSION = 3  # v2 adds "diff"; v3 adds "ego" (camera-motion transforms)


@dataclass
class VideoFeatures:
    video_id: str
    times: np.ndarray          # (T,) grid timestamps in seconds
    visual: np.ndarray         # (T, Dv) float16 CNN embeddings
    motion: np.ndarray         # (T, Dm) float32 hand-crafted motion features
    tracks: np.ndarray         # (N, 8) [step, track_id, cls, conf, x1, y1, x2, y2] normalised coords
    meta: dict = field(default_factory=dict)
    diff: np.ndarray | None = None  # (T, 20) frame-difference descriptor (src/features/frame_diff.py); None in v1 caches
    ego: np.ndarray | None = None   # (T, 30) camera-motion vector (src/features/ego_motion.py); None before v3
    derived: dict = field(default_factory=dict, repr=False)  # memo for on-the-fly derived arrays

    @property
    def fps(self) -> float:
        return float(self.meta.get("target_fps", 10.0))

    def __len__(self) -> int:
        return len(self.times)


def cache_path(features_dir: str | Path, video_id: str, tag: str = "clean") -> Path:
    return Path(features_dir) / tag / f"{video_id}.npz"


def save_features(feat: VideoFeatures, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, times=feat.times, visual=feat.visual.astype(np.float16),
                        motion=feat.motion.astype(np.float32), tracks=feat.tracks.astype(np.float32),
                        diff=(feat.diff if feat.diff is not None else np.zeros((len(feat.times), 0))).astype(np.float32),
                        ego=(feat.ego if feat.ego is not None else np.zeros((len(feat.times), 0))).astype(np.float32),
                        meta=json.dumps({**feat.meta, "cache_version": CACHE_VERSION, "video_id": feat.video_id}))
    tmp.replace(path)  # atomic: a crash never leaves a half-written cache file


def load_features(path: str | Path) -> VideoFeatures:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        diff = z["diff"] if "diff" in z.files and z["diff"].shape[-1] > 0 else None
        ego = z["ego"] if "ego" in z.files and z["ego"].shape[-1] > 0 else None
        return VideoFeatures(meta["video_id"], z["times"], z["visual"], z["motion"], z["tracks"], meta, diff, ego)
