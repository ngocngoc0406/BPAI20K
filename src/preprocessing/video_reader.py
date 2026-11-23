"""Robust OpenCV video reading with explicit errors and metadata."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoReadError(RuntimeError):
    pass


@dataclass
class VideoMeta:
    path: str
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0


def open_video(path: str | Path) -> tuple[cv2.VideoCapture, VideoMeta]:
    path = Path(path)
    if not path.exists():
        raise VideoReadError(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoReadError(f"Cannot open video (corrupt or unsupported codec): {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or fps > 240:
        fps = 30.0  # some containers miss FPS; documented fallback
    meta = VideoMeta(str(path), fps, n, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    ok, _ = cap.read()
    if not ok:
        cap.release()
        raise VideoReadError(f"Video has no decodable frames: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap, meta


def get_meta(path: str | Path) -> VideoMeta:
    cap, meta = open_video(path)
    cap.release()
    return meta


def iter_frames(path: str | Path, start_s: float = 0.0, end_s: float | None = None) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, timestamp_s, BGR frame) sequentially; decoding is sequential for accuracy."""
    cap, meta = open_video(path)
    try:
        start_idx = int(round(start_s * meta.fps))
        if start_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        idx = start_idx
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            t = idx / meta.fps
            if end_s is not None and t > end_s + 1e-6:
                break
            yield idx, t, frame
            idx += 1
    finally:
        cap.release()
