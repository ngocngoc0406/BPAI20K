"""Extract frames (e.g. thumbnails for the UI, or a frame at a given timestamp)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.preprocessing.video_reader import VideoReadError, open_video


def frame_at(path: str | Path, t_s: float) -> np.ndarray:
    cap, meta = open_video(path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t_s * meta.fps)))
        ok, frame = cap.read()
        if not ok:
            raise VideoReadError(f"Cannot read frame at {t_s:.2f}s from {path}")
        return frame
    finally:
        cap.release()


def resize_width(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == width:
        return frame
    return cv2.resize(frame, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def thumbnails(path: str | Path, n: int = 6, width: int = 320) -> list[tuple[float, np.ndarray]]:
    cap, meta = open_video(path)
    cap.release()
    ts = np.linspace(0.5, max(meta.duration_s - 0.5, 0.5), n)
    return [(float(t), cv2.cvtColor(resize_width(frame_at(path, t), width), cv2.COLOR_BGR2RGB)) for t in ts]
