"""Test fixtures. The synthetic clip is ONLY for software tests — never used as experimental data."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def sample_image() -> np.ndarray:
    from ultralytics.utils import ASSETS
    img = cv2.imread(str(Path(ASSETS) / "bus.jpg"))
    assert img is not None
    return img


def make_zoom_video(path: str | Path, seconds: float = 4.0, fps: float = 30.0, size=(640, 360)) -> Path:
    """Progressive zoom into a real street photo: objects grow (looming) over time."""
    img = sample_image()
    h, w = img.shape[:2]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    n = int(seconds * fps)
    for i in range(n):
        z = 1.0 + 0.8 * i / n
        cw, ch = int(w / z), int(w / z * size[1] / size[0])
        ch = min(ch, h)
        x0, y0 = (w - cw) // 2, max((h - ch) // 2, 0)
        vw.write(cv2.resize(img[y0:y0 + ch, x0:x0 + cw], size))
    vw.release()
    return path


def make_blank_video(path: str | Path, seconds: float = 2.0, fps: float = 30.0, size=(320, 180)) -> Path:
    path = Path(path)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for _ in range(int(seconds * fps)):
        vw.write(np.full((size[1], size[0], 3), 90, np.uint8))
    vw.release()
    return path
