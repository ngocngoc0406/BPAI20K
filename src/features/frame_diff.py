"""Cheap appearance-change descriptor (no CNN): statistics of |I_t - I_{t-1}| on a 160x90 grayscale frame.

20 values per step: 4x4 grid of mean absolute difference (16) + global mean, centre-ROI mean,
fraction of changed pixels (>15/255), and mean-brightness change. Normalised to [0, 1] units.
"""
from __future__ import annotations

import cv2
import numpy as np

FRAME_DIFF_FEATURES = [f"diff_g{r}{c}" for r in range(4) for c in range(4)] + [
    "diff_mean", "diff_center", "diff_changed_frac", "diff_brightness"]


class FrameDiff:
    def __init__(self, size: tuple[int, int] = (160, 90)):
        self.size = tuple(size)
        self.prev: np.ndarray | None = None

    def reset(self) -> None:
        self.prev = None

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(cv2.resize(frame_bgr, self.size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if self.prev is None:
            self.prev = g
            return np.zeros(len(FRAME_DIFF_FEATURES), np.float32)
        d = np.abs(g - self.prev)
        bright = float(g.mean() - self.prev.mean())
        self.prev = g
        h, w = d.shape
        grid = d[: h - h % 4, : w - w % 4].reshape(4, h // 4, 4, w // 4).mean(axis=(1, 3)).ravel()
        center = d[int(h * 0.3): int(h * 0.9), int(w * 0.3): int(w * 0.7)].mean()
        return np.concatenate([grid, [d.mean(), center, (d > 15 / 255).mean(), bright]]).astype(np.float32)
