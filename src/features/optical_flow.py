"""Dense optical-flow summary statistics (Farneback, low resolution)."""
from __future__ import annotations

import cv2
import numpy as np

FLOW_FEATURES = ["flow_mag_mean", "flow_mag_center", "flow_div_center", "flow_dx_median"]


class FlowComputer:
    def __init__(self, size: tuple[int, int] = (160, 90)):
        self.size = tuple(size)
        self.prev: np.ndarray | None = None
        self.last_field: np.ndarray | None = None  # pixel flow (prev -> cur) at self.size, for ego-motion

    def reset(self) -> None:
        self.prev = None
        self.last_field = None

    def __call__(self, frame_bgr: np.ndarray, dt: float) -> np.ndarray:
        g = cv2.cvtColor(cv2.resize(frame_bgr, self.size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if self.prev is None:
            self.prev = g
            self.last_field = None
            return np.zeros(len(FLOW_FEATURES), np.float32)
        flow = cv2.calcOpticalFlowFarneback(self.prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.prev = g
        self.last_field = flow
        w, h = self.size
        # convert to normalised image units per second
        fx, fy = flow[..., 0] / w / dt, flow[..., 1] / h / dt
        mag = np.sqrt(fx ** 2 + fy ** 2)
        ys, xs = slice(int(h * 0.3), int(h * 0.9)), slice(int(w * 0.3), int(w * 0.7))
        div = np.gradient(fx, axis=1) * w + np.gradient(fy, axis=0) * h  # d(fx)/dx + d(fy)/dy (1/s)
        return np.array([mag.mean(), mag[ys, xs].mean(), div[ys, xs].mean(), np.median(fx)], np.float32)
