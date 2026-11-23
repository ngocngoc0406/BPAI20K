"""Camera input-quality monitor (DEGRADED INPUT), separate from perception reliability and from risk.

Per frame (cheap, < 2 ms at 640 px):
  stalled        no new frame for > stall_s, or the median frame interval > max_interval_s (models assume 10 Hz)
  black          mean brightness < 0.05            saturated   share of pixels > 250 is > 0.6
  frozen         mean absolute difference to the previous frame < 0.5 grey level for >= frozen_frames frames
  blurred        variance of the Laplacian < blur_min (lens dirty / out of focus)
  dark           mean brightness < 0.12 (night without light: still usable, reported as a reason, not degraded alone)
DEGRADED when any of stalled / black / saturated / frozen / blurred holds. Thresholds are fixed engineering values (not tuned on
evaluation data); they are listed in the uncertainty doc and checked with synthetic frames in tests.
Replay mode (no frames, v3 caches): brightness and the frame-difference channel of the cache are used instead.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class QualityState:
    degraded: bool
    reasons: list[str]
    metrics: dict = field(default_factory=dict)


class InputQualityMonitor:
    def __init__(self, stall_s: float = 1.0, max_interval_s: float = 0.25, frozen_frames: int = 10, blur_min: float = 15.0):
        self.stall_s, self.max_iv, self.frozen_n, self.blur_min = stall_s, max_interval_s, frozen_frames, blur_min
        self.reset()

    def reset(self) -> None:
        self.prev: np.ndarray | None = None
        self.still = 0
        self.last_t: float | None = None
        self.iv: deque = deque(maxlen=20)

    def update(self, frame: np.ndarray | None, t_wall: float) -> QualityState:
        import cv2
        reasons: list[str] = []
        if self.last_t is not None:
            self.iv.append(t_wall - self.last_t)
        if frame is None:
            stalled = self.last_t is not None and t_wall - self.last_t > self.stall_s
            return QualityState(True, ["no frame from the camera" + (" (stalled)" if stalled else "")], {})
        self.last_t = t_wall
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        small = cv2.resize(g, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
        bright = float(small.mean() / 255.0)
        sat = float((small > 250).mean())
        lap = float(cv2.Laplacian(cv2.resize(g, (320, 180)), cv2.CV_64F).var())
        diff = float(np.abs(small - self.prev).mean()) if self.prev is not None else 255.0
        self.prev = small
        self.still = self.still + 1 if diff < 0.5 else 0
        med_iv = float(np.median(self.iv)) if self.iv else 0.1
        if med_iv > self.max_iv:
            reasons.append(f"frame interval {med_iv:.2f} s > {self.max_iv} s")
        if bright < 0.05:
            reasons.append("image black (lens covered / camera off)")
        if sat > 0.6:
            reasons.append("image saturated")
        if self.still >= self.frozen_n:
            reasons.append(f"image frozen for {self.still} frames")
        if lap < self.blur_min:
            reasons.append("image blurred (dirty lens / out of focus)")
        degraded = bool(reasons)
        if bright < 0.12 and not degraded:
            reasons.append("dark scene (usable, lower detection reliability)")
        return QualityState(degraded, reasons, {"brightness": bright, "saturated": sat, "sharpness": lap, "frame_diff": diff,
                                                "median_interval_s": med_iv, "frozen_frames": self.still})


def replay_quality(brightness: float, frame_diff_mean: float | None, frozen_run: int) -> QualityState:
    """Replay (cache) approximation: black / saturated from brightness, frozen from the cached frame-difference channel."""
    reasons = []
    if brightness < 0.05:
        reasons.append("image black")
    if brightness > 0.95:
        reasons.append("image saturated")
    if frame_diff_mean is not None and frozen_run >= 10:
        reasons.append(f"image frozen for {frozen_run} steps")
    return QualityState(bool(reasons), reasons, {"brightness": brightness, "frozen_run": frozen_run})
