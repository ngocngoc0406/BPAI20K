"""Controlled image corruptions for robustness experiments (RQ4).

Each corruption is deterministic given (name, severity, seed, step) so runs are reproducible.
Severity levels are documented in docs/EXPERIMENT_DESIGN.md.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

# name -> {severity -> parameter}
CORRUPTIONS: dict[str, dict[int, float]] = {
    "frame_drop": {1: 0.25, 2: 0.50},     # probability a grid frame is replaced by the previous one
    "motion_blur": {1: 9, 2: 21},         # horizontal blur kernel length (px at 1280 width)
    "low_light": {1: 0.45, 2: 0.25},      # brightness gain applied in linear-ish space + gamma
    "occlusion": {1: 0.10, 2: 0.25},      # fraction of image area covered by random black blocks
    "noise": {1: 15.0, 2: 35.0},          # Gaussian noise std (0-255 scale)
    "shake": {1: 6.0, 2: 15.0},           # camera shake: per-frame jitter std in px at 1280 width (+ 0.1 deg per px-unit rotation)
}
# v3.3 stress conditions (kept OUT of CORRUPTIONS so the original robustness benchmark is unchanged).
# light_change: tunnel entry/exit-like exposure swings applied per 10 Hz step: the gain switches between 1.0 and the
# severity's (dark, bright) gains every 2.5-4.5 s with a 0.3 s ramp (sev 1: 0.35 / 1.0; sev 2: 0.15 / 1.8 = over-exposure).
STRESS_V33: dict[str, dict[int, tuple]] = {"light_change": {1: (0.35, 1.0), 2: (0.15, 1.8)}}


def _blur(frame: np.ndarray, k: float) -> np.ndarray:
    k = int(k) | 1
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(frame, -1, kernel)


def _low_light(frame: np.ndarray, gain: float, rng: np.random.Generator) -> np.ndarray:
    f = (frame.astype(np.float32) / 255.0) ** 1.5 * gain  # darken non-linearly
    f = f + rng.normal(0, 0.01, f.shape).astype(np.float32)  # sensor noise in the dark
    return np.clip(f * 255.0, 0, 255).astype(np.uint8)


def _occlude(frame: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    target = frac * h * w
    covered = 0.0
    while covered < target:
        bw, bh = int(rng.uniform(0.08, 0.2) * w), int(rng.uniform(0.08, 0.2) * h)
        x, y = int(rng.uniform(0, w - bw)), int(rng.uniform(0, h - bh))
        out[y:y + bh, x:x + bw] = 0
        covered += bw * bh
    return out


def _noise(frame: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    return np.clip(frame.astype(np.float32) + rng.normal(0, std, frame.shape), 0, 255).astype(np.uint8)


class Corruptor:
    """Stateful per-video corruption applied to each grid frame in order."""

    def __init__(self, name: str, severity: int, seed: int = 0):
        if name != "clean" and name not in CORRUPTIONS and name not in STRESS_V33:
            raise ValueError(f"Unknown corruption {name}")
        self.name, self.severity = name, severity
        self.param = CORRUPTIONS.get(name, STRESS_V33.get(name, {})).get(severity, 0.0)
        self._step = 0
        self._light_plan: list[tuple[int, float]] = []
        self.rng = np.random.default_rng(seed)
        self._prev: np.ndarray | None = None
        # occlusion blocks are static over a clip (like dirt / a sticker on the lens)
        self._occ_seed = int(self.rng.integers(1 << 30))
        self._shake_state = np.zeros(3)

    def _shake(self, frame: np.ndarray) -> np.ndarray:
        """Random-walk-with-damping jitter (high-frequency dashcam vibration): translation and small rotation."""
        s = self.param * frame.shape[1] / 1280
        self._shake_state = 0.5 * self._shake_state + self.rng.normal(0, [s, s, 0.1 * s], 3)
        dx, dy, rot = self._shake_state
        return _shake_warp(frame, dx, dy, rot)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        n = self.name
        if n == "clean":
            return frame
        if n == "frame_drop":
            if self._prev is not None and self.rng.random() < self.param:
                return self._prev
            self._prev = frame
            return frame
        if n == "motion_blur":
            return _blur(frame, self.param * frame.shape[1] / 1280)
        if n == "low_light":
            return _low_light(frame, self.param, self.rng)
        if n == "occlusion":
            return _occlude(frame, self.param, np.random.default_rng(self._occ_seed))
        if n == "noise":
            return _noise(frame, self.param, self.rng)
        if n == "shake":
            return self._shake(frame)
        if n == "light_change":
            return self._light_change(frame)
        raise AssertionError

    def _light_gain(self, step: int) -> float:
        """Piecewise-constant gain schedule (seeded) with 3-step (0.3 s) linear ramps between levels."""
        dark, bright = self.param
        while not self._light_plan or self._light_plan[-1][0] <= step + 3:
            last_t = self._light_plan[-1][0] if self._light_plan else 0
            k = len(self._light_plan)
            level = 1.0 if k % 2 == 0 else (dark if (k // 2) % 2 == 0 else bright)
            self._light_plan.append((last_t + int(self.rng.integers(25, 46)), level))
        prev_level, start = 1.0, 0
        for end, level in self._light_plan:
            if step < end:
                ramp = min(1.0, (step - start) / 3.0)
                return prev_level + (level - prev_level) * ramp
            prev_level, start = level, end
        return 1.0

    def _light_change(self, frame: np.ndarray) -> np.ndarray:
        g = self._light_gain(self._step)
        self._step += 1
        if abs(g - 1.0) < 1e-6:
            return frame
        f = frame.astype(np.float32) / 255.0
        f = f ** 1.5 * g if g < 1 else 1 - (1 - f) ** g  # darken non-linearly / over-expose (highlights clip)
        return np.clip(f * 255.0, 0, 255).astype(np.uint8)


def _shake_warp(frame: np.ndarray, dx: float, dy: float, rot_deg: float) -> np.ndarray:
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot_deg, 1.0)
    M[:, 2] += (dx, dy)
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def make_transform(name: str, severity: int, seed: int = 0) -> Callable[[np.ndarray], np.ndarray]:
    return Corruptor(name, severity, seed)
