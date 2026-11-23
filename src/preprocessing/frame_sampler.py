"""Resample a native-FPS frame stream onto a fixed time grid (FPS normalisation)."""
from __future__ import annotations

from typing import Iterable, Iterator

import numpy as np


def sample_on_grid(frames: Iterable[tuple[int, float, np.ndarray]], target_fps: float,
                   start_s: float = 0.0) -> Iterator[tuple[int, float, int, np.ndarray]]:
    """Yield (step, grid_time_s, source_frame_index, frame) choosing the first source frame
    whose timestamp reaches each grid point. Grid: start_s + k / target_fps."""
    dt = 1.0 / target_fps
    k = 0
    next_t = start_s
    for idx, t, frame in frames:
        if t + 1e-6 >= next_t:
            yield k, round(next_t, 4), idx, frame
            k += 1
            next_t = start_s + k * dt
