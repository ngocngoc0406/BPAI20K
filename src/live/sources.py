"""Frame sources for live operation: an OS camera (OpenCV) or a demo video file played in real time.

Only one frame is held at a time; nothing is accumulated in RAM.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class SourceError(RuntimeError):
    pass


def list_cameras(max_index: int = 4) -> list[dict]:
    """Probe camera indices 0..max_index-1 (DirectShow on Windows). A camera held by another app may fail to open."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        try:
            if cap.isOpened():
                ok, f = cap.read()
                found.append({"index": i, "readable": bool(ok), "shape": None if f is None else list(f.shape)})
        finally:
            cap.release()
    return found


@dataclass
class CameraSource:
    index: int = 0
    width: int | None = None
    height: int | None = None

    def __post_init__(self):
        self.cap = None

    @property
    def key(self) -> str:
        return f"camera:{self.index}"

    def open(self) -> None:
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            raise SourceError(f"Camera {self.index} cannot be opened (missing, disabled, or in use by another application).")
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep the driver queue short -> lower latency
        ok, _ = cap.read()
        if not ok:
            cap.release()
            raise SourceError(f"Camera {self.index} opened but returns no frames (it may be in use by another application).")
        self.cap = cap

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        if self.cap is None:
            return False, None, time.time()
        ok, f = self.cap.read()
        return ok, f, time.time()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


@dataclass
class VideoFileSource:
    """Demo mode: a local video played at its native speed (real-time pacing) and optionally looped.
    With realtime=False frames are returned as fast as possible (used for equivalence tests)."""
    path: str
    realtime: bool = True
    loop: bool = True

    def __post_init__(self):
        self.cap = None
        self.t0 = 0.0
        self.idx = 0
        self.fps = 30.0
        self.eof = False

    @property
    def key(self) -> str:
        return f"file:{Path(self.path).resolve()}"

    def open(self) -> None:
        if not Path(self.path).exists():
            raise SourceError(f"Demo video not found: {self.path}")
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise SourceError(f"Demo video cannot be decoded: {self.path}")
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
        self.cap, self.t0, self.idx, self.eof = cap, time.time(), 0, False

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        if self.cap is None:
            return False, None, time.time()
        if self.realtime:  # pace to the source frame rate
            due = self.t0 + self.idx / self.fps
            wait = due - time.time()
            if wait > 0:
                time.sleep(wait)
        ok, f = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.t0, self.idx = time.time(), 0
            ok, f = self.cap.read()
        elif not ok:
            self.eof = True  # normal end of a non-looping demo video (not a disconnection)
        self.idx += 1
        return ok, f, time.time()

    def media_time(self) -> float:
        return (self.idx - 1) / self.fps

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
