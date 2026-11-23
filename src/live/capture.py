"""Capture thread: reads frames from a source continuously and keeps only the LATEST frame (no queue growth).

- one capture per device in the whole process (registry); a second start on the same device is refused
- automatic reconnect with back-off after read failures; interruptions are counted
- the source is always released on stop or on fatal error
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from src.live.sources import SourceError

_REGISTRY: dict[str, "CaptureThread"] = {}
_REG_LOCK = threading.Lock()


class DeviceBusyError(RuntimeError):
    pass


@dataclass
class CaptureStats:
    frames: int = 0
    interruptions: int = 0
    reconnects: int = 0
    state: str = "idle"           # idle | running | reconnecting | failed | stopped
    error: str = ""
    started: float = 0.0
    last_frame_t: float = 0.0
    fps_window: list = field(default_factory=list)

    def capture_fps(self) -> float:
        w = self.fps_window
        return (len(w) - 1) / (w[-1] - w[0]) if len(w) > 1 and w[-1] > w[0] else 0.0


class CaptureThread(threading.Thread):
    def __init__(self, source, max_failures: int = 10, reconnect_backoff_s: float = 1.0, max_reconnect_s: float = 30.0):
        super().__init__(daemon=True, name=f"capture-{source.key}")
        self.source = source
        self.max_failures = max_failures
        self.backoff = reconnect_backoff_s
        self.max_reconnect_s = max_reconnect_s
        self.stats = CaptureStats()
        self._lock = threading.Lock()
        self._latest: tuple[int, float, np.ndarray] | None = None
        self._stop_evt = threading.Event()

    # ---------- lifecycle ----------
    def start(self) -> None:  # type: ignore[override]
        with _REG_LOCK:
            other = _REGISTRY.get(self.source.key)
            if other is not None and other.is_alive():
                raise DeviceBusyError(f"{self.source.key} is already being captured by this application.")
            self.source.open()  # raises SourceError with a clear message
            _REGISTRY[self.source.key] = self
        self.stats.state, self.stats.started = "running", time.time()
        super().start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout)
        self._release()

    def _release(self) -> None:
        try:
            self.source.release()
        finally:
            with _REG_LOCK:
                if _REGISTRY.get(self.source.key) is self:
                    del _REGISTRY[self.source.key]
            if self.stats.state not in ("failed", "ended"):
                self.stats.state = "stopped"

    # ---------- frames ----------
    def latest(self) -> tuple[int, float, np.ndarray] | None:
        with self._lock:
            return self._latest

    def run(self) -> None:
        fails = 0
        try:
            while not self._stop_evt.is_set():
                ok, frame, t = self.source.read()
                if ok and frame is not None:
                    fails = 0
                    with self._lock:
                        self.stats.frames += 1
                        self._latest = (self.stats.frames, t, frame)
                    self.stats.last_frame_t = t
                    self.stats.fps_window = (self.stats.fps_window + [t])[-60:]
                    continue
                if getattr(self.source, "eof", False):  # demo video finished: end cleanly, no reconnect
                    self.stats.state = "ended"
                    return
                fails += 1
                if fails < self.max_failures:
                    time.sleep(0.02)
                    continue
                # treat as a disconnection: release and try to reopen with back-off
                self.stats.interruptions += 1
                self.stats.state = "reconnecting"
                self.source.release()
                t_start, delay = time.time(), self.backoff
                while not self._stop_evt.is_set():
                    try:
                        self.source.open()
                        self.stats.reconnects += 1
                        self.stats.state, fails = "running", 0
                        break
                    except SourceError as e:
                        self.stats.error = str(e)
                        if time.time() - t_start > self.max_reconnect_s:
                            self.stats.state = "failed"
                            return
                        self._stop_evt.wait(delay)
                        delay = min(delay * 2, 5.0)
        except Exception as e:  # noqa: BLE001 - never leave the device locked
            self.stats.state, self.stats.error = "failed", repr(e)
        finally:
            self._release()


def active_devices() -> list[str]:
    with _REG_LOCK:
        return [k for k, t in _REGISTRY.items() if t.is_alive()]
