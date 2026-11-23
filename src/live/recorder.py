"""Event recording for live operation: a bounded rolling buffer of annotated frames (JPEG in RAM) + clip writer.

- Pre-event frames come from the rolling buffer (default 5 s); post-event frames are collected after the trigger (default 5 s).
  Nothing else is written to disk: there is no continuous recording.
- Before writing, free disk space and the clip-store quota are checked. If a clip cannot be written, the event metadata is
  still stored (the event did happen) with an explicit `clip_status` describing the failure; no clip is fabricated.
- Event categories (honest naming):
    risk_event           P >= CAUTION level for >= k steps without an alarm
    warning_event        an alarm (P >= profile threshold for k steps)
    potential_near_miss  an alarm episode that reached HIGH RISK (P >= HIGH threshold). "Potential": NOT verified
    confirmed_collision  ONLY set manually by a user with evidence (never automatic)
"""
from __future__ import annotations

import json
import queue
import shutil
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import resolve

LIVE_STORE = resolve("data/memory")
CATEGORIES = ("risk_event", "warning_event", "potential_near_miss", "confirmed_collision")


@dataclass
class RecorderConfig:
    pre_s: float = 5.0
    post_s: float = 5.0
    fps: float = 10.0
    jpeg_quality: int = 80
    max_buffer_mb: float = 64.0        # cap on the RAM used by the rolling buffer
    max_store_mb: float = 2048.0       # quota for all stored clips
    min_free_mb: float = 1024.0        # refuse to write when the disk has less free space than this
    record_risk_events: bool = True


@dataclass
class LiveEvent:
    event_id: str
    session_id: str
    category: str
    started_wall: str
    t_stream: float
    level_peak: str
    p_peak: float
    mutual_info: float
    factors: list
    explanation: str
    lead_object: dict | None
    model: str
    profile: str
    pre_s: float
    post_s: float
    clip: str | None = None
    clip_status: str = "pending"
    clip_frames: int = 0
    note: str = ""
    # fields shared with the offline Near-Miss Memory schema
    trip_id: str = ""
    type: str = ""
    video: str = "live"
    t_start: float = 0.0
    t_end: float = 0.0
    t_peak: float = 0.0
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    mutual_info_mean: float = 0.0
    risk_score_peak: float = 0.0


class RollingBuffer:
    """Last `pre_s` seconds of annotated frames as JPEG bytes, bounded by count AND bytes."""

    def __init__(self, cfg: RecorderConfig):
        self.cfg = cfg
        self.frames: deque[tuple[float, bytes]] = deque(maxlen=max(int(cfg.pre_s * cfg.fps), 1))
        self.bytes = 0

    def push(self, t: float, frame: np.ndarray) -> bytes:
        ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
        b = enc.tobytes() if ok else b""
        if len(self.frames) == self.frames.maxlen:
            self.bytes -= len(self.frames[0][1])
        self.frames.append((t, b))
        self.bytes += len(b)
        while self.bytes > self.cfg.max_buffer_mb * 1e6 and self.frames:
            self.bytes -= len(self.frames.popleft()[1])
        return b

    def snapshot(self) -> list[tuple[float, bytes]]:
        return list(self.frames)


def store_size_mb(root: Path) -> float:
    d = root / "clips"
    return sum(p.stat().st_size for p in d.glob("*.mp4")) / 1e6 if d.exists() else 0.0


class EventRecorder:
    """Call `push(frame)` every grid step and `trigger(...)` when an event starts. Clips are written by a background thread."""

    def __init__(self, cfg: RecorderConfig | None = None, root: str | Path = LIVE_STORE):
        self.cfg = cfg or RecorderConfig()
        self.root = Path(root)
        (self.root / "clips").mkdir(parents=True, exist_ok=True)
        self.ring = RollingBuffer(self.cfg)
        self.pending: list[tuple[LiveEvent, list[tuple[float, bytes]], int]] = []
        self.done: list[LiveEvent] = []
        self.errors: list[str] = []
        self._q: queue.Queue = queue.Queue()
        self._writer = threading.Thread(target=self._write_loop, daemon=True, name="clip-writer")
        self._writer.start()

    # ---------- per step ----------
    def push(self, t: float, frame: np.ndarray) -> None:
        jpg = self.ring.push(t, frame)
        still = []
        for ev, frames, remaining in self.pending:
            frames.append((t, jpg))
            remaining -= 1
            if remaining <= 0:
                self._q.put((ev, frames))
            else:
                still.append((ev, frames, remaining))
        self.pending = still

    def active_event(self) -> LiveEvent | None:
        return self.pending[-1][0] if self.pending else None

    def trigger(self, ev: LiveEvent) -> LiveEvent:
        ev.pre_s, ev.post_s = self.cfg.pre_s, self.cfg.post_s
        self.pending.append((ev, self.ring.snapshot(), max(int(self.cfg.post_s * self.cfg.fps), 1)))
        return ev

    def flush(self, timeout: float = 30.0) -> None:
        """Finish pending events with the frames collected so far (e.g. on stop) and wait for the writer."""
        for ev, frames, _ in self.pending:
            ev.note = (ev.note + " post-event window truncated by stop").strip()
            self._q.put((ev, frames))
        self.pending = []
        t0 = time.time()
        while self._q.unfinished_tasks and time.time() - t0 < timeout:  # wait until every clip is written (not just dequeued)
            time.sleep(0.02)

    # ---------- writer ----------
    def _write_loop(self) -> None:
        while True:
            ev, frames = self._q.get()
            try:
                self._write(ev, frames)
            except Exception as e:  # noqa: BLE001
                ev.clip_status = f"not saved: {e!r}"
                self.errors.append(ev.clip_status)
            try:
                self._append_metadata(ev)
                self.done.append(ev)
            finally:
                self._q.task_done()

    def _write(self, ev: LiveEvent, frames: list[tuple[float, bytes]]) -> None:
        free_mb = shutil.disk_usage(self.root).free / 1e6
        if free_mb < self.cfg.min_free_mb:
            raise OSError(f"insufficient disk space ({free_mb:.0f} MB free < {self.cfg.min_free_mb:.0f} MB minimum)")
        if store_size_mb(self.root) > self.cfg.max_store_mb:
            raise OSError(f"clip store quota exceeded ({self.cfg.max_store_mb:.0f} MB); delete old events in Near-Miss Memory")
        imgs = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for _, b in frames if b]
        imgs = [i for i in imgs if i is not None]
        if not imgs:
            raise ValueError("no frames available for the clip")
        from src.visualization.overlay import write_video
        path = write_video(imgs, self.root / "clips" / f"{ev.event_id}.mp4", self.cfg.fps)
        cap = cv2.VideoCapture(str(path))  # verify the file really decodes
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if n <= 0:
            raise OSError("written clip cannot be decoded")
        ev.clip, ev.clip_status, ev.clip_frames = str(path), "saved", len(imgs)

    def _append_metadata(self, ev: LiveEvent) -> None:
        with open(self.root / "events.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(ev), ensure_ascii=False, default=str) + "\n")


def new_event(session_id: str, category: str, t: float, res, model: str, profile: str) -> LiveEvent:
    a = res.assessment
    return LiveEvent(uuid.uuid4().hex[:12], session_id, category, datetime.now().isoformat(timespec="seconds"), t,
                     res.level, float(res.p), float(res.mutual_info), list(getattr(a, "triggering_factors", []) or []),
                     getattr(a, "explanation", ""), None, model, profile, 0.0, 0.0, trip_id=f"live-{session_id}", type=category,
                     t_start=t, t_end=t, t_peak=t, mutual_info_mean=float(res.mutual_info), risk_score_peak=float(res.risk_score))
