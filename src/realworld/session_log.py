"""Per-step session timeline (CSV), audio log (JSONL) and optional full-session video (segmented MP4) for real-world evaluation.

- Timeline: one row per processed 10 Hz step (buffered writes), enough to recompute alarms / false alarms per hour offline.
- Full-session video (opt-in, consent required): raw (un-annotated) frames at 10 fps in 5-minute H.264 segments, written by a
  background thread with a streaming writer (constant memory). People can be blurred (detector 'person' boxes).
  Licence plates are NOT detected by the current models, so they are NOT blurred: see the privacy section of the protocol.
"""
from __future__ import annotations

import csv
import json
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

COLUMNS = ["step", "t_stream", "wall", "level", "p", "risk_score", "mi", "alarm", "degraded", "capture_fps", "step_ms", "e2e_ms",
           "n_objects", "lead_in_corridor", "rss_mb", "camera_state", "camera_interruptions"]
SYSTEM_COLUMNS = ["wall", "rss_mb", "cpu_percent_machine", "gpu_util", "gpu_mem_mb", "disk_free_gb", "session_mb"]


class TimelineLogger:
    def __init__(self, session_dir: Path, flush_every: int = 50):
        self.dir = Path(session_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "timeline.csv"
        self.audio_path = self.dir / "audio_log.jsonl"
        self.rows: list[list] = []
        self.flush_every = flush_every
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(COLUMNS)

    def log(self, row: dict) -> None:
        self.rows.append([row.get(c, "") for c in COLUMNS])
        if len(self.rows) >= self.flush_every:
            self.flush()

    def log_audio(self, rec: dict, t_stream: float) -> None:
        with open(self.audio_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({**rec, "t_stream": t_stream}, ensure_ascii=False) + "\n")

    def flush(self) -> None:
        if not self.rows:
            return
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(self.rows)
        self.rows = []


class SystemLogger:
    """System resources every `every_s` seconds on a background thread (system_log.csv): process RSS, CPU % of the machine,
    GPU utilisation / memory (nvidia-smi, empty if unavailable), free disk space and the session folder size."""

    def __init__(self, session_dir: Path, every_s: float = 5.0):
        import psutil
        self.dir = Path(session_dir)
        self.path = self.dir / "system_log.csv"
        self.every_s = every_s
        self._proc = psutil.Process()
        self._proc.cpu_percent(None)
        self._stop = threading.Event()
        with open(self.path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(SYSTEM_COLUMNS)
        self._thread = threading.Thread(target=self._run, daemon=True, name="session-system-log")
        self._thread.start()

    @staticmethod
    def gpu() -> tuple:
        import subprocess
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            u, m = out.stdout.strip().splitlines()[0].split(",")
            return float(u), float(m)
        except Exception:  # noqa: BLE001
            return "", ""

    def sample(self) -> list:
        import os
        import shutil
        import psutil
        size = sum(f.stat().st_size for f in self.dir.rglob("*") if f.is_file()) / 1e6
        u, m = self.gpu()
        return [round(time.time(), 2), round(self._proc.memory_info().rss / 1e6, 1),
                round(self._proc.cpu_percent(None) / (os.cpu_count() or 1), 1), u, m,
                round(shutil.disk_usage(self.dir).free / 1e9, 2), round(size, 1)] if psutil else []

    def _run(self) -> None:
        while not self._stop.wait(self.every_s):
            try:
                row = self.sample()
                with open(self.path, "a", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerow(row)
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        self._stop.set()
        self._thread.join(10)


def blur_people(frame: np.ndarray, rows: np.ndarray | None) -> np.ndarray:
    """Blur detected persons (COCO class 0) in a frame (rows: [step, id, cls, conf, x1, y1, x2, y2], normalised)."""
    if rows is None or len(rows) == 0:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    for r in rows:
        if int(r[2]) != 0:
            continue
        x1, y1, x2, y2 = int(r[4] * w), int(r[5] * h), int(r[6] * w), int(r[7] * h)
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
        if x2 - x1 > 2 and y2 - y1 > 2:
            roi = out[y1:y2, x1:x2]
            k = max(15, (min(roi.shape[:2]) // 3) | 1)
            out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return out


class SessionVideoWriter:
    """Streams frames into 5-minute MP4 segments on a background thread. Drops frames (and counts them) if the writer falls behind."""

    def __init__(self, session_dir: Path, fps: float = 10.0, segment_s: float = 300.0, max_queue: int = 100):
        self.dir = Path(session_dir) / "video"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fps, self.segment_frames = fps, int(segment_s * fps)
        self.q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.dropped = 0
        self.written = 0
        self.errors: list[str] = []
        self.segments: list[dict] = []
        self._thread = threading.Thread(target=self._run, daemon=True, name="session-video")
        self._thread.start()

    def push(self, t_stream: float, frame: np.ndarray) -> None:
        try:
            self.q.put_nowait((t_stream, frame))
        except queue.Full:
            self.dropped += 1

    def close(self, timeout: float = 30.0) -> dict:
        self.q.put(None)
        self._thread.join(timeout)
        info = {"segments": self.segments, "frames_written": self.written, "frames_dropped": self.dropped, "errors": self.errors}
        (self.dir / "segments.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        return info

    def _run(self) -> None:
        import imageio.v2 as imageio
        wr, n_in_seg, seg = None, 0, None
        while True:
            item = self.q.get()
            if item is None:
                break
            t, frame = item
            try:
                if wr is None or n_in_seg >= self.segment_frames:
                    if wr is not None:
                        wr.close()
                        seg["t_end"] = t
                    path = self.dir / f"segment_{len(self.segments):03d}.mp4"
                    wr = imageio.get_writer(str(path), fps=self.fps, codec="libx264", quality=6, macro_block_size=8,
                                            ffmpeg_log_level="error")
                    seg = {"file": str(path), "t_start": t, "t_end": t, "frames": 0}
                    self.segments.append(seg)
                    n_in_seg = 0
                wr.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                n_in_seg += 1
                seg["frames"] += 1
                seg["t_end"] = t
                self.written += 1
            except Exception as e:  # noqa: BLE001
                self.errors.append(repr(e))
        if wr is not None:
            wr.close()
        _ = time
