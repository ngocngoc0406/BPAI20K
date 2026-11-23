"""Driving session: statistics accumulated while the live assistant runs + JSON / PDF report.

No safety score is computed (no validated basis). Counts are *onsets* of each level / event, not frames.
"""
from __future__ import annotations

import io
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from src.utils.config import resolve

SESSIONS_DIR = resolve("data/sessions")


@dataclass
class SessionStats:
    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:4])
    started: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    ended: str = ""
    source: str = ""
    model: str = ""
    profile: str = ""
    device: str = ""
    wall_start: float = field(default_factory=time.time)
    wall_end: float = 0.0
    steps: int = 0
    level_onsets: dict = field(default_factory=lambda: {"CAUTION": 0, "WARNING": 0, "HIGH RISK": 0, "UNCERTAIN": 0})
    alarms: int = 0
    events: dict = field(default_factory=lambda: {"risk_event": 0, "warning_event": 0, "potential_near_miss": 0})
    clips_saved: int = 0
    clips_failed: int = 0
    audio_played: int = 0
    audio_suppressed: int = 0
    degraded_steps: int = 0          # steps with input slower than the 10 Hz the models assume (outputs suppressed)
    camera_interruptions: int = 0
    camera_reconnects: int = 0
    step_total_ms: list = field(default_factory=list)
    e2e_ms: list = field(default_factory=list)
    step_intervals_s: list = field(default_factory=list)
    capture_fps_samples: list = field(default_factory=list)

    _prev_level: str = "NORMAL"
    _last_step_wall: float = 0.0

    def on_step(self, res, e2e_ms: float, wall_now: float) -> None:
        self.steps += 1
        self.step_total_ms.append(res.timings_ms.get("total", 0.0))
        self.e2e_ms.append(e2e_ms)
        if self._last_step_wall:
            self.step_intervals_s.append(wall_now - self._last_step_wall)
        self._last_step_wall = wall_now
        for buf in (self.step_total_ms, self.e2e_ms, self.step_intervals_s):  # bounded memory for long sessions
            if len(buf) > 20000:
                del buf[: len(buf) - 20000]
        if res.predicted:
            if res.level != self._prev_level and res.level in self.level_onsets:
                self.level_onsets[res.level] += 1
            self._prev_level = res.level
        if res.alarm:
            self.alarms += 1

    def summary(self) -> dict:
        dur = (self.wall_end or time.time()) - self.wall_start
        pct = lambda a, q: float(np.percentile(a, q)) if len(a) else None  # noqa: E731
        d = {k: v for k, v in asdict(self).items() if not k.startswith("_") and not isinstance(v, list)}
        d.update({"duration_s": dur, "processed_steps_per_s": self.steps / dur if dur > 0 else 0.0,
                  "mean_step_interval_s": float(np.mean(self.step_intervals_s)) if self.step_intervals_s else None,
                  "late_steps_fraction": float(np.mean(np.array(self.step_intervals_s) > 0.15)) if self.step_intervals_s else None,
                  "step_ms_p50": pct(self.step_total_ms, 50), "step_ms_p95": pct(self.step_total_ms, 95),
                  "e2e_ms_p50": pct(self.e2e_ms, 50), "e2e_ms_p95": pct(self.e2e_ms, 95),
                  "capture_fps_mean": float(np.mean(self.capture_fps_samples)) if self.capture_fps_samples else None,
                  "events_total": int(sum(self.events.values())),
                  "disclaimer": "Research prototype. Levels and events are model outputs on image-plane cues; they are not verified "
                                "incidents. No automatic braking or steering; no guarantee of collision avoidance."})
        return d


def save_report(stats: SessionStats, events: list[dict], root: Path = SESSIONS_DIR) -> dict[str, Path]:
    d = Path(root) / stats.session_id
    d.mkdir(parents=True, exist_ok=True)
    rep = {"summary": stats.summary(), "events": events}
    pj = d / "session_report.json"
    pj.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    pp = d / "session_report.pdf"
    pp.write_bytes(report_pdf(rep))
    return {"json": pj, "pdf": pp}


def _l(s) -> str:
    return str(s).encode("latin-1", "replace").decode("latin-1")


def report_pdf(rep: dict) -> bytes:
    from fpdf import FPDF
    s = rep["summary"]
    pdf = FPDF()
    pdf.set_auto_page_break(True, 14)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 9, _l(f"SafeDrive AI - Driving session {s['session_id']}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    fmt = lambda v, n=1: "-" if v is None else f"{v:.{n}f}"  # noqa: E731
    rows = [("Started / ended", f"{s['started']} / {s['ended']}"), ("Duration", f"{s['duration_s'] / 60:.1f} min"),
            ("Source / model / profile / device", f"{s['source']} / {s['model']} / {s['profile']} / {s['device']}"),
            ("Level onsets (CAUTION / WARNING / HIGH / UNCERTAIN)",
             " / ".join(str(s['level_onsets'][k]) for k in ("CAUTION", "WARNING", "HIGH RISK", "UNCERTAIN"))),
            ("Alarms", str(s["alarms"])),
            ("Events (risk / warning / potential near-miss)", " / ".join(str(s['events'][k]) for k in ("risk_event", "warning_event", "potential_near_miss"))),
            ("Clips saved / failed", f"{s['clips_saved']} / {s['clips_failed']}"),
            ("Audio played / suppressed", f"{s['audio_played']} / {s['audio_suppressed']}"),
            ("Processed steps per s (target 10)", fmt(s["processed_steps_per_s"])),
            ("Capture FPS (mean)", fmt(s["capture_fps_mean"])),
            ("Step time p50 / p95 (ms)", f"{fmt(s['step_ms_p50'])} / {fmt(s['step_ms_p95'])}"),
            ("End-to-end latency p50 / p95 (ms)", f"{fmt(s['e2e_ms_p50'])} / {fmt(s['e2e_ms_p95'])}"),
            ("Camera interruptions / reconnects", f"{s['camera_interruptions']} / {s['camera_reconnects']}"),
            ("Degraded-input steps (outputs suppressed)", str(s.get("degraded_steps", 0)))]
    for k, v in rows:
        pdf.cell(80, 6, _l(k))
        pdf.multi_cell(0, 6, _l(v), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "Events", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    for e in rep["events"][:200]:
        pdf.multi_cell(0, 4.2, _l(f"[{e['category']}] t={e['t_stream']:.1f}s peak P={e['p_peak']:.2f} ({e['level_peak']}) "
                                  f"clip: {e['clip_status']} - {e['explanation']}"), new_x="LMARGIN", new_y="NEXT")
    if not rep["events"]:
        pdf.cell(0, 6, "No events recorded.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _l(s["disclaimer"]), new_x="LMARGIN", new_y="NEXT")
    return bytes(io.BytesIO(pdf.output()).getvalue())
