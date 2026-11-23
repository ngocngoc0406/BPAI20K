"""LiveController: capture thread -> inference thread (10 Hz grid, newest frame only) -> shared state for the UI,
with event recording, audio warnings and driving-session statistics. One controller per Streamlit server process.
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import psutil

from src.detection.detector import COCO_ROAD
from src.live.audio import AudioAlerter, AudioConfig
from src.live.capture import CaptureThread
from src.live.engine import LiveEngine
from src.live.recorder import EventRecorder, RecorderConfig, new_event
from src.live.session import SESSIONS_DIR, SessionStats, save_report
from src.preprocessing.frame_extractor import resize_width
from src.realworld.session_log import SessionVideoWriter, SystemLogger, TimelineLogger, blur_people

GAP_RESET_S = 1.0  # a capture gap longer than this resets the stream state (tracks / motion history are no longer valid)
DEGRADED_INTERVAL_S = 0.25  # median input interval above this (models assume 0.1 s steps) -> DEGRADED INPUT, outputs suppressed


def lead_object(rows: np.ndarray | None) -> tuple[dict | None, bool]:
    """Largest road user, and whether a road user is in the (fixed, image-plane) driving corridor."""
    if rows is None or len(rows) == 0:
        return None, False
    area = (rows[:, 6] - rows[:, 4]) * (rows[:, 7] - rows[:, 5])
    r = rows[int(np.argmax(area))]
    cx, cy = (rows[:, 4] + rows[:, 6]) / 2, (rows[:, 5] + rows[:, 7]) / 2
    in_corr = bool(np.any((np.abs(cx - 0.5) < 0.2) & (cy > 0.35)))
    return {"track_id": int(r[1]), "class": COCO_ROAD.get(int(r[2]), str(int(r[2]))), "confidence": float(r[3]),
            "box": [float(x) for x in r[4:8]], "area_fraction": float(area.max())}, in_corr


class LiveController:
    def __init__(self):
        self.lock = threading.Lock()
        self.capture: CaptureThread | None = None
        self.engine: LiveEngine | None = None
        self.recorder: EventRecorder | None = None
        self.audio: AudioAlerter | None = None
        self.session: SessionStats | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.state: dict = {"running": False, "error": ""}
        self.latest_frame: np.ndarray | None = None
        self.record_events = True
        self._episode = {"caution_run": 0, "risk_recorded": False, "alarm_active": False}
        self._prev_level = "NORMAL"
        self._proc = psutil.Process()
        self.metrics: list[dict] = []   # periodic samples for evaluation (bounded)
        self.trace: list | None = None  # optional per-prediction trace (t, p, alarm, wall) for fidelity evaluation
        self.timeline: TimelineLogger | None = None   # per-step log of the current session (v3.3 real-world evaluation)
        self.video: SessionVideoWriter | None = None  # optional consented full-session video
        self.trip_meta = None
        self.sessions_root = SESSIONS_DIR

    # ---------------- lifecycle ----------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, source, model_dir: str | Path, profile: str = "balanced", device: str | None = None,
              rec_cfg: RecorderConfig | None = None, audio_cfg: AudioConfig | None = None, record_events: bool = True,
              audio_backend: str = "auto", store_root: str | Path | None = None) -> None:
        with self.lock:
            if self.running:
                raise RuntimeError("The live assistant is already running. Stop it first.")
            self._stop.clear()
            self.state = {"running": False, "error": "", "status": "loading model"}
            engine = LiveEngine(model_dir, profile, device)  # load before opening the camera
            cap = CaptureThread(source)
            cap.start()  # raises SourceError / DeviceBusyError with a clear message
            self.capture, self.engine = cap, engine
            self._model_dir = model_dir
            self.recorder = EventRecorder(rec_cfg, store_root) if store_root else EventRecorder(rec_cfg)
            self.audio = AudioAlerter(audio_cfg, backend=audio_backend)
            self.record_events = record_events
            self.state = {"running": True, "error": "", "status": "running", "source": source.key, "model": Path(str(model_dir)).name,
                          "profile": engine.profile, "device": engine.device, "audio": self.audio.reason if not self.audio.available else "ok"}
            self._thread = threading.Thread(target=self._loop, daemon=True, name="live-inference")
            self._thread.start()

    def stop(self) -> dict | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(10)
        if self.capture is not None:
            self.capture.stop()
        if self.recorder is not None:
            self.recorder.flush()
        rep = self.stop_session() if self.session else None
        self.state["running"] = False
        self.state["status"] = "stopped"
        return rep

    # ---------------- session ----------------
    def start_session(self, meta=None, sessions_root: str | Path | None = None) -> SessionStats:
        """Start a session. `meta` (src.realworld.trip_meta.TripMeta, validated, consent-checked) is DECLARED by the user;
        its session_id is overwritten with the new session's id. A per-step timeline is always written; the full-session
        video only when meta.record_full_video is set (requires consent)."""
        if self.session is not None:
            return self.session
        from src.realworld.trip_meta import save_meta
        st = self.state
        sess = SessionStats(source=st.get("source", ""), model=st.get("model", ""), profile=st.get("profile", ""), device=st.get("device", ""))
        root = Path(sessions_root) if sessions_root else self.sessions_root
        if meta is not None:
            meta.validate()  # raises ConsentError before anything is recorded
            if meta.data_kind == "real_driving":  # v3.4.3: complete pre-session checklist (while parked) + re-checked hashes
                from src.realworld import checklist
                checklist.require_ready(getattr(self, "_model_dir", "models/safedrive_v3"), getattr(self, "checklist_dir", None))
            meta.session_id = sess.session_id
            save_meta(meta, root)
        self.trip_meta = meta
        from src.realworld import provenance
        provenance.save_session(root / sess.session_id, self._model_dir, self.engine.profile if self.engine else "")  # v3.4
        self.timeline = TimelineLogger(root / sess.session_id)
        self.syslog = SystemLogger(root / sess.session_id)
        self.video = SessionVideoWriter(root / sess.session_id) if (meta is not None and meta.record_full_video) else None
        self._sess_root = root
        if self.capture is not None:
            self._sess_base = (self.capture.stats.interruptions, self.capture.stats.reconnects)
        self.session = sess
        return self.session

    def stop_session(self) -> dict | None:
        s = self.session
        if s is None:
            return None
        if self.recorder is not None:
            self.recorder.flush()
        s.wall_end = time.time()
        s.ended = time.strftime("%Y-%m-%dT%H:%M:%S")
        if self.capture is not None:
            b = getattr(self, "_sess_base", (0, 0))
            s.camera_interruptions = self.capture.stats.interruptions - b[0]
            s.camera_reconnects = self.capture.stats.reconnects - b[1]
        self.session = None  # stop logging before closing the writers
        video_info = None
        if self.timeline is not None:
            self.timeline.flush()
        if getattr(self, "syslog", None) is not None:
            self.syslog.close()
            self.syslog = None
        if self.video is not None:
            video_info = self.video.close()
            self.video = None
        evs = [asdict(e) for e in (self.recorder.done if self.recorder else []) if e.session_id == s.session_id]
        s.clips_saved = sum(e["clip_status"] == "saved" for e in evs)
        s.clips_failed = sum(e["clip_status"] != "saved" for e in evs)
        paths = save_report(s, evs, getattr(self, "_sess_root", SESSIONS_DIR))
        if self.timeline is not None:
            paths["timeline"] = self.timeline.path
        self.timeline = None
        return {"summary": s.summary(), "paths": {k: str(v) for k, v in paths.items()}, "events": evs, "video": video_info}

    # ---------------- inference loop ----------------
    def _loop(self) -> None:
        eng, cap = self.engine, self.capture
        period = 1.0 / eng.fps
        next_t, last_id, last_wall = time.time(), None, None
        self._intervals: list[float] = []
        self._t_start = time.time()
        sample_t = time.time()
        try:
            while not self._stop.is_set():
                if cap.stats.state == "failed":
                    self.state.update(error=f"Camera failed: {cap.stats.error}", status="camera failed")
                    break
                if cap.stats.state == "ended" and cap.latest() is not None and cap.latest()[0] == last_id:
                    self.state.update(status="demo video ended")
                    break
                now = time.time()
                if now < next_t:
                    time.sleep(min(next_t - now, 0.01))
                    continue
                item = cap.latest()
                if item is None or item[0] == last_id:
                    stalled = time.time() - (last_wall or self._t_start)
                    if stalled > 2.0:  # no new frame for > 2 s: tell the user instead of showing a stale level
                        self.state.update(stalled_s=stalled, degraded=True,
                                          status=f"no new frames from the source for {stalled:.0f} s "
                                                 "(camera busy in another application, disconnected, or end of demo video)")
                    time.sleep(0.003)
                    continue
                self.state["stalled_s"] = 0.0
                fid, t_cap, frame = item
                if last_wall is not None and t_cap - last_wall > GAP_RESET_S:
                    eng.reset()  # stream interrupted: tracks / motion history no longer valid; warm-up restarts
                    self._episode = {"caution_run": 0, "risk_recorded": False, "alarm_active": False}
                if last_wall is not None:
                    self._intervals.append(t_cap - last_wall)
                    self._intervals = self._intervals[-10:]
                degraded = len(self._intervals) >= 1 and float(np.median(self._intervals)) > DEGRADED_INTERVAL_S
                last_id, last_wall = fid, t_cap
                res = eng.process(frame, t_cap)
                if self.trace is not None and res.predicted:
                    self.trace.append((res.t, res.p, res.alarm, t_cap))
                done = time.time()
                e2e = 1000 * (done - t_cap)
                if degraded:
                    # input slower than the 10 Hz the models were trained for: outputs are not valid -> no audio, no events
                    self._prev_level = "NORMAL"
                    self._episode = {"caution_run": 0, "risk_recorded": False, "alarm_active": False}
                    if self.session is not None:
                        self.session.degraded_steps += 1
                else:
                    self._events_and_audio(res)
                if self.recorder is not None:
                    self.recorder.push(res.t, res.frame)
                tl, vw = self.timeline, self.video
                if tl is not None and self.session is not None:
                    _, in_corr = lead_object(res.rows)
                    tl.log({"step": res.step, "t_stream": round(res.t, 3), "wall": round(t_cap, 3), "level": res.level,
                            "p": "" if not res.predicted else round(float(res.p), 5), "risk_score": round(float(res.risk_score), 4),
                            "mi": round(float(res.mutual_info), 5), "alarm": int(bool(res.alarm)), "degraded": int(degraded),
                            "capture_fps": round(cap.stats.capture_fps(), 2), "step_ms": round(res.timings_ms["total"], 1),
                            "e2e_ms": round(e2e, 1), "n_objects": res.n_objects, "lead_in_corridor": int(in_corr),
                            "rss_mb": round(self._proc.memory_info().rss / 1e6, 1) if res.step % 50 == 0 else "",
                            "camera_state": cap.stats.state, "camera_interruptions": cap.stats.interruptions})
                    if vw is not None:
                        raw = resize_width(frame, eng.width)
                        vw.push(res.t, blur_people(raw, res.rows) if getattr(self.trip_meta, "blur_people", True) else raw)
                if self.session is not None:
                    self.session.on_step(res, e2e, done)
                    self.session.capture_fps_samples = (self.session.capture_fps_samples + [cap.stats.capture_fps()])[-2000:]
                with self.lock:
                    self.latest_frame = res.frame
                    self.state.update(status="running", degraded=degraded,
                                      input_interval_s=float(np.median(self._intervals)) if self._intervals else None,
                                      t=res.t, level=res.level, p=res.p, risk_score=res.risk_score, mi=res.mutual_info,
                                      objects=res.n_objects, step_ms=res.timings_ms["total"], timings=res.timings_ms, e2e_ms=e2e,
                                      capture_fps=cap.stats.capture_fps(), camera=cap.stats.state, interruptions=cap.stats.interruptions,
                                      pending_events=len(self.recorder.pending) if self.recorder else 0,
                                      recorder_errors=list(self.recorder.errors[-3:]) if self.recorder else [], step=res.step)
                if done - sample_t >= 5.0:
                    sample_t = done
                    self.metrics.append({"t": done, "rss_mb": self._proc.memory_info().rss / 1e6, "cpu_percent": self._proc.cpu_percent(None),
                                         "capture_fps": cap.stats.capture_fps(), "step": res.step, "e2e_ms": e2e,
                                         "step_ms": res.timings_ms["total"], "interruptions": cap.stats.interruptions})
                    self.metrics = self.metrics[-5000:]
                next_t = max(next_t + period, time.time() - period)  # keep the 10 Hz grid; never burst to catch up
        except Exception as e:  # noqa: BLE001
            self.state.update(error=f"Inference error: {e!r}", status="error", trace=traceback.format_exc()[-1500:])
        finally:
            cap.stop()  # always release the camera
            self.state["running"] = False

    def _events_and_audio(self, res) -> None:
        if not res.predicted:
            return
        eng, ep = self.engine, self._episode
        lead, in_corr = lead_object(res.rows)
        caution = eng.caution_threshold()
        past_warmup = res.t >= eng.warmup - 1e-6
        # --- audio on level onsets ---
        if res.level != self._prev_level and res.level in ("CAUTION", "WARNING", "HIGH RISK") and past_warmup:
            rec = self.audio.alert(res.level, in_corr)
            self.state["last_audio"] = rec
            if self.timeline is not None:
                self.timeline.log_audio(rec, res.t)
            if self.session is not None:
                if rec["played"]:
                    self.session.audio_played += 1
                else:
                    self.session.audio_suppressed += 1
        self._prev_level = res.level
        # --- events ---
        active = self.recorder.active_event() if self.recorder else None
        if active is not None and res.p > active.p_peak:  # keep the episode peak up to date while collecting post frames
            active.p_peak, active.level_peak, active.t_peak = res.p, res.level, res.t
            active.factors = list(getattr(res.assessment, "triggering_factors", []) or [])
            active.explanation = getattr(res.assessment, "explanation", "")
        if active is not None:
            active.t_end = res.t
            if active.category == "warning_event" and res.level == "HIGH RISK":
                active.category = active.type = "potential_near_miss"
                if self.session is not None and active.session_id == self.session.session_id:
                    self.session.events["warning_event"] -= 1
                    self.session.events["potential_near_miss"] += 1
        if res.p >= caution and past_warmup:
            ep["caution_run"] += 1
        else:
            ep.update(caution_run=0, risk_recorded=False, alarm_active=False)
        if not (self.record_events and self.recorder):
            return
        sess = self.session
        sid = sess.session_id if sess else time.strftime("unsessioned-%Y%m%d")

        def count(cat: str, delta: int = 1) -> None:
            if sess is not None:
                sess.events[cat] += delta
        if res.alarm:
            ep["alarm_active"] = True
            cat = "potential_near_miss" if res.level == "HIGH RISK" else "warning_event"
            if active is not None and active.category in ("risk_event",):
                active.category = active.type = cat  # escalate the ongoing episode instead of duplicating it
                count(cat)
                count("risk_event", -1)
            elif active is None:
                ev = new_event(sid, cat, res.t, res, self.state.get("model", ""), eng.profile)
                ev.lead_object = lead
                self.recorder.trigger(ev)
                count(cat)
        elif (self.recorder.cfg.record_risk_events and ep["caution_run"] >= eng.risk.cfg.consecutive_steps
              and not ep["risk_recorded"] and not ep["alarm_active"] and active is None):
            ep["risk_recorded"] = True
            ev = new_event(sid, "risk_event", res.t, res, self.state.get("model", ""), eng.profile)
            ev.lead_object = lead
            self.recorder.trigger(ev)
            count("risk_event")

    def snapshot(self) -> tuple[dict, np.ndarray | None]:
        with self.lock:
            return dict(self.state), self.latest_frame
