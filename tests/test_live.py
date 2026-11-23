"""v3.2 live assistant: alarm equivalence, live-vs-offline equivalence, capture lifecycle / reconnect, recording,
audio anti-repetition, sessions, controller end-to-end, manual collision confirmation."""
from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from src.evaluation.early_warning import alarm_onsets
from src.live.audio import MESSAGES, AudioAlerter, AudioConfig, tone_wav
from src.live.capture import CaptureThread, DeviceBusyError, active_devices
from src.live.engine import AlarmState
from src.live.recorder import EventRecorder, LiveEvent, RecorderConfig
from src.live.session import SessionStats, report_pdf
from src.live.sources import SourceError, VideoFileSource
from src.utils.config import resolve

V3 = resolve("models/safedrive_v3")


def test_alarm_state_matches_offline_rule():
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = rng.random(200)
        t = np.arange(200) / 5.0
        st = AlarmState(3)
        live = [t[i] for i in range(200) if st.update(s[i], 0.7)]
        assert live == alarm_onsets(t, s, 0.7, 3)


def test_live_engine_equals_offline_pipeline(zoom_video):
    from src.live.engine import LiveEngine
    from src.pipeline import SafeDrivePipeline
    from src.preprocessing.frame_sampler import sample_on_grid
    from src.preprocessing.video_reader import iter_frames
    off = SafeDrivePipeline(V3, device="cpu").analyze(zoom_video, keep_frames=False)
    eng = LiveEngine(V3, device="cpu")
    ps, alarms = [], []
    for _, _, _, f in sample_on_grid(iter_frames(zoom_video), 10.0):
        r = eng.process(f)
        if r.predicted:
            ps.append(r.p)
        if r.alarm:
            alarms.append(round(r.t, 2))
    po = off.timeline.p_mean.to_numpy()
    assert len(ps) == len(po)
    assert np.max(np.abs(np.array(ps) - po)) < 1e-6
    assert alarms == [round(a, 2) for a in off.alarms]


def test_capture_latest_frame_and_single_owner(zoom_video):
    c = CaptureThread(VideoFileSource(str(zoom_video), realtime=True, loop=True))
    c.start()
    try:
        time.sleep(0.5)
        a = c.latest()
        assert a is not None and a[2].ndim == 3
        with pytest.raises(DeviceBusyError):
            CaptureThread(VideoFileSource(str(zoom_video))).start()
        assert any("file:" in k for k in active_devices())
    finally:
        c.stop()
    assert not any(str(zoom_video.name) in k for k in active_devices())
    assert c.stats.state == "stopped"


def test_missing_source_raises():
    with pytest.raises(SourceError):
        CaptureThread(VideoFileSource("does_not_exist.mp4")).start()


class FlakySource:
    """Delivers frames, then fails `fail_reads` times (disconnect), then works again after reopening."""

    def __init__(self, fail_after=20, fail_reads=30, reopen_ok=True):
        self.n, self.fail_after, self.fail_reads, self.reopen_ok, self.opened = 0, fail_after, fail_reads, reopen_ok, 0
        self.img = np.full((90, 160, 3), 100, np.uint8)

    key = "flaky:0"

    def open(self):
        if self.opened and not self.reopen_ok:
            raise SourceError("device gone")
        self.opened += 1

    def read(self):
        self.n += 1
        time.sleep(0.005)
        if self.fail_after < self.n <= self.fail_after + self.fail_reads:
            return False, None, time.time()
        return True, self.img, time.time()

    def release(self):
        pass


def test_capture_reconnects_and_counts_interruptions():
    src = FlakySource()
    c = CaptureThread(src, max_failures=5, reconnect_backoff_s=0.05)
    c.start()
    time.sleep(1.5)
    c.stop()
    assert c.stats.interruptions >= 1 and c.stats.reconnects >= 1 and c.stats.frames > 20


def test_capture_gives_up_and_releases():
    c = CaptureThread(FlakySource(fail_after=5, fail_reads=10**9, reopen_ok=False), max_failures=3, reconnect_backoff_s=0.05,
                      max_reconnect_s=0.3)
    c.start()
    time.sleep(1.5)
    assert c.stats.state == "failed" and "device gone" in c.stats.error
    assert "flaky:0" not in active_devices()
    c.stop()


def _event(eid="e1"):
    return LiveEvent(eid, "s1", "warning_event", "now", 3.0, "WARNING", 0.95, 0.01, [], "x", None, "v3", "balanced", 0, 0)


def test_recorder_writes_pre_and_post_clip(tmp_path):
    rec = EventRecorder(RecorderConfig(pre_s=2, post_s=1, min_free_mb=0), tmp_path)
    img = np.zeros((120, 160, 3), np.uint8)
    for i in range(30):
        rec.push(i / 10, cv2.putText(img.copy(), str(i), (10, 60), 0, 1, (255, 255, 255), 2))
    ev = rec.trigger(_event())
    for i in range(30, 45):
        rec.push(i / 10, img)
    rec.flush()
    assert ev.clip_status == "saved" and ev.clip_frames == 20 + 10
    cap = cv2.VideoCapture(ev.clip)
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
    cap.release()
    assert '"warning_event"' in (tmp_path / "events.jsonl").read_text(encoding="utf-8")


def test_recorder_disk_full_and_quota_report_errors_without_fake_clip(tmp_path):
    rec = EventRecorder(RecorderConfig(pre_s=1, post_s=1, min_free_mb=10**12), tmp_path)  # simulate an almost full disk
    img = np.zeros((60, 80, 3), np.uint8)
    for i in range(15):
        rec.push(i / 10, img)
    ev = rec.trigger(_event("e2"))
    for i in range(15, 30):
        rec.push(i / 10, img)
    rec.flush()
    assert ev.clip is None and ev.clip_status.startswith("not saved") and "disk space" in ev.clip_status
    assert not list((tmp_path / "clips").glob("e2*.mp4"))
    rec2 = EventRecorder(RecorderConfig(pre_s=1, post_s=1, min_free_mb=0, max_store_mb=-1), tmp_path)
    for i in range(15):
        rec2.push(i / 10, img)
    ev2 = rec2.trigger(_event("e3"))
    rec2.flush()
    assert "quota" in ev2.clip_status


def test_rolling_buffer_is_bounded():
    from src.live.recorder import RollingBuffer
    rb = RollingBuffer(RecorderConfig(pre_s=1, fps=10, max_buffer_mb=0.001))
    noise = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    for i in range(100):
        rb.push(i, noise)
    assert len(rb.frames) <= 10 and rb.bytes <= 0.001e6 + 1e6  # count bound, and the byte bound evicts old frames


def test_audio_cooldown_and_escalation():
    a = AudioAlerter(AudioConfig(cooldown_s={"CAUTION": 20, "WARNING": 8, "HIGH RISK": 5}, global_gap_s=2), backend="none")
    assert not a.available and "disabled" in a.reason
    t = 1000.0
    assert a.alert("CAUTION", now=t)["reason"].startswith("unavailable")      # decision ok, but no backend
    assert a.decide("CAUTION", t + 1)[0] is False                            # global gap
    assert a.decide("CAUTION", t + 5)[1] == "level cooldown"                 # same level within its cooldown
    assert a.decide("WARNING", t + 3)[0] is True                             # escalation after the global gap
    a.alert("WARNING", now=t + 3)
    assert a.decide("CAUTION", t + 25)[0] is True                            # both cooldowns over (CAUTION 20 s, WARNING 8 s)
    a.alert("HIGH RISK", now=t + 10)
    assert a.decide("WARNING", t + 12)[1].startswith("higher level")         # own cooldown over, but HIGH RISK 2 s ago
    assert a.decide("NORMAL", t + 100)[0] is False
    assert a.message("WARNING", True) == MESSAGES["WARNING_LEAD"] and a.message("WARNING", False) != MESSAGES["WARNING_LEAD"]
    assert "va chạm" in a.message("HIGH RISK", False) and "sẽ" not in a.message("HIGH RISK", False)  # no certainty claim
    assert tone_wav([(440, 0.1)], 0.5)[:4] == b"RIFF"


def test_session_report_has_no_score():
    s = SessionStats(source="demo", model="v3", profile="balanced", device="cpu")

    class R:
        timings_ms = {"total": 20.0}
        predicted, level, alarm = True, "WARNING", True
    for i in range(5):
        s.on_step(R(), 40.0, 1000 + i * 0.1)
    sm = s.summary()
    assert sm["alarms"] == 5 and sm["level_onsets"]["WARNING"] == 1 and "score" not in " ".join(sm)
    assert report_pdf({"summary": sm, "events": []})[:4] == b"%PDF"


def test_controller_end_to_end_demo(zoom_video, tmp_path):
    from src.live.controller import LiveController
    c = LiveController()
    c.start(VideoFileSource(str(zoom_video), realtime=True, loop=True), V3, "sensitive", "cpu",
            RecorderConfig(pre_s=1, post_s=1, min_free_mb=0), AudioConfig(enabled=False), audio_backend="none", store_root=tmp_path)
    c.start_session(sessions_root=tmp_path / "sessions")  # never write test sessions into data/sessions
    time.sleep(4)
    s, frame = c.snapshot()
    rep = c.stop()
    assert s["running"] and frame is not None and s["step"] > 10 and not s["error"]
    assert rep["summary"]["steps"] > 10 and rep["summary"]["processed_steps_per_s"] > 0
    assert not c.running and not any(zoom_video.name in k for k in active_devices())
    with pytest.raises(RuntimeError):  # second start while running is refused
        c2 = LiveController()
        c2.start(VideoFileSource(str(zoom_video), realtime=True, loop=True), V3, audio_backend="none", store_root=tmp_path)
        c2.start(VideoFileSource(str(zoom_video), realtime=True, loop=True), V3, audio_backend="none", store_root=tmp_path)
    c2.stop()


def test_manual_collision_confirmation_requires_evidence(tmp_path):
    from src.safety.near_miss import NearMissMemory
    mem = NearMissMemory(tmp_path)
    (tmp_path / "events.jsonl").write_text('{"event_id": "x1", "type": "warning_event", "trip_id": "t"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        mem.confirm_collision("x1", "")
    assert mem.confirm_collision("x1", "visible contact at 12.3 s")
    assert mem.all()[0]["category"] == "confirmed_collision"


class SlowSource:
    """Delivers a frame only every 0.5 s (like a camera shared with another app, ~2 fps)."""
    key = "slow:0"

    def __init__(self):
        self.img = np.full((360, 640, 3), 90, np.uint8)

    def open(self):
        pass

    def read(self):
        time.sleep(0.5)
        return True, self.img, time.time()

    def release(self):
        pass


def test_degraded_input_suppresses_outputs(tmp_path):
    from src.live.controller import LiveController
    c = LiveController()
    c.start(SlowSource(), V3, "sensitive", "cpu", RecorderConfig(min_free_mb=0), AudioConfig(enabled=False),
            audio_backend="none", store_root=tmp_path)
    c.start_session(sessions_root=tmp_path / "sessions")  # never write test sessions into data/sessions
    time.sleep(4)
    s, _ = c.snapshot()
    rep = c.stop()
    assert s["degraded"] is True and s["input_interval_s"] > 0.25
    assert rep["summary"]["degraded_steps"] >= 1 and rep["summary"]["events_total"] == 0


def test_demo_video_without_loop_ends_cleanly(zoom_video):
    """Regression: a non-looping demo video must END (not 'reconnect' and replay forever)."""
    from src.live.controller import LiveController
    c = LiveController()
    c.start(VideoFileSource(str(zoom_video), realtime=True, loop=False), V3, "balanced", "cpu", audio_backend="none",
            record_events=False, audio_cfg=AudioConfig(enabled=False))
    t0 = time.time()
    while c.running and time.time() - t0 < 20:
        time.sleep(0.2)
    assert not c.running and time.time() - t0 < 15          # the clip is 4 s long
    assert c.capture.stats.state == "ended" and c.capture.stats.interruptions == 0
    assert c.state["status"] == "demo video ended"
    c.stop()
