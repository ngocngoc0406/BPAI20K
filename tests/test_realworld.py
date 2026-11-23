"""v3.3 real-world validation tooling: consent gating, declared metadata, annotation rules, agreement, metrics from logs,
session deletion, controller timeline + consented full-session video."""
from __future__ import annotations

import csv
import json
import time

import numpy as np
import pytest

from src.realworld import annotation as ann
from src.realworld import metrics as rwm
from src.realworld.session_log import COLUMNS, SessionVideoWriter, TimelineLogger, blur_people
from src.realworld.trip_meta import ConsentError, TripMeta, delete_session, load_meta, save_meta
from src.utils.config import resolve

V3 = resolve("models/safedrive_v3")


def test_real_driving_requires_consent_supervisor_and_setup_before_driving(tmp_path):
    with pytest.raises(ConsentError):
        save_meta(TripMeta("s1", data_kind="real_driving"), tmp_path)
    with pytest.raises(ConsentError):
        save_meta(TripMeta("s1", data_kind="real_driving", driver_consent=True, supervisor_present=True), tmp_path)
    m = TripMeta("s1", data_kind="real_driving", driver_consent=True, supervisor_present=True, configured_before_driving=True)
    save_meta(m, tmp_path)
    assert load_meta("s1", tmp_path).data_kind == "real_driving"
    with pytest.raises(ConsentError):  # full video additionally needs informed occupants
        save_meta(TripMeta("s2", data_kind="live_camera_test", record_full_video=True, driver_consent=True), tmp_path)


def test_conditions_default_to_unknown_never_inferred(tmp_path):
    m = TripMeta("s3")
    assert (m.lighting, m.weather, m.traffic, m.road_types) == ("unknown", "unknown", "unknown", [])
    with pytest.raises(ValueError):
        save_meta(TripMeta("s4", weather="sunny-ish"), tmp_path)


def E(sid, who, cat, t0, t1, ct=None, **kw):
    """An annotation with valid visual evidence for event categories."""
    ev = dict(evidence="lead car brake lights, gap closes", evidence_frames=[ct if ct is not None else t0]) if cat in ann.EVENT_CATEGORIES else {}
    return ann.Annotation(sid, who, cat, t0, t1, ct, **{**ev, **kw})


def test_annotation_rules_v34(tmp_path):
    with pytest.raises(ann.AnnotationError):
        ann.add(tmp_path, ann.Annotation("s", "A", "collision", 1, 2, 1.5))  # no visual evidence
    with pytest.raises(ann.AnnotationError):
        ann.add(tmp_path, ann.Annotation("s", "A", "near_miss", 1, 4, None, evidence="gap", evidence_frames=[2]))  # no event time
    with pytest.raises(ann.AnnotationError):
        ann.add(tmp_path, E("s", "A", "risk_event", 5, 2, 3))
    with pytest.raises(ann.AnnotationError):
        ann.add(tmp_path, E("s", "A", "risk_event", 1, 2, 3))  # event time outside the situation
    with pytest.raises(ann.AnnotationError):
        ann.add(tmp_path, ann.Annotation("s", "A", "near_miss", 1, 4, 3, evidence="close gap", evidence_frames=[60]))
    a = ann.add(tmp_path, E("s", "A", "near_miss", 1, 4, 3, tags=["lead_braking"]))
    ann.add(tmp_path, ann.Annotation("s", "B", "uncertain", 1, 4))
    assert ann.annotators(tmp_path) == ["A", "B"] and len(ann.load(tmp_path, "A")) == 1
    assert ann.delete(tmp_path, "A", a.annot_id) and ann.load(tmp_path, "A") == []
    old = ann.Annotation("s", "A", "warning_worthy", 1, 2)  # v3.3 label is mapped, original kept
    assert old.category == "risk_event" and old.legacy_label == "warning_worthy"


def test_agreement_kappa_offsets_and_unmatched():
    A = [E("s", "A", c, t, t + 3, t + 2) for c, t in [("risk_event", 10), ("normal", 30), ("risk_event", 60), ("near_miss", 90)]]
    B = [E("s", "B", c, t, t + 3, t + 2.5) for c, t in [("risk_event", 10.5), ("normal", 30), ("near_miss", 60), ("uncertain", 200)]]
    r = ann.agreement(A, B)
    assert r["matched"] == 3 and r["only_first"] == 1 and r["only_second"] == 1
    assert abs(r["exact_category_agreement"] - 2 / 3) < 1e-9
    assert ann.cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    assert abs(r["offset_start"]["mean_abs_s"] - 0.5 / 3) < 1e-9
    assert r["offset_event_time"]["n"] == 3 and abs(r["offset_event_time"]["mean_abs_s"] - 2 / 3) < 1e-9  # offsets 1.0, 0.5, 0.5 s
    assert r["confusion"]["near_miss"]["none"] == 1 and r["confusion"]["none"]["uncertain"] == 1
    assert r["kappa_with_unmatched"] < r["kappa_category"]


def test_consensus_never_forces_agreement(tmp_path):
    A = [E("s", "A", "risk_event", 10, 13, 12), E("s", "A", "near_miss", 50, 53, 52), E("s", "A", "risk_event", 80, 83, 82)]
    B = [E("s", "B", "risk_event", 10.2, 13, 12.4), E("s", "B", "risk_event", 50, 53, 52)]
    c = ann.consensus(A, B)
    assert [x["status"] for x in c] == ["confirmed", "disputed", "single"]
    assert abs(c[0]["critical_t"] - 12.2) < 1e-9
    with pytest.raises(ann.AnnotationError):
        ann.add_disagreement(tmp_path, ann.Disagreement("s", c[1]["key"], "A", ""))  # a reason is required
    ann.add_disagreement(tmp_path, ann.Disagreement("s", c[1]["key"], "A", "I saw a hard swerve", "unresolved"))
    assert ann.consensus(A, B, ann.load_disagreements(tmp_path))[1]["status"] == "disputed"
    ann.add_disagreement(tmp_path, ann.Disagreement("s", c[1]["key"], "B", "after review: swerve visible", "agreed_after_discussion", "near_miss"))
    c2 = ann.consensus(A, B, ann.load_disagreements(tmp_path))
    assert c2[1]["status"] == "confirmed" and c2[1]["category"] == "near_miss"


def _write_session(root, sid, alarms_at, hours_steps=3000, meta=None):
    d = root / sid
    tl = TimelineLogger(d)
    on = set()
    for t0 in alarms_at:
        on |= set(range(int(t0 * 10), int(t0 * 10) + 5))
    for i in range(hours_steps):
        tl.log({"step": i, "t_stream": i / 10, "wall": 1000 + i / 10, "level": "WARNING" if i in on else "NORMAL",
                "p": 0.95 if i in on else 0.1, "risk_score": 0, "mi": 0, "alarm": int(i in on), "degraded": 0, "capture_fps": 30,
                "step_ms": 20, "e2e_ms": 25, "n_objects": 2, "lead_in_corridor": 0, "rss_mb": 500 if i % 50 == 0 else ""})
    tl.flush()
    save_meta(meta or TripMeta(sid), root)
    return d


def test_metrics_from_logs(tmp_path):
    d = _write_session(tmp_path, "sx", [20, 40, 100, 150, 160])  # 5 alarms in 300 s
    r = rwm.session_metrics(d)
    assert r["alarms"] == 5 and r["fa_per_hour"] is None  # not annotated: no FA claim
    assert r["integrity"]["missing_steps"] == 0 and r["integrity"]["completeness"] == 1.0
    ann.add(d, E("sx", "A", "risk_event", 18, 23, 22))       # alarm at 20 -> valid, lead 2 s
    ann.add(d, ann.Annotation("sx", "A", "normal", 99, 103))  # alarm at 100 -> false
    ann.add(d, ann.Annotation("sx", "A", "uncertain", 149, 152))  # alarm at 150 -> uncertain
    ann.add(d, E("sx", "A", "near_miss", 250, 260, 255))     # missed
    m = load_meta("sx", tmp_path)
    m.fully_reviewed_by = ["A"]
    save_meta(m, tmp_path)
    r = rwm.session_metrics(d)
    assert r["alarm_status"] == {"valid": 1, "false": 3, "uncertain": 1}
    assert r["warnings_without_hazard"] == 3  # 40, 100 (reviewed normal stretch) and 160
    assert abs(r["fa_per_hour"] - 3 / (3000 * 0.1 / 3600)) < 1e-6
    assert r["warning_recall"] == 0.5 and r["lead_times_s"] == [2.0]
    assert r["warned_ge"]["1s"] == 0.5 and r["warned_ge"]["3s"] == 0.0
    assert r["repeated_warnings"] == 2  # onset gaps 20, 60, 50, 10 s -> two below 30 s
    agg = rwm.aggregate([r])
    assert agg.iloc[0]["warning_recall"] == 0.5


def test_integrity_counts_gaps_and_duplicates():
    import pandas as pd
    tl = pd.DataFrame({"step": [0, 1, 2, 5, 0, 1], "t_stream": [0.0, 0.1, 0.1, 0.5, 0.0, 0.1]})
    r = rwm.frame_integrity(tl)
    assert r["missing_steps"] == 3 and r["duplicate_steps"] == 1 and r["stream_resets"] == 1


def test_delete_session_removes_folder_events_and_clips(tmp_path):
    root, mem = tmp_path / "sessions", tmp_path / "memory"
    _write_session(root, "sdel", [5], hours_steps=100)
    (mem / "clips").mkdir(parents=True)
    clip = mem / "clips" / "e1.mp4"
    clip.write_bytes(b"x")
    (mem / "events.jsonl").write_text(json.dumps({"event_id": "e1", "session_id": "sdel", "clip": str(clip)}) + "\n"
                                      + json.dumps({"event_id": "e2", "session_id": "other"}) + "\n", encoding="utf-8")
    r = delete_session("sdel", root, mem, requested_by="tester", reason="test", log_path=tmp_path / "dellog.jsonl")
    assert "sdel" in (tmp_path / "dellog.jsonl").read_text(encoding="utf-8")
    assert r == {"session_dir": True, "events": 1, "clips": 1}
    assert not (root / "sdel").exists() and not clip.exists()
    assert "e2" in (mem / "events.jsonl").read_text(encoding="utf-8")


def test_blur_people_only_blurs_person_boxes():
    rng = np.random.default_rng(0)
    f = rng.integers(0, 255, (100, 200, 3), dtype=np.uint8)
    rows = np.array([[0, 1, 0, 0.9, 0.0, 0.0, 0.25, 0.5], [0, 2, 2, 0.9, 0.5, 0.5, 1.0, 1.0]])
    out = blur_people(f, rows)
    assert not np.array_equal(out[:50, :50], f[:50, :50])
    assert np.array_equal(out[50:, 100:], f[50:, 100:])


def test_session_video_writer_segments(tmp_path):
    w = SessionVideoWriter(tmp_path, fps=10, segment_s=1.0)
    for i in range(25):
        w.push(i / 10, np.full((96, 128, 3), i * 8, np.uint8))
    info = w.close()
    assert info["frames_written"] + info["frames_dropped"] == 25 and not info["errors"]
    assert len(info["segments"]) == 3 and all((tmp_path / "video" / f"segment_{k:03d}.mp4").exists() for k in range(3))


def test_controller_writes_timeline_and_consented_video(zoom_video, tmp_path):
    from src.live.audio import AudioConfig
    from src.live.controller import LiveController
    from src.live.recorder import RecorderConfig
    from src.live.sources import VideoFileSource
    c = LiveController()
    c.start(VideoFileSource(str(zoom_video), realtime=True, loop=True), V3, "balanced", "cpu",
            RecorderConfig(pre_s=1, post_s=1, min_free_mb=0), AudioConfig(enabled=False), audio_backend="none", store_root=tmp_path / "mem")
    with pytest.raises(ConsentError):
        c.start_session(TripMeta("x", data_kind="real_driving"), sessions_root=tmp_path / "sessions")
    assert c.session is None
    m = TripMeta("x", data_kind="demo_video", record_full_video=True, driver_consent=True, occupants_informed=True)
    s = c.start_session(m, sessions_root=tmp_path / "sessions")
    time.sleep(6)
    rep = c.stop()
    d = tmp_path / "sessions" / s.session_id
    rows = list(csv.DictReader(open(d / "timeline.csv", encoding="utf-8")))
    assert len(rows) > 10 and list(rows[0].keys()) == COLUMNS
    assert rep["video"]["frames_written"] > 10 and (d / "video" / "segment_000.mp4").exists()
    assert load_meta(s.session_id, tmp_path / "sessions").data_kind == "demo_video"
    prov = json.loads((d / "provenance.json").read_text(encoding="utf-8"))  # v3.4: hashes recorded at session start
    assert prov["model_dir"] == "safedrive_v3" and len(prov["threshold_config_sha256"]) == 64
    assert (d / "system_log.csv").exists() and rows[0]["camera_state"]
    r = rwm.session_metrics(d)
    assert r["has_timeline"] and r["hours"] > 0 and r["runtime"]["processed_rate_hz"] > 3


def test_partial_review_marks_unreviewed_alarms_and_gives_no_fa_rate(tmp_path):
    d = _write_session(tmp_path, "sp", [20, 100])
    ann.add(d, ann.Annotation("sp", "A", "normal", 19, 23))  # the reviewer looked at the alarm at 20 s only
    r = rwm.session_metrics(d, annotator="A")
    assert r["alarm_status"] == {"valid": 0, "false": 1, "uncertain": 0, "unreviewed": 1}
    assert r["fa_per_hour"] is None and r["warning_recall"] is None
