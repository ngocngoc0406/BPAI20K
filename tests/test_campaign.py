"""v3.4 data-collection campaign: provenance hashes, frozen reference, group-wise frozen partitions, leakage / duplicate checks,
validation flags, legacy audit (no metadata assigned), consensus metrics with the 'insufficient events' rule, final-test
access logging, privacy review and access tiers."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.realworld import annotation as ann
from src.realworld import campaign, privacy, provenance
from src.realworld.session_log import SessionVideoWriter, TimelineLogger
from src.realworld.trip_meta import ConsentError, TripMeta, save_meta
from src.utils.config import resolve

V3 = resolve("models/safedrive_v3")


@pytest.fixture()
def frozen(tmp_path, monkeypatch):
    ref = {"safedrive_v3": provenance.compute(V3)}
    p = tmp_path / "frozen.json"
    p.write_text(json.dumps(ref), encoding="utf-8")
    monkeypatch.setattr(provenance, "FROZEN_PATH", p)
    monkeypatch.setattr(provenance, "frozen", lambda path=p: json.loads(p.read_text(encoding="utf-8")))
    return ref


def test_provenance_is_deterministic_and_detects_changes(tmp_path):
    a, b = provenance.compute(V3), provenance.compute(V3)
    assert a == b and len(a["model_sha256"]) == 64
    ref = {"safedrive_v3": a}
    assert provenance.check(a, ref)["eligible"]
    bad = {**a, "threshold_config_sha256": "0" * 64}
    c = provenance.check(bad, ref)
    assert not c["eligible"] and not c["thresholds_ok"] and c["model_ok"]
    assert not provenance.check({**a, "model_dir": "other"}, ref)["eligible"]  # no frozen reference
    p = tmp_path / "f.json"
    provenance.freeze(["models/safedrive_v3"], p)
    with pytest.raises(FileExistsError):
        provenance.freeze(["models/safedrive_v3"], p)


def test_campaign_consent_rules(tmp_path):
    with pytest.raises(ConsentError):  # campaign real driving needs the recording-rules confirmation
        save_meta(TripMeta("s", data_kind="real_driving", driver_consent=True, supervisor_present=True,
                           configured_before_driving=True, campaign_id="c1"), tmp_path)
    with pytest.raises(ValueError):
        save_meta(TripMeta("s", driver_id="Nguyen Van A"), tmp_path)  # pseudonyms only


def test_partitions_are_group_wise_frozen_and_manual_needs_reason(tmp_path):
    parts = {campaign.assign_partition(f"driver:D{i:02d}", tmp_path) for i in range(40)}
    assert parts <= set(campaign.PARTITIONS) and len(parts) == 3
    first = campaign.assign_partition("driver:D01", tmp_path)
    assert campaign.assign_partition("driver:D01", tmp_path, manual="train", reason="x") == first  # never reassigned
    with pytest.raises(ValueError):
        campaign.assign_partition("driver:NEW", tmp_path, manual="dev")


def _session(root, sid, meta, alarms=(), steps=3000, video=True, frames=None, seed=0):
    d = root / sid
    tl = TimelineLogger(d)
    on = set()
    for t0 in alarms:
        on |= set(range(int(t0 * 10), int(t0 * 10) + 5))
    for i in range(steps):
        tl.log({"step": i, "t_stream": i / 10, "wall": 1000 + i / 10, "level": "WARNING" if i in on else "NORMAL",
                "p": 0.95 if i in on else 0.1, "alarm": int(i in on), "degraded": 0, "step_ms": 20, "e2e_ms": 25, "capture_fps": 30})
    tl.flush()
    meta.session_id = sid
    save_meta(meta, root)
    if video:
        w = SessionVideoWriter(d, fps=10, segment_s=1000, max_queue=100000)  # burst push in tests (live: 10 Hz)
        rng = np.random.default_rng(seed)
        base = rng.integers(0, 255, (72, 96, 3), dtype=np.uint8)
        for i in range(frames if frames is not None else steps):
            w.push(i / 10, np.roll(base, i, axis=1))
        w.close()
    provenance.save_session(d, V3, "balanced")
    return d


def real(trip, driver, **kw):
    return TripMeta("x", data_kind="real_driving", driver_consent=True, occupants_informed=True, supervisor_present=True,
                    configured_before_driving=True, regulations_checked=True, campaign_id="c1", trip_id=trip, driver_id=driver,
                    record_full_video=True, **kw)


def test_manifest_validation_leakage_and_duplicates(tmp_path, frozen):
    root, cdir = tmp_path / "sessions", tmp_path / "campaign"
    _session(root, "s1", real("T1", "D01"), steps=200, seed=1)
    _session(root, "s2", real("T2", "D01"), steps=200, seed=2)            # same driver -> same partition
    _session(root, "s3", real("T3", "D02"), steps=200, frames=150, seed=3)  # desynchronised video
    _session(root, "s4", TripMeta("x", data_kind="demo_video"), steps=100, video=False)
    (root / "legacy1").mkdir(parents=True)
    (root / "legacy1" / "session_report.json").write_text(json.dumps({"summary": {"source": "camera:0", "duration_s": 30}, "events": []}))
    man = campaign.build_manifest(root, cdir)
    by = {e["session_id"]: e for e in man["sessions"]}
    assert by["s1"]["partition"] == by["s2"]["partition"] != "excluded"
    assert by["s4"]["partition"] == "excluded" and by["legacy1"]["data_kind"] == "legacy_unknown"
    assert any("desynchronised" in i for i in by["s3"]["issues"])
    assert by["s1"]["provenance_check"]["eligible"] and len(by["s1"]["video_segments"][0]["sha256"]) == 64
    assert man["leakage"]["ok"]
    val = campaign.validation_report(man)
    assert val["real_driving_sessions"] == 3 and val["drivers"] == 2 and val["sessions_desynchronised"] == 1 and val["legacy_unknown"] == 1
    # duplicate video across sessions + a forced cross-partition trip -> leakage reported
    import shutil
    shutil.copy(root / "s1" / "video" / "segment_000.mp4", root / "s3" / "video" / "segment_000.mp4")
    parts = json.loads((cdir / "partitions.json").read_text())
    man2 = campaign.build_manifest(root, cdir)
    assert man2["leakage"]["exact_duplicate_videos"] and not man2["leakage"]["ok"]
    for e in man2["sessions"]:
        if e["session_id"] == "s2":
            e["partition"] = next(p for p in campaign.PARTITIONS if p != e["partition"])
    assert any("D01" in x for x in campaign.leakage_check(man2["sessions"])["group_leaks"])
    assert parts == json.loads((cdir / "partitions.json").read_text())  # rebuilding never reassigns


def test_legacy_audit_assigns_no_metadata(tmp_path):
    for sid, src in (("a", "C:/Temp/pytest-of-x/pytest-1/vid0/zoom.mp4"), ("b", "camera:0"), ("c", "D:/x/data/raw/nexar/train/positive/1.mp4")):
        (tmp_path / sid).mkdir()
        (tmp_path / sid / "session_report.json").write_text(json.dumps({"summary": {"source": src}, "events": []}))
    rows = campaign.legacy_audit(tmp_path)
    assert [r["classification"] for r in rows] == ["legacy_unknown"] * 3
    assert rows[0]["observed_origin"].startswith("automated test") and not any((tmp_path / s / "trip_meta.json").exists() for s in "abc")
    assert all(r["official_evaluation"] is False for r in rows)


def _ev(who, cat, t0, t1, ct):
    return ann.Annotation("s", who, cat, t0, t1, ct, evidence="visible close gap", evidence_frames=[ct])


def test_campaign_metrics_consensus_and_insufficient_events(tmp_path, frozen):
    root, cdir = tmp_path / "sessions", tmp_path / "campaign"
    d = _session(root, "r1", real("T1", "D01"), alarms=[20, 40, 100], steps=3600, video=False)
    for who, shift in (("A", 0.0), ("B", 0.3)):
        ann.add(d, _ev(who, "risk_event", 18 + shift, 23, 22))        # confirmed, alarm 20 valid (lead 2 s)
        ann.add(d, _ev(who, "near_miss", 300, 305, 303))              # confirmed, missed
        ann.add(d, ann.Annotation("r1", who, "normal", 95, 105))       # alarm 100 -> false (both reviewed as normal)
    ann.add(d, _ev("A", "risk_event", 38, 42, 41))                     # single -> alarm 40 excluded
    from src.realworld.trip_meta import load_meta
    m = load_meta("r1", root)
    m.fully_reviewed_by = ["A", "B"]
    save_meta(m, root)
    man = campaign.build_manifest(root, cdir)
    part = man["sessions"][0]["partition"]
    met = campaign.campaign_metrics(man, partitions=(part,), root=root, cdir=cdir) if part != "final_test" else \
        campaign.campaign_metrics(man, partitions=(), root=root, cdir=cdir, include_final_test=True, requested_by="t")
    ov = met["overall"]
    assert met["double_annotated_sessions"] == 1
    assert ov["false_alarms"] == 1 and ov["excluded_alarms"] == 1 and ov["alarms"] == 3
    assert abs(ov["fa_per_hour"] - 1 / 0.1) < 1e-6 and ov["fa_per_hour_ci95"][0] < 10 < ov["fa_per_hour_ci95"][1]
    assert ov["by_event_type"]["all_events"]["confirmed_events"] == 2
    assert str(ov["by_event_type"]["all_events"]["recall"]).startswith("insufficient")  # 2 < 10 confirmed events
    agr = campaign.agreement_report(man, root)
    assert agr["sessions"][0]["confirmed"] == 3 and agr["sessions"][0]["single"] == 1


def test_recall_reported_with_enough_confirmed_events(tmp_path, frozen):
    root, cdir = tmp_path / "sessions", tmp_path / "campaign"
    alarms = [10 + 30 * k for k in range(6)]  # 6 of 12 events warned 1 s before the event time
    d = _session(root, "r2", real("T9", "D09"), alarms=alarms, steps=4000, video=False)
    for who in ("A", "B"):
        for k in range(12):
            ann.add(d, _ev(who, "risk_event", 9 + 30 * k, 13 + 30 * k, 11 + 30 * k))
    from src.realworld.trip_meta import load_meta
    m = load_meta("r2", root)
    m.fully_reviewed_by = ["A", "B"]
    save_meta(m, root)
    man = campaign.build_manifest(root, cdir)
    met = campaign.campaign_metrics(man, partitions=tuple(campaign.PARTITIONS)[:2], root=root, cdir=cdir,
                                    include_final_test=True, requested_by="tester")
    o = met["overall"]["by_event_type"]["risk_event"]
    assert o["confirmed_events"] == 12 and o["recall"] == 0.5 and o["warned_ge_1s"] == 0.5 and o["warned_ge_2s"] == 0.0
    assert o["recall_ci95"][0] < 0.5 < o["recall_ci95"][1]
    assert "tester" in (cdir / "final_test_access_log.jsonl").read_text()


def test_privacy_review_access_and_blur(tmp_path, monkeypatch):
    monkeypatch.setattr(privacy, "load_access", lambda path=None: {"raw_access": ["A1"], "coordinators": ["R1"]})
    f = np.full((100, 200, 3), 200, np.uint8)
    f[60:100, 0:100] = np.random.default_rng(0).integers(0, 255, (40, 100, 3), dtype=np.uint8)
    f[0:40, 0:100] = np.random.default_rng(1).integers(0, 255, (40, 100, 3), dtype=np.uint8)
    out = privacy.blur_regions(f, [(2, 10, 10, 90, 90)])  # car: WHOLE box (+8 % padding) blurred since the 2026-11-22 audit
    assert not np.array_equal(out[10:40, 10:90], f[10:40, 10:90]) and not np.array_equal(out[60:90, 10:90], f[60:90, 10:90])
    assert np.array_equal(out[:, 150:], f[:, 150:])  # outside the padded box untouched
    d = tmp_path / "s"
    (d / "shared").mkdir(parents=True)
    for k in range(2):
        (d / "shared" / f"segment_{k:03d}.mp4").write_bytes(b"x")
    with pytest.raises(ValueError):
        privacy.review(d, "R1", True, ["segment_000.mp4"], identifiable_found=False)  # not every segment checked
    assert not privacy.shareable(d, policy_path=tmp_path / "pol.json")
    privacy.review(d, "R1", True, ["segment_000.mp4", "segment_001.mp4"], identifiable_found=False)
    assert privacy.shareable(d, policy_path=tmp_path / "pol.json")
    monkeypatch.undo()
    acc = tmp_path / "access.json"
    acc.write_text(json.dumps({"raw_access": ["A1"], "coordinators": ["C1"]}))
    assert privacy.can_view_raw("A1", acc) and not privacy.can_view_raw("B2", acc) and privacy.is_coordinator("C1", acc)


def test_make_shared_copy_runs_on_real_video(tmp_path, zoom_video):
    import shutil
    d = tmp_path / "s"
    (d / "video").mkdir(parents=True)
    shutil.copy(zoom_video, d / "video" / "segment_000.mp4")
    log = privacy.make_shared_copy(d, device="cpu")
    assert (d / "shared" / "segment_000.mp4").exists() and log["segments"][0]["frames"] > 10
    assert not privacy.shareable(d)  # a new copy is never shareable before a human review


def test_export_only_approved_copy_by_coordinator(tmp_path, monkeypatch):
    monkeypatch.setattr(privacy, "load_access", lambda path=None: {"raw_access": ["A1"], "coordinators": ["C1"]})
    d = tmp_path / "s1"
    (d / "shared").mkdir(parents=True)
    (d / "video").mkdir()
    (d / "video" / "segment_000.mp4").write_bytes(b"RAW")
    (d / "shared" / "segment_000.mp4").write_bytes(b"BLURRED")
    log = tmp_path / "export_log.jsonl"
    with pytest.raises(privacy.ExportRefused):
        privacy.export_shared(d, tmp_path / "out", "C1", log)            # not reviewed
    monkeypatch.setattr(privacy, "SHARING_POLICY_PATH", tmp_path / "policy.json")
    privacy.review(d, "C1", True, ["segment_000.mp4"], identifiable_found=False)
    with pytest.raises(privacy.ExportRefused):
        privacy.export_shared(d, tmp_path / "out", "A1", log)            # not a coordinator
    (d / "shared" / "segment_001.mp4").write_bytes(b"NEW")
    with pytest.raises(privacy.ExportRefused):
        privacy.export_shared(d, tmp_path / "out", "C1", log)            # changed after review
    (d / "shared" / "segment_001.mp4").unlink()
    rec = privacy.export_shared(d, tmp_path / "out", "C1", log)
    exported = sorted(p.name for p in (tmp_path / "out" / "s1").iterdir())
    assert "segment_000.mp4" in exported and (tmp_path / "out" / "s1" / "segment_000.mp4").read_bytes() == b"BLURRED"
    assert rec["by"] == "C1" and "s1" in log.read_text() and not any("RAW" in p.read_text(errors="ignore") for p in (tmp_path / "out" / "s1").glob("*.mp4"))


def test_backup_verify_restore(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("cb", resolve("scripts/campaign_backup.py"))
    cb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cb)
    src, camp, dest = tmp_path / "sessions", tmp_path / "campaign", tmp_path / "ext"
    _session(src, "b1", real("T1", "D01"), steps=50, video=False)
    camp.mkdir()
    (camp / "manifest.json").write_text("{}")
    r = cb.backup(dest, None, True, src, camp)
    assert r["copied"] > 3 and r["unchanged"] == 0
    assert cb.backup(dest, None, True, src, camp)["copied"] == 0          # incremental
    assert cb.verify(dest)["ok"]
    (dest / "SafeDrive_backup" / "sessions" / "b1" / "timeline.csv").write_text("tampered")
    v = cb.verify(dest)
    assert not v["ok"] and v["hash_mismatch"] == ["sessions/b1/timeline.csv"]
    cb.backup(dest, None, True, src, camp)                                 # repaired from the source
    out = cb.restore(dest, "b1", tmp_path / "restored")
    assert out["restored_files"] == 3 and (tmp_path / "restored" / "b1" / "trip_meta.json").exists()
    with pytest.raises(cb.BackupError):
        cb.backup(tmp_path / "same", None, False, src, camp)               # same drive refused without the flag
