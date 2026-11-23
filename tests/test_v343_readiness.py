"""v3.4.3 pre-campaign readiness: backup destination rules, backup / restore failures, privacy review workflow, access refusals,
provenance mismatch, operator checklist gating and the GO rule. System probes (drives, BitLocker, ACL) are injected."""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta

import pytest

from src.realworld import campaign, checklist, privacy, provenance, readiness, training
from src.realworld.trip_meta import TripMeta, save_meta
from src.utils.config import resolve

V3 = "models/safedrive_v3"


def _cb():
    spec = importlib.util.spec_from_file_location("cb343", resolve("scripts/campaign_backup.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _bl(state):
    return lambda letter: {"method": "probe", "protection": state(letter) if callable(state) else state}


@pytest.fixture()
def acc(tmp_path, monkeypatch):
    p = tmp_path / "access.json"
    p.write_text(json.dumps({"raw_access": ["ANN-A", "ANN-B"], "coordinators": ["COORD-01"]}))
    monkeypatch.setattr(privacy, "ACCESS_PATH", p)
    monkeypatch.setattr(privacy, "SHARING_POLICY_PATH", tmp_path / "sharing_policy.json")
    monkeypatch.setattr(privacy, "EXPORT_LOG", tmp_path / "export_log.jsonl")
    monkeypatch.setattr(privacy, "REVOCATION_LOG", tmp_path / "revocation_log.jsonl")
    return p


def _real(sid="r1"):
    return TripMeta(sid, data_kind="real_driving", driver_consent=True, occupants_informed=True, supervisor_present=True,
                    configured_before_driving=True, regulations_checked=True, campaign_id="c1", trip_id="T1", driver_id="D01",
                    record_full_video=True)


def _shared(tmp_path, sid="s1", kind="live_camera_test", n=2):
    root = tmp_path / "sessions"
    save_meta(TripMeta(sid, data_kind=kind, driver_consent=True, occupants_informed=True, supervisor_present=True,
                       configured_before_driving=True), root)
    d = root / sid
    (d / "shared").mkdir(parents=True)
    (d / "video").mkdir()
    (d / "video" / "segment_000.mp4").write_bytes(b"RAW")
    for k in range(n):
        (d / "shared" / f"segment_{k:03d}.mp4").write_bytes(b"BLURRED%d" % k)
    return root, d


def _all_confirmed(items_):
    return {k: {"status": "confirmed", "basis": "checked on site, note N-1"} for k in items_}


def _acl_evidence(tmp_path, data_dir, acl_probe, app="PC\\app"):
    """Synthetic (tmp) ACL evidence: account map + snapshot of the current fingerprint + a denied standard-account probe and an
    allowed app-account probe recorded after the snapshot."""
    amap = tmp_path / "accounts.json"
    amap.write_text(json.dumps({"app_account": app, "accounts": {}}))
    snaps = tmp_path / "acl_snaps.jsonl"
    t0 = datetime.now() - timedelta(minutes=5)
    fp = readiness.acl_fingerprint(readiness.measure_acls(data_dir, acl_probe))
    snaps.write_text(json.dumps({"at": t0.isoformat(timespec="seconds"), "fingerprint": fp}) + "\n")
    pd = tmp_path / "probes"
    pd.mkdir()
    t1 = (t0 + timedelta(minutes=1)).isoformat(timespec="seconds")
    (pd / "probe_std.json").write_text(json.dumps({"at": t1, "user": "PC\\tester", "elevated": False, "admin_member": False,
                                                   "can_list_sessions": False, "can_read_raw": False, "can_write": False,
                                                   "outcome": "denied"}))
    (pd / "probe_app.json").write_text(json.dumps({"at": t1, "user": app, "elevated": False, "admin_member": True,
                                                   "can_list_sessions": True, "can_read_raw": True, "can_write": True,
                                                   "outcome": "allowed"}))
    return {"account_map_path": amap, "acl_snapshot_path": snaps, "acl_probe_dir": pd}


# ------------------------------------------------------------------ backup
def test_backup_refused_without_external_drive(tmp_path):
    cb = _cb()
    src, camp = tmp_path / "sessions", tmp_path / "campaign"
    save_meta(_real(), src)
    camp.mkdir()
    drives = [{"letter": "C", "disk": 0, "bus": "NVMe", "external": False, "free": 10**12, "serial": "A"},
              {"letter": "D", "disk": 0, "bus": "NVMe", "external": False, "free": 10**12, "serial": "B"}]
    chk = cb.check_destination(resolve("."), src, 10, drives, bitlocker_probe=_bl("off"))
    assert not chk["ok"] and any("inside the project" in p for p in chk["problems"])
    with pytest.raises(cb.BackupError) as e:
        cb.backup(tmp_path / "dest", None, False, src, camp, drives=drives, bitlocker_probe=_bl("on"))
    assert "same physical disk" in str(e.value) and e.value.recovery
    assert not (tmp_path / "dest" / "SafeDrive_backup").exists()                     # nothing copied, no fake backup
    rec = readiness.backup_records(camp / "backup_status.jsonl")[-1]
    assert rec["ok"] is False and rec["refused"]
    items = {i["key"]: i for i in readiness.check_all(src, drives=drives, bitlocker_probe=_bl("unknown"),
                                                      acl_probe=lambda p: {"ok": True, "entries": [], "broad_principals": []},
                                                      backup_path=camp / "backup_status.jsonl", training_report={},
                                                      attest_path=tmp_path / "att.jsonl", camera_path=tmp_path / "cam.jsonl")}
    assert items["8_external_backup_destination"]["status"] == "BLOCKED"
    assert items["9_backup_verification"]["status"] == "BLOCKED" and items["10_restore_verification"]["status"] == "BLOCKED"


def test_external_drive_not_enough_space_or_unencrypted_is_refused(tmp_path):
    cb = _cb()
    letter = str(tmp_path.drive).rstrip(":").upper()
    ext = [{"letter": letter, "disk": 1, "bus": "USB", "external": True, "free": 1000, "serial": "X"},
           {"letter": "Q", "disk": 0, "bus": "NVMe", "external": False, "free": 10**12, "serial": "Y"}]
    data_root = tmp_path / "sessions"
    data_root.mkdir()
    # pretend the data lives on drive Q (disk 0) and the destination is the USB disk 1
    small = cb.check_destination(tmp_path / "dest", data_root, 10_000, ext, allow_same_drive=True, bitlocker_probe=_bl("on"))
    assert any("not enough free space" in p for p in small["problems"])
    unenc = cb.check_destination(tmp_path / "dest", data_root, 10, [dict(ext[0], free=10**12), ext[1]], allow_same_drive=True,
                                 bitlocker_probe=_bl("off"))
    assert any("encryption" in w for w in unenc["warnings"])  # test mode downgrades to a warning; a real run refuses
    real_run = cb.check_destination(tmp_path / "dest", tmp_path / "Q_data", 10, [dict(ext[0], free=10**12)], bitlocker_probe=_bl("off"))
    assert any("encryption" in p for p in real_run["problems"]) or any("same physical disk" in p for p in real_run["problems"])
    # readiness: external drive present but too small -> FAIL
    items = {i["key"]: i for i in readiness.check_all(data_root, drives=[dict(ext[0], letter="E"), ext[1]], bitlocker_probe=_bl("on"),
                                                      acl_probe=lambda p: {"ok": True, "entries": [], "broad_principals": []},
                                                      backup_path=tmp_path / "b.jsonl", training_report={}, data_dir=tmp_path,
                                                      attest_path=tmp_path / "att.jsonl", camera_path=tmp_path / "cam.jsonl")}
    assert items["8_external_backup_destination"]["status"] == "FAIL"


def test_backup_hash_mismatch_and_failed_restore_are_reported(tmp_path):
    cb = _cb()
    src, camp, dest = tmp_path / "sessions", tmp_path / "campaign", tmp_path / "ext"
    save_meta(_real(), src)
    (src / "r1" / "timeline.csv").write_text("step\n0\n")
    camp.mkdir()
    cb.backup(dest, None, True, src, camp, drives=[])
    assert cb.verify(dest, camp)["ok"]
    (dest / "SafeDrive_backup" / "sessions" / "r1" / "timeline.csv").write_text("tampered")
    v = cb.verify(dest, camp)
    assert not v["ok"] and v["hash_mismatch"] == ["sessions/r1/timeline.csv"] and "recovery" in v
    rt = cb.restore_test(dest, src, camp, tmp_parent=tmp_path)
    assert rt["ok"] is False and rt["hash_mismatch"] and rt["temp_folder_removed"]
    (dest / "SafeDrive_backup" / "sessions" / "r1" / "timeline.csv").unlink()
    rt2 = cb.restore_test(dest, src, camp, tmp_parent=tmp_path)
    assert rt2["ok"] is False and rt2["missing"] == ["sessions/r1/timeline.csv"]
    with pytest.raises(cb.BackupError):
        cb.restore(dest, "r1", tmp_path / "restored")
    # readiness reads the records: a test-mode (same-drive) backup never counts, a failed verify / restore is FAIL
    recs = camp / "backup_status.jsonl"
    items = {i["key"]: i for i in readiness.check_all(src, drives=[], bitlocker_probe=_bl("on"), backup_path=recs,
                                                      acl_probe=lambda p: {"ok": True, "entries": [], "broad_principals": []},
                                                      training_report={}, attest_path=tmp_path / "a", camera_path=tmp_path / "c")}
    assert items["9_backup_verification"]["status"] == "BLOCKED"   # test-mode backup is not a valid backup
    assert items["10_restore_verification"]["status"] == "FAIL"
    good = {"action": "backup", "at": "2026-11-22T10:00:00", "ok": True, "external": True, "same_disk": False, "test_mode": False,
            "sessions": ["r1"], "dest_serial": "S"}
    bad_verify = {"action": "verify", "at": "2026-11-22T10:05:00", "ok": False, "dest_serial": "S", "hash_mismatch": ["x"]}
    p = tmp_path / "recs.jsonl"
    p.write_text(json.dumps(good) + "\n" + json.dumps(bad_verify) + "\n")
    items = {i["key"]: i for i in readiness.check_all(src, drives=[], bitlocker_probe=_bl("on"), backup_path=p,
                                                      acl_probe=lambda p_: {"ok": True, "entries": [], "broad_principals": []},
                                                      training_report={}, attest_path=tmp_path / "a", camera_path=tmp_path / "c")}
    assert items["9_backup_verification"]["status"] == "FAIL"


def test_interrupted_backup_keeps_only_verified_files(tmp_path, monkeypatch):
    cb = _cb()
    src, camp, dest = tmp_path / "sessions", tmp_path / "campaign", tmp_path / "ext"
    save_meta(_real(), src)
    (src / "r1" / "a.csv").write_text("1")
    (src / "r1" / "b.csv").write_text("2")
    camp.mkdir()
    real_copy, calls = cb.shutil.copy2, {"n": 0}

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError(28, "No space left on device")
        return real_copy(a, b)
    monkeypatch.setattr(cb.shutil, "copy2", flaky)
    with pytest.raises(cb.BackupError) as e:
        cb.backup(dest, None, True, src, camp, drives=[])
    assert "source data is unchanged" in e.value.recovery
    idx = json.loads((dest / "SafeDrive_backup" / "backup_index.json").read_text())
    assert len(idx["files"]) == 1                                  # only the verified file is indexed
    monkeypatch.setattr(cb.shutil, "copy2", real_copy)
    cb.backup(dest, None, True, src, camp, drives=[])               # resume
    assert cb.verify(dest)["ok"]


# ------------------------------------------------------------------ access / privacy
def test_unauthorised_people_are_refused(tmp_path, acc):
    root, d = _shared(tmp_path)
    assert not privacy.can_view_raw("Stranger") and privacy.can_view_raw("ANN-A")
    with pytest.raises(PermissionError):
        privacy.review(d, "ANN-A", True, ["segment_000.mp4", "segment_001.mp4"], identifiable_found=False)  # annotator, not coordinator
    with pytest.raises(PermissionError):
        readiness.attest("legal_review", "ANN-B", "checked", "doc-1", path=tmp_path / "att.jsonl")
    with pytest.raises(PermissionError):
        privacy.set_sharing_policy(False, "ANN-A", "x")
    privacy.review(d, "COORD-01", True, ["segment_000.mp4", "segment_001.mp4"], identifiable_found=False)
    with pytest.raises(privacy.ExportRefused):
        privacy.export_shared(d, tmp_path / "out", "ANN-A")
    with pytest.raises(PermissionError):
        privacy.revoke_shared(d, "ANN-A", "x")


def test_unreviewed_copy_is_pending_and_never_exported(tmp_path, acc):
    root, d = _shared(tmp_path)
    assert privacy.shared_status(d) == "pending_review" and not privacy.shareable(d)
    with pytest.raises(privacy.ExportRefused, match="pending_review"):
        privacy.export_shared(d, tmp_path / "out", "COORD-01")
    with pytest.raises(ValueError):  # approval without the explicit 'nothing readable' finding
        privacy.review(d, "COORD-01", True, ["segment_000.mp4", "segment_001.mp4"])
    assert not (tmp_path / "out").exists()


def test_readable_face_or_plate_rejects_copy_and_revocation_blocks_export(tmp_path, acc):
    root, d = _shared(tmp_path)
    r = privacy.review(d, "COORD-01", True, ["segment_000.mp4", "segment_001.mp4"], "plate readable at 12 s", identifiable_found=True)
    assert r["decision"] == "rejected" and privacy.shared_status(d) == "rejected"
    with pytest.raises(privacy.ExportRefused, match="REJECTED"):
        privacy.export_shared(d, tmp_path / "out", "COORD-01")
    # approved, exported, then found insufficient -> revoke (deletes reachable exports, raw untouched)
    privacy.review(d, "COORD-01", True, ["segment_000.mp4", "segment_001.mp4"], identifiable_found=False)
    rec = privacy.export_shared(d, tmp_path / "out", "COORD-01")
    assert (tmp_path / "out" / "s1" / "segment_000.mp4").exists()
    rv = privacy.revoke_shared(d, "COORD-01", "face visible in segment 1", delete_exports=True)
    assert rv["deleted_exports"] == [rec["destination"]] and not (tmp_path / "out" / "s1").exists()
    assert privacy.shared_status(d) == "revoked" and (d / "video" / "segment_000.mp4").read_bytes() == b"RAW"
    with pytest.raises(privacy.ExportRefused, match="revoked"):
        privacy.export_shared(d, tmp_path / "out2", "COORD-01")
    # a changed copy after approval is stale
    root2, d2 = _shared(tmp_path / "b", "s2")
    privacy.review(d2, "COORD-01", True, ["segment_000.mp4", "segment_001.mp4"], identifiable_found=False)
    (d2 / "shared" / "segment_001.mp4").write_bytes(b"CHANGED")
    assert privacy.shared_status(d2) == "stale"
    with pytest.raises(privacy.ExportRefused, match="changed after the review"):
        privacy.export_shared(d2, tmp_path / "out3", "COORD-01")


def test_no_sharing_decision_blocks_every_export_and_demo_never_validates_blur(tmp_path, acc):
    root, d = _shared(tmp_path, "demo1", kind="demo_video")
    privacy.review(d, "COORD-01", True, ["segment_000.mp4", "segment_001.mp4"], identifiable_found=False)
    kw = dict(drives=[], bitlocker_probe=_bl("on"), acl_probe=lambda p: {"ok": True, "entries": [], "broad_principals": []},
              training_report={}, attest_path=tmp_path / "a", camera_path=tmp_path / "c", backup_path=tmp_path / "b",
              access_path=acc, sharing_policy_path=tmp_path / "sharing_policy.json")
    it = {i["key"]: i for i in readiness.check_all(root, **kw)}
    assert it["11_privacy_review"]["status"] == "NOT VERIFIED"   # an approved DEMO copy does not validate the blur
    privacy.set_sharing_policy(False, "COORD-01", "no sharing until the blur is reviewed on campaign-camera footage")
    with pytest.raises(privacy.ExportRefused, match="disabled"):
        privacy.export_shared(d, tmp_path / "out", "COORD-01")
    it = {i["key"]: i for i in readiness.check_all(root, **kw)}
    assert it["11_privacy_review"]["status"] == "PASS" and "sharing disabled" in it["11_privacy_review"]["evidence"]["mode"]


# ------------------------------------------------------------------ provenance / checklist / GO
def test_provenance_mismatch_fails_hash_items_and_blocks_session(tmp_path, acc, monkeypatch):
    ref = {m.split("/")[-1]: provenance.compute(m) for m in readiness.MODELS}
    bad = json.loads(json.dumps(ref))
    bad["safedrive_v3"]["threshold_config_sha256"] = "0" * 64
    it = {i["key"]: i for i in readiness.check_all(tmp_path / "s", frozen_ref=bad, drives=[], bitlocker_probe=_bl("on"),
                                                   acl_probe=lambda p: {"ok": True, "entries": [], "broad_principals": []},
                                                   training_report={}, attest_path=tmp_path / "a", camera_path=tmp_path / "c",
                                                   backup_path=tmp_path / "b")}
    assert it["3_threshold_hash"]["status"] == "FAIL" and it["1_model_hash"]["status"] == "PASS"
    cl = tmp_path / "cl"
    rec = checklist.save("OP1", "SUP1", {k: True for k in checklist.MANUAL_ITEMS},
                         {"model_hash": {"ok": True, "detail": ""}}, cl)
    assert rec["complete"]
    monkeypatch.setattr(provenance, "frozen", lambda path=None: bad)
    with pytest.raises(checklist.ChecklistError, match="frozen reference"):
        checklist.require_ready(V3, cl)


def test_incomplete_or_old_checklist_refuses_real_driving_session(tmp_path, acc):
    cl = tmp_path / "cl"
    with pytest.raises(checklist.ChecklistError, match="no pre-session checklist"):
        checklist.require_ready(V3, cl)
    with pytest.raises(ValueError):
        checklist.save("D01", "D01", {}, {}, cl)                    # supervisor must be another person
    rec = checklist.save("OP1", "SUP1", {k: True for k in checklist.MANUAL_ITEMS if k != "camera_fixed"},
                         {"disk_space": {"ok": True, "detail": ""}, "precampaign_go": {"ok": False, "detail": "NO-GO"}}, cl)
    assert not rec["complete"] and rec["missing_manual"] == ["camera_fixed"] and rec["failed_auto"] == ["precampaign_go"]
    with pytest.raises(checklist.ChecklistError, match="incomplete"):
        checklist.require_ready(V3, cl)
    checklist.save("OP1", "SUP1", {k: True for k in checklist.MANUAL_ITEMS}, {"x": {"ok": True, "detail": ""}}, cl)
    with pytest.raises(checklist.ChecklistError, match="older"):
        checklist.require_ready(V3, cl, now=datetime.now() + timedelta(hours=3))
    # the controller refuses a real-driving session and records nothing
    from src.live.controller import LiveController
    ctl = LiveController()
    ctl.checklist_dir = tmp_path / "empty"
    with pytest.raises(checklist.ChecklistError):
        ctl.start_session(_real(), sessions_root=tmp_path / "sessions")
    assert ctl.session is None and not (tmp_path / "sessions").exists()
    # automatic items: with no readiness GO the checklist can never be complete
    auto = checklist.auto_checks(V3, tmp_path / "sessions", acc, tmp_path / "b", tmp_path / "c", tmp_path / "rd.json", tmp_path)
    assert auto["precampaign_go"]["ok"] is False and auto["camera_check"]["ok"] is False and auto["model_hash"]["ok"] is True


def test_go_only_when_everything_is_pass(tmp_path, acc):
    mk = lambda st: readiness.item("k", "t", st, {})  # noqa: E731
    assert readiness.go_decision([mk("PASS")] * 15)["verdict"] == "GO"
    for bad in ("BLOCKED", "NOT VERIFIED", "FAIL"):
        dec = readiness.go_decision([mk("PASS")] * 14 + [mk(bad)])
        assert dec["verdict"] == "NO-GO" and dec["blocking"][0]["status"] == bad
    assert readiness.go_decision([])["verdict"] == "NO-GO"
    # a configuration value 'true' is not evidence: a camera check alone (no coordinator attestation) stays NOT VERIFIED;
    # an attestation by someone who is no longer a coordinator does not count
    cam = tmp_path / "cam.jsonl"
    cam.write_text(json.dumps({"at": datetime.now().isoformat(), "ok": True}) + "\n")
    att = tmp_path / "att.jsonl"
    base = dict(drives=[], bitlocker_probe=_bl("on"), acl_probe=lambda p: {"ok": True, "entries": [], "broad_principals": []},
                training_report={"passed": True, "complete": True}, attest_path=att, camera_path=cam, backup_path=tmp_path / "b",
                access_path=acc)
    it = {i["key"]: i for i in readiness.check_all(tmp_path / "s", **base)}
    assert it["12_camera_readiness"]["status"] == "NOT VERIFIED" and it["13_legal_review"]["status"] == "NOT VERIFIED"
    readiness.attest("legal_review", "COORD-01", "rules checked", "memo-2026-11", acc, att)
    # v3.4.4: a single legal tick is no longer enough; the 8-point legal checklist is required
    assert {i["key"]: i for i in readiness.check_all(tmp_path / "s", **base)}["13_legal_review"]["status"] == "NOT VERIFIED"
    readiness.attest_checklist("legal_checklist", "COORD-01", _all_confirmed(readiness.LEGAL_ITEMS), "memo-2026-11", acc, att)
    assert {i["key"]: i for i in readiness.check_all(tmp_path / "s", **base)}["13_legal_review"]["status"] == "PASS"
    acc.write_text(json.dumps({"raw_access": ["ANN-A"], "coordinators": ["COORD-02"]}))
    assert {i["key"]: i for i in readiness.check_all(tmp_path / "s", **base)}["13_legal_review"]["status"] == "NOT VERIFIED"


def test_full_evidence_gives_go_and_acl_with_broad_groups_fails(tmp_path, acc):
    """GO is reachable, but only with every piece of evidence (all synthetic here, in a temporary folder)."""
    ref = {m.split("/")[-1]: provenance.compute(m) for m in readiness.MODELS}
    root = tmp_path / "sessions"
    root.mkdir()
    att = tmp_path / "att.jsonl"
    for k in ("legal_review", "camera_mount"):
        readiness.attest(k, "COORD-01", "done", "ref", acc, att)
    for k, items_ in (("legal_checklist", readiness.LEGAL_ITEMS), ("camera_safety", readiness.CAMERA_SAFETY_ITEMS)):
        readiness.attest_checklist(k, "COORD-01", _all_confirmed(items_), "ref", acc, att)
    privacy.set_sharing_policy(False, "COORD-01", "no sharing yet", tmp_path / "pol.json", acc)
    cam = tmp_path / "cam.jsonl"
    cam.write_text(json.dumps({"at": datetime.now().isoformat(), "ok": True}) + "\n")
    bk = tmp_path / "bk.jsonl"
    t0 = datetime.now()
    rows = [{"action": "backup", "at": t0.isoformat(timespec="seconds"), "ok": True, "external": True, "same_disk": False,
             "test_mode": False, "sessions": [], "dest_serial": "S"},
            {"action": "verify", "at": (t0 + timedelta(seconds=5)).isoformat(timespec="seconds"), "ok": True, "dest_serial": "S"},
            {"action": "restore_test", "at": (t0 + timedelta(seconds=9)).isoformat(timespec="seconds"), "ok": True}]
    bk.write_text("".join(json.dumps(r) + "\n" for r in rows))
    data_dir = resolve("data")
    letter = str(data_dir.drive).rstrip(":").upper()
    drives = [{"letter": letter, "disk": 0, "bus": "NVMe", "external": False, "free": 10**12},
              {"letter": "Z", "disk": 1, "bus": "USB", "external": True, "free": 10**12}]
    good_acl = lambda p: {"ok": True, "entries": [{"principal": "PC\\app", "rights": "(OI)(CI)(M)"},  # noqa: E731
                                                  {"principal": "NT AUTHORITY\\SYSTEM", "rights": "(OI)(CI)(F)"}], "broad_principals": []}
    acl_ev = _acl_evidence(tmp_path, data_dir, good_acl)
    kw = dict(sessions_root=root, access_path=acc, attest_path=att, backup_path=bk, camera_path=cam, drives=drives,
              bitlocker_probe=_bl("on"), acl_probe=good_acl, data_dir=data_dir,
              frozen_ref=ref, sharing_policy_path=tmp_path / "pol.json", training_report={"passed": True, "complete": True}, **acl_ev)
    items = readiness.check_all(**kw)
    dec = readiness.go_decision(items)
    if any(i["key"] == "15_disk_space" and i["status"] != "PASS" for i in items):
        pytest.skip("not enough free disk on the test machine for the 10 h estimate")
    assert dec["verdict"] == "GO", dec["blocking"]
    kw["acl_probe"] = lambda p: {"ok": True, "entries": [], "broad_principals": ["NT AUTHORITY\\Authenticated Users"]}
    assert readiness.go_decision(readiness.check_all(**kw))["verdict"] == "NO-GO"


def test_training_pack_report_and_isolation(tmp_path):
    from src.realworld import annotation as ann
    d = tmp_path / "PACK"
    (d / "video").mkdir(parents=True)
    save_meta(TripMeta("PACK", data_kind="demo_video", notes=training.MARK), tmp_path)
    key = {"pack": "PACK", "clips": [
        {"clip": 0, "kind": "event", "t_start": 0, "t_end": 10, "accepted": ["collision"], "reference_event_time": 8.0},
        {"clip": 1, "kind": "normal", "t_start": 11, "t_end": 21, "accepted": ["normal"], "reference_event_time": None},
        {"clip": 2, "kind": "uncertain", "t_start": 22, "t_end": 32, "accepted": ["uncertain"], "reference_event_time": 30.0}]}
    (d / "training_key.json").write_text(json.dumps(key))
    for n, dt in (("ANN-A", 0.3), ("ANN-B", -0.4)):
        ann.add(d, ann.Annotation("PACK", n, "collision", 6, 9, 8 + dt, evidence="car hits barrier", evidence_frames=[8.0]))
        ann.add(d, ann.Annotation("PACK", n, "near_miss", 2, 4, 3, evidence="close pass", evidence_frames=[3.0]))
        ann.add(d, ann.Annotation("PACK", n, "risk_event", 4.2, 5.5, 5, evidence="lead brakes", evidence_frames=[5.0]))
        ann.add(d, ann.Annotation("PACK", n, "normal", 11, 21))
        ann.add(d, ann.Annotation("PACK", n, "uncertain", 25, 31))
    rep = training.report(d, ["ANN-A", "ANN-B"])
    assert not rep["complete"] and not rep["passed"]                 # nobody marked the full review yet
    from src.realworld.trip_meta import load_meta
    m = load_meta("PACK", tmp_path)
    m.fully_reviewed_by = ["ANN-A", "ANN-B"]
    save_meta(m, tmp_path)
    rep = training.report(d, ["ANN-A", "ANN-B"])
    assert rep["complete"] and rep["per_annotator"]["ANN-A"]["timing_median_abs_s"] == 0.3
    assert rep["per_annotator"]["ANN-B"]["criteria"]["categories_practised"] and rep["passed"], rep["criteria_results"]
    # the pack never appears in the campaign manifest / real-driving hours
    assert training.TRAINING_SESSIONS != resolve("data/sessions") and "data/sessions" not in str(training.TRAINING_SESSIONS).replace("\\", "/")


def test_dashboard_status_counts_only_real_driving(tmp_path, acc):
    man = {"sessions": [{"session_id": "a", "data_kind": "demo_video", "hours": 0.5, "eligible": False, "model_sha256": "x",
                         "partition": "excluded", "double_annotated": False},
                        {"session_id": "b", "data_kind": "legacy_unknown", "hours": 0.2, "eligible": False, "model_sha256": "",
                         "partition": "excluded", "double_annotated": False}], "leakage": {"ok": True}}
    st = campaign.campaign_status(man, tmp_path / "s", tmp_path)
    assert st["real_driving_hours"] == 0.0 and st["eligible_sessions"] == 0 and st["excluded_sessions"] == 2
    assert st["missing_provenance"] == 0 and st["final_test"]["label"] == "locked" and st["backup"]["label"] == "none"
