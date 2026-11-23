"""v3.4.4 NO-GO closure: folder permissions (least privilege + real-account probes), encryption evidence, backup failure modes,
privacy review (rejected / stale / parked / log), legal and camera-safety checklists, annotator training submissions, checklist
expiry, provenance mismatch and the GO rule. Every probe and record is synthetic and lives in a temporary folder."""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta

import pytest

from src.realworld import checklist, privacy, provenance, readiness, training
from src.realworld import annotation as ann
from src.realworld.trip_meta import TripMeta, load_meta, save_meta
from src.utils.config import resolve


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, resolve(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _cb():
    return _load("cb344", "scripts/campaign_backup.py")


def _bl(state):
    return lambda letter: {"method": "probe", "protection": state}


@pytest.fixture()
def acc(tmp_path, monkeypatch):
    p = tmp_path / "access.json"
    p.write_text(json.dumps({"raw_access": ["ANN-A", "ANN-B"], "coordinators": ["COORD-01"]}))
    monkeypatch.setattr(privacy, "ACCESS_PATH", p)
    monkeypatch.setattr(privacy, "SHARING_POLICY_PATH", tmp_path / "sharing_policy.json")
    monkeypatch.setattr(privacy, "EXPORT_LOG", tmp_path / "export_log.jsonl")
    monkeypatch.setattr(privacy, "REVOCATION_LOG", tmp_path / "revocation_log.jsonl")
    return p


def _kw(tmp_path, acc, **over):
    """check_all arguments with every evidence store in tmp (nothing from the real data/campaign)."""
    kw = dict(sessions_root=tmp_path / "s", access_path=acc, attest_path=tmp_path / "att.jsonl", backup_path=tmp_path / "bk.jsonl",
              camera_path=tmp_path / "cam.jsonl", training_report={}, drives=[], bitlocker_probe=_bl("on"),
              acl_probe=lambda p: {"ok": True, "entries": [], "broad_principals": []}, data_dir=tmp_path,
              sharing_policy_path=tmp_path / "pol.json", account_map_path=tmp_path / "amap.json",
              acl_snapshot_path=tmp_path / "snaps.jsonl", acl_probe_dir=tmp_path / "probes", encryption_path=tmp_path / "enc.jsonl")
    kw.update(over)
    return kw


def _items(tmp_path, acc, **over):
    return {i["key"]: i for i in readiness.check_all(**_kw(tmp_path, acc, **over))}


def _confirmed(items_):
    return {k: {"status": "confirmed", "basis": "checked on site, note N-1"} for k in items_}


# ================================================================== folder permissions
BAD_ACL = {"ok": True, "broad_principals": ["NT AUTHORITY\\Authenticated Users"], "entries": [
    {"principal": "BUILTIN\\Administrators", "rights": "(I)(OI)(CI)(F)"},
    {"principal": "NT AUTHORITY\\SYSTEM", "rights": "(I)(OI)(CI)(F)"},
    {"principal": "NT AUTHORITY\\Authenticated Users", "rights": "(I)(OI)(CI)(M)"},
    {"principal": "BUILTIN\\Users", "rights": "(I)(OI)(CI)(RX)"}]}
GOOD_ACL = {"ok": True, "broad_principals": [], "entries": [
    {"principal": "NT AUTHORITY\\SYSTEM", "rights": "(OI)(CI)(F)"},
    {"principal": "BUILTIN\\Administrators", "rights": "(OI)(CI)(F)"},
    {"principal": "PC\\app", "rights": "(OI)(CI)(M)"},
    {"principal": "PC\\ann_a", "rights": "(OI)(CI)(RX,W)"}]}


def test_rights_are_split_into_read_write_delete_and_permissions():
    assert readiness.rights_of("(I)(OI)(CI)(M)") == {"rights": {"read", "write", "delete"}, "deny": False, "inherited": True,
                                                     "inherit_only": False}
    assert readiness.rights_of("(OI)(CI)(IO)(GR,GE)")["rights"] == {"read"}
    assert readiness.rights_of("(F)")["rights"] == {"read", "write", "delete", "change_permissions"}
    assert readiness.rights_of("(DENY)(W)")["deny"]


def test_unsafe_acl_fails_with_inherited_cause(tmp_path, acc):
    it = _items(tmp_path, acc, acl_probe=lambda p: BAD_ACL)["7_folder_permissions"]
    assert it["status"] == "FAIL"
    assert any("Authenticated Users" in p and "inherited" in p for p in it["evidence"]["problems"])
    assert any("BUILTIN\\Users" in p for p in it["evidence"]["problems"])


def test_least_privilege_extra_rights_unknown_accounts_and_app_lockout(tmp_path, acc):
    amap = {"app_account": "PC\\app", "accounts": {"ANN-A": "PC\\ann_a", "COORD-01": "PC\\coord"}}
    over = {"acls": {"d": {"entries": GOOD_ACL["entries"] + [{"principal": "PC\\ann_a", "rights": "(OI)(CI)(F)"},
                                                             {"principal": "PC\\visitor", "rights": "(OI)(CI)(RX)"}]}}}
    ev = readiness.evaluate_acl(over["acls"], amap, ["ANN-A", "ANN-B"], ["COORD-01"])
    assert any("ann_a" in p and "more than needed" in p for p in ev["problems"])      # annotator with Full control
    assert any("visitor" in p and "not an authorised" in p for p in ev["problems"])    # unknown account
    locked = readiness.evaluate_acl({"d": {"entries": [{"principal": "NT AUTHORITY\\SYSTEM", "rights": "(F)"}]}}, amap, [], [])
    assert any("would lose read / write" in p for p in locked["problems"])              # the plan must not lock out the app
    assert not readiness.evaluate_acl({"d": GOOD_ACL}, amap, ["ANN-A", "ANN-B"], ["COORD-01"])["problems"]


def _acl_store(tmp_path, data_dir, acl, probes):
    amap = tmp_path / "amap.json"
    amap.write_text(json.dumps({"app_account": "PC\\app", "accounts": {"ANN-A": "PC\\ann_a"}}))
    t0 = datetime.now() - timedelta(minutes=10)
    fp = readiness.acl_fingerprint(readiness.measure_acls(data_dir, lambda p: acl))
    (tmp_path / "snaps.jsonl").write_text(json.dumps({"at": t0.isoformat(timespec="seconds"), "fingerprint": fp}) + "\n")
    pd = tmp_path / "probes"
    pd.mkdir(exist_ok=True)
    for i, pr in enumerate(probes):
        (pd / f"probe_{i}.json").write_text(json.dumps({"at": (t0 + timedelta(minutes=1)).isoformat(timespec="seconds"), **pr}))


DENIED = {"user": "PC\\tester", "elevated": False, "admin_member": False, "can_list_sessions": False, "can_read_raw": False,
          "can_write": False, "outcome": "denied"}
APP_OK = {"user": "PC\\app", "can_list_sessions": True, "can_read_raw": True, "can_write": True, "outcome": "allowed"}
ANN_OK = {"user": "PC\\ann_a", "can_list_sessions": True, "can_read_raw": True, "can_write": True, "outcome": "allowed"}


def test_unauthorised_account_reaching_raw_video_fails_and_probes_are_required(tmp_path, acc):
    probe = lambda p: GOOD_ACL  # noqa: E731
    # narrow ACL but no account map / snapshot / probes -> NOT VERIFIED (a clean ACL listing alone is not proof)
    assert _items(tmp_path, acc, acl_probe=probe)["7_folder_permissions"]["status"] == "NOT VERIFIED"
    _acl_store(tmp_path, tmp_path, GOOD_ACL, [APP_OK, ANN_OK])
    it = _items(tmp_path, acc, acl_probe=probe)["7_folder_permissions"]
    assert it["status"] == "NOT VERIFIED" and "STANDARD account" in it["action"]           # no denied standard account yet
    # an admin-member or elevated "denied" probe does not count as the standard-account test
    _acl_store(tmp_path, tmp_path, GOOD_ACL, [APP_OK, ANN_OK, dict(DENIED, admin_member=True)])
    assert _items(tmp_path, acc, acl_probe=probe)["7_folder_permissions"]["status"] == "NOT VERIFIED"
    # a non-authorised account that could read the raw video -> FAIL
    _acl_store(tmp_path, tmp_path, GOOD_ACL, [APP_OK, ANN_OK, dict(DENIED, can_read_raw=True, outcome="allowed")])
    assert _items(tmp_path, acc, acl_probe=probe)["7_folder_permissions"]["status"] == "FAIL"
    # everything measured -> PASS
    for f in (tmp_path / "probes").glob("*.json"):
        f.unlink()
    _acl_store(tmp_path, tmp_path, GOOD_ACL, [APP_OK, ANN_OK, DENIED])
    it = _items(tmp_path, acc, acl_probe=probe)["7_folder_permissions"]
    assert it["status"] == "PASS" and it["evidence_at"]
    # the ACL changes afterwards -> the old snapshot / probes no longer apply
    changed = dict(GOOD_ACL, entries=GOOD_ACL["entries"] + [{"principal": "PC\\coord", "rights": "(OI)(CI)(RX)"}])
    assert _items(tmp_path, acc, acl_probe=lambda p: changed)["7_folder_permissions"]["status"] in ("FAIL", "NOT VERIFIED")


def test_probe_records_what_the_current_account_can_do(tmp_path):
    dacl = _load("dacl344", "scripts/data_acl.py")
    data = tmp_path / "data"
    (data / "sessions" / "x" / "video").mkdir(parents=True)
    (data / "sessions" / "x" / "video" / "segment_000.mp4").write_bytes(b"RAW")
    r = dacl.probe(data, tmp_path / "probes")
    assert r["outcome"] == "allowed" and r["can_read_raw"] and r["can_write"] and r["user"]
    assert len(list((tmp_path / "probes").glob("probe_*.json"))) == 1
    assert not any((data / "sessions").glob(".acl_probe_*"))                              # test file removed


def test_acl_plan_is_never_executed_and_grants_the_app_first(tmp_path, monkeypatch):
    dacl = _load("dacl344b", "scripts/data_acl.py")
    amap = tmp_path / "amap.json"
    amap.write_text(json.dumps({"app_account": "PC\\app", "accounts": {"ANN-A": "PC\\ann_a"}}))
    monkeypatch.setattr(dacl.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("plan must not run anything")))
    r = dacl.plan(tmp_path / "data", amap, tmp_path / "out")
    cmds = r["commands"]
    gi = next(i for i, c in enumerate(cmds) if "/grant" in c and "PC\\app" in c)
    ii = next(i for i, c in enumerate(cmds) if "/inheritance:d" in c)
    ri = next(i for i, c in enumerate(cmds) if "/remove:g" in c)
    assert gi < ii < ri and any("/restore" in c for c in cmds) and any("(RX,W)" in c and "ann_a" in c for c in cmds)
    assert "NOTHING HAS BEEN RUN" in cmds[0]


# ================================================================== encryption
BDE_ON = "Volume D: [Data]\n    Conversion Status:    Fully Encrypted\n    Percentage Encrypted: 100.0%\n" \
         "    Encryption Method:    XTS-AES 128\n    Protection Status:    Protection On\n    Lock Status:          Unlocked\n"


def test_manage_bde_parsing_is_strict():
    assert readiness.parse_manage_bde(BDE_ON)["protection"] == "on"
    waiting = BDE_ON.replace("Protection On", "Protection Off")      # e.g. Device encryption waiting for activation
    assert readiness.parse_manage_bde(waiting)["protection"] == "suspended"
    prog = BDE_ON.replace("Fully Encrypted", "Encryption in Progress").replace("100.0%", "42.0%")
    assert readiness.parse_manage_bde(prog)["protection"] == "encrypting"
    off = "Conversion Status: Fully Decrypted\nPercentage Encrypted: 0.0%\nProtection Status: Protection Off\n"
    assert readiness.parse_manage_bde(off)["protection"] == "off"
    assert readiness.parse_manage_bde("ERROR: An attempt to access a required resource was denied.")["protection"] == "unknown"
    assert readiness.parse_manage_bde(BDE_ON.replace("100.0%", "99.0%"))["protection"] != "on"


def test_unverified_encryption_is_blocked_and_only_admin_records_count(tmp_path, acc, monkeypatch):
    monkeypatch.setattr(readiness, "bitlocker_status", lambda letter, probe=None: {
        "method": "shell_property", "protection": "on", "shell_value": "1", "admin_required": True})
    monkeypatch.setattr(readiness, "windows_edition", lambda: "Windows 11 Home")
    enc = tmp_path / "enc.jsonl"
    kw = dict(bitlocker_probe=None, drives=[{"letter": str(tmp_path.drive).rstrip(":"), "serial": "S1", "disk": 0}])
    it = _items(tmp_path, acc, **kw)["6_bitlocker_data_drive"]
    assert it["status"] == "NOT VERIFIED" and it["evidence"]["shell_indication"] == "on"   # Shell 'on' is a hint, not proof
    monkeypatch.setattr(readiness, "bitlocker_status", lambda letter, probe=None: {
        "method": "shell_property", "protection": "unknown", "shell_value": None, "admin_required": True})
    assert _items(tmp_path, acc, **kw)["6_bitlocker_data_drive"]["status"] == "BLOCKED"
    letter = str(tmp_path.drive).rstrip(":").upper()
    rec = {"at": datetime.now().isoformat(timespec="seconds"), "letter": letter, "serial": "S1", "method": "manage-bde",
           **readiness.parse_manage_bde(BDE_ON)}
    enc.write_text(json.dumps(dict(rec, elevated=False)) + "\n")                             # not elevated -> ignored
    assert _items(tmp_path, acc, **kw)["6_bitlocker_data_drive"]["status"] == "BLOCKED"
    enc.write_text(json.dumps(dict(rec, elevated=True, serial="OTHER")) + "\n")              # other volume -> ignored
    assert _items(tmp_path, acc, **kw)["6_bitlocker_data_drive"]["status"] == "BLOCKED"
    old = (datetime.now() - timedelta(days=readiness.ENCRYPTION_VALID_DAYS + 1)).isoformat(timespec="seconds")
    enc.write_text(json.dumps(dict(rec, elevated=True, at=old)) + "\n")                      # too old -> ignored
    assert _items(tmp_path, acc, **kw)["6_bitlocker_data_drive"]["status"] == "BLOCKED"
    enc.write_text(json.dumps(dict(rec, elevated=True)) + "\n")
    it = _items(tmp_path, acc, **kw)["6_bitlocker_data_drive"]
    assert it["status"] == "PASS" and it["evidence"]["source"] == "recorded administrator check"
    enc.write_text(json.dumps(dict(rec, elevated=True, **readiness.parse_manage_bde(BDE_ON.replace("Protection On", "Protection Off")))) + "\n")
    assert _items(tmp_path, acc, **kw)["6_bitlocker_data_drive"]["status"] == "FAIL"


# ================================================================== backup
def _real(sid="r1"):
    return TripMeta(sid, data_kind="real_driving", driver_consent=True, occupants_informed=True, supervisor_present=True,
                    configured_before_driving=True, regulations_checked=True, campaign_id="c1", trip_id="T1", driver_id="D01",
                    record_full_video=True)


def _src(tmp_path, n=3):
    src, camp = tmp_path / "sessions", tmp_path / "campaign"
    save_meta(_real(), src)
    for k in range(n):
        (src / "r1" / f"f{k}.csv").write_text(f"row {k}\n" * 50)
    camp.mkdir()
    return src, camp


def test_no_external_drive_and_too_small_drive_are_refused_before_copying(tmp_path, acc):
    cb = _cb()
    src, camp = _src(tmp_path)
    letter = str(tmp_path.drive).rstrip(":").upper()
    internal = [{"letter": letter, "disk": 0, "bus": "NVMe", "external": False, "free": 10**12}]
    with pytest.raises(cb.BackupError) as e:
        cb.backup(tmp_path / "dest", None, False, src, camp, drives=internal, bitlocker_probe=_bl("on"))
    assert "same physical disk" in str(e.value) and not (tmp_path / "dest" / "SafeDrive_backup").exists()
    assert _items(tmp_path, acc, drives=internal)["8_external_backup_destination"]["status"] == "BLOCKED"
    small = [{"letter": letter, "disk": 1, "bus": "USB", "external": True, "free": 10, "serial": "U"},
             {"letter": "Q", "disk": 0, "bus": "NVMe", "external": False, "free": 10**12}]
    chk = cb.check_destination(tmp_path / "dest", tmp_path / "Q_x", 10_000, small, allow_same_drive=True, bitlocker_probe=_bl("on"))
    assert any("not enough free space" in p for p in chk["problems"]) and not chk["ok"]


def test_drive_removed_or_full_mid_copy_keeps_only_verified_files(tmp_path, monkeypatch):
    cb = _cb()
    src, camp = _src(tmp_path)
    dest = tmp_path / "ext"
    real_copy, calls = cb.shutil.copy2, {"n": 0}

    def unplugged(a, b):
        calls["n"] += 1
        if calls["n"] == 3:
            raise FileNotFoundError(2, "The system cannot find the path specified (drive removed)")
        return real_copy(a, b)
    monkeypatch.setattr(cb.shutil, "copy2", unplugged)
    with pytest.raises(cb.BackupError) as e:
        cb.backup(dest, None, True, src, camp, drives=[])
    assert "interrupted" in str(e.value) and "reconnect" in e.value.recovery
    root = dest / "SafeDrive_backup"
    idx = json.loads((root / "backup_index.json").read_text())
    man = (root / cb.MANIFEST).read_text().splitlines()
    assert len(idx["files"]) == 2 and len(man) == 2                                        # manifest = verified files only
    rec = readiness.backup_records(camp / "backup_status.jsonl")[-1]
    assert rec["action"] == "backup" and rec["ok"] is False and "error" in rec
    monkeypatch.setattr(cb.shutil, "copy2", real_copy)
    out = cb.backup(dest, None, True, src, camp, drives=[])
    assert out["unchanged"] == 2 and out["copied"] >= 1
    v = cb.verify(dest, camp)
    assert v["ok"] and v["manifest_matches_index"] and v["files"] == v["files_present"] == len(idx["files"]) + out["copied"]


def test_corrupted_file_manifest_mismatch_and_restore_into_data_are_refused(tmp_path):
    cb = _cb()
    src, camp = _src(tmp_path)
    dest = tmp_path / "ext"
    cb.backup(dest, None, True, src, camp, drives=[])
    root = dest / "SafeDrive_backup"
    lines = (root / cb.MANIFEST).read_text().splitlines()
    assert all(len(x.split("  ")[0]) == 64 for x in lines)                                 # sha256  path
    (root / "sessions" / "r1" / "f1.csv").write_bytes(b"\x00corrupted")
    v = cb.verify(dest, camp)
    assert not v["ok"] and v["hash_mismatch"] == ["sessions/r1/f1.csv"]
    rt = cb.restore_test(dest, src, camp, tmp_parent=tmp_path)
    assert not rt["ok"] and rt["temp_folder_removed"]
    cb.backup(dest, None, True, src, camp, drives=[])                                      # repairs from the source
    assert cb.verify(dest, camp)["ok"]
    (root / cb.MANIFEST).write_text(lines[0] + "\n")                                       # manifest no longer matches
    assert not cb.verify(dest, camp)["manifest_matches_index"]
    (root / "backup_index.json").write_text("{broken")
    assert cb.verify(dest, camp)["ok"] is False
    with pytest.raises(cb.BackupError) as e:
        cb.restore(dest, "r1", src / "restored", protected=(src,))
    assert "never overwritten" in str(e.value)


def test_backup_report_is_written(tmp_path):
    cb = _cb()
    src, camp = _src(tmp_path)
    cb.backup(tmp_path / "ext", None, True, src, camp, drives=[])
    cb.verify(tmp_path / "ext", camp)
    cb.restore_test(tmp_path / "ext", src, camp, tmp_parent=tmp_path)
    r = cb.report(camp, tmp_path / "rep")
    from pathlib import Path
    md = Path(r["files"][0]).read_text(encoding="utf-8")
    assert "| backup |" in md and "| verify |" in md and "| restore_test |" in md
    assert r["valid_external_backups"] == 0                                               # test-mode backups never count


# ================================================================== privacy
def _shared(tmp_path, sid="s1", kind="controlled_test", n=2):
    root = tmp_path / "s"
    save_meta(TripMeta(sid, data_kind=kind, driver_consent=True, occupants_informed=True, supervisor_present=True,
                       configured_before_driving=True), root)
    d = root / sid
    (d / "shared").mkdir(parents=True)
    (d / "video").mkdir()
    (d / "video" / "segment_000.mp4").write_bytes(b"RAW")
    for k in range(n):
        (d / "shared" / f"segment_{k:03d}.mp4").write_bytes(b"BLURRED%d" % k)
    return root, d


SEGS = ["segment_000.mp4", "segment_001.mp4"]


def test_rejected_review_stale_copy_parked_requirement_and_review_log(tmp_path, acc):
    root, d = _shared(tmp_path)
    log = tmp_path / "review_log.jsonl"
    r = privacy.review(d, "COORD-01", True, SEGS, "plate readable at 3 s", identifiable_found=True, log_path=log)
    assert r["decision"] == "rejected" and privacy.shared_status(d, acc) == "rejected" and not privacy.shareable(d, acc)
    assert _items(tmp_path, acc)["11_privacy_review"]["status"] == "NOT VERIFIED"
    # approved, but the reviewer did not confirm the clip was recorded while PARKED -> does not count
    privacy.review(d, "COORD-01", True, SEGS, identifiable_found=False, log_path=log)
    assert _items(tmp_path, acc)["11_privacy_review"]["status"] == "NOT VERIFIED"
    privacy.review(d, "COORD-01", True, SEGS, identifiable_found=False, recorded_parked=True, log_path=log)
    it = _items(tmp_path, acc)["11_privacy_review"]
    assert it["status"] == "PASS" and it["evidence_at"]
    # the copy changes after the review -> stale, export refused, item back to NOT VERIFIED
    (d / "shared" / "segment_001.mp4").write_bytes(b"RE-ENCODED")
    assert privacy.shared_status(d, acc) == "stale" and not privacy.shareable(d, acc)
    assert _items(tmp_path, acc)["11_privacy_review"]["status"] == "NOT VERIFIED"
    assert (d / "video" / "segment_000.mp4").read_bytes() == b"RAW"                       # raw video untouched
    L = privacy.review_log(log)
    assert [x["decision"] for x in L] == ["rejected", "approved", "approved"]
    assert all(x["reviewer"] == "COORD-01" and x["reviewed_at"] and len(x["segments_sha256"]) == 2 for x in L)


def test_demo_copy_never_validates_and_no_sharing_blocks_export(tmp_path, acc):
    root, d = _shared(tmp_path, "demo1", "demo_video")
    privacy.review(d, "COORD-01", True, SEGS, identifiable_found=False, recorded_parked=True, log_path=tmp_path / "l.jsonl")
    assert _items(tmp_path, acc)["11_privacy_review"]["status"] == "NOT VERIFIED"
    privacy.set_sharing_policy(False, "COORD-01", "nothing is shared", tmp_path / "pol.json", acc)
    assert not privacy.shareable(d, acc, tmp_path / "pol.json")
    with pytest.raises(PermissionError):
        privacy.export_shared(d, tmp_path / "out", "COORD-01", log_path=tmp_path / "e.jsonl", policy_path=tmp_path / "pol.json")


# ================================================================== legal / camera checklists
def test_legal_checklist_needs_every_point_and_keeps_legal_advice_open(tmp_path, acc):
    att = tmp_path / "att.jsonl"
    assert _items(tmp_path, acc)["13_legal_review"]["status"] == "NOT VERIFIED"           # nothing confirmed
    ans = _confirmed(readiness.LEGAL_ITEMS)
    with pytest.raises(PermissionError):
        readiness.attest_checklist("legal_checklist", "ANN-A", ans, "memo", acc, att)
    with pytest.raises(ValueError):
        readiness.attest_checklist("legal_checklist", "COORD-01", {k: v for k, v in ans.items() if k != "retention"}, "memo", acc, att)
    with pytest.raises(ValueError):
        readiness.attest_checklist("legal_checklist", "COORD-01", dict(ans, local_rules={"status": "not_applicable", "basis": "n/a here"}),
                                   "memo", acc, att)
    with pytest.raises(ValueError):
        readiness.attest_checklist("legal_checklist", "COORD-01", dict(ans, purpose={"status": "confirmed", "basis": ""}), "memo", acc, att)
    with pytest.raises(ValueError):
        readiness.attest("legal_checklist", "COORD-01", "all fine", "memo", acc, att)    # no single-tick shortcut
    rec = readiness.attest_checklist("legal_checklist", "COORD-01", dict(ans, local_rules={"status": "needs_legal_advice"}),
                                     "memo", acc, att)
    assert not rec["complete"] and rec["needs_legal_advice"] == ["local_rules"]
    it = _items(tmp_path, acc)["13_legal_review"]
    assert it["status"] == "NOT VERIFIED" and "local_rules" in it["action"]
    readiness.attest_checklist("legal_checklist", "COORD-01", dict(ans, sharing={"status": "not_applicable", "basis": "no sharing at all"}),
                               "memo-2", acc, att)
    assert _items(tmp_path, acc)["13_legal_review"]["status"] == "PASS"
    readiness.withdraw("legal_checklist", "COORD-01", "rules changed", acc, att)
    assert _items(tmp_path, acc)["13_legal_review"]["status"] == "NOT VERIFIED"


def test_camera_not_verified_until_check_and_every_safety_point(tmp_path, acc):
    att, cam = tmp_path / "att.jsonl", tmp_path / "cam.jsonl"
    assert _items(tmp_path, acc)["12_camera_readiness"]["status"] == "NOT VERIFIED"
    cam.write_text(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), "ok": True}) + "\n")
    readiness.attest("camera_mount", "COORD-01", "fixed", "photo-1", acc, att)            # old single tick: not enough
    assert _items(tmp_path, acc)["12_camera_readiness"]["status"] == "NOT VERIFIED"
    ans = dict(_confirmed(readiness.CAMERA_SAFETY_ITEMS), airbag_clear={"status": "not_confirmed"})
    readiness.attest_checklist("camera_safety", "COORD-01", ans, "photo-1", acc, att)
    it = _items(tmp_path, acc)["12_camera_readiness"]
    assert it["status"] == "NOT VERIFIED" and it["evidence"]["safety_points_open"] == ["airbag_clear"]
    readiness.attest_checklist("camera_safety", "COORD-01", _confirmed(readiness.CAMERA_SAFETY_ITEMS), "photo-2", acc, att)
    assert _items(tmp_path, acc)["12_camera_readiness"]["status"] == "PASS"
    with open(cam, "a") as fh:
        fh.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), "ok": False, "problems": ["image very dark"]}) + "\n")
    assert _items(tmp_path, acc)["12_camera_readiness"]["status"] == "FAIL"


# ================================================================== annotator training
def _pack(tmp_path):
    d = tmp_path / training.PACK_ID
    (d / "video").mkdir(parents=True)
    save_meta(TripMeta(training.PACK_ID, data_kind="demo_video", notes=training.MARK), tmp_path)
    key = {"pack": training.PACK_ID, "clips": [
        {"clip": 0, "kind": "event", "t_start": 0, "t_end": 10, "accepted": ["collision", "near_miss", "risk_event"],
         "reference_event_time": 8.0},
        {"clip": 1, "kind": "normal", "t_start": 11, "t_end": 21, "accepted": ["normal"], "reference_event_time": None},
        {"clip": 2, "kind": "uncertain", "t_start": 22, "t_end": 32, "accepted": ["uncertain"], "reference_event_time": 30.0}]}
    (d / "training_key.json").write_text(json.dumps(key))
    return d


def _label(d, n, dt=0.2):
    ann.add(d, ann.Annotation(training.PACK_ID, n, "collision", 6, 9, 8 + dt, evidence="car hits barrier", evidence_frames=[8.0]))
    ann.add(d, ann.Annotation(training.PACK_ID, n, "near_miss", 2, 4, 3, evidence="close pass", evidence_frames=[3.0]))
    ann.add(d, ann.Annotation(training.PACK_ID, n, "risk_event", 4.2, 5.5, 5, evidence="lead brakes", evidence_frames=[5.0]))
    ann.add(d, ann.Annotation(training.PACK_ID, n, "normal", 11, 21))
    ann.add(d, ann.Annotation(training.PACK_ID, n, "uncertain", 25, 31))


def test_training_incomplete_until_both_submit_and_locked_by_hash(tmp_path, acc):
    d = _pack(tmp_path)
    subs = tmp_path / "subs.jsonl"
    _label(d, "ANN-A")
    with pytest.raises(training.TrainingError):
        training.submit(d, "Stranger", subs, acc)                                          # not a campaign annotator
    with pytest.raises(training.TrainingError):
        training.submit(d, "ANN-B", subs, acc)                                             # nothing labelled yet
    training.submit(d, "ANN-A", subs, acc)
    with pytest.raises(training.TrainingError):
        training.submit(d, "ANN-A", subs, acc)                                             # one submission only
    assert training.is_locked(d, "ANN-A", subs) and not training.key_visible(d, ["ANN-A", "ANN-B"], subs)
    rep = training.report_if_available(tmp_path, subs, acc)
    assert not rep["complete"] and rep["criteria_results"] is None and rep["submitted"] == ["ANN-A"]   # no score leaks
    assert "per_annotator" not in rep
    it = _items(tmp_path, acc, training_report=rep)["14_annotation_training"]
    assert it["status"] == "NOT VERIFIED"
    _label(d, "ANN-B", -0.3)
    training.submit(d, "ANN-B", subs, acc)
    rep = training.report_if_available(tmp_path, subs, acc)
    assert rep["complete"] and rep["passed"], rep["criteria_results"]
    disc = rep["per_annotator"]["ANN-A"]["discrimination_by_clip_kind"]
    assert disc["normal"]["clips_with_accepted_label"] == 1 and disc["uncertain"]["clips_with_accepted_label"] == 1
    training.save_report(rep, tmp_path / "rep.json", tmp_path / "results")
    assert sorted(p.name for p in (tmp_path / "results").glob("*.json")) == ["ANN-A.json", "ANN-B.json"]
    assert _items(tmp_path, acc, training_report=rep)["14_annotation_training"]["status"] == "PASS"
    # a label edited after submission invalidates the result
    ann.add(d, ann.Annotation(training.PACK_ID, "ANN-B", "normal", 0, 1))
    rep = training.report_if_available(tmp_path, subs, acc)
    assert not rep["complete"] and rep["invalid"]["ANN-B"] == "labels changed after submission"
    assert _items(tmp_path, acc, training_report=rep)["14_annotation_training"]["status"] == "NOT VERIFIED"
    # isolation: the pack is never under data/sessions
    assert "data/sessions" not in str(training.TRAINING_SESSIONS).replace("\\", "/")


def test_training_failing_annotator_gives_fail(tmp_path, acc):
    d = _pack(tmp_path)
    subs = tmp_path / "subs.jsonl"
    _label(d, "ANN-A")
    ann.add(d, ann.Annotation(training.PACK_ID, "ANN-B", "normal", 0, 32))                # misses the event
    for n in ("ANN-A", "ANN-B"):
        training.submit(d, n, subs, acc)
    rep = training.report_if_available(tmp_path, subs, acc)
    assert rep["complete"] and not rep["passed"]
    assert _items(tmp_path, acc, training_report=rep)["14_annotation_training"]["status"] == "FAIL"


# ================================================================== checklist / provenance / GO
def test_expired_or_incomplete_checklist_refuses_real_driving(tmp_path, acc):
    cl = tmp_path / "cl"
    manual = {k: True for k in checklist.MANUAL_ITEMS}
    assert {"airbag_clear", "no_view_obstruction", "cable_safe", "no_laptop_while_moving", "stop_procedure"} <= set(manual)
    rec = checklist.save("OP1", "SUP1", {k: v for k, v in manual.items() if k != "airbag_clear"}, {"x": {"ok": True, "detail": ""}}, cl)
    assert not rec["complete"] and rec["missing_manual"] == ["airbag_clear"]
    with pytest.raises(checklist.ChecklistError):
        checklist.require_ready("models/safedrive_v3", cl, data_dir=resolve("data"))
    checklist.save("OP1", "SUP1", manual, {"x": {"ok": True, "detail": ""}}, cl)
    later = datetime.now() + timedelta(hours=checklist.CHECKLIST_VALID_H, minutes=1)
    with pytest.raises(checklist.ChecklistError) as e:
        checklist.require_ready("models/safedrive_v3", cl, now=later, data_dir=resolve("data"))
    assert "older than" in str(e.value)


def test_provenance_hash_mismatch_fails_hash_items(tmp_path, acc):
    ref = {m.split("/")[-1]: dict(provenance.compute(m)) for m in readiness.MODELS}
    for m in ref:
        ref[m]["threshold_config_sha256"] = "0" * 64
    it = _items(tmp_path, acc, frozen_ref=ref)
    assert it["3_threshold_hash"]["status"] == "FAIL" and it["1_model_hash"]["status"] == "PASS"


def test_no_go_while_anything_is_blocked_or_not_verified(tmp_path, acc):
    items = readiness.check_all(**_kw(tmp_path, acc))
    dec = readiness.go_decision(items)
    assert dec["verdict"] == "NO-GO" and len(items) == 15
    assert {"BLOCKED", "NOT VERIFIED"} & {b["status"] for b in dec["blocking"]}
    assert all("checked_at" in i and "evidence_at" in i for i in items)
    for bad in ("BLOCKED", "NOT VERIFIED", "FAIL"):
        mixed = [readiness.item("k", "t", "PASS", {})] * 14 + [readiness.item("z", "t", bad, {})]
        assert readiness.go_decision(mixed)["verdict"] == "NO-GO"


# ================================================================== VeraCrypt (Windows Home alternative to BitLocker)
def test_veracrypt_volume_needs_driver_device_and_mount_list(tmp_path, monkeypatch):
    monkeypatch.setattr(readiness, "_dos_device", lambda l: "\Device\VeraCryptVolumeX" if l == "X" else "\Device\HarddiskVolume5")
    monkeypatch.setattr(readiness, "_vc_driver_list", lambda: {"X": {"container": "E:\safedrive_backup.hc", "size": 30 << 30, "ea": 1}})
    v = readiness.veracrypt_volume("X")
    assert v["container_drive"] == "E" and v["device"].endswith("VeraCryptVolumeX")
    assert readiness.veracrypt_volume("E") is None                               # an ordinary volume
    monkeypatch.setattr(readiness, "_vc_driver_list", lambda: {})                # device name alone (driver says not mounted)
    assert readiness.veracrypt_volume("X") is None
    monkeypatch.setattr(readiness, "_vc_driver_list", lambda: {"X": {"container": "E:\safedrive_backup.hc", "size": 1, "ea": 1}})
    monkeypatch.setattr(readiness, "bitlocker_status", lambda letter, probe=None: {"method": "shell_property", "protection": "unknown"})
    assert readiness.encryption_evidence("X", path=tmp_path / "none.jsonl")["protection"] == "on"
    assert readiness.encryption_evidence("E", path=tmp_path / "none.jsonl")["protection"] == "unknown"


def test_backup_to_veracrypt_volume_on_usb_is_accepted_but_not_when_container_is_on_the_data_disk(tmp_path):
    cb = _cb()
    letter = str(tmp_path.drive).rstrip(":").upper()
    vc = lambda l: {"method": "VeraCrypt driver", "protection": "on"} if l == letter else {"protection": "off"}  # noqa: E731
    on_usb = [{"letter": letter, "disk": 1, "bus": "USB", "external": True, "free": 10**12, "serial": "V1",
               "veracrypt_container": "E:\safedrive_backup.hc"},
              {"letter": "Q", "disk": 0, "bus": "NVMe", "external": False, "free": 10**12}]
    from pathlib import Path
    ok = cb.check_destination(tmp_path / "dest", Path("Q:/data"), 10, on_usb, bitlocker_probe=vc)
    assert not any("same physical disk" in p or "encryption" in p or "external" in p for p in ok["problems"]), ok["problems"]
    on_data_disk = [dict(on_usb[0], disk=0, bus="NVMe", external=False), on_usb[1]]
    bad = cb.check_destination(tmp_path / "dest", Path("Q:/data"), 10, on_data_disk, bitlocker_probe=vc)
    assert any("same physical disk" in p for p in bad["problems"])


def test_probe_written_by_powershell_with_bom_is_read(tmp_path):
    pd = tmp_path / "p"
    pd.mkdir()
    (pd / "probe_x.json").write_bytes(b"\xef\xbb\xbf" + json.dumps(DENIED).encode())
    assert readiness.acl_probes(pd)[0]["outcome"] == "denied"


def test_session_store_behind_a_junction_is_what_the_audit_checks(tmp_path, acc, monkeypatch):
    """data/sessions may point into an encrypted volume; items 6 / 7 / 15 must check the real location, not data/."""
    import subprocess
    data, vault = tmp_path / "data", tmp_path / "vault"
    data.mkdir()
    (vault / "sessions").mkdir(parents=True)
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(data / "sessions"), str(vault / "sessions")], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("junctions not available")
    assert readiness.sessions_location(data) == (vault / "sessions").resolve()
    assert (vault / "sessions").resolve() in [p.resolve() for p in readiness.acl_targets(data)]
    seen = []
    readiness.check_all(**_kw(tmp_path, acc, data_dir=data, bitlocker_probe=lambda l: seen.append(l) or {"protection": "on"}))
    assert seen and seen[0] == str(vault.resolve().drive).rstrip(":").upper()
