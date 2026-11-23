"""v3.4.3 / v3.4.4 campaign backup / verify / restore / restore-test / report with SHA-256 verification (no compression, no upload).

  drives        list drives with bus type, physical disk, free space and encryption state (read-only)
  check         check a destination without copying anything
  backup        copy data/sessions/<id>/ (every session with trip metadata, or --sessions ...) and data/campaign/ to
                <dest>/SafeDrive_backup/, verify every copied file by SHA-256 and keep backup_index.json (path -> sha256).
                Incremental: files already present with the same hash are skipped.
  verify        re-hash a backup against its backup_index.json (detects bit rot / tampering / missing files).
  restore       copy a backed-up session (or all) to --to <folder> (never overwrites an existing file), then verify.
  restore-test  restore everything into a NEW temporary folder, compare every hash with the index (and with the live source
                files that still exist), then delete the temporary folder.

  report        write experiments/backup_reports/backup_report_<time>.md/.json from the action log.

Every action appends a record to data/campaign/backup_status.jsonl; the readiness audit reads it.
v3.4.4: each backup also writes <dest>/SafeDrive_backup/MANIFEST.sha256 ('sha256  path', checkable with any tool); verify checks
file count, hashes and that the manifest equals the index; restore refuses any target inside data/; the destination's
encryption is read with the system tool (manage-bde, administrator) or taken from a recent administrator record
(scripts/encryption_status.py) - the Shell property alone is never accepted.

Destination rules (checked before anything is copied):
  * the drive must be connected and writable, and the path must not be inside the project;
  * it must be on a DIFFERENT physical disk than the data (C: and D: of this laptop are the SAME disk) - refused otherwise;
  * it must be an external drive (USB / SD) - refused otherwise;
  * it must have enough free space for what will be copied (+10 %);
  * it should be encrypted (BitLocker To Go or equivalent); an unencrypted drive is refused unless --allow-unencrypted, and such
    a backup is recorded as unencrypted (the audit then reports the destination as FAIL).
--allow-same-drive / --allow-internal exist ONLY for automated tests; such records are marked and never count for the audit.

Examples:
  python scripts/campaign_backup.py drives
  python scripts/campaign_backup.py backup       --dest E:\\
  python scripts/campaign_backup.py verify       --dest E:\\
  python scripts/campaign_backup.py restore-test --dest E:\\
  python scripts/campaign_backup.py restore      --dest E:\\ --session 20261201_090000_ab12 --to D:\\restore_test
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.realworld import readiness  # noqa: E402
from src.realworld.provenance import sha256_file  # noqa: E402
from src.utils.config import resolve  # noqa: E402

SESSIONS, CAMPAIGN = resolve("data/sessions"), resolve("data/campaign")
PROJECT = resolve(".")


class BackupError(RuntimeError):
    """A refused or failed backup action. .recovery says what to do."""

    def __init__(self, msg: str, recovery: str = ""):
        super().__init__(msg)
        self.recovery = recovery


def _files(root: Path) -> list[Path]:
    return sorted(f for f in Path(root).rglob("*") if f.is_file())


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _status_path(campaign_root: Path) -> Path:
    return Path(campaign_root) / "backup_status.jsonl"


def _log(campaign_root: Path, rec: dict) -> dict:
    readiness._append_jsonl(_status_path(campaign_root), rec)
    return rec


def _letter(p: Path) -> str:
    return Path(p).resolve().drive.rstrip(":").upper()


def check_destination(dest: Path, data_root: Path = SESSIONS, need_bytes: int = 0, drives: list[dict] | None = None,
                      allow_same_drive: bool = False, allow_internal: bool = False, allow_unencrypted: bool = False,
                      bitlocker_probe=None) -> dict:
    """Return {ok, problems, warnings, ...}. Nothing is copied."""
    dest = Path(dest)
    problems, warnings = [], []
    drv = drives if drives is not None else readiness.list_drives()
    dl, sl = _letter(dest), _letter(data_root)
    dinfo = next((d for d in drv if str(d.get("letter", "")).upper() == dl), None)
    sinfo = next((d for d in drv if str(d.get("letter", "")).upper() == sl), None)
    if not Path(dest.anchor or dest).exists():
        problems.append(f"drive {dl}: is not connected")
    if drv and dinfo is None and dest.anchor and not allow_same_drive:
        problems.append(f"drive {dl}: not found in the drive list")
    try:
        Path(dest).resolve().relative_to(PROJECT)
        problems.append("destination is inside the project folder")
    except ValueError:
        pass
    same_letter = dl == sl
    same_disk = same_letter or (dinfo is not None and sinfo is not None and dinfo.get("disk") is not None
                                and dinfo.get("disk") == sinfo.get("disk"))
    if same_disk and not allow_same_drive:
        problems.append(f"destination {dl}: is on the same physical disk as the data ({sl}:); a backup there does not protect "
                        "against disk failure or theft")
    external = bool(dinfo and dinfo.get("external"))
    if not external and not same_disk and not allow_internal:
        problems.append(f"destination {dl}: is not an external (USB / SD) drive")
    enc = readiness.encryption_evidence(dl, (dinfo or {}).get("serial"), bitlocker_probe).get("protection") if dinfo else "unknown"
    if enc not in ("on", "on_locked"):
        (warnings if allow_unencrypted or allow_same_drive else problems).append(
            f"destination encryption is '{enc}' (BitLocker To Go or equivalent required; --allow-unencrypted records it as unencrypted)")
    free = (dinfo or {}).get("free")
    if free is None and Path(dest.anchor or dest).exists():
        free = shutil.disk_usage(dest.anchor or dest).free
    if free is not None and need_bytes and free < need_bytes * 1.1:
        problems.append(f"not enough free space on {dl}: {free / 1e9:.2f} GB free, {need_bytes * 1.1 / 1e9:.2f} GB needed")
    if not problems:
        try:
            Path(dest).mkdir(parents=True, exist_ok=True)
            t = Path(dest) / ".safedrive_write_test"
            t.write_bytes(b"x")
            t.unlink()
        except OSError as e:
            problems.append(f"destination not writable: {e}")
    return {"ok": not problems, "problems": problems, "warnings": warnings, "dest": str(dest), "letter": dl,
            "dest_serial": (dinfo or {}).get("serial"), "external": external, "same_disk": same_disk, "encryption": enc,
            "free_bytes": free, "need_bytes": need_bytes, "test_mode": allow_same_drive or allow_internal}


MANIFEST = "MANIFEST.sha256"
OWN_FILES = {"backup_index.json", MANIFEST}


def _write_index(root: Path, idx: dict) -> None:
    """backup_index.json (tool) + MANIFEST.sha256 (plain 'sha256  path' lines, checkable without this software)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "backup_index.json").write_text(json.dumps(idx, indent=1), encoding="utf-8")
    (root / MANIFEST).write_text("".join(f"{h}  {rel}\n" for rel, h in sorted(idx["files"].items())), encoding="utf-8")


def _sources(sessions: list[str], sessions_root: Path, campaign_root: Path):
    return [(Path(sessions_root) / s, Path("sessions") / s) for s in sessions] + \
        ([(Path(campaign_root), Path("campaign"))] if Path(campaign_root).exists() else [])


def backup(dest: Path, sessions: list[str] | None, allow_same_drive: bool, sessions_root: Path = SESSIONS,
           campaign_root: Path = CAMPAIGN, drives: list[dict] | None = None, allow_internal: bool = False,
           allow_unencrypted: bool = False, bitlocker_probe=None) -> dict:
    sessions_root, campaign_root = Path(sessions_root), Path(campaign_root)
    root = Path(dest) / "SafeDrive_backup"
    idx_p = root / "backup_index.json"
    idx = json.loads(idx_p.read_text(encoding="utf-8")) if idx_p.exists() else {"files": {}, "runs": []}
    if sessions is None:
        sessions = readiness.campaign_sessions(sessions_root)
    todo = []
    for src_dir, rel_dir in _sources(sessions, sessions_root, campaign_root):
        for f in _files(src_dir):
            if f.name == "backup_status.jsonl" and f.parent == campaign_root:
                continue  # copied at the end of the run (it changes during the run)
            todo.append((f, (rel_dir / f.relative_to(src_dir)).as_posix()))
    need = sum(f.stat().st_size for f, rel in todo if not (root / rel).exists())
    chk = check_destination(dest, sessions_root, need, drives, allow_same_drive, allow_internal, allow_unencrypted, bitlocker_probe)
    base = {"action": "backup", "at": _now(), "dest": str(dest), "dest_serial": chk["dest_serial"], "external": chk["external"],
            "same_disk": chk["same_disk"], "encryption": chk["encryption"], "test_mode": chk["test_mode"], "sessions": sessions}
    if not chk["ok"]:
        _log(campaign_root, {**base, "ok": False, "refused": chk["problems"]})
        raise BackupError("destination refused: " + "; ".join(chk["problems"]),
                          "Connect an external encrypted drive with enough space and run the command again. Nothing was copied.")
    copied = skipped = 0
    failed: list[str] = []
    try:
        for f, rel in todo:
            h = sha256_file(f)
            tgt = root / rel
            if idx["files"].get(rel) == h and tgt.exists() and sha256_file(tgt) == h:
                skipped += 1
                continue
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, tgt)
            if sha256_file(tgt) != h:
                idx["files"].pop(rel, None)
                failed.append(rel)
                continue
            idx["files"][rel] = h
            copied += 1
    except OSError as e:
        rec = _log(campaign_root, {**base, "ok": False, "copied": copied, "unchanged": skipped, "error": str(e)})
        idx["runs"].append({k: rec[k] for k in ("at", "copied", "unchanged", "error")})
        try:
            _write_index(root, idx)  # the index lists only files verified so far
        except OSError:
            pass  # drive gone: the next run rebuilds from what is still verifiable
        raise BackupError(f"backup interrupted: {e}", "The source data is unchanged. Free space / reconnect the drive and run "
                          "backup again: verified files are skipped, the rest is copied.") from e
    ok = not failed
    run = {"at": base["at"], "sessions": sessions, "copied": copied, "unchanged": skipped, "failed_verification": failed}
    idx["runs"].append(run)
    _write_index(root, idx)
    rec = _log(campaign_root, {**base, "ok": ok, "copied": copied, "unchanged": skipped, "failed_verification": failed,
                               "index_files": len(idx["files"]), "warnings": chk["warnings"]})
    st = _status_path(campaign_root)
    if st.exists():
        (root / "campaign").mkdir(parents=True, exist_ok=True)
        shutil.copy2(st, root / "campaign" / st.name)
    if not ok:
        raise BackupError(f"{len(failed)} file(s) differed after copy: {failed[:5]}", "The drive may be failing. Run verify; "
                          "use another drive if the problem repeats.")
    return {**run, "warnings": chk["warnings"], "record": rec}


def verify(dest: Path, campaign_root: Path | None = None) -> dict:
    root = Path(dest) / "SafeDrive_backup"
    idx_p = root / "backup_index.json"
    if not idx_p.exists():
        r = {"action": "verify", "at": _now(), "dest": str(dest), "ok": False, "error": "no backup_index.json at destination"}
        if campaign_root is not None:
            _log(campaign_root, r)
        return r
    try:
        idx = json.loads(idx_p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        r = {"action": "verify", "at": _now(), "dest": str(dest), "ok": False, "error": f"backup index unreadable: {e}",
             "recovery": "Run backup again to rebuild the index from the files that still match the source."}
        if campaign_root is not None:
            _log(campaign_root, r)
        return r
    missing, bad = [], []
    for rel, h in idx["files"].items():
        p = root / rel
        try:
            if not p.exists():
                missing.append(rel)
            elif sha256_file(p) != h:
                bad.append(rel)
        except OSError:
            missing.append(rel)  # unreadable (drive removed / bad sector)
    # the plain-text manifest must list exactly the same files and hashes
    man_p = root / MANIFEST
    manifest = {}
    if man_p.exists():
        for line in man_p.read_text(encoding="utf-8").splitlines():
            if "  " in line:
                h_, rel_ = line.split("  ", 1)
                manifest[rel_] = h_
    manifest_ok = manifest == idx["files"]
    on_disk = {f.relative_to(root).as_posix() for f in _files(root)} - OWN_FILES - {"campaign/backup_status.jsonl"}
    extra = sorted(on_disk - set(idx["files"]))
    r = {"action": "verify", "at": _now(), "dest": str(dest), "dest_serial": _serial(dest) if campaign_root is not None else None,
         "files": len(idx["files"]), "files_present": len(idx["files"]) - len(missing), "manifest_matches_index": manifest_ok,
         "unindexed_files": extra[:20], "missing": missing, "hash_mismatch": bad,
         "ok": not missing and not bad and manifest_ok and len(idx["files"]) > 0}
    if not r["ok"]:
        r["recovery"] = ("Run backup again: missing / damaged files are re-copied from data/ if the source still exists. If the "
                         "source is gone, the damaged files cannot be repaired from this drive.")
    if campaign_root is not None:
        _log(campaign_root, r)
    return r


def _serial(dest: Path):
    try:
        d = next((x for x in readiness.list_drives() if str(x.get("letter", "")).upper() == _letter(dest)), None)
        return (d or {}).get("serial")
    except Exception:  # noqa: BLE001
        return None


def restore(dest: Path, session: str | None, to: Path, protected: tuple[Path, ...] | None = None) -> dict:
    """Restore into an EMPTY folder outside data/ (the original data is never overwritten)."""
    for p in (protected if protected is not None else (resolve("data"),)):
        try:
            Path(to).resolve().relative_to(Path(p).resolve())
            raise BackupError(f"refusing to restore into {to}: it is inside {p} (original data is never overwritten)",
                              "Restore into a new empty folder outside data/, check it, then copy single files by hand if needed.")
        except ValueError:
            pass
    root = Path(dest) / "SafeDrive_backup"
    idx = json.loads((root / "backup_index.json").read_text(encoding="utf-8"))
    pref = f"sessions/{session}/" if session else "sessions/"
    n = 0
    for rel, h in idx["files"].items():
        if not rel.startswith(pref):
            continue
        tgt = Path(to) / Path(rel).relative_to("sessions")
        if tgt.exists():
            raise BackupError(f"refusing to overwrite existing file {tgt}", "Restore into an empty folder.")
        if not (root / rel).exists():
            raise BackupError(f"backup file missing: {rel}", "Run verify; re-run backup while the source still exists.")
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, tgt)
        if sha256_file(tgt) != h:
            raise BackupError(f"restored file differs from backup index: {rel}", "The backup copy is damaged; run verify.")
        n += 1
    return {"restored_files": n, "to": str(to), "verified": True}


def restore_test(dest: Path, sessions_root: Path = SESSIONS, campaign_root: Path = CAMPAIGN, tmp_parent: Path | None = None) -> dict:
    """Restore EVERYTHING into a new temporary folder, compare with the index and with live sources, then delete the folder."""
    root = Path(dest) / "SafeDrive_backup"
    rec = {"action": "restore_test", "at": _now(), "dest": str(dest), "dest_serial": _serial(dest)}
    tmp = Path(tempfile.mkdtemp(prefix="safedrive_restore_test_", dir=str(tmp_parent) if tmp_parent else None))
    try:
        idx = json.loads((root / "backup_index.json").read_text(encoding="utf-8"))
        n, mism, live_diff, missing = 0, [], [], []
        for rel, h in idx["files"].items():
            src = root / rel
            if not src.exists():
                missing.append(rel)
                continue
            tgt = tmp / rel
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, tgt)
            hh = sha256_file(tgt)
            if hh != h:
                mism.append(rel)
            live = (Path(sessions_root) / Path(rel).relative_to("sessions")) if rel.startswith("sessions/") else \
                (Path(campaign_root) / Path(rel).relative_to("campaign"))
            if live.exists() and live.name != "backup_status.jsonl" and sha256_file(live) != hh:
                live_diff.append(rel)  # source changed after the backup (e.g. new annotations): back up again
            n += 1
        rec.update({"restored_files": n, "missing": missing, "hash_mismatch": mism, "newer_in_source": live_diff,
                    "ok": n > 0 and not mism and not missing})
        if live_diff:
            rec["note"] = "files changed in data/ since the backup; run backup again (the restored copies themselves are intact)"
    except (OSError, json.JSONDecodeError) as e:
        rec.update({"ok": False, "error": str(e), "recovery": "check that the backup drive is connected and run backup + verify"})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        rec["temp_folder_removed"] = not tmp.exists()
    _log(campaign_root, rec)
    return rec


def report(campaign_root: Path = CAMPAIGN, out_dir: Path | None = None) -> dict:
    """Backup / restore report from backup_status.jsonl (what was done, when, to which drive, with which result)."""
    recs = readiness.backup_records(_status_path(campaign_root))
    last = {a: readiness._latest(recs, a) for a in ("backup", "verify", "restore_test")}
    counted = [r for r in recs if r.get("action") == "backup" and r.get("ok") and r.get("external") and not r.get("test_mode")
               and not r.get("same_disk")]
    rep = {"at": _now(), "records": len(recs), "valid_external_backups": len(counted), "last": last,
           "refused_or_failed": [{k: r.get(k) for k in ("action", "at", "dest", "refused", "error", "missing", "hash_mismatch")}
                                 for r in recs if not r.get("ok")][-10:]}
    lines = ["# SafeDrive backup / restore report", "", f"Generated {rep['at']} from `{_status_path(campaign_root)}` "
             f"({len(recs)} records).", "", f"Valid external backups (counted by the readiness audit): **{len(counted)}**", "",
             "| Action | Last run | Result | Detail |", "|---|---|---|---|"]
    for a, r in last.items():
        if r is None:
            lines.append(f"| {a} | never | — | — |")
        else:
            det = {k: r.get(k) for k in ("dest", "copied", "files", "restored_files", "missing", "hash_mismatch", "refused", "error")
                   if r.get(k) not in (None, [], "")}
            lines.append(f"| {a} | {r.get('at')} | {'OK' if r.get('ok') else 'NOT OK'} | {json.dumps(det, ensure_ascii=False)[:300]} |")
    if rep["refused_or_failed"]:
        lines += ["", "## Refused or failed actions (latest 10)", ""] + \
                 [f"- {x['at']} {x['action']}: {json.dumps({k: v for k, v in x.items() if v and k not in ('action', 'at')}, ensure_ascii=False)[:300]}"
                  for x in rep["refused_or_failed"]]
    out = Path(out_dir) if out_dir else resolve("experiments/backup_reports")
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / f"backup_report_{stamp}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / f"backup_report_{stamp}.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    rep["files"] = [str(out / f"backup_report_{stamp}.md"), str(out / f"backup_report_{stamp}.json")]
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["drives", "check", "backup", "verify", "restore", "restore-test", "report"])
    ap.add_argument("--dest")
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--session")
    ap.add_argument("--to")
    ap.add_argument("--allow-same-drive", action="store_true", help="TESTS ONLY")
    ap.add_argument("--allow-internal", action="store_true", help="TESTS ONLY")
    ap.add_argument("--allow-unencrypted", action="store_true", help="records the backup as unencrypted")
    a = ap.parse_args()
    try:
        if a.action == "drives":
            rows = readiness.list_drives()
            for d in rows:
                d["encryption"] = readiness.encryption_evidence(d["letter"], d.get("serial")).get("protection")
            print(json.dumps(rows, indent=1))
            if not any(d["external"] for d in rows):
                print("BLOCKED: no external drive connected.")
            return 0
        if a.action == "report":
            r = report()
            print("\n".join(r["files"]))
            return 0
        if not a.dest:
            raise SystemExit("--dest is required")
        if a.action == "check":
            r = check_destination(Path(a.dest), allow_unencrypted=a.allow_unencrypted)
            print(json.dumps(r, indent=1))
            return 0 if r["ok"] else 2
        if a.action == "backup":
            print(json.dumps(backup(Path(a.dest), a.sessions, a.allow_same_drive, allow_internal=a.allow_internal,
                                    allow_unencrypted=a.allow_unencrypted), indent=1, default=str))
        elif a.action == "verify":
            r = verify(Path(a.dest), CAMPAIGN)
            print(json.dumps(r, indent=1))
            return 0 if r["ok"] else 2
        elif a.action == "restore-test":
            r = restore_test(Path(a.dest))
            print(json.dumps(r, indent=1))
            return 0 if r["ok"] else 2
        else:
            if not a.to:
                raise SystemExit("restore needs --to")
            print(json.dumps(restore(Path(a.dest), a.session, Path(a.to)), indent=1))
    except BackupError as e:
        print(f"ERROR: {e}\nRECOVERY: {e.recovery}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
