"""v3.4.4 folder-permission tool for D:\\SanphambpAI\\data (raw campaign video). It NEVER changes permissions itself.

  inspect    show the measured ACL of data/, data/sessions, data/campaign: every principal, its rights split into
             read / write / delete / change_permissions, whether the entry is inherited (and from where), and the
             least-privilege evaluation against data/campaign/windows_accounts.json.
  snapshot   record the current ACL (fingerprint + entries) in data/campaign/acl_snapshots.jsonl and save a restorable copy
             (icacls /save) under backups/acl/. Run it BEFORE and AFTER any change.
  compare    show what changed between the current ACL and a snapshot (default: the latest one).
  plan       print (and save to experiments/acl/) the exact icacls commands for least privilege, with the rollback command.
             Nothing is executed: a person reads the plan, confirms, and runs it.
  probe      run this AS THE ACCOUNT BEING TESTED (log in as that Windows user): tries to list data/sessions, read one byte of a
             raw video and create + delete a test file; writes the result to acl_probes/ (outside data/, so a denied account can
             still write it). A standard, non-authorised account must be DENIED; authorised accounts must be allowed.

Why data/ currently fails: D:\\ (a non-system NTFS volume) carries the Windows default ACL "Authenticated Users: Modify" and
"Users: Read & execute"; D:\\SanphambpAI and data/ inherit it, so every Windows account on this PC can read and change raw video.

Examples:
  python scripts/data_acl.py inspect
  python scripts/data_acl.py snapshot
  python scripts/data_acl.py plan
  python scripts/data_acl.py compare
  python scripts/data_acl.py probe          (as the tested account)
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.realworld import privacy, readiness  # noqa: E402
from src.utils.config import resolve  # noqa: E402

DATA = resolve("data")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def whoami() -> str:
    try:
        r = subprocess.run(["whoami"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return f"{os.environ.get('USERDOMAIN', '')}\\{getpass.getuser()}".strip("\\")


def admin_member() -> bool:
    """True if the account belongs to BUILTIN\\Administrators (even when the current token is not elevated)."""
    try:
        r = subprocess.run(["whoami", "/groups"], capture_output=True, text=True, timeout=10)
        return "S-1-5-32-544" in r.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


def inspect(data_dir: Path = DATA, account_map_path: Path = readiness.ACCOUNT_MAP_PATH, probe=None) -> dict:
    acls = readiness.measure_acls(data_dir, probe)
    acc = privacy.load_access()
    amap = readiness.load_account_map(account_map_path)
    ev = readiness.evaluate_acl(acls, amap, acc.get("raw_access", []), acc.get("coordinators", []))
    chain = []
    p = Path(data_dir).resolve()
    for q in [p, *p.parents]:
        a = readiness.folder_acl(q, probe)
        chain.append({"path": str(q), "explicit": [f"{e['principal']} {e['rights']}" for e in a.get("entries", [])
                                                   if "(I)" not in e["rights"]]})
    return {"at": _now(), "user": whoami(), "fingerprint": readiness.acl_fingerprint(acls), "evaluation": ev,
            "inheritance_chain_explicit_entries": chain, "account_map": amap or None}


def snapshot(data_dir: Path = DATA, path: Path = readiness.ACL_SNAPSHOTS, save_dir: Path | None = None, probe=None,
             icacls_save: bool = True) -> dict:
    acls = readiness.measure_acls(data_dir, probe)
    rec = {"at": _now(), "by": whoami(), "fingerprint": readiness.acl_fingerprint(acls),
           "acls": {k: v.get("entries", []) for k, v in acls.items()}}
    if icacls_save and sys.platform == "win32":
        sd = Path(save_dir) if save_dir else resolve("backups/acl")
        sd.mkdir(parents=True, exist_ok=True)
        f = sd / f"data_acl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        r = subprocess.run(["icacls", str(data_dir), "/save", str(f), "/t", "/c", "/q"], capture_output=True, text=True, timeout=600)
        rec["icacls_save"] = {"file": str(f), "rc": r.returncode,
                              "restore_command": f'icacls "{Path(data_dir).parent}" /restore "{f}" /c'}
    readiness._append_jsonl(path, rec)
    return rec


def compare(data_dir: Path = DATA, path: Path = readiness.ACL_SNAPSHOTS, probe=None, ref: dict | None = None) -> dict:
    snaps = readiness._read_jsonl(path)
    ref = ref or (snaps[-1] if snaps else None)
    cur = {k: v.get("entries", []) for k, v in readiness.measure_acls(data_dir, probe).items()}
    if ref is None:
        return {"error": "no snapshot yet: run snapshot first", "current": cur}
    diff = {}
    for p in sorted(set(cur) | set(ref["acls"])):
        a = {f"{e['principal']} {e['rights']}" for e in ref["acls"].get(p, [])}
        b = {f"{e['principal']} {e['rights']}" for e in cur.get(p, [])}
        if a != b:
            diff[p] = {"removed": sorted(a - b), "added": sorted(b - a)}
    return {"snapshot_at": ref["at"], "changed": bool(diff), "diff": diff}


def plan(data_dir: Path = DATA, account_map_path: Path = readiness.ACCOUNT_MAP_PATH, out_dir: Path | None = None) -> dict:
    amap = readiness.load_account_map(account_map_path)
    app = amap.get("app_account") or whoami()
    acc = privacy.load_access()
    d = str(Path(data_dir).resolve())
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save = str(resolve("backups/acl") / f"data_acl_before_{stamp}.txt")
    lines = [
        "REM SafeDrive v3.4.4 - least-privilege plan for the raw-data folder. NOTHING HAS BEEN RUN.",
        "REM Read every line, confirm, then run them one by one in a terminal of the folder owner (or an administrator).",
        "REM 0. record the state before (fingerprint + restorable copy)",
        "python scripts\\data_acl.py snapshot",
        f'icacls "{d}" /save "{save}" /t /c',
        f"REM 1. FIRST give the app account its own explicit right, so the app keeps working ({app})",
        f'icacls "{d}" /grant "{app}:(OI)(CI)M"',
        "REM 2. stop inheriting from D:\\ (the inherited entries are copied as explicit ones)",
        f'icacls "{d}" /inheritance:d',
        "REM 3. remove the broad groups (every Windows account)",
        f'icacls "{d}" /remove:g "NT AUTHORITY\\Authenticated Users" "BUILTIN\\Users" "Everyone"',
    ]
    people = amap.get("accounts") or {}
    extra = [(ps, a) for ps, a in people.items() if a and readiness._norm(a) != readiness._norm(app)]
    if extra:
        lines.append("REM 4. separate Windows accounts of campaign people (least privilege)")
        for ps, a in extra:
            if ps in acc.get("coordinators", []):
                lines.append(f'icacls "{d}" /grant "{a}:(OI)(CI)M"   REM {ps}: coordinator (read/write/delete)')
            elif ps in acc.get("raw_access", []):
                lines.append(f'icacls "{d}" /grant "{a}:(OI)(CI)(RX,W)"   REM {ps}: annotator (read + write labels, no delete)')
    else:
        lines.append("REM 4. (no separate Windows accounts in data/campaign/windows_accounts.json: everybody uses the app account)")
    lines += [
        "REM 5. check the result and record the new state",
        "python scripts\\data_acl.py compare",
        "python scripts\\data_acl.py inspect",
        "python scripts\\data_acl.py snapshot",
        "REM 6. access test with REAL accounts: log in as a standard, non-authorised account -> python scripts\\data_acl.py probe "
        "(must say DENIED); as the app account -> probe (must say allowed)",
        "REM ROLLBACK (restores the saved ACL of data\\ and everything below):",
        f'REM   icacls "{Path(d).parent}" /restore "{save}" /c',
    ]
    out = Path(out_dir) if out_dir else resolve("experiments/acl")
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"acl_plan_{stamp}.cmd.txt"
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"file": str(f), "app_account": app, "commands": lines}


def probe(data_dir: Path = DATA, out_dir: Path = readiness.ACL_PROBE_DIR) -> dict:
    """Run as the account under test. Records what this account can actually do on the raw data."""
    sess = Path(data_dir) / "sessions"
    rec = {"at": _now(), "user": whoami(), "elevated": readiness.is_elevated(), "admin_member": admin_member(),
           "target": str(sess)}
    try:
        names = [p.name for p in sess.iterdir()]
        rec["can_list_sessions"] = True
    except OSError as e:
        names, rec["can_list_sessions"], rec["list_error"] = [], False, type(e).__name__
    raw = None
    for n in names:
        vs = sorted((sess / n / "video").glob("segment_*.mp4")) if (sess / n / "video").exists() else []
        if vs:
            raw = vs[0]
            break
    if raw is None:
        # cannot list -> try a known path from the manifest-free layout (no listing rights): try any file directly
        rec["can_read_raw"] = False if not rec["can_list_sessions"] else None
        if rec["can_list_sessions"]:
            rec["note"] = "no raw video file found to read (record a parked test session first)"
    else:
        try:
            with open(raw, "rb") as fh:
                fh.read(1)
            rec["can_read_raw"] = True
        except OSError as e:
            rec["can_read_raw"], rec["read_error"] = False, type(e).__name__
        rec["raw_file_tested"] = raw.relative_to(data_dir).as_posix()
    t = sess / f".acl_probe_{os.getpid()}.tmp"
    try:
        t.write_bytes(b"x")
        t.unlink()
        rec["can_write"] = True
    except OSError as e:
        rec["can_write"], rec["write_error"] = False, type(e).__name__
    rec["outcome"] = "denied" if not rec["can_list_sessions"] and not rec.get("can_read_raw") and not rec["can_write"] else "allowed"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in rec["user"])
    (Path(out_dir) / f"probe_{safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(rec, indent=1),
                                                                                              encoding="utf-8")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["inspect", "snapshot", "compare", "plan", "probe"])
    a = ap.parse_args()
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    fn = {"inspect": inspect, "snapshot": snapshot, "compare": compare, "plan": plan, "probe": probe}[a.action]
    r = fn()
    if a.action == "plan":
        print("\n".join(r["commands"]))
        print(f"\n(saved to {r['file']}; NOTHING was executed)")
    else:
        print(json.dumps(r, indent=1, ensure_ascii=False, default=str))
    if a.action == "inspect":
        print("\nRESULT:", "FAIL - " + str(len(r["evaluation"]["problems"])) + " problem(s)" if r["evaluation"]["problems"]
              else "no ACL problem found (still needs snapshot + probes for PASS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
