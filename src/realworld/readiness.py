"""v3.4.3 pre-campaign readiness: 15 evidence-based checks, human attestations, camera check records and the GO / NO-GO rule.

Statuses
  PASS          verified with evidence (a measurement, a hash, a tool-produced verification record, or - only where the item IS a
                human confirmation - a named coordinator's attestation with an evidence reference)
  FAIL          checked and not met
  BLOCKED       cannot be checked or done now: missing rights (administrator) or a missing resource (external drive)
  NOT VERIFIED  not confirmed by a person / no evidence yet

Rules that this module enforces
  * a configuration value set to true is never evidence on its own;
  * GO only when every mandatory item is PASS; any FAIL, BLOCKED or NOT VERIFIED gives NO-GO;
  * nothing here opens the camera (camera_check() runs only when a person presses the button / passes --camera-check),
    starts a recording, changes Windows permissions or writes to the frozen reference.

System probes (drives, BitLocker, ACL) use Windows tools and are injectable so that tests never depend on the machine.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from src.utils.config import resolve

PASS, FAIL, BLOCKED, NV = "PASS", "FAIL", "BLOCKED", "NOT VERIFIED"
STATUSES = (PASS, FAIL, BLOCKED, NV)
CAMPAIGN_DIR = resolve("data/campaign")
ATTEST_PATH = CAMPAIGN_DIR / "attestations.jsonl"
BACKUP_STATUS_PATH = CAMPAIGN_DIR / "backup_status.jsonl"
CAMERA_CHECK_PATH = CAMPAIGN_DIR / "camera_checks.jsonl"
READINESS_LATEST = CAMPAIGN_DIR / "readiness_latest.json"
CHANGELOG = resolve("docs/SAFEDRIVE_SOFTWARE_CHANGELOG.md")
MODELS = ("models/safedrive_v3", "models/safedrive_v3_stabilized")

# ---- storage model (measured; see docs/SAFEDRIVE_V343_STORAGE_PLAN.md) ----
VIDEO_MB_PER_H = 665.0        # full-session video, 10 fps, measured on the 30-min v3.3 run (330.6 MB / 0.499 h)
SHARED_MB_PER_H = 665.0       # blurred copy: same encoder settings, assumed equal (not measured on campaign footage)
TIMELINE_MB_PER_H = 3.0       # measured 2.74-2.85 MB/h
SYSLOG_MB_PER_H = 0.2         # 720 rows/h of ~8 numbers
ANNOT_MB_PER_H = 5.0          # labels + evidence JPEGs (~50 frames/h x ~80 kB), planning value
CLIP_MB = 4.1                 # one event clip (pre 5 s + post 5 s), measured 223 MB / 54 clips
CLIPS_PER_H = {"low (balanced, few alarms)": 9.0, "high (demo-like density)": 108.0}
RESERVE = 0.30                # 30 % safety margin on top of the estimate
MIN_FREE_GB_PER_SESSION = 3.0  # a 60-min session with clips + headroom; checked before every session

BROAD_PRINCIPALS = ("Everyone", "BUILTIN\\Users", "NT AUTHORITY\\Authenticated Users", "BUILTIN\\Guests", "Guests")
EXTERNAL_BUS = {"USB", "SD", "MMC", "1394"}
BITLOCKER_SHELL = {1: "on", 2: "off", 3: "encrypting", 4: "decrypting", 5: "suspended", 6: "on_locked"}

# ---- v3.4.4 evidence stores ----
ACCOUNT_MAP_PATH = CAMPAIGN_DIR / "windows_accounts.json"   # owner fills: app_account + pseudonym -> Windows account
ACL_SNAPSHOTS = CAMPAIGN_DIR / "acl_snapshots.jsonl"        # scripts/data_acl.py snapshot (measured ACL + fingerprint)
ACL_PROBE_DIR = resolve("acl_probes")                       # OUTSIDE data/: an account that is denied data/ can still write here
ENCRYPTION_PATH = CAMPAIGN_DIR / "encryption_status.jsonl"  # scripts/encryption_status.py run AS ADMINISTRATOR (raw output kept)
ENCRYPTION_VALID_DAYS = 30
SYSTEM_PRINCIPALS = ("NT AUTHORITY\\SYSTEM", "BUILTIN\\Administrators", "CREATOR OWNER")
ROLE_RIGHTS = {  # least privilege on data/ for each role (Windows level)
    "app": {"read", "write", "delete"},          # the account that runs the app (writes sessions, labels, metadata)
    "coordinator": {"read", "write", "delete"},  # retention / deletion requests
    "raw_access": {"read", "write"},             # watch raw video, write labels; no delete, no permission changes
}

# v3.4.4 structured human checklists: every point is answered separately; a single tick is never enough
CAMERA_SAFETY_ITEMS = {
    "camera_fixed": "The camera is fixed firmly (does not move when touched, does not fall when braking)",
    "no_view_obstruction": "Camera, mount and cable do not obstruct the driver's view (windscreen sweep area, mirrors)",
    "airbag_clear": "Nothing is mounted on or in front of an airbag cover / deployment zone (dashboard, A-pillar, steering wheel)",
    "cable_safe": "Cables are routed and fixed; they do not hang near pedals, gear lever, steering wheel or door",
    "no_laptop_while_moving": "Nobody needs to operate the laptop while moving: it is started and stopped only while parked, "
                              "the screen is out of the driver's sight",
    "supervisor_present": "A supervisor (not the driver) is present for every session and knows the stop procedure",
    "parked_stable_run": "Camera and app ran stably while PARKED (camera check OK, no DEGRADED state, session started and stopped)",
    "stop_procedure": "A stop / incident procedure exists: recording stops only after parking; on any problem the driver ignores "
                      "the prototype and drives normally",
}
LEGAL_ITEMS = {
    "purpose": "Purpose of recording stated in writing (research on early collision warning; no other use)",
    "consent": "Consent of the driver and information / consent of occupants obtained where required",
    "faces_plates": "Handling of faces and licence plates of third parties defined (raw video restricted, blurred copies reviewed)",
    "retention": "Retention period defined, with deletion at the end",
    "access": "Who may access raw video is defined (access.json) and enforced (folder permissions)",
    "sharing": "Sharing rules defined (no sharing, or reviewed blurred copies only, to named recipients)",
    "deletion_on_request": "Deletion on request (driver / occupant) is possible and the procedure is known",
    "local_rules": "Rules that apply where the vehicle is driven (dash-camera, personal data, traffic rules on devices) checked",
}
CHECKLIST_ATTESTS = {"camera_safety": CAMERA_SAFETY_ITEMS, "legal_checklist": LEGAL_ITEMS}
ANSWERS = {"camera_safety": ("confirmed", "not_confirmed"),
           "legal_checklist": ("confirmed", "not_applicable", "needs_legal_advice", "not_confirmed")}
NEVER_NOT_APPLICABLE = {"local_rules", "consent", "retention", "access"}

ATTEST_ITEMS = {
    "legal_review": "Local rules on in-vehicle / dash-camera recording and on processing images of people and plates were "
                    "checked; recording is permitted under the stated conditions.",
    "camera_mount": "The camera is fixed (does not move by hand), the view covers the road ahead, the laptop is out of the "
                    "driver's reach, power is secured.",
    "parked_dry_run": "A complete session was started and stopped while PARKED with the campaign setup.",
    "supervisor_briefed": "The supervisor was briefed on the protocol (watches the road, never relies on the prototype).",
}


def now() -> datetime:
    return datetime.now()


def item(key: str, title: str, status: str, evidence: dict, action: str | None = None, who: str | None = None,
         evidence_at: str | None = None) -> dict:
    """checked_at = when this check ran; evidence_at = when the evidence it relies on was produced (a record, a measurement)."""
    assert status in STATUSES, status
    return {"key": key, "title": title, "status": status, "evidence": evidence, "action": action, "who": who,
            "checked_at": now().isoformat(timespec="seconds"), "evidence_at": evidence_at}


def _read_jsonl(p: Path) -> list[dict]:
    p = Path(p)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _append_jsonl(p: Path, rec: dict) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


# =============================== system probes (Windows) ===============================
def _ps(cmd: str, timeout: int = 30) -> tuple[int, str]:
    if sys.platform != "win32":
        return 127, "not windows"
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd], capture_output=True, text=True,
                           timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, str(e)


def windows_edition() -> str:
    rc, out = _ps("(Get-CimInstance Win32_OperatingSystem).Caption")
    return out.strip() if rc == 0 else "unknown"


def list_drives() -> list[dict]:
    """Logical drives with their physical disk and bus type (readable without administrator rights)."""
    cmd = ("$p=@{}; Get-Partition | Where-Object DriveLetter | ForEach-Object { $p[[string]$_.DriveLetter]=$_.DiskNumber };"
           "$b=@{}; Get-Disk | ForEach-Object { $b[[int]$_.Number]=[string]$_.BusType };"
           "Get-CimInstance Win32_LogicalDisk | ForEach-Object { $l=$_.DeviceID.Substring(0,1); $n=$p[$l];"
           "[pscustomobject]@{letter=$l; drive_type=$_.DriveType; fs=$_.FileSystem; free=$_.FreeSpace; size=$_.Size;"
           "serial=$_.VolumeSerialNumber; disk=$n; bus=$(if($n -ne $null){$b[[int]$n]}else{$null})} } | ConvertTo-Json -Compress")
    rc, out = _ps(cmd)
    if rc != 0 or not out.strip():
        return []
    try:
        data = json.loads(out.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return []
    data = data if isinstance(data, list) else [data]
    for d in data:
        d["external"] = bool(d.get("drive_type") == 2 or str(d.get("bus") or "").upper() in EXTERNAL_BUS)
    # v3.4.4: a mounted VeraCrypt volume has no partition of its own; it lives on the drive that holds its container file,
    # so it inherits that drive's physical disk, bus and external flag (verified through the VeraCrypt driver)
    for d in data:
        vc = veracrypt_volume(d["letter"])
        if vc and vc.get("container_drive"):
            host = next((x for x in data if str(x.get("letter", "")).upper() == vc["container_drive"]), None)
            if host:
                d.update(disk=host.get("disk"), bus=host.get("bus"), external=host.get("external"), veracrypt_container=vc["container"],
                         host_letter=host.get("letter"), host_serial=host.get("serial"))
    return data


def bitlocker_status(letter: str, probe=None) -> dict:
    """Try manage-bde (needs administrator), then the Shell volume property (readable without rights on most systems)."""
    if probe is not None:
        return probe(letter)
    rc, out = _ps(f"manage-bde -status {letter}:")
    if rc == 0 and "Protection Status" in out:
        return {"method": "manage-bde", **parse_manage_bde(out), "admin_required": False, "raw": out[-600:]}
    admin_denied = "denied" in out.lower() or "administrative" in out.lower()
    rc2, out2 = _ps(f"(New-Object -ComObject Shell.Application).NameSpace('{letter}:\\').Self.ExtendedProperty("
                    "'System.Volume.BitLockerProtection')")
    v = out2.strip()
    prot = BITLOCKER_SHELL.get(int(v), "unknown") if v.isdigit() else "unknown"
    return {"method": "shell_property", "protection": prot, "shell_value": v or None, "admin_required": admin_denied,
            "manage_bde": "access denied (administrator rights needed)" if admin_denied else out.strip()[-200:]}


def parse_manage_bde(out: str) -> dict:
    """Parse `manage-bde -status X:` (English output). 'on' ONLY when protection is on AND the volume is fully encrypted; a volume
    that is encrypted but whose protection is off (suspended, or Device encryption waiting for activation / a Microsoft-account
    key backup) is 'suspended' (= not protected). Unknown wording -> 'unknown' (never PASS by default)."""
    def g(k):
        m = re.search(rf"{k}:\s*(.+)", out)
        return m.group(1).strip() if m else None
    prot, conv, pct, meth, lock = (g("Protection Status"), g("Conversion Status"), g("Percentage Encrypted"),
                                   g("Encryption Method"), g("Lock Status"))
    try:
        pctv = float(pct.replace("%", "").replace(",", ".").strip()) if pct else None
    except ValueError:
        pctv = None
    full = pctv == 100.0 and bool(conv) and ("Fully Encrypted" in conv or "Used Space Only Encrypted" in conv)
    if prot and "Protection On" in prot and full:
        state = "on"
    elif conv and "Encryption in Progress" in conv:
        state = "encrypting"
    elif conv and "Decryption in Progress" in conv:
        state = "decrypting"
    elif prot and "Protection Off" in prot and pctv and pctv > 0:
        state = "suspended"
    elif prot and "Protection Off" in prot:
        state = "off"
    else:
        state = "unknown"
    return {"protection": state, "protection_status": prot, "conversion_status": conv, "percentage_encrypted": pctv,
            "encryption_method": meth, "lock_status": lock}


def is_elevated() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:  # noqa: BLE001
        return False


VC_IOCTL_GET_MOUNTED_VOLUMES = 0x222018  # CTL_CODE(FILE_DEVICE_UNKNOWN, 0x800 + 6, METHOD_BUFFERED, FILE_ANY_ACCESS)


def _dos_device(letter: str) -> str | None:
    if sys.platform != "win32":
        return None
    import ctypes
    buf = ctypes.create_unicode_buffer(1024)
    n = ctypes.windll.kernel32.QueryDosDeviceW(f"{letter.upper()}:", buf, 1024)
    return buf.value if n else None


def _vc_driver_list() -> dict[str, dict]:
    """Ask the VeraCrypt kernel driver (\\\\.\\VeraCrypt) which volumes are mounted: letter -> backing container / device."""
    if sys.platform != "win32":
        return {}
    import ctypes
    from ctypes import wintypes
    k = ctypes.windll.kernel32
    k.CreateFileW.restype = wintypes.HANDLE
    h = k.CreateFileW("\\\\.\\VeraCrypt", 0, 3, None, 3, 0, None)
    if h in (None, wintypes.HANDLE(-1).value):
        return {}

    # MOUNT_LIST_STRUCT grew across VeraCrypt versions (1.26: 17,424 bytes); only its stable head is parsed:
    # ULONG ulMountedDrives (bit i = drive letter A+i), then wchar wszVolume[26][260] (backing container / device path)
    buf = ctypes.create_string_buffer(65536)
    got = wintypes.DWORD(0)
    try:
        ok = k.DeviceIoControl(wintypes.HANDLE(h), VC_IOCTL_GET_MOUNTED_VOLUMES, buf, len(buf), buf, len(buf),
                               ctypes.byref(got), None)
    finally:
        k.CloseHandle(wintypes.HANDLE(h))
    if not ok or got.value < 4 + 26 * 260 * 2:
        return {}
    raw = buf.raw
    mask = int.from_bytes(raw[:4], "little")
    out = {}
    for i in range(26):
        if mask & (1 << i):
            off = 4 + i * 520
            vol = raw[off:off + 520].decode("utf-16-le", errors="ignore").split("\x00", 1)[0]
            out[chr(65 + i)] = {"container": vol[4:] if vol.startswith("\\??\\") else vol, "size": None, "ea": None}
    return out


def veracrypt_volume(letter: str, probe=None) -> dict | None:
    """A drive letter that is a MOUNTED VeraCrypt volume, verified with system facts: the kernel device name created by the
    VeraCrypt driver and the driver's own mount list (backing container). None otherwise."""
    if probe is not None:
        return probe(letter)
    dev = _dos_device(letter) or ""
    if "VeraCryptVolume" not in dev:
        return None
    info = _vc_driver_list().get(letter.upper())
    if not info:
        return None
    cont = info["container"]
    cl = cont[0].upper() if len(cont) > 1 and cont[1] == ":" else None
    return {"letter": letter.upper(), "device": dev, "container": cont, "container_drive": cl, "size": info["size"],
            "encryption_algorithm_id": info["ea"]}


def encryption_records(path: Path = ENCRYPTION_PATH) -> list[dict]:
    return _read_jsonl(path)


def encryption_evidence(letter: str, serial: str | None = None, probe=None, path: Path = ENCRYPTION_PATH,
                        at: datetime | None = None) -> dict:
    """Encryption state of a volume, from (1) an injected probe (tests), (2) a live manage-bde read (works only when elevated),
    or (3) the latest record written by scripts/encryption_status.py run AS ADMINISTRATOR for the same volume (same serial when
    known) within ENCRYPTION_VALID_DAYS. Anything else stays 'unknown'."""
    if probe is not None:
        return {**probe(letter), "source": "probe"}
    vc = veracrypt_volume(letter)
    if vc:
        return {"protection": "on", "method": "VeraCrypt driver", "source": "live VeraCrypt kernel driver", **vc}
    live = bitlocker_status(letter)
    if live.get("method") == "manage-bde" and live.get("protection") != "unknown":
        return {**live, "source": "live manage-bde"}
    at = at or now()
    recs = [r for r in encryption_records(path) if str(r.get("letter", "")).upper() == letter.upper() and r.get("elevated")
            and r.get("method") in ("manage-bde", "Get-BitLockerVolume") and (serial is None or r.get("serial") in (None, serial))]
    if recs:
        r = recs[-1]
        days = (at - datetime.fromisoformat(r["at"])).total_seconds() / 86400
        if days <= ENCRYPTION_VALID_DAYS:
            return {**{k: r.get(k) for k in ("protection", "protection_status", "conversion_status", "percentage_encrypted",
                                              "encryption_method", "serial", "at")},
                    "method": r["method"], "source": "recorded administrator check", "record_age_days": round(days, 1)}
        live["note"] = f"latest administrator record is {days:.0f} days old (> {ENCRYPTION_VALID_DAYS}); run it again"
    if live.get("method") == "shell_property":
        # the Shell property is an indication only (no percentage, no protector check): never accepted as verification
        live = {**live, "shell_indication": live.get("protection"), "protection": "unknown"}
    return {**live, "source": "live (not elevated)"}


RIGHT_TOKENS = {"F": {"read", "write", "delete", "change_permissions"}, "GA": {"read", "write", "delete", "change_permissions"},
                "M": {"read", "write", "delete"}, "RX": {"read"}, "R": {"read"}, "GR": {"read"}, "GE": {"read"}, "RD": {"read"},
                "RA": {"read"}, "REA": {"read"}, "X": {"read"}, "W": {"write"}, "GW": {"write"}, "WD": {"write"},
                "AD": {"write"}, "WA": {"write"}, "WEA": {"write"}, "D": {"delete"}, "DE": {"delete"}, "DC": {"delete"},
                "WDAC": {"change_permissions"}, "WO": {"change_permissions"}}
ACE_FLAGS = {"OI", "CI", "IO", "NP", "I"}


def rights_of(rights: str) -> dict:
    """icacls rights string -> {'rights': set, 'deny': bool, 'inherited': bool, 'inherit_only': bool}."""
    toks = [x.strip().upper() for t in re.findall(r"\(([^)]*)\)", rights) for x in t.split(",")]
    out: set[str] = set()
    for x in toks:
        if x in ACE_FLAGS or x == "DENY":
            continue
        out |= RIGHT_TOKENS.get(x, set())
    return {"rights": out, "deny": "DENY" in toks, "inherited": "I" in toks, "inherit_only": "IO" in toks}


def load_account_map(path: Path = ACCOUNT_MAP_PATH) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}
    except json.JSONDecodeError:
        return {}


def _norm(p: str) -> str:
    return (p or "").strip().lower()


def evaluate_acl(acls: dict[str, dict], account_map: dict, raw_access: list[str], coordinators: list[str]) -> dict:
    """Least-privilege evaluation of measured ACLs (path -> folder_acl()). Problems are concrete sentences."""
    app = account_map.get("app_account")
    people = account_map.get("accounts", {}) or {}
    roles: dict[str, str] = {}
    if app:
        roles[_norm(app)] = "app"
    for pseud, acct in people.items():
        if not acct:
            continue
        role = "coordinator" if pseud in coordinators else ("raw_access" if pseud in raw_access else "unlisted")
        if roles.get(_norm(acct)) != "app":
            roles[_norm(acct)] = role
    problems, rows = [], []
    broad = {_norm(b) for b in BROAD_PRINCIPALS}
    system = {_norm(s) for s in SYSTEM_PRINCIPALS}
    app_rights: set[str] = set()
    for path, acl in acls.items():
        for e in acl.get("entries", []):
            r = rights_of(e["rights"])
            p = e["principal"]
            if r["deny"] or not r["rights"]:
                continue
            role = "broad" if _norm(p) in broad else ("system" if _norm(p) in system else roles.get(_norm(p), "not_authorised"))
            rows.append({"path": path, "principal": p, "rights": sorted(r["rights"]), "inherited": r["inherited"], "role": role})
            src = " (inherited from the parent folder / drive root)" if r["inherited"] else ""
            if role == "broad":
                problems.append(f"{path}: {p} has {'/'.join(sorted(r['rights']))}{src}: every Windows account can reach the raw video")
            elif role == "not_authorised" and not app:
                rows[-1]["role"] = "unmapped"  # no account map yet: cannot judge named accounts (item stays NOT VERIFIED)
            elif role in ("not_authorised", "unlisted"):
                problems.append(f"{path}: {p} is not an authorised campaign account ({role}) but has {'/'.join(sorted(r['rights']))}")
            elif role in ROLE_RIGHTS:
                extra = r["rights"] - ROLE_RIGHTS[role]
                if extra:
                    problems.append(f"{path}: {p} ({role}) has more than needed: {'/'.join(sorted(extra))}")
                if role == "app":
                    app_rights |= r["rights"]
    if app and not {"read", "write"} <= app_rights:
        problems.append(f"the app account {app} would lose read / write on data/ (the app would stop working)")
    return {"problems": list(dict.fromkeys(problems)), "rows": rows, "roles": roles, "app_account": app}


def acl_fingerprint(acls: dict[str, dict]) -> str:
    import hashlib
    canon = sorted((Path(p).name or p, e["principal"].lower(), e["rights"]) for p, a in acls.items() for e in a.get("entries", []))
    return hashlib.sha256(json.dumps(canon).encode()).hexdigest()


def sessions_location(data_dir: Path) -> Path:
    """Where session data REALLY lives: data/sessions may be a junction into an encrypted volume (v3.4.4: V:/sessions)."""
    p = Path(data_dir) / "sessions"
    try:
        return p.resolve(strict=True) if p.exists() else Path(data_dir)
    except OSError:
        return Path(data_dir)


def acl_targets(data_dir: Path) -> list[Path]:
    d = Path(data_dir)
    sess = sessions_location(d) if (d / "sessions").exists() else d / "sessions"
    return [p for p in (d, sess, d / "campaign") if p.exists()] or [d]


def measure_acls(data_dir: Path, probe=None) -> dict[str, dict]:
    return {str(p): folder_acl(p, probe) for p in acl_targets(data_dir)}


def acl_probes(probe_dir: Path = ACL_PROBE_DIR) -> list[dict]:
    out = []
    for f in sorted(Path(probe_dir).glob("probe_*.json")) if Path(probe_dir).exists() else []:
        try:
            out.append(json.loads(f.read_text(encoding="utf-8-sig")))  # PowerShell 5.1 writes a BOM
        except json.JSONDecodeError:
            continue
    return out


def folder_acl(path: Path, probe=None) -> dict:
    if probe is not None:
        return probe(path)
    try:
        r = subprocess.run(["icacls", str(path)], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "error": str(e)}
    lines = [ln.strip() for ln in r.stdout.splitlines() if ":" in ln and "Successfully" not in ln]
    entries = []
    for ln in lines:
        ln = ln.replace(str(path), "").strip()
        m = re.match(r"(.+?):(\(.+\))$", ln)
        if m:
            entries.append({"principal": m.group(1).strip(), "rights": m.group(2)})
    broad = sorted({e["principal"] for e in entries if any(e["principal"].lower() == b.lower() for b in BROAD_PRINCIPALS)})
    return {"ok": r.returncode == 0, "entries": entries, "broad_principals": broad}


# =============================== attestations (human confirmations) ===============================
def attest(item_key: str, by: str, statement: str, evidence_ref: str, access_path: Path | None = None,
           path: Path = ATTEST_PATH) -> dict:
    """Record a HUMAN confirmation. Only a listed coordinator may attest, and a statement plus an evidence reference (a document,
    a note in the trip file, a photo id kept locally) are required. The software never attests by itself."""
    from src.realworld import privacy
    if item_key not in ATTEST_ITEMS:
        raise ValueError(f"unknown attestation item {item_key}")
    if item_key in CHECKLIST_ATTESTS:
        raise ValueError("use attest_checklist() for structured checklists")
    if not privacy.is_coordinator(by, access_path or privacy.ACCESS_PATH):
        raise PermissionError("only a coordinator listed in data/campaign/access.json can record an attestation")
    if not statement.strip() or not evidence_ref.strip():
        raise ValueError("a statement and an evidence reference are required")
    rec = {"item": item_key, "by": by, "at": now().isoformat(timespec="seconds"), "statement": statement.strip(),
           "evidence_ref": evidence_ref.strip(), "withdrawn": False}
    _append_jsonl(path, rec)
    return rec


def withdraw(item_key: str, by: str, reason: str, access_path: Path | None = None, path: Path = ATTEST_PATH) -> dict:
    from src.realworld import privacy
    if not privacy.is_coordinator(by, access_path or privacy.ACCESS_PATH):
        raise PermissionError("only a coordinator can withdraw an attestation")
    rec = {"item": item_key, "by": by, "at": now().isoformat(timespec="seconds"), "statement": reason, "evidence_ref": "",
           "withdrawn": True}
    _append_jsonl(path, rec)
    return rec


def latest_attestation(item_key: str, path: Path = ATTEST_PATH, access_path: Path | None = None) -> dict | None:
    """Latest record for the item; None if absent, withdrawn, or made by someone who is no longer a coordinator."""
    from src.realworld import privacy
    recs = [r for r in _read_jsonl(path) if r.get("item") == item_key]
    if not recs or recs[-1].get("withdrawn"):
        return None
    r = recs[-1]
    if not privacy.is_coordinator(r.get("by", ""), access_path or privacy.ACCESS_PATH):
        return None
    return r


def attest_checklist(kind: str, by: str, answers: dict, evidence_ref: str, access_path: Path | None = None,
                     path: Path = ATTEST_PATH) -> dict:
    """v3.4.4 structured HUMAN confirmation (camera_safety / legal_checklist). Every point needs its own answer:
      confirmed          + basis (what was checked / which document)
      not_applicable     + reason (legal only; never for local_rules / consent / retention / access)
      needs_legal_advice (legal only): stays open -> the item remains NOT VERIFIED
      not_confirmed      stays open
    Only a coordinator may record it; the software never fills it in."""
    from src.realworld import privacy
    if kind not in CHECKLIST_ATTESTS:
        raise ValueError(f"unknown checklist {kind}")
    if not privacy.is_coordinator(by, access_path or privacy.ACCESS_PATH):
        raise PermissionError("only a coordinator listed in data/campaign/access.json can record this checklist")
    if not evidence_ref.strip():
        raise ValueError("an evidence reference is required (document, photo id kept locally, note)")
    items = CHECKLIST_ATTESTS[kind]
    missing = [k for k in items if k not in answers]
    if missing:
        raise ValueError(f"every point needs an answer; missing: {missing}")
    clean = {}
    for k in items:
        a = answers[k] if isinstance(answers[k], dict) else {"status": answers[k], "basis": ""}
        st, basis = a.get("status"), str(a.get("basis", "")).strip()
        if st not in ANSWERS[kind]:
            raise ValueError(f"{k}: answer must be one of {ANSWERS[kind]}")
        if st == "not_applicable" and k in NEVER_NOT_APPLICABLE:
            raise ValueError(f"{k} cannot be 'not applicable'")
        if st in ("confirmed", "not_applicable") and len(basis) < 5:
            raise ValueError(f"{k}: '{st}' needs a basis / reason (what was checked, which document)")
        clean[k] = {"status": st, "basis": basis}
    open_ = [k for k, v in clean.items() if v["status"] not in ("confirmed", "not_applicable")]
    rec = {"item": kind, "by": by, "at": now().isoformat(timespec="seconds"), "answers": clean, "evidence_ref": evidence_ref.strip(),
           "complete": not open_, "open": open_,
           "needs_legal_advice": [k for k, v in clean.items() if v["status"] == "needs_legal_advice"], "withdrawn": False}
    _append_jsonl(path, rec)
    return rec


def checklist_attestation(kind: str, path: Path = ATTEST_PATH, access_path: Path | None = None) -> dict | None:
    """Latest structured record (None if absent, withdrawn, malformed or by a non-coordinator)."""
    r = latest_attestation(kind, path, access_path)
    if r is None or set(r.get("answers", {})) != set(CHECKLIST_ATTESTS[kind]):
        return None
    return r


# =============================== camera check (human-initiated only) ===============================
def camera_check(index: int, seconds: float = 3.0, operator: str = "", path: Path = CAMERA_CHECK_PATH, source=None) -> dict:
    """Open the camera for a few seconds (ONLY when a person asks for it), measure frame rate, resolution, brightness and
    sharpness, release it and log the numbers. No image is stored."""
    import time

    import cv2
    import numpy as np
    cap = source if source is not None else cv2.VideoCapture(int(index), cv2.CAP_DSHOW if sys.platform == "win32" else 0)
    frames, bright, sharp, size = 0, [], [], None
    t0 = time.time()
    try:
        if not cap.isOpened():
            rec = {"at": now().isoformat(timespec="seconds"), "index": index, "operator": operator, "opened": False, "ok": False,
                   "problems": ["camera could not be opened"]}
            _append_jsonl(path, rec)
            return rec
        while time.time() - t0 < seconds:
            ok, fr = cap.read()
            if not ok or fr is None:
                continue
            frames += 1
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            size = (int(fr.shape[1]), int(fr.shape[0]))
            bright.append(float(g.mean()))
            sharp.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
    finally:
        cap.release()
    dt = max(time.time() - t0, 1e-6)
    fps = frames / dt
    problems = []
    if frames == 0:
        problems.append("no frame received")
    else:
        if fps < 8:
            problems.append(f"frame rate {fps:.1f} < 8 fps")
        if size and (size[0] < 640 or size[1] < 360):
            problems.append(f"resolution {size} below 640x360")
        if float(np.median(bright)) < 25:
            problems.append("image very dark (lens covered or night without light)")
        if float(np.median(bright)) > 235:
            problems.append("image saturated")
        if float(np.median(sharp)) < 15:
            problems.append("image blurred / out of focus")
    rec = {"at": now().isoformat(timespec="seconds"), "index": index, "operator": operator, "opened": True, "frames": frames,
           "fps": round(fps, 2), "resolution": size, "brightness_median": round(float(np.median(bright)), 1) if bright else None,
           "sharpness_median": round(float(np.median(sharp)), 1) if sharp else None, "problems": problems, "ok": not problems}
    _append_jsonl(path, rec)
    return rec


def latest_camera_check(path: Path = CAMERA_CHECK_PATH) -> dict | None:
    r = _read_jsonl(path)
    return r[-1] if r else None


# =============================== storage ===============================
def storage_need(hours: float = 10.0, clip_scenario: str = "high (demo-like density)", shared_copies: bool = True) -> dict:
    v = VIDEO_MB_PER_H * hours
    sh = SHARED_MB_PER_H * hours if shared_copies else 0.0
    clips = CLIPS_PER_H[clip_scenario] * CLIP_MB * hours
    small = (TIMELINE_MB_PER_H + SYSLOG_MB_PER_H + ANNOT_MB_PER_H) * hours
    local = v + sh + clips + small
    return {"hours": hours, "video_mb": round(v), "shared_mb": round(sh), "clips_mb": round(clips), "small_mb": round(small),
            "local_mb": round(local), "local_with_reserve_gb": round(local * (1 + RESERVE) / 1024, 1),
            "backup_with_reserve_gb": round(local * (1 + RESERVE) / 1024, 1), "clip_scenario": clip_scenario, "reserve": RESERVE}


# =============================== backup records ===============================
def backup_records(path: Path = BACKUP_STATUS_PATH) -> list[dict]:
    return _read_jsonl(path)


def _latest(recs: list[dict], action: str) -> dict | None:
    xs = [r for r in recs if r.get("action") == action]
    return xs[-1] if xs else None


def campaign_sessions(root: Path) -> list[str]:
    root = Path(root)
    return sorted(p.name for p in root.iterdir() if (p / "trip_meta.json").exists()) if root.exists() else []


def newest_mtime(root: Path, sessions: list[str]) -> float:
    m = 0.0
    for s in sessions:
        for f in (Path(root) / s).rglob("*"):
            if f.is_file():
                m = max(m, f.stat().st_mtime)
    return m


# =============================== the 15 checks ===============================
def _changelog_has(h: str, path: Path = CHANGELOG) -> bool:
    return Path(path).exists() and h[:16] in Path(path).read_text(encoding="utf-8")


def check_all(sessions_root: Path | None = None, access_path: Path | None = None, attest_path: Path = ATTEST_PATH,
              backup_path: Path = BACKUP_STATUS_PATH, camera_path: Path = CAMERA_CHECK_PATH, training_report: dict | None = None,
              drives: list[dict] | None = None, bitlocker_probe=None, acl_probe=None, data_dir: Path | None = None,
              frozen_ref: dict | None = None, sharing_policy_path: Path | None = None, account_map_path: Path = ACCOUNT_MAP_PATH,
              acl_snapshot_path: Path = ACL_SNAPSHOTS, acl_probe_dir: Path = ACL_PROBE_DIR,
              encryption_path: Path = ENCRYPTION_PATH) -> list[dict]:
    from src.realworld import privacy, provenance, training
    from src.realworld.trip_meta import SESSIONS_DIR
    root = Path(sessions_root or SESSIONS_DIR)
    data_dir = Path(data_dir or resolve("data"))
    access_path = access_path or privacy.ACCESS_PATH
    ref = frozen_ref if frozen_ref is not None else provenance.frozen()
    out: list[dict] = []

    # 1-4 hashes / provenance
    cur = {Path(m).name: provenance.compute(m) for m in MODELS if resolve(m).exists()}
    for key, title, field in (("1_model_hash", "Model hash = frozen reference", "model_sha256"),
                              ("2_feature_hash", "Feature-configuration hash = frozen reference", "feature_config_sha256"),
                              ("3_threshold_hash", "Threshold / alert / calibration hash = frozen reference", "threshold_config_sha256")):
        ev = {m: {"current": c[field][:16], "frozen": ref.get(m, {}).get(field, "")[:16], "same": c[field] == ref.get(m, {}).get(field)}
              for m, c in cur.items()}
        ok = bool(ev) and bool(ref) and all(v["same"] for v in ev.values())
        out.append(item(key, title, PASS if ok else FAIL, ev, None if ok else
                        "A frozen hash differs: the frozen system was changed. Do not record campaign data; restore the frozen files."))
    sw = {m: c["software_sha256"] for m, c in cur.items()}
    h = next(iter(sw.values()), "")
    same = bool(ref) and all(sw[m] == ref.get(m, {}).get("software_sha256") for m in sw)
    logged = _changelog_has(h)
    out.append(item("4_software_provenance", "Software hash equals the frozen one or is recorded in the changelog",
                    PASS if (h and (same or logged)) else FAIL, {"current": h[:16], "same_as_frozen": same, "in_changelog": logged},
                    None if (same or logged) else "Add the current software_sha256 to docs/SAFEDRIVE_SOFTWARE_CHANGELOG.md with the change "
                    "description (or revert the change).", "developer"))

    # 5 access lists
    acc = privacy.load_access(access_path)
    raw, coord = acc.get("raw_access", []), acc.get("coordinators", [])
    bad = [n for n in raw + coord if not n or any(ch.isspace() for ch in n)]
    ok = bool(raw) and bool(coord) and not bad
    out.append(item("5_access_lists", "raw_access and coordinators filled with pseudonyms", PASS if ok else (FAIL if bad else BLOCKED),
                    {"raw_access": raw, "coordinators": coord, "invalid_names": bad}, None if ok else
                    "The owner fills data/campaign/access.json with pseudonyms (no spaces, no real names).", "owner"))

    # 6 encryption of the data drive (BitLocker / Device encryption), read with system tools only
    store = sessions_location(data_dir)  # the drive that really holds the session video (junction followed)
    letter = store.resolve().drive.rstrip(":") or "D"
    drv = drives if drives is not None else list_drives()
    data_serial = next((d.get("serial") for d in drv if str(d.get("letter", "")).upper() == letter.upper()), None)
    bl = encryption_evidence(letter, data_serial, bitlocker_probe, encryption_path)
    prot = bl.get("protection")
    if prot in ("on", "on_locked"):
        st6, act = PASS, None
    elif prot in ("off", "decrypting", "suspended"):
        st6, act = FAIL, (f"{letter}: is not protected ({prot}). As administrator: Settings > Privacy & security > Device encryption "
                          f"(Windows Home) or BitLocker (Pro), then python scripts/encryption_status.py --drive {letter} as administrator.")
    elif prot == "encrypting":
        st6, act = NV, "Encryption is in progress; re-run scripts/encryption_status.py as administrator when it has finished."
    elif bl.get("shell_indication") in ("on", "on_locked"):
        st6, act = NV, "The Shell property reports 'on' but this is not a full verification; confirm as administrator."
    else:
        st6, act = BLOCKED, (f"Reading the encryption state needs administrator rights: run python scripts/encryption_status.py "
                             f"--drive {letter} --save in an elevated terminal (see docs/SAFEDRIVE_V344_SECURITY_GUIDE.md §3).")
    ev6 = {k: v for k, v in bl.items() if k != "raw"}
    if bitlocker_probe is None:
        ev6["edition"] = windows_edition()
    out.append(item("6_bitlocker_data_drive", f"Session-data drive {letter}: encrypted (BitLocker on + 100 %, or a mounted VeraCrypt volume)", st6, ev6, act,
                    "Windows administrator", bl.get("at")))

    # 7 folder permissions: measured ACL + least-privilege evaluation + access probes from real Windows accounts
    acls = measure_acls(data_dir, acl_probe)
    acc_map = load_account_map(account_map_path)
    ev_acl = evaluate_acl(acls, acc_map, raw, coord)
    fp = acl_fingerprint(acls)
    snaps = [s for s in _read_jsonl(acl_snapshot_path) if s.get("fingerprint") == fp]
    since = snaps[0]["at"] if snaps else None
    probes = [p for p in acl_probes(acl_probe_dir) if since and p.get("at", "") >= since]
    app_n = _norm(acc_map.get("app_account", ""))
    authorised = {app_n} | {_norm(a) for a in (acc_map.get("accounts") or {}).values() if a}
    denied_std = [p for p in probes if _norm(p.get("user")) not in authorised and not p.get("elevated")
                  and not p.get("admin_member") and p.get("outcome") == "denied"]
    leaked = [p for p in probes if _norm(p.get("user")) not in authorised and p.get("outcome") != "denied"]
    app_ok = [p for p in probes if _norm(p.get("user")) == app_n and p.get("can_read_raw") and p.get("can_write")]
    people_missing = [ps for ps, a in (acc_map.get("accounts") or {}).items() if a and _norm(a) != app_n
                      and not any(_norm(p.get("user")) == _norm(a) and p.get("can_read_raw") for p in probes)]
    ev7 = {"paths": list(acls), "problems": ev_acl["problems"][:10], "account_map": acc_map or None,
           "acl_fingerprint": fp[:16], "acl_measured_since": since, "probes_since_change": len(probes),
           "denied_standard_account_probe": [p.get("user") for p in denied_std],
           "unauthorised_access_probe": [p.get("user") for p in leaked],
           "app_account_probe_ok": bool(app_ok), "person_accounts_not_probed": people_missing,
           "entries": [f"{Path(r['path']).name}: {r['principal']} {'/'.join(r['rights'])}{' (I)' if r['inherited'] else ''}"
                       for r in ev_acl["rows"]][:15]}
    if not all(a.get("ok") for a in acls.values()):
        st7, act7 = BLOCKED, "icacls could not read the folder permissions; run the audit as the folder owner."
    elif ev_acl["problems"] or leaked:
        st7, act7 = FAIL, ("Restrict data/ (docs/SAFEDRIVE_V344_SECURITY_GUIDE.md §2: python scripts/data_acl.py plan, run the printed "
                           "icacls commands after confirming, then snapshot + probe).")
    elif not acc_map.get("app_account"):
        st7, act7 = NV, "The owner fills data/campaign/windows_accounts.json (app_account + Windows account of each pseudonym)."
    elif not since:
        st7, act7 = NV, "Record the current permissions: python scripts/data_acl.py snapshot"
    elif not (denied_std and app_ok and not people_missing):
        st7, act7 = NV, ("Access test with real Windows accounts is missing: log in as a STANDARD account that is not authorised and "
                         "run python scripts/data_acl.py probe (must be denied), and run it as the app account"
                         + (f" and as {people_missing}" if people_missing else "") + ".")
    else:
        st7, act7 = PASS, None
    out.append(item("7_folder_permissions", "data/ reachable only by authorised accounts (least privilege, verified with real accounts)",
                    st7, ev7, act7, "Windows administrator / owner",
                    max([p.get("at", "") for p in probes], default=since)))

    # 8-10 backup
    data_disk = next((d.get("disk") for d in drv if d.get("letter", "").upper() == letter.upper()), None)
    ext = [d for d in drv if d.get("external") and d.get("disk") != data_disk]
    need = storage_need()
    recs = backup_records(backup_path)
    if not ext:
        st8, ev8 = BLOCKED, {"external_drives": [], "drives_seen": [f"{d.get('letter')}: bus={d.get('bus')} disk={d.get('disk')}" for d in drv]}
        act8 = "Connect an external USB drive (≥ %.0f GB free) encrypted with BitLocker To Go (or another vetted encryption)." % need["backup_with_reserve_gb"]
    else:
        rows = []
        for d in ext:
            b = encryption_evidence(d["letter"], d.get("serial"), bitlocker_probe, encryption_path)
            rows.append({"letter": d["letter"], "bus": d.get("bus"), "free_gb": round((d.get("free") or 0) / 1e9, 1),
                         "enough_space": (d.get("free") or 0) / 1e9 >= need["backup_with_reserve_gb"] * 1.074, "encryption": b.get("protection")})
        good = [r for r in rows if r["enough_space"] and r["encryption"] in ("on", "on_locked")]
        st8 = PASS if good else (FAIL if any(r["encryption"] in ("off",) or not r["enough_space"] for r in rows) else NV)
        ev8, act8 = {"external_drives": rows, "need_gb": need["backup_with_reserve_gb"]}, (None if good else
                    "Use an external drive with enough space and encryption on (BitLocker To Go needs Windows Pro to create; "
                    "see the backup operations guide).")
    out.append(item("8_external_backup_destination", "External, encrypted backup drive with enough space", st8, ev8, act8, "user"))

    sess = campaign_sessions(root)
    lb, lv, lr = _latest(recs, "backup"), _latest(recs, "verify"), _latest(recs, "restore_test")
    valid_b = lb is not None and lb.get("ok") and lb.get("external") and not lb.get("same_disk") and not lb.get("test_mode")
    if not valid_b:
        st9 = BLOCKED if not ext else NV
        ev9 = {"last_backup": lb, "last_verify": lv}
        act9 = "python scripts/campaign_backup.py backup --dest <drive>:\\  then  verify --dest <drive>:\\"
    else:
        covered = set(lb.get("sessions", [])) >= set(sess)
        fresh = newest_mtime(root, sess) <= datetime.fromisoformat(lb["at"]).timestamp() + 1
        verified = lv is not None and lv.get("ok") and lv["at"] >= lb["at"] and lv.get("dest_serial") == lb.get("dest_serial")
        if lv is not None and not lv.get("ok") and lv["at"] >= lb["at"]:
            st9 = FAIL
        else:
            st9 = PASS if (covered and fresh and verified) else NV
        ev9 = {"last_backup_at": lb["at"], "covers_all_sessions": covered, "no_newer_files": fresh, "verified_after_backup": verified,
               "verify": {k: lv.get(k) for k in ("at", "ok", "files", "missing", "hash_mismatch")} if lv else None}
        act9 = None if st9 == PASS else "Run backup again (incremental) and then verify."
    out.append(item("9_backup_verification", "Latest backup covers every session and was re-verified by SHA-256", st9, ev9, act9, "user"))
    if lr is not None and valid_b and lr.get("ok") and lr["at"] >= lb["at"]:
        st10 = PASS
    elif lr is not None and not lr.get("ok"):
        st10 = FAIL
    else:
        st10 = BLOCKED if not valid_b else NV
    out.append(item("10_restore_verification", "Test restore into a temporary folder matched every hash", st10,
                    {"last_restore_test": lr}, None if st10 == PASS else
                    "python scripts/campaign_backup.py restore-test --dest <drive>:\\", "user"))

    # 11 privacy review
    pol = privacy.sharing_policy(sharing_policy_path)
    approved, others = [], []
    for s in sess:
        d = root / s
        m = json.loads((d / "trip_meta.json").read_text(encoding="utf-8"))
        st_ = privacy.shared_status(d, access_path)
        if st_ == "no_shared_copy":
            continue
        rv = privacy.load_review(d)
        row = {"session": s, "data_kind": m.get("data_kind"), "status": st_, "reviewer": rv.get("reviewer"),
               "reviewed_at": rv.get("reviewed_at"), "recorded_parked": rv.get("recorded_parked")}
        parked_ok = m.get("data_kind") == "real_driving" or rv.get("recorded_parked") is True
        if st_ == "approved" and m.get("data_kind") in privacy.CAMPAIGN_CAMERA_KINDS and parked_ok:
            approved.append(row)
        else:
            others.append(row)
    if approved:
        st11, ev11, act11, at11 = PASS, {"approved_campaign_camera_copies": approved, "other_copies": others,
                                         "sharing_policy": pol}, None, approved[-1]["reviewed_at"]
    elif not pol.get("sharing_enabled", True) and privacy.is_coordinator(pol.get("decided_by", ""), access_path):
        st11, ev11, act11, at11 = PASS, {"mode": "sharing disabled by coordinator decision (export refused by software)",
                                         "policy": pol, "other_copies": others,
                                         "blur_quality": "not verified (no reviewed campaign-camera copy)"}, None, pol.get("decided_at")
    else:
        st11, at11 = NV, None
        ev11 = {"approved_campaign_camera_copies": [], "other_copies": others, "sharing_policy": pol,
                "note": "demo / Nexar video never counts: it is pre-anonymised and is not campaign-camera footage"}
        act11 = ("COORD-01: either review a blurred copy of a consented PARKED campaign-camera clip (page 11) or record the "
                 "no-sharing decision (page 14).")
    out.append(item("11_privacy_review", "Blurred copy of campaign-camera footage approved by a coordinator, or sharing disabled",
                    st11, ev11, act11, "COORD-01", at11))

    # 12 camera readiness: measured camera check + structured safety checklist (coordinator)
    cc = latest_camera_check(camera_path)
    saf = checklist_attestation("camera_safety", attest_path, access_path)
    if cc and not cc.get("ok"):
        st12 = FAIL
    elif cc and cc.get("ok") and saf and saf.get("complete"):
        st12 = PASS
    else:
        st12 = NV
    open12 = (saf or {}).get("open", list(CAMERA_SAFETY_ITEMS)) if not (saf and saf.get("complete")) else []
    out.append(item("12_camera_readiness", "Camera check measured OK and the 8-point camera / safety checklist confirmed", st12,
                    {"camera_check": cc, "safety_checklist_by": (saf or {}).get("by"), "safety_points_open": open12,
                     "evidence_ref": (saf or {}).get("evidence_ref")},
                    None if st12 == PASS else ("Camera check failed: " + "; ".join(cc.get("problems", [])) if st12 == FAIL else
                    "In the vehicle, parked: page 14 → 'Test camera'; COORD-01 answers every point of the camera / safety checklist."),
                    "operator + COORD-01", max([x for x in ((cc or {}).get("at"), (saf or {}).get("at")) if x], default=None)))

    # 13 legal: structured checklist; any 'needs legal advice' keeps it open
    lg = checklist_attestation("legal_checklist", attest_path, access_path)
    st13 = PASS if lg and lg.get("complete") else NV
    out.append(item("13_legal_review", "Legal checklist (8 points) answered by a coordinator; nothing left for legal advice", st13,
                    {"by": (lg or {}).get("by"), "evidence_ref": (lg or {}).get("evidence_ref"),
                     "open": (lg or {}).get("open", list(LEGAL_ITEMS)), "needs_legal_advice": (lg or {}).get("needs_legal_advice", []),
                     "note": "a human confirmation; the software cannot verify legality"},
                    None if st13 == PASS else ("Points needing legal advice: " + ", ".join(lg["needs_legal_advice"]) if lg and
                    lg.get("needs_legal_advice") else "COORD-01 answers the legal checklist on page 14 (docs/SAFEDRIVE_V344_CAMPAIGN_CHECKLIST.md §3)."),
                    "COORD-01 / owner", (lg or {}).get("at")))

    # 14 annotation training (submissions locked by hash; per-annotator results)
    tr = training_report if training_report is not None else training.report_if_available()
    if tr and tr.get("passed"):
        st14 = PASS
    elif tr and tr.get("complete") and not tr.get("passed"):
        st14 = FAIL
    else:
        st14 = NV
    out.append(item("14_annotation_training", "Both annotators completed the training pack independently and met the criteria", st14,
                    {k: tr.get(k) for k in ("pack", "annotators", "submitted", "complete", "passed", "criteria_results", "invalid")}
                    if tr else {"report": None},
                    None if st14 == PASS else "ANN-A and ANN-B label TRAINING_PACK_01 blind and submit (page 11, workspace "
                    "'Annotator training'); COORD-01 then opens the result.", "ANN-A, ANN-B, COORD-01", (tr or {}).get("at")))

    # 15 disk space
    try:
        free = shutil.disk_usage(store).free / 1e9
        need_gb = need["local_with_reserve_gb"] * 1.074
        out.append(item("15_disk_space", "Free space on the data drive ≥ 10 h estimate + 30 % reserve",
                        PASS if free >= need_gb else FAIL, {"free_gb": round(free, 1), "need_gb": round(need_gb, 1), "estimate": need}))
    except OSError as e:
        out.append(item("15_disk_space", "Free space on the data drive", BLOCKED, {"error": str(e)}))
    return out


def go_decision(items: list[dict]) -> dict:
    blocking = [{"key": i["key"], "status": i["status"], "action": i.get("action"), "who": i.get("who")} for i in items if i["status"] != PASS]
    return {"verdict": "GO" if items and not blocking else "NO-GO", "blocking": blocking,
            "counts": {s: sum(i["status"] == s for i in items) for s in STATUSES}}


def save_latest(items: list[dict], path: Path = READINESS_LATEST, run_dir: str | None = None) -> dict:
    rec = {"at": now().isoformat(timespec="seconds"), "run_dir": run_dir, "decision": go_decision(items), "items": items}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return rec


def load_latest(path: Path = READINESS_LATEST) -> dict | None:
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None


def age(rec_at: str) -> timedelta:
    return now() - datetime.fromisoformat(rec_at)
