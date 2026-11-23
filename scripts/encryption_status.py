"""v3.4.4 read the REAL encryption state of a volume with Windows' own tools. It never changes encryption settings.

Run it in an ELEVATED terminal (right-click Terminal > Run as administrator):
    python scripts\\encryption_status.py --drive D --save
    python scripts\\encryption_status.py --drive E --save      (the external backup drive)

What it does
  * `manage-bde -status X:` and `Get-BitLockerVolume` (both need administrator rights) - the full output is kept;
  * parses Protection Status / Conversion Status / Percentage Encrypted / Encryption Method;
  * with --save AND elevation AND a parsable result, appends a record to data/campaign/encryption_status.jsonl
    (letter, volume serial, parsed fields, SHA-256 of the raw output, raw output). The readiness audit accepts a record for
    ENCRYPTION_VALID_DAYS days; 'on' requires Protection On AND 100 % encrypted.
  * without elevation it prints the non-admin indications (Shell property, BitLockerStatus registry value, sleep states) and
    says BLOCKED; nothing is saved, and nothing is reported as encrypted.

Windows 11 Home: BitLocker management (and creating BitLocker To Go on USB drives) is not available, but "Device encryption"
(BitLocker-based) may be: Settings > Privacy & security > Device encryption. Whether it exists on this PC depends on the hardware
(TPM 2.0, Modern Standby or HSTI compliance) and is shown by msinfo32 > "Device Encryption Support" (run as administrator).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.realworld import readiness  # noqa: E402


def indications(letter: str) -> dict:
    """Non-admin hints only (never verification)."""
    out = {}
    rc, v = readiness._ps(f"(New-Object -ComObject Shell.Application).NameSpace('{letter}:\\').Self.ExtendedProperty("
                          "'System.Volume.BitLockerProtection')")
    out["shell_property"] = v.strip() or None
    rc, v = readiness._ps("(Get-ItemProperty HKLM:\\SYSTEM\\CurrentControlSet\\Control\\BitLockerStatus -ErrorAction "
                          "SilentlyContinue).BootStatus")
    out["bitlockerstatus_bootstatus"] = v.strip() or None
    rc, v = readiness._ps("powercfg /a")
    out["modern_standby_available"] = ("Standby (S0 Low Power Idle)" in v.split("not available")[0]) if v else None
    out["edition"] = readiness.windows_edition()
    return out


def read_status(letter: str, elevated: bool | None = None) -> dict:
    elevated = readiness.is_elevated() if elevated is None else elevated
    rec = {"at": datetime.now().isoformat(timespec="seconds"), "letter": letter.upper(), "elevated": elevated}
    drv = next((d for d in readiness.list_drives() if str(d.get("letter", "")).upper() == letter.upper()), {})
    rec["serial"] = drv.get("serial")
    rec["bus"] = drv.get("bus")
    rc, out = readiness._ps(f"manage-bde -status {letter}:")
    if rc == 0 and "Protection Status" in out:
        rec.update(method="manage-bde", **readiness.parse_manage_bde(out))
        rec["raw"] = out.strip()
    else:
        rec["manage_bde_error"] = out.strip()[-300:]
        rc2, out2 = readiness._ps(f"Get-BitLockerVolume -MountPoint '{letter}:' | Select-Object MountPoint,VolumeStatus,"
                                  "ProtectionStatus,EncryptionPercentage,EncryptionMethod,LockStatus | ConvertTo-Json -Compress")
        try:
            j = json.loads(out2.strip().splitlines()[-1]) if rc2 == 0 and out2.strip() else None
        except (json.JSONDecodeError, IndexError):
            j = None
        if j:
            # numeric enums from the BitLocker module: ProtectionStatus 1 = On; VolumeStatus 1 = FullyEncrypted
            on = str(j.get("ProtectionStatus")) in ("1", "On") and float(j.get("EncryptionPercentage") or 0) == 100.0
            rec.update(method="Get-BitLockerVolume", protection="on" if on else ("off" if str(j.get("ProtectionStatus")) in
                       ("0", "Off") else "unknown"), percentage_encrypted=j.get("EncryptionPercentage"),
                       protection_status=str(j.get("ProtectionStatus")), conversion_status=str(j.get("VolumeStatus")),
                       encryption_method=str(j.get("EncryptionMethod")))
            rec["raw"] = out2.strip()
        else:
            rec.update(method=None, protection="unknown", get_bitlockervolume_error=out2.strip()[-300:])
    rec["raw_sha256"] = hashlib.sha256(rec.get("raw", "").encode()).hexdigest() if rec.get("raw") else None
    if not elevated or rec.get("method") is None:
        rec["indications_not_verification"] = indications(letter)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="D")
    ap.add_argument("--save", action="store_true", help="append to data/campaign/encryption_status.jsonl (only when elevated)")
    a = ap.parse_args()
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    letter = a.drive.rstrip(":\\").upper()
    rec = read_status(letter)
    print(json.dumps({k: v for k, v in rec.items() if k != "raw"}, indent=1, ensure_ascii=False))
    if rec.get("raw"):
        print("\n--- raw output ---\n" + rec["raw"])
    if not rec["elevated"] or rec.get("method") is None:
        print(f"\nBLOCKED: reading the encryption state of {letter}: needs an ELEVATED terminal (Run as administrator). Nothing saved.")
        return 2
    verdict = {"on": "ENCRYPTED and protection ON", "off": "NOT encrypted", "suspended": "encrypted but protection OFF (not protected)",
               "encrypting": "encryption in progress", "decrypting": "decryption in progress"}.get(rec["protection"], "UNKNOWN wording")
    print(f"\n{letter}: {verdict}")
    if a.save:
        readiness._append_jsonl(readiness.ENCRYPTION_PATH, rec)
        print(f"saved to {readiness.ENCRYPTION_PATH}")
    return 0 if rec["protection"] == "on" else 1


if __name__ == "__main__":
    raise SystemExit(main())
