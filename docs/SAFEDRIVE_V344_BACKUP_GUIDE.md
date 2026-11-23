# SafeDrive AI v3.4.4: Backup Guide

| | |
|---|---|
| For | the user (operator) and COORD-01 |
| Tool | `python scripts/campaign_backup.py <action>`; every action is logged to `data/campaign/backup_status.jsonl` |
| Covers | readiness items **8** (destination), **9** (backup verified), **10** (restore verified) |
| State 2026-11-22 | **no external drive connected** (C: and D: are the same NVMe disk 0); no backup exists; items 8–10 BLOCKED |

## 1. What you need
- An external USB drive with **≥ 23 GB free** for 10 h. 64 GB is the minimum; 128 GB is recommended. Sizes are in
  [SAFEDRIVE_V343_STORAGE_PLAN.md](SAFEDRIVE_V343_STORAGE_PLAN.md).
- **Encrypted with BitLocker To Go.** Windows Home cannot create it: encrypt the drive once on a Windows Pro PC. Then run
  `python scripts/encryption_status.py --drive <letter> --save` in an **administrator** terminal here, so that the tool can confirm
  the encryption ([SAFEDRIVE_V344_SECURITY_GUIDE.md](SAFEDRIVE_V344_SECURITY_GUIDE.md) §3).
- Never use C: or D: of this laptop (the same physical disk), a folder inside the project, or a cloud folder.

## 2. Commands, in order
| Step | Command | What it does / what must happen |
|---|---|---|
| 1 | `campaign_backup.py drives` | Lists drives with bus type, physical disk, free space and encryption. The USB drive must show `external: true` and a **different** `disk` |
| 2 | `campaign_backup.py check --dest E:\` | Checks the destination and **copies nothing**: connected, not in the project, different physical disk, external, enough space (+10 %), encrypted, writable |
| 3 | `campaign_backup.py backup --dest E:\` | Copies every session with trip metadata plus `data/campaign` to `E:\SafeDrive_backup\`. Re-hashes **every** copied file. Writes `backup_index.json` and **`MANIFEST.sha256`** (`sha256  path` lines, checkable with any tool). Incremental: unchanged verified files are skipped |
| 4 | `campaign_backup.py verify --dest E:\` | Re-hashes the backup. Checks that every indexed file is present, the **file count**, that `MANIFEST.sha256` equals the index, and lists unindexed files |
| 5 | `campaign_backup.py restore-test --dest E:\` | Restores **everything into a new temporary folder**, compares each hash with the index and with the live files, then deletes the folder. The original data is never touched |
| 6 | `campaign_backup.py report` | Writes `experiments/backup_reports/backup_report_<time>.md/.json` (last backup / verify / restore-test, refusals and failures) |

Repeat steps 3–4 after **every** recording day, and step 5 at least weekly. The pre-session checklist refuses a real-driving
session if earlier real sessions are not in a verified external backup.

To restore one session for use: `campaign_backup.py restore --dest E:\ --session <id> --to F:\restore_check`. The target must be
an **empty folder outside `data\`**. Restoring into `data\` is refused, so original data can never be overwritten.

## 3. Failure handling
| Situation | What the tool does | What you do |
|---|---|---|
| no external drive / same physical disk / internal drive | `check` and `backup` refuse **before copying** (exit 3, `ERROR` + `RECOVERY`); the refusal is logged | connect a real external drive |
| not enough space | refused before copying (free < needed × 1.1) | use a larger drive or free space on it |
| unencrypted or unknown encryption | refused (`--allow-unencrypted` records the backup as unencrypted, and the audit then reports FAIL) | encrypt the drive, then run the administrator encryption check |
| **drive removed or full during the copy** | stops; the index and manifest list **only files already verified**; logged as `ok: false` with the error | reconnect / free space and run `backup` again: verified files are skipped, the rest is copied |
| file differs right after the copy | not indexed; the backup is `ok: false` | run `verify`; if it repeats, the drive is failing: use another one |
| **corrupted file / hash mismatch** later | `verify` lists it under `hash_mismatch`; `restore-test` is not ok | run `backup` again (re-copies from `data\` while the source exists), then `verify` |
| damaged index or manifest | `verify` fails (`index unreadable` / `manifest_matches_index: false`) | run `backup` again |
| restore target exists or is inside `data\` | refused | restore into a new empty folder |

Synthetic tests for every row: `tests/test_v344_closure.py` and `tests/test_v343_readiness.py`. They use temporary folders only;
**no real backup was simulated as successful**.

## 4. What the audit counts
- **Item 8 PASS:** an external drive on another physical disk, with enough space and **encryption confirmed by a Windows tool**.
- **Item 9 PASS:** the latest backup is ok, external, not same-disk and not test-mode; it covers every session, no file is newer than
  the backup, and a `verify` ran afterwards on the same drive serial and was ok.
- **Item 10 PASS:** a `restore-test` after that backup was ok.
- Test-mode records (`--allow-same-drive`, `--allow-internal`) never count.
