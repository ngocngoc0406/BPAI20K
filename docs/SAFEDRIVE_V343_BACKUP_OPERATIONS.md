# SafeDrive AI v3.4.3: Backup Operations

**Tool:** `scripts/campaign_backup.py`. It uses SHA-256 throughout, never compresses and never uploads.

**Who:** the campaign operator, as the normal Windows user, plus a Windows administrator for §5 and §6.

**Current state (2026-11-22):**
- **BLOCKED.** No external drive is connected.
- No backup exists yet. The tool did not create a placeholder.

## 1. What the tool checks before copying anything (`check`)
| Check | Refused if |
|---|---|
| Drive connected and listed | The drive letter does not exist |
| Not inside the project | The destination is under `D:\SanphambpAI` |
| **Different physical disk** | It is on the same disk as the data. On this laptop C: and D: are the same NVMe disk, so both are refused |
| **External** | The bus type is not USB / SD, or the drive is not removable |
| **Free space** | Free space is below the size of the files still to copy, plus 10 % |
| **Encryption** | BitLocker protection is not *on*. `--allow-unencrypted` copies anyway, but records the backup as unencrypted; the audit then reports FAIL |
| Writable | A test file cannot be written and removed |

- `--allow-same-drive` and `--allow-internal` exist **only for automated tests**.
- Records made with them carry `test_mode` and never count for readiness.

## 2. Daily routine (after every driving day, parked, at home)
```powershell
cd D:\SanphambpAI
.venv\Scripts\python scripts\campaign_backup.py drives                 # find the external drive letter (e.g. E)
.venv\Scripts\python scripts\campaign_backup.py check  --dest E:\      # nothing is copied
.venv\Scripts\python scripts\campaign_backup.py backup --dest E:\      # incremental, every file hash-verified after copy
.venv\Scripts\python scripts\campaign_backup.py verify --dest E:\      # re-hash the whole backup
```
**First time and once a week, also run:**
```powershell
.venv\Scripts\python scripts\campaign_backup.py restore-test --dest E:\
```
This restores **everything** into a new temporary folder and compares every hash with the index. It also compares with the
live files and reports any source that changed since the backup. Afterwards it deletes the temporary folder.

Every action appends a line to `data/campaign/backup_status.jsonl`. That file is the evidence the audit and the checklist use
for items 9 and 10. **Afterwards, unplug the drive and store it away from the laptop.**

## 3. Errors and recovery
| Message | Meaning | What to do |
|---|---|---|
| `destination refused: ... same physical disk` | Target is C: or D: | Use the external drive |
| `... is not an external (USB / SD) drive` | Internal second disk, or a network drive | Use the external drive |
| `not enough free space` | The drive is too small | Use a larger drive. Nothing was copied |
| `destination encryption is 'off' / 'unknown'` | Not encrypted, or not readable | See §4 |
| `backup interrupted: ...` (exit code 3) | Disk full, or drive unplugged during the copy | The source data is unchanged and the index lists only verified files. Fix the cause and run `backup` again: verified files are skipped |
| `file(s) differed after copy` | Write error on the drive | Run `verify`; if it repeats, replace the drive |
| `verify`: `missing` / `hash_mismatch` | Bit rot, tampering, deleted files | Run `backup` again: files are re-copied from `data/` while the source still exists. If the source is gone, those files are lost from this copy |
| `restore-test`: `newer_in_source` | Annotations etc. changed after the backup | Run `backup` again |
| `refusing to overwrite existing file` (`restore`) | The target folder is not empty | Restore into an empty folder |

## 4. Encryption of the backup drive
- **BitLocker To Go** needs Windows Pro / Enterprise / Education to **create** it. This laptop runs **Windows 11 Home**, which
  can only unlock such a drive. Options, in order of preference:
  1. Encrypt the drive once on a Pro / Enterprise PC, with a strong password and the recovery key stored separately. On this
     laptop, unlock it with the password before each backup.
  2. Use another vetted full-volume encryption tool chosen by the owner. That is a human decision: the project does not install
     software. Then run with `--allow-unencrypted` only if the tool cannot report BitLocker status, and record the decision in
     the changelog. **The audit will still report item 8 as FAIL** because it cannot verify that encryption.
- The encryption state is read with `manage-bde`, which needs administrator rights. Without administrator rights it falls back
  to the Windows Shell property. If neither gives a clear answer, the state is "unknown" and never counts as encrypted.

## 5. Laptop data drive encryption (administrator)
1. Open PowerShell **as administrator** and run `manage-bde -status D:` and `manage-bde -status C:`.
2. On Windows Home, *Settings → Privacy & security → Device encryption* shows whether it is available and on. If it is not
   available, the owner must decide whether to upgrade or to use other encryption. Do not record unencrypted campaign video
   until this is settled.
3. Re-run `python scripts/precampaign_audit.py` **as administrator once**, so that item 6 can be read.

## 6. Folder permissions (administrator or folder owner)
**Measured on 2026-11-22:**
- `D:\SanphambpAI\data` inherits **Authenticated Users: Modify** and **Users: Read & execute**.
- Every account on this PC can therefore read the raw video.
- Item 7 is **FAIL**.

The owner decides which Windows account(s) run the campaign. Then:
```powershell
# run as the folder owner or as administrator; replace CAMPAIGN_ACCOUNT with the real Windows account name
icacls "D:\SanphambpAI\data" /inheritance:d
icacls "D:\SanphambpAI\data" /remove:g "Authenticated Users" "Users" "Everyone"
icacls "D:\SanphambpAI\data" /grant:r "CAMPAIGN_ACCOUNT:(OI)(CI)F" "SYSTEM:(OI)(CI)F" "Administrators:(OI)(CI)F"
icacls "D:\SanphambpAI\data"        # check: no Everyone / Users / Authenticated Users
```
- **Undo:** `icacls "D:\SanphambpAI\data" /reset /T`.
- The software does **not** run these commands, because changing Windows permissions is the owner's decision.
- Annotators who use the same Windows account still rely on the application-level raw-access list.

## 7. Restoring after a loss
1. Install the project on the replacement machine, then check the hashes in `configs/frozen_v3_hashes.json` with the audit.
2. Run `restore --dest E:\ --to D:\SanphambpAI\data\sessions_restored`.
3. Compare, then move the sessions into `data/sessions/` by hand. Restore never overwrites existing files.
4. Copy the backed-up `campaign/` folder to `data/campaign/` **only** if the local one is lost. It holds the partitions, logs
   and access lists.
