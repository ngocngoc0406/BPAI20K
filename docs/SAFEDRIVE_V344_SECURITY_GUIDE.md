# SafeDrive AI v3.4.4: Security Guide (folder permissions and drive encryption)

| | |
|---|---|
| For | the owner, the Windows administrator, COORD-01 |
| Covers | readiness items **6** (data drive encrypted) and **7** (folder permissions) |
| Machine | Windows 11 **Home** Single Language, one local account `gigabyte\ngocn` (member of Administrators, works with UAC = not elevated) |

The software reads permissions and encryption state; it **never changes them**. Every change in this guide is done by a person,
after reading the command, in the order given.

## 1. Why item 7 FAILS today (measured 2026-11-22)
`icacls` shows this chain:

| Folder | Entry | Origin |
|---|---|---|
| `D:\` | `NT AUTHORITY\Authenticated Users:(M)` and `(OI)(CI)(IO)(M)` | explicit: the Windows default ACL of a non-system NTFS volume |
| `D:\` | `BUILTIN\Users:(RX)` and `(OI)(CI)(IO)(GR,GE)` | explicit: same default |
| `D:\SanphambpAI` | the same entries with `(I)` | inherited from `D:\` |
| `D:\SanphambpAI\data` (and `sessions`, `campaign`) | the same entries with `(I)` | inherited |

So **every Windows account** on this PC (any account that can log in = *Authenticated Users*) can **read, change and delete** raw
video. Nobody set this on purpose: it is inherited from the drive root. Owner of `data\` is `gigabyte\ngocn`.

Rights, as the tool reports them (`python scripts/data_acl.py inspect`):

| icacls | read | write | delete | change permissions |
|---|---|---|---|---|
| `F` (full) | ✔ | ✔ | ✔ | ✔ |
| `M` (modify) | ✔ | ✔ | ✔ | — |
| `RX`, `R`, `GR,GE` | ✔ | — | — | — |
| `W` | — | ✔ | — | — |
| `D` | — | — | ✔ | — |

## 2. Restricting `D:\SanphambpAI\data` (least privilege)

### 2.1 Roles
| Role | Who | Needed on `data\` |
|---|---|---|
| app account | the Windows account that runs the app (today `gigabyte\ngocn`) | read + write + delete (writes sessions, labels, metadata, deletion requests) |
| coordinator | COORD-01 (if they have their own Windows account) | read + write + delete |
| annotator (raw_access) | ANN-A, ANN-B (if they have their own Windows accounts) | read + write (labels); **no delete, no permission changes** |
| SYSTEM, Administrators | Windows itself / administrators | full (cannot be prevented by an ACL anyway) |
| everybody else | — | **nothing** |

### 2.2 Tell the software which Windows account is which
Create `data/campaign/windows_accounts.json` (owner; pseudonyms are the ones in `access.json`, values are Windows account names
as `whoami` prints them):
```json
{"app_account": "gigabyte\\ngocn",
 "accounts": {"ANN-A": "gigabyte\\ngocn", "ANN-B": "gigabyte\\ngocn", "COORD-01": "gigabyte\\ngocn"}}
```
If all people use the app on the one laptop account, map them all to the app account (as above). If someone has their own
Windows account, write it there; the plan then grants exactly the role's rights.

### 2.3 Steps (owner or administrator, after confirming each line)
1. `python scripts/data_acl.py snapshot` — records the current ACL (fingerprint) and a restorable copy in `backups\acl\`.
   *(Done once on 2026-11-22 23:19: `backups/acl/data_acl_20261122_231922.txt`.)*
2. `python scripts/data_acl.py plan` — prints and saves the commands (`experiments/acl/acl_plan_<time>.cmd.txt`). **Nothing is run.**
   The plan, generated on this PC:
   ```
   icacls "D:\SanphambpAI\data" /save "D:\SanphambpAI\backups\acl\data_acl_before_<time>.txt" /t /c
   icacls "D:\SanphambpAI\data" /grant "gigabyte\ngocn:(OI)(CI)M"
   icacls "D:\SanphambpAI\data" /inheritance:d
   icacls "D:\SanphambpAI\data" /remove:g "NT AUTHORITY\Authenticated Users" "BUILTIN\Users" "Everyone"
   ```
   **Order matters.** The explicit grant for the app account comes **first**. With UAC, the Administrators group is *deny-only* in
   a normal (non-elevated) session. If the broad groups were removed first, the app account could lose write access.
3. Run the lines one by one. The owner of the folder may change its permissions without elevation.
4. `python scripts/data_acl.py compare` (what changed), `inspect` (must report no problem), `snapshot` (records the new state).
5. Start the app and record a short **parked** camera-test session to confirm the app still writes normally.
6. **Rollback** if anything goes wrong:
   `icacls "D:\SanphambpAI" /restore "D:\SanphambpAI\backups\acl\data_acl_before_<time>.txt" /c`.

Only `data\` is restricted. The project folder keeps its permissions (the code, the `.venv` and `acl_probes\` stay usable).

### 2.4 Verify with real accounts (required for PASS)
An ACL listing alone is not proof. Item 7 is PASS only when, **after** the latest ACL snapshot:
- a **standard** (not administrator, not elevated) account that is **not** authorised ran `python scripts/data_acl.py probe` and was
  **denied** (cannot list `data\sessions`, cannot read a raw video, cannot write);
- the app account ran `probe` and was **allowed** (read + write);
- every separately mapped person account ran `probe` and could read.

This PC has **only one enabled account**, so a test account is needed. The administrator creates it, uses it, then deletes it:
```
net user sd_acl_test * /add          (asks for a password; new accounts are standard users)
  log in as sd_acl_test, open a terminal in D:\SanphambpAI, run:  .venv\Scripts\python scripts\data_acl.py probe
net user sd_acl_test /delete
```
Results are written to `acl_probes\probe_<user>_<time>.json` (outside `data\`, so a denied account can still write its result).
Until this is done, item 7 is at best **NOT VERIFIED**.

## 3. Encryption of the data drive D: (item 6)

### 3.1 What exists on Windows 11 Home
| Feature | Home | Pro | What it protects |
|---|---|---|---|
| **Device encryption** (BitLocker-based, automatic) | yes, **if the hardware supports it** | yes | the OS drive and, when active, fixed data drives; the recovery key is backed up to a Microsoft account |
| **BitLocker** (full management, `manage-bde -on`, Control Panel) | no management UI | yes | any fixed drive |
| **BitLocker To Go** (USB drives) | can **unlock / read / write** an encrypted USB drive, **cannot create** one | yes | removable drives |

So Windows Home is **not** "without encryption". Whether Device encryption is available here depends on the hardware (TPM 2.0, plus
Modern Standby or HSTI compliance).

### 3.2 What was measured without administrator rights (indications only, not verification)
| Probe | Result |
|---|---|
| `manage-bde -status D:` | *access denied* (needs administrator) |
| `Get-BitLockerVolume -MountPoint D:` | access denied |
| Shell property `System.Volume.BitLockerProtection` | C: `0`, D: empty |
| Registry `BitLockerStatus\BootStatus` | `0` |
| `powercfg /a` | S3 standby available, **no Modern Standby (S0)** |

No Modern Standby can mean that *automatic* Device encryption is not supported on this laptop. This is **not** a conclusion. Only
the administrator check below decides.

### 3.3 Administrator steps
1. Open **Terminal (Admin)** in `D:\SanphambpAI` and run:
   `.venv\Scripts\python scripts\encryption_status.py --drive D --save`.
   The tool runs `manage-bde -status D:` and `Get-BitLockerVolume` and keeps the raw output and its SHA-256. It records the
   result **only** when the terminal is elevated and the output can be parsed.
2. Read the result:
   - **on**: *Protection On* **and** 100 % encrypted. Item 6 becomes **PASS** for 30 days (re-run monthly).
   - **suspended**: encrypted but *Protection Off*. On Home this is typical when Device encryption is "waiting for activation"
     (sign in with a Microsoft account to back up the key). **Not protected** → FAIL.
   - **off** → FAIL; **encrypting** → NOT VERIFIED until 100 %.
3. If it is off, open `msinfo32` (as administrator) → *Device Encryption Support*.
   - "Meets prerequisites" → Settings > Privacy & security > **Device encryption** → On. Store the recovery key safely.
     Then repeat step 1.
   - Otherwise Device encryption cannot be used on this hardware. Options:
     - upgrade to Windows **Pro** (BitLocker for D: and BitLocker To Go for the backup drive);
     - keep campaign data on another, encrypted machine.
4. The **backup drive** must be encrypted too. On Home, create BitLocker To Go on a Pro PC, then run step 1 with `--drive <USB letter>`
   here as administrator (`campaign_backup.py` reads that record).

**Limitation.** v3.4.4 accepts only BitLocker / Device-encryption evidence from Windows tools. Third-party containers (for example
VeraCrypt) are not detected and would need a reviewed code change.

## 4. What the software never does
- change ACLs, create accounts, or turn encryption on or off;
- count a configuration value, a Shell property or a clean-looking ACL listing as PASS;
- run an ACL change in bulk: the plan is text that a person reads and runs.
