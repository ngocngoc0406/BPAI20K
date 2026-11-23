# SafeDrive AI v3.4.4: NO-GO Closure Report

| | |
|---|---|
| Date | 2026-11-22 |
| Audit run | `experiments/audit_v344/20261122_232020/` (`readiness_results.json`, `audit.log`); latest copy in `data/campaign/readiness_latest.json` |
| Real-world driving recorded | **0.0 h**: 0 eligible sessions; 25 excluded (23 legacy_unknown, 2 demo) |
| Model | v3 LightGBM ×5 and v3-stabilized, **frozen** (not retrained, not changed) |
| **Decision** | **NO-GO**: 6 PASS · 1 FAIL · 4 BLOCKED · 4 NOT VERIFIED on 2026-11-22. **Latest (2026-11-23 18:06): 11 PASS · 0 FAIL · 0 BLOCKED · 4 NOT VERIFIED** (items 11–14, human confirmations) |

v3.4.4 closes every NO-GO condition that **software** can close. What remains needs:
- an external drive;
- administrator rights;
- a second, standard Windows account;
- the camera in the vehicle (parked);
- COORD-01's checklists;
- two annotators.

None of these was simulated. Nothing was changed in permissions or encryption, the camera was not opened, and nothing was recorded.

## 1. Issues addressed (software and procedure)
| Area | Before (v3.4.3) | Now (v3.4.4) |
|---|---|---|
| Folder permissions (item 7) | FAIL on broad groups; a clean listing would have given PASS | **Cause found:** `D:\` carries the Windows default *Authenticated Users: Modify* and *Users: Read*, and `SanphambpAI\data` inherits it. `scripts/data_acl.py` adds inspect, snapshot (restorable), compare, plan and probe. The plan grants the app account explicitly **first** (UAC deny-only Administrators), never runs anything, and includes rollback. PASS requires least-privilege ACL **and** probes by real accounts after the snapshot: a standard account denied, the app account and each mapped person allowed |
| Encryption (item 6) | `manage-bde` "Protection On" alone = PASS; the Shell property could give PASS | Strict parsing: *on* = Protection On **and** 100 %; *protection off* on an encrypted volume = suspended (FAIL). The Shell property is an indication only. `scripts/encryption_status.py` records an administrator reading (raw output + SHA-256, volume serial), valid for 30 days. Guide: Device encryption vs BitLocker vs BitLocker To Go on Home |
| Backup (items 8–10) | backup / verify / restore-test | `MANIFEST.sha256`; verify checks count, manifest = index, unindexed files and an unreadable index; drive removed or full mid-copy keeps only verified files; restore never into `data\`; encryption from Windows tools only; `report` action |
| Privacy (item 11) | coordinator review, states, revocation, no-sharing | Append-only review log (reviewer, time, decision, finding, segment SHA-256). Test footage counts only if the reviewer confirms it was recorded **parked** |
| Camera & safety (item 12) | camera check + one "camera_mount" tick | 8-point camera / safety checklist (airbag zone, view, cables, no laptop while moving, supervisor, parked stable run, stop procedure), each with a basis. 5 safety points added to the pre-session checklist |
| Legal (item 13) | one "legal_review" tick | 8-point legal checklist: purpose, consent, faces / plates, retention, access, sharing, deletion on request, local rules. *needs_legal_advice* keeps it open; `local_rules`, `consent`, `retention` and `access` cannot be "not applicable" |
| Training (item 14) | report against a hidden key | Locked submission per annotator (SHA-256). Key and results only after both submitted. Invalid if labels change. Per-annotator result files; discrimination by clip kind |
| Dashboard | status per item | Evidence time and check time per item, evidence per item, checklist tab; REAL-WORLD DRIVING HOURS 0.0 (page 13, real_driving only) |

## 2. Issues not resolved (cannot be resolved by software)
| # | Item | Status | Why | Who |
|---|---|---|---|---|
| 6 | Data drive D: encrypted | **BLOCKED** | `manage-bde` and `Get-BitLockerVolume`: access denied without elevation. The Shell property for D: is empty. No Modern Standby, so Device encryption support is uncertain | Windows administrator |
| 7 | Folder permissions | **FAIL** | inherited *Authenticated Users: delete/read/write* and *Users: read* on `data`, `data\sessions`, `data\campaign`. Also missing: `windows_accounts.json` and a standard test account (the PC has one enabled account) | owner + administrator |
| 8 | External backup destination | **BLOCKED** | only C: and D: (same NVMe disk 0); no USB drive | user |
| 9 | Backup verified | **BLOCKED** | no backup (0 records) | user |
| 10 | Restore verified | **BLOCKED** | no backup | user |
| 11 | Privacy review | **NOT VERIFIED** | no reviewed campaign-camera copy, no no-sharing decision | COORD-01 |
| 12 | Camera readiness | **NOT VERIFIED** | no camera check in the vehicle, no safety checklist | operator + COORD-01 |
| 13 | Legal | **NOT VERIFIED** | legal checklist not answered; local rules need qualified advice | COORD-01 / owner |
| 14 | Annotator training | **NOT VERIFIED** | 0 / 2 submitted | ANN-A, ANN-B, COORD-01 |

## 3. Evidence actually measured on this PC
| Evidence | Result | File |
|---|---|---|
| ACL inspect | 6 distinct problems: Authenticated Users (delete/read/write) and Users (read), inherited, on 3 folders | `python scripts/data_acl.py inspect` |
| ACL snapshot (before any change) | fingerprint 47b42f72…; restorable copy | `backups/acl/data_acl_20261122_231922.txt`, `data/campaign/acl_snapshots.jsonl` |
| ACL plan | generated, **not executed** | `experiments/acl/acl_plan_20261122_231923.cmd.txt` |
| Probe as `gigabyte\ngocn` (app account, admin member, not elevated) | allowed: list / read raw / write | `acl_probes/probe_gigabyte_ngocn_*.json` |
| Encryption, not elevated | BLOCKED, nothing saved. Indications: Shell D: empty, BootStatus 0, no Modern Standby, Windows 11 Home Single Language | `python scripts/encryption_status.py --drive D` |
| Drives | C: and D: NVMe disk 0, not external; "BLOCKED: no external drive connected" | `campaign_backup.py drives` |
| Backup check on C: | refused: same physical disk, encryption unknown; nothing created | `campaign_backup.py check --dest C:\SafeDriveBackupTest` |
| Backup report | 0 records, 0 valid external backups | `experiments/backup_reports/backup_report_20261122_231950.md` |
| Readiness audit | NO-GO, 6 / 1 / 4 / 4 | `experiments/audit_v344/20261122_232020/` |
| **2026-11-23: USB drive E: connected** | Kingston, FAT32, 58 GB, 40.9 GB free, USB disk 1 (separate physical disk); `check` refused it (encryption) | `campaign_backup.py drives / check` |
| **2026-11-23: elevated encryption read** (owner approved UAC) | C:, D: and E: **not encrypted** (Fully Decrypted, Protection Off, no key protectors); TPM 2.0 enabled and activated | `experiments/encryption_admin_check.log`, `data/campaign/encryption_status.jsonl` |
| **2026-11-23: attempt to turn on BitLocker** on E: and D: (owner request) | `manage-bde -on` **refused by Windows** (0x8031005A) on both; state re-checked, nothing changed | `experiments/encryption_enable.log` |
| Audit after these steps | NO-GO **6 PASS · 3 FAIL (6, 7, 8) · 1 BLOCKED (10) · 5 NOT VERIFIED** | `experiments/audit_v344/20261123_001102/` |
| **2026-11-23: USB contents saved before formatting** (owner request) | 276,243 files / 11.4 GB copied to `D:\USB_E_backup_20261123` and **SHA-256 verified file by file**: missing 0, differ 0 | `D:\USB_E_backup_20261123_copy.log` |
| **2026-11-23: VeraCrypt 1.26.31 installed** | official MSI; SHA-256 equal to the published checksum; Authenticode signature valid (IDRIX SARL) | `experiments/veracrypt_install.log` |
| **2026-11-23: encrypted backup container** (owner request; the owner chose the password) | E: formatted to exFAT; container `E:\safedrive_backup.hc` (40 GiB, AES, SHA-512) mounted as **X:**. Detected through the VeraCrypt driver: `\Device\VeraCryptVolumeX`, container on USB disk 1 | `experiments/veracrypt_setup.log` |
| **2026-11-23: real backup to X:** | backup 29 files (2 demo sessions + `data/campaign`) ok; verify ok (manifest = index, 0 missing, 0 mismatch); restore-test ok (29 files, temp folder removed) | `data/campaign/backup_status.jsonl`, `experiments/backup_reports/backup_report_20261123_162045.md` |
| Audit after the backup | NO-GO **9 PASS · 2 FAIL (6, 7) · 0 BLOCKED · 4 NOT VERIFIED (11–14)** | `experiments/audit_v344/20261123_162312/` |

| **2026-11-23: folder permissions restricted** (owner request) | ACL saved (`backups/acl/data_acl_before_20261123_170405.txt`). Then: grant `gigabyte\ngocn:(OI)(CI)M`, `/inheritance:d`, remove *Authenticated Users* and *Users*. `data\` and everything below now: ngocn (M), Administrators, SYSTEM. `inspect`: no problem | `data/campaign/acl_snapshots.jsonl`, `data/campaign/windows_accounts.json` |
| **2026-11-23: real-account probes** | app account `gigabyte\ngocn`: allowed (list / read raw / write). Temporary **standard** account `sd_acl_test` (not admin, not elevated): **denied** (UnauthorizedAccess on list, read and write). The account and its profile were deleted right after | `acl_probes/`, `experiments/acl_standard_probe.log` |
| Audit after these steps | NO-GO **10 PASS · 1 FAIL (6: D: not encrypted) · 4 NOT VERIFIED (11–14)** | `experiments/audit_v344/20261123_171455/` |

| **2026-11-23: encrypted session store** (owner request; the owner chose the password) | VeraCrypt container `D:\SafeDriveVault.hc` (60 GiB, AES, SHA-512, NTFS) mounted as **V:**; `V:\sessions` restricted (ngocn M, Administrators, SYSTEM). The 67 session files (25 sessions, 0.34 GB) were copied and SHA-256-verified twice, then `data\sessions` was replaced by a **junction** to `V:\sessions` and the unencrypted original deleted | `experiments/vault_setup.log` |
| Audit after the move | NO-GO **10 PASS · 0 FAIL · 5 NOT VERIFIED** (7: the standard-account probe must be repeated for the new location; 11–14 human items) | `experiments/audit_v344/20261123_173358/` |

| **2026-11-23 18:06: standard-account probe repeated for the new location** | temporary standard account `sd_acl_test`: **denied** (list, read and write); the account and profile were deleted. App account: allowed | `acl_probes/probe_gigabyte_sd_acl_test_20261123_180635.json` |
| **Audit (latest)** | NO-GO **11 PASS · 0 FAIL · 0 BLOCKED · 4 NOT VERIFIED**. The 4 open items (11–14) are human confirmations | `experiments/audit_v344/20261123_180645/` |

**Daily use from now on.** After every start of Windows, open VeraCrypt and mount `D:\SafeDriveVault.hc` as **V:** before starting the app; without V: the app cannot write sessions (fail closed). For a backup, also mount `E:\safedrive_backup.hc` as **X:**. Dismount both before unplugging the USB. The deleted unencrypted copy may still be recoverable from free space on D: (normal deletion); it contained only the 23 legacy and 2 demo sessions.

**Note on the container password.** The owner chose a 4-digit password. Such a password can be guessed quickly, so the container's
protection is weak. Item 8 is PASS because the software can verify that encryption is on; it cannot measure password strength.
Recommended: change it in VeraCrypt (*Volumes → Change Volume Password*) to ≥ 20 characters.

## 4. Steps for the user
1. Get an external USB drive (≥ 64 GB, 128 GB recommended) and encrypt it with BitLocker To Go **on a Windows Pro PC**.
2. Plug it in here, and let the administrator confirm its encryption: `encryption_status.py --drive <letter> --save` (elevated).
3. Run `campaign_backup.py drives` → `check` → `backup` → `verify` → `restore-test` → `report`
   ([backup guide](SAFEDRIVE_V344_BACKUP_GUIDE.md)).
4. Fill `data/campaign/windows_accounts.json` ([security guide](SAFEDRIVE_V344_SECURITY_GUIDE.md) §2.2).
5. In the vehicle, **parked**:
   - mount the camera and run *Test camera* on page 14;
   - do a parked dry run (start / stop a session);
   - record a short consented parked test clip for the privacy review.
6. Re-run `python scripts/precampaign_audit.py`.

## 5. Steps for the Windows administrator
1. Encryption of D: run `.venv\Scripts\python scripts\encryption_status.py --drive D --save` in **Terminal (Admin)**.
   - If it is not on: check `msinfo32` → *Device Encryption Support*, then Settings → Device encryption, or upgrade to Pro.
   - Then re-run the check (security guide §3).
2. Permissions: review `python scripts/data_acl.py plan`, then run its lines one by one (app account first). Then `compare`,
   `inspect`, `snapshot` and a parked test recording (security guide §2.3).
3. Create a temporary **standard** account (`net user sd_acl_test * /add`), log in, run `data_acl.py probe` (must be DENIED),
   then delete the account (§2.4).
4. Run the audit once as administrator.

## 6. Steps for COORD-01
1. Page 14 → *Camera / safety & legal checklists*:
   - answer all 8 safety points after inspecting the mounted camera (with a basis and an evidence reference);
   - answer all 8 legal points; get qualified advice at least for `local_rules`, and for `consent`, `faces_plates` and `retention`
     where they apply ([checklists](SAFEDRIVE_V344_CAMPAIGN_CHECKLIST.md) §3).
2. Privacy: review the blurred copy of the parked test clip on page 11, segment by segment, with the explicit finding and the
   parked confirmation. Or record the no-sharing decision on page 14.
3. Training: make sure ANN-A and ANN-B work separately; after both submit, open the result on page 11 and hold the debrief
   ([training report](SAFEDRIVE_V344_ANNOTATOR_TRAINING_REPORT.md)).

## 7. Tests before / after
| | Before | After |
|---|---|---|
| Full suite | **139 passed** (v3.4.3) | **160 passed**, 0 failed: 21 new tests in `tests/test_v344_closure.py`; no test deleted |

Two v3.4.3 tests were adapted to the stricter rules. The single legal tick is now asserted to stay NOT VERIFIED, and the GO test
supplies checklists and ACL probes. No assertion was removed.

| Required case | Test |
|---|---|
| unsafe ACL | `test_unsafe_acl_fails_with_inherited_cause`, `test_least_privilege_extra_rights_unknown_accounts_and_app_lockout`, `test_rights_are_split_into_read_write_delete_and_permissions` |
| unauthorised account reaching raw video | `test_unauthorised_account_reaching_raw_video_fails_and_probes_are_required`, `test_probe_records_what_the_current_account_can_do`, `test_acl_plan_is_never_executed_and_grants_the_app_first` |
| encryption not verified | `test_unverified_encryption_is_blocked_and_only_admin_records_count`, `test_manage_bde_parsing_is_strict` |
| no external drive / too small | `test_no_external_drive_and_too_small_drive_are_refused_before_copying` |
| hash mismatch / restore failure | `test_corrupted_file_manifest_mismatch_and_restore_into_data_are_refused` |
| drive removed or full mid-copy | `test_drive_removed_or_full_mid_copy_keeps_only_verified_files` |
| backup report | `test_backup_report_is_written` |
| privacy review rejected / file changed after review | `test_rejected_review_stale_copy_parked_requirement_and_review_log`, `test_demo_copy_never_validates_and_no_sharing_blocks_export` |
| legal not confirmed | `test_legal_checklist_needs_every_point_and_keeps_legal_advice_open` |
| camera not verified | `test_camera_not_verified_until_check_and_every_safety_point` |
| training not completed / failed | `test_training_incomplete_until_both_submit_and_locked_by_hash`, `test_training_failing_annotator_gives_fail` |
| checklist expired | `test_expired_or_incomplete_checklist_refuses_real_driving` |
| provenance hash mismatch | `test_provenance_hash_mismatch_fails_hash_items` |
| no GO while BLOCKED / NOT VERIFIED | `test_no_go_while_anything_is_blocked_or_not_verified` |

## 8. Model hashes and benchmarks before / after
| | Before (v3.4.3) | After (v3.4.4) |
|---|---|---|
| v3 model_sha256 | 77a3361d3ebba791… | 77a3361d3ebba791… (= frozen) |
| v3-stabilized model_sha256 | 206373f53b52af2b… | 206373f53b52af2b… (= frozen) |
| feature / threshold hashes | = frozen | = frozen (audit items 2–3 PASS) |
| `configs/frozen_v3_hashes.json` | SHA-256 1d15d1cc… | SHA-256 1d15d1cc… (untouched) |
| software_sha256 | 9fac2f8f856a5dca… | 0ee9dc6074f8cbe8… (recorded in the software changelog) |

**Benchmarks re-run after the v3.4.4 changes.** The outputs before the run are kept in `experiments/regression_v344/before/`;
the logs are in `experiments/regression_v344/`.

| Benchmark | Result |
|---|---|
| V3.3 strata (`scripts/realworld_benchmark_strata.py`, 224 held-out videos) | `v33_benchmark_strata.json` and `v33_benchmark_per_video.csv` are **byte-identical** |
| ↳ conservative profile | recall 0.089 [0.049, 0.157] at 2.61 FA/h |
| ↳ balanced profile | recall 0.116 [0.069, 0.189] at 8.69 FA/h |
| ↳ sensitive profile | recall 0.179 [0.119, 0.26] at 16.51 FA/h |
| V3.4.2 class-stabiliser check (`scripts/verify_class_stabilizer.py --n 20`) | `v342_class_stabilizer_check.json` **byte-identical** |
| ↳ model inputs | `all_inputs_identical = True` |
| ↳ label flips | within-group 578 → 82; cross-group 46 → 46; tracks with a class change 163 → 74 |

## 9. Real-world driving hours
**0.0 h.** The manifest has 25 sessions: 23 legacy_unknown and 2 demo_video, all excluded. Demo, Nexar and training data are never
counted.

## 10. Verdict
**NO-GO.** GO requires all 15 items to be PASS with real evidence. Today:
- item 7 is **FAIL**;
- items 6, 8, 9 and 10 are **BLOCKED** (administrator rights, external drive);
- items 11–14 are **NOT VERIFIED** (COORD-01's checklists and review, camera in the vehicle, annotator training).

After the steps in §4–6, run `python scripts/precampaign_audit.py`, once as administrator. Only a 15 × PASS result allows the
first real-driving session.
