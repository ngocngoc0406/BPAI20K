# SafeDrive AI v3.4.3: Pre-Campaign Readiness Report

| | |
|---|---|
| Date | 2026-11-22 |
| Audit run | `experiments/audit_v343/20261122_201627/` (`readiness_results.json`, `audit.log`); the latest copy is in `data/campaign/readiness_latest.json` |
| Real-world driving recorded | **0.0 h** |
| Model | v3 LightGBM ×5, **frozen** (not retrained, not changed) |
| **Decision** | **NO-GO**: 6 PASS · 1 FAIL · 4 BLOCKED · 4 NOT VERIFIED |

Nothing in this version starts the camera or a recording. Nothing in it changes Windows permissions, and nothing turns NO-GO
into GO. GO requires **all 15** items to be PASS with evidence (`readiness.go_decision`). A configuration value set to true
is never accepted as evidence.

## 1. The 15 readiness items (measured on this laptop)
| # | Item | Status | Evidence / reason | Who acts |
|---|---|---|---|---|
| 1 | Model hash | **PASS** | v3 77a3361d…, v3-stabilized 206373f5… = frozen | — |
| 2 | Feature hash | **PASS** | equal to frozen (both models) | — |
| 3 | Threshold / calibration / alert hash | **PASS** | equal to frozen (both models) | — |
| 4 | Software provenance | **PASS** | software_sha256 9fac2f8f… ≠ frozen 3cc98c39…, but **recorded in the changelog** (v3.4.3) | — |
| 5 | raw_access / coordinators | **PASS** | ANN-A, ANN-B / COORD-01 (chosen by the owner); valid pseudonyms | — |
| 6 | BitLocker on the data drive D: | **BLOCKED** | `manage-bde`: *access denied* (needs administrator); the Shell property gives no value. **Windows 11 Home Single Language** | Windows admin |
| 7 | Folder permissions `data/` | **FAIL** | inherited **Authenticated Users: Modify** and **Users: Read**: every Windows account can read the raw video | Windows admin / owner |
| 8 | External backup destination | **BLOCKED** | only C: and D: exist, both on **the same NVMe disk 0**; no USB drive | user |
| 9 | Backup verification | **BLOCKED** | no valid backup (needs item 8) | user |
| 10 | Restore verification | **BLOCKED** | no backup to restore | user |
| 11 | Privacy review | **NOT VERIFIED** | no blurred copy of campaign-camera footage reviewed, no no-sharing decision. Demo / Nexar copies never count | COORD-01 |
| 12 | Camera readiness | **NOT VERIFIED** | no camera check in the vehicle, no mount confirmation (the audit never opens the camera) | operator + COORD-01 |
| 13 | Legal review acknowledgement | **NOT VERIFIED** | no coordinator attestation | COORD-01 / owner |
| 14 | Annotator training | **NOT VERIFIED** | the pack is built; nobody has labelled it yet | ANN-A, ANN-B, COORD-01 |
| 15 | Disk space | **PASS** | D: 106.7 GB free ≥ 24.3 GB (10 h worst case + 30 %) | — |

## 2. What was built (v3.4.3)
- **Readiness checks.** `src/realworld/readiness.py` holds the 15 checks and the GO rule. `scripts/precampaign_audit.py`
  runs them, plus an active test that the no-sharing decision blocks export; `--full` re-runs the v3.4 mechanism tests.
- **Backup.** `scripts/campaign_backup.py` adds the actions `drives`, `check` and `restore-test`. A destination must be
  connected, outside the project, on a **different physical disk**, external, large enough, encrypted and writable.
  - Interrupted copies keep only verified files in the index.
  - Every action is logged to `backup_status.jsonl`, and errors come with recovery hints.
  - **Verified on this laptop:** `check --dest C:\…` is refused (same physical disk, encryption unknown) and nothing is created.
  - Guide: [SAFEDRIVE_V343_BACKUP_OPERATIONS.md](SAFEDRIVE_V343_BACKUP_OPERATIONS.md).
- **Privacy workflow.** Only a coordinator can review a blurred copy.
  - Approval needs every segment checked plus the explicit finding that nothing is readable. *Readable* gives
    **rejected**.
  - The review stores per-segment SHA-256, so a changed copy becomes **stale**.
  - New: the states pending_review / approved / rejected / stale / revoked, the no-sharing decision, and **revocation**
    (blocks export, lists exports, can delete reachable exports). Raw video is never touched.
  - Guide: [SAFEDRIVE_V343_PRIVACY_REVIEW_WORKFLOW.md](SAFEDRIVE_V343_PRIVACY_REVIEW_WORKFLOW.md).
- **Operator checklist.** Page 14 and `src/realworld/checklist.py`, done while parked.
  - 6 automatic items: access, ≥ 3 GB free, earlier real sessions in a verified external backup, a camera check within
    60 min, frozen hashes, readiness GO within 7 days.
  - 7 manual confirmations by a named operator and a different named supervisor.
  - **Enforced in `LiveController.start_session`:** a *real_driving* session is refused, and nothing is recorded, unless a
    complete checklist from the last 2 h exists and the disk and hashes still pass. Demo and camera-test sessions are unaffected.
- **Annotator training.** `TRAINING_PACK_01` holds 9 Nexar train-split clips (99 s): 4 events, 3 normal and 2 degraded
  ("uncertain"). It has a hidden key and pass criteria, and is stored **outside** `data/sessions`.
  - Page 11 has a new *Annotator training* workspace.
  - Guide: [SAFEDRIVE_V343_ANNOTATOR_TRAINING.md](SAFEDRIVE_V343_ANNOTATOR_TRAINING.md).
- **Storage plan.** It uses measured rates: 630–665 MB/h of video and 4.1 MB per clip. 10 h needs 9.0–22.6 GiB with a 30 %
  reserve. Details: [SAFEDRIVE_V343_STORAGE_PLAN.md](SAFEDRIVE_V343_STORAGE_PLAN.md).
- **Dashboard.** Page 13 shows **REAL-WORLD DRIVING HOURS: 0.0**, eligible / excluded sessions, missing provenance, and the
  status of backup, privacy, annotation, training, split and final_test, plus the NO-GO list.
  - Only sessions declared as *real driving* count. Demo, Nexar, training and legacy sessions never count.

## 3. Tests
| | Before | After |
|---|---|---|
| Test suite | **123 passed** | **139 passed**: 14 new in `tests/test_v343_readiness.py` + 2 smoke renders of page 14 |

**Required test cases:**

| Required case | Test |
|---|---|
| no external drive | `test_backup_refused_without_external_drive` (refused, nothing copied, items 8 / 9 / 10 BLOCKED) |
| external drive too small / unencrypted | `test_external_drive_not_enough_space_or_unencrypted_is_refused` (item 8 FAIL) |
| backup hash mismatch | `test_backup_hash_mismatch_and_failed_restore_are_reported` (verify fails with a recovery hint; item 9 FAIL) |
| restore failure | same test (`restore-test` with a damaged or missing file is not ok; item 10 FAIL; `restore` refused) |
| interrupted copy | `test_interrupted_backup_keeps_only_verified_files` |
| unauthorised person | `test_unauthorised_people_are_refused` |
| unreviewed video | `test_unreviewed_copy_is_pending_and_never_exported` |
| readable face / plate | `test_readable_face_or_plate_rejects_copy_and_revocation_blocks_export` |
| provenance mismatch | `test_provenance_mismatch_fails_hash_items_and_blocks_session` |
| checklist incomplete | `test_incomplete_or_old_checklist_refuses_real_driving_session` (the controller records nothing) |
| no GO while BLOCKED / NOT VERIFIED | `test_go_only_when_everything_is_pass` and `test_full_evidence_gives_go_and_acl_with_broad_groups_fails` |

`tests/test_campaign.py` was updated for the stricter review (a coordinator plus the explicit finding is required).

## 4. Regression: model and earlier benchmarks
- **Hashes.** The model, feature and threshold hashes of v3 and v3-stabilized equal the frozen reference.
- **Frozen file untouched.** `configs/frozen_v3_hashes.json` has SHA-256 1d15d1cc… and modification time 2026-11-21 11:42.
- **V3.3 benchmark** (`scripts/realworld_benchmark_strata.py`, 224 held-out videos): re-run after the changes. The JSON is
  **identical** apart from timestamps, and the per-video CSV is byte-identical.
  - Balanced profile: recall 0.116 [0.069, 0.189] at 8.69 FA/h, unchanged.
- **V3.4.2 class-stabiliser check** (`scripts/verify_class_stabilizer.py --n 20`): identical, see §4.1.
- Earlier outputs are kept in `experiments/regression_v343/before/`.

### 4.1 V3.4.2 re-run
`scripts/verify_class_stabilizer.py --n 20` was re-run after the v3.4.3 changes. The output is **identical** to the v3.4.2
output (`experiments/latest/v342_class_stabilizer_check.json` vs `experiments/regression_v343/before/`):
- `all_inputs_identical = True`;
- every per-video comparison is unchanged: motion, frame difference, ego-motion, track boxes / ids / confidences, P and alarms;
- label flips are unchanged:
  - within-group flips 578 → 82;
  - cross-group flips 46 → 46;
  - tracks with a class change 163 → 74.

## 5. Answers
1. **Done:**
   - the 15-item audit with the GO rule;
   - backup destination checks, restore-test, recovery hints and logging;
   - the privacy review states, rejection, revocation and the no-sharing decision;
   - the operator checklist, enforced at session start;
   - the camera check (person-initiated);
   - the annotator training pack and report;
   - the storage plan;
   - the dashboard status;
   - 16 new tests;
   - the documentation.
2. **Not verifiable yet:**
   - BitLocker (administrator rights);
   - the backup and restore on a real external drive;
   - the blur on campaign-camera footage;
   - the camera in the vehicle;
   - legality;
   - annotator skill.
3. **NO-GO conditions left:** items 6, 7, 8, 9, 10, 11, 12, 13 and 14.
4. **User:**
   - buy or connect an external drive of ≥ 64 GB (128 GB recommended), encrypted (see the backup guide §4; Windows Home cannot
     create BitLocker To Go);
   - run `backup` → `verify` → `restore-test`;
   - in the vehicle, parked: the camera test and the checklist;
   - a parked dry run.
5. **Windows administrator:**
   - read or enable encryption of D: (`manage-bde -status D:`; on Home: *Device encryption*);
   - restrict `D:\SanphambpAI\data` (icacls commands in the backup guide §6);
   - run the audit once as administrator.
6. **COORD-01:**
   - privacy: either review a blurred copy of a consented **parked** campaign-camera clip, or record the no-sharing decision on page 14;
   - confirm the camera mount and the legal check, with evidence references, on page 14;
   - supervise the annotator training and debrief.
7. **Tests:** 123 before, **139 after**, all passing.
8. **Model hash unchanged:** yes, for the model, features and thresholds of both v3 and v3-stabilized.
9. **Earlier benchmarks unchanged:** yes. The V3.3 strata benchmark and the V3.4.2 stabiliser check are both identical.
10. **GO?** **No: NO-GO.**
    - The external backup drive, administrator rights, a reviewed campaign-camera copy (or the no-sharing decision), the legal
      acknowledgement, camera readiness and annotator training are all missing.
    - The folder permissions currently **FAIL**.

**Next:** after the human actions, run `python scripts/precampaign_audit.py`, once as administrator. GO is shown only when
all 15 items are PASS.
