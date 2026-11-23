# SafeDrive AI v3.4: Pre-Campaign Readiness Audit

> **Superseded on 2026-11-22 by [SAFEDRIVE_V343_PRECAMPAIGN_REPORT.md](SAFEDRIVE_V343_PRECAMPAIGN_REPORT.md)** (15-item readiness audit, still NO-GO). Kept as the record of rounds 1 and 2.

| Field | Value |
|---|---|
| Date | 2026-11-22 |
| Script | `scripts/precampaign_audit.py` (read-only towards the project: all tests run on copies; no camera opened, no recording session started) |
| Round 1 | `experiments/audit_v34/20261122_173701/`: NO-GO (access BLOCKED, privacy FAIL, backup BLOCKED) |
| Round 2 (current) | `experiments/audit_v34/20261122_181715/` (`audit_results.json`, `audit.log`, `manual_findings.json`), after the fixes listed below |
| Statuses | PASS = verified with evidence; FAIL = requirement not met; BLOCKED = needs a human action; NOT VERIFIED = insufficient evidence |

**Conclusion: still NO-GO.**
- **Backup is BLOCKED:** no external drive is connected.
- **Not verified:** the privacy blur (it needs a human review on campaign-camera footage), BitLocker / OS protection
  (admin only), the legal check and the in-vehicle setup.

## Fixes between round 1 and round 2 (owner decisions of 2026-11-22)
| Problem in round 1 | Fix | Model / features / thresholds |
|---|---|---|
| Access lists empty | The owner chose the pseudonyms: `raw_access` = ANN-A, ANN-B; `coordinators` = COORD-01 | unaffected |
| Blur missed rear plates (lower 45 % band) | Vehicles are blurred over the **whole box**; every blur box is padded by 8 % (`src/realworld/privacy.py`) | unaffected |
| No sanctioned share path | `export_shared()` plus a coordinator export form on page 11. Only an approved, unchanged blurred copy can be exported, by a coordinator, with SHA-256 verification and a log entry | unaffected |
| No backup tool | `scripts/campaign_backup.py`: backup / verify / restore with SHA-256, incremental; it refuses the data drive without `--allow-same-drive` | unaffected |

- **Software changes** are recorded in `docs/SAFEDRIVE_SOFTWARE_CHANGELOG.md` (v3.4.1, software_sha256 cfc40cf4743d2873…).
- **Model, feature and threshold hashes are unchanged.**
- **Tests:** 119 passed (+2: export guard, backup / verify / restore).

## Results (round 2)
| # | Item | Status |
|---|---|---|
| 1 | Frozen hashes and provenance | **PASS** |
| 2 | Access control (application level) | **PASS** |
| 2b | OS protection (BitLocker, folder permissions) | **NOT VERIFIED** |
| 3 | Privacy | **NOT VERIFIED** (mechanisms pass; blur quality needs a human review on campaign footage) |
| 4 | Data integrity, storage, backup | **BLOCKED** (no external backup drive) |
| 5 | Annotation | **PASS** (workflow; simulated labels) |
| 6 | Dataset governance | **PASS** |
| — | Project unchanged by the audit | **PASS** |
| — | Legal / recording-rule check; in-vehicle setup | **NOT VERIFIED** (human only) |

### 1. Frozen hashes: PASS
- **Model, features, thresholds.** The hashes of v3 and v3-stabilized equal the frozen reference.
- **Software.** The software hash differs from the frozen one (cfc40cf4… vs 3cc98c39…). The new hash **is recorded in the
  changelog**, as the plan requires.
- **Frozen file intact.**
  - `freeze()` refuses to overwrite it.
  - Its modification time equals `frozen_at` (2026-11-21 11:42).
  - Its SHA-256 is unchanged (1d15d1cc…).
- **Models still match the verified backups.** v1, v3 and v3-stabilized are byte-identical; the v2-light weights are identical.
- **Mismatch detection.** A copy-session with a tampered threshold hash was marked ineligible ("model / feature / threshold
  hash differs from the frozen reference").

### 2. Access control: PASS / 2b OS protection: NOT VERIFIED
- **Lists filled.** `access.json` contains raw_access ANN-A, ANN-B and coordinators COORD-01. The owner chose these;
  the audit did not.
- **Unlisted name.** An AppTest of page 11 showed no video ("not on the raw-access list"), and the coordinator functions
  were refused.
- **Limitation:** a typed name, no authentication.
- **BitLocker.** `Get-BitLockerVolume` returned *Access denied* because it needs administrator rights, so encryption and folder
  permissions could not be checked.

### 3. Privacy: NOT VERIFIED
**Mechanisms (all PASS):**
- A new shared copy is not shareable.
- An approval with unchecked segments is refused.
- Export of an unreviewed copy is refused ("no approved privacy review").
- Export by a non-coordinator (ANN-A) is refused.
- The deletion test removes the folder and its events, and logs ids and counts only.

**Blur quality.** The blur was re-run on the same 3 Nexar urban clips.
- In the two close-up rear views that failed in round 1, the plate positions are now **fully covered** (`review_frames/zoom_plates.jpg`).
- **Side effect:** in one frame, a very large detection (probably the ego hood or dashboard detected as a car) blurs most of the
  image. Shared copies may therefore be heavily over-blurred and of limited research value. Annotators use the raw video,
  so labelling is not affected.
- **Why not PASS:**
  - this was a spot check by the AI assistant, not the required human review;
  - Nexar clips are already anonymised, so residual readability cannot be measured on them.
  - **A human review of a shared copy made from campaign-camera footage is required before anything is shared.**

### 4. Data integrity: BLOCKED (all checks pass; no backup drive)
- **Existing 30-min demo session:** 17,979 steps = 17,979 frames written = 17,979 frames decoded; 0 missing; timestamps monotonic;
  332 audio records within the timeline span.
- **v3.4 recording path.** A 2-min **demo file replay** into the audit folder (no camera):
  - 1,200 steps = 1,200 frames, 0 dropped, 0 missing, sync difference 0;
  - provenance eligible;
  - system log with GPU fields;
  - audio log present.
- **Fault injection** on a copy: 40 removed steps were detected, and a video truncated to 80 % was reported as desynchronised.
- **Storage for 10 h:** 7.8–11.1 GB raw (14.5–17.7 GB with shared copies), with 107 GB free on D:.
- **Backup tool:**
  - first run copied 9 files; the second run copied 0 (incremental);
  - verify OK;
  - restore of 9 files, hash-verified;
  - tampering with a file is detected (unit test).
- **No external drive** is connected (only C: and D:, both fixed). → **BLOCKED.**

### 5. Annotation: PASS
Two accounts (ANN-A, ANN-B) were used on a demo audit sample.
- **Blind view:** only one's own 4 rows, none of the other annotator's notes, no model output, no kappa.
- **Coordinator view:** results are hidden until both full reviews are done. Afterwards: 3 matched situations, κ (with unmatched) 0.27,
  κ event vs not 0.50.
- **Kappa and offsets:** κ on matched categories = 0.50, equal to a hand computation. Event-time offset 0.5 s, as expected.
- **Disagreement:** the disputed situation stays disputed after both annotators gave reasons. *Uncertain* is handled as an exclusion zone.
- **Limitation:** the labels are simulated and only exercise the software.

### 6. Dataset governance: PASS
- **Manifest:** 25 sessions; every video segment has a SHA-256 and a perceptual hash; the leakage check is OK.
- **legacy_unknown:** all 23 sessions are excluded and ineligible. None was modified or deleted.
- **Split:** group-wise split, stable on reassignment.
- **final_test locked by default:** default metrics use train and dev only, and `--final-test` without `--by` is refused.
- **Limitation:** the perceptual hash does not detect overlapping recordings at different offsets.

## Remaining human actions (mandatory before the first session)
1. **Backup drive.** Connect an external drive, encrypted with BitLocker To Go, then:
   - run `python scripts/campaign_backup.py backup --dest <drive>:\`;
   - run `verify`;
   - run one `restore --to <test folder>`;
   - re-run the audit.
2. **OS protection.** As administrator, enable BitLocker on D: (or confirm it is on), and limit `D:\SanphambpAI\data` to the
   campaign Windows account(s).
3. **Privacy decision.** Either:
   - record a short consented **parked** clip with the campaign camera that shows plates and people, create the shared copy, and
     have COORD-01 review it (approve only if nothing is readable); or
   - decide in writing that nothing is shared until that has been done. Recording itself keeps raw video local.
4. **Legal check.** Confirm the local rules for in-vehicle recording and for processing images of people and plates.
5. **In-vehicle setup.**
   - Fixed mount, power, the laptop out of the driver's reach.
   - A supervisor briefed on the protocol.
   - A **parked** dry run.
6. **Annotators.** Two people trained on the annotation guide; each uses only their own pseudonym (ANN-A, ANN-B).

## Decision: **NO-GO**
- GO requires items 1, 2b, 3 (or the written no-sharing decision), the legal check and the vehicle setup to be PASS or settled.
- Re-run `python scripts/precampaign_audit.py` afterwards.
- No recording session was started by this audit.
