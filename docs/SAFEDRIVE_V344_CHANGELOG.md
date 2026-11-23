# SafeDrive AI v3.4.4: Changelog (NO-GO closure & campaign GO readiness)

**This version does not change the AI model.**
- No retraining.
- No change to model weights, feature extraction, thresholds, calibration or alert logic.
- `configs/frozen_v3_hashes.json` was not touched (SHA-256 1d15d1cc…).
- No session data was deleted or edited: 23 legacy_unknown + 2 demo sessions are unchanged.
- No ACL, account or encryption setting was changed. The camera was not opened.

**Source backup before the changes:** `backups/src_v343_before_v344_20261122_225817.zip` (`src`, `pages`, `ui`, `scripts`,
`tests`, `configs`, `docs`, `app.py`, `data/campaign/*.json*`). The project is not a git repository.

## New
| File | What |
|---|---|
| `scripts/data_acl.py` | `inspect` / `snapshot` / `compare` / `plan` / `probe` for `data\` permissions. Rights are split into read / write / delete / change-permissions, and the inheritance source is reported. `plan` prints the icacls commands (app account granted first) with a rollback command and **executes nothing**. `probe` is run by the account under test and writes to `acl_probes\` |
| `scripts/encryption_status.py` | Reads `manage-bde -status` / `Get-BitLockerVolume` (administrator). It records the parsed state, the volume serial and the raw output plus its SHA-256 only when elevated. Without elevation it prints indications and says BLOCKED |
| `tests/test_v344_closure.py` | 21 tests (see the closure report §7) |
| `docs/SAFEDRIVE_V344_*.md` | closure report, security guide, backup guide, campaign checklists, annotator training report, this changelog |

## Changed
| File | Change |
|---|---|
| `src/realworld/readiness.py` | **Item 6:** strict `manage-bde` parsing. *On* requires Protection On **and** 100 %; protection off on an encrypted volume = suspended (FAIL). Administrator records are accepted for 30 days for the same volume serial. The Shell property is an indication only (never PASS). **Item 7:** least-privilege evaluation of `data`, `data/sessions` and `data/campaign` against `windows_accounts.json`, an ACL fingerprint, and probes by real accounts after the latest snapshot (a standard account denied, the app account allowed, each mapped person allowed). A clean listing alone is NOT VERIFIED. **Items 12 / 13:** structured coordinator checklists (`attest_checklist`): 8 camera / safety points and 8 legal points, each with an answer and a basis. *needs_legal_advice* keeps the item open. Single-tick attestations no longer count for these items. **Item 11:** an approved test copy counts only if the reviewer confirmed it was recorded parked. **Item 14:** submission state. Every item carries `checked_at` and `evidence_at` |
| `src/realworld/privacy.py` | `review(..., recorded_parked, log_path)`; append-only `privacy_review_log.jsonl` (reviewer, time, decision, finding, segment hashes); `load_review`, `review_log` |
| `src/realworld/training.py` | `submit` (raw-access annotators only, once, labels locked by SHA-256), `submissions`, `is_locked`, `key_visible` (results / key only after all submitted). Invalid if labels change after submission. Per-annotator result files; `discrimination_by_clip_kind`; collision / near-miss / risk-event marked "not scorable" on this pack |
| `src/realworld/checklist.py` | 5 safety points added to the pre-session checklist: no_view_obstruction, airbag_clear, cable_safe, no_laptop_while_moving, stop_procedure |
| `scripts/campaign_backup.py` | `MANIFEST.sha256`. Verify checks file count, manifest = index, unindexed files and an unreadable index. A drive removed mid-copy keeps the verified-only index. `restore` refuses targets inside `data\`. Destination encryption comes from Windows tools or an administrator record. New `report` action |
| `scripts/precampaign_audit.py` | Runs go to `experiments/audit_v344/`; the elevation state is logged; the audit's own throw-away reviews log into the run folder (never the real review log) |
| `pages/14_Campaign_Readiness.py` | Evidence time and check time per item, an evidence expander per item, a new tab for the camera / safety and legal checklists |
| `pages/11_Ground_Truth_Annotation.py` | Training workspace: submit button, locked labels after submission, coordinator result only when every annotator submitted |
| `tests/test_v343_readiness.py` | Two tests adapted to the stricter rules: a single legal tick is now asserted to stay NOT VERIFIED; the full-evidence GO test supplies checklists and ACL probes. No assertion was removed |
| `tests/conftest.py` | Autouse fixture: tests never write to the real review log, training submissions or encryption records |
| `docs/SAFEDRIVE_SOFTWARE_CHANGELOG.md` | v3.4.4 row: new software_sha256 **0ee9dc6074f8cbe8…** |

## Measured on this PC (read-only)
- `backups/acl/data_acl_20261122_231922.txt`: the ACL snapshot before any change (restorable).
- `experiments/acl/acl_plan_20261122_231923.cmd.txt`: the generated plan (not executed).
- `acl_probes/probe_gigabyte_ngocn_*.json`: the probe as the app account, allowed (before any ACL change).
- `experiments/backup_reports/backup_report_20261122_231950.md`: 0 backup records.
- `experiments/audit_v344/20261122_232020/`: audit **NO-GO** (6 PASS · 1 FAIL · 4 BLOCKED · 4 NOT VERIFIED).
