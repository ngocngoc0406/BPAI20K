# SafeDrive AI v3.4.3: Changelog (Pre-Campaign Readiness & Operations)

**This version does not change the AI model.** None of the following was changed:
- no retraining;
- no change to model weights, features, thresholds, calibration or alert logic;
- `configs/frozen_v3_hashes.json` was not touched;
- none of the 23 legacy_unknown or 2 demo sessions was deleted.

**Source backup before the changes:** `backups/src_v342_before_v343_20261122_195753.zip`. It holds `src`, `pages`, `ui`,
`scripts`, `tests`, `configs`, `docs`, `app.py` and the `data/campaign` JSON files.

## New
| File | What |
|---|---|
| `src/realworld/readiness.py` | The 15 readiness checks, with PASS / FAIL / BLOCKED / NOT VERIFIED and the GO rule (GO only if all 15 are PASS). Windows probes for drives, bus type, physical disk, BitLocker and ACL, which can be injected in tests. Coordinator-only attestations, with an evidence reference and withdrawal. A camera check that runs only when a person asks and stores numbers, never images. The storage model |
| `src/realworld/checklist.py` | The operator checklist before every real-driving session: 6 automatic items (access, disk, backup, a camera check within 60 min, model hashes, readiness GO within 7 days) and 7 manual confirmations. `require_ready()` |
| `src/realworld/training.py` | The annotator-training pack (built outside `data/sessions`), the hidden key, the training report and the pass criteria |
| `pages/14_Campaign_Readiness.py` | Readiness table and verdict, pre-session checklist and camera test, coordinator confirmations and the sharing decision |
| `tests/test_v343_readiness.py` | 14 tests (see the precampaign report) |
| `data/annotator_training/sessions/TRAINING_PACK_01/` | The practice pack: 9 Nexar train-split clips, 99 s, marked as training data |

## Changed
| File | Change |
|---|---|
| `src/realworld/privacy.py` | **Review:** only a coordinator may review. Approval needs every segment checked plus the explicit finding that nothing is readable; *readable* → rejected. The review stores per-segment SHA-256. **New:** `shared_status` (pending_review / approved / rejected / stale / revoked), `sharing_policy` / `set_sharing_policy` (the no-sharing decision blocks every export), and `revoke_shared` (blocks export, lists exports, can delete reachable exports, keeps a log). `export_shared` now uses the status and the policy. Re-creating a copy clears an earlier revocation |
| `src/live/controller.py` | `start_session`: meta is validated first. For **real_driving** sessions, `checklist.require_ready()` must pass first: a complete checklist from the last 2 h, at least 3 GB free and the frozen hashes still matching. Otherwise nothing is recorded |
| `src/realworld/campaign.py` | `campaign_status()` for the dashboard (read-only) |
| `pages/11_Ground_Truth_Annotation.py` | Workspace switch (campaign / annotator training); training result for the coordinator; privacy form with the explicit finding, status line and revocation |
| `pages/13_Data_Collection_Campaign.py` | Campaign status block: **REAL-WORLD DRIVING HOURS**, eligible / excluded sessions, missing provenance, backup, privacy, annotation, training, split, final_test and the NO-GO list |
| `scripts/campaign_backup.py` | New actions `drives`, `check` and `restore-test`. Destination rules: connected, not in the project, a different **physical** disk, external, enough space, encrypted (or explicitly recorded as unencrypted), writable. Interrupted copies keep only verified files in the index. Every action is logged to `backup_status.jsonl`. Errors come with recovery hints |
| `scripts/precampaign_audit.py` | The default run is the 15 readiness checks plus an active test that the no-sharing decision blocks export. Writes `experiments/audit_v343/<run>/readiness_results.json` and `data/campaign/readiness_latest.json`. `--full` re-runs the v3.4 mechanism tests |
| `tests/test_campaign.py` | Updated for the stricter review: a coordinator plus the explicit finding is required, and the backup refusal is now a `BackupError` |

## Software hash
- The new `software_sha256` is recorded in [SAFEDRIVE_SOFTWARE_CHANGELOG.md](SAFEDRIVE_SOFTWARE_CHANGELOG.md) (entry v3.4.3).
- The model, feature and threshold hashes are unchanged.
