# SafeDrive AI v3.4: Collection Report

**Date:** 2026-11-21.

**Campaign:** `v34-realworld`.

**Status:** tooling complete. **Real-driving data collected: 0 h. The campaign has not started on the road.**

Generated sources (all in `data/campaign/`, produced by `scripts/campaign_report.py`):

| File | Content |
|---|---|
| `manifest.json` | dataset manifest |
| `validation_report.json` | data validation |
| `agreement_report.json` | annotation agreement |
| `campaign_metrics.json` | metrics on consensus labels |
| `legacy_audit.json` | audit of the 23 older sessions |
| `collection_summary.md` | generated summary |

Dashboard: page **Data Collection Campaign**.

## 1. Model preservation
- **No retraining, no change** to features, thresholds, alert logic or calibration.
- **Models byte-identical to the backups:** v1, v3 and v3-stabilized (every file), and the v2-light weights.
- **Frozen reference** written once: `configs/frozen_v3_hashes.json` (v3: model 77a3361d3ebba791…, features 870ab60075fbe67a…,
  thresholds 050ad4ca855e44d3…, software 3cc98c394530751f…). The current code matches it on all four hashes.
- **V3.3 benchmark unchanged.** Re-running `realworld_benchmark_strata.py` and `v33_temporal_overlap.py` after the v3.4 changes
  reproduced `v33_benchmark_strata.json`, `v33_benchmark_per_video.csv` and `v33_temporal_overlap.json` **byte for byte**.
- **Software changes** are recorded in [SAFEDRIVE_SOFTWARE_CHANGELOG.md](SAFEDRIVE_SOFTWARE_CHANGELOG.md).

## 2. What was built (v3.3 tools reused, not rebuilt)
| Need | Reused from v3.3 | Added in v3.4 |
|---|---|---|
| Session metadata, consent | `TripMeta`, consent gating | campaign / trip / driver-pseudonym ids, `regulations_checked`, `share_consent`; names refused as driver ids |
| 10 Hz timeline, video | `TimelineLogger`, `SessionVideoWriter` | camera state and interruption columns; `system_log.csv` (RSS, CPU, GPU utilisation / memory, free disk, session size, every 5 s) |
| Model and config hashes | — | `provenance.py`: per-session `provenance.json` checked against the frozen reference |
| Privacy | people blurring at recording, deletion | `privacy.py`: blurred shared copies (region blur, low threshold, ±2-frame persistence), mandatory human review, raw / shared access tiers, deletion log |
| Annotation | annotation store, kappa, tIoU matching | schema v3.4 (collision / near_miss / risk_event / normal / uncertain), required event time and visual evidence with saved frames, enforced blind mode (no model output or clips), kappa with unmatched situations, start / event / end offsets, confusion table, disagreement records with reasons, consensus without forced agreement |
| Splits | — | `campaign.py`: manifest with SHA-256 and perceptual hashes, group-wise frozen partitions (train / dev / final_test), leakage and near-duplicate checks, final-test access log |
| Metrics | `metrics.py` building blocks | consensus-based FA/h (Poisson and session-bootstrap CIs), recall per event type only with ≥ 10 confirmed events (Wilson CI), warned ≥ 1 / 2 s, strata by declared lighting / weather / road |
| Dashboard | page 12 | page 13 **Data Collection Campaign** (quality, partitions and leakage, agreement, metrics, legacy, deletion log); pages 10 / 11 updated |

**Tests:** 117 passed (105 in v3.3). The 12 new tests are:
- 9 in `tests/test_campaign.py`;
- 1 in `tests/test_realworld.py` (consensus);
- 2 page tests for the new dashboard.

Updated tests: the v3.4 annotation rules, agreement, metrics and the controller test (provenance, system log).

**Fixed:** automated tests no longer write sessions into `data/sessions`. The session count was 25 before and after the full suite.

## 3. Campaign figures (from `validation_report.json` and `campaign_metrics.json`)
| Metric | Value |
|---|---|
| Real-driving hours / sessions / trips / drivers | **0.00 h / 0 / 0 / 0** (target ≥ 10 h) |
| Eligible sessions | 0 |
| Events per type (confirmed) | collision 0, near_miss 0, risk_event 0 |
| Sessions with missing frames / desync | 0 / 0 (no real sessions) |
| False alarms per hour | **not computed** (no double-annotated session) |
| Recall by event type, lead time, warned ≥ 1 s / 2 s | **not computed:** "insufficient (n = 0 < 10 confirmed events)" for every type |
| By lighting / weather / road | no data |
| Inter-annotator agreement | not available |
| Partitions | train 0, dev 0, final_test 0; excluded 25 sessions (0.52 h) |
| Leakage / duplicate check | passed (no partitioned data) |
| Privacy-approved shared copies | 0 |

## 4. The 23 older sessions (decision)
- **Data:** nothing was deleted.
- **Classification:** each session without `trip_meta.json` is `legacy_unknown`.
  - No metadata was assigned.
  - The only thing recorded is the observed `source` field of its report.

  | Observed origin (from the `source` field) | Sessions | Total length | Model events |
  |---|---|---|---|
  | automated test (pytest fixture) | 13 | 0.9 min | 0 |
  | camera source, no record of what was filmed | 6 | 24.5 min | 0 |
  | replay of a recorded Nexar clip | 4 | 7.6 min | 13 |

- **Decision:**
  - all 23 are kept unchanged;
  - they are excluded from every partition and from official metrics, and they show as legacy_unknown on the dashboards.
- **Other excluded sessions:** the 2 v3.3 demo_video sessions are also excluded (not real driving; recorded before provenance existed).

## 5. Failed, incomplete or unverified
1. **No real driving: 0 h.** Every road metric of §6 of the brief is unmeasured. This is not a software failure: no supervised
   drive has taken place.
2. **Two independent annotators:** the workflow exists and is tested, but has not been used on real data.
3. **Privacy blurring.**
   - There is no face or plate detector, so the method is region-based, and its blur recall has **not been measured**.
   - The human review is mandatory.
   - `make_shared_copy` was only run on a test clip (tests).
4. **Access control** is application-level on one computer. Real protection needs Windows account permissions and BitLocker.
   `data/campaign/access.json` has been created with **empty** lists: the coordinator must add pseudonyms.
5. **Legal check:** this report does not establish which recording rules apply. The operator must check them before the first
   session (`regulations_checked`).
6. **The perceptual-hash duplicate threshold** (≤ 6 of 64 bits) is a design choice. It has not been calibrated on real data.
7. **Expected limitations once data exists** (see the dataset card):
   - few drivers make the partitions coarse;
   - rare events will probably leave recall "insufficient" after 10 h;
   - night, rain, curves and specific manoeuvres may have no data.

## 6. Next step
- **Begin supervised sessions** following the collection plan.
- **After each session:**
  - two blind full reviews;
  - privacy review of any copy to be shared;
  - re-run `scripts/campaign_report.py`.
- **After ≥ 10 h:** apply the end conditions of the plan (§8).
- **No training starts automatically.** An algorithm change may only be proposed with evidence from train / dev, a named
  failure, and a dev design. final_test stays sealed until the single final evaluation.
