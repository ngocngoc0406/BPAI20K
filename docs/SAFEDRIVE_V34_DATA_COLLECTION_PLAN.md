# SafeDrive AI v3.4: Real-World Data Collection Plan

This phase **collects data only. It does not improve the model.**
- The system under evaluation is frozen: v3 LightGBM ×5, its features, thresholds, alert logic and calibration.
- The campaign never trains a model.
- Algorithm changes may only be proposed after the end conditions in §8, and must say which failure they address and which dev set they use.

## 1. Frozen system and provenance
- **Frozen reference.** `configs/frozen_v3_hashes.json` was written once on 2026-11-21 and is refused if overwritten.
  It records, for v3 and v3-stabilized:
  - `model_sha256`: every file of the bundle;
  - `feature_config_sha256`: feature version, groups, window, feature names, preprocessing / detection / tracking configuration,
    and the detector weights hash;
  - `threshold_config_sha256`: risk thresholds, profiles, persistence, calibration, EMA and warm-up;
  - `software_sha256`: `src/`, `pages/` and `ui/`.
- **Identity check.** The bundles are byte-identical to the backups that were verified earlier: v1, v3, v3-stabilized, and the
  v2-light weights.
- **Per session.** Every session writes `provenance.json` at its start, with the same hashes and a comparison against the frozen reference.
  - A model, feature or threshold mismatch makes the session **ineligible**.
  - A software difference is allowed only with an entry in [SAFEDRIVE_SOFTWARE_CHANGELOG.md](SAFEDRIVE_SOFTWARE_CHANGELOG.md).

## 2. Campaign design
| Item | Plan |
|---|---|
| Volume | **≥ 10 h** of supervised real driving with full-session video (initial target) |
| Session | 30–60 min, one fixed camera mount, the same laptop, model v3, the **balanced** profile throughout (one profile, so FA/h is comparable) |
| Spread (declared, not inferred) | day and night, urban / highway / curves, sparse and dense traffic; record whatever occurs, never force it |
| Drivers | pseudonyms only (D01, D02, …). Several drivers make the group-wise split possible: with one driver, all data is in one partition |
| Trips | `trip_id` for one drive (a trip can have several sessions). Partitions are assigned per driver, else per trip, else per session |
| Never | create dangerous situations, brake hard, cut in, or test the system against other road users |

Expected yield, as a planning assumption (not a measurement):
- Warning-worthy events are rare in normal driving.
- 10 h will probably give **fewer than 10 confirmed events per type**, so recall will be reported as *insufficient*.
- FA/h will be estimable: 10 h with 0 false alarms gives an upper bound of 0.37 FA/h; 50 false alarms give 3.7–6.6 FA/h.

## 3. Session procedure (driver never touches the screen while moving)
1. **Before driving, parked.**
   - Check that the camera is mounted and the laptop is powered.
   - On *Live Driving Assistant*, choose the model **v3** and the **balanced** profile.
2. **Open *Trip / session metadata*.**
   - Fill in: data kind = *Real driving*; lighting, weather, road types, traffic (or **unknown**); camera mount; campaign id; trip id; driver pseudonym.
   - Tick:
     - driver consent;
     - occupants informed;
     - supervisor present;
     - configured before moving;
     - recording rules checked;
     - full-session video;
     - (optional) share consent.
3. **Press ▶ Start camera, then ● Start session.**
   - The software refuses to start the session if consent, supervisor, setup or recording-rule flags are missing.
   - It writes `trip_meta.json` and `provenance.json`, and starts the timeline, system log and full video.
4. **Drive normally.** The supervisor watches the road, not the screen. The prototype is never the only safety aid.
5. **After parking:** press ■ Stop session, then ⏹ Stop camera.
6. **Back up the session.** On the external encrypted drive, run:
   - `python scripts/campaign_backup.py backup --dest <drive>:\`;
   - then `verify`.
   Keep the drive separate from the laptop.

## 4. What every session stores (`data/sessions/<id>/`)
| File | Content |
|---|---|
| `trip_meta.json` | declared metadata, consent flags, campaign / trip / driver ids |
| `provenance.json` | model, feature, threshold and software hashes, plus the frozen-reference check |
| `timeline.csv` | one row per 10 Hz step: level, P, risk score, MI, alarm, DEGRADED, capture FPS, step / e2e ms, objects, corridor flag, RSS (every 5 s), **camera state**, **camera interruptions** |
| `system_log.csv` | every 5 s: process RSS, CPU % of the machine, GPU utilisation and memory (nvidia-smi), free disk, session size |
| `audio_log.jsonl` | every audio decision |
| `video/segment_NNN.mp4`, `segments.json` | raw full-session video, 10 fps, 5-min segments; frames written and dropped |
| `session_report.{json,pdf}` | summary and model events |
| `annotations_<name>.jsonl`, `evidence/`, `disagreements.jsonl` | independent labels, evidence frames, disagreement records |
| `shared/` | blurred copy + `blur_log.json` + `privacy_review.json` (see the privacy protocol) |

Dropped or missing data is measured, not assumed:
- missing 10 Hz steps and stream resets (timeline);
- video frames dropped by the writer;
- video-frame vs timeline-step desynchronisation (> 1 % is flagged).

## 5. Dataset manifest, partitions and leakage (`scripts/campaign_report.py`)
- **`data/campaign/manifest.json`**, per session:
  - session / trip / driver / campaign ids, data kind, declared conditions, hours;
  - timeline steps vs video frames, missing steps;
  - SHA-256 and a perceptual hash (8 sampled dHash values) for every video segment;
  - provenance hashes and their check, annotators and full reviews, privacy status;
  - issues, eligibility, partition.
- **Eligibility.** All of: real driving, consent / supervisor / setup / recording-rule flags, provenance matching the frozen
  reference, a timeline without missing steps, video in sync.
  - Not eligible: legacy_unknown sessions, demo, controlled and camera tests.
- **Partitions** (`data/campaign/partitions.json`):
  - assigned once per group (driver, else trip, else session) by a salted hash: train 50 % / dev 20 % / **final_test 30 %**, or
    manually with a written reason;
  - never reassigned, never split by frame or clip.
- **Leakage check.** Fails the manifest if any of the following holds:
  - a trip, driver or group appears in two partitions;
  - a video segment is byte-identical (SHA-256) to another session's;
  - two segments of different partitions are near-duplicates (mean dHash Hamming distance ≤ 6 of 64).
- **final_test** is evaluated only with `--final-test --by <coordinator>` at the end of the campaign. Every access is logged
  in `final_test_access_log.jsonl`.

## 6. Metrics (`campaign_metrics.json`, consensus labels, independent events, never frames)
- **Scope.** Computed only on eligible sessions that were fully reviewed by **two** independent annotators.
- **Reference events.** Taken from the two annotators' consensus (see the annotation guide):
  - only *confirmed* events count;
  - disputed, single-annotator and uncertain situations are exclusion zones: an alarm there is neither true nor false.
- **Rates.**
  - FA/h with an exact Poisson CI, and a session-bootstrap CI when there are ≥ 5 sessions;
  - recall per event type (collision / near_miss / risk_event / all) with a Wilson CI, only when ≥ 10 confirmed events of that
    type exist, otherwise "insufficient (n = k)".
- **Timing and repetition.** Lead times, warned ≥ 1 s and ≥ 2 s (Wilson CIs), repeated warnings.
- **By declared condition.** The same metrics by lighting, weather and road type. Groups are not pooled across data kinds.
- **Data quality.** Total hours, trips, sessions, events per type, the share of sessions with missing steps or desync.
- **Agreement.** `agreement_report.json` with kappa and time offsets.

## 7. Software changes during the campaign
- Any change to `src/`, `pages/` or `ui/` changes `software_sha256` and must be entered in the changelog, with its date, the files
  and why it does not touch the model, features, thresholds or alerts.
- A change that does touch them is not allowed during the campaign. It would start a new campaign with a new frozen reference.

## 8. End conditions (after ≥ 10 h)
1. Data-quality review: `validation_report.json` with no unexplained issue.
2. Every eligible session fully reviewed by two independent annotators; disagreements recorded, none forced.
3. Agreement computed.
4. The leakage check passes.
5. A report of the limitations and of the situations with no data: night, rain, curves, specific manoeuvres.
6. **No automatic move to training.** A proposal for an algorithm change must:
   - name the failure it fixes, using evidence from train / dev;
   - define the dev set used for selection;
   - keep final_test untouched until the single final evaluation.
