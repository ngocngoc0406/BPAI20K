# SafeDrive AI v3.3: Real-World Driving Validation

**Date:** 2026-11-21.

**Models:** frozen, not retrained.
- v1, v2-light, v3 and v3-stabilized are byte-identical to their backups (SHA-256 of every file in each bundle).
- The one exception is a `feature_version` field in the v2-light `bundle.json`, added by the v3.1 audit before v3.2.
  The v2-light weights are identical.

**No feature definition, threshold or benchmark was changed.**
- The new stress condition `light_change` is kept outside the original corruption list.

**Real-driving data recorded: 0 hours.** No supervised road session has been carried out. All road-related metrics in this report
are therefore **not measured**.

Companion documents:
- [SAFEDRIVE_V33_FAILURE_ANALYSIS.md](SAFEDRIVE_V33_FAILURE_ANALYSIS.md): details and sources;
- [SAFEDRIVE_V33_EVALUATION_PROTOCOL.md](SAFEDRIVE_V33_EVALUATION_PROTOCOL.md): how to collect, label and score real data.

## 1. What was built
| Deliverable | Where | Status |
|---|---|---|
| Session metadata, declared only (lighting, weather, road, traffic, mount), plus consent and supervisor gating | `src/realworld/trip_meta.py`, Live page "Trip / session metadata" | done, tested |
| Per-step timeline, audio log, consented full-session video (people blurred) | `src/realworld/session_log.py`, `LiveController.start_session(meta)` | done, tested, 30-min run |
| Ground-truth annotation: 6 categories incl. *uncertain*, risk start / critical / end, confidence, note, evidence rule, blind mode, second annotator, agreement (kappa, tIoU, onset difference) | `src/realworld/annotation.py`, page **11 Ground Truth Annotation** | done, tested; **not yet used on real data** |
| Real-world metrics: FA/h, repeated warnings, warnings without hazard, warning / CAUTION recall, warned ≥ 1/2/3 s, lead times, runtime, recording completeness, missing / duplicate steps, storage per hour | `src/realworld/metrics.py` | done, tested on synthetic logs and recorded sessions |
| Dashboard **12 Real-world Evaluation**: tabs ① benchmark, ② live camera, ③ controlled, ④ real driving, failure gallery; filters by model, lighting, road, session, category | `pages/12_Real_World_Evaluation.py` | done; tab ④ shows **0 h** |
| Trip-data deletion (session folder, events and clips) | page 11, `delete_session` | done, tested |
| Session dataset | `data/sessions/` | 2 v3.3 sessions with metadata (1-min smoke, 30-min run; both demo_video), plus 23 older sessions without metadata (v3.2 live runs and sessions written by an automated test, fixed in v3.3); **no real-driving session** |

## 2. What was tested
The four kinds of evidence are reported separately.

### ① Benchmark video: held-out Nexar, 224 videos, deployed v3
| Profile | Recall [95 % CI] | FA/h [95 % CI] | Warned ≥ 2 s | Median lead |
|---|---|---|---|---|
| conservative | 8.9 % [4.9, 15.7] | 2.6 [0.5, 7.6] | 1.8 % | 0.78 s |
| balanced | 11.6 % [6.9, 18.9] | 8.7 [4.2, 16.0] | 2.7 % | 1.00 s |
| sensitive | 17.9 % [11.9, 26.0] | 16.5 [9.9, 25.8] | 4.5 % | 1.07 s |

- These are reproduced by `scripts/realworld_benchmark_strata.py` and match the published values.
- FA/h differs slightly (8.7 vs 8.5) because of the normal-hour definition.
- Strata by Nexar's verified lighting, weather and scene, and by two H3 proxies: see Failure Analysis §1–2. No stratum differs
  measurably from the overall value. Dark and twilight have only 11 collision videos.

### ② Live camera
- No new webcam runs were made in v3.3.
- The v3.2 webcam results still stand: 5 and 15 min, CPU-only, disconnect and busy camera (`experiments/live/`).

### ③ Controlled tests
| Test | Outcome |
|---|---|
| **30 min continuous run** with event recording **and** full-session video (demo replay, GPU, sensitive profile) | **PASS.** 1800.6 s, no error, 9.98 processed steps/s, step 36 / 92 ms and end-to-end 52 / 111 ms (p50 / p95), CPU 18 % of machine, GPU memory ≤ 1349 MB. 54/54 event clips saved with 100/100 frames. Full video: 17,979 frames in 6 segments, equal to the 17,979 timeline steps, 0 dropped, 0 encoder errors. Timeline: 0 missing, 0 duplicate steps, 1 stream reset (start-up). RSS 3039 → 2984 MB (peak 3912 MB at ~7.5 min, non-monotonic). Storage 555 MB in 30 min: full video 316 MB (about 630 MB/h), clips 223 MB, timeline 1.4 MB |
| Rapid light change, window level (60 test videos) | mAP 0.805 → 0.695 / 0.657, ROC-AUC 0.905 → 0.870 / 0.837, ECE 0.093 → 0.114 / 0.135 (sev 1 / 2). Exposure is partial, see FA §4.2 |
| Rapid light change, **full streams, deployed alarms, paired with the same videos** | Detections per step −4.5 % / −10.3 %. Mean P shift per video −0.029 [−0.037, −0.022] / −0.039 [−0.049, −0.028]. P falls more on collision videos than on normal ones. Recall (balanced) 2/30 → 1/30 → 1/30 (McNemar p = 1.0). **The alarm-level effect cannot be tested with 30 + 30 videos** |
| Shake, occlusion, low light, noise, blur, frame drop | Unchanged from v3.1 (`robustness_v3*.json`) |
| Disconnect / reconnect, busy camera, near-full disk | Unchanged from v3.2 (passed) |

**Event-only vs full-video recording mode.** This is not a controlled A/B: the runs differ in length and version.

| Measure | v3.2 event-only (5 min) | v3.3 events + full video (30 min) |
|---|---|---|
| Late steps | 19 % | 23 % |
| Step time p50 / p95 | 29 / 80 ms | 36 / 92 ms |
| End-to-end p50 / p95 | 49 / 101 ms | 52 / 111 ms |
| CPU | 17.9 % | 18.0 % |

- Full-video recording costs about 7–12 ms per step at p50 / p95 and roughly 630 MB of disk per hour.
- The 10 Hz rate was held in both modes.

### ④ Real driving
**0 h, 0 sessions, 0 annotated events.** The target is ≥ 10 h.

## 3. Results summary
| Item | Result |
|---|---|
| Tooling for real-world collection, annotation, agreement, metrics, dashboard, deletion | **Passed** (unit and integration tests, 30-min run) |
| 30-min continuous recording | **Passed** |
| Consent gating, never inferring conditions, labelling rules | **Passed** (tests) |
| Robustness to rapid light change | **Not shown.** Perception and probabilities degrade measurably; the alarm-level effect cannot be measured |
| H3 fixed-corridor failures | **Not measurable on the benchmark proxies.** The proxy differences lie within the CIs. Road curvature is not measured |
| Real-world FA/h, recall, lead time, repeated warnings | **Not measured** (0 h) |
| Inter-annotator agreement on real data | **Not measured** (no annotations yet) |
| Test suite | **105 passed** (90 before v3.3 + 15 new) |

## 4. Has the model improved?
**No.** v3.3 changed no model, and none was meant to change.
- The deployed v3 still warns about 1 in 9 collisions (11.6 %) at about 9 FA/h on held-out data, with a median lead of 1.0 s.
- v3.3 adds the ability to measure real-world performance. It does not improve performance.

## 5. Decision gate: which branch next
Each criterion is scored 1–5, where higher is better for the branch. For "computational cost" and "regression risk",
a high score means low cost and low risk.

| Criterion | A. Road Geometry | B. Adaptive Warning | C. Near-Miss Intelligence | D. More data first |
|---|---|---|---|---|
| Evidence of the problem | **1**: H3 proxies show no measurable effect (FA §2) | **2**: 17 persistence misses, but the rule removes short runs equally in both classes (65 % / 66 %); with a pre-event AUC of 0.63, rule changes trade recall for FA (FA §3) | **1**: no annotated near-misses exist; "potential near-miss" is model-defined only | **5**: 0 h of real driving; real FA/h unknown; discrimination is the bottleneck; v3's held-out gain over v2 came from more data (ΔS −0.003 vs the same-data baseline) |
| Expected product value | 3: better corridor messages in curves | 3: fewer repeated warnings (17 repeats < 30 s in the 30-min demo; demo only) | 4 if it works: a useful trip review | 4: enables every later decision and gives the first honest road numbers |
| Technical feasibility | 2: needs lane or road segmentation and calibration; none is available | 5: cheap, within the current pipeline | 2: needs labelled near-misses to learn or validate | 4: the tooling is ready; needs a supervised driver, a vehicle and time |
| Computational cost | 2: an extra segmentation model per frame; step p95 is already 92 ms with full video | 5: negligible | 4: offline | 5: none at run time; about 0.63 GB per hour of video |
| Evaluation feasibility | 1: no curve-labelled data | 4: on dev streams | 1: no ground truth | 4: protocol, metrics and agreement are implemented |
| Regression risk | 2: new perception component | 3: thresholds are bound to FA budgets; must be re-selected on dev | 4: does not touch warnings | 5: no change to the system |
| **Total (of 30)** | **11** | **22** | **16** | **27** |

**Decision: Branch D, collect supervised real-driving data before any algorithm change.**
- Follow the protocol: ≥ 10 h of normal driving with full-session video, then a blind annotation pass and a second annotator.
- Then compute:
  - the first real FA/h with an interval;
  - repeated warnings and warnings without a hazard;
  - the corridor-related errors in sessions tagged `curves`.

**B is the runner-up, under conditions.** It may be explored later, but only:
- after D gives a real FA/h and real repeated-warning rate (the problem B would address);
- with every rule selected on dev streams and a measured FA cost;
- without B being presented as a way to improve separation, which the evidence does not support.

**A and C are not supported by current evidence.**
- A would need curve-tagged real data first.
- C would need annotated near-misses first.

**The underlying bottleneck is discrimination** (AUC 0.63).
- Improving it is a modelling and data problem: more labelled data, especially real data from the target camera.
- No newer algorithm is chosen simply because it is newer.

## 6. Failed, incomplete or unverified
1. **Real driving: 0 h.** Everything road-specific is unverified:
   - FA/h on the road, recall and lead time;
   - curves, lane changes, cut-ins, lead braking, many vehicles;
   - the camera mount.
2. **Annotation tool:** not used on real data, so inter-annotator agreement is unmeasured.
3. **Light-change robustness:** degradation measured; the alarm-level effect is untestable at this sample size.
   The window-level test under-exposes (only about 4 s clips).
4. **No clean A/B of event-only vs full-video mode:** the comparison mixes run length and version.
5. **Memory:** no leak trend over 30 min, but swings of about 1 GB occurred. Runs longer than 30 min (e.g. 2 h) have not been done.
6. **Privacy:** **licence plates are not blurred**, because the models have no plate detector. People blurring has not been checked
   by a human for misses.
7. **Unchanged from v3.2:**
   - no spoken Vietnamese (no vi-VN voice);
   - audibility in a vehicle not verified.
8. **Unrepresentative event rate:** the 30-min run used a looping collision clip, so its 63 alarms (126 per hour) and
   17 repeated warnings say nothing about real driving.

## 7. Reproduce
```powershell
.venv\Scripts\python scripts\realworld_benchmark_strata.py   # benchmark strata + H3 proxies
.venv\Scripts\python scripts\v33_temporal_overlap.py          # score overlap / temporal consistency
bash scripts/v33_stress_suite.sh                               # 30-min run + window-level light change
.venv\Scripts\python scripts\v33_light_change_stream.py --keep-awake   # full-stream paired light change
.venv\Scripts\python -m pytest tests -q
```
