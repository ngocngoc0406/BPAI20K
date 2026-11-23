# SafeDrive AI — Next Version (v3.1) and Plan for v4

This iteration implements only the roadmap items with a clear reason and **no change to any model's predictions**: P0 consistency
fixes and P1 product features with zero inference cost. v1, v2-light, v3 and v3-stabilized are all kept.

**Backup before changes:** `backups/v3_20261121/` (bundles, result files, source).

## 1. Implemented in v3.1
| Item | Why (audit ref.) | What changed | Verification |
|---|---|---|---|
| Feature-definition versioning | M7 | `FEATURE_VERSION` in `src/features/window_features.py`; bundles record `feature_version`; `TabularPredictor` refuses mismatches | `tests/test_safety.py::test_feature_version_guard` |
| Consistent calibration in the study | M1 | `run_ew_study.py` held-out streams now use mean(calibrated member probabilities), the same as the product | Product numbers re-measured with the deployed code (§2) |
| **Warning-sensitivity profiles** | Product: adaptive warning | Conservative / Balanced / Sensitive = alarm thresholds at 5 / 10 / 20 false alarms per hour on dev OOF streams (already in the bundle); sidebar selector; `pipeline.analyze(profile=...)` | Held-out, deployed logic (§2); test |
| **Near-Miss Memory** | Product gap M5 | `src/safety/near_miss.py`: events (alert; near-miss candidate = P ≥ CAUTION for ≥ 0.6 s without an alarm), each with clip (−3 / +1 s), measured cues, explanation, lead road user, model, profile. Local JSONL store and a new **Near-Miss Memory** page | Tests; held-out validation (§2); end-to-end on held-out videos 00288 / 00901 |
| **Trip safety report** | Product gap M5 | `src/safety/trip_report.py`: aggregates the clips of a trip; alerts and candidates per hour, cues, lead-object classes, event list; PDF + JSON. **No safety score** (not validated) | Test; example `reports/trip_demo-heldout_report.pdf` |

Benchmark after the changes (5 test videos, v3, balanced profile): **44.5 fps GPU, 20.0 fps CPU-only** — unchanged within run-to-run
variation. Event extraction takes about 1 ms per clip. Tests: **73 passed**.

## 2. Measured results (held-out 224 videos; deployed code; no choice made on these data)
**Warning-sensitivity profiles, v3** (`experiments/latest/profiles_heldout_deployed.json`). All realised FA/h are within their dev budgets.

| Profile (dev budget) | Recall: valid alarm ≤ 5 s before the event [95 % CI] | Alarm ≥ 2 s ahead | False alarms / h [95 % CI] | Median lead |
|---|---|---|---|---|
| Conservative (5) | 8.9 % [4.9, 15.7] | 1.8 % | 2.5 [0.5, 7.4] | 0.8 s |
| Balanced (10, default) | 11.6 % [6.9, 18.9] | 2.7 % | 8.5 [4.1, 15.6] | 1.0 s |
| Sensitive (20) | 17.9 % [11.9, 26.0] | 4.5 % | 16.1 [9.7, 25.1] | 1.1 s |

For v3-stabilized, Conservative and Balanced share the same threshold (equal on dev). Sensitive gives 11.6 % recall at 11.0 FA/h.

**Near-Miss Memory, v3 balanced** (`near_miss_memory_heldout.json`):
- 22 of 112 collision videos (19.6 % [13.3, 28.0]) have at least one stored event within 5 s before the collision (alerts alone: 13).
- Normal driving produces 17.8 events per hour [11.0, 27.2] (8.5 alerts + 9.3 candidates).

So the memory roughly doubles the pre-collision episodes available for review, at roughly double the event rate. It is a
**review / coaching tool with low precision**, not a detector of verified incidents.

**Product alarm logic = evaluated logic** (`audit_deployed_alarm_logic.json`): v3 recall 11.6 % at 8.5 FA/h both ways; UNCERTAIN
suppressed no alarm step. For v2-light the product alarm (risk score + trend / looming) operates at 18.8 % recall and 15.7 FA/h,
which differs from its probability-only evaluation.

## 3. Not done in v3.1 (reasons)
| Item | Reason |
|---|---|
| Fix `lead_brake_proxy` (L1) | Changes a feature v3 was trained on. Now possible safely via `FEATURE_VERSION`, but needs retraining and re-evaluation → v4 |
| OOF thresholds from bagged fold models (M2) | Changes thresholds; must be re-validated on dev → v4 |
| Lane / road-geometry context (H3) | A modelling change with unknown benefit; must go through the dev protocol first → v4 experiment |
| Live camera input, audible alert | Engineering effort with no evidence needed, but outside this iteration's "no unproven change" scope → v4 product track |

## 4. Plan for v4
**Model track.** Each item must pass the v3 dev protocol: 5-fold CV on 1,276 videos, paired video bootstrap, keep only if the 90 % CI of ΔS > 0.
1. Dynamic driving corridor from a per-clip vanishing-point / horizon estimate (cheap) or a light lane model; replaces the fixed corridor in interaction cues.
2. `FEATURE_VERSION = 2` with the `lead_brake_proxy` fix; retrain and compare against v3 on dev.
3. Thresholds and calibration from bagged fold models (M2), to reduce the FA/h transfer error.
4. Bigger held-out: the current 224 videos cannot resolve 2–3 point recall changes (FA/h CI spans about 4×). Collect or record ≥ 10 h of normal driving for FA/h estimation.

**Product track.**
5. Live input (webcam / RTSP), with a ring buffer feeding the existing extractor (already 1.8–4.4× real time on the test laptop) and an alert banner plus sound.
6. Field test protocol: fixed model version, logged alerts and clips, driver annotations of true / false alerts, FA/h per profile. No safety claims.

**Hardware.** Export YOLOv8n to ONNX / TensorRT and measure on the target device before any edge claim. Today only the RTX 3050
Laptop (42–45 fps) and the i5-11400H CPU (20 fps; 18 fps on 2 threads) are measured.

## 5. Answers (final)
1. **What it does.** Real-time road-user detection and tracking; image-plane approach and interaction cues; calibrated window risk
   (held-out ROC-AUC 0.816); FA-budgeted alerts with selectable sensitivity; grounded explanations; and now a near-miss memory with
   trip reports. It does **not** reliably warn before collisions (≈ 9–18 % of collisions at 2.5–16 FA/h).
2. **Most serious weaknesses.** Low streaming recall and lead time; a fixed corridor without lane / road geometry; no metric distance;
   the v3 gains are unconfirmed on held-out (v3 ≈ v2 retrained on the same data); no live input.
3. **Difference from dashcam ADAS.** It does not beat calibrated FCW / LDW systems at warning, and does not claim to. Its differences are
   *measured* false-alarm budgets that the user can choose, per-event uncertainty and grounded explanations, and a local near-miss
   memory with clips and trip reports for post-drive review.
4. **Most worthwhile next improvements.** Live input plus a field test, which is the only way to measure real FA/h. Then
   road-geometry context, which addresses the main modelling flaw.
5. **Hardware.** Real time on a mid-range laptop GPU (RTX 3050 4 GB: ~42–45 fps) and on a 6-core laptop CPU without a GPU (~20 fps;
   18 fps on 2 threads). Edge boards are not measured.
6. **To demo and field-test.** Live input + alert channel; a per-vehicle corridor / horizon setup; a fixed model version; a logging
   protocol with driver feedback; ≥ 10 h of normal-driving recordings to measure FA/h. Position it as a research prototype for
   driver assistance and near-miss review, with no braking and no avoidance guarantee.
