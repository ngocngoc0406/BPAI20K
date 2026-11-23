# Early-Warning Optimisation (v3)

> **Audit correction (2026-11-21, see [SAFEDRIVE_V3_AUDIT.md](SAFEDRIVE_V3_AUDIT.md)).**
> 1. On held-out at matched false-alarm rates, v3 is **not** better than the v2 configuration retrained on the same 1,276 videos:
>    ΔS −0.003 [−0.053, +0.044]. The held-out gain over the deployed v2-light (+0.030, n.s.) is attributable to more training data.
>    The v3 feature and label changes are supported on dev only.
> 2. Product numbers measured with the deployed code: recall 11.6 % [6.9, 18.9] at **8.5 FA/h** [4.1, 15.6]. The 11.3 FA/h in §5a came
>    from a different calibration aggregation (audit M1), now aligned in `run_ew_study.py`.

**Goal:** improve collision warning in **continuous (streaming) operation** under a false-alarm budget, rather than window-level mAP.
The default LightGBM family is kept unless an experiment shows something better.

**Sources:**
- `experiments/latest/results_ew_full.json` (run `ew_full_20261120_004859`);
- `ew_heldout_compare.json`, `robustness_v2_v3caches.json`, `robustness_v3.json`, `robustness_v3_stabilized.json`;
- `shake_dev_probe.json`, `ew_negatives_probe_orig.json`, `pipeline_benchmark_v3.json`;
- tables: [RESULTS_TABLES.md](RESULTS_TABLES.md), Part C.

**Backup:** v2-light (code, bundle, results) is in `backups/v2_light_20261119/`. v2-light and v1 remain selectable in the app.

## 1. Data
- **Dataset.** All 1,500 Nexar training videos (24 GB, Nexar Open Data License). All 1,100 new videos are readable, with no annotation inconsistencies.
- **Split** (pre-declared, by video; see DATASET_REPORT.md):
  - the original 400-video split is unchanged;
  - the new videos are split 85 / 15 (label-stratified, seed 2026);
  - **dev = 1,276 videos** (5-fold grouped CV) and **held-out = 224 videos** (original test 60 + test_ext 164), never used for any choice.
- **Causality.** No future frame is used: windows are causal, timelines are evaluated from t = 0, and nothing is observed at or after the event.

## 2. Metrics (defined before the experiments; `src/evaluation/streaming.py`)
- **Valid alarm:** an onset (≥ 3 consecutive 0.2 s steps above threshold) within [t_event − 5 s, t_event).
  The longest annotated alert lead is 4.5 s, so earlier onsets are not anticipations.
- **Recall@B:** fraction of collision videos with a valid alarm, at the threshold giving ≤ B false alarms per hour on normal driving (B = 5, 10, 20).
- **EDR@X:** fraction of collision videos with a valid alarm at or before t_event − X (X = 1, 2, 3 s).
- **Primary selection score:** S = mean(Recall@5, @10, @20) on out-of-fold dev streams.
- **Acceptance rule.** A change is kept only if the paired video-bootstrap 90 % CI of ΔS is above 0. A secondary rule was declared after
  the 340-video pilot and before the full run: ΔS non-inferior (CI lower bound > −0.02) **and** OOF mAP better (95 % CI > 0).
  The secondary rule was never needed.

## 3. What was implemented
| Component | Implementation | Code |
|---|---|---|
| Ego-motion | KLT corners on background (road users and hood masked) → RANSAC homography; and dense Farnebäck flow → RANSAC partial affine. Stats per step: pan, tilt, zoom (forward motion), roll, inlier ratio, residual. ~4 ms/frame | `src/features/ego_motion.py` |
| Scene-relative motion | Box at t−k mapped through the accumulated camera transform; residual centre velocity and residual scale growth = motion not explained by the camera | `src/features/relational.py` |
| Vehicle interactions | Corridor occupancy; lead gap and its rate of change; lead "brake" proxy (increase of growth rate); cut-in score/count (objects beside the corridor moving inward); pairwise image-based time-to-meet; converging pairs; relative heading; image-based time-to-reach of the ego point. All image-plane. **No physical distance or TTC is claimed.** | `src/features/relational.py` |
| Label schemes | Horizons 0.3–1.5 / 2 / 3 / 4 s; alert-aligned positives; hard negatives before the annotated alert; **dense negatives** (40 windows per normal video); dense + mined negatives | `src/prediction/windows.py`, `scripts/run_ew_study.py` |
| Alarm post-processing | k ∈ {1, 2, 3, 5} consecutive steps × causal EMA α ∈ {1, 0.6, 0.3} | `src/evaluation/streaming.py` |
| Camera-shake corruption | Damped random-walk jitter (translation + rotation), 2 severities | `src/preprocessing/video_transform.py` |

## 4. Results on dev (1,276 videos, OOF streams; Δ vs reference, 90 % CI)
| Stage | Change | ΔS | Kept |
|---|---|---|---|
| A | + camera ego-motion stats | +0.003 [−0.016, +0.022] | no |
| A | **+ interaction cues (uncompensated)** | **+0.025 [+0.001, +0.049]** | **yes (selected)** |
| A | + interactions, homography-compensated | +0.010 [−0.014, +0.032] | no |
| A | + interactions, flow-compensated | +0.016 [−0.006, +0.039] | no |
| A | + ego(H) + interactions(H) | +0.032 [+0.007, +0.059] | passed; not selected (lower point S + ΔmAP) |
| A | + ego(F) + interactions(F) | +0.013 [−0.011, +0.037] | no |
| B | horizon 0.3–1.5 s | +0.005 [−0.013, +0.024] | no |
| B | horizon 0.3–3.0 s / 0.3–4.0 s | −0.027 / −0.040 (CI < 0) | no: **significantly worse** |
| B | alert-aligned positives | +0.014 [−0.005, +0.036] | no |
| B | hard negatives (L0 / L2 horizon) | +0.022 / +0.016 (CI includes 0) | no |
| B | **dense negatives (40 / video)** | **+0.023 [+0.003, +0.044]** | **yes** |
| B | dense + mined negatives | +0.009 [−0.010, +0.030] | no |
| B | window 2 s | −0.013 [−0.035, +0.014] | no |
| C | k ∈ {1, 2, 5}, EMA | all below k = 3 / no EMA | no |

**Selected (v3):** motion + frame difference + uncompensated interaction cues, W = 1 s, dense-negative training, k = 3, no EMA.
On dev: S 0.101 → 0.166; Recall@5 / 10 / 20 = 0.000 / 0.114 / 0.190 → 0.099 / 0.166 / 0.234.

The 340-video pilot found no significant change. Its OOF data contained only ~0.85 h of normal driving, too little to measure
recall at 5–20 FA/h reliably. The full dataset (~6 h of normal driving in dev) made the effects measurable.

## 5. Held-out (224 videos, 1.15 h normal driving)
**(a) Thresholds fixed on dev streams** (the deployable setting):

| | v2 config retrained on 1,276 videos | **v3** |
|---|---|---|
| mAP / ROC-AUC / ECE | 0.607 / 0.809 / 0.055 | **0.620 / 0.816 / 0.028** |
| target 5 FA/h → recall (actual FA/h) | 0.000 (0.0) | **0.098 (2.6)** |
| target 10 FA/h → recall (actual FA/h) | 0.196 (16.5) | 0.116 (11.3) |
| target 20 FA/h → recall (actual FA/h) | 0.259 (30.5) | 0.196 (17.4) |

v3's dev-chosen thresholds transfer much better: its actual FA/h stays close to target, while the baseline overshoots 1.5–1.7×.
This is the expected effect of dense-negative training.

**(b) At matched false-alarm rates** (each model's own held-out curve; descriptive):

| Model | Recall@5 / 10 / 20 | S | EDR@1 / 2 / 3 s (10 FA/h) |
|---|---|---|---|
| v2-light deployed (400 videos) | 0.036 / 0.116 / 0.179 | 0.110 | 0.071 / 0.027 / 0.000 |
| v2 config, 1,276 dev videos | 0.062 / 0.170 / 0.196 | 0.143 | 0.080 / 0.045 / 0.018 |
| **v3** | **0.107** / 0.116 / **0.241** | **0.155** | 0.062 / 0.027 / 0.027 |

Paired ΔS for v3 − v2-light deployed: +0.030 [−0.034, +0.092], P = 0.78. **On 224 held-out videos none of the streaming differences is
statistically significant.** The direction agrees with dev, but the held-out set is too small to confirm effects of 2–3 recall points.

**Absolute level.** Early warning remains weak. At a usable false-alarm rate, about 1 in 10 to 1 in 4 collisions gets a valid alarm,
and fewer than 5 % get one ≥ 2 s ahead. Window-level ranking is good (ROC-AUC 0.82), but separating the seconds before a collision from
the busiest moments of normal driving needs more specific evidence than image-plane cues provide.

## 6. Ego-motion compensation and camera shake
- **Stable footage (dev streams).** Compensation brings no streaming gain over uncompensated interaction cues (A3/A4 vs A2).
- **Dev shake probe** (`shake_dev_probe.json`; train 280 → val 60 clips; mAP): compensated interaction features stay ahead under shake
  (shake-1: rel(H) 0.499, ego+rel(F) 0.544 vs base 0.426), and are best under 50 % frame drop and strong low light.
- **Test robustness** (identical corrupted clips, full perception re-run; mAP):

  | | clean | shake-1 | shake-2 | low light-2 | occlusion-2 | frame drop-2 |
  |---|---|---|---|---|---|---|
  | v2-light | 0.683 | 0.416 | 0.502 | 0.537 | 0.540 | 0.518 |
  | **v3** | 0.805 | 0.421 | 0.487 | 0.758 | 0.692 | 0.588 |
  | **v3-stabilized** (ego(H) + compensated interactions) | **0.825** | **0.637** | **0.596** | **0.803** | 0.690 | **0.613** |

**Conclusion.**
- Ego-motion compensation is what makes the system robust to camera shake: +0.10 to +0.22 mAP over v3.
- On stable footage it does not improve streaming early warning, and v3-stabilized's dev recall is lower (Recall@5 / 10 / 20 =
  0.080 / 0.080 / 0.182, vs 0.099 / 0.166 / 0.234).
- So **v3 is the default** and **v3-stabilized is offered for poorly mounted or vibrating cameras**. That choice rests on the dev shake
  probe; the test robustness numbers are reported only.

## 7. Efficiency (5 test videos, end-to-end)
| Pipeline | GPU fps | CPU-only fps | Prediction ms/frame | Peak VRAM |
|---|---|---|---|---|
| **v3** | 42.0 (4.2× real time) | 19.3 (1.9×) | 0.38 | 0.07 GB |
| v3-stabilized | 34.0 | 17.3 | 0.40 | 0.07 GB |
| v2-light | 42.7 | 19.5 | 0.04 | 0.07 GB |

Interaction cues cost about 0.3 ms per frame. Ego-motion costs about 4 ms per frame (≈ −20 % throughput). Every variant runs in real time on CPU.

## 8. Product changes
- New default model `models/safedrive_v3`; optional `models/safedrive_v3_stabilized`; v2-light and v1 are kept in the selector.
- Risk levels are now **false-alarm-budgeted on dev streams**: CAUTION at 20 FA/h, WARNING (alarm) at 10 FA/h, HIGH RISK at 5 FA/h,
  applied to the calibrated probability (`level_signal = probability`). The trend/looming risk score is still shown as a separate quantity.
- UNCERTAIN: ensemble mutual information above its 90th percentile on dev streams.
- The extractor computes ego-motion only when the selected model needs it.
- XAI page: descriptions and attribution groups for camera-motion and interaction cues. A bug in temporal occlusion with ego/interaction
  features was found by an end-to-end check, fixed, and covered by a regression test.
- Demos:
  - `reports/demo_annotated_00288_v3.mp4`: one valid alarm 3.7 s before the collision;
  - `reports/demo_annotated_00901_v3.mp4`: collision missed, kept to show the limitation.

## 9. Bottom line
On 1,276 dev videos, two changes improved streaming early warning with statistical support:
- vehicle-interaction cues: +2.5 points of mean recall;
- dense-negative training: +2.3 points, and much better transfer of false-alarm thresholds.

Ego-motion compensation did not help on stable footage but strongly improves robustness to camera shake. Held-out results point the
same way but are not significant, and absolute early-warning performance remains low. The most promising next steps target
specificity: per-object (not frame-level) risk heads, depth / scale-aware cues, and video foundation-model features distilled to a
lightweight student.
