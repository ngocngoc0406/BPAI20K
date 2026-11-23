# Lightweight Optimisation (v2) — Study and Decision

Goal: find the **lightest architecture that is still effective**, runs stably on a personal computer, and is scientifically
defensible. No v1 algorithm was kept by default.

Sources:
- `experiments/latest/results_light.json` (run `light_cv_20261119_095514`);
- `cv_study.json`, `robustness_light.json`, `perception_profile.json`, `pipeline_benchmark.json`;
- auto-generated tables in [RESULTS_TABLES.md](RESULTS_TABLES.md), Part B.

The v1 code, model and results are archived in `backups/v1_lstm_ensemble/`. The v1 model stays selectable in the app.

## 1. What was evaluated

| Stage | Candidates | Finding | Decision |
|---|---|---|---|
| Detector | YOLOv8n and YOLO11n at input 640 / 480 / 320 | All ≈ 9–12 ms/frame. The cost is overhead-bound, so smaller inputs were **not faster**, but lost 31 % (480) and 64 % (320) of the boxes found at 640. YOLO11n@640 was slower (12.4 vs 9.3 ms). | Keep **YOLOv8n@640** |
| Tracker | ByteTrack (Ultralytics) | Included in the 9.3 ms. Nothing lighter worth the loss of IDs, which the motion cues need. | Keep **ByteTrack** |
| Appearance | ResNet-18 embedding (3.3 ms/frame) vs frame-difference descriptor (0.5 ms/frame) | Under CV, absolute CNN features add nothing: XGB motion 0.614 vs motion + CNN 0.612 OOF mAP over all 400 videos | **Drop ResNet**, use frame difference |
| Classifier | LogReg, LightGBM, XGBoost, CatBoost, light GRU / TCN | See §2 | **LightGBM** |
| Window | 1, 2, 3 s | Within noise | 1 s, the cheapest within tolerance |

## 2. Model selection — and why the protocol changed
**First attempt (v2.0).** Selection used the fixed 60-video validation split, picking the cheapest model within the seed std of the best.
- That rule picked LogReg (motion, W = 3 s).
- Its test mAP was 0.533.
- Validation and test rankings of the 54 tabular configurations barely agreed (`reports/figures`; Evaluation page).

**Check.** A 5-fold grouped cross-validation over all 400 videos (`run_cv_study.py`, ~1,200 out-of-fold windows) showed:
- gradient boosting on motion features clearly beats LogReg: LightGBM 0.618 vs LogReg 0.476 OOF mAP, Δ = +0.142, 95 % CI [+0.066, +0.219];
- absolute CNN features bring no gain (0.612);
- the light GRU on motion + frame difference reaches only 0.491.

**Revised protocol (v2.1, deployed).**
- Every choice uses 5-fold grouped CV on the **340 train+val videos only**: model, features, window, calibration method, alarm and risk thresholds.
- The test set is evaluated once.
- Rule: the cheapest configuration whose OOF mAP ≥ best − max(0.01, fold SE of the best).
- Cost order: no-CNN before CNN; LogReg < LightGBM < XGBoost < CatBoost; shorter window first.

> **Transparency note.** The protocol change was motivated by the v2.0 outcome, and by the all-video CV, which includes the test videos.
> The v2.1 rule itself sees no test data. But because the decision to revise came after test numbers were visible, the v2.1 test
> result should be read with that caveat. All three model versions are reported.

v2.1 selected **LightGBM on motion + frame-difference window features, W = 1 s, no CNN**:
- OOF mAP 0.592; the best was CatBoost + CNN at 0.611, and the tolerance 0.055;
- 384 inputs = 48 channels × 8 temporal aggregates;
- 5 bootstrap-bagged members;
- **Platt** calibration, chosen by CV.

## 3. Held-out test (same 180 windows as v1)
| | **v2.1-light** | v1 LSTM ensemble | v2.0 LogReg | XGBoost (v1 baseline) |
|---|---|---|---|---|
| mAP [95 % CI] | **0.610** [0.43, 0.83] | 0.582 [0.42, 0.77] | 0.533 [0.35, 0.72] | 0.622 [0.42, 0.84] |
| ROC-AUC | **0.834** | 0.731 | 0.726 | 0.828 |
| Precision / Recall / F1 | 0.724 / 0.933 / **0.816** | 0.676 / 0.767 / 0.719 | 0.673 / 0.756 / 0.712 | 0.689 / 0.811 / 0.745 |
| ECE (after calibration) | **0.114** (0.261 raw) | 0.128 | 0.171 | 0.205 |
| Error-detection AUROC (entropy) | **0.679** | 0.588 | 0.624 | — |
| Accuracy 100 % → 50 % coverage | 0.789 → **0.889** | 0.700 → 0.789 | 0.694 → 0.778 | — |

Paired bootstrap Δ mAP (v2.1 − other):
- vs v1: **+0.028** [−0.083, +0.150];
- vs XGBoost: −0.012 [−0.165, +0.115];
- vs v2.0: +0.076 [−0.105, +0.266].

**On window-level mAP, v2.1 is statistically indistinguishable from v1 and from XGBoost, at a fraction of the cost.**
Its ROC-AUC, F1, calibration and uncertainty usefulness are the best observed.

## 4. Streaming early warning (full test videos)
Thresholds were chosen on out-of-fold dev streams: 340 videos, 1.68 h of normal driving vs 18 min for v1.

| System | Detection | Timely (≤ 3 s) | FA / h (test) | Median TTA |
|---|---|---|---|---|
| v2.1 probability alarms | 26.7 % | 13.3 % | 10.0 | 6.95 s |
| v2.1 risk engine | 20.0 % | 6.7 % | 10.0 | 5.35 s |
| v1 probability alarms | 33.3 % | 16.7 % | 10.0 | 7.45 s |
| v1 risk engine | 36.7 % | 20.0 % | 29.9 | 11.2 s |

**Early warning did not improve, and is the system's main weakness.**
- At about 10 false alarms per hour, only 1 in 4 to 1 in 3 events gets any alarm, and fewer get a timely one.
- Differences of 1–2 videos (30 positives) are within noise.
- Test video 00901, timely for v1, is missed by v2.1: its alarms come at 1.8 s, 19.2 s and 30.2 s.
- 00288 is caught by v2.1 at 18.0 s, before the event at 18.92 s.

## 5. Efficiency (same laptop)
| | v2.1-light | v1 |
|---|---|---|
| Prediction latency / window | **4.1 ms on CPU** (5 members), 0.95 ms with 1 member | 73 ms on GPU (5 nets × 20 MC passes) |
| End-to-end throughput, 5 test videos | **42.9 fps on GPU (4.3× real time)**; **20.2 fps CPU-only (2.0× real time)** | 39.0 fps on GPU (3.9×) |
| Peak VRAM | **0.07 GB** (YOLO only) | 0.20 GB |
| CNN weights needed | none besides YOLOv8n (6 MB) | ResNet-18 (45 MB) |
| Prediction model size | 3.7 MB | 1.5 MB + ResNet |

The whole system now runs **in real time on a CPU-only laptop**. Perception (decoding + YOLO) dominates the cost.

## 6. Robustness (identical corrupted clips, full perception re-run)
v2.1 has equal or higher absolute mAP than v1 under every corruption. It is more sensitive to strong low light:

| Condition | v2.1 mAP | v1 mAP |
|---|---|---|
| clean (clip re-extraction) | 0.681 | 0.587 |
| low light, gain 0.45 / 0.25 | 0.602 / 0.547 | 0.612 / 0.593 |
| occlusion 10 % / 25 % | 0.626 / 0.538 | 0.537 / 0.477 |
| Gaussian noise σ = 15 / 35 | 0.723 / 0.671 | 0.575 / 0.540 |

Also:
- Calibration degrades under corruption: ECE up to 0.27 under strong low light.
- Ensemble disagreement (MI) does **not** rise with corruption, as in v1.

## 7. Product calibration of risk levels
CAUTION, derived from window recall ≥ 0.9, covered 49 % of normal-driving time. It was recalibrated on dev OOF streams so that
CAUTION-or-above covers ≤ 20 % of normal driving: threshold 0.318 → 0.686. The WARNING / alarm threshold is unchanged.

On test, normal-driving videos are now 88.8 % NORMAL, 10.5 % CAUTION and 0.5 % WARNING or HIGH RISK. Collision videos show
20.5 % CAUTION and 2.4 % WARNING or HIGH RISK.

## 8. Conclusion
The simplest architecture that matches v1's accuracy is:
- YOLOv8n + ByteTrack;
- 28 image-plane motion cues and a 20-d frame-difference descriptor;
- 8 temporal aggregates per cue;
- a bagged LightGBM with Platt calibration.

It removes the CNN backbone and the neural temporal model, runs in real time on CPU, and is better calibrated.
- Heavier models — LSTM ensemble, CatBoost, CNN features — showed **no statistically supported improvement**.
- The bottleneck is no longer the model but the **signal**: streaming early warning at a usable false-alarm rate remains poor.
  More data (the full 1,500 videos), ego-motion compensation and better object-interaction cues are the most promising next steps.
