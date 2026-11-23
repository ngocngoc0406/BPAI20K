# Results and Discussion

> **Update (v2, lightweight optimisation).** The deployed model is now *v2-light*: LightGBM on motion + frame-difference
> features, with no CNN. Test mAP 0.610 vs 0.582 for the v1 LSTM ensemble below (difference not significant). ROC-AUC 0.834,
> F1 0.816, ECE 0.114. End-to-end 20 fps on CPU. See [LIGHTWEIGHT_OPTIMIZATION.md](LIGHTWEIGHT_OPTIMIZATION.md) and
> [RESULTS_TABLES.md](RESULTS_TABLES.md), Part B. The sections below document the v1 study, kept for comparison.

Every number below comes from these experiment files:
- `experiments/latest/results.json` (run `20261119_020750`, RTX 3050 4 GB, 4.0 min of training and evaluation);
- `experiments/latest/robustness.json`;
- `experiments/latest/bootstrap_ci.json`.

The complete auto-generated tables are in [RESULTS_TABLES.md](RESULTS_TABLES.md), and the figures in `reports/figures/`.
Data: 400 Nexar videos, split by video into 280 / 60 / 60. The test set has 30 collision / near-miss and 30 normal videos,
giving 180 evaluation windows that end 0.5, 1.0 and 1.5 s before the event.

## 1. Headline numbers (held-out test)
| System | mAP [95 % CI] | ROC-AUC | F1 | ECE |
|---|---|---|---|---|
| Majority class | 0.250 | 0.500 | 0.667 | — |
| Logistic regression, pooled features | 0.568 [0.386, 0.814] | 0.795 | 0.731 | 0.107 |
| **XGBoost, pooled features** | **0.622** [0.417, 0.842] | 0.828 | 0.745 | 0.205 |
| XGBoost, motion only | 0.580 [0.401, 0.876] | **0.845** | **0.774** | 0.201 |
| **SafeDrive E**: LSTM ensemble ×5, visual-delta + motion, MC-Dropout, TS | 0.582 [0.419, 0.765] | 0.731 | 0.719 | 0.128 |

**The proposed neural system does not outperform the gradient-boosted baseline on window-level discrimination.**
The paired bootstrap difference E − XGBoost is −0.040, with 95 % CI [−0.197, 0.097]. With 60 test videos, the CIs are about ± 0.18 wide,
so no model in this study is significantly better than another, except that all models clearly beat the majority baseline (0.25).
What SafeDrive adds over XGBoost is not accuracy. It adds a calibrated ensemble with uncertainty decomposition,
a causal streaming timeline, a graded risk policy with abstention, and gradient-based explanations.

## 2. RQ1 — Do motion and temporal context help?
**Yes for motion and appearance change; no for absolute appearance.** See [ABLATION_STUDY.md](ABLATION_STUDY.md).
- GRU with absolute visual embedding: 0.446. Adding motion: 0.451. Motion only: 0.562. Visual-delta only: 0.538.
- LSTM / TCN on visual-delta + motion: 0.578 / 0.596.
- Permutation importance (test AP drop): the visual-delta block matters most (0.131), followed by road-ahead flow magnitude (0.014),
  flow expansion (0.014), lead-object bottom position (0.013), apparent τ (0.010), lead offset, acceleration and looming (0.006–0.008).
- Interpretation: frozen ImageNet embeddings of single frames mostly encode scene and camera identity. That identity is spuriously
  predictive at window level but hurts temporal specificity (validation temporal AUC ≈ 0.56 vs ≈ 0.70 for motion-only).
  Motion and appearance-change signals generalise better.
- Window length matters little (validation mAP 0.538–0.554 for W = 1–5 s).

## 3. RQ2 — Early warning in streaming use (full test videos)
The alarm threshold was selected on **validation streams** to maximise timely detection with ≤ 30 false alarms / h, using
3 consecutive steps (0.6 s) and a W-second warm-up.

| System (threshold from val) | Detection before event | Timely (≤ 3 s before) | Mean / median TTA | False alarms / h |
|---|---|---|---|---|
| E + risk engine (0.88) | 36.7 % | 20.0 % | 9.8 / 11.2 s | 29.9 |
| E probability only (0.83) | 33.3 % | 16.7 % | 8.3 / 7.5 s | 10.0 |
| Single LSTM, deterministic (0.80) | 20.0 % | 10.0 % | 7.3 / 6.8 s | 6.6 |

- At a practical false-alarm level, **most events are missed**. Many of the alarms that do fire come far earlier than the event:
  the median alarm comes 10.1 s before the annotated alert time, so these are not specific anticipations. 26 of 30 normal test videos
  produced no alarm. The FA/h estimate rests on only 18 min of normal driving: 9 alarms for E + risk engine.
- The operating curve (`reports/figures/operating_curve.png`) shows the tradeoff. Detection rises to about 90 % only at more than 200 FA/h.
- Window-level AP decreases with horizon for E (0.627 at 0.5 s, 0.601 at 1.0 s, 0.517 at 1.5 s): earlier is harder, as expected.
- The risk engine (trend + looming bonus) raises detection over probability-only (36.7 % vs 33.3 %, timely 20.0 % vs 16.7 %) at the cost of
  more false alarms (29.9 vs 10.0 / h). That is a tradeoff, not a strict gain.
- Qualitative example (test video 00901, `reports/demo_annotated_00901.mp4`): WARNING at 16.2 s, 2.7 s before the annotated collision
  at 18.93 s. The triggering factors are apparent acceleration, pairwise closing rate and lateral motion toward the ego lane.

## 4. RQ3 — Is the uncertainty useful?
- **Calibration.** Ensemble ECE is 0.102 before temperature scaling, and **0.128 after** (T = 0.60, fitted on 180 validation windows).
  The single model went from 0.121 to 0.125 (T = 1.03). Temperature scaling **did not transfer** from validation to test. This is a
  negative result, attributable to the small validation set. The un-scaled ensemble is already better calibrated than XGBoost (0.205).
- **Error detection.** AUROC for separating wrong from correct predictions: total entropy 0.588, p-std 0.555, **mutual information 0.501 (chance)**.
- **Selective prediction** (ensemble). Abstaining on the 50 % most uncertain windows by *entropy* raises accuracy from 0.700 to 0.789.
  By *MI*, it only reaches 0.711.
- **Conclusion.** Total predictive uncertainty is moderately useful for abstention. The epistemic component (MI), which drives the
  UNCERTAIN state, was **not** shown to identify errors in this data. On test timelines, UNCERTAIN covered 8.0 % of steps
  (885 of 11,046).

## 5. RQ4 — Robustness (full perception pipeline re-run on corrupted clips)
| Condition | mAP | F1 | ECE | mean MI |
|---|---|---|---|---|
| clean (re-extracted) | 0.587 | 0.701 | 0.108 | 0.104 |
| frame drop 25 % / 50 % | 0.583 / 0.561 | 0.689 / 0.687 | 0.126 / 0.171 | 0.098 / 0.092 |
| motion blur k = 9 / 21 | 0.561 / 0.528 | 0.681 / 0.609 | 0.127 / 0.210 | 0.101 / 0.102 |
| low light gain 0.45 / 0.25 | 0.612 / 0.593 | 0.742 / 0.698 | 0.133 / 0.160 | 0.098 / 0.094 |
| occlusion 10 % / 25 % | 0.537 / **0.477** | 0.644 / 0.616 | 0.132 / 0.175 | 0.097 / 0.096 |
| Gaussian noise σ = 15 / 35 | 0.575 / 0.540 | 0.685 / **0.468** | 0.147 / **0.297** | 0.108 / 0.114 |

- The system degrades gracefully under frame dropping and low light: no measurable drop, given CIs of about ± 0.18.
- Static occlusion of 25 % of the image (−0.11 mAP) and strong noise (F1 collapse, ECE 0.30) are the main weaknesses.
- Calibration worsens under every corruption, yet **mean MI does not rise** (0.092–0.114 vs 0.104 clean). The epistemic signal does
  not warn about degraded input, which reinforces the RQ3 finding.

## 6. Efficiency (RTX 3050 Laptop 4 GB)
- Perception (decode, YOLOv8n, ByteTrack, Farnebäck flow, ResNet-18) runs at **40.2 frames/s** on the 10 Hz grid, i.e. about 4× real time.
- Prediction latency: 1.2 ms per window for a single model; 73 ms per window for the ensemble ×5 × MC 20 (100 passes).
  At a 0.2 s stride, this fits real time.
- Training the full study (36 models) took 4.0 min. Peak training VRAM was 0.11 GB, and process RSS 4.2 GB.

## 7. Discussion
- **What worked.** It is a complete, reproducible, causal pipeline on real dashcam data. The finding that appearance *change* plus
  motion beats absolute appearance held consistently on both validation and test. The ensemble improved over single models.
  Explanations rest on measured cues. The system runs in real time on a laptop.
- **What did not.** Window discrimination (mAP ≈ 0.58) is moderate and not above a pooled-feature XGBoost. Streaming early warning
  at a usable false-alarm rate detects only about a third of events, and few of them timely. Temperature scaling and the MI-based
  UNCERTAIN state did not deliver their expected benefits on test.
- **Why (hypotheses for future work).**
  - The data is small: 280 training videos.
  - The visual encoder is frozen and frame-level.
  - Motion is image-plane only, with no ego-motion compensation.
  - Positive videos contain long stretches of normal driving, so the model must learn a very local signal.

  Recent work on the same dataset uses large video foundation models trained end-to-end (BADAS, arXiv 2510.14876).
- **Claims we do *not* make.** We do not claim novelty of uncertainty-aware or explainable anticipation (UString, DRIVE), nor
  superiority over baselines, nor physical time-to-collision.
