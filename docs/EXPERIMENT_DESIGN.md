# Experiment Design

## Data protocol
- Balanced random subset of the Nexar training set: 200 positive + 200 negative videos, sampled with seed 42.
- **Video-level** stratified split, 70 / 15 / 15 (train / val / test), seed 42. No video contributes windows to more than one split.
- Training windows are re-sampled every epoch:
  - positives end at τ_e − h with h ~ U(0.3, 2.0) s;
  - negatives end at random times in normal-driving videos.
- Evaluation windows are deterministic:
  - each positive gives 3 windows, ending at τ_e − {0.5, 1.0, 1.5} s (the Nexar test protocol);
  - each negative gives 3 windows at one random anchor with offsets {0, 0.5, 1.0} s, so both classes share the same clip geometry.
- **Validation is used for all selection**: window length, architecture, early stopping, temperature, classification threshold, risk thresholds, uncertainty threshold. The test split is evaluated once.

## Metric definitions
| Metric | Definition |
|---|---|
| Accuracy / Precision / Recall / F1 | At the F1-optimal threshold on **validation**, applied to test windows |
| ROC-AUC, PR-AUC | Threshold-free, over all test windows |
| AP@h | Average precision of positives at horizon h vs all negative windows |
| **mAP** | Mean of AP@0.5, AP@1.0, AP@1.5 (Nexar challenge metric) |
| Brier | mean (p − y)² |
| ECE | Σ_b (n_b / N) · \|freq_b − conf_b\| over 10 equal-width bins of p |
| Alarm onset | First step of a run of ≥ 3 consecutive steps (0.6 s at stride 0.2 s) with score ≥ threshold. Re-arms after 3 steps below |
| Time-to-alert (TTA) | τ_e − first onset before τ_e (positive videos) |
| Detection rate | Fraction of positive test videos with an onset before τ_e |
| **Timely detection rate** | Fraction of positive test videos with an onset in [τ_e − 3 s, τ_e) |
| Warm-up | No onset possible before t = W (one full observation window) |
| Alarm threshold | Chosen on **validation streams**: maximise timely detection s.t. FA/h ≤ 30. The same rule is used for every system |
| False alarms per hour | Onsets on negative test videos ÷ their total duration (h) |
| Error-detection AUROC | AUROC of an uncertainty score at separating wrong from correct predictions |
| Risk–coverage | Accuracy on the (1 − c) least-uncertain fraction of windows, for coverage c |
| Latency | Wall time per window for model inference (perception throughput is reported separately as frames/s) |

## Models compared
| ID | Model | Purpose |
|---|---|---|
| — | Majority class | Lower bound |
| — | Logistic regression / XGBoost on pooled window features | Classical baselines |
| — | XGBoost on motion features only | Classical, motion only |
| A | Frame-level MLP on the last step (visual + motion) | **No temporal context** |
| B | GRU, visual features only | Temporal, appearance only |
| — | GRU, motion features only | Motion only (RQ1) |
| C | GRU, visual (absolute) + motion | Temporal + motion |
| C2 | GRU, visual-**delta** + motion | Proposed core (design chosen in the validation study) |
| — | GRU, visual-delta only | Visual dynamics alone |
| — | LSTM / TCN / temporal pooling, visual-delta + motion | Architecture selection (on val) |
| D | Selected architecture, 1 model + MC-Dropout + temperature scaling | Uncertainty, single model |
| E | Ensemble ×5 + MC-Dropout ×20 + temperature scaling + risk engine | Full proposed system |

Neural models A–C and the architecture variants are each trained with 3 seeds (42, 43, 44); we report mean ± std.
Ensemble members use seeds 100–104.

## Research questions → evidence
- **RQ1** (motion and temporal context help?): A vs C, B vs C, motion-only, XGB motion-only.
- **RQ2** (earlier warning?): TTA, detection rate and FA/h for E vs C; AP@1.5 vs AP@0.5; the operating curve.
- **RQ3** (uncertainty useful?): ECE/Brier before vs after temperature scaling; error-detection AUROC; risk–coverage for D vs E; the UNCERTAIN level.
- **RQ4** (robustness?): `scripts/run_robustness.py`.

## Validation design study (before the final run; test split not loaded)
`scripts/tune_val.py` → `experiments/latest/val_design_study.json`, and `scripts/diag_val.py` → `val_streaming_diagnostics.json`.
They compared: hard negatives (off / margin 0, 1.5, 3 s), regularisation (lr, wd, input dropout, size), W ∈ {2, 3}, and visual mode
(abs / delta / none), using validation window mAP plus two threshold-free streaming diagnostics:
*temporal AUC* (scores in the 2 s before the event vs scores > 3 s before the annotated alert, within positive videos) and
*video AUC* (per-video peak score). The adopted settings are listed in METHODOLOGY.md §3.

## Robustness protocol
For every test video, the clip that covers its evaluation windows (plus 2 s of tracker warm-up) is decoded and corrupted at the
640-px working resolution. Detection, tracking, flow and CNN are then re-run on the corrupted frames, and the bundle (E) is scored.
`clean` is re-extracted the same way, so every condition shares the same clip boundaries.

| Corruption | Severity 1 | Severity 2 |
|---|---|---|
| frame_drop: grid frame replaced by the previous one | p = 0.25 | p = 0.50 |
| motion_blur: horizontal kernel, px at 1280 width | 9 | 21 |
| low_light: x^1.5 · gain + N(0, 0.01) | gain 0.45 | gain 0.25 |
| occlusion: static black blocks, fraction of area | 10 % | 25 % |
| noise: Gaussian σ on 0–255 | 15 | 35 |

Per-video seeds are deterministic, so the corruptions can be reproduced exactly.

## Reproducibility
`python scripts/prepare_dataset.py && python scripts/extract_features.py && python scripts/run_experiments.py && python scripts/run_robustness.py`.
Each run writes `experiments/<timestamp>/{results.json, train.log, training_histories.json, test_predictions.csv, early_warning_per_video.csv}`
and copies the latest run to `experiments/latest/`. The config, seeds, environment and timings are stored inside `results.json`.

## Protocol v2.1 (lightweight optimisation)
- **Why it changed.** A first lightweight study selected on the fixed 60-video validation split. Validation and test rankings of 54
  configurations barely agreed, and a 400-video cross-validation showed the selected LogReg to be significantly worse than
  gradient boosting (Δ OOF mAP +0.14, 95 % CI [+0.07, +0.22]).
- **New protocol.** 5-fold grouped, label-stratified CV on the 340 train+val videos decides model, features, window, calibration
  method and every threshold, using out-of-fold windows and out-of-fold streaming timelines (1.68 h of normal driving).
  The chosen config is retrained on all 340 videos and evaluated **once** on the unchanged 60-video test set, with the same
  180 windows as v1.
- **Selection rule.** The cheapest configuration with OOF mAP ≥ best − max(0.01, fold SE of the best).
  Cost order: no-CNN < CNN; LogReg < LightGBM < XGBoost < CatBoost; shorter window first.
- **Caveat.** The decision to revise the protocol was taken after v2.0 test numbers had been seen. The v2.1 rule itself never uses test data.
- **Efficiency.** Measured on the same laptop:
  - per-component profile on 1,207 real frames (`profile_perception.py`);
  - end-to-end benchmark on 5 test videos, GPU and CPU-only (`benchmark_pipeline.py`).

## Protocol v3 (early warning, full dataset)
- **Data.** 1,500 videos. The new 1,100 were split before any experiment into dev_new 936 / test_ext 164 (seed 2026). Dev = 1,276,
  held-out = 224. Every choice uses 5-fold grouped CV on dev (fold seed 13); the held-out set is evaluated once.
- **Metrics.** See EARLY_WARNING_V3.md §2. S = mean Recall@{5, 10, 20} FA/h on OOF dev streams.
- **Stages.** A: feature groups (base, +ego, +interactions none / H / F, combinations). B: labels / horizons (0.3–1.5 / 2 / 3 / 4 s,
  alert-aligned, hard negatives, dense negatives, mined negatives) and a 2 s window. C: k ∈ {1, 2, 3, 5} × EMA α ∈ {1, 0.6, 0.3}.
- **Rule.** Keep a change only if the paired video-bootstrap 90 % CI of ΔS > 0. The secondary non-inferiority + mAP rule was declared
  after the 340-video pilot, before the full run, and was not needed.
- **Held-out reporting.** (a) thresholds fixed on dev (deployable); (b) recall at matched held-out FA/h with a paired bootstrap (descriptive).
- **Robustness.** 60 test videos, 13 conditions including camera shake (damped random-walk jitter: 6 / 15 px std at 1280 width,
  plus rotation). The dev shake probe trains on 280 videos and evaluates on 60 val clips, and is what informed the v3-stabilized recommendation.
