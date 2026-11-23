# Ablation Study

Source: `experiments/latest/results.json` (run `20261119_020750`), `bootstrap_ci.json`, validation studies in `val_design_study.json`
and `val_streaming_diagnostics.json`. Test set: 60 held-out videos (30 / 30), 180 windows. Observation window W = 1.0 s
(selected on validation; validation mAP for W = 1 / 2 / 3 / 5 s: 0.554 / 0.550 / 0.538 / 0.552, so window length barely matters).

## A–E (test, Nexar mAP = mean AP at 0.5 / 1.0 / 1.5 s; neural rows are mean ± std over 3 seeds)

| ID | Configuration | mAP | ROC-AUC | F1 | Brier | ECE |
|---|---|---|---|---|---|---|
| A | Frame-level MLP, last step only (visual + motion) | 0.520 ± 0.014 | 0.750 | 0.695 | 0.222 | 0.153 |
| B | GRU, visual (absolute embedding) only | 0.446 ± 0.023 | 0.723 | 0.703 | 0.237 | 0.168 |
| C | GRU, visual (absolute) + motion | 0.451 ± 0.020 | 0.715 | 0.709 | 0.241 | 0.162 |
| C2 | GRU, visual-delta + motion | 0.493 ± 0.106 | 0.696 | 0.625 | 0.231 | 0.083 |
| — | GRU, visual-delta only | 0.538 ± 0.026 | 0.774 | 0.729 | 0.238 | 0.197 |
| — | GRU, motion only | 0.562 ± 0.064 | 0.722 | 0.660 | 0.223 | 0.112 |
| — | LSTM, visual-delta + motion (**selected on val**) | 0.578 ± 0.011 | 0.717 | 0.656 | 0.226 | 0.137 |
| — | TCN, visual-delta + motion | 0.596 ± 0.040 | 0.739 | 0.674 | 0.212 | 0.113 |
| D | LSTM single + MC-Dropout + temperature scaling | 0.565 | 0.692 | 0.656 | 0.235 | 0.125 |
| E | LSTM ensemble ×5 + MC-Dropout ×20 + temperature scaling | **0.582** | 0.731 | 0.719 | 0.218 | 0.128 |

Reference baselines: majority 0.250, logistic regression (pooled) 0.568, XGBoost (pooled) **0.622**, XGBoost (motion only) 0.580.

## Findings
1. **Absolute appearance embeddings do not help temporal models.** Adding motion to them gives no gain (B 0.446 → C 0.451).
   Replacing them with *appearance change* (delta) gives 0.538 (visual-delta only), and motion alone gives 0.562. The validation
   streaming diagnostics show why: absolute embeddings have a temporal AUC of ≈ 0.56 vs ≈ 0.70 for motion-only, meaning they learn
   *what kind of video* rather than *what is happening now*.
2. **Temporal context.** A (no temporal context) at 0.520 outperforms the temporal models fed absolute embeddings (B, C). The
   temporal models using delta + motion (LSTM 0.578, TCN 0.596) are above A. So temporal modelling helps only when the inputs are dynamic.
3. **Ensembling (D → E)** adds +0.017 mAP and +0.04 ROC-AUC over a single member, and improves F1 from 0.656 to 0.719.
4. **C2 has high seed variance** (± 0.106). The LSTM / TCN variants are much more stable (± 0.011 / ± 0.040).
5. **Statistical caution.** Video-level bootstrap 95 % CIs (1,000 resamples) are about ± 0.18 wide. For example E = 0.582 [0.419, 0.765],
   XGBoost = 0.622 [0.417, 0.842]; the paired difference E − XGBoost = −0.040 [−0.197, 0.097], with P(E > XGB) = 0.28.
   **None of the pairwise differences above is statistically significant on this 60-video test set.**

## Validation design study (choices made before the final run)
| Choice | Validation evidence | Adopted |
|---|---|---|
| lr 1e-3 / wd 1e-4 → lr 3e-4 / wd 1e-2 / input dropout 0.3 | best epoch was often 0 (overfitting); val mAP 0.445–0.463 → 0.486–0.513 | yes |
| Hard negatives (margins 0 / 1.5 / 3 s before alert) | val mAP drops (e.g. 0.513 → 0.434 at W = 3); small temporal-AUC gain only for visual models | no |
| Visual delta vs absolute | val mAP 0.486 → 0.572 (W = 2); temporal AUC 0.555 → 0.661 | yes |

## v2 — lightweight study (5-fold CV on train+val, out-of-fold mAP; see LIGHTWEIGHT_OPTIMIZATION.md)
| Factor | Evidence | Conclusion |
|---|---|---|
| Classifier | Train+val CV: LogReg 0.47–0.55; LightGBM 0.54–0.61; XGBoost 0.54–0.61; CatBoost 0.55–0.61. Light GRU 0.49 in the 400-video CV | Gradient boosting > linear ≈ light temporal NN |
| Absolute CNN features | XGB motion 0.614 vs motion + CNN 0.612 (400-video CV); in the train+val CV the best config used CNN (0.611), but within 1 SE of no-CNN configs | No supported gain, so dropped |
| Frame-difference descriptor | LightGBM motion + diff W1 selected (0.592); motion-only variants within noise | Kept: cheap (0.5 ms/frame) and important (6 of the top-10 permutation channels on test are frame-difference cells) |
| Window 1 / 2 / 3 s | Differences within fold SE | Shortest (1 s) |
| Calibration | Platt chosen by CV; test ECE 0.261 → 0.114 | Kept (v1 temperature scaling did not transfer) |
| Bagging ×5 | Gives an uncertainty estimate; entropy error-detection AUROC 0.679 | Kept (4 ms/window on CPU) |

