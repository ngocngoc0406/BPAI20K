# SafeDrive AI V4: Experiment Plan and Results

All experiments use the v3 protocol unchanged, and the hypotheses and acceptance rules were declared **before** running. Every
result is reported, including those that show no improvement.

## Verdicts
| Hypothesis | Test | Verdict |
|---|---|---|
| **H1** Traffic knowledge + dynamic scene graph improve interaction representation and risk prediction | Ablation of SG / KG feature fusion, graph-only models and a temporal GNN against v3 (§3, §4) | **Not supported.** No configuration beats v3. Graph-only models and the GNN are clearly worse |
| **H2** Uncertainty-aware graph reasoning limits the effect of perception errors | Train on clean data; OOF evaluation on clean and on perception-error tracks; paired bootstrap of the degradation difference A3 vs A2 (§5) | **Not supported.** ΔS 90 % CI [−0.035, 0.004]; ΔmAP ≈ 0 |
| **H3** Evidence-grounded XAI is more faithful / verifiable than feature importance alone | Faithfulness, stability, validity, integrity and completeness metrics on 145 held-out warnings (§6) | **Partly supported.** See §6 |
| **H4** Adaptive warning reduces false alarms while keeping detection | Policies compared at equal false-alarm budgets; parameters chosen on dev, held-out evaluated once (§7) | **Not supported.** Evidence gating lowers recall; the reliability gate chosen on dev changes nothing on held-out |

**Decision.** The product predictor stays the frozen v3 model. V4 contributes:
- **What V4 adds to the product:** the explanation and evidence layer, the uncertainty states (INSUFFICIENT_EVIDENCE,
  cautious HIGH, DEGRADED_INPUT), Near-Miss Intelligence and the single app.
- **These do not improve detection.** Their value is transparency, honest uncertainty and reviewability. That value is
  measured in §6; it is not a gain in detection.

## 1. Protocol
- **Data.** Nexar dashcam collision prediction (Moura et al. 2025).
  - Dev: 1,276 videos, in 5 video-level folds (seed 13).
  - Held-out: 224 videos (test 60 + test_ext 164), evaluated once per reported configuration and never used for a choice.
- **Split level.** Videos, never frames.
- **Labels.** L7: positives end 0.3–2.0 s before the event, with 40 dense negatives per normal video. W = 1 s, k = 3, no EMA.
- **No future information.** Features at t use only steps ≤ t (tested).
- **Metrics.**
  - Window: Nexar mAP (AP at 0.5 / 1.0 / 1.5 s), ROC-AUC, PR-AUC, precision / recall / F1, Brier, ECE.
  - Streaming: recall at 5 / 10 / 20 FA/h (thresholds from dev OOF streams), EDR@1/2/3 s (early-warning recall), median lead
    time; missed-event rate = 1 − recall.
- **Primary score.** S = mean recall@{5, 10, 20} FA/h on dev OOF streams.
- **Acceptance rule ("better than A0").** The paired video-bootstrap 90 % CI of ΔS is > 0, or S is non-inferior (CI lower bound
  > −0.02) **and** the paired ΔmAP 95 % CI is > 0.
- **Reproduction guard.** A0 must reproduce v3 exactly, and it does:
  - OOF mAP 0.6170491 and recall@5/10/20 of 0.0987 / 0.1661 / 0.2335;
  - held-out mAP 0.6198, ROC-AUC 0.816, ECE 0.028 and recall@10 0.116 at 8.70 FA/h.

  The H2 / H4 script stops if A0 does not reproduce. It caught one bug: one random generator per draw instead of a shared one.
  The invalid run was discarded and rerun.

## 2. Configurations (numbering of the V4 product prompt)
| ID | Prediction input | Model |
|---|---|---|
| A0 | v3 features (motion + frame difference + interaction cues, 528) | LightGBM (v3 recipe) |
| A1 | A0 + dynamic scene graph (SG, 288) | LightGBM |
| A2 | A0 + knowledge-graph reasoning (KG, plain confidences, 120) | LightGBM |
| A3 | A0 + uncertainty-aware KG reasoning (KGU, subjective logic, 136) | LightGBM |
| A4 | A0 + SG + KG (936) | LightGBM |
| A5 | A0 + SG + KGU (KG-guided prediction, 952) | LightGBM |
| A6 | product: v3 predictor + KGU evidence + uncertainty states + XAI + adaptive warning | frozen v3 + V4 layers |
| B-* | baselines on v3 features: logistic regression, random forest, XGBoost, motion-only; temporal R-GCN + GRU (direction B) | — |
| W0.5 / W2.0 | A4 with graph windows of 0.5 / 2.0 s | LightGBM |

Direction A (feature fusion) is A1–A5. Direction B is the GNN. Direction C (hybrid: graph + KG reasoning + temporal predictor +
uncertainty) is A5 / A6.

## 3. Dev OOF results (1,276 videos)
Runs:
- `experiments/v4/ablation_20261123_182909/` (round 1)
- `experiments/v4/hypotheses_20261123_195013/` (round 2: A3, A5, H2, H4)
- `experiments/v4/gnn_oof.json`

| Config | mAP | ROC-AUC | S | R@5 | R@10 | R@20 | EDR@2s (10 FA/h) | ΔS 90 % CI | ΔmAP 95 % CI | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| **A0 v3** | 0.617 | 0.823 | **0.166** | 0.099 | 0.166 | 0.234 | 0.036 | — | — | reference |
| A1 +SG | 0.622 | 0.825 | 0.112 | 0.000 | 0.100 | 0.237 | 0.013 | [−0.035, 0.006] | [−0.009, 0.018] | no improvement |
| A2 +KG | 0.624 | 0.822 | 0.119 | 0.083 | 0.083 | 0.191 | 0.011 | [−0.029, 0.004] | [−0.003, 0.017] | no improvement |
| A3 +KGU | 0.614 | 0.825 | 0.105 | 0.000 | 0.097 | 0.218 | 0.014 | [−0.016, 0.018] | [−0.016, 0.009] | no improvement |
| A4 +SG+KG | 0.621 | 0.824 | 0.099 | 0.000 | 0.089 | 0.207 | 0.013 | [−0.036, 0.001] | [−0.011, 0.017] | no improvement |
| A5 +SG+KGU | 0.613 | 0.823 | 0.136 | 0.099 | 0.099 | 0.210 | 0.014 | [−0.024, 0.017] | [−0.017, 0.009] | no improvement |
| SG + KG only | 0.510 | 0.741 | 0.084 | 0.041 | 0.085 | 0.127 | 0.020 | [−0.105, −0.041] | [−0.145, −0.066] | worse |
| KG only | 0.422 | 0.677 | 0.052 | 0.011 | 0.039 | 0.107 | 0.014 | [−0.141, −0.071] | [−0.235, −0.153] | worse |
| B-logreg | 0.544 | 0.763 | 0.119 | 0.069 | 0.124 | 0.165 | 0.025 | [−0.068, −0.005] | [−0.107, −0.039] | worse |
| B-rf | 0.620 | 0.821 | 0.113 | 0.000 | 0.114 | 0.224 | 0.014 | [−0.026, 0.025] | [−0.017, 0.025] | no improvement |
| B-xgb | 0.616 | 0.822 | 0.107 | 0.002 | 0.097 | 0.223 | 0.014 | [−0.019, 0.017] | [−0.012, 0.009] | no improvement |
| B-motion-only | 0.617 | 0.822 | 0.086 | 0.002 | 0.072 | 0.185 | 0.016 | [−0.060, −0.010] | [−0.020, 0.021] | worse (S) |
| B-GNN (R-GCN + GRU) | 0.478 | 0.671 | 0.043 | 0.000 | 0.038 | 0.092 | — | [−0.152, −0.083] | [−0.177, −0.096] | worse |
| W0.5 | 0.612 | 0.820 | 0.111 | 0.000 | 0.091 | 0.243 | 0.013 | [−0.023, 0.015] | [−0.017, 0.006] | no improvement |
| W2.0 | 0.611 | 0.818 | 0.086 | 0.000 | 0.078 | 0.179 | 0.013 | [−0.046, −0.007] | [−0.019, 0.008] | worse (S) |

Calibrated OOF Brier and ECE are 0.171–0.175 and 0.012–0.024 for all LightGBM configurations: calibration is not a
differentiator. The tables of round 1 also list PR-AUC, precision, recall and F1 (`ablation_oof.json`).

## 4. Held-out results (224 videos, evaluated once)
| Config | mAP | ROC-AUC | ECE | R@5 (FA/h) | R@10 (FA/h) | R@20 (FA/h) | paired ΔmAP vs A0 (95 % CI) |
|---|---|---|---|---|---|---|---|
| **A0 = v3 = A6 predictor** | **0.620** | 0.816 | 0.028 | 0.089 (2.6) | 0.116 (8.7) | 0.179 (16.5) | — |
| A1 | 0.606 | 0.818 | 0.050 | 0.000 (0.0) | 0.071 (3.5) | 0.170 (11.3) | −0.014 [−0.037, 0.007] |
| A2 | 0.611 | 0.814 | 0.038 | 0.063 (2.6) | 0.063 (2.6) | 0.143 (11.3) | −0.009 [−0.029, 0.007] |
| A3 | 0.613 | 0.817 | 0.030 | 0.000 (0.0) | 0.062 (1.7) | 0.188 (10.4) | −0.007 [−0.025, 0.011] |
| A4 | 0.597 | 0.813 | 0.055 | 0.000 (0.0) | 0.080 (4.4) | 0.152 (13.9) | −0.023 [−0.050, 0.003] |
| A5 | 0.599 | 0.813 | 0.036 | 0.062 (4.4) | 0.062 (4.4) | 0.152 (13.1) | −0.020 [−0.050, 0.004] |
| SG + KG only | 0.436 | 0.701 | 0.036 | 0.018 (0.0) | 0.018 (4.4) | 0.045 (7.8) | −0.183 [−0.264, −0.098] |

**Early warning stays below 5 %** at 2 s for every configuration. With v3, about 88 % of events are missed at about 9 FA/h.

## 5. H2: perception errors (dev OOF, models trained on clean folds)
**Error model** (track level):
- 15 % detection dropout;
- box jitter of 3 % of the box size;
- 2 % id switches per step;
- 5 % ghost detections per step.

The v3 motion features are recomputed from the noisy tracks; pixel channels are unchanged.

| Config | S clean → noisy | ΔS | mAP clean → noisy | ΔmAP |
|---|---|---|---|---|
| A0 v3 | 0.166 → 0.106 | −0.060 | 0.617 → 0.588 | −0.029 |
| A2 KG (plain) | 0.119 → 0.117 | −0.003 | 0.624 → 0.590 | −0.034 |
| A3 KG (uncertainty-aware) | 0.105 → 0.115 | +0.010 | 0.614 → 0.581 | −0.033 |

**Paired bootstrap of D(A3) − D(A2):**
- S: mean −0.015, 90 % CI [−0.035, 0.004];
- mAP: mean 0.000, 90 % CI [−0.009, 0.010].

**H2: not supported.** Uncertainty-aware reasoning does not make the predictor measurably more robust than plain KG features.

The KG-feature models lose less S under noise than A0 (A0 − A2: CI [−0.029, 0.007]). That difference is **not significant**
either, and they start from a lower clean S.

## 6. H3: evidence-grounded explanations (held-out, 145 warnings of the v3 predictor)
| Metric | Feature importance (SHAP) | Graph / KG evidence |
|---|---|---|
| Exactness / validity | local accuracy error 1.2e-14 | 100 % of 112 chain steps reference a real node, edge and timestamp |
| Faithfulness to the model | deletion: top-10 features lower the logit by 1.32 vs 0.01 for random (100 % of warnings) | **not established**: the v3 predictor does not use KG features, so a chain is observed evidence consistent with the warning, not the model's computation |
| Stability | single features: Spearman 0.06 (unstable); groups: 0.90 | deterministic given the tracks (same input, same chain) |
| Verifiability by a person | a feature value is hard to check against the video | each step points to a track, a frame time and a measured value; trace hash and references verify in 100 % |
| Completeness | 100 % of warnings have SHAP | **47 %** of warnings have a KG chain |
| Counterfactual validity | — | 100 % of 60 (identity reproduction exact); closing speed halved lowers risk in 95 % |

**H3: partly supported.**
- **Supported:** graph evidence is **more verifiable** than feature importance (every claim checkable against frames and
  tracks), and SHAP is faithful.
- **Not supported:** that graph explanations are more **faithful** to the model. They are not a description of the v3 model,
  and they cover only half of the warnings. No human evaluation was possible (no trained annotators yet).

## 7. H4: adaptive warning (v3 predictor; policies at equal false-alarm budgets)
| Policy (parameter chosen on dev) | dev S | held-out S | R@5 (FA/h) | R@10 (FA/h) | R@20 (FA/h) | EDR@2s @10 | Coverage |
|---|---|---|---|---|---|---|---|
| P0 v3 rule | 0.166 | 0.128 | 0.089 (2.6) | 0.116 (8.7) | 0.179 (16.5) | 0.027 | 100 % |
| P1 KG evidence gate (E ≥ 0.20) | 0.158 | 0.116 | 0.080 (3.5) | 0.107 (11.3) | 0.161 (15.7) | 0.018 | 100 % |
| P2 reliability gate (≥ 0.40) | 0.164 | 0.128 | identical to P0 | identical | identical | 0.027 | 90.5 % |

**H4: not supported.**
- Requiring KG evidence removes true warnings as well as false ones, so it lowers recall at equal false alarms.
- The reliability gate chosen on dev covers 90.5 % of steps and changes no alarm on held-out.
- **Product setting.** The gates stay **off**. INSUFFICIENT_EVIDENCE and the cautious HIGH flag are display states that do not
  change the v3 recall / false-alarm behaviour.
- **Selective risk.** Selective risk at 90.5 % coverage equals full-coverage risk.

## 8. System (CPU i5-11400H, RTX 3050 4 GB)
- **Video analysis of a raw 40 s Nexar clip, end to end:** 46 s (YOLO + ByteTrack + flow + V4 layers, first run including model
  load). That is 0.87× real time.
- **V4 layers:** 8.9 ms per step mean (16.6 ms p95), with the graph 0.6–1 ms, reasoning 0.03–0.9 ms and SHAP 6–9 ms per
  warning.
- **Live demo (10 Hz pacing):** 374 steps for 40 s of video. Live warnings can differ from offline analysis because frames are
  sampled by wall clock.

Details: [SAFEDRIVE_V4_TEST_REPORT.md](SAFEDRIVE_V4_TEST_REPORT.md).

## 9. Reproduce
```
python -m safedrive_v4.evaluation.precompute --workers 4         # SG / KG channels
python -m safedrive_v4.evaluation.precompute_u --workers 4       # KGU channels + perception-error versions (H2)
python -m safedrive_v4.prediction.graph_tensors                 # GNN tensors
python -m safedrive_v4.evaluation.ablation                      # round 1 (A0, A1, A2, A4, baselines, windows)
python -m safedrive_v4.evaluation.hypotheses                    # round 2 (A3, A5, H2, H4); stops if A0 does not reproduce v3
python -m safedrive_v4.prediction.gnn --epochs 6
python -m safedrive_v4.evaluation.rule_stats; python -m safedrive_v4.evaluation.shap_global
python -m safedrive_v4.evaluation.xai_eval --model a3 --max-videos 224
```
**Seeds:**
- folds 13; training 600+k, 900 and 300; GNN 42+k;
- perception noise: `crc32(video_id) + 20261123`.

## 10. Pending
- Real-world driving data (0.0 h).
- Human evaluation of the explanations.
- A Nexar comparison with published anticipation models (UString, DSTA, Graph(Graph), SG2VEC).
- Rules with lower firing rates (the current rules fire 5–19 times per minute in normal driving), designed on dev and tested
  once on held-out.
