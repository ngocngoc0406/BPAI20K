# SafeDrive AI v4: Model Card

## Recommended configuration: A3 = frozen v3 predictor + v4 explanation layer
| | |
|---|---|
| Predictor | v3 LightGBM ×5 with Platt calibration (`models/safedrive_v3`, **read-only**). v4 outputs are identical to v3's `TabularPredictor`: max difference < 1e-9 (test `test_a3_v3_readonly_predictions_equal_v3_and_files_untouched`) |
| Explanation layer (v4) | Dynamic Traffic Scene Graph (sg-1.0), Traffic KG + 10 rules (kg-1.0), exact Tree SHAP (group level first), trajectory counterfactuals, evidence trace, perception reliability / OOD / insufficient-evidence state |
| Intended use | Research prototype: explain and audit early-collision-risk estimates on recorded dashcam video, and support annotation and analysis. **Not** a safety device; it **cannot prevent accidents**; it is not certified; it must never be relied on while driving |
| Inputs | Monocular dashcam video, processed by the v3 perception (YOLO road users + ByteTrack + homography ego motion) |
| Outputs | Risk probability (calibrated), model uncertainty (mutual information), perception reliability, warning level (LOW / MODERATE / HIGH / INSUFFICIENT_EVIDENCE), explanation and evidence trace |

## Research model: A6 (kept for comparison, not recommended)
`models/safedrive_v4_research/`: LightGBM ×5 on v3 + scene-graph + KG features (936). Trained on 1,276 dev videos.
- Held-out: mAP 0.597 vs 0.620 for v3; paired ΔmAP −0.023, 95 % CI [−0.050, 0.003].
- Recall@10 FA/h: 0.080 at 4.4 FA/h, vs 0.116 at 8.7 FA/h for v3.

**Not better than v3.**

## Performance (held-out Nexar, 224 videos; full tables in the experiment plan)
| | A3 (= v3) |
|---|---|
| mAP (AP at 0.5 / 1.0 / 1.5 s) | 0.620 |
| ROC-AUC / PR-AUC | 0.816 / 0.813 |
| Brier / ECE | 0.176 / 0.028 |
| Recall @5 / 10 / 20 FA/h | 0.089 / 0.116 / 0.179 |
| Early warning ≥ 2 s (10 FA/h) | 2.7 % |

**Early warning stays weak:** about 1 in 9 events is caught at about 9 false alarms per hour, and fewer than 3 % are caught
2 s ahead.

## Explanation quality (held-out, 145 warning events; `experiments/v4/xai_eval_a3.json`)
| Check | Result |
|---|---|
| SHAP local accuracy | max error 1.2e-14 (exact) |
| Deletion fidelity | removing the top-10 SHAP features lowers the logit by 1.32 vs 0.01 for 10 random features; top beats random in 100 % of events |
| Attribution stability | single features: Spearman 0.06 (unstable, so shown as indicative only); feature groups: 0.90 (stable, shown first) |
| Graph-evidence validity | 100 % of 112 chain steps reference real nodes / edges at the stated time |
| Trace integrity | 100 % of sealed traces verify |
| Counterfactual validity | 100 % of 60 (identity reproduction exact, valid boxes, plausible steps) |
| Counterfactual direction | halving closing speed lowers risk in 95 % of cases (mean Δp −0.026); removing lateral motion lowers it in 80 % (mean Δp −0.003) |
| Explanation completeness | **47 %** of warnings have a KG reasoning chain. For the rest, no rule fired at that step and only SHAP and the reliability statement explain the warning; the UI says so |
| Human evaluation | **pending** (no trained annotators yet) |

## Limitations and risks
- **Image-plane only.** Image-based TTC is not a physical TTC, and there are no lanes and no traffic lights. The corridor is a
  fixed image region.
- **No real-world driving data** (0.0 h). Nexar is US dashcam footage, so traffic in Vietnam (dense motorcycles) is
  out of distribution.
- **Counterfactuals keep the pixel channels** (optical flow, frame difference) as observed.
- **The insufficient-evidence state does not discriminate:** it is shown on about 16 % of steps before events and about 18 % in
  normal driving. It describes input quality, not risk.
- **Rule base.** The risk rules fire 5–19 times per minute in normal driving. They are evidence, not alarms.

## Provenance
- Bundle fields: `feature_schema` (v4feat-1.0), scene-graph schema sg-1.0, KG schema kg-1.0, `v4_config_hash`, training stamp
  and the experiment directory.
- The v3 artefacts and `configs/frozen_v3_hashes.json` are unchanged (verified by hash).

## Product round update (2026-11-23)
**Numbering.** The product prompt renumbers the configurations. The research bundle `models/safedrive_v4_research/` (v3 + SG + KG
features) is **A4** in the new numbering (it was called A6 in round 1). **A6** is now the product configuration:
- frozen v3 predictor;
- uncertainty-aware KG evidence (subjective logic);
- 5 warning states (LOW / MODERATE / HIGH with a cautious flag / INSUFFICIENT_EVIDENCE / DEGRADED_INPUT);
- evidence-grounded XAI;
- Near-Miss Intelligence.

**Detection performance of A6.** It equals v3: the policy gates were tested (H4) and are off.

**Hypotheses:**
- H1 (graph and KG improve prediction): not supported;
- H2 (uncertainty-aware reasoning gives robustness to perception errors): not supported;
- H3 (evidence-grounded explanations): partly supported (verifiable, but not shown faithful to the model and present for only
  47 % of warnings);
- H4 (adaptive warning): not supported.

**Failure cases to know:**
- late or missing warnings (88 % of events missed at about 9 FA/h; example: clip 00063 warned 0.45 s **after** the annotated
  event);
- rules firing in normal traffic (5–19 times per minute);
- live warnings differing from offline analysis (frames sampled by wall clock);
- no clips when the encrypted volume is not mounted;
- unsupported inputs: night without light, heavy rain, unusual camera mounts (OOD).
