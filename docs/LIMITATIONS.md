# Limitations

## Data
- **Subset and size.** 400 of the 1,500 Nexar training videos (balanced), split 280/60/60 by video. The test set has 30 positive and
  30 negative videos (90 + 90 windows), so test metrics carry wide sampling uncertainty: one video changes recall by about 3 points.
  Seed std is reported, but there is no cross-validation.
- **No official test labels used.** The Nexar test-public / test-private ground truth (`solution.csv`) was not used, so the numbers
  are not directly comparable to the Kaggle leaderboard or published Nexar results.
- **Binary task.** Collision and near-collision share one positive label in this dataset, so near-miss prediction cannot be evaluated separately.
- **Balanced sampling.** The 50/50 class prior is far from real-world event rates. Precision and FA/h would be worse at deployment priors.
  The FA/h estimate comes from only ~18 min of normal driving.
- **Condition coverage.** Few night (Dark: 16) and rain (18) videos, so per-condition analysis is not statistically meaningful.

## Method
- **Image-plane motion only.** No camera calibration, depth or ego-speed. Looming and apparent τ are scale-based cues, valid only under
  constant closing speed for rigid, fronto-parallel objects. **No physical TTC is claimed.** Ego-motion is not compensated
  (optical-flow statistics are only a proxy).
- **Off-the-shelf perception.** YOLOv8n is not fine-tuned on dashcam data, and small or distant objects are missed. Without MOT ground truth,
  ID switches cannot be measured.
- **Frozen frame-level CNN.** ResNet-18 embeddings ignore fine spatial layout. Larger video backbones (e.g. BADAS / V-JEPA2) are
  stronger but do not fit the 4 GB budget for training.
- **Risk-engine heuristics.** The trend and looming bonuses use fixed weights (0.15 / 0.10), not learned ones. Only the thresholds are tuned on validation.
- **Explanations.** Grad-CAM covers only the last frame's appearance pathway. IG depends on the baseline choice. Triggering factors are
  statistical outliers, not causal attributions.

## Evaluation
- **Streaming metrics.** Alarm onsets need 3 consecutive steps (0.6 s), which delays alarms by design. TTA is measured against the annotated
  event time, whose precision is limited by human annotation.
- **Robustness.** Corruptions are synthetic and applied at the 640-px working resolution. Real night or rain footage differs.
- **Single dataset.** No cross-dataset generalisation test (DAD, CCD).

## Findings that limit claims (from the experiments)
- The proposed ensemble (test mAP 0.582) is **not** better than an XGBoost baseline on pooled features (0.622). The 95 % CIs overlap heavily.
- Streaming detection at ≤ 30 FA/h is low: 36.7 % of events, 20 % timely.
- Temperature scaling fitted on 180 validation windows **worsened** test ECE (0.102 → 0.128).
- The MI-based UNCERTAIN state does not identify errors (error-detection AUROC 0.50) and does not rise under corruption.
- The CAUTION threshold (window recall ≥ 0.90 on val) is low (0.34), so CAUTION covers about 36 % of test timeline steps.
  In a product, CAUTION would need re-tuning against user tolerance.

## Product
- Research prototype, not a certified ADAS. No real-time camera input yet (file upload only). Annotated videos keep all frames
  in RAM, which caps uploads at 120 s.

## v2-light specific
- **Early warning.** Streaming detection at ≤ 10 FA/h is low: 27 % of events, 13 % timely. It did not improve over v1; this is the main open problem.
- **Low light.** v2 is more sensitive to strong low light than v1: mAP 0.547 vs 0.593 at gain 0.25. Frame-difference features shrink in dark scenes.
- **Selection caveat.** The protocol was revised after the first test evaluation (see EXPERIMENT_DESIGN.md).
- **Explanations.** The frame-change heatmap is input evidence, not a model saliency map. Occlusion attributions depend on the
  training-median baseline.
- **Throughput.** The CPU-only figure (20 fps) was measured on this laptop's CPU; slower CPUs may fall below real time.

## v3 specific
- **Early warning.** Still low: held-out recall 0.11–0.24 at 5–20 FA/h, and EDR@2 s ≤ 0.05. Improvements are significant only on dev;
  the 224-video held-out set cannot confirm effects of 2–3 recall points.
- **Ego-motion.** Monocular, uncalibrated, image-plane only. The homography assumes a dominant planar or distant background. With very
  few background corners (night, fog) it falls back to identity (flagged per step). Compensation helps under shake but not on stable footage.
- **Interaction cues.** These are image-based proxies ("time to meet", "lead gap"). They are not physical distances or TTC. Lane geometry
  is approximated by a fixed central corridor.
- **Risk-level thresholds** are FA-budgeted on dev normal driving. Different cameras, regions or traffic densities will change the actual FA/h.
- **Run-to-run variation.** Re-extracting identical corrupted clips changed some v2 robustness numbers by up to 0.05 mAP
  (GPU detector/tracker nondeterminism), so small robustness differences should not be over-read.
- **Sleep interruption.** Long jobs were paused once by laptop sleep. Results are unaffected, because every job resumes deterministically.
