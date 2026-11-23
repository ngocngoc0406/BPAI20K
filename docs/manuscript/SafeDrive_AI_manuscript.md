# SafeDrive AI: An Empirical Study of Explainable, Uncertainty-Aware Early Collision Anticipation from Dashcam Video on Commodity Hardware

*Draft manuscript (IEEE conference structure). All numbers come from `experiments/latest/`:
v1 run 20261119_020750 and v2 run light_cv_20261119_095514. Section IX reports the lightweight optimisation and Section X the early-warning optimisation (run ew_full_20261120_004859).*

---

## Abstract
Anticipating traffic collisions from a monocular dashcam is only useful if the warning is early, rare when nothing happens,
honest about its confidence, and explainable. We present SafeDrive AI, a reproducible pipeline:
- YOLOv8 detection and ByteTrack tracking;
- 28 interpretable image-plane motion cues (looming, apparent τ, lateral motion toward the ego path, pairwise closing, optical-flow expansion);
- ResNet-18 appearance features;
- a deep ensemble of temporal models with Monte-Carlo dropout and temperature scaling;
- a graded risk policy with an explicit UNCERTAIN (abstain) state;
- Grad-CAM and Integrated-Gradients explanations.

We evaluate it on a balanced 400-video subset of the Nexar Dashcam Collision Prediction dataset, using its horizon protocol (0.5/1.0/1.5 s before
the event) together with streaming alarm metrics. Our findings:
- **Feature design.** Absolute frame embeddings hurt temporal specificity, while appearance *change* combined with motion cues improves
  validation mAP from 0.49 to 0.57.
- **Accuracy.** The final ensemble reaches a test mAP of 0.582 (95 % CI 0.42–0.77). This is statistically indistinguishable from a
  pooled-feature XGBoost baseline (0.622).
- **Streaming.** At ≤ 30 false alarms per hour, 37 % of events are detected.
- **Uncertainty.** Total predictive entropy supports selective prediction (accuracy 0.70 → 0.79 at 50 % coverage), but epistemic mutual
  information neither detects errors nor rises under input corruption, and temperature scaling fitted on a small validation set did not transfer.
- **Efficiency.** The full system runs at 40 frames/s on a 4 GB laptop GPU.
- **Lightweight optimisation.** A systematic comparison selected by grouped cross-validation (logistic regression, LightGBM,
  XGBoost, CatBoost, light GRU/TCN; with and without CNN features) shows that ResNet features and neural temporal models bring no
  supported gain. The final *v2-light* model is a bagged LightGBM on motion and frame-difference window aggregates with no CNN.
  It reaches test mAP 0.610 (CI 0.43–0.83), ROC-AUC 0.834 and ECE 0.114. Prediction takes 4 ms per window on CPU, and the
  complete pipeline runs at 2× real time on a CPU-only laptop. Streaming early warning, however, remains weak
  (27 % of events detected at 10 false alarms per hour).
- **Early-warning optimisation (full 1,500-video dataset).** Vehicle-interaction cues and dense-negative training improved streaming
  recall on 1,276 development videos (+2.5 and +2.3 points, bootstrap-significant). Ego-motion compensation added robustness to camera
  shake (+0.1–0.2 mAP). Held-out confirmation (224 videos) is directionally consistent but not significant, and recall at 5–20 false
  alarms per hour stays at 0.1–0.25.

We report these mixed results to calibrate expectations for laptop-scale, explainable collision anticipation.

**Index Terms**: accident anticipation, dashcam, uncertainty estimation, calibration, explainable AI, ADAS.

## I. Introduction
Rear-end, cut-in and vulnerable-road-user conflicts develop over 1–3 s, so a warning even half a second earlier can change outcomes.
Learning-based accident anticipation from dashcams has progressed since DAD [1]. Three properties that decide whether a warning
system is usable are rarely studied together:
- the false-alarm rate under continuous (streaming) operation;
- whether the system's confidence is calibrated and can be used to abstain;
- whether its explanations are grounded in measurable evidence.

Recent strong results rely on large video foundation models [10], which are out of reach for low-cost hardware.

**Contributions.** We do not claim a new model component. Uncertainty-aware [5] and visually explained [6] anticipation already exist.
Our contributions are:
1. A reproducible, causal, laptop-scale pipeline that integrates perception, interpretable motion cues, temporal ensembles, calibrated
   uncertainty with abstention, and grounded explanations, released with tests and a dashboard.
2. A controlled study on Nexar [9] with baselines, ablations, video-level bootstrap CIs, streaming early-warning metrics, and robustness
   under corruptions that re-runs the whole perception chain.
3. Empirical findings, including negative ones:
   - absolute CNN embeddings hurt temporal specificity, while appearance-change plus motion cues help;
   - a pooled-feature gradient-boosting baseline is competitive;
   - epistemic uncertainty did not flag errors or corrupted inputs, and temperature scaling did not transfer.

## II. Related Work
**Accident anticipation.**
- DSA-RNN [1] introduced DAD and attention over object features.
- AdaLEA [2] proposed an earliness-adaptive loss with the NIDB near-miss data.
- DSTA [7] adds dynamic spatial-temporal attention.
- Nexar [9] released 1,500 annotated clips with event *and* alert times, and a horizon-based mAP protocol.
- BADAS [10] trains V-JEPA2-based models on Nexar and re-annotated benchmarks.

**Uncertainty.** UString [5] uses Bayesian GCN-RNNs with an uncertainty ranking loss. General tools include MC-Dropout [12],
deep ensembles [13], the uncertainty decomposition [14], temperature scaling [15] and selective classification [16].

**Explainability.** DRIVE [6] learns attention via reinforcement learning. Grad-CAM [17] and Integrated Gradients [18] are generic
attribution methods.

**Motion cues.** Lee's τ [11] underlies classical forward-collision warning.

## III. Method
**Perception.**
- Frames are resampled to a 10 Hz grid; the source FPS in Nexar varies from 23.6 to 31.0.
- YOLOv8n (COCO) keeps six road-user classes, and ByteTrack [19] links them into tracks.
- From the tracks, with finite differences over 0.3 s, we compute 28 image-plane cues per step: counts and sizes; looming ṡ/s and
  apparent τ = s/ṡ in the driving corridor; lateral velocity toward the image centre; apparent speed and acceleration; minimum pairwise
  distance, IoU and closing rate; track churn; Farnebäck flow magnitude, expansion and horizontal median; brightness and detector confidence.
- These are apparent quantities. Without calibration or depth, no physical TTC is claimed.
- A frozen ResNet-18 provides 512-d frame embeddings.

**Temporal model.**
- A causal window of W s feeds per-modality projections into a GRU, LSTM, TCN or pooling encoder.
- The visual stream is either the embedding vₜ (*abs*) or its change vₜ − vₜ₋₁ (*delta*).
- Training uses BCE with AdamW (lr 3e-4, wd 1e-2), input dropout 0.3 and early stopping on validation AP.
- Positive windows end 0.3–2.0 s before the annotated event; negatives come from normal-driving videos.

**Uncertainty.** We train five members, draw 20 MC-dropout passes each, and fit one temperature on validation. We report the
predictive mean, total entropy, expected entropy and mutual information (MI).

**Risk policy.**
- risk = p + 0.15·max(0, dp/dt) + 0.10·clip(z_looming/3, 0, 1).
- WARNING is chosen on validation *streams* to maximise timely detection subject to ≤ 30 FA/h. CAUTION and HIGH RISK come from
  window recall and precision targets.
- The state is UNCERTAIN when MI exceeds its validation 90th percentile within the ambiguous band.
- Alarms require 3 consecutive steps and a W-second warm-up.

**Explanations.**
- Grad-CAM through the full model: the last frame's embedding is recomputed with gradients.
- Integrated Gradients over time and features.
- Permutation importance.
- Template explanations filled with measured cue values that exceed training-set percentiles; no language model is used.

## IV. Dataset and Experimental Setup
- **Data.** 400 videos from the Nexar train set, a balanced random sample with seed 42: 1280×720 resolution, mean duration 37.4 s.
  Labels come from the positive/negative folders. Annotated alert lead (event − alert) has a median of 1.49 s. The split is
  stratified by video: 280 / 60 / 60.
- **Evaluation windows.** Following [9], positives end at τ_e − {0.5, 1.0, 1.5} s; negatives use matched offsets in normal videos.
  This gives 180 test windows.
- **Metrics.**
  - Window level: mAP (mean of AP per horizon versus all negatives), ROC-AUC, F1 at the validation-optimal threshold, Brier score,
    and ECE (10 bins).
  - Streaming: detection, timely detection (≤ 3 s), TTA and FA/h.
  - Uncertainty: error-detection AUROC and risk–coverage.
  - CIs: video-level bootstrap with 1,000 resamples.
- **Protocol.** All choices — window, architecture, regularisation, temperature, thresholds — are made on validation.
  A validation design study preceded the final run.
- **Hardware.** RTX 3050 Laptop with 4 GB of VRAM.

## V. Results
**Main comparison (Table I, test).**

| System | mAP [95 % CI] | ROC-AUC | F1 | ECE |
|---|---|---|---|---|
| Majority | 0.250 | 0.500 | 0.667 | — |
| LogReg (pooled) | 0.568 [0.39, 0.81] | 0.795 | 0.731 | 0.107 |
| XGBoost (pooled) | 0.622 [0.42, 0.84] | 0.828 | 0.745 | 0.205 |
| XGBoost (motion) | 0.580 [0.40, 0.88] | 0.845 | 0.774 | 0.201 |
| SafeDrive E (LSTM ×5, Δvisual + motion) | 0.582 [0.42, 0.77] | 0.731 | 0.719 | 0.128 |

The paired difference E − XGBoost is −0.040 [−0.197, 0.097].

**Streaming (Table II, test).** With the WARNING threshold from validation (0.88):
- detection before the event is 36.7 %, and timely detection 20.0 %;
- false alarms are 29.9 per hour, i.e. 9 alarms in 18 min of normal driving;
- the median time-to-alert is 11.2 s.

Probability-only alarms give 33.3 %, 16.7 % and 10.0 FA/h. The operating curve reaches about 90 % detection only above 200 FA/h.

**Uncertainty.**
- ECE is 0.102 before and 0.128 after temperature scaling (T = 0.60).
- Error-detection AUROC is 0.588 for entropy and 0.501 for MI.
- Abstaining by entropy raises accuracy from 0.700 to 0.789 at 50 % coverage.

**Robustness (Table III).** Clean re-extracted clips score 0.587.
- 25 % occlusion: 0.477 (largest drop).
- σ = 35 noise: mAP 0.540, with F1 falling to 0.468 and ECE rising to 0.297.
- Blur k = 21: 0.528.
- 50 % frame drop: 0.561.
- Low light: 0.593–0.612.

Mean MI stays between 0.092 and 0.114 under every corruption, against 0.104 when clean.

**Efficiency.** Perception runs at 40.2 frames/s. Inference takes 1.2 ms per window for one model and 73 ms for the ensemble with MC sampling.

## VI. Discussion
Three lessons stand out.
1. **Representation matters more than architecture.** On 280 training videos, absolute frame embeddings act as a video-identity shortcut.
   Differencing them, together with explicit motion cues, restores temporal specificity: validation temporal AUC rises from 0.56 to 0.66.
   Architecture differences (GRU, LSTM, TCN) are within noise.
2. **Window-level AP overstates streaming usefulness.** A model with mAP ≈ 0.58 still produces either many false alarms or few
   timely detections. Streaming metrics with validation-chosen thresholds should be standard.
3. **Uncertainty tools need data to work.** Temperature fitted on 180 validation windows overfitted. MI from five small members
   did not capture the distribution shift caused by corruptions. Total entropy remained useful for abstention.

## VII. Ablation Study
Test mAP, mean ± std over 3 seeds:
- A, frame-level (no temporal): 0.520 ± 0.014.
- B, GRU with absolute visual: 0.446 ± 0.023.
- C, B with motion added: 0.451 ± 0.020.
- C2, GRU with Δvisual + motion: 0.493 ± 0.106.
- Δvisual only: 0.538 ± 0.026. Motion only: 0.562 ± 0.064.
- LSTM / TCN with Δvisual + motion: 0.578 ± 0.011 / 0.596 ± 0.040.
- D, single model + MC + TS: 0.565. E, ensemble: 0.582.

The validation design study rejected hard negatives (lower mAP) and adopted stronger regularisation and Δvisual.

## VIII. Limitations
- Data: a subset of Nexar (400 / 1,500 videos), a small test set with wide CIs, no official test labels, binary labels only,
  and a balanced prior.
- Perception: image-plane motion without ego-motion compensation; the detector is not fine-tuned.
- Robustness: synthetic corruptions only.
- Risk policy: heuristic risk-score weights.
- Status: a research prototype, not a certified ADAS.

## IX. Lightweight Optimisation (v2)
**Question.** What is the lightest architecture that keeps the accuracy of the v1 system?

**Perception cost.** We profiled each component on 1,207 real frames:
- decode + resize: 5.8 ms;
- YOLOv8n@640 + ByteTrack: 9.3 ms. Inputs of 480 or 320 were not faster, because the cost is overhead-bound, but lost 31 % and 64 %
  of the boxes; YOLO11n took 12.4 ms;
- flow: 3.2 ms; ResNet-18: 3.3 ms; a frame-difference descriptor: 0.5 ms.

**Candidates.**
- Classifiers: logistic regression, LightGBM, XGBoost, CatBoost and light GRU/TCN.
- Feature sets: motion; motion + frame difference; + CNN appearance change; + absolute CNN.
- Windows: 1, 2 and 3 s.
- Each channel is summarised by 8 temporal aggregates.

**Selection protocol.** Selection on the 60-video validation split proved unreliable: its rankings disagreed with test rankings, and
the chosen logistic regression (test mAP 0.533) was later shown by cross-validation to be significantly worse than gradient boosting
(Δ OOF mAP +0.14, CI [+0.07, +0.22]). We therefore selected by grouped 5-fold CV on the 340 train+val videos: the cheapest
configuration within one fold-SE of the best. All thresholds were also set from out-of-fold streams. The test set was evaluated once.
We disclose that the protocol was revised after the first test evaluation.

**Outcome.** Bagged LightGBM ×5 on motion + frame-difference aggregates (W = 1 s, 384 inputs, no CNN), with Platt calibration.

**Test (same 180 windows).**

| | v2-light | v1 LSTM ensemble |
|---|---|---|
| mAP | 0.610 [0.43, 0.83] | 0.582 (Δ +0.028 [−0.083, +0.150]) |
| ROC-AUC | 0.834 | 0.731 |
| F1 | 0.816 | 0.719 |
| ECE | 0.114 | 0.128 |
| Error-detection AUROC (entropy) | 0.679 | 0.588 |
| Accuracy at 50 % coverage | 0.889 | 0.789 |
| Streaming detection at 10 FA/h (timely) | 26.7 % (13.3 %) | 33.3 % (16.7 %) |
| End-to-end throughput | 42.9 fps GPU, **20.2 fps CPU-only** | 39.0 fps GPU |
| Prediction latency | 4.1 ms per window, CPU | 73 ms per window, GPU |

**Robustness.** On identical corrupted clips, v2's absolute mAP is equal to or above v1's in every condition, but v2 is more sensitive
to strong low light (0.547 vs 0.593).

**Conclusion of the optimisation.** CNN appearance features and neural temporal models gave no supported improvement on this data.
A tree ensemble on interpretable motion and frame-change statistics is sufficient, better calibrated, and CPU-deployable.

## X. Early-Warning Optimisation on the Full Dataset (v3)
**Setup.** All 1,500 Nexar videos were split by video before any experiment into 1,276 development and 224 held-out videos
(the original test 60 + 164 new). Every choice used 5-fold grouped CV on development streams.

**Primary metric.** Mean recall at 5, 10 and 20 false alarms per hour. An alarm counts as valid when its onset falls within 5 s before
the event. A change was kept only if the paired video-bootstrap 90 % CI of the improvement excluded 0.

**Candidates.**
- Camera ego-motion: KLT/RANSAC homography and dense-flow affine.
- Scene-relative object motion and 18 image-plane vehicle-interaction cues: cut-in, lead gap and closing rate, lead braking proxy,
  pairwise time-to-meet.
- Prediction horizons of 1.5–4 s, alert-aligned labels, hard / dense / mined negatives, and alarm smoothing.

**Results on development data.** Two changes passed:
- interaction cues: ΔS = +0.025 [+0.001, +0.049];
- dense negatives, i.e. matching the training negative distribution to streaming inference: ΔS = +0.023 [+0.003, +0.044].

Longer horizons were significantly worse (−0.027 and −0.040). Ego-motion compensation did not add streaming gains on stable footage.

**Held-out.** With thresholds fixed on development streams, v3 reached mAP 0.620, ROC-AUC 0.816 and ECE 0.028. Its realised false-alarm
rates stayed close to target (2.6 / 11.3 / 17.4 FA/h at targets 5 / 10 / 20), whereas the retrained baseline overshot (0.0 / 16.5 / 30.5).
At matched false-alarm rates, recall was 0.107 / 0.116 / 0.241, vs 0.036 / 0.116 / 0.179 for the deployed v2. That difference is not
significant on 224 videos (ΔS +0.030 [−0.034, +0.092]).

**Camera shake.** A compensated variant (ego homography + compensated interactions) raised test mAP under synthetic shake from
0.42 / 0.49 to 0.64 / 0.60, and was also favoured on a development shake probe.

**Cost.** End-to-end throughput: v3 runs at 42 fps on GPU and 19 fps CPU-only; the compensated variant at 34 and 17 fps.

**Conclusion.** Interaction cues and streaming-matched negatives measurably help early warning, and ego-motion compensation provides
robustness to camera shake. Absolute early-warning recall at practical false-alarm rates nevertheless remains low (≈ 0.1–0.25).

**Audit note.** At matched false-alarm rates on the held-out set, v3 does not differ from the v2 configuration retrained on the
same development data (ΔS −0.003 [−0.053, +0.044]). The held-out difference to the earlier v2 model reflects the larger training set.
The deployed code realises 11.6 % recall [6.9, 18.9] at 8.5 false alarms per hour [4.1, 15.6].

## XI. Conclusion
SafeDrive AI shows that an explainable, uncertainty-aware collision-anticipation system can run in real time on a laptop, with
every claim traceable to experiment files. After optimisation it no longer needs a GPU. Its window-level accuracy is moderate and
statistically tied across model families. Streaming early warning remains the open problem. Two common uncertainty techniques
(temperature scaling on a small validation set, and MI-based abstention) did not behave as hoped, while Platt calibration chosen by
cross-validation did. We hope these calibrated, partly negative findings, together with the open pipeline, serve as a reproducible
baseline for future work: full-dataset training, ego-motion-aware cues and stronger video encoders.

## References
[1] F.-H. Chan, Y.-T. Chen, Y. Xiang, M. Sun, "Anticipating Accidents in Dashcam Videos," ACCV, 2016.
[2] T. Suzuki, H. Kataoka, Y. Aoki, Y. Satoh, "Anticipating Traffic Accidents with Adaptive Loss and Large-scale Incident DB," CVPR, 2018, doi:10.1109/CVPR.2018.00371.
[3] Y. Yao, M. Xu, Y. Wang, D. J. Crandall, E. M. Atkins, "Unsupervised Traffic Accident Detection in First-Person Videos," IROS, 2019.
[4] J. Fang et al., "DADA-2000: Can Driving Accident be Predicted by Driver Attention? Analyzed by A Benchmark," ITSC, 2019.
[5] W. Bao, Q. Yu, Y. Kong, "Uncertainty-based Traffic Accident Anticipation with Spatio-Temporal Relational Learning," ACM MM, 2020, doi:10.1145/3394171.3413827.
[6] W. Bao, Q. Yu, Y. Kong, "DRIVE: Deep Reinforced Accident Anticipation with Visual Explanation," ICCV, 2021, doi:10.1109/ICCV48922.2021.00752.
[7] M. M. Karim, Y. Li, R. Qin, Z. Yin, "A Dynamic Spatial-Temporal Attention Network for Early Anticipation of Traffic Accidents," IEEE T-ITS, 23(7):9590–9600, 2022.
[8] Y. Yao et al., "DoTA: Unsupervised Detection of Traffic Anomaly in Driving Videos," IEEE TPAMI, 2022.
[9] D. C. Moura, S. Zhu, O. Zvitia, "Nexar Dashcam Collision Prediction Dataset and Challenge," CVPR Workshops, 2025; arXiv:2503.03848.
[10] R. Goldshmidt, H. Scott, L. Niccolini, S. Zhu, D. Moura, O. Zvitia, "BADAS: Context Aware Collision Prediction Using Real-World Dashcam Data," arXiv:2510.14876, 2025.
[11] D. N. Lee, "A theory of visual control of braking based on information about time-to-collision," Perception, 5(4), 1976.
[12] Y. Gal, Z. Ghahramani, "Dropout as a Bayesian Approximation," ICML, 2016.
[13] B. Lakshminarayanan, A. Pritzel, C. Blundell, "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles," NeurIPS, 2017.
[14] A. Kendall, Y. Gal, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?," NeurIPS, 2017.
[15] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, "On Calibration of Modern Neural Networks," ICML, 2017.
[16] Y. Geifman, R. El-Yaniv, "Selective Classification for Deep Neural Networks," NeurIPS, 2017.
[17] R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," ICCV, 2017.
[18] M. Sundararajan, A. Taly, Q. Yan, "Axiomatic Attribution for Deep Networks," ICML, 2017.
[19] Y. Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box," ECCV, 2022.
[20] K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image Recognition," CVPR, 2016.
[21] G. Farnebäck, "Two-Frame Motion Estimation Based on Polynomial Expansion," SCIA, 2003.
[22] D. Hendrycks, T. Dietterich, "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations," ICLR, 2019.
[23] G. Jocher, A. Chaurasia, J. Qiu, "Ultralytics YOLOv8," software, 2023 (AGPL-3.0).
[24] T. Chen, C. Guestrin, "XGBoost: A Scalable Tree Boosting System," KDD, 2016.
[25] G. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," NeurIPS, 2017.
[26] L. Prokhorenkova et al., "CatBoost: unbiased boosting with categorical features," NeurIPS, 2018.
[27] J. Platt, "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods," Advances in Large Margin Classifiers, 1999.
