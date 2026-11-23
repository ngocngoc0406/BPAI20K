# Literature Review — Early Traffic-Collision Anticipation

Scope: traffic-accident anticipation, near-miss prediction, motion-based risk cues, predictive uncertainty,
explainability. Only works whose bibliographic details were checked are listed. DOIs are given only where
they were verified during this project (Nov 2026); for the rest, use the venue and title to look them up.

## 1. Overview table

| # | Paper | Year | Dataset | Method | Metrics | Limitation (w.r.t. this project) |
|---|---|---|---|---|---|---|
| 1 | Chan, Chen, Xiang, Sun — *Anticipating Accidents in Dashcam Videos* (ACCV) | 2016 | DAD (introduced) | Dynamic-spatial-attention RNN (DSA-RNN) over object features | AP, mean time-to-accident (mTTA) | No uncertainty; attention isn't validated as an explanation; fixed-length clips |
| 2 | Suzuki, Kataoka, Aoki, Satoh — *Anticipating Traffic Accidents with Adaptive Loss and Large-scale Incident DB* (CVPR), DOI 10.1109/CVPR.2018.00371 | 2018 | NIDB (near-miss, introduced) | Quasi-RNN + adaptive loss for early anticipation (AdaLEA) | mAP, ATTC | Dataset not public at scale; no calibration or uncertainty |
| 3 | Yao, Xu, Wang, Crandall, Atkins — *Unsupervised Traffic Accident Detection in First-Person Videos* (IROS) | 2019 | A3D (introduced) | Future bounding-box prediction; anomaly = prediction error | AUC | Detection (during/after the event), not early anticipation |
| 4 | Fang, Yan, Qiao, Xue, Li et al. — *DADA-2000: Can Driving Accident be Predicted by Driver Attention?* (ITSC) | 2019 | DADA-2000 (introduced) | Driver-attention-guided prediction | AUC, attention metrics | Attention needs eye-tracking labels |
| 5 | Bao, Yu, Kong — *Uncertainty-based Traffic Accident Anticipation with Spatio-Temporal Relational Learning* (ACM MM), DOI 10.1145/3394171.3413827 | 2020 | DAD, A3D, CCD (introduced) | Graph conv + RNN over agents; Bayesian NN uncertainty; uncertainty ranking loss (UString) | AP, mTTA | Uses uncertainty as a training signal, not for abstention or alarm decisions; calibration not reported |
| 6 | Bao, Yu, Kong — *DRIVE: Deep Reinforced Accident Anticipation with Visual Explanation* (ICCV), DOI 10.1109/ICCV48922.2021.00752 | 2021 | DADA-2000, DAD | Deep RL (SAC) with joint fixation prediction → visual explanation | AP, TTA, saliency metrics | Explanation is saliency only; no calibrated probability or uncertainty |
| 7 | Karim, Li, Qin, Yin — *A Dynamic Spatial-Temporal Attention Network for Early Anticipation of Traffic Accidents* (IEEE T-ITS 23(7)) | 2022 | DAD, CCD | Dynamic temporal + spatial attention with a GRU (DSTA) | AP, mTTA | Attention-based; no uncertainty; video-level clips |
| 8 | Yao et al. — *DoTA: Unsupervised Detection of Traffic Anomaly in Driving Videos* (IEEE TPAMI) | 2022 | DoTA (introduced) | Unsupervised anomaly detection; anomaly categories | AUC, STAUC | Anomaly detection, not anticipation |
| 9 | Moura, Zhu, Zvitia — *Nexar Dashcam Collision Prediction Dataset and Challenge* (CVPR Workshops; arXiv 2503.03848) | 2025 | Nexar (1,500 annotated ~40 s train clips; introduced) | Dataset + Kaggle challenge; test clips end 0.5/1.0/1.5 s before the event | mAP over 500/1000/1500 ms horizons | Dataset paper; baselines don't address calibration or explanation |
| 10 | Goldshmidt, Scott, Niccolini, Zhu, Moura, Zvitia — *BADAS: Context Aware Collision Prediction Using Real-World Dashcam Data* (arXiv 2510.14876) | 2025 | Nexar, re-annotated DAD/DADA-2000/DoTA | V-JEPA2 video backbone, end-to-end; ego-involvement labels | AP, AUC, TTA | Large video foundation model, which doesn't fit 4 GB VRAM; no uncertainty-driven abstention |

### Methods SafeDrive builds on

| Topic | Reference | Used for |
|---|---|---|
| Detection | Jocher et al., *Ultralytics YOLOv8* (software, AGPL-3.0, 2023) | Road-user detection |
| Tracking | Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box* (ECCV 2022) | Track IDs |
| Visual features | He et al., *Deep Residual Learning for Image Recognition* (CVPR 2016) | ResNet-18 embeddings |
| Optical flow | Farnebäck, *Two-Frame Motion Estimation Based on Polynomial Expansion* (SCIA 2003) | Ego / scene motion |
| Looming | Lee, *A theory of visual control of braking based on information about time-to-collision* (Perception 5(4), 1976) | Scale-based apparent τ |
| Recurrent / temporal | Cho et al., GRU (EMNLP 2014); Hochreiter & Schmidhuber, LSTM (Neural Computation 1997); Bai, Kolter, Koltun, *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling* (arXiv 2018) | Temporal models |
| MC-Dropout | Gal & Ghahramani, *Dropout as a Bayesian Approximation* (ICML 2016) | Epistemic uncertainty |
| Ensembles | Lakshminarayanan, Pritzel, Blundell, *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles* (NeurIPS 2017) | Ensemble uncertainty |
| Uncertainty types | Kendall & Gal, *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* (NeurIPS 2017) | Aleatoric vs epistemic |
| Calibration | Guo, Pleiss, Sun, Weinberger, *On Calibration of Modern Neural Networks* (ICML 2017) | Temperature scaling, ECE |
| Selective prediction | Geifman & El-Yaniv, *Selective Classification for Deep Neural Networks* (NeurIPS 2017) | Risk–coverage, abstention |
| Grad-CAM | Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization* (ICCV 2017) | Spatial heatmaps |
| Integrated Gradients | Sundararajan, Taly, Yan, *Axiomatic Attribution for Deep Networks* (ICML 2017) | Temporal / feature attribution |
| Corruptions | Hendrycks & Dietterich, *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations* (ICLR 2019) | Robustness protocol design |

## 2. Synthesis

**Anticipation formulation.** Starting with DAD [1], most work frames anticipation as per-frame accident
probability on short positive clips that end at the accident. Earliness is measured as mean time-to-accident
at a recall/threshold operating point. Nexar [9] instead fixes the *observation end* at 0.5/1.0/1.5 s before
the event and scores AP per horizon. That design removes the ambiguity of choosing a threshold for
earliness, and we adopt it as our primary window-level metric.

**Object and motion cues.** [1], [5] and [7] pool object-level features and model agent relations
with attention or graphs. Explicit, interpretable *image-plane motion cues* (looming / scale expansion,
lateral motion toward the ego path, pairwise closing rate) are common in classical forward-collision
warning, which builds on Lee's τ [Lee 1976]. They are rarely reported as separate, ablated inputs in
learned anticipation models.

**Uncertainty.** UString [5] is the closest prior work: it estimates Bayesian uncertainty for accident
anticipation. **Uncertainty-aware anticipation is therefore not new.** What [5] does not evaluate: (a)
calibration (ECE, Brier, reliability) of the anticipation probability; (b) whether uncertainty identifies
wrong predictions (error-detection AUROC, risk–coverage); (c) using uncertainty to *abstain* (an explicit
UNCERTAIN output) inside a deployed alarm policy.

**Explainability.** DRIVE [6] provides visual (saliency) explanations, and attention maps in [1, 7] are
often read as explanations. **Explainable anticipation is therefore not new either.** Most explanations
are spatial only. Few works report temporal attribution, or tie the explanation to measured,
human-readable motion quantities.

**Resource-constrained deployment.** Recent state of the art (BADAS [10]) uses large video foundation
models. Pipelines that run on a 4 GB laptop GPU, with transparent and auditable intermediate signals,
are less explored.

## 3. Datasets considered

| Dataset | Access | Why (not) used |
|---|---|---|
| Nexar Collision Prediction [9] | Hugging Face (gated, Nexar Open Data License) / Kaggle | **Used.** Real dashcam clips, event *and* alert timestamps, balanced classes |
| DAD [1] | Public research release | Fallback; positive clips end at the accident, 20 fps, no alert time |
| CCD [5] | GitHub (research) | Fallback |
| DADA-2000 [4], DoTA [8], A3D [3] | Research licences | Not used: oriented to attention or anomaly detection |
