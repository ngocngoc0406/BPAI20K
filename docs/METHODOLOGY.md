# Methodology

## 1. Perception (per frame on a 10 Hz grid)
- **FPS normalisation.** Frames are decoded sequentially. For each grid time k/10 s we take the first source frame whose timestamp reaches it,
  then resize to 640 px width.
- **Detection.** YOLOv8n (COCO-pretrained, Ultralytics, AGPL-3.0), confidence ≥ 0.25, NMS IoU 0.5. Classes kept: person, bicycle,
  car, motorcycle, bus, truck. The detector is not fine-tuned, because Nexar has no box labels.
- **Tracking.** ByteTrack (Ultralytics implementation), reset for each video. Without MOT ground truth, ID switches cannot be measured.
  We report fragmentation proxies instead: track count, length and gaps.
- **Appearance.** ResNet-18 (ImageNet weights, frozen) on the 224×224 frame, giving a 512-d embedding in fp16 on GPU.
- **Optical flow.** Farnebäck at 160×90. Summary statistics: global magnitude, magnitude in the road-ahead region, expansion
  (divergence) in that region, and median horizontal flow.

## 2. Motion features (28 per step; image-plane, **not physical**)
All features are in normalised image units per second, computed by finite differences over k = 3 steps (0.3 s) of each track.
- **Counts.** All road users, vehicles, persons, two-wheelers.
- **Size.** Largest area, total area, plus the lead (largest) object's area, horizontal offset and bottom edge.
- **Looming.** Scale-expansion rate ṡ/s of the linear box size, maximum over the corridor (|cx − 0.5| < 0.2, cy > 0.35) and over the whole view.
  The **apparent τ = s / ṡ** (Lee, 1976) is capped at 10 s. It equals physical time-to-contact only under constant closing speed and
  a rigid object, so it is used purely as a learned input.
- **Lateral and speed.** Lateral velocity toward the image centre (lead object, and maximum), apparent speed (maximum, mean), apparent acceleration.
- **Interactions.** Minimum pairwise centre distance, maximum pairwise IoU, maximum pairwise closing rate.
- **Track churn.** New tracks, lost tracks.
- **Scene.** Brightness, mean detector confidence.

Before z-scoring with training statistics, motion features go through a signed log transform sign(x)·log(1 + |x|),
because they are heavy-tailed.

## 3. Temporal collision model
The input is a causal window of W s (W ∈ {1, 2, 3, 5}, chosen on validation): visual (T × 512) and motion (T × 28).
- **Visual stream mode.** `abs` uses the embedding vₜ directly. `delta` (proposed) uses the appearance change vₜ − vₜ₋₁ (first step = 0).
  The validation design study (`scripts/diag_val.py`, `experiments/latest/val_streaming_diagnostics.json`) showed that absolute
  embeddings let the model recognise *which kind of video* it sees (scene / camera identity) rather than *what is happening now*:
  they lower temporal specificity. The delta stream keeps the dynamics and removes the static identity.
- **Input dropout** 0.3 on both input streams (feature-level), on top of 0.3 dropout after each projection and before the head.
- **Projection.** Linear → ReLU → Dropout per modality (512 → 64, 28 → 32), then concatenation.
- **Encoder.** GRU, LSTM or TCN (4 causal dilated residual blocks), or mean+max temporal pooling; for variant A, the last step only (no temporal context).
- **Head.** Dropout → Linear → logit, trained with BCE, AdamW (lr 3e-4, wd 1e-2), gradient clipping 1.0, up to 40 epochs,
  early stopping on validation AP (patience 8). Training windows are re-sampled every epoch for augmentation.
  The lr / wd / input dropout came from the validation design study. With the initial lr 1e-3 / wd 1e-4, the best validation epoch
  was often epoch 0, i.e. immediate overfitting.
- **Hard negatives** (windows from positive videos that end before the annotated alert time) were evaluated on validation and
  **not adopted**. They lowered window mAP, and with a small margin they conflict with the 0.3–2.0 s positive horizon, because the
  annotated alert lead has a median of 1.5 s.

## 4. Uncertainty
- **Deep ensemble.** M = 5 independently seeded members.
- **MC-Dropout.** S = 20 stochastic passes per member, so 100 samples p_s.
- **Decomposition.** Predictive mean p̄; total entropy H[p̄]; aleatoric E_s H[p_s]; epistemic mutual information MI = H[p̄] − E_s H[p_s].
- **Temperature scaling.** One scalar T fitted on validation by minimising the NLL of the MC-averaged probability: p_s = σ(z_s / T).
- **Abstention.** If MI ≥ its 90th validation percentile **and** the risk score lies in the ambiguous band [CAUTION, HIGH), the output is **UNCERTAIN**.

## 5. Risk engine
risk = clip(p̄ + 0.15 · max(0, slope of p̄ over the last 1 s) + 0.10 · clip(z(looming_center) / 3, 0, 1), 0, 1).
Thresholds are fitted on **validation** data:
- **WARNING (alarm)**: chosen on validation **streams** (full videos). It maximises timely detection (an onset within 3 s before
  the event) subject to ≤ 30 false alarms per hour of normal driving. The window-level F1-optimal threshold was found to be
  degenerate for streaming: it fires on nearly every window.
- **CAUTION**: min(threshold at window recall ≥ 0.90, 0.8 × WARNING).
- **HIGH RISK**: max(threshold at window precision ≥ 0.90, WARNING + ½(1 − WARNING)).

An alarm onset needs ≥ 3 consecutive steps (0.6 s) at or above WARNING. UNCERTAIN steps do not trigger alarms, and no alarm is possible
during the **warm-up**, before one full observation window has been seen. All compared systems use the same validation selection rule.
Triggering factors are the motion cues above their training-set 90th percentile (or below the 10th, for τ), ranked by
their z-distance from the median. Explanation text is filled from a template with those measured values only.

## 6. Explainability
- **Grad-CAM.** The ResNet-18 embedding of the window's last frame is recomputed with gradients, standardised, and inserted
  into the window. The ensemble-mean logit is back-propagated to layer4.
- **Integrated Gradients.** 32 steps from a zero baseline (the training mean in standardised space), averaged over members. This gives
  per-timestep (temporal) and per-feature (motion contribution) attributions, plus the visual vs motion share.
- **Permutation importance.** Drop in test AP when one feature group is shuffled across windows.

## 7. Baselines
- Majority class.
- Logistic regression and XGBoost on pooled window features: motion mean, max, last and last−first, plus a 32-component PCA of the mean visual embedding.
- XGBoost on motion features only.

## 8. v2-light (deployed model)
- **Perception.** Unchanged (YOLOv8n@640 + ByteTrack + Farnebäck), **without ResNet-18**.
  Added a frame-difference descriptor: |I_t − I_{t−1}| on 160×90 grey frames, giving a 4×4 grid of means, the global mean,
  the centre-ROI mean, the fraction of changed pixels and the brightness change (20 values, 0.5 ms/frame).
- **Window features.** For the last W s, each of the 48 channels (28 motion cues, signed-log transformed, plus 20 frame-difference values)
  is summarised by last, mean, max, min, std, least-squares slope, last − first, and the trend (mean of the last third − mean of the
  first third). This gives 384 inputs. Windows are causal and left-padded at the start.
- **Classifier.** LightGBM (400 trees, 15 leaves, lr 0.05, row/column subsampling) in 5 bootstrap-bagged members, resampled by video.
- **Uncertainty.** Spread of the member probabilities (entropy, MI).
- **Calibration.** Platt scaling fitted on out-of-fold dev predictions; the method was chosen by grouped CV between none, Platt and isotonic.
- **Selection.** 5-fold grouped CV on train+val (see EXPERIMENT_DESIGN.md, "Protocol v2.1").
- **Explanations.** Exact occlusion attributions. Per cue: set all aggregates of a channel to the training median and measure Δp.
  Per time step: hold the previous step's values and measure Δp. Also a frame-change heatmap with tracker boxes, and permutation
  importance. Grad-CAM / IG remain available for the v1 model.
- **Risk levels.** WARNING and alarm thresholds come from out-of-fold dev streams (max timely detection with ≤ 30 FA/h).
  CAUTION is set so that ≤ 20 % of normal driving is at CAUTION or above (dev streams).

## 9. v3 (deployed default)
- **Features.** v2 features plus 18 interaction cues computed from tracks (`src/features/relational.py`): scene-relative speed / lateral
  motion / looming, corridor occupancy, lead gap and its rate, lead brake proxy, cut-in score and count, pairwise time-to-meet,
  converging pairs, relative heading, and time-to-reach of the ego point. In v3 these use raw image motion; in v3-stabilized they are
  compensated by the per-step camera homography, which is accumulated over k = 3 steps and maps the box at t−k into frame t.
- **Ego-motion** (`src/features/ego_motion.py`): KLT + RANSAC homography and Farnebäck + RANSAC partial affine, computed on background
  pixels (detections and hood masked).
- **Training.** Positives end 0.3–2.0 s before the event. **Dense negatives**: 40 random windows per normal-driving video instead of 8,
  so the training distribution matches streaming inference. LightGBM ×5 bagged, Platt calibration fitted on OOF dev predictions.
- **Alarm.** Calibrated probability ≥ threshold for 3 consecutive steps. Thresholds are set on OOF dev streams at 20 / 10 / 5 false alarms
  per hour (CAUTION / WARNING / HIGH RISK).
