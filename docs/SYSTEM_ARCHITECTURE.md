# System Architecture

```
video ──► video_reader ──► frame_sampler (10 Hz grid) ──► resize 640w ──► [optional corruption]
                                                              │
         ┌────────────────────────────┬───────────────────────┼──────────────────────┐
         ▼                            ▼                       ▼                      ▼
  YOLOv8n detection ─► ByteTrack   Farnebäck flow        ResNet-18 (frozen)     brightness
         │  track ids, boxes          │ 4 stats               │ 512-d               │
         └──────────► MotionFeatureComputer (28 features) ◄───┘                     │
                              │                                                     │
                     feature_cache (.npz per video) ◄───────────────────────────────┘
                              │
             windows (W s, causal) + Standardizer (train stats)
                              │
          CollisionModel ×5 (deep ensemble; GRU/TCN/... chosen on val)
                              │ MC-Dropout ×20 per member, temperature T
                              ▼
          p_mean, p_std, entropy, aleatoric, mutual information
                              │
                RiskEngine (val-tuned thresholds, trend, looming, UNCERTAIN)
                              │
      ┌───────────────┬───────┴─────────┬───────────────────┐
      ▼               ▼                 ▼                   ▼
  Streamlit UI   Grad-CAM / IG     exports (CSV/JSON/   early-warning
  (8 pages)      explanations      PDF/annotated MP4)   evaluation
```

## Modules
| Package | Responsibility |
|---|---|
| `src/data` | Annotation parsing, video discovery, validation, video-level stratified split, statistics, HF download |
| `src/preprocessing` | Robust decoding, FPS normalisation onto a time grid, corruptions, feature cache |
| `src/detection`, `src/tracking` | Ultralytics YOLO + ByteTrack, track-quality statistics |
| `src/features` | Optical flow, CNN embeddings, image-plane motion features, the per-video extraction driver (OOM → CPU fallback, cancellation, duration limit) |
| `src/prediction` | Window construction and labelling, models, trainer, baselines, model bundle and streaming predictor |
| `src/uncertainty` | MC-Dropout / ensemble sampling and decomposition, temperature scaling, ECE, Brier, reliability |
| `src/explainability` | Integrated Gradients, Grad-CAM through the full model, permutation importance |
| `src/risk` | `risk_engine.py`: levels, risk score, triggering factors, template explanations |
| `src/evaluation` | Classification, Nexar mAP, selective risk, early-warning metrics |
| `src/visualization`, `src/reporting` | Overlays, H.264 writer, CSV/JSON/PDF reports |
| `src/pipeline.py` | `SafeDrivePipeline.analyze()` / `render()`, used by the app |
| `ui/`, `app.py`, `pages/` | Streamlit dashboard |
| `scripts/` | download → prepare → extract → experiments → robustness → reports |

## Model bundle (`models/safedrive_v1/`)
`member_i.pt` (state dict + hyper-parameters), `scaler.npz` (train statistics), and `bundle.json`
(window length, temperature, MC samples, F1 threshold, risk configuration including cue reference
quantiles, training stamp, experiment directory).

## Engineering
Settings come from `configs/default.yaml`; logs go to `experiments/logs`; secrets are read from `.env` (never committed).
Without a working GPU the system runs on CPU (or force it with `SAFEDRIVE_DEVICE=cpu`). The app is bound to localhost.

## v2-light (default) vs v1
- `src/features/frame_diff.py` computes the frame-difference descriptor (computed by the extractor in both modes).
- `FeatureExtractor(use_cnn=False)` is the light mode: ResNet-18 is not loaded.
- `src/features/window_features.py` builds the vectorised window aggregates.
- `src/prediction/light_models.py` holds the classifiers, bagging and calibration.
- `src/prediction/tabular_predictor.py` provides streaming prediction and occlusion explanations.
- `src/pipeline.load_predictor` dispatches on `bundle.json` `type` (`tabular` → v2, otherwise v1) and only loads the CNN if the model needs it.
- The dashboard sidebar selects the bundle: `models/safedrive_v2_light` (default) or `models/safedrive_v1`. Each analysis result remembers
  the bundle that produced it, so explanations always use the same model.
- Feature cache v2: `.npz` files gain a `diff` array (`scripts/add_framediff.py` back-fills v1 caches).

## v3 additions
- Feature cache v3 adds an `ego` array (T × 30: homography and flow-affine transforms + stats). `scripts/add_ego_features.py` back-fills it.
- `src/features/relational.py` derives interaction cues on the fly from cached tracks (+ ego for compensated variants), memoised per video.
- `FeatureExtractor(use_ego=...)`: ego-motion is computed only when the selected model needs it (v3-stabilized).
- `RiskConfig.level_signal = "probability"` for v3: levels and alarms use the calibrated probability with FA-budgeted thresholds.
  The trend/looming risk score is still computed and displayed separately.
- Model selector: v3 (default), v3-stabilized, v2-light, v1.
