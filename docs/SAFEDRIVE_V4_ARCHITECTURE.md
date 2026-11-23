# SafeDrive AI v4: Architecture

## 1. Repository audit: the v3 system (unchanged, frozen)
```
dashcam video ─► src/detection (YOLO, road-user classes only) ─► src/tracking (ByteTrack + ClassStabilizer v3.4.2)
       │                                                                  │
       ├─► src/features/optical_flow, frame_diff (pixels) ──┐             ▼
       ├─► src/features/ego_motion (homography, flow-affine) ┼─► src/features/motion_features (object motion, 28 ch)
       │                                                     └─► src/features/relational (interaction cues, 18 ch)
       ▼
src/preprocessing/feature_cache (.npz per video: tracks, motion, diff, ego)
       ▼
src/features/window_features (W = 1 s, 8 statistics per channel → 528 features)
       ▼
src/prediction (LightGBM ×5, Platt) ─► src/risk/risk_engine (thresholds at 5/10/20 FA/h) ─► src/live (Live Driving Assistant),
src/safety/near_miss (Near-Miss Memory), src/reporting (PDF/JSON), pages/ (Streamlit), src/realworld (campaign tooling)
```
Frozen: `configs/frozen_v3_hashes.json` covers the model, feature and threshold hashes. The software hash covers `src/`,
`pages/` and `ui/`.

**Reused by v4, read-only:**

| Module | Used for |
|---|---|
| feature caches | tracks, ego transforms, pixel channels |
| `window_indices` and `_aggregate` | identical window construction |
| `derive_relational` and `MotionFeatureComputer` | recomputation for counterfactuals |
| `scripts/run_ew_study.Study` | folds, samples, OOF streams, paired bootstrap |
| `streaming.operating_points` / `evaluate_at` | FA/h-budgeted recall |
| `BaggedEnsemble`, `Calibrator` | ensemble and calibration |

## 2. Isolation of v4
- **Location.** All v4 code is in `safedrive_v4/`, outside `src/`, `pages/` and `ui/`. The v3 software hash and campaign
  provenance therefore do not change (verified: the readiness audit hash items still PASS).
- **What v4 writes:** only `data/processed/v4_graph/`, `models/safedrive_v4_research/` and `experiments/v4/`. It never writes
  `models/safedrive_v3*` or `configs/`.
- **App.** The single V4 product app (`safedrive_v4/app/`, port 8502) replaces the earlier research dashboard. The frozen v3 app on
  8501 is untouched and is kept for the campaign; it is not a second V4 product.

## 3. v4 pipeline
```
tracked detections + camera transform (v3 perception, per step)
  ▼ scene_graph/builder.py        incremental Dynamic Traffic Scene Graph (nodes, 12 typed relations, evidence, temporal validity)
  ▼ reasoning/engine.py + rules   10 rules over the graph + KG (ontology.py) → findings, conflicts, risk evidence, reliability
  ▼ features/graph_features.py    36 SG + 15 KG per-step channels → window statistics (same 8 as v3)
  ▼ prediction (research bundle)  LightGBM ×5 on v3 features ⊕ SG ⊕ KG, Platt   |   GNN (direction B), experimental
  ▼ uncertainty/reliability.py    risk probability · model uncertainty (MI) · perception reliability · OOD percentile
  ▼ warning/policy.py             LOW / MODERATE / HIGH / INSUFFICIENT_EVIDENCE (thresholds from dev OOF at 20 / 10 FA/h)
  ▼ explainability/               Tree SHAP (per warning) · graph chains · counterfactuals (on request)
  ▼ event_memory/trace.py         sealed evidence trace per warning (near-miss memory)
  ▼ dashboard/app_v4.py           replay UI
```

| Folder | Content |
|---|---|
| `safedrive_v4/configs` | versioned JSON config (`config_version`; its SHA-256 is recorded in every artefact) |
| `scene_graph/` | schema and builder |
| `knowledge_graph/` | ontology, rules export |
| `reasoning/` | rules and engine |
| `features/` | graph and KG channels and windows |
| `prediction/` | GNN, graph tensors |
| `explainability/` | SHAP, graph chains, counterfactuals |
| `uncertainty/` | reliability, OOD |
| `warning/` | policy |
| `event_memory/` | evidence trace |
| `dashboard/` | Streamlit app |
| `evaluation/` | precompute, ablation, XAI evaluation |
| `pipeline.py` | streaming orchestration and latency |

Tests are in `tests/test_v4_*.py`.

## 4. Engineering properties
- **Incremental graph.** Only the current step's tracks are processed. Per-track history is bounded (20 steps), the persistence
  deques are bounded (3), and lost tracks are forgotten.
- **Asynchronous explanation.** The warning decision does not wait for explanations. SHAP runs only for new warning onsets (a few
  ms); counterfactuals run only on request (UI button or `xai_eval`). Graph drawing is UI-only.
- **Provenance.** The bundle records `feature_schema` (`v4feat-1.0`), the scene-graph schema (`sg-1.0`), the KG schema
  (`kg-1.0`), the config hash and the training stamp. Loading a bundle with another feature schema raises an error (tested).
- **Logging, typing and errors.** The code is type-annotated. Experiments log to `experiments/v4/*.log`. A missing model gives a
  clear message in the UI.
- **Resource use.** Measured values are in [SAFEDRIVE_V4_TEST_REPORT.md](SAFEDRIVE_V4_TEST_REPORT.md) §4.

## 5. Product round: single app, live mode, near-miss intelligence (V4 product prompt)
```
camera / demo video ─► src/live CaptureThread (v3, newest frame) ─► 10 Hz grid ─► V4LiveEngine (= v3 LiveEngine + ego motion)
     │                                                                              │ rows, camera transform, v3 probability
     └► InputQualityMonitor ─────────────────────────────────────────────────────┐ ▼
                                  V4Pipeline.step: scene graph ─► uncertainty-aware reasoning ─► 5-state policy ─► trace
                                                                                   ▼
                          v3 EventRecorder (5 s before / 5 s after) ─► EventStore (inside the encrypted volume V:)
video file ─► v3 FeatureExtractor (YOLO, ByteTrack, flow, frame difference, ego) ─► V4Pipeline.process_video ─► TripStore
```

| Module | Role |
|---|---|
| `uncertainty/subjective_logic.py` | opinions, conjunction, fusion, discounting |
| `reasoning/uncertain_engine.py` | opinion-based findings, step / temporal risk, observation reliability |
| `uncertainty/input_quality.py` | DEGRADED_INPUT |
| `warning/policy.py` | LOW / MODERATE / HIGH (+ cautious) / INSUFFICIENT_EVIDENCE / DEGRADED_INPUT |
| `event_memory/store.py` | automatic categories (never collision), review (confirm / reject / relabel; collision needs evidence), clips only in the encrypted store, export |
| `app/core.py` | trips, video / clip analysis, annotated clips re-read from the source, trip report and PDF |
| `app/live.py` | live controller on a thread with 10 Hz pacing |
| `app/Home.py` + `pages/1–7` | the 7 product screens |

**Reused from v3, read-only:** `LiveEngine` (subclassed, only `needs_ego`), `CaptureThread`, `CameraSource` / `VideoFileSource`,
`EventRecorder`, `FeatureExtractor`, `draw_frame` / `write_video`, `TabularPredictor`, and the fpdf report style.

**Graceful degradation:**
- camera failure: DEGRADED_INPUT and the loop stops;
- stream gap > 1 s: engine and graph reset;
- encrypted store missing: no clips and no uploads;
- missing replay data: a message instead of a graph;
- errors in the live thread: status "error" with the traceback in the UI.
