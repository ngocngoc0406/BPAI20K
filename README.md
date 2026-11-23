# 🛡️ SafeDrive AI

**Explainable and Uncertainty-Aware Early Traffic Collision Prediction System**
*(Hệ thống AI dự đoán sớm nguy cơ va chạm giao thông, có khả năng giải thích và định lượng độ tin cậy)*

SafeDrive AI analyses dashcam video and estimates, **before it happens**, the risk of a collision or near-miss.
The pipeline: YOLOv8 detection → ByteTrack tracking → image-plane motion features + ResNet-18 appearance →
temporal deep ensemble (MC-Dropout, temperature-calibrated) → risk engine (NORMAL / CAUTION / WARNING / HIGH RISK /
**UNCERTAIN**) → Grad-CAM / Integrated-Gradients explanations. It runs on a laptop with a 4 GB GPU, or on CPU.

> Research prototype for the Vingroup applied-AI programme. Not a certified safety system.

## Current model: v3 (default) — see [docs/EARLY_WARNING_V3.md](docs/EARLY_WARNING_V3.md)
YOLOv8n + ByteTrack → 28 image-plane motion cues + 20-d frame-difference descriptor + 18 vehicle-interaction cues
(cut-in, lead gap / closing, pairwise time-to-meet, ...; no CNN) → 8 temporal aggregates per cue (1 s window)
→ LightGBM ×5 trained with dense negatives on 1,276 dev videos → Platt calibration.
Alarm levels are false-alarm-budgeted on dev streams: CAUTION 20 / WARNING 10 / HIGH 5 false alarms per hour.
**v3-stabilized** adds camera ego-motion compensation; use it for shaky cameras.

| Held-out (224 Nexar videos never used for choices) | v3 | v2-light (previous default) |
|---|---|---|
| mAP / ROC-AUC / ECE | 0.620 / 0.816 / 0.028 | 0.607 / 0.809 / 0.055 (v2 config retrained on the same data) |
| Recall at 5 / 10 / 20 false alarms per hour (matched FA) | 0.107 / 0.116 / 0.241 | 0.036 / 0.116 / 0.179 (differences not significant) |
| Camera shake, mAP on test clips | 0.42–0.49 (v3-stabilized **0.60–0.64**) | 0.42–0.50 |
| End-to-end throughput | 42 fps GPU · 19 fps CPU-only | 43 fps GPU · 19 fps CPU-only |

**v3.1 (product features, no model change):** selectable warning sensitivity (Conservative / Balanced / Sensitive = 5 / 10 / 20
false alarms per hour on dev; held-out 2.5 / 8.5 / 16.1 FA/h measured), a **Near-Miss Memory** page (events with clips, measured cues
and explanations), and per-trip safety reports (PDF / JSON). See [docs/SAFEDRIVE_NEXT_VERSION.md](docs/SAFEDRIVE_NEXT_VERSION.md)
and the audit [docs/SAFEDRIVE_V3_AUDIT.md](docs/SAFEDRIVE_V3_AUDIT.md).

**v3.2 Live Driving Assistant** (page *Live Driving Assistant*): webcam or demo video → live boxes / IDs / risk level,
tone warnings with cooldowns, event clips from a 5 s rolling pre-buffer + 5 s post, driving-session reports.
Measured on the test laptop: 9.95 steps/s at the 10 Hz model rate, end-to-end latency 18.5 / 22.8 ms (p50 / p95, GPU) and
56 / 62 ms (CPU only), no RAM growth over 15 min. See [docs/SAFEDRIVE_V32_USER_GUIDE.md](docs/SAFEDRIVE_V32_USER_GUIDE.md) and
[docs/SAFEDRIVE_V32_TEST_REPORT.md](docs/SAFEDRIVE_V32_TEST_REPORT.md).

**v3.3 Real-world validation:** session metadata and consent gating, timeline and full-session recording, a ground-truth annotation
page with a second annotator, real-world metrics, and the *Real-world Evaluation* dashboard. Real driving recorded so far: **0 h**.
See [docs/SAFEDRIVE_V33_REAL_WORLD_VALIDATION.md](docs/SAFEDRIVE_V33_REAL_WORLD_VALIDATION.md),
[docs/SAFEDRIVE_V33_FAILURE_ANALYSIS.md](docs/SAFEDRIVE_V33_FAILURE_ANALYSIS.md) and
[docs/SAFEDRIVE_V33_EVALUATION_PROTOCOL.md](docs/SAFEDRIVE_V33_EVALUATION_PROTOCOL.md).

**v3.4 Data collection campaign** (page *Data Collection Campaign*): frozen-model provenance per session, blind two-annotator
labelling, group-wise partitions with leakage checks, privacy-reviewed blurred copies. Real driving collected: **0 h**. See
[docs/SAFEDRIVE_V34_DATA_COLLECTION_PLAN.md](docs/SAFEDRIVE_V34_DATA_COLLECTION_PLAN.md) and
[docs/SAFEDRIVE_V34_COLLECTION_REPORT.md](docs/SAFEDRIVE_V34_COLLECTION_REPORT.md).

**v3.4.3 Pre-campaign readiness** (page *Campaign Readiness*): 15 evidence-based checks with GO / NO-GO, external-backup
rules with restore tests, a privacy review workflow with rejection and revocation, an operator checklist enforced before every
real-driving session, and an annotator-training pack. Current decision: **NO-GO** (see
[docs/SAFEDRIVE_V343_PRECAMPAIGN_REPORT.md](docs/SAFEDRIVE_V343_PRECAMPAIGN_REPORT.md)).

Early warning remains the open problem: at about 10 false alarms per hour, roughly 1 in 8 collisions gets a valid alarm.
Earlier versions are documented in [docs/LIGHTWEIGHT_OPTIMIZATION.md](docs/LIGHTWEIGHT_OPTIMIZATION.md) (v2) and
[docs/RESULTS_AND_DISCUSSION.md](docs/RESULTS_AND_DISCUSSION.md) (v1). All versions are selectable in the app.

## Quick start (Windows, PowerShell)
```powershell
cd D:\SanphambpAI
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118   # or /whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest -q          # test suite
.venv\Scripts\streamlit run app.py         # http://localhost:8501 (bound to localhost)
```

## Data (Nexar Dashcam Collision Prediction — gated, Nexar Open Data License)
1. Accept the conditions at https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction.
2. Copy `.env.example` to `.env` and set `HF_TOKEN=...`, or run `.venv\Scripts\huggingface-cli login`.
3. Run the pipeline:
```powershell
.venv\Scripts\python scripts\download_nexar.py        # balanced 200 + 200 subset (~8 GB) -> data\raw\nexar
.venv\Scripts\python scripts\prepare_dataset.py       # annotations, validation, video-level split, statistics
.venv\Scripts\python scripts\extract_features.py      # detection + tracking + flow + CNN, cached (resumable)
.venv\Scripts\python scripts\run_experiments.py       # baselines, ablation, ensemble, risk engine, early warning
.venv\Scripts\python scripts\run_robustness.py        # v1 corruption robustness (full pipeline re-run)
# --- v2 lightweight optimisation ---
.venv\Scripts\python scripts\add_framediff.py         # add frame-difference features to caches (decode only)
.venv\Scripts\python scripts\profile_perception.py    # detector / CNN / flow cost profile
.venv\Scripts\python scripts\run_light_cv_select.py   # v2.1: CV selection on train+val, single test evaluation
.venv\Scripts\python scripts\recalibrate_caution.py   # product calibration of CAUTION (dev data only)
.venv\Scripts\python scripts\run_robustness.py --model-dir models/safedrive_v2_light --out robustness_light.json
.venv\Scripts\python scripts\run_cv_study.py          # corroborating 400-video cross-validation
.venv\Scripts\python scripts\benchmark_pipeline.py    # end-to-end FPS / VRAM, all models
# --- v3 early-warning optimisation (full dataset) ---
.venv\Scripts\python scripts\download_nexar.py --per-class 750
.venv\Scripts\python scripts\prepare_full_split.py    # pre-declared split of the 1,100 new videos (dev_new / test_ext)
.venv\Scripts\python scripts\extract_features.py --splits data/processed/all_ids.json --no-cnn
.venv\Scripts\python scripts\add_ego_features.py      # back-fill camera ego-motion into older caches
.venv\Scripts\python scripts\run_ew_study.py --dev full   # stages A/B/C + held-out evaluation
.venv\Scripts\python scripts\deploy_v3.py && .venv\Scripts\python scripts\ew_heldout_compare.py
.venv\Scripts\python scripts\train_v3_variant.py --groups motion,diff,ego_h,rel_h --labels "L7 dense negatives (40/video)" --out models/safedrive_v3_stabilized
.venv\Scripts\python scripts\shake_dev_probe.py
.venv\Scripts\python scripts\run_robustness.py --model-dir models/safedrive_v3 --out robustness_v3.json
.venv\Scripts\python scripts\make_reports.py          # figures, results doc, PDF report, slides
```
Videos are not redistributed with this repository; see the dataset licence.

## Layout
```
app.py, pages/, ui/        Streamlit dashboard (8 pages)
configs/default.yaml       all settings
src/                       data · preprocessing · detection · tracking · features · prediction · uncertainty
                           explainability · risk · evaluation · visualization · reporting · pipeline.py
scripts/                   reproducible pipeline steps
tests/                     unit + integration + dashboard tests
models/safedrive_v1/       trained bundle (ensemble members, scaler, calibration, thresholds)
experiments/               timestamped runs; experiments/latest = most recent
docs/                      audit, dataset report, literature review, methodology, results, manuscript …
reports/                   figures, PDF report, slides
```

## Docker (CPU)
```powershell
docker build -t safedrive-ai .
docker run -p 127.0.0.1:8501:8501 -v ${PWD}/models:/app/models -v ${PWD}/experiments:/app/experiments safedrive-ai
```

## Licences
- Code: research use.
- YOLOv8 (Ultralytics): AGPL-3.0.
- ResNet-18 weights: torchvision (BSD).
- Data: Nexar Open Data License, which requires attribution and prohibits resale and re-identification.
