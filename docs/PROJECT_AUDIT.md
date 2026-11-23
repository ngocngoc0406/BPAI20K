# Project Audit — SafeDrive AI (Stage 1)

Date: 2026-11-19 · Workspace: `D:\SanphambpAI` (the expected `D:\SafeDrive_AI` does not exist; this is the open workspace)

## 1. Starting state
| Item | Finding |
|---|---|
| Existing source code | **None.** The workspace was empty: no `app.py`, `src/`, tests or docs. The project is built from scratch. |
| Existing checkpoints | None |
| Existing dataset | None |
| Existing tests | None |
| Git repository | No |

## 2. Environment
| Item | Value |
|---|---|
| OS | Windows 11 Home 10.0.26200 |
| Python | 3.12.10 (also 3.7 / 3.10 / 3.13 / 3.14 installed; the project uses 3.12 in `.venv`) |
| GPU | NVIDIA GeForce RTX 3050 Laptop, **4 GB VRAM**, driver 462.75 (reports CUDA 11.2) |
| CUDA in PyTorch | torch 2.5.1+cu118 → `cuda.is_available() == True`; works through CUDA minor-version compatibility (verified by running inference and training) |
| RAM | 16.9 GB |
| Disk | D: 137 GB free; **C: only 21 GB free**, so the venv and pip cache are placed on D: |
| Credentials | No Hugging Face / Kaggle token on the machine at audit time |

## 3. Dataset access
- Nexar Collision Prediction is on Hugging Face as `nexar-ai/nexar_collision_prediction`: **gated**, Nexar Open Data License,
  1,500 train videos (750 positive / 750 negative), ~31.4 GB, folders `train/positive`, `train/negative`, plus test-public / test-private.
- Downloading needs the user to accept the conditions and log in. The plan is a balanced subset of 200 + 200 videos (~8 GB), chosen with the user.

## 4. What was built (status at the end of Stage 1)
| Component | Status | Evidence |
|---|---|---|
| Data layer (`src/data/*`) | Working | `tests/test_data.py` |
| Preprocessing (`src/preprocessing/*`) | Working | `tests/test_perception.py` |
| Detection (YOLOv8n) + tracking (ByteTrack) | Working on GPU, ~48 frames/s end-to-end | smoke run + tests |
| Motion features (28) + optical flow + ResNet-18 embeddings | Working | tests |
| Temporal models (frame / pool / GRU / LSTM / TCN) | Working | tests |
| MC-Dropout, ensembles, temperature scaling, ECE / Brier | Working | tests |
| Risk engine (5 levels incl. UNCERTAIN) | Working | tests |
| Explainability (Grad-CAM, IG, permutation) | Working | tests |
| Experiment pipeline | Runs end-to-end (dry run on toy features in scratch space) | — |
| Dashboard (8 pages) | Renders (empty states) | `tests/test_dashboard.py` |
| Real training / results | **Blocked: needs dataset access** | — |

## 5. Final audit (after training)
| Component | Status | Evidence |
|---|---|---|
| Dataset | 400 Nexar videos downloaded, 0 invalid | `docs/DATASET_REPORT.md`, `data/processed/validation.json` |
| Features | 400 / 400 extracted, 0 failures, 40.2 frames/s | `data/processed/extract_summary.json` |
| Training and evaluation | 36 models plus baselines, 4.0 min | `experiments/20261119_020750/` |
| Robustness | 11 conditions × 60 test videos | `experiments/latest/robustness.json` |
| Tests | 50 passed (unit + integration + dashboard with the real model) | `pytest -q` |
| App | Runs on localhost:8501 with the trained bundle | `streamlit run app.py` |

## 6. Issues found and fixed (whole project)
- Hugging Face access check probed a public file (README), so it reported access that didn't exist. It now probes `metadata.csv`.
- Nexar metadata has no label column. Labels are now taken from the folder (dataset-defined ground truth).
- Sequential download ran at about 90 MB/min. Switched to a parallel snapshot download.
- Streaming timelines began at t = W, and could look up a *future* timeline point for events in the first seconds. Timelines now start
  at 0, and a causality assertion was added.
- Alarm warm-up artefact: padded windows at t < W triggered alarms. Alarms are now masked before one full window.
- The window-F1 threshold was degenerate for streaming (0.001). WARNING is now selected on validation streams under an FA/h budget.
- Overfitting: the best epoch was often 0. Fixed with lr, weight decay and input dropout chosen on validation.
- Explainable AI page: the float time selector crashed. Now uses index-based selection, caught by the new dashboard test with a real model.
- fpdf2 cursor positioning broke the PDF report. Fixed.

## 7. Issues found and fixed during the initial audit
1. Shell heredoc writes failed in this environment, so files are written with the editor tool instead.
2. `imageio` was missing, so annotated videos silently fell back to mp4v, which browsers can't play. Installed it, and added it to requirements.
3. `st.page_link` raised an error when a page runs standalone. Now guarded.
4. `results.json` serialization failed on pandas MultiIndex keys. Fixed.
5. The temperature grid lower bound (0.25) was too narrow. Widened to [0.05, 20].

## 8. Optimisation audit (v2, 2026-11-19)
- **Issue:** the first v2 model selection used a 60-video validation split whose ranking did not predict test or CV rankings.
  **Fix:** selection moved to 5-fold CV on train+val. Disclosed in EXPERIMENT_DESIGN.md.
- **Issue:** CAUTION covered 49 % of normal driving, which is unusable. **Fix:** recalibrated on dev streams to ≤ 20 %; the alarm threshold is unchanged.
- **Issue:** explanations could use the wrong model after switching models in the sidebar. **Fix:** each result stores its `model_dir`.
- **Issue:** a backup step briefly started zipping the whole workspace, including the videos. It was stopped, the partial zip was deleted, and the backup was redone selectively (13 MB).
- **Issue:** the evaluation tab broke on the new results format. Caught by the dashboard tests and fixed.

## 9. v3 audit (2026-11-20)
- **GPU out of memory.** Three extraction shards plus a probe on the 4 GB GPU caused CUDA OOM for ~770 videos; no corrupt caches were
  written. Re-run with 2 shards: 1,500/1,500 caches, 0 failures.
- **Laptop sleep** paused the full study for ~5.5 h; it resumed without error. Later jobs kept the system awake with a process-scoped
  `SetThreadExecutionState`; the user's power settings were not changed.
- **Slow bootstrap.** The paired bootstrap recomputed alarms for every resample. It was replaced by per-video, per-threshold tables;
  identical results were verified.
- **Explanations.** Temporal occlusion dropped the ego / interaction arrays for v3-stabilized. Found by an end-to-end check, fixed, and
  covered by a regression test.
- **Unit mismatch.** Alarm levels were thresholded on the risk score although the budgets were fitted on the probability. Fixed with `level_signal`.
