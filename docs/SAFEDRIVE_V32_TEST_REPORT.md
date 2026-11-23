# SafeDrive AI v3.2 — Live Driving Assistant: Test Report

Machine: laptop with Intel i5-11400H (6 cores), RTX 3050 Laptop 4 GB, built-in webcam (camera 0, 640×480), Windows 11.
Default model v3, Balanced profile, unless stated otherwise.

Raw results:
- `experiments/live/*.json` (`scripts/live_benchmark.py`, `live_eval_suite.sh`, `live_occupied_test.py`, `live_fidelity.py`);
- summary printout: `scripts/live_summarize.py`.

The older model benchmarks (`experiments/latest/*`) were **not** modified.

## 1. Automated tests
**90 passed** (`pytest -q`, 59 s): the 73 existing tests + 17 new ones (`tests/test_live.py`, plus the new page in `tests/test_dashboard.py`). They cover:
- `AlarmState` vs `alarm_onsets` (50 random sequences);
- **live engine vs offline pipeline: identical probabilities and alarm times**;
- capture latest-frame behaviour and single owner per device;
- missing source; reconnect after a simulated disconnect; give-up and release;
- recorder pre + post clip with the frame count verified by decoding;
- disk-full and quota errors with **no clip fabricated**;
- bounded rolling buffer;
- audio cooldown / escalation / suppression and the wording (no certainty claim);
- a session report without a score;
- controller end-to-end on a demo video; duplicate start refused;
- degraded input suppressing outputs;
- a non-looping demo ending cleanly;
- manual collision confirmation requiring evidence.

## 2. Measured performance (end to end through the real controller)
| Scenario | Duration | Capture FPS | Processed steps/s (target 10) | Late steps | Step time p50 / p95 | **End-to-end latency p50 / p95** | CPU (machine) | GPU util | VRAM (device) | Process RAM |
|---|---|---|---|---|---|---|---|---|---|---|
| Camera, GPU | 5 min | 13.0 | 9.89 | 0 % | 17.5 / 29.0 ms | **19.6 / 54.5 ms** | 13.9 % | 5.1 % | 1.35 GB | 2.96 → 3.72 GB, then flat (+0.18 MB/min) |
| Camera, GPU | **15 min** | 10.0 | 9.95 | 0 % | 17.1 / 20.7 ms | **18.5 / 22.8 ms** | 13.8 % | 5.1 % | 1.35 GB | 2.94 → 3.72 GB, then flat (**−0.09 MB/min**) |
| Camera, **CPU only** | 2 min | 10.0 | 9.80 | 0 % | 54.1 / 59.5 ms | **56.0 / 61.5 ms** | 71.6 % | 0 % | — | 0.70 → 0.72 GB (+1.8 MB/min over 1 min) |
| Demo video (1280×720 @ 30 fps), recording on | 5 min | 30.6 | 9.89 | 19 % | 29.2 / 80.4 ms | 49.5 / 101.3 ms | 17.9 % | 4.2 % | 1.35 GB | 2.99 → 3.90 GB (+2.6 MB/min in the 2nd half) |

Definitions:
- **End-to-end latency** = frame capture timestamp → annotated frame and risk level ready. It excludes display: the UI refreshes at ≤ 4 Hz, which adds ≤ 250 ms on screen.
- **Step time** = detection + tracking + features + prediction + overlay for one 10 Hz step.

The single-model inference speed reported earlier (44.5 fps) is **not** the system rate. The live system processes at the 10 Hz
the models were trained for, and its headroom is step time vs the 100 ms budget: ~17 ms on GPU, ~54 ms CPU-only.

**Interpretation:**
- **Long run.** No growth in camera mode over 15 minutes. The ~0.8 GB increase in the first 30 s is model and CUDA initialisation.
- **Demo with recording.** RAM grew by +2.6 MB/min over 5 minutes, likely from transient clip decode / encode buffers (~70 MB per clip).
  A leak cannot be excluded from a 5-minute run; a ≥ 30-minute recording test is still needed (open item).
- **Demo late steps (19 %).** These come from decoding 1280×720 @ 30 fps in the capture thread on the same CPU. Camera mode had 0 % late steps.

## 3. Audio warnings
| Check | Result |
|---|---|
| Plays without blocking | winsound asynchronous call: **≤ 2.8 ms** API time (max over all runs). Acoustic output delay was not measured; use the "Test sound" button |
| Repetition | Demo 5 min: 60 level onsets → **14 played**, 28 suppressed by the 2 s global gap, 18 by level cooldowns. **Minimum gap between played warnings: 2.2 s.** No repeat of the same level within its cooldown |
| No audio device / backend | `--audio-backend none`: every decision is reported as "unavailable: audio disabled (no backend)", and the UI shows the reason |
| Muted | Reported as "muted by user"; nothing played |
| Spoken Vietnamese | **Not available**: only en-US speech voices are installed. Tones + on-screen Vietnamese text are used instead |
| Audible output | The automated runs used volume 0 so as not to disturb the user. **Audibility was not verified by a human listener** |

## 4. Event recording
| Check | Result |
|---|---|
| Clips saved and correct | Demo 5 min (Sensitive): 9 events (3 warning, 6 potential near-miss), **9/9 clips saved**. 8 have 100 frames = 5 s pre + 5 s post at 10 Hz (re-decoded count = written count); 1 has 95 frames because the stop arrived inside its post window (truncation is noted in the event) |
| No-audio run | 2 events, 2/2 clips with 100 decoded frames |
| Disk almost full (simulated: minimum free space above the real free space) | 2 events logged with `clip_status = not saved: insufficient disk space (…)`; **no clip file written**; reported in the UI as a recording error |
| Quota exceeded | Unit test: event logged with "quota exceeded"; no clip |
| Categories | Only risk_event / warning_event / potential_near_miss are produced automatically; confirmed_collision is manual with evidence (test) |

## 5. Camera robustness
| Scenario | Result |
|---|---|
| **Disconnect** (device force-released under the capture thread at 30 s) | 1 interruption detected, **reconnected after 2.3 s**; stream state reset (warm-up again); run continued to the end |
| Permanent loss (unit test) | Gives up after the timeout, state "failed", device released, clear error |
| **Camera held by another application** | Windows allowed shared access. SafeDrive received only **~1 frame per 1.5 s**. Before the fix the UI showed a stale "NORMAL". **Fixed:** a DEGRADED INPUT state (median input interval > 0.25 s) suspends warnings and events and shows a clear message. In one of four runs no frame arrived within 5 s; the new "no new frames for N s" warning covers that case. After the other app releases the camera, SafeDrive starts normally |
| Normal camera, false degraded flags | 2 of 273 steps at camera start-up (first frames slow), within the 1 s warm-up when no alarm is possible |
| No GPU | CPU-only works: 9.8 steps/s, step 54 ms, e2e 56 ms |
| Duplicate camera stream | Refused (same device in the same process: `DeviceBusyError`; second controller start: `RuntimeError`) |
| Non-looping demo video | **Bug found and fixed:** at the end of the file the capture thread treated EOF as a disconnection and "reconnected", replaying forever (and counting an interruption). It now ends cleanly with status "demo video ended" (regression test) |

## 6. Fidelity: live vs evaluated (offline) model
- **Frame-identical input** (frames fed on the offline grid): probabilities and alarm times are **identical** for v3, v3-stabilized and
  v2-light (`tests/test_live.py`; held-out 00288 on CPU). This required a 100-step prediction buffer; 40 steps gave deviations up to 0.0097.
- **Real-time playback** (newest-frame sampling, wall-clock jitter), 6 held-out videos (`experiments/live/fidelity.json`):
  - valid-alarm agreement **6/6**: the 2 offline alarms are reproduced at the same times (15.2 s and 19.4 s); 0 vs 0 on normal videos;
  - per-step |ΔP| mean **0.034**, max 0.23.

  This sample is small. In an earlier real-time run of 00288 the alarm did **not** fire (P peaked at about 0.9 vs threshold 0.925), so
  results near the threshold vary between runs.

## 7. Effect on existing models
- No model, threshold, feature definition (FEATURE_VERSION = 1), held-out data or earlier benchmark file was changed.
- The offline pipeline code path is unchanged.
- The live engine reproduces offline predictions exactly when given the same frames.

## 8. Not done or open
| Item | Status |
|---|---|
| Spoken Vietnamese warnings | Not available (no vi-VN voice); tones + on-screen text |
| Human check of audibility | Not done; "Test sound" button provided |
| ≥ 30 min recording session (memory trend with clips) | Not done |
| Real vehicle / road test | Not done. The webcam runs measured system behaviour on an indoor scene, **not** warning quality |
| Real disk-full | Simulated via the minimum-free-space threshold |
| Physical camera unplug | Simulated via forced release |
