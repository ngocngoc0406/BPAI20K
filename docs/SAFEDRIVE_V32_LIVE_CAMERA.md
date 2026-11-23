# SafeDrive AI v3.2 — Live Driving Assistant: Architecture

Status: research prototype. No braking or steering, and no claim of collision prevention. All processing is local.
Backup before the changes: `backups/v3_1_before_live_camera/` (v3 / v3-stabilized bundles, experiment results, Near-Miss store, source).

## 1. Audit of the existing system (Phase 1)
| Area | Finding | Consequence for live mode |
|---|---|---|
| Offline inference | `SafeDrivePipeline.analyze()` → `FeatureExtractor._extract()` decodes the **whole file**, keeps all frames, then predicts the timeline | Not usable for a camera; frame-by-frame logic must be re-hosted |
| Per-frame components | `Tracker` (YOLOv8n + ByteTrack), `MotionFeatureComputer`, `FlowComputer`, `FrameDiff`, `EgoMotionEstimator`, `VisualEncoder` are stateful per-step objects | **Reused unchanged**, called in the same order as the extractor |
| Model calls | `load_predictor()` → `TabularPredictor` (v2 / v3) or `EnsemblePredictor` (v1); both expose `predict_at(VideoFeatures, t_ends)` | Reused; a ring buffer is wrapped as a small `VideoFeatures` |
| Risk / alarms | `RiskEngine.assess()` on a timeline (1 s trend); alarms via `alarm_onsets` (k consecutive steps, re-arm), warm-up, profiles, optional EMA | Re-implemented **incrementally** (`AlarmState`), proven identical to `alarm_onsets` |
| Near-Miss Memory | `data/memory/events.jsonl` (JSON lines) + `data/memory/clips/*.mp4`; `NearMissMemory` reads, updates and deletes | Live events are written with the same core fields plus live fields (category, clip_status, …) |
| Overlay / video | `draw_frame`, `write_video` (H.264 via imageio-ffmpeg) | Reused |
| Feature definitions / models / thresholds | unchanged (FEATURE_VERSION = 1); no retraining | Live mode must reproduce offline predictions |

## 2. Design (new package `src/live/`, separate from the old pipeline)
```
 camera / demo file ──► CaptureThread ──(latest frame only)──► inference thread (10 Hz grid) ──► shared state ──► Streamlit fragment (≤ 4 Hz)
                         reconnect, 1 owner per device            LiveEngine.process()                 │
                                                                   ├─ ring buffers (100 steps)          ├─ AudioAlerter (cooldowns)
                                                                   ├─ predict every 2 steps             ├─ EventRecorder (rolling JPEG buffer → clip writer thread)
                                                                   └─ RiskEngine + AlarmState           └─ SessionStats → JSON / PDF report
```
| Module | Responsibility |
|---|---|
| `sources.py` | `CameraSource` (OpenCV DirectShow, buffer size 1) and `VideoFileSource` (demo; real-time pacing, loop); `list_cameras()` |
| `capture.py` | `CaptureThread`: reads continuously and keeps **only the newest frame** (no queue growth); one capture per device (registry, `DeviceBusyError`); reconnect with back-off; counts interruptions; always releases the device |
| `engine.py` | `LiveEngine`: per grid step resize → detect + track → flow → frame diff → (ego) → motion → (CNN) → ring buffers. Every 2 steps: `predict_at` on the last 100 steps, risk assessment on the last 20 predictions, profile threshold, warm-up, EMA, `AlarmState`. Renders the annotated frame |
| `controller.py` | `LiveController`: owns the threads; 10 Hz scheduling with the newest frame, **never processing a frame twice** and never bursting to catch up; resets stream state after a capture gap > 1 s; event categories; audio on level onsets; session statistics; periodic metric samples |
| `recorder.py` | `RollingBuffer` (pre-event JPEGs bounded by seconds and MB) + `EventRecorder` (post-event collection, disk-space and quota checks, background clip writer, decode verification, explicit `clip_status`) |
| `audio.py` | `AudioAlerter`: synthesized tones (winsound, local), per-level cooldown + global gap + escalation rule, evidence-based Vietnamese on-screen messages |
| `session.py` | `SessionStats` (level onsets, alarms, events, clips, audio, FPS, step / end-to-end latency, interruptions) and the JSON / PDF report, **without** a safety score |
| `pages/10_Live_Driving_Assistant.py` | UI: source, model, sensitivity, device, recording and audio settings; Start / Stop camera; Start / Stop session; live view in `st.fragment(run_every=0.25)` |

## 3. Equivalence with the evaluated (offline) models
- `AlarmState` matches `alarm_onsets` on 50 random sequences (`tests/test_live.py`).
- With frames fed on the offline grid, `LiveEngine` reproduces the offline probabilities **exactly** (max |ΔP| = 0) and the same alarm
  times for v3, v3-stabilized and v2-light (`tests/test_live.py`; held-out video 00288).
- **Buffer length matters.** With a 40-step prediction buffer, v3 differed by up to 0.0097 (4 of 157 steps), because the interaction
  cues carry state across steps. A 100-step buffer gives exact equality at about +25 ms per prediction (every 0.2 s). The only
  remaining divergence is a lead track that vanishes for > 10 s and then returns.
- **Real-time operation is not frame-identical to offline.** Live mode takes the newest available frame at each 0.1 s tick, so the
  frames — and therefore the probabilities — differ slightly from the offline grid. The size of this effect is measured in the test report.

## 4. Event semantics (honest naming)
| Category | Rule |
|---|---|
| risk_event | P ≥ CAUTION level for ≥ k steps (0.6 s) without an alarm (optional) |
| warning_event | an alarm onset (P ≥ profile threshold for k steps, after warm-up, not UNCERTAIN) |
| potential_near_miss | an alarm episode whose level reached HIGH RISK; "potential" because it is not verified |
| confirmed_collision | manual only (Near-Miss Memory page), and requires an evidence note |

One clip per episode: an alarm during an ongoing risk episode upgrades it instead of creating a duplicate. The clip contains the
pre-event buffer plus the post-event frames. If the clip cannot be written, the metadata is still stored, with
`clip_status = not saved: <reason>`.

## 5. Privacy
- Nothing leaves the computer: no upload, no cloud, no network calls in the live path.
- There is no continuous recording. Only event clips (pre + post seconds) are written, and only when recording is enabled.
- The camera benchmarks were run with recording disabled, so no camera image was written to disk.

## 6. Changes made during testing
- **DEGRADED INPUT.** When the median interval between processed frames exceeds 0.25 s (the models assume 0.1 s), warnings, audio and
  events are suspended and the UI says so. When no new frame arrives for > 2 s, a "no new frames" warning is shown. This was found with
  the camera shared with another application (~1 frame per 1.5 s); before the fix the UI showed a stale NORMAL.
- **End of a demo video.** The capture thread treated EOF as a disconnection and replayed the file forever. Sources now report `eof`,
  and the stream ends with status "demo video ended".
- **Clip writer flush.** It waited only until the queue was empty, not until each clip was written. It now uses task accounting
  (`unfinished_tasks`), so a stop never leaves an event "pending".
