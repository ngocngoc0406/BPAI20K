# SafeDrive AI v3.3: Real-World Evaluation Protocol

This protocol says how SafeDrive AI is evaluated outside the benchmark, how data are collected and labelled, and how each
number is computed. It applies to the frozen v3 / v3-stabilized models. During evaluation, no model, feature definition,
threshold or benchmark is changed.

## 1. Four kinds of evidence (never pooled)
| Kind | What it is | What it can show | What it cannot show |
|---|---|---|---|
| **Benchmark video** | Held-out Nexar dashcam videos (224): real recorded driving, labelled by the dataset | Recall, FA/h and lead time under the dataset's sampling; results by verified dataset metadata | Behaviour on *this* camera, mount and road network; the true real-world FA/h (Nexar negatives are short, selected clips) |
| **Live camera test** | Laptop webcam or demo replay through the full live pipeline, not in a vehicle | Runtime, latency, stability, recording, audio, fault handling | Accuracy |
| **Controlled test** | Synthetic corruptions of held-out videos (shake, occlusion, light change, …); staged system faults (disconnect, busy camera, full disk, 30-min run) | Sensitivity to each factor in isolation; failure handling | Real frequency of these conditions |
| **Real driving** | Supervised sessions recorded in a vehicle with this system | Real FA/h, repeated warnings, recall / lead time on annotated situations, runtime in the car | Anything beyond the declared conditions and sample size |

Each kind is reported in its own table and its own dashboard tab (**Real-world Evaluation**, tabs ① to ④).
Groups recorded under different conditions are never merged into one number.

## 2. Safety and consent rules (mandatory)
1. **Configure before the vehicle moves.**
   - Before driving, fill in the trip metadata on the Live Driving Assistant page and press *Start camera* and *Start session*.
   - The driver never operates the screen while driving. Stop the session only after parking.
2. **A second person (supervisor) is present** in every real-driving session. They watch the road, not the screen.
3. **The prototype is never the only safety aid.** Its warnings may be wrong or missing. The driver drives normally.
4. **No dangerous situations are created** on public roads. Cut-ins and hard braking are only labelled when they happen naturally.
   Staged scenarios belong to a closed area (data kind *controlled_test*).
5. **Consent.**
   - A real-driving or controlled session cannot start without: driver consent, a supervisor, and setup before moving.
     The software enforces this (`ConsentError`).
   - Full-session video additionally requires that all occupants are informed.
6. **Privacy.**
   - Everything is processed and stored locally. Nothing is uploaded.
   - Detected people are blurred in the full-session video by default.
   - **Licence plates are not detected by the current models, so they are not blurred.** Do not share full-session video outside
     the research team. Before any release, a plate-blurring step is required.
   - Avoid recording precise personal locations: the location note is free text and optional.
   - **Deletion:** *Ground Truth Annotation → Delete this trip's data* removes the session folder (timeline, video, annotations,
     report) and its event clips and metadata in Near-Miss Memory.

## 3. Session metadata (declared, never inferred)
Stored in `data/sessions/<id>/trip_meta.json`. Every field has the value **unknown** unless the person declares it.
The software never fills it in from the images.

| Field | Values |
|---|---|
| data kind | real_driving / controlled_test / live_camera_test / demo_video |
| lighting | unknown, day, night, low_light, backlight, dawn_dusk, tunnel_changing |
| weather | unknown, clear, cloudy, rain, fog, snow |
| road types | straight, curves, urban, highway, rural, intersection_heavy, roundabout (multiple) |
| traffic | unknown, sparse, moderate, dense |
| camera mount | unknown, fixed_windshield, fixed_dashboard, handheld, vibrating_loose |
| consent | driver_consent, occupants_informed, supervisor_present, configured_before_driving |
| recording | record_full_video, blur_people |
| review | fully_reviewed_by (annotators who watched the whole session) |

## 4. What is recorded per session
| File | Content |
|---|---|
| `timeline.csv` | One row per processed 10 Hz step: stream time, wall time, level, P(collision), risk score, mutual information, alarm (0/1), DEGRADED flag, capture FPS, step ms, end-to-end ms, number of objects, road user in the corridor, RSS memory every 5 s |
| `audio_log.jsonl` | Every audio decision: level, message, played or suppressed, and the reason |
| `session_report.{json,pdf}` | Summary and events (category, P peak, clip status, frames in the clip) |
| `video/segment_NNN.mp4` | (Opt-in, consent) the full session at 10 fps in 5-min H.264 segments, without overlays, with people blurred |
| `annotations_<name>.jsonl` | Each annotator's labels (one file per annotator, so reviewers stay independent) |

## 5. Data-collection plan
- **Target:** ≥ 10 h of normal driving with full-session video, which gives a real FA/h with a usable interval.
  With 0 false alarms in 10 h, the 95 % upper bound is 0.37 FA/h. With 50 false alarms, the interval is about 3.7 to 6.6 FA/h.
- **Spread over the declared conditions:**
  - day and night;
  - urban, highway and curves;
  - sparse and dense traffic.
- **Fixed camera mount.** Note the model and profile used. Run sessions of about 30 to 60 min.
- **Warning-worthy situations are rare.** Estimating recall to ± 10 % needs about 100 annotated warning-worthy events.
  This takes far more than 10 h of normal driving, and must not be forced by creating danger. Until then, recall on real roads
  is reported with its (wide) interval or as "not estimable".

## 6. Annotation procedure
1. **Blind first pass.**
   - The annotator enters their name. *Blind mode* is on by default: model output, event categories and the other reviewer's
     labels are hidden.
   - The annotator watches the full-session video segments. Session time = the segment's start time + the video position.
2. For every situation that needs the driver's attention, add an annotation:
   - **Category:**
     - `normal`: a reviewed stretch with nothing notable;
     - `potential_risk`: worth watching, no warning needed;
     - `warning_worthy`: a warning would have helped;
     - `potential_near_miss`: a close call (evasive action or a very small gap);
     - `confirmed_collision`: requires an evidence note;
     - `uncertain`: insufficient evidence.
   - **Times:**
     - risk start: when it became visible;
     - most critical moment: when a warning was needed at the latest, e.g. the start of braking or the closest approach;
     - risk end.
   - **Tags, confidence and note.**
3. After watching the whole session, press **"I reviewed the WHOLE session"**. Recall is only computed for fully reviewed sessions.
4. **Second annotator.**
   - A different person annotates the same session independently, also in blind mode.
   - After both finish, switch blind mode off to see the agreement:
     - one-to-one matching by temporal IoU ≥ 0.3;
     - Cohen's kappa on categories, and on warning-worthy vs. not;
     - situations found by only one reviewer;
     - mean onset difference.
   - Disagreements are discussed. The reference annotator is recorded. Labels are never changed to match the model.
5. **Clip-only review** (no full video) can label individual alarms, but cannot support recall or an FA rate (§7).

## 7. Metric definitions (src/realworld/metrics.py)
| Metric | Definition |
|---|---|
| hours | non-DEGRADED processed steps × 0.1 s (wall-clock hours are also reported) |
| alarm onset | a 0 → 1 transition of `alarm` on non-DEGRADED steps (deployed rule: P ≥ profile threshold for 3 consecutive steps) |
| reference time *t\** of a situation | the annotated most critical moment, else the risk end |
| **valid warning** | an onset in [*t\** − 5 s, *t\**) of a warning-worthy situation (warning_worthy / potential_near_miss / confirmed_collision), same 5 s horizon as the benchmark |
| **false alarm** | an onset valid for no warning-worthy situation and not inside an `uncertain` situation. Uncertain onsets are counted separately and excluded |
| **FA/h** | false alarms / hours, **only when every alarm was reviewed** (full review). Otherwise unreviewed alarms are shown as such and no rate is given |
| warning without hazard | an onset overlapping no annotated situation of any non-normal category (window [start − 5 s, end]) |
| repeated warning | an onset < 30 s after the previous onset |
| warning recall | warning-worthy situations with a valid warning / all warning-worthy situations (full review only) |
| CAUTION event recall | warning-worthy situations with P ≥ CAUTION in [*t\** − 5 s, *t\**) |
| warned ≥ 1 / 2 / 3 s | share of warning-worthy situations whose first valid onset was ≥ x s before *t\** |
| lead-time distribution | *t\** − first valid onset, for warned situations |
| runtime | step ms and end-to-end ms p50 / p95, processed rate, DEGRADED share, RSS trend (MB/min) |
| recording | event clips saved / events, clip completeness (frames / expected), timeline missing and duplicate steps (stream resets after camera gaps counted separately), storage MB per hour, full-video frames written / dropped |

Intervals:
- recall uses a 95 % Wilson interval;
- FA/h uses an exact Poisson interval;
- when comparing sessions or models, bootstrap over sessions (not steps).

## 8. Stress-test matrix
| Factor | Where it is tested | Status in v3.3 |
|---|---|---|
| Camera shake | Controlled: `shake` corruption sev 1/2 (v3, v3-stabilized) | measured (v3.1) |
| Partial occlusion | Controlled: `occlusion` sev 1/2 | measured |
| Rapid light change (tunnel) | Controlled: new `light_change` sev 1/2 (STRESS_V33, separate from the original benchmark), window level + full streams paired with clean | measured in v3.3: degradation of detections / P; alarm-level effect not testable (n = 30 + 30) |
| Low light, noise, blur, frame drop | Controlled corruptions | measured |
| Curves, turning | Benchmark: turn proxy (measured horizontal camera shift); real: `curves` road type + `curve` tag | proxy only; real data 0 h |
| Lane change, many vehicles, motorcycle cut-in, lead braking | Real driving tags (natural occurrences only) | 0 h |
| Disconnect / reconnect | Live: cam_disconnect | measured (v3.2) |
| Camera used by another app | Live: cam_occupied (DEGRADED INPUT) | measured (v3.2) |
| Near-full disk | Live: demo_disk_full (simulated) | measured (v3.2) |
| ≥ 30 min continuous run with recording | Live: demo_30min_record (event clips + full-session video) | passed in v3.3 (1800 s, 54/54 clips, 0 dropped frames) |

## 9. Commands
```powershell
.venv\Scripts\python scripts\realworld_benchmark_strata.py        # benchmark strata + H3 proxies -> experiments/latest/v33_*
bash scripts/v33_stress_suite.sh                                    # 30-min run + window-level light change
.venv\Scripts\python scripts\v33_light_change_stream.py --keep-awake # full-stream paired light change
.venv\Scripts\python scripts\v33_temporal_overlap.py               # score overlap / temporal consistency
.venv\Scripts\python scripts\live_benchmark.py --name <n> --source camera:0 --minutes 30 --data-kind live_camera_test --keep-awake
.venv\Scripts\python -m pytest tests -q
```

## 10. Status at the end of v3.3 (2026-11-21)
- Real-driving hours: **0**.
- Annotated sessions: **0**.
- Second-annotator sessions: **0**.
- The first real-data step is a supervised session with full video, following §2–§6.
