# SafeDrive AI v3.2 — Live Driving Assistant: User Guide

> **Research prototype.** Warnings are model estimates from image-plane cues and can be missed or false. With the default
> model, about 1 in 9 collisions was warned at about 9 false alarms per hour on held-out data (offline evaluation).
> The system **does not brake or steer** and does **not** guarantee collision avoidance. Never look at the screen while driving.

## 1. Start the app
```powershell
cd D:\SanphambpAI
.venv\Scripts\streamlit run app.py
```
Open http://localhost:8501 and choose **Live Driving Assistant** in the sidebar. Everything runs on this computer:
no video, clip or report is uploaded anywhere.

## 2. Settings (fixed while running unless noted)
| Setting | Meaning |
|---|---|
| Input: Camera / Demo video | **Camera** = a camera recognised by Windows (index 0, 1, …). **Demo video** = a held-out test clip or an uploaded MP4, played in real time (use it when no camera is available) |
| Prediction model | v3 (default), v3-stabilized (for shaky mounts), v2-light, v1 |
| Warning sensitivity | Conservative / Balanced / Sensitive = alarm thresholds set at about 5 / 10 / 20 false alarms per hour on development data (v3 and v3-stabilized only) |
| Device | auto (GPU if available) or cpu |
| Record events | Saves a clip (pre + post seconds) and metadata for each event; pre-event frames come from a rolling buffer in RAM |
| Pre / post-event | 1–15 s each (default 5 / 5) |
| Clip storage quota | Maximum size of all stored clips; when exceeded, new events are logged **without** a clip, with the reason shown |
| Also record risk events | Record CAUTION episodes too (otherwise only alarm-based events) |
| Audio warnings, Volume, Sound for CAUTION | **Can be changed while running.** "Test sound" plays the WARNING tone once |

## 3. Running
1. Press **▶ Start camera**. If the camera is missing or used by another application, a clear error is shown; close the other app or switch to *Demo video*.
2. The live view refreshes up to 4 times per second. It shows boxes with tracking IDs, the current level (NORMAL / CAUTION / WARNING / HIGH RISK / UNCERTAIN), P(collision), uncertainty, capture FPS, step time, end-to-end latency, the number of objects, and camera interruptions.
3. Press **● Start session** to collect statistics, then **■ Stop session** for the session report (PDF / JSON). Events are recorded even without a session, but statistics and the report need one.
4. Press **⏹ Stop camera** to release the camera. It is also released automatically on errors.

## 4. Audio warnings
| Level | Tone | On-screen message |
|---|---|---|
| CAUTION | 1 soft low chime | "Có tình huống giao thông cần quan sát." |
| WARNING | 2 medium beeps | "Chú ý phương tiện phía trước." (when a road user is in the driving corridor), otherwise "Có tình huống giao thông cần quan sát kỹ." |
| HIGH RISK | 3 fast high beeps | "Nguy cơ va chạm, hãy kiểm tra tình hình phía trước." |

- **Anti-repetition.** Per-level cooldown (CAUTION 20 s, WARNING 8 s, HIGH RISK 5 s) plus a 2 s global gap. A higher level may sound after the gap; the same or a lower level is suppressed while its cooldown (or a higher level's) runs.
- **No spoken Vietnamese.** This computer has only English (en-US) speech voices, so the message is shown on screen. Installing a Vietnamese voice is a possible future addition.

## 5. Events and Near-Miss Memory
| Category | When |
|---|---|
| risk_event | P ≥ CAUTION level for ≥ 0.6 s without an alarm |
| warning_event | an alarm (P ≥ profile threshold for 3 steps) |
| potential_near_miss | an alarm episode that reached HIGH RISK. *Potential*: not verified |
| confirmed_collision | **only set manually** in Near-Miss Memory, with an evidence note |

Live events appear in **Near-Miss Memory** under the trip `live-<session id>` (or `unsessioned-<date>`), with a clip, measured cues and explanation.
If a clip could not be written (disk space below the minimum, quota exceeded, write error), the event is shown with `clip_status = not saved: <reason>`. No clip is fabricated.

## 6. Where data is stored
| What | Location |
|---|---|
| Events (metadata) | `data/memory/events.jsonl` |
| Clips | `data/memory/clips/<event_id>.mp4` |
| Session reports | `data/sessions/<session_id>/session_report.{json,pdf}` |

Delete events from the Near-Miss Memory page, or delete these folders to remove all recordings.

## 7. Troubleshooting
| Symptom | What to do |
|---|---|
| "Camera … cannot be opened" | Close Zoom / Teams / Camera app; check Windows camera privacy settings; try another index; use Demo video |
| Low processed rate (< 10 Hz) or high step time | Use GPU (device auto); close other heavy apps; the CPU-only mode was measured at the rate in the test report |
| No sound | Use "Test sound"; check the Windows output device and volume; audio status is shown in the live panel |
| "Recording error" | Free disk space or raise the quota; delete old events |

## 8. DEGRADED INPUT and "no new frames"
- If frames arrive slower than the models need (median interval > 0.25 s instead of 0.1 s), the panel shows **DEGRADED INPUT**.
  Warnings and event recording are paused until the rate recovers. The usual cause is another application using the same camera.
- If no new frame arrives for more than 2 s, a warning says the displayed level is not current.
- A non-looping demo video stops at its end with "demo video ended".
