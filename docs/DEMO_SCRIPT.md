# Demo Script — SafeDrive AI (5–7 minutes)

**Model.** The default is v2-light (sidebar: *Prediction model*). Suggested clips:
- **00288**: v2 raises one WARNING at 18.0 s, before the collision at 18.92 s.
- **00901**: v1 warns 2.7 s early, v2 misses it. Use this to show honestly that early warning is the open problem.
- Mention that v2 runs without a GPU: 20 fps end-to-end on CPU.

**Setup (before the demo).** Run `.venv\Scripts\streamlit run app.py` and open http://localhost:8501.
Pre-analyse one positive and one negative **test** video so the cache is warm.
Suggested clips: pick from *Video Analysis → Dataset test sample*. The list shows which clips are collision / near-miss and which are normal.

| Time | Page | What to show | What to say |
|---|---|---|---|
| 0:00–0:45 | Overview | Hero, architecture diagram, status cards | "SafeDrive predicts collision risk **before** it happens from a single dashcam. It also says how sure it is and shows why it warned. Everything runs on this laptop's 4 GB GPU." |
| 0:45–2:00 | Video Analysis | Select a positive test clip → **Analyse**. Show the progress bar, the annotated video with IDs and boxes, and the frame inspector at the alarm time | "YOLOv8 detects road users, ByteTrack keeps their IDs, and 28 motion cues are measured every 0.1 s. This clip was never seen in training." |
| 2:00–3:15 | Collision Prediction | Probability band, risk score, thresholds, the ALARM line vs the GT event line, the level strip | "The red line is our first alarm and the white line is the annotated collision, so the gap is the warning time. The CAUTION / WARNING / HIGH thresholds were fixed on validation data." |
| 3:15–4:15 | Explainable AI | Grad-CAM at the alarm, factors text, temporal IG bars, motion-contribution bars | "The heatmap is where the model looked. The factors are measured values, for example 'object in the driving corridor growing 0.8 /s, typical 0.0'. No language model invents reasons." |
| 4:15–5:00 | Uncertainty | MI timeline, UNCERTAIN markers; reliability diagram, ECE before/after | "When the five ensemble members disagree, the system says UNCERTAIN instead of guessing. After temperature scaling, its probabilities are calibrated." |
| 5:00–6:00 | Evaluation | Baselines table, ablation A–E, early-warning tab (FA/h), robustness tab | "Every number here comes from experiment files: baselines, ablation, false alarms per hour of normal driving, and corrupted-video robustness." |
| 6:00–6:30 | Reports | Download the PDF and the annotated video | "Each analysis exports to CSV, JSON, PDF and an annotated video." |
| 6:30–7:00 | (wrap-up) | Back to Overview | Limitations: Nexar subset, image-plane motion, research prototype. Next steps: full dataset, depth, on-vehicle testing. |

**Fallback.** If the GPU is busy, set `SAFEDRIVE_DEVICE=cpu`: a 40 s clip then takes about 2–3 min. You can also play `reports/demo_annotated.mp4`.

## v3 notes
- The default model is v3. Use test clip **00288**: one WARNING at 15.2 s, 3.7 s before the collision at 18.92 s; the rest of the clip stays NORMAL.
- Show **00901** as an honest miss.
- To demonstrate stabilisation, switch the sidebar to v3-stabilized and explain the camera-shake robustness chart (Evaluation / Reports).

## v3.1 additions
- In the sidebar, show **Warning sensitivity** and explain that each profile is bound to a measured false-alarm budget (held-out: 2.5 / 8.5 / 16.1 FA/h).
- Analyse 00288 with "Save safety events" enabled, open **Near-Miss Memory**, replay the stored clip, read the measured cues, and download the trip report.
- Say clearly that this is a research prototype: about 1 in 9 collisions is warned at about 9 false alarms per hour; it does not brake.
