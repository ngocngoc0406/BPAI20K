# SafeDrive AI — Product Assessment and Roadmap

Based on [SAFEDRIVE_V3_AUDIT.md](SAFEDRIVE_V3_AUDIT.md). All numbers come from files in `experiments/latest/`.

## 1. Product capability review (driver-assistance perspective)
| Capability | Status today | Evidence | Gap |
|---|---|---|---|
| Detect vehicles and road users | YES: 6 COCO classes, 10 Hz | YOLOv8n + ByteTrack, 9.3 ms/frame | Not fine-tuned or measured on dashcam data; weaker at night |
| Distance and approach speed | PARTIAL: image-plane only (looming, apparent τ, lead gap rate, closing) | `motion_features.py`, `relational.py` | No metric distance or speed; no calibration or depth |
| Recognise dangerous situations | PARTIAL: window ROC-AUC 0.816 on held-out | `results_ew_full.json` | Fixed corridor; no lane context |
| Early warning with few false alarms | WEAK: 11.6 % recall at 8.7 FA/h (deployed, held-out) | `audit_deployed_alarm_logic.json` | Main open problem |
| Near-Miss Memory | NO | — | Needed for review and coaching |
| Explain the warning | YES: grounded cues, occlusion attributions, change heatmap | XAI page | Not validated with users |
| Safety report after each trip | PARTIAL: per-clip PDF | `reports/` | No multi-clip trip aggregation or safety score |
| Live operation | NO: file upload only (≤ 120 s) | `pages/2_Video_Analysis.py` | Needs camera / RTSP input |

## 2. Candidate differentiators
YOLO + ByteTrack + LightGBM is **not** novel and is not claimed as such. Each candidate below was assessed on what can be **proven**.

| Direction | What it would mean | Novelty (honest) | Feasibility / cost | Real-time impact | Verifiable with our data? | Verdict |
|---|---|---|---|---|---|---|
| **Explainable Safety Events** | Every alert or near-miss stored with measured cues, attribution, uncertainty and clip | Low as research; uncommon as a transparent product | Low: components exist | None (post-alarm) | Yes: functional tests; explanations reproducible | **Do now** (P1) |
| **Near-Miss Intelligence** | Persistent memory of risky episodes (alarm or high-risk peaks), trip report, frequency per hour, top factors | Low–medium (fleet products exist; transparent local version is uncommon) | Low–medium | None | Partly: episode detection on held-out normal / collision videos (precision of "risky episode" vs label) | **Do now** (P1) |
| **Adaptive Warning** (driver-selectable sensitivity) | Profiles bound to measured FA budgets (5 / 10 / 20 FA/h) | Low | Very low: thresholds already exist | None | Yes: dev-calibrated, held-out FA/h and recall per profile | **Do now** (P1) |
| **Context-aware Risk** (lane / road geometry instead of a fixed corridor) | Ego lane from lane detection or a vanishing-point estimate; corridor follows the road | Low (standard in ADAS) but fixes H3 | Medium: needs a light lane model and validation | +2–5 ms/frame (to be measured) | Partly: streaming S on dev; no lane labels | **Next** (P2), experiment first |
| **Vehicle Interaction Understanding** | Already in v3 (cut-in, closing pairs, lead gap) | Low | Done | ~0.3 ms/frame | Dev gain significant (+0.025 S); held-out unconfirmed | Keep; improve with lane context |
| Per-object risk head / scale-aware depth | Risk per track instead of per frame | Medium | High | Unknown | Needs per-object labels (not in Nexar) | Later (P3) |
| Larger video backbone, distilled | Better specificity | Low | High (training cost) | Must stay real-time | Yes | Later (P3) |

## 3. Prioritised roadmap
**P0: consistency and honesty.** No model change, no latency.
1. Document H2: v3's held-out gain is attributable to more data.
2. Record the feature-definition version in bundles and refuse mismatches (M7). This is a prerequisite for fixing L1 safely.
3. Make the study's held-out evaluation use the product's calibration aggregation (M1), and report the **deployed-logic** held-out numbers as the product numbers.

**P1: product value at zero inference cost** (implemented in this iteration if tests pass; see SAFEDRIVE_NEXT_VERSION.md).
4. **Warning sensitivity profiles** (Conservative @5 / Balanced @10 / Sensitive @20 FA/h on dev) with held-out evidence per profile.
5. **Near-Miss Memory**: a persistent local event store (JSON lines) of alarm episodes and high-risk peaks, with time, level,
   probability, uncertainty, top cues, explanation, model version, and a short clip.
6. **Trip Safety Report**: aggregate several analysed clips into one trip, with events per hour, event timeline, top contributing cues
   and a PDF/JSON export. This is descriptive, with no "safety score" claim unless validated.

**P2: fix the main modelling weakness (experiment first; keep only if dev evidence is clear).**
7. Road-geometry context: dynamic corridor from a vanishing point or lane estimate; re-run the v3 study protocol (dev CV, paired bootstrap).
8. OOF thresholds from bagged fold models, matching the deployed model (M2).
9. Fix `lead_brake_proxy` (L1) under a new feature version, then re-train and re-evaluate.

**P3: research.**
10. Per-object risk; scale-aware depth; distilled video features; detector fine-tuning with dashcam boxes.

## 4. Demo and field-test readiness checklist
| Need | Status | Plan |
|---|---|---|
| Live camera / RTSP input with a rolling buffer | missing | P1/P2: OpenCV capture loop on the existing extractor (already 1.8–4× real time) |
| Alert channel (sound + visual banner) | visual only | P2 |
| Near-miss memory + trip report | missing | P1 (this iteration) |
| Per-vehicle setup (horizon / corridor) | missing | P2, with context-aware risk |
| Field protocol | missing | Record ≥ 10 h of normal driving to measure the real FA/h with a fixed model version; log every alert with its clip; no safety claims |
| Positioning | — | "Research prototype for driver assistance and near-miss review. It does not brake or steer and does not guarantee collision avoidance." |

## 5. Hardware
| Hardware | Measured | v3 throughput |
|---|---|---|
| RTX 3050 Laptop 4 GB + i5-11400H | yes | 42.0 fps (4.2× real time at 10 Hz) |
| i5-11400H CPU only, 6 cores | yes | 19.3 fps (1.9×) |
| Same CPU limited to 2 threads | yes | 18.0 fps (1.8×) |
| Jetson / Raspberry Pi / phone | **no** | not claimed; this needs a measurement on the device (the YOLOv8n export to ONNX/TensorRT is the main work) |
