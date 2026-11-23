# SafeDrive AI V4: Product Specification

## 1. Product
**One application:** an intelligent driving-safety copilot for a private car driver with a dashcam.
- **Inputs:** a camera, or a recorded dashcam video.
- **What it does:**
  - understands the traffic situation and estimates collision risk;
  - says when its evidence is weak;
  - explains every warning with real evidence;
  - keeps dangerous moments for review;
  - analyses each trip.

**What it is not:**
- not a vehicle controller: it never brakes, steers or intervenes;
- not a certified ADAS, and it cannot prevent accidents;
- not a fleet-management product, and it has no second dashboard.

Run: `streamlit run safedrive_v4/app/Home.py --server.port 8502 --server.address localhost` (localhost only).

The frozen v3 app (port 8501) is the previous version and is kept unchanged for the data-collection campaign. It is not a
second V4 product.

## 2. Screens
| # | Screen | Content |
|---|---|---|
| 1 | Live Driving Assistant | Camera (opened only on Start) or demo video; video with boxes and track ids; risk state (5 states); FPS; latency; input quality; evidence explanation. Events are recorded with 5 s before and 5 s after |
| 2 | Video Analysis | Nexar clip (cached perception) or any video file / upload (full YOLO + ByteTrack perception). Risk timeline, states and events |
| 3 | Traffic Scene Graph | Frame with boxes; dynamic graph at any time (edge width = confidence; hover: since when, belief, uncertainty, evidence); road users interacting with the ego car; "is it getting worse?" (3 s trend of KG risk evidence and uncertainty) |
| 4 | Risk Explanation | Per warning: level and reasons, trace integrity, evidence frame, road users, KG reasoning path with opinions, SHAP by group (then features), uncertainty, what the system could not determine, counterfactuals on request, full sealed trace |
| 5 | Near-Miss Memory | Filter by category and status; clip with before / event / after; risk timeline; track history; review (confirm / reject / relabel; a collision needs written evidence); review log; JSON export |
| 6 | Trip Report | Counts (events, to review, confirmed near-miss, rejected false alarms); time share per state; frequent risk factors; event timeline; limitations; PDF and JSON export. **No safety score** |
| 7 | Model & System Diagnostics | Provenance and frozen-hash check; held-out benchmark table; hypothesis verdicts; latency per trip; CPU / RAM / VRAM; encrypted-store status; state shares |

## 3. Warning states
| State | When | Shown as |
|---|---|---|
| LOW | risk below the MODERATE threshold | 🟢 |
| MODERATE | p ≥ threshold at 20 FA/h (dev) for 3 steps | 🟠 |
| HIGH | p ≥ threshold at 10 FA/h (dev) for 3 steps | 🔴. If perception is unreliable: 🔴 **low confidence (cautious)**. A high risk is never hidden |
| INSUFFICIENT_EVIDENCE | not HIGH, and perception reliability < 0.45, or OOD, or high model disagreement | ⚪ |
| DEGRADED_INPUT | camera stall / black / saturated / frozen / blurred | ⚫. Risk is not interpreted, and the onset counter is reset |

## 4. Events (Near-Miss Intelligence)
**Automatic proposals:**
- `potential_near_miss`: HIGH, not cautious, with KG risk evidence;
- `risk_event`: MODERATE, or HIGH without KG evidence;
- `uncertain`: a cautious HIGH.

**Human categories:** collision (written evidence required), near_miss, risk_event, normal (false alarm), uncertain. Every
review is logged.

**Storage:** events and clips live **inside the encrypted session volume** (VeraCrypt V:, next to `data/sessions`). When it is
not mounted, clips are not written and uploads are refused.

## 5. Data honesty
- **Data kind.** Each trip records its data kind. Only `real_driving` (a consented campaign session) counts as real-world
  driving; Nexar and demo videos never do.
- **Measured quality.** The model's measured limits are shown in the app: about 12 % recall at about 9 FA/h, and fewer than 5 %
  of events warned 2 s ahead.
- **Unknowns.** Every warning lists what the system could not determine.
