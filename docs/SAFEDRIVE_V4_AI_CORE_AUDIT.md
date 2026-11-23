# SafeDrive V4: AI-Core Audit (Phase A)

| | |
|---|---|
| Date | 2026-11-23 |
| Branch | `v4-ai-core` (git initialised for this work; baseline commit `a898dca` on `master`; data, models and private artefacts are excluded from git) |
| Measurements | `python -m safedrive_v4.evaluation.core_audit` → `experiments/v4/core_audit.json` (4.9 min) |
| Changes to models / thresholds / labels | **none** (audit only) |

## 0. Ground truth
Nexar provides clip-level labels (collision / near-collision, event and alert times) and conditions (light, weather, scene). It
has **no boxes, no track ids and no relation labels**.

- **Cannot be computed:** detection precision / recall / mAP, IDF1 / HOTA and relation accuracy. They are **not computed**, and
  no automatic labels were created.
- **Proxies instead.** Where a proxy is used, it is named as one.
- **Data use.**
  - Diagnostics use **dev** videos only.
  - Calibration, uncertainty and XAI-consistency figures on **held-out** are descriptive: frozen v3 ensemble, no parameter chosen.

## 1. Actual pipeline (as implemented)
```
video ─► [v3] YOLOv8n (conf 0.25, road-user classes) ─► ByteTrack ─► per 10 Hz step:
          ├─ object-motion features (28 ch; ALL detections incl. low conf / untracked)      ┐
          ├─ optical flow (4 ch), frame difference (20 ch), brightness (pixels)            ├─► [v3] window stats (W = 1 s, 8 stats)
          ├─ interaction cues (18 ch, from tracks, uncompensated)                          ┘      ─► LightGBM ×5 ─► Platt ─► p
          └─ ego motion (homography; computed for V4 only)
[V4] tracks + homography ─► scene graph (k = 3 finite differences, 2-of-3 persistence) ─► rules R01–R10
     ─► (a) plain findings (confidence products)  (b) subjective-logic opinions ─► explanation chains, event category
[V4] decision: p (frozen v3) + SL reliability + input quality ─► LOW / MODERATE / HIGH(+cautious) / INSUFFICIENT / DEGRADED
[V4] SHAP (TreeSHAP of the v3 members) at warning onsets ─► sealed trace
```

**Coupling.**
- The **risk decision is entirely the frozen v3 model**.
- The scene graph and KG affect only explanations, event categories and the INSUFFICIENT / cautious states.
- **Reliability gates.** The INSUFFICIENT and cautious states depend on the SL reliability (§A4).
- **OOD gate.** Documented as active, but not wired (§A7).

## 2. Findings by module (measured)

### A1. Perception: proxies, dev 1,276 videos (raw caches, extracted before the v3.4.2 class stabiliser)
| Proxy (median per video) | All | Normal light (1,153) | Twilight (72) | **Dark (41)** | Rain |
|---|---|---|---|---|---|
| detections per frame | 2.74 | 2.80 | 2.53 | **0.61** | — |
| mean detector confidence | 0.58 | 0.58 | 0.57 | **0.44** | 0.56 |
| share of detections with conf < 0.4 | 0.20 | | | | |
| median track length (steps of 0.1 s) | 8 | 8 | 7 | **4.75** | |
| short tracks (< 3 steps) | 0.23 | 0.23 | 0.24 | **0.33** | 0.24 |
| id-switch proxy (a same-class track restarts within 0.3 s at IoU > 0.3), per min | 3.0 | 3.0 | 3.0 | 0.0 | 3.0 |
| mid-frame track loss (large box, not at the border), per min | 9.1 | 9.2 | 10.4 | 3.4 | **14.8** |
| class changes per track | 0.27 | | | | |

**What the proxies show.**
- **Perception degrades sharply at night.** There are 4.6× fewer detections and shorter tracks.
- **Rain doubles mid-frame track loss.**
- **Confidence.** One detection in five has conf < 0.4, and v3 object-motion features use all of them (threshold 0.25).
- **Camera conditions.** Motion blur, shake and frame drops were measured in v3 (`experiments/latest/robustness_v3*.json`) and
  are not repeated here.

### A2. Scene graph: 300 dev videos, 186 min
| Relation | per min | median lifetime | 10th percentile | flicker share (re-appears ≤ 0.3 s after vanishing) |
|---|---|---|---|---|
| APPROACHING | 215 | **0.2 s** | 0.1 s | 0.18 |
| MOVING_AWAY | 217 | 0.2 s | 0.1 s | 0.28 |
| CUTTING_IN_FRONT_OF | 33 | 0.3 s | 0.1 s | 0.16 |
| CROSSING_PATH | 2.5 | 0.3 s | 0.1 s | 0.25 |
| FOLLOWING | 262 | 0.3 s | 0.1 s | **0.35** |
| NEAR | 821 | 0.2 s | 0.1 s | **0.35** |
| POTENTIAL_CONFLICT | 64 | 0.2 s | 0.1 s | 0.18 |
| INTERACTING_WITH | 88 | 0.2 s | 0.1 s | 0.19 |
| OCCLUDING | 48 | 0.2 s | 0.1 s | 0.19 |
| SAME_DIRECTION | 19 | 0.1 s | 0.1 s | 0.22 |

- **Structural invariants: 0 violations** (programmatic checks on every step):
  - no edge referencing a missing node, and no timestamp mismatch;
  - edge confidence never exceeds node reliability;
  - at most one lead, and no APPROACHING with MOVING_AWAY on the same pair.
- **Temporal stability: poor.**
  - **Lifetime.** Relations live 0.2–0.3 s at the median.
  - **Flicker.** 16–35 % flicker.
  - **Noise symmetry.** APPROACHING and MOVING_AWAY are equally frequent.
  - **Cause.** The kinematic relations are dominated by estimation noise: scale growth and velocity are 3-step finite
    differences of jittery boxes, and persistence is only 2-of-3.
- **Relation classes** (documented in the scene-graph doc):
  - geometric: NEAR, OCCLUDING, LOCATED_IN, FOLLOWING (lead = lowest box in the corridor);
  - kinematic: APPROACHING, MOVING_AWAY, CUTTING_IN, CROSSING, SAME_DIRECTION;
  - inferred: POTENTIAL_CONFLICT, INTERACTING_WITH.

### A3. Risk assessment: dev OOF streams of v3 (= A0, reproduced exactly), threshold at 10 FA/h
| | Value |
|---|---|
| positives warned 0–5 s before the event | 106 / 638 (16.6 %) |
| warned **only after** the event (late) | 49 (7.7 %) |
| **missed** | 483 (75.7 %) |
| lead time of hits: median [IQR] | **0.78 s** [0.39, 1.52] |
| median calibrated p at −3 / −2 / −1 / −0.5 s | 0.52 / 0.60 / 0.71 / 0.77 (risk becomes visible only in the last ~1 s) |
| best single cue, univariate AUROC (dev OOF windows) | lead_gap_rate 0.68, looming_all 0.67, lead_bottom_y 0.64, res_tau_center 0.63, **tau_center 0.62**, ego_ttr 0.60, cut-in score 0.56, lateral-in 0.51 |
| normal driving: level transitions / min, episodes / h (20 FA/h), re-onsets within 2 s | 0.85 / 19.3 / 4 % |

- **Immediate risk is visible; developing risk is not.** Discrimination comes from cues of the final second: looming, the lead
  gap closing. Cut-in and lateral cues barely discriminate.
- **TTC is image-based.** The image TTC (tau) is a weak cue (AUROC 0.62). No physical distance is assumed anywhere, and none is
  available.
- **Calibration** (dev OOF): ECE 0.287 raw, 0.016 after Platt. Held-out (frozen model): ECE 0.028, Brier 0.176.

### A4. Uncertainty (held-out windows, frozen v3 ensemble, descriptive)
| Signal | Error-detection AUROC | Error rate at 100 / 70 / 50 % coverage |
|---|---|---|
| entropy (total) | **0.70** | 0.262 / 0.185 / 0.146 |
| aleatoric (entropy − MI) | 0.69 | — |
| mutual information (epistemic) | 0.65 | 0.262 / 0.213 / 0.179 |
| **1 − SL perception reliability (V4)** | **0.52** | 0.262 / 0.257 / 0.247 (≈ no selective benefit) |

**Response of SL reliability to injected perception errors** (dev, 300 videos, same steps):
- mean 0.570 clean vs 0.561 noisy;
- AUROC between noisy and clean steps 0.52.

**The V4 perception-reliability signal does not reflect perception errors and does not predict model errors.** It currently
drives INSUFFICIENT_EVIDENCE and the cautious HIGH flag through an **unvalidated** threshold (0.45). The other uncertainty
parameters were also chosen without validation:
- u > 0.5 / 0.8 for "insufficient";
- temporal decay 0.8, evidence decay 0.9, base rate 0.1;
- the rules' base confidences.

The ensemble entropy, by contrast, is informative (selective error 26 % → 15 % at 50 % coverage).

### A5. Knowledge graph (rules R01–R10; dev rule statistics as reported in the KG doc)
- **Duplicate rule.** R03 fires **only** when R02 fires (subset 100 %, Jaccard 0.58): R03 duplicates R02 with a stricter
  threshold. The plain engine's noisy-OR double-counts it; the SL engine keeps the maximum per road user.
- **Overlapping rule.** R06 overlaps R02 / R03 (Jaccard 0.20–0.21), because POTENTIAL_CONFLICT includes "tau < 2 s in the
  corridor".
- **R09 (insufficient evidence)** fires at 34–36 % of steps in both normal and pre-event driving. It is uninformative.
- **R04 (pedestrian crossing)** almost never fires on Nexar, so it is untested.
- **R10 (interaction ahead)** is weak (lift 1.2).
- **Provenance.**
  - The ontology triples are marked "hypothesis".
  - The rules' base confidences (0.4–0.9) were set by hand, with no data provenance.
  - No rule creates an event without an edge: 0 risk findings without an edge.
- **Missing information.** Tracks younger than 3 steps produce no relation (tested). Darkness only triggers R08 (perception
  degradation), never risk.

### A6. XAI
| Check | Result |
|---|---|
| SHAP faithful to the v3 computation | yes: local accuracy 1e-14; deletion test 100 % |
| SHAP temporal consistency (consecutive steps in the 3 s before events, 40 held-out positives) | top group unchanged in 86 % of steps; group-rank Spearman 0.82 |
| Rule chains faithful to the model | **no**: the v3 model does not use them. They are observed evidence, but the UI shows them under "Why" / "Reasoning path" without saying that the model did not use them |
| Explanation coverage | SHAP 100 %; KG chain 47 % of warnings |
| Live mode | **no SHAP** (the trace has graph evidence only) |
| Group labels | the "v3_motion" group mixes object-motion (track) and optical-flow / brightness (pixel) channels, so the label is misleading |
| Links to frame / time / object / track | yes (trace integrity 100 %) |

### A7. Decision engine (code review + dev streams)
- **INSUFFICIENT override.** INSUFFICIENT_EVIDENCE replaces MODERATE warnings whenever the SL reliability is < 0.45. The signal
  is uninformative (A4), so this hides MODERATE warnings for no validated reason.
- **OOD gate not wired.** It is documented (UNCERTAINTY.md: "OOD ≥ 0.99 → INSUFFICIENT") but **not wired**: the pipeline always
  passes `ood=None`. Documentation and code disagree.
- **Duplicate traces.** There is no hysteresis or minimum dwell. A trace is created at every level change, so MODERATE ↔ HIGH
  alternation creates duplicate traces.
- **Duplicate events.** Event episodes split on a single-step gap.
- **What already holds.**
  - High uncertainty never raises the level (tested).
  - DEGRADED_INPUT overrides all other states.
  - On normal driving, re-onsets within 2 s are rare (4 %).
- **Level names.** V4 "HIGH" is v3's WARNING level (10 FA/h). V3's own HIGH (5 FA/h) is unused.
- **Replay degraded-input detection** is approximate: black / saturated / frozen are detected from cache channels, but blur and
  stall are not.

## 3. Issues by severity
| ID | Severity | Module | Issue | Evidence |
|---|---|---|---|---|
| C1 | **Critical** | Uncertainty / Decision | The SL perception reliability is uninformative but gates warnings (INSUFFICIENT replaces MODERATE; cautious HIGH) with an unvalidated threshold | AUROC 0.52 error detection; 0.52 vs injected errors |
| C2 | **Critical** | Scene graph | Kinematic relations are dominated by noise (median lifetime 0.2 s, flicker up to 35 %, APPROACHING ≈ MOVING_AWAY); every chain, rule and event category inherits this | A2 table |
| H1 | High | Decision / docs | OOD gate documented but not wired | `pipeline.step` passes `ood=None` |
| H2 | High | XAI | Rule evidence is shown as "Why" without separating it from the model explanation; no SHAP live | A6 |
| H3 | High | Uncertainty / KG | All uncertainty and rule parameters are unvalidated | A4, A5 |
| H4 | High | Risk | 76 % of events missed and 8 % warned only after the event at 10 FA/h; lead time 0.78 s; developing risk is not captured (capability limit of v3 features) | A3 |
| H5 | High | Perception | Night: 4.6× fewer detections, short tracks; no "dark" input state influences the decision | A1 |
| M1 | Medium | KG | Duplicate / overlapping rules (R03 ⊂ R02, R06 ∩ R02); R09 uninformative; R04 untested | A5 |
| M2 | Medium | Decision | No hysteresis / dwell; duplicate traces at level changes; events split on single-step gaps | A7 |
| M3 | Medium | Perception | id-switch proxy 3 / min, mid-frame loss 9 / min (15 / min in rain), 23 % short tracks; offline caches pre-date the class stabiliser (live has it): an offline / live mismatch | A1 |
| M4 | Medium | Perception / features | v3 motion features use detections down to conf 0.25 (20 % < 0.4); the scene graph uses 0.30; the node-reliability formula is ad hoc | A1 |
| M5 | Medium | XAI | The "v3_motion" SHAP group mixes object-motion and pixel channels | A6 |
| M6 | Medium | System | V4 live vs offline differences (wall-clock sampling) are not quantified | test report |
| L1 | Low | XAI / UI | Two confidence notions are shown (edge confidence and SL belief) | — |
| L2 | Low | Decision | V4 level names differ from v3's | — |
| L3 | Low | Uncertainty | Replay degraded-input detection is approximate | — |

## 4. Proposed order of work and acceptance criteria (Phase B; nothing implemented yet)
Each step gets a baseline, an ablation, a latency measurement and a regression check. All choices are made on dev OOF; held-out
is evaluated once at the end. The v3 model and thresholds stay frozen.

| # | Fixes | Method (hypothesis) | Acceptance criterion |
|---|---|---|---|
| 1 | C2 | Robust kinematics: 1 s least-squares fit of scale and position instead of 3-step differences; enter / exit hysteresis for kinematic relations | On dev: kinematic flicker < 10 %; median lifetime ≥ 0.8 s; APPROACHING / MOVING_AWAY asymmetry in the 2 s before events; rule lift not lower; latency < +1 ms / step |
| 2 | C1, H3 | Validate or replace perception reliability: build candidate signals (track age, confidence trend, box jitter, darkness, detection count) and test them on dev against (a) injected errors and (b) model errors. Use a signal only if its dev error-detection AUROC beats entropy's, with the threshold chosen on dev risk-coverage. Otherwise remove the reliability gate and show the ensemble entropy instead | Dev AUROC reported; gate on or off decided by the rule; held-out once |
| 3 | H1 | Wire the OOD percentile, with the threshold validated on dev (risk-coverage), or remove it from docs and code | Code = docs; dev validation reported |
| 4 | H2, M5 | Separate "model explanation (SHAP of the v3 model)" from "rule-based evidence (not used by the model)" in traces and UI; split the SHAP groups (object motion / camera & pixel / interaction); compute SHAP asynchronously live | Trace schema has both, each labelled; live traces carry SHAP; latency of the warning path unchanged |
| 5 | M2 | Decision hysteresis (minimum dwell; merge episodes with gaps ≤ 1 s; one trace per episode) | Dev streams: recall and FA/h unchanged (same onsets); duplicate traces and events = 0 |
| 6 | M1 | Rule clean-up: R03 becomes a severity of R02; remove the tau duplicate from R06; redefine or drop R09; rerun the rule statistics | No duplicate rule (Jaccard of the risk rules < 0.3); rule statistics re-reported |
| 7 | H5, M3, M4 | Perception: darkness / low-detection input state validated against Nexar light metadata (dev); quantify the class-stabiliser offline / live mismatch; confidence-weighted features only as a tested v4 variant (v3 frozen) | Dev evidence for each; no v3 change |
| 8 | Phase C | AI-core scenario suite (synthetic tracks with expected behaviour). Real-video scenarios wait for human-confirmed ground truth | Each scenario: expected vs actual recorded |

**Not planned:**
- a new predictor, GNN, Transformer or LLM: H1 showed no gain, and no new hypothesis justifies one yet;
- any change to v3.
