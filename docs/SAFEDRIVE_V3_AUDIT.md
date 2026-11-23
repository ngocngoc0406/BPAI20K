# SafeDrive AI v3 — Independent Audit

Date: 2026-11-21 · Scope: code under `src/`, `scripts/`, `pages/`, `ui/`; result files under `experiments/latest/`; all docs.
Method: re-run tests, check splits and leakage, recompute reported numbers from raw result files, read the core modules, and run the
**deployed** product logic on the held-out videos (`experiments/latest/audit_deployed_alarm_logic.json`).
No code was changed before this audit was written.

## 1. Verified facts
| Claim | Check | Result |
|---|---|---|
| 67 tests pass | `pytest -q` | **67 passed** (44.8 s) |
| 1,500 videos; dev 1,276 / held-out 224 | `splits_full.json` | train 280, val 60, test 60, dev_new 936, test_ext 164 = 1,500; label-balanced in every split |
| No leakage between splits | ID overlap, plus content hash (size + first/last MB) of all 1,500 files | **0 duplicate IDs, 0 duplicate files**; the original 400-video split is unchanged |
| Choices made only on dev | Read `run_ew_study.py`, `deploy_v3.py`, `train_v3_variant.py`, `recalibrate_caution.py` | Calibration, thresholds, CAUTION and the MI threshold all use dev data only. Test windows are built only in the final stage |
| No future frames | `window_indices`, timelines from t = 0, positives end before the event, alarms counted before the event | Confirmed; a unit test also asserts it |
| Report numbers (EARLY_WARNING_V3, RESULTS_TABLES Part C) | Recomputed from `results_ew_full.json`, `ew_heldout_compare.json`, `robustness_*.json`, `pipeline_benchmark_v3.json` | **All quoted numbers match the files** |
| Real-time | `pipeline_benchmark_v3.json` + a new 2/4-thread CPU run | v3: 42.0 fps GPU (RTX 3050 Laptop 4 GB), 19.3 fps CPU (i5-11400H); **18.0 fps limited to 2 CPU threads**, 19.3 fps at 4 threads |

## 2. What SafeDrive AI actually does today
1. **Perceives road users.** YOLOv8n (COCO) detects person, bicycle, car, motorcycle, bus and truck at 10 Hz; ByteTrack assigns IDs.
   Detection and tracking accuracy on dashcam data is **not measured**, because there are no box or ID labels.
2. **Measures image-plane motion and interactions.** Looming / apparent τ, lateral motion toward a *fixed central corridor*, pairwise
   closing, cut-in proxy, lead gap and its rate, optical-flow statistics, and frame difference. There is **no metric distance or speed**
   and no lane detection.
3. **Predicts window-level risk.** LightGBM ×5 (bagged) with Platt calibration. Held-out (224 videos): mAP 0.620, ROC-AUC 0.816, ECE 0.028.
4. **Warns in streaming mode.** Calibrated probability ≥ the threshold set at 10 FA/h on dev, for 3 consecutive steps (0.6 s).
   **Measured on held-out with the deployed code:** 13 of 112 collisions get a valid alarm (≤ 5 s before the event) = **11.6 %
   [6.9, 18.9]**, at **8.7 false alarms per hour [4.2, 16.0]** (10 alarms in 1.15 h of normal driving). Alarms ≥ 2 s ahead: ≤ 5 %.
5. **Explains.** Occlusion attributions per cue and per time step, a frame-change heatmap with boxes, grounded factor text (no LLM),
   and permutation importance.
6. **Quantifies uncertainty.** Ensemble entropy and MI, plus an UNCERTAIN state. MI-based UNCERTAIN suppressed **0** alarm steps on held-out.
7. **Reports.** Per-video CSV / JSON / PDF and an annotated MP4.
   **Not implemented:** live camera input, a persistent near-miss memory, multi-clip trip reports, audio alerts.

## 3. Issues by severity
### High
| ID | Issue | Evidence | Impact |
|---|---|---|---|
| H1 | **Early-warning performance is far from a usable safety product** | Deployed v3: recall 11.6 % at 8.7 FA/h; EDR@2 s ≤ 0.05 | It cannot be presented as a collision-warning system; only as a research prototype or a post-trip analysis tool |
| H2 | **v3's held-out gain over v2-light comes from more data, not from v3's changes** | Matched-FA held-out ΔS: v3 − (v2 config retrained on the same 1,276 videos) = **−0.003 [−0.053, +0.044]**; v2 retrained − v2 deployed = +0.035 | The v3 feature/label changes are supported on dev only; EARLY_WARNING_V3.md overstated "held-out points the same way" |
| H3 | **Fixed central "driving corridor"; no lane / road geometry** | `relational.py`, `motion_features.py`: corridor = \|cx − 0.5\| < 0.2, cy > 0.35 | Lead-vehicle and cut-in cues are wrong on curves, during lane changes, and with off-centre mounting; not validated |

### Medium
| ID | Issue | Evidence | Impact |
|---|---|---|---|
| M1 | Calibration applied inconsistently | Study held-out used `cal(mean of members)`; the product uses `mean(cal(member))` | Reported held-out FA/h at the 10 FA/h target was 11.3; the product gives 8.7 on the same videos |
| M2 | Thresholds fitted on OOF streams of **single** fold models but deployed on the **bagged full-dev** model | `run_ew_study.py` final stage | Realised FA/h deviates from the budget (≈ 0.5–0.9 × on held-out). It is conservative, but not guaranteed |
| M3 | Low statistical power for streaming metrics | 112 positives, 1.15 h of normal driving held-out | FA/h CIs span ~4×; recall CIs ±6 points; small model differences cannot be resolved |
| M4 | v2-light's product alarm differs from its evaluated rule | v2 bundle uses risk score + trend/looming bonus: deployed 18.8 % @ 15.7 FA/h vs probability rule 3.6 % @ 0.9 FA/h | v2 numbers in the docs describe a different operating point than what the app does |
| M5 | Product gaps | No live input, near-miss memory, trip report, or audible alert; uploads capped at 120 s, all frames held in RAM | Cannot demo continuous driving; review workflow is per clip |
| M6 | Perception quality unmeasured | No detection or tracking ground truth; YOLOv8n is not fine-tuned (night: 47 Dark, 88 Twilight videos) | Failure modes of the feature extractor are unknown |
| M7 | No feature-definition versioning | Bundles do not record the relational / motion feature version | Any future fix to a feature (e.g. L1) would silently change deployed v3 predictions |

### Low
| ID | Issue | Evidence |
|---|---|---|
| L1 | `lead_brake_proxy` divides a one-step difference by the 3-step interval, and the interval is irregular if the lead drops out | `relational.py` lines 138–141 |
| L2 | UNCERTAIN (MI) threshold computed on dev videos the model was trained on (in-sample) | `deploy_v3.py` |
| L3 | GPU detection/tracking nondeterminism: up to ±0.05 mAP on re-extracted corrupted clips | `robustness_light.json` vs `robustness_v2_v3caches.json` |
| L4 | Operational: laptop sleep pauses long jobs; the app process dies with the agent session | logs; `PROJECT_AUDIT.md` §9 |

No **critical** issue was found: no leakage, no test-set selection, and no number that disagrees with its source file.

## 4. Answers to the six questions (short form; details in the roadmap and next-version docs)
1. **What it really does.** It detects and tracks road users and measures image-plane approach and interaction cues in real time.
   It ranks risky moments well at window level (ROC-AUC ≈ 0.82), produces calibrated probabilities with grounded explanations, and
   exports per-clip reports. It does **not** yet reliably warn before collisions: about 12 % of collisions at ~9 false alarms per hour.
2. **Most serious weaknesses.** H1 (weak streaming early warning), H3 (no lane / road geometry, no metric distance), H2 (the v3
   improvements are unconfirmed on held-out), and M5 (no live input or near-miss memory).
3. **Difference from dashcam ADAS.** Typical dashcam ADAS functions (forward-collision and lane-departure warnings) rely on camera
   calibration and lane detection to give distance-based alerts. SafeDrive has neither, so it should **not** compete on FCW accuracy.
   What it can offer — and partly does — is *measured* false-alarm budgets, uncertainty, grounded explanations, and (once built)
   near-miss review. This is a difference in transparency and post-event analysis, not in warning accuracy.
4. **Most worthwhile improvements.** See `SAFEDRIVE_PRODUCT_ROADMAP.md`:
   - first, consistency fixes (M1, M2, M7) and honest documentation (H2);
   - then Near-Miss Memory with a trip safety report, and selectable warning sensitivity, which add product value at no inference cost;
   - then lane / road-geometry context to fix H3.
5. **Real-time hardware.** Measured on a laptop with an RTX 3050 4 GB (42 fps) and on its i5-11400H CPU alone (19 fps; 18 fps on 2
   threads). Embedded boards (Jetson, Raspberry Pi) were **not** measured, so no claim is made for them.
6. **From prototype to field-testable demo.**
   - live camera / RTSP input with a rolling buffer;
   - near-miss event memory with clips;
   - an audible or visual alert channel;
   - a trip report;
   - a per-vehicle calibration step (corridor / horizon);
   - a recorded field test with logged false alarms.

   It must be positioned as a driver-assistance **research prototype**, with no automatic braking and no collision-avoidance guarantee.
