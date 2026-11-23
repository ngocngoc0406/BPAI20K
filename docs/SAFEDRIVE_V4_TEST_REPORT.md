# SafeDrive AI V4: Test Report (2026-11-23, product round)

## 1. Test suite
| Suite | Count | Result |
|---|---|---|
| Existing tests (v1–v3.4.4), unchanged | 164 | **164 passed** |
| V4 tests | 40 | **40 passed** |
| **Total** (`python -m pytest -q`) | **204** | **204 passed**, 0 failed |

| V4 test file | Count |
|---|---|
| `test_v4_graph_reasoning.py` | 13 |
| `test_v4_xai_uncertainty.py` | 7 |
| `test_v4_pipeline.py` | 4 |
| `test_v4_uncertainty_events.py` | 7 |
| `test_v4_app.py` | 9 (8 pages + report / clip) |

The earlier `test_v4_dashboard.py` was removed together with the research dashboard; the single app replaces it.

| Required case | Test(s) |
|---|---|
| detection / tracking (v3, reused) | v3 suites `test_live.py`, `test_pipeline*.py` (unchanged, passing) |
| graph consistency, temporal updates, occlusion, missing evidence, rule conflicts | `test_v4_graph_reasoning.py` |
| uncertainty propagation | `test_subjective_logic_operators_are_consistent`, `test_edges_and_nodes_carry_opinions_and_low_confidence_raises_uncertainty`, `test_temporal_risk_opinion_accumulates_and_ages` |
| SHAP, counterfactual validity, trace integrity, data leakage | `test_v4_xai_uncertainty.py` |
| adaptive warning (5 states, cautious HIGH, degraded input, evidence gate) | `test_warning_policy_five_states`, `test_warning_policy_separates_uncertainty_from_risk` |
| input quality | `test_input_quality_monitor_detects_degraded_input` |
| live inference, memory | `test_streaming_pipeline_matches_offline_features_latency_and_traces`, `test_a3_v3_readonly_predictions_equal_v3_and_files_untouched` |
| event recording, review rules, encryption requirement | `test_event_store_categories_review_rules_and_export`, `test_clips_are_never_written_outside_the_encrypted_store`, `test_trip_report_exports_and_event_clip` |
| PDF / JSON export | `test_trip_report_exports_and_event_clip`, page tests |
| single app | `test_page_renders[*]` (all 8 pages, temporary store) |
| v3 isolation | `test_v4_does_not_touch_v3_artefacts` |

## 2. Regression (v3 unchanged)
- **A0 reproduces v3** in both experiment rounds:
  - OOF mAP 0.6170491 and recall@5/10/20 of 0.0987 / 0.1661 / 0.2335;
  - held-out mAP 0.6198, ROC-AUC 0.816, ECE 0.028 and recall@10 0.116 at 8.70 FA/h.
- **Hashes unchanged:**
  - v3 model, feature and threshold hashes, both eligible;
  - `configs/frozen_v3_hashes.json` 1d15d1cc… (mtime 2026-11-21);
  - software hash 2032bf7b….
- **A3 path.** v3 predictions through the V4 pipeline are identical (< 1e-9), and the v3 files are byte-identical before and
  after.
- **Graph regression.** After the opinions were added, the SG / KG channels are identical to the round-1 precompute (8 videos
  checked).

## 3. Manual end-to-end checks (performed)
| Check | Result |
|---|---|
| Video analysis of a cached Nexar clip (00063) | 1 potential near-miss at 20.0 s (annotated event at 19.55 s: the warning was **late**); clip saved (5 s before, 5 s after); report PDF / JSON |
| Video analysis of the **raw** video of the same clip (full v3 perception) | 46 s for 40 s of video; same event at 20.0 s |
| Live mode on the demo video (10 Hz pacing, v3 capture thread) | 374 steps; states LOW 363, INSUFFICIENT 6, MODERATE 4, DEGRADED 1; 2 risk events with saved clips; clean stop. The first attempt was faulty (looping source; processing faster than 10 Hz, 552 steps) and was fixed |
| App on localhost:8502 | HTTP 200; all 8 pages render (AppTest) |
| Camera | **not opened** in any test (requires the user's Start press) |

## 4. System measurements
| Item | Value |
|---|---|
| V4 layers per step (replay) | 8.9 ms mean, 16.6 ms p95 |
| V4 layers per step (raw-video analysis) | 24 ms mean, including v3 window features |
| Graph / reasoning | 0.6–2.7 ms / 0.03–0.9 ms |
| SHAP per warning | 6–9 ms |
| Raw-video analysis, end to end | 0.87× real time (GPU YOLO; first run includes model load) |
| Memory | bounded graph state; process RSS ≈ 0.5 GB; Python peak 49 MB per clip |
| Model size | v3 3.8 MB (frozen); A4 research 3.9 MB; GNN 75k parameters |

**Not measured:** sustained live operation on a camera in a vehicle, and the CPU-only perception speed in the V4 app.
