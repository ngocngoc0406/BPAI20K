# SafeDrive AI v3.4.4: Annotator Training Report

| | |
|---|---|
| Pack | `TRAINING_PACK_01`, in `data/annotator_training/sessions/` (**outside** `data/sessions`). 9 Nexar **train-split** clips, 99 s: 4 events, 3 normal, 2 degraded |
| Marked | "ANNOTATOR TRAINING DATA - NOT REAL DRIVING - NEVER EVALUATION DATA" |
| Status 2026-11-22 | **No annotator has submitted (0 / 2).** Readiness item 14: **NOT VERIFIED** |
| Results | none yet. Nothing in this report is an annotator result |

## 1. Procedure
1. **ANN-A and ANN-B each work alone**, on page 11 → workspace *Annotator training*, with their own pseudonym.
   - The annotator view shows only their own labels.
   - The key (`training_key.json`) is never shown there.
   - Annotators must not open the key file, and they label **before** any discussion.
2. Each annotator labels the whole pack, with evidence text and frames for every event. They use **uncertain** when the image does
   not show enough.
3. **Submit** (button, once). The labels are locked by their SHA-256 in `data/annotator_training/submissions.jsonl`.
   - A second submission is refused.
   - Any label change after submission marks the result **invalid** ("labels changed after submission").
   - Only names on the raw-access list can submit.
4. COORD-01 sees results and the key **only when both have submitted**.
   - Before that the readiness audit shows only *who* has submitted, never a score.
   - Results are saved as `training_report.json` (pair) plus `results/ANN-A.json` and `results/ANN-B.json` (one per annotator).
5. Debrief: COORD-01 goes through every clip with each annotator, including the event-type question (§3).

## 2. Pass criteria (published before training; unchanged from v3.4.3)
| Criterion | Per annotator / pair | Threshold |
|---|---|---|
| submitted, labels unchanged since | each | required |
| event detection (an event label inside each event clip) | each | ≥ 0.80 |
| timing: median \|labelled event time − reference\| | each | ≤ 1.0 s |
| false events on normal clips | each | ≤ 1 |
| all 5 categories used (collision, near_miss, risk_event, normal, uncertain) | each | required |
| evidence complete for every event label | each | required |
| Cohen's κ event vs not-event between ANN-A and ANN-B | pair | ≥ 0.60 |

The audit passes item 14 only if **all** criteria hold. If both submitted and any criterion fails, item 14 is **FAIL**: retrain
and use a new pack.

## 3. What the pack can and cannot score
| Distinction | Scored? | How |
|---|---|---|
| event vs normal | **yes** | event detection on event clips; false events on normal clips; `discrimination_by_clip_kind` |
| uncertain on degraded evidence | **yes** (reported) | share of degraded clips labelled *uncertain* (`uncertain_used_on_degraded`) |
| event timing | **yes** | offsets to the Nexar reference event time |
| collision vs near_miss vs risk_event | **no** | Nexar labels do not separate them, so all three are accepted on event clips. Reported as "not scorable"; covered in the debrief |

## 4. Isolation from evaluation
- The pack is outside `data/sessions`, so it is never in the campaign manifest, the partitions, agreement or metrics, or the
  real-driving hours.
- Its clips come from the Nexar **train** split, never the held-out split.
- Training scores measure annotator readiness only. They are never reported as model or real-world results.

## 5. Current result
| Annotator | Submitted | Result |
|---|---|---|
| ANN-A | no | — |
| ANN-B | no | — |

Item 14 stays **NOT VERIFIED** until two real people have done this. No result was simulated.
