# SafeDrive AI v3.4: Annotation Guide

## 1. Independence (blind annotation)
- **Two annotators** label every eligible session **independently**.
- **The annotation page enforces blindness.** An annotator sees only the video:
  - no model output;
  - no event clips (their timing reveals where the model warned);
  - no other annotator's labels;
  - no agreement figures.
- **Agreement results appear only for a coordinator**, and only after both annotators marked the session as fully reviewed.
- **Do not discuss a session** with the other annotator before both have finished.
- **Raw access.**
  - Annotators on the raw-access list see the raw video.
  - Others see only a privacy-approved shared copy, where blurring may hide brake lights, which lowers label quality.

## 2. Categories
| Category | When to use it | Event time (`critical_t`) |
|---|---|---|
| **collision** | visible contact between the ego vehicle and another road user or object | moment of contact |
| **near_miss** | no contact, but evasive action (hard braking, swerve) by anyone, or a gap so small that contact was avoided only by chance | closest approach, or start of the evasive action |
| **risk_event** | a developing conflict where a warning would have helped the driver: cut-in, lead vehicle braking hard, pedestrian stepping out, … | latest moment a warning would still have been useful (e.g. start of the needed braking) |
| **normal** | a reviewed stretch with nothing notable. Optional; use it to mark checked stretches, e.g. around a model alarm you noticed later | — |
| **uncertain** | something may have happened, but the evidence is insufficient (occluded, out of view, too dark, ambiguous) | optional |

Rules:
- A near-miss is **never** labelled because a warning sounded or because the model said so. Annotators work blind.
- If in doubt between two categories, choose the lower one and write why. If evidence is missing, choose **uncertain**.
- One annotation = one independent situation. Do not split a situation into frames or repeat it.

## 3. Fields
| Field | Meaning |
|---|---|
| Start | when the risk first became visible (session time, s) |
| Event time | see the table above. **Required** for collision / near_miss / risk_event |
| End | when the situation was over |
| Confidence | low / medium / high: how sure you are of the category and the times |
| Visual evidence | **required** for events: what in the image shows it, e.g. "lead car brake lights at 123.4 s, gap halves within 1 s" |
| Evidence frame times | **required** for events: ≥ 1 session time within (or ≤ 5 s around) the situation. The page saves those frames to `evidence/` |
| Tags | motorcycle / vehicle cut-in, lead braking, ego / other lane change, curve, intersection, pedestrian, cyclist, dense traffic, low light, backlight, glare, rain, camera shake, occlusion, other |
| Note | anything else |

- **Session time** = segment start (shown on the page) + video position.
- **Full review:** after watching the **whole** session, press *"I watched the WHOLE session"*. Without it the session gives no
  false-alarm rate and no recall.

## 4. Agreement and disagreements (coordinator)
- **Matching.** Situations of the two annotators are matched one-to-one by temporal IoU ≥ 0.3.
- **Agreement measures** (`src/realworld/annotation.py`):
  - **Cohen's kappa** on matched categories;
  - **kappa with unmatched**: a situation found by only one annotator is paired with "none";
  - **kappa event vs not**;
  - **time offsets**: mean and median absolute difference of start, event time and end;
  - a confusion table.
- **Consensus:**
  - **confirmed**: same category from both annotators, or agreed after discussion;
  - **disputed**: matched with different categories;
  - **single**: found by one annotator only.
- **Disagreements** are recorded with **each annotator's own reason**, and a status of *open*, *unresolved* or
  *agreed_after_discussion* (with the agreed category).
  - **Agreement is never forced.** If the evidence does not settle it, the status stays *unresolved*.
  - Unresolved, disputed and single situations are excluded from recall and are exclusion zones for false alarms.
- **Labels are never changed** to match the model.

## 5. v3.3 labels
v3.3 categories are mapped on load and the original name is kept in `legacy_label`:
- `confirmed_collision` → collision;
- `potential_near_miss` → near_miss;
- `warning_worthy` / `potential_risk` → risk_event.

No real annotations existed when the schema changed.

## 6. Quality checks before campaign metrics
- Both annotators fully reviewed the session.
- Every event has an event time, evidence text and evidence frames.
- Disagreement reasons have been recorded for every disputed or single situation that matters for the metrics.
- `agreement_report.json` has been produced. Report kappa, its sample size and the time offsets. A kappa from very few
  situations is not meaningful.
