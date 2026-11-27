# SafeDrive V4: Evidence-Grounded Intelligent Annotation: Design (Phase A)

**Purpose.** Produce human ground truth for **driving-risk anticipation**: the temporal extent of risk, the actors involved and
their interactions, annotator uncertainty and visual evidence. It is produced by **two independent annotators**, with separate
adjudication. It is integrated as page **8 · Annotation Intelligence** of the single SafeDrive V4 app.

**What the system never does.** It never uses model output as ground truth and never labels automatically.

## 1. Audit: what exists and what is reused
| Existing (v3.4, `src/realworld/`) | Status | Reuse in V4 |
|---|---|---|
| `annotation.py`: event schema (onset, event time, end, 5 categories, confidence, evidence text + frame times) | works; tested | concepts and categories kept. `cohen_kappa` and the tIoU **matching** algorithm are imported read-only. `delete()` rewrites the file, which loses history, so V4 uses its own append-only store |
| `annotation.agreement` / `consensus` / `Disagreement` | works | V4 agreement extends it (boundaries, actors, relations, ordinal attributes); disagreement records keep the same idea (no forced agreement) |
| `training.py`: submission lock (labels hashed at submit; key hidden until all submitted) | works | the same lock decides when an annotator's work becomes visible to the other annotator / adjudicator |
| `privacy.py`: raw-access list, coordinators, sharing policy | works | raw video only for raw-access annotators; adjudication and export by coordinators; export gated by the sharing policy |
| `pages/11_Ground_Truth_Annotation.py` (v3 app) | works for campaign sessions | **not modified** (it belongs to the frozen v3 app). The V4 page is new |
| V4 `event_memory`, `app/core.TripStore`, encrypted store (`default_root`) | works | the annotation store lives next to them, inside the VeraCrypt volume |
| V4 scene graph / KG | works (with the known relation-instability issue C2) | the annotation → KG triple mapping and the scene-graph validation (E6) |

Not modified: `src/`, `pages/`, `ui/`, v3 models, thresholds, calibration and `configs/frozen_v3_hashes.json`.

## 2. Data model (schema `ann-v4.0`; versioned)
**Records.** Every record carries `schema`, `record_id`, `video_id`, `annotator` (pseudonym), `version`, `created`, and
`provenance` (tool version, video SHA-256, frame rate).

| Record | Fields |
|---|---|
| **Actor** | `label` (A, B, …), `cls` (car / truck / bus / motorcycle / bicycle / pedestrian / other / ego), `track_id` (v3 tracker id, if one is visible), `ref_frame_t`, `box` (manual, normalised) |
| **TemporalEvent** (A) | `category` ∈ {normal, risk_event, near_miss, collision, uncertain}; `t_start`, `t_peak`, `t_end` (peak required for risk_event / near_miss / collision; start = when the risk becomes visible, **not** the impact); `duration` (derived); `actors` [actor labels]; `evidence` [EvidenceRef ≥ 1 for event categories]; `rationale` (required); `observation` ∈ {observed, inferred_from_sequence, insufficient_evidence}; `uncertainty` (C) |
| **Interaction** (B) | `subject` (actor), `object` (actor or ego), `relation` ∈ {APPROACHING, MOVING_AWAY, CUTTING_IN, FOLLOWING, CROSSING, BRAKING, YIELDING, BLOCKING, COLLISION_INTERACTION}; `t_start`, `t_end`; `evidence` (≥ 2 frames: a relation is never asserted from one frame); `annotator_confidence` (low / medium / high); `observation` (as above); `event_id` (optional link) |
| **Uncertainty** (C, embedded) | `visibility_level` ∈ {clear, partial, poor, not_visible}; `occlusion_level` ∈ {none, partial, heavy}; `motion_blur` ∈ {none, mild, severe}; `lighting_quality` ∈ {good, low, very_low, glare}; `annotation_confidence` ∈ {low, medium, high}; `uncertainty_reason` (required when the category is uncertain, the confidence is low, or the visibility is poor / not_visible) |
| **Uncertain vs not visible vs no event** | `category = normal` means "reviewed, no event". `uncertain` + `uncertainty_reason` means "cannot be determined". `visibility_level = not_visible` means "not visible". Disagreement between annotators is a **computed** state, never a label. **uncertain is never mapped to normal** (export and metrics keep it separate) |
| **EvidenceRef** (D) | `video_id`, `t`, `frame_index`, `track_id` / `box` (optional), `image` (JPEG of the frame stored in the encrypted store), `image_sha256`, `note` |
| **Counterfactual** (E, research only) | `event_id`; `observed_facts` (references to evidence and interactions); `interpretation` (annotator's reading); `hypothetical` (the what-if); `answerable` ∈ {yes, partly, not_answerable_from_video}; `research_only = true`. **Never exported as ground truth; never used for training** |
| **Submission** | annotator, video, timestamp, SHA-256 of the annotator's current records. A submitted video is locked for that annotator (later edits are recorded as a new revision with a reason) |
| **Adjudication** | by a coordinator, only after both annotators submitted. References the two annotators' record ids, the decision (accept_A / accept_B / merged / unresolved), and the adjudicated record. Originals are never changed |

## 3. Storage and audit trail
- **Append-only operation log.** One log per video and annotator: `ops/<video>/<annotator>.jsonl`.
  - Each entry is `{op: create | update | retract, record, at, annotator, reason, prev_hash, hash}`.
  - The **hash chain** makes edits and deletions detectable.
  - The current state is the fold of the log; any earlier version can be **restored**, which appends a new op. Nothing is
    deleted.
- **Evidence images** are saved as `evidence/<video>/<sha256>.jpg` and referenced by hash.
- **Location.** The store lives inside the encrypted session volume (`default_root()` next to `data/sessions`, i.e. on V:). If
  the volume is not mounted, the page refuses to write.

## 4. Blindness and roles
| Role | Can see | Cannot see |
|---|---|---|
| Annotator (pseudonym on the raw-access list for campaign videos; any pseudonym for Nexar dev) | the video, their own records | model predictions, warnings, SHAP, scene graph, KG output; **other annotators' records**, until both have submitted |
| Coordinator (adjudicator) | after submission: both annotators' records, agreement, disagreements; then model / graph comparison (E6) | — |
| Export | coordinator only; privacy gate (sharing policy) for any image or video content | — |

The **active annotation queue** may use model outputs to **order** videos, but it never shows them in the annotation view and
never writes labels.

## 5. Agreement (D): metrics chosen per label type
| Label type | Metric | Reason |
|---|---|---|
| event category (matched events) | Cohen's κ (with and without unmatched events as "none"); κ event vs not | nominal |
| event existence / extent | temporal IoU of matched events; F1 of event detection between annotators | interval data: κ is not meaningful |
| boundaries | mean / median absolute deviation of start, peak and end (s) | continuous |
| actors | actor matching by track id or box IoU ≥ 0.3 at the reference frame; actor-set F1 per matched event | set data |
| relations | relation tuples (subject, object, relation) matched on actors + tIoU ≥ 0.3; relation F1; κ only on relation type for matched actor pairs | structured data |
| ordinal uncertainty attributes | quadratic-weighted κ | ordinal |
| uncertainty rate | share of uncertain events and of low-confidence labels per annotator | descriptive |

## 6. Knowledge graph and XAI integration (E)
- **Triples.** Adjudicated (or per-annotator, marked so) records become triples with `source`, `target`, `relation`,
  `valid_from`, `valid_to`, `evidence_ref` (image hash + time), `provenance` (record id, version, annotator, adjudication) and
  `confidence`. Examples:
  - `Actor A —APPROACHING→ Actor B`;
  - `Actor B —BRAKING→ EvidenceFrame@12.0 s`;
  - `RiskEvent —SUPPORTED_BY→ EvidenceFrame`;
  - `RiskEvent —INVOLVES→ Actor A`.
- **Evidence lookup.** `evidence_for(triple)` returns the stored frames.
- **Scene-graph validation (E6).** Automatic relations of the scene graph are compared with human interactions on the same
  track ids: per-relation precision / recall and boundary deviation. It runs only on adjudicated human data.
- **Three explanation types, shown separately and never merged:**
  1. **model explanation:** TreeSHAP of the v3 model;
  2. **evidence explanation:** what was observed (human evidence frames; measured track values);
  3. **rule-based interpretation:** KG rules.

## 7. Active annotation (F)
**Priority score per dev video (no labels used).** It combines:
- v3 ensemble uncertainty (entropy, mutual information) and member disagreement;
- rarity of conditions (light, weather, scene metadata) and of rule patterns (R01 / R04 / R05);
- annotation disagreement, once it exists.

**Evaluation against random sampling under the same budget:**
- **Data.** A simulation on the **dev** split with the existing Nexar clip labels as the only oracle. These are human labels
  that already exist; no new labels are created.
- **Metrics.** Events found per budget, coverage of rare conditions, and a learning curve of a research LightGBM trained on
  the selected dev videos and evaluated on the held-out dev folds.
- **Not affected.** v3 is untouched.

## 8. Experiments (E1–E6)
Detailed in `SAFEDRIVE_V4_ANNOTATION_EXPERIMENTS.md`. Everything that needs real annotators is **NOT EVALUATED** until two
trained annotators have worked; no numbers are simulated. Only E5 can be evaluated now, and only as a simulation with existing
labels.

## 9. Implementation plan
| Phase | Content | Commit |
|---|---|---|
| B | schema, store (append-only log, hash chain, restore), temporal events and interactions, evidence capture | ✓ |
| C | uncertainty attributes, evidence images with hashes, validation rules | ✓ |
| D | submission lock, blindness, agreement, disagreements, adjudication | ✓ |
| E | KG triples with evidence lookup, the three-way XAI split in the Risk Explanation page | ✓ |
| F | counterfactual research records, active queue and the E5 simulation | ✓ |
| G | page 8, tests, experiments doc, schema, user guide, test report | ✓ |
