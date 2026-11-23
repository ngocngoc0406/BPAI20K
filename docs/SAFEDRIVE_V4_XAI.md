# SafeDrive AI v4: Explainable-AI Design

Every explanation element is computed from pipeline data. No language model generates causes or evidence. A text rendering,
if one is ever added, may only paraphrase stored fields.

## 1. Feature level: exact Tree SHAP
- **Method.** LightGBM's built-in `pred_contrib=True` is the TreeSHAP algorithm (Lundberg et al. 2020). It is exact for tree
  ensembles and needs no extra dependency (the `shap` package is not installed).
- **Space.** Log-odds of the raw member output; the ensemble explanation is the mean over the 5 members. The Platt calibration is
  monotone, so the signs and the ranking of contributions carry over to the probability.
- **Outputs.**
  - Local: top factors with value, contribution and direction.
  - Global: mean |SHAP| per feature, and the share per group (v3 motion, v3 frame difference, v3 interaction, scene graph,
    knowledge graph).
- **Consistency checks** (computed in `safedrive_v4/explainability/shap_tree.py` and reported in experiment plan §5.3):
  - *local accuracy:* Σφ + φ0 = logit, with max error reported;
  - *attribution stability:* Spearman correlation of |φ| rankings between ensemble members;
  - *deletion fidelity:* setting the top-10 positive features to the dev median must lower the logit more than 10 random ones.

## 2. Graph level: reasoning chains
For a warning, `graph_explain.chains` lists the strongest risk findings. Each chain gives:
- the rule id, the conclusion and the finding's confidence;
- the filled explanation template;
- the KG path, for example `obj:12 → CUTTING_IN_FRONT_OF → ego → INVOLVES LateralIntrusion → INCREASES PotentialCollisionRisk`;
- for every graph step: node id, node type, relation, target, edge id, edge confidence, `observed_since_s`, `timestamp_s` and
  the edge's numeric evidence;
- the exceptions applied (turning, occlusion, conflicts).

`validate_chain` checks that every referenced node and edge exists in the snapshot at that timestamp. This is the
**graph-evidence validity** metric.

## 3. Counterfactuals: trajectory-level and physically consistent
Edits act on the **observed trajectory of one road user** over the last 2 s:

| What-if | Edit |
|---|---|
| closing speed halved | box scale s(t) = s(t0) · (s_obs / s(t0))^0.5 |
| image TTC doubled | growth exponent 1/2 |
| no cut-in | lateral centre frozen at its value 2 s earlier |

**All** features are then recomputed from the edited tracks with the factual code (v3 object-motion and interaction features,
scene graph, rules, windows). Features are never edited independently, which avoids impossible combinations.

**Validity checks** (per counterfactual):
- **identity reproduction:** an empty edit must give the factual prediction, error < 1e-4. Recomputation of the v3 motion
  features from tracks reproduces the cache **exactly**: max difference 0.0 on the tested videos (unit test).
- box validity, clipping to the frame when the edited object leaves the view (as a detector would observe it), and per-step
  change not larger than observed.

**Stated limitation.** The optical-flow and frame-difference channels are computed from pixels and cannot be regenerated for an
edited trajectory, so they are kept as observed (`pixel_channels_kept = True`).

## 4. Evidence trace (per warning)
`safedrive_v4/event_memory/trace.py` stores:
- event id, video, timestamp, step, frame references and track ids;
- the graph nodes and edges used, rule findings and KG paths;
- the prediction, the top SHAP factors, uncertainty (mutual information, entropy, perception reliability, SHAP local-accuracy
  error) and the warning decision with its reasons;
- **unknowns**: what the system could not determine, for example physical distance and TTC, traffic lights, lanes, occluded
  extents and unresolved tracks;
- provenance: version, feature schema and config hash.

The trace is sealed with SHA-256 over its canonical JSON. `verify()` re-checks the hash and the references (tested: tampering and
dangling references are detected).

## 5. Uncertainty kept separate from risk
| Quantity | Source |
|---|---|
| risk probability | calibrated mean of the 5 members |
| model uncertainty | ensemble mutual information + entropy |
| perception reliability | geometric mean of KG reliability, detector confidence, corridor track reliability and track stability |
| out-of-distribution | robust Mahalanobis percentile against dev windows |
| warning severity | the policy |

**Policy rule.** Low reliability gives INSUFFICIENT_EVIDENCE, never HIGH. A HIGH risk is never hidden by low reliability; its
reliability is displayed next to it.

## 6. What is not evaluated
Human evaluation of explanation usefulness is **pending**: it needs the trained annotators (V3.4.4 item 14), and no rating was
collected.

## 7. Product round additions
**Traces.** They now also store:
- the subjective-logic opinions of the edges used, of each finding, and of the step and temporal KG risk;
- the input-quality reasons;
- the 5-state decision, including the cautious flag.

**Hypothesis H3**, evaluated in experiment plan §6:
- graph evidence is **verifiable** (100 % valid references, sealed traces);
- SHAP is **faithful** (deletion test);
- graph chains are **not** an account of the v3 model's computation, and they exist for 47 % of warnings. The UI says so when no
  rule fired.
