# SafeDrive AI V4: Uncertainty

**Four quantities are kept separate and never merged into "risk":**
- collision risk;
- model uncertainty;
- perception reliability;
- input quality.

## 1. Quantities
| Quantity | Computation | Used for |
|---|---|---|
| Collision risk p | calibrated mean of the 5 v3 LightGBM members (frozen model) | warning thresholds |
| Model uncertainty | ensemble mutual information + entropy | INSUFFICIENT_EVIDENCE / cautious HIGH (optional threshold) |
| Perception reliability | subjective-logic opinion "the observations of the road users in the path are correct" (§2) | INSUFFICIENT_EVIDENCE (< 0.45) / cautious HIGH |
| KG risk evidence | temporal subjective-logic opinion of the risk findings (§3) | explanation, event category, optional evidence gate (H4-P1) |
| Input quality | `InputQualityMonitor`: stall, black, saturated, frozen, blurred (fixed engineering thresholds) | DEGRADED_INPUT |
| OOD percentile | robust Mahalanobis distance of the window features to dev windows | INSUFFICIENT_EVIDENCE (≥ 0.99) |

## 2. Uncertainty-aware knowledge graph: subjective logic
Reference: Jøsang, *Subjective Logic* (Springer, 2016).

**Binomial opinion.** An opinion is w = (b, d, u, a) with b + d + u = 1. It is equivalent to a Beta distribution with r positive
and s negative pieces of evidence and prior weight W = 2:
- b = r / (r + s + 2), d = s / (r + s + 2), u = 2 / (r + s + 2);
- the expected probability is E = b + a·u.

**Why not multiply confidences.**
- A product of confidences is a probability only for independent probabilities.
- It cannot tell "unlikely" (high d) apart from "no evidence" (high u).
- It does not accumulate repeated observations.

**Where the opinions come from:**
- **Edge opinion.** For each (source, target, relation), the last 10 steps of the condition (held / not held) are weighted by
  the reliability of the involved observations and aged by 0.9 per step. An unreliable observation is partly *missing*
  evidence, so it raises u, not d. Base rate a = 0.1 (prior that a relation holds; fixed, not tuned).
- **Node reliability opinion.** Detector confidence over the last 10 steps is used as evidence that "this detection is
  correct" (conf for, 1 − conf against). It is discounted for unstable motion (1 / (1 + 5·σ_speed)) and by 0.7 for partial
  occlusion or truncation.

## 3. Reasoning with opinions (`reasoning/uncertain_engine.py`)
- **Finding.** finding = discount(discount(edge ∧ node_reliability, m_exceptions), trust(rule)).
  - ∧ is the binomial conjunction, where E[x ∧ y] = E[x]·E[y] (verified to 3e-16).
  - Exceptions (ego turning, occlusion, far road user) and the rule's base confidence are applied as **trust discounting**:
    the evidence becomes less informative, and uncertainty rises.
- **Step risk opinion.**
  - For each road user, the strongest finding is kept. Rules about the same road user share evidence, so they are not fused.
  - Across road users, cumulative fusion is used: independent evidence adds up.
- **Temporal risk opinion.** R_t = R_step ⊕ discount(R_{t−1}, 0.8). Evidence accumulates while a situation lasts and ages into
  uncertainty afterwards (tested).
- **Observation reliability.** The node reliability opinions of the road users in the path are fused with the reliability
  findings (occlusion, perception degradation) as negative evidence.
- **Insufficient evidence.** Road users are in the path and the reliability opinion has u > 0.5, or risk findings exist and the
  temporal risk opinion has u > 0.8.
- **Cost.** 0.9 ms per step including the graph.

**Interpretation note.** A low detector confidence counts as evidence *against* the detection, so it raises disbelief. It is
not a lack of evidence. A short, unstable or occluded track yields **uncertainty** (tests: `test_v4_uncertainty_events.py`).

## 4. Input quality (`uncertainty/input_quality.py`)
DEGRADED_INPUT is triggered by any of:
- median frame interval > 0.25 s, or no frame for > 1 s;
- mean brightness < 0.05 (black);
- more than 60 % of pixels saturated;
- frame-to-frame mean absolute difference < 0.5 grey level for ≥ 10 frames (frozen);
- Laplacian variance < 15 (blurred).

"Dark" (< 0.12) is reported but does not by itself degrade the input. In replay mode the cache brightness and frame-difference
channels are used.

## 5. Hypotheses
- H2 (uncertainty-aware reasoning limits the effect of perception errors) and H4 (adaptive warning) are tested in
  [SAFEDRIVE_V4_EXPERIMENT_PLAN.md](SAFEDRIVE_V4_EXPERIMENT_PLAN.md).
- **Perception-error model used for H2** (track level, deterministic per video):
  - 15 % detection dropout;
  - box jitter of 3 % of the box size;
  - 2 % id switches per step;
  - 5 % ghost detections per step.
  - The v3 object-motion features are recomputed from the noisy tracks; pixel channels stay unchanged.
