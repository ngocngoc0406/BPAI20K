"""Uncertainty-aware reasoning (A3 / H2): the same rule triggers as the plain engine, but every conclusion is a subjective-logic
opinion instead of a product of confidences.

Per risk finding f about road user x:
  premise   = edge_opinion(relation)  ∧  reliability_opinion(x)            (binomial conjunction: relation holds AND x is observed
                                                                            correctly; both from weighted evidence)
  exception = discount(premise, m_f)                                       (ego turning, occlusion, far road user: evidence is
                                                                            made less informative -> uncertainty rises, belief
                                                                            does not turn into disbelief)
  finding   = discount(exception, trust(rule))                             (trust discounting by the rule's base confidence)
Step risk opinion:   per road user the strongest finding (rules on one road user share evidence -> not fused);
                     across road users: cumulative fusion (independent sources).
Temporal opinion:    R_t = R_step ⊕ discount(R_{t-1}, λ)                    (evidence accumulates while a situation persists and
                                                                            ages away otherwise)
Observation reliability: fusion of the corridor road users' reliability opinions with the reliability findings (R07, R08) as
                     negative evidence. Insufficient evidence: road users in the path but the reliability opinion's uncertainty
                     is high, or the temporal risk opinion is mostly uncertainty while risk findings exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from safedrive_v4.reasoning.engine import ReasoningEngine, ReasoningResult
from safedrive_v4.reasoning.rules import RULES
from safedrive_v4.scene_graph.schema import REGION_IDS, Snapshot
from safedrive_v4.uncertainty.subjective_logic import Opinion, conjunction, cumulative_fusion, discount

RISK_RULES = [r.rule_id for r in RULES if r.kind == "risk"]
TRUST = {r.rule_id: r.base_confidence for r in RULES}


def _op(d: dict | None) -> Opinion:
    if not d:
        return Opinion.vacuous()
    b, dd, u = d["belief"], d["disbelief"], d["uncertainty"]
    s = b + dd + u
    return Opinion(b / s, dd / s, u / s, d["base_rate"])


@dataclass
class UncertainResult:
    base: ReasoningResult
    finding_opinions: dict[int, Opinion]          # index into base.findings -> opinion
    step_risk: Opinion
    temporal_risk: Opinion
    reliability: Opinion
    insufficient: bool
    per_rule: dict[str, float] = field(default_factory=dict)   # max expected probability per risk rule


class UncertainReasoningEngine:
    def __init__(self, cfg: dict, decay: float = 0.8, u_insufficient: float = 0.5):
        self.plain = ReasoningEngine(cfg)
        self.decay, self.u_ins = decay, u_insufficient
        self.prev: Opinion | None = None

    def reset(self) -> None:
        self.plain.reset()
        self.prev = None

    def evaluate(self, s: Snapshot) -> UncertainResult:
        base = self.plain.evaluate(s)
        edges = {e.edge_id: e for e in s.edges}
        ops: dict[int, Opinion] = {}
        best: dict[str, Opinion] = {}
        per_rule = {r: 0.0 for r in RISK_RULES}
        for i, f in enumerate(base.findings):
            if f.kind != "risk" or not f.edge_ids or f.edge_ids[0] not in edges:
                continue
            e = edges[f.edge_ids[0]]
            subj = f.subjects[0]
            node = s.nodes.get(subj)
            prem = conjunction(_op(e.opinion), _op(node.attrs.get("reliability_opinion") if node else None))
            denom = TRUST[f.rule_id] * max(e.confidence, 1e-6)
            m = float(np.clip(f.confidence / denom, 0.0, 1.0)) if e.confidence > 0 else 1.0   # exception / conflict factors
            op = discount(discount(prem, m), TRUST[f.rule_id])
            ops[i] = op
            per_rule[f.rule_id] = max(per_rule[f.rule_id], op.expectation)
            if subj not in best or op.expectation > best[subj].expectation:
                best[subj] = op
        step = cumulative_fusion(*best.values()) if best else Opinion.vacuous(0.1)
        temporal = cumulative_fusion(step, discount(self.prev, self.decay)) if self.prev is not None else step
        self.prev = temporal
        corr = [n for n in s.nodes.values() if n.object_id is not None and n.region in REGION_IDS]
        rel_ops = [_op(n.attrs.get("reliability_opinion")) for n in corr]
        neg = sum(f.confidence for f in base.findings if f.kind == "reliability")
        rel = cumulative_fusion(*(rel_ops + ([Opinion.from_evidence(0.0, neg)] if neg > 0 else []))) if (rel_ops or neg > 0) else Opinion.vacuous()
        insufficient = bool(corr) and (rel.u > self.u_ins or (bool(best) and temporal.u > 0.8))
        return UncertainResult(base, ops, step, temporal, rel, insufficient, per_rule)


KGU_FEATURES = ([f"kgu_rule_{r.lower()}" for r in RISK_RULES] +
                ["kgu_step_belief", "kgu_step_disbelief", "kgu_step_uncertainty", "kgu_step_expectation",
                 "kgu_temporal_belief", "kgu_temporal_uncertainty", "kgu_temporal_expectation",
                 "kgu_reliability_expectation", "kgu_reliability_uncertainty", "kgu_insufficient"])


def kgu_vector(r: UncertainResult) -> np.ndarray:
    v = [r.per_rule[k] for k in RISK_RULES]
    v += [r.step_risk.b, r.step_risk.d, r.step_risk.u, r.step_risk.expectation, r.temporal_risk.b, r.temporal_risk.u,
          r.temporal_risk.expectation, r.reliability.expectation, r.reliability.u, float(r.insufficient)]
    return np.array(v, np.float32)
