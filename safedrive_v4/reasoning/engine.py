"""Reasoning engine: evaluates the rule base on each scene-graph snapshot, detects rule conflicts and summarises the evidence.

Outputs per step (ReasoningResult):
  findings            all rule findings (with node / edge / timestamp references)
  conflicts           contradictory evidence about the same object (e.g. risk evidence on a partially observed object,
                      or 'co-moving' SAME_DIRECTION together with a low-TTC finding); conflicts LOWER the risk evidence
  risk_evidence       noisy-OR of the risk findings' confidences (0..1); NOT a calibrated probability
  reliability         1 - noisy-OR of the reliability findings (0..1)
  insufficient        True when the path contains objects whose motion cannot be estimated yet
The engine never decides a warning on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from safedrive_v4.reasoning.rules import RULES, Finding, Rule
from safedrive_v4.scene_graph.schema import Snapshot


@dataclass
class ReasoningResult:
    step: int
    timestamp: float
    findings: list[Finding]
    conflicts: list[dict]
    risk_evidence: float
    reliability: float
    insufficient: bool
    per_rule: dict[str, float] = field(default_factory=dict)


def noisy_or(ps) -> float:
    ps = [p for p in ps if p > 0]
    return float(1.0 - np.prod([1.0 - min(p, 1.0) for p in ps])) if ps else 0.0


class ReasoningEngine:
    def __init__(self, cfg: dict, rules: list[Rule] | None = None):
        self.c = cfg["reasoning"] if "reasoning" in cfg else cfg
        self.rules = rules or RULES
        self.ctx: dict = {}

    def reset(self) -> None:
        self.ctx = {}

    def evaluate(self, s: Snapshot) -> ReasoningResult:
        findings: list[Finding] = []
        for r in self.rules:
            findings.extend(r.fn(r, s, self.ctx, self.c))
        conflicts = []
        occluded = {sub for f in findings if f.rule_id == "R07" for sub in f.subjects[:1]}
        co_moving = {e.source_node for e in s.edges_of("SAME_DIRECTION")}
        for f in findings:
            if f.kind != "risk" or not f.subjects:
                continue
            subj = f.subjects[0]
            if subj in occluded:
                conflicts.append({"type": "risk_on_partially_observed_object", "rule": f.rule_id, "subject": subj})
                f.confidence *= 0.8
                f.exceptions_applied.append("conflict: object partially occluded (x0.8)")
            if f.rule_id in ("R02", "R03") and subj in co_moving:
                conflicts.append({"type": "low_ttc_vs_same_direction", "rule": f.rule_id, "subject": subj})
                f.confidence *= 0.5
                f.exceptions_applied.append("conflict: object also judged co-moving (x0.5)")
        per_rule = {r.rule_id: 0.0 for r in self.rules}
        for f in findings:
            per_rule[f.rule_id] = max(per_rule[f.rule_id], f.confidence)
        risk = noisy_or(f.confidence for f in findings if f.kind == "risk")
        rel = 1.0 - noisy_or(f.confidence for f in findings if f.kind == "reliability")
        insufficient = any(f.kind == "insufficient" and f.confidence >= 0.45 for f in findings)
        return ReasoningResult(s.step, s.timestamp, findings, conflicts, risk, rel, insufficient, per_rule)
