"""Uncertainty-aware adaptive warning policy (v4). States:

  LOW                    risk below the MODERATE threshold (or not yet k consecutive steps)
  MODERATE               risk >= MODERATE threshold (dev OOF, 20 FA/h) for k = 3 consecutive steps
  HIGH                   risk >= HIGH threshold (dev OOF, 10 FA/h) for k steps; flagged `cautious` when perception reliability is
                         low or the ensemble disagrees: the warning is SHOWN (a high risk is never hidden) with "low confidence"
  INSUFFICIENT_EVIDENCE  risk not HIGH and the observation is not reliable enough to conclude (low perception reliability,
                         out-of-distribution window, or high model uncertainty)
  DEGRADED_INPUT         the camera input itself is unusable (stall, black, saturated, frozen, blurred): risk outputs are not
                         interpreted; this overrides every other state
Optional evidence gate (H4, policy P1): a MODERATE / HIGH onset additionally needs uncertainty-aware KG risk evidence
E_t >= evidence_min. It is OFF by default and only switched on if the held-out test of H4 supports it
(docs/SAFEDRIVE_V4_EXPERIMENT_PLAN.md §H4).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

LEVELS = ("LOW", "MODERATE", "HIGH", "INSUFFICIENT_EVIDENCE", "DEGRADED_INPUT")


@dataclass
class WarningDecision:
    level: str
    risk_probability: float
    model_uncertainty: float
    perception_reliability: float
    ood_percentile: float | None
    reasons: list[str]
    cautious: bool = False
    input_degraded: bool = False
    kg_evidence: float | None = None
    flags: dict = field(default_factory=dict)


class WarningPolicy:
    def __init__(self, moderate_thr: float, high_thr: float, consecutive: int = 3, reliability_min: float = 0.45,
                 mi_max: float | None = None, ood_max: float = 0.99, evidence_min: float | None = None):
        self.mod, self.high, self.k = float(moderate_thr), float(high_thr), int(consecutive)
        self.rel_min, self.mi_max, self.ood_max, self.evidence_min = reliability_min, mi_max, ood_max, evidence_min
        self.buf: deque = deque(maxlen=self.k)

    def reset(self) -> None:
        self.buf.clear()

    def decide(self, p: float, mi: float, reliability: float, ood: float | None = None, input_degraded: bool = False,
               input_reasons: list[str] | None = None, kg_evidence: float | None = None) -> WarningDecision:
        if input_degraded:
            self.buf.clear()   # risk outputs on unusable input are not interpreted (and do not count toward an onset)
            return WarningDecision("DEGRADED_INPUT", float(p), float(mi), float(reliability), ood, list(input_reasons or ["input degraded"]),
                                   input_degraded=True, kg_evidence=kg_evidence)
        self.buf.append(p)
        full = len(self.buf) == self.k
        reasons: list[str] = []
        gate_ok = self.evidence_min is None or (kg_evidence is not None and kg_evidence >= self.evidence_min)
        if full and min(self.buf) >= self.high and gate_ok:
            level = "HIGH"
            reasons.append(f"risk >= {self.high:.2f} for {self.k} consecutive steps")
        elif full and min(self.buf) >= self.mod and gate_ok:
            level = "MODERATE"
            reasons.append(f"risk >= {self.mod:.2f} for {self.k} consecutive steps")
        else:
            level = "LOW"
            if full and min(self.buf) >= self.mod and not gate_ok:
                reasons.append("risk above threshold but no knowledge-graph risk evidence (evidence gate)")
        weak = []
        if reliability < self.rel_min:
            weak.append(f"perception reliability {reliability:.2f} < {self.rel_min:.2f}")
        if ood is not None and ood >= self.ood_max:
            weak.append(f"input outside {100 * self.ood_max:.0f} % of training windows")
        if self.mi_max is not None and mi > self.mi_max:
            weak.append("ensemble members disagree (high model uncertainty)")
        cautious = False
        if weak:
            if level == "HIGH":
                cautious = True
                reasons += ["cautious: " + w for w in weak]
            else:
                level = "INSUFFICIENT_EVIDENCE"
                reasons += weak
        return WarningDecision(level, float(p), float(mi), float(reliability), None if ood is None else float(ood), reasons,
                               cautious=cautious, kg_evidence=kg_evidence)
