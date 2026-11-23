"""Risk assessment: maps calibrated collision probability + uncertainty + temporal trend + motion cues
to a graded risk level with grounded triggering factors.

risk_score is deliberately NOT the collision probability:
    risk_score = clip( p + w_trend * max(0, dp/dt over the last 1 s) + w_loom * L, 0, 1 )
    with L = clip(z(looming_center) / 3, 0, 1) the standardised image-plane looming cue.
Thresholds are selected on the validation split (see scripts/run_experiments.py):
    CAUTION  : score at which validation recall >= 0.90
    WARNING  : F1-optimal validation threshold
    HIGH RISK: score at which validation precision >= 0.90
UNCERTAIN  : epistemic uncertainty (mutual information) above its validation quantile while the
             score is in the ambiguous band [CAUTION, HIGH RISK) — the system abstains instead of
             forcing a safe/dangerous verdict.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from src.features.motion_features import FEATURE_DESCRIPTIONS, MOTION_FEATURES

LEVELS = ["NORMAL", "CAUTION", "WARNING", "HIGH RISK", "UNCERTAIN"]
LEVEL_COLORS = {"NORMAL": "#22c55e", "CAUTION": "#eab308", "WARNING": "#f97316", "HIGH RISK": "#ef4444",
                "UNCERTAIN": "#a855f7"}

# Motion cues that can be reported as triggering factors, with the direction that indicates danger.
CUES = {"looming_center": +1, "looming_all": +1, "tau_center": -1, "lead_lateral_in": +1, "max_lateral_in": +1,
        "max_pair_closing": +1, "max_pair_iou": +1, "lead_bottom_y": +1, "lead_area": +1, "flow_div_center": +1,
        "n_persons": +1, "n_two_wheelers": +1, "max_accel": +1}


@dataclass
class RiskConfig:
    caution: float = 0.3
    warning: float = 0.5
    high: float = 0.8
    mi_threshold: float = 0.05
    trend_weight: float = 0.15
    looming_weight: float = 0.10
    consecutive_steps: int = 3
    level_signal: str = "risk_score"  # 'risk_score' (v1/v2) or 'probability' (v3: thresholds are FA-budgeted on calibrated p)
    cue_reference: dict = field(default_factory=dict)  # feature -> {"q50":..,"q90":..,"q10":..} from train set

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RiskConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RiskAssessment:
    timestamp: float
    level: str
    risk_score: float
    prediction_score: float
    uncertainty: float
    entropy: float
    trend: float
    triggering_factors: list[dict]
    explanation: str


def cue_reference(motion: np.ndarray) -> dict:
    ref = {}
    for name in CUES:
        col = motion[:, MOTION_FEATURES.index(name)]
        ref[name] = {"q10": float(np.quantile(col, 0.10)), "q50": float(np.quantile(col, 0.5)),
                     "q90": float(np.quantile(col, 0.90)), "std": float(col.std() + 1e-6)}
    return ref


class RiskEngine:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    def _trend(self, times: np.ndarray, p: np.ndarray, i: int) -> float:
        m = (times > times[i] - 1.0) & (times <= times[i])
        if m.sum() < 2:
            return 0.0
        return float(np.polyfit(times[m], p[m], 1)[0])

    def factors(self, motion_row: np.ndarray) -> list[dict]:
        """Motion cues that are unusually high (above train q90, or below q10 for tau) at this step."""
        out = []
        for name, sign in CUES.items():
            ref = self.cfg.cue_reference.get(name)
            if not ref:
                continue
            v = float(motion_row[MOTION_FEATURES.index(name)])
            hit = v >= ref["q90"] and v > ref["q50"] if sign > 0 else v <= ref["q10"] and v < ref["q50"]
            if hit:
                strength = abs(v - ref["q50"]) / ref["std"]
                out.append({"feature": name, "value": v, "typical": ref["q50"], "strength": float(strength),
                            "description": FEATURE_DESCRIPTIONS[name]})
        return sorted(out, key=lambda d: -d["strength"])[:4]

    def level_for(self, score: float, mi: float) -> str:
        c = self.cfg
        if c.caution <= score < c.high and mi >= c.mi_threshold:
            return "UNCERTAIN"
        if score >= c.high:
            return "HIGH RISK"
        if score >= c.warning:
            return "WARNING"
        if score >= c.caution:
            return "CAUTION"
        return "NORMAL"

    def risk_scores(self, times: np.ndarray, p: np.ndarray, loom: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        tr = np.array([self._trend(times, p, i) for i in range(len(p))])
        s = p + self.cfg.trend_weight * np.maximum(tr, 0.0)
        if loom is not None and "looming_center" in self.cfg.cue_reference:
            ref = self.cfg.cue_reference["looming_center"]
            s = s + self.cfg.looming_weight * np.clip((loom - ref["q50"]) / ref["std"] / 3.0, 0, 1)
        return np.clip(s, 0.0, 1.0), tr

    def assess(self, times: np.ndarray, pred: dict[str, np.ndarray], motion_rows: np.ndarray) -> list[RiskAssessment]:
        """pred: output of EnsemblePredictor.predict_video; motion_rows: raw motion feature rows at each time."""
        p, mi, ent = pred["p_mean"], pred["mutual_info"], pred["entropy"]
        loom = motion_rows[:, MOTION_FEATURES.index("looming_center")]
        score, tr = self.risk_scores(times, p, loom)
        out = []
        for i, t in enumerate(times):
            sig = float(p[i]) if self.cfg.level_signal == "probability" else float(score[i])
            lvl = self.level_for(sig, float(mi[i]))
            fac = self.factors(motion_rows[i]) if lvl != "NORMAL" else []
            out.append(RiskAssessment(float(t), lvl, float(score[i]), float(p[i]), float(mi[i]), float(ent[i]),
                                      float(tr[i]), fac, explain(lvl, float(p[i]), float(mi[i]), float(tr[i]), fac)))
        return out


def explain(level: str, p: float, mi: float, trend: float, factors: list[dict]) -> str:
    """Template explanation built only from computed quantities (no free-text generation)."""
    parts = [f"{level}: calibrated collision probability {p:.0%}"]
    if trend > 0.05:
        parts.append(f"rising {trend:+.2f}/s over the last second")
    if level == "UNCERTAIN":
        parts.append(f"model disagreement (mutual information {mi:.3f}) is high - more evidence needed")
    for f in factors[:3]:
        parts.append(f"{f['description']} = {f['value']:.3g} (typical {f['typical']:.3g})")
    return "; ".join(parts) + "."
