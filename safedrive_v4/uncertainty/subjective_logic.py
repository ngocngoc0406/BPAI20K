"""Binomial subjective-logic opinions (Jøsang, "Subjective Logic", Springer 2016) for uncertainty-aware graph reasoning.

An opinion w = (b, d, u, a) about a proposition (e.g. "obj:7 is cutting in front of ego") with b + d + u = 1:
  b belief, d disbelief, u uncertainty (lack of evidence), a base rate (prior probability).
It is equivalent to a Beta distribution built from evidence: r positive and s negative observations with non-informative
prior weight W = 2:   b = r / (r + s + W),  d = s / (r + s + W),  u = W / (r + s + W);   E[p] = b + a u.

Why this instead of multiplying confidences: a product of confidences has no probabilistic meaning when the factors are not
independent probabilities, and it cannot distinguish "the relation is unlikely" from "we have no evidence". Opinions keep the
two apart (d vs u), aggregate repeated observations correctly (evidence counts add up) and have standard operators:
  conjunction (w_x ∧ w_y)   binomial multiplication: opinion that both premises hold (rule premises)
  cumulative fusion (⊕)     combine opinions from independent evidence sources (several rules / objects)
  discount(w, λ)            evidence aging: older evidence counts λ^Δt (temporal propagation)
  weighted evidence         each observation contributes with weight = observation reliability in [0, 1]
                            (detector confidence x track stability), i.e. an unreliable observation is partly missing
                            evidence, which raises u instead of raising d.
"""
from __future__ import annotations

from dataclasses import dataclass

W_PRIOR = 2.0


@dataclass(frozen=True)
class Opinion:
    b: float
    d: float
    u: float
    a: float = 0.5

    def __post_init__(self):
        s = self.b + self.d + self.u
        if not (abs(s - 1.0) < 1e-6 and min(self.b, self.d, self.u) >= -1e-9 and 0.0 <= self.a <= 1.0):
            raise ValueError(f"invalid opinion {self}")

    @property
    def expectation(self) -> float:
        return self.b + self.a * self.u

    @classmethod
    def vacuous(cls, a: float = 0.5) -> "Opinion":
        return cls(0.0, 0.0, 1.0, a)

    @classmethod
    def from_evidence(cls, r: float, s: float, a: float = 0.5, W: float = W_PRIOR) -> "Opinion":
        r, s = max(r, 0.0), max(s, 0.0)
        k = r + s + W
        return cls(r / k, s / k, W / k, a)

    def evidence(self, W: float = W_PRIOR) -> tuple[float, float]:
        if self.u <= 1e-12:
            return float("inf"), float("inf")
        return W * self.b / self.u, W * self.d / self.u

    def to_dict(self) -> dict:
        return {"belief": round(self.b, 4), "disbelief": round(self.d, 4), "uncertainty": round(self.u, 4),
                "base_rate": round(self.a, 4), "expected_probability": round(self.expectation, 4)}


def conjunction(x: Opinion, y: Opinion) -> Opinion:
    """Binomial multiplication (Jøsang 2016, normal multiplication): opinion that x AND y hold.
    Expectation satisfies E[x ∧ y] = E[x] · E[y] (assumes x and y are independent propositions)."""
    ax, ay = x.a, y.a
    a = ax * ay
    b = x.b * y.b
    d = x.d + y.d - x.d * y.d
    if a < 1.0 - 1e-12:
        b = x.b * y.b + ((1 - ax) * ay * x.b * y.u + ax * (1 - ay) * x.u * y.b) / (1 - a)
        u = x.u * y.u + ((1 - ay) * x.b * y.u + (1 - ax) * x.u * y.b) / (1 - a)
    else:
        u = x.u * y.u
    # numerical clean-up so that b + d + u = 1
    tot = b + d + u
    return Opinion(max(b, 0) / tot, max(d, 0) / tot, max(u, 0) / tot, a)


def cumulative_fusion(*ops: Opinion) -> Opinion:
    """Cumulative fusion of opinions from INDEPENDENT sources = adding their evidence (Jøsang 2016, ch. 12)."""
    ops = [o for o in ops if o is not None]
    if not ops:
        return Opinion.vacuous()
    if len(ops) == 1:
        return ops[0]
    r = s = 0.0
    for o in ops:
        ri, si = o.evidence()
        r, s = r + ri, s + si
    a = sum(o.a for o in ops) / len(ops)
    if r == float("inf") or s == float("inf"):
        dog = [o for o in ops if o.u <= 1e-12]
        return Opinion(sum(o.b for o in dog) / len(dog), sum(o.d for o in dog) / len(dog), 0.0, a)
    return Opinion.from_evidence(r, s, a)


def discount(o: Opinion, factor: float) -> Opinion:
    """Evidence aging / trust discounting: evidence is multiplied by `factor` in [0, 1]; lost evidence becomes uncertainty."""
    r, s = o.evidence()
    if r == float("inf"):
        return Opinion(o.b * factor, o.d * factor, 1 - factor * (o.b + o.d), o.a)
    return Opinion.from_evidence(r * factor, s * factor, o.a)


def weighted_observations(obs: list[tuple[bool, float]], a: float = 0.5, decay: float = 1.0) -> Opinion:
    """obs: oldest .. newest (condition held?, reliability weight in [0,1]). Older observations are aged by decay^age."""
    r = s = 0.0
    n = len(obs)
    for i, (hit, w) in enumerate(obs):
        wt = max(0.0, min(1.0, w)) * decay ** (n - 1 - i)
        if hit:
            r += wt
        else:
            s += wt
    return Opinion.from_evidence(r, s, a)
