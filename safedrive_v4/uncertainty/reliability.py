"""Uncertainty and perception reliability, kept SEPARATE from risk.

  risk_probability      calibrated mean of the ensemble members (what the model predicts)
  model_uncertainty     ensemble mutual information (epistemic: members disagree) + entropy
  perception_reliability 0..1 from the scene graph / KG: detector confidence, track stability, occlusion, darkness, ego motion
  ood_score             Mahalanobis distance of the window feature vector to the dev distribution (diagonal, robust scales),
                        reported as a percentile of dev distances (>= 0.99 = outside 99 % of the training windows)
High uncertainty / low reliability is NEVER turned into high risk: it is reported as 'insufficient evidence / degraded input'.
"""
from __future__ import annotations

import numpy as np


def perception_reliability(kg_reliability: float, mean_det_conf: float | None, n_objects: float, mean_unc_corridor: float,
                           n_new: float) -> dict:
    parts = {"kg_reliability": float(np.clip(kg_reliability, 0, 1))}
    if n_objects >= 1 and mean_det_conf is not None:
        parts["detector_confidence"] = float(np.clip((mean_det_conf - 0.25) / 0.5, 0, 1))
    if mean_unc_corridor > 0:
        parts["corridor_track_reliability"] = float(np.clip(1 - mean_unc_corridor, 0, 1))
    if n_objects >= 3:
        parts["track_stability"] = float(np.clip(1 - n_new / n_objects, 0, 1))
    score = float(np.prod(list(parts.values()))) ** (1.0 / max(len(parts), 1))  # geometric mean
    return {"score": score, "components": parts}


class OODScorer:
    """Diagonal robust Mahalanobis distance to the dev windows; percentile against dev distances."""

    def fit(self, X: np.ndarray) -> "OODScorer":
        self.med = np.median(X, 0)
        mad = np.median(np.abs(X - self.med), 0) * 1.4826
        self.scale = np.where(mad > 1e-6, mad, X.std(0) + 1e-6)
        d = self._d(X)
        self.ref = np.sort(d)
        return self

    def _d(self, X: np.ndarray) -> np.ndarray:
        z = np.clip((X - self.med) / self.scale, -20, 20)
        return np.sqrt((z ** 2).mean(1))

    def percentile(self, X: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.ref, self._d(np.atleast_2d(X))) / len(self.ref)
