"""Non-neural baselines: majority class, logistic regression and XGBoost on pooled window features."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class PooledFeaturizer:
    """Window -> [motion mean, max, last, last-first] + PCA(32) of mean visual embedding."""

    def __init__(self, n_pca: int = 32, use_visual: bool = True, use_motion: bool = True):
        self.n_pca, self.use_visual, self.use_motion = n_pca, use_visual, use_motion
        self.pca: PCA | None = None

    def _pool(self, V: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mot = np.concatenate([M.mean(1), M.max(1), M[:, -1], M[:, -1] - M[:, 0]], 1)
        return V.mean(1), mot

    def fit(self, V: np.ndarray, M: np.ndarray) -> "PooledFeaturizer":
        v, _ = self._pool(V, M)
        if self.use_visual:
            self.pca = PCA(self.n_pca, random_state=0).fit(v)
        return self

    def transform(self, V: np.ndarray, M: np.ndarray) -> np.ndarray:
        v, m = self._pool(V, M)
        parts = []
        if self.use_visual:
            parts.append(self.pca.transform(v))
        if self.use_motion:
            parts.append(m)
        return np.concatenate(parts, 1)


class MajorityBaseline:
    def fit(self, y: np.ndarray) -> "MajorityBaseline":
        self.p = float(np.mean(y))
        return self

    def predict_proba(self, n: int) -> np.ndarray:
        return np.full(n, self.p)


def logistic_regression(seed: int = 0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, random_state=seed))


def xgboost(seed: int = 0):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         random_state=seed, n_jobs=4, eval_metric="logloss")
