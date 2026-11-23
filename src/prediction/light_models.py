"""Lightweight tabular classifiers, seed-bagged ensembles (for uncertainty) and post-hoc calibration."""
from __future__ import annotations

import numpy as np

# Fixed, modest hyper-parameters (no early stopping on validation, so validation stays clean for model selection).
GBM_PARAMS = {
    "xgb": dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5, min_child_weight=2),
    "lgbm": dict(n_estimators=400, num_leaves=15, learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                 min_child_samples=20),
    "catboost": dict(iterations=400, depth=5, learning_rate=0.05, rsm=0.5),
}
COST_RANK = {"logreg": 0, "lgbm": 1, "xgb": 2, "catboost": 3}  # rough inference-cost order, used as a tie-breaker


def make_model(kind: str, seed: int = 0):
    if kind == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(**GBM_PARAMS["xgb"], random_state=seed, n_jobs=4, eval_metric="logloss", tree_method="hist")
    if kind == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**GBM_PARAMS["lgbm"], random_state=seed, n_jobs=4, verbose=-1)
    if kind == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**GBM_PARAMS["catboost"], random_seed=seed, verbose=0, thread_count=4)
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000, random_state=seed))
    raise ValueError(kind)


class BaggedEnsemble:
    """M members trained on bootstrap resamples (grouped by video) with different seeds.
    predict_samples -> (M, N) probabilities; the spread gives an ensemble uncertainty estimate."""

    def __init__(self, kind: str, n_members: int = 5, seed: int = 0):
        self.kind, self.n_members, self.seed = kind, n_members, seed
        self.members: list = []

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "BaggedEnsemble":
        rng = np.random.default_rng(self.seed)
        ug = np.unique(groups)
        rows = {g: np.where(groups == g)[0] for g in ug}
        self.members = []
        for i in range(self.n_members):
            pick = rng.choice(ug, size=len(ug), replace=True) if self.n_members > 1 else ug
            ii = np.concatenate([rows[g] for g in pick])
            m = make_model(self.kind, self.seed + i)
            m.fit(X[ii], y[ii])
            self.members.append(m)
        return self

    def predict_samples(self, X: np.ndarray) -> np.ndarray:
        return np.stack([m.predict_proba(X)[:, 1] for m in self.members])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.predict_samples(X).mean(0)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Calibrator:
    """method: 'none' | 'platt' (logistic on logit p) | 'isotonic'. Fitted on validation only."""

    def __init__(self, method: str = "none"):
        self.method = method
        self.model = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        if self.method == "platt":
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(C=1e3).fit(_logit(p)[:, None], y)
        elif self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p, y)
        return self

    def __call__(self, p: np.ndarray) -> np.ndarray:
        shape = np.shape(p)
        p = np.asarray(p, float).ravel()
        if self.method == "platt":
            out = self.model.predict_proba(_logit(p)[:, None])[:, 1]
        elif self.method == "isotonic":
            out = self.model.predict(p)
        else:
            out = p
        return np.asarray(out).reshape(shape)


def choose_calibration(p_val: np.ndarray, y_val: np.ndarray, groups: np.ndarray, folds: int = 5, seed: int = 0) -> tuple[str, dict]:
    """Pick none / platt / isotonic by grouped K-fold cross-validated NLL *within* the validation set."""
    from src.uncertainty.calibration import nll
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    rng.shuffle(ug)
    fold_of = {g: i % folds for i, g in enumerate(ug)}
    f = np.array([fold_of[g] for g in groups])
    scores = {}
    for method in ("none", "platt", "isotonic"):
        pred = np.zeros_like(p_val)
        for k in range(folds):
            tr, te = f != k, f == k
            pred[te] = Calibrator(method).fit(p_val[tr], y_val[tr])(p_val[te])
        scores[method] = nll(pred, y_val)
    return min(scores, key=scores.get), scores


def summarize_probs(ps: np.ndarray) -> dict[str, np.ndarray]:
    """ps: (S, N) member probabilities -> same keys as src.uncertainty.mc_dropout.summarize."""
    from src.uncertainty.mc_dropout import binary_entropy
    p = ps.mean(0)
    ent = binary_entropy(p)
    ale = binary_entropy(ps).mean(0)
    return {"p_mean": p, "p_std": ps.std(0), "entropy": ent, "aleatoric": ale, "mutual_info": np.maximum(ent - ale, 0.0)}
