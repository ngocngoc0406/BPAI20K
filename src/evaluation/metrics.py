"""Metric definitions (see docs/EXPERIMENT_DESIGN.md for formal definitions)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from src.uncertainty.calibration import brier, ece


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[i]) if len(thr) else 0.5


def threshold_for_precision(y: np.ndarray, p: np.ndarray, target: float) -> float:
    prec, _rec, thr = precision_recall_curve(y, p)
    ok = np.where(prec[:-1] >= target)[0]
    return float(thr[ok[0]]) if len(ok) else float(thr[-1]) if len(thr) else 0.9


def threshold_for_recall(y: np.ndarray, p: np.ndarray, target: float) -> float:
    _prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target)[0]
    return float(thr[ok[-1]]) if len(ok) else float(thr[0]) if len(thr) else 0.1


def classification_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict[str, float]:
    yh = (p >= thr).astype(int)
    two = len(np.unique(y)) > 1
    return {
        "accuracy": float(accuracy_score(y, yh)),
        "precision": float(precision_score(y, yh, zero_division=0)),
        "recall": float(recall_score(y, yh, zero_division=0)),
        "f1": float(f1_score(y, yh, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if two else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if two else float("nan"),
        "brier": brier(p, y),
        "ece": ece(p, y),
        "threshold": float(thr),
    }


def nexar_map(y: np.ndarray, p: np.ndarray, groups: np.ndarray, horizons: list[float]) -> dict[str, float]:
    """Nexar-style mAP: AP computed separately per horizon (positives at that horizon vs all negatives),
    then averaged over horizons."""
    neg = groups == "neg"
    out = {}
    for h in horizons:
        m = (groups == f"h{h}") | neg
        if m.sum() and len(np.unique(y[m])) > 1:
            out[f"ap_h{h}"] = float(average_precision_score(y[m], p[m]))
    out["mAP"] = float(np.mean(list(out.values()))) if out else float("nan")
    return out


def selective_risk(y: np.ndarray, p: np.ndarray, unc: np.ndarray, thr: float,
                   coverages=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)) -> list[dict]:
    """Risk-coverage: abstain on the most uncertain (1-coverage) fraction, report accuracy/F1 on the rest."""
    order = np.argsort(unc)
    rows = []
    for c in coverages:
        k = max(int(round(c * len(y))), 1)
        keep = order[:k]
        yh = (p[keep] >= thr).astype(int)
        rows.append({"coverage": c, "n": k, "accuracy": float((yh == y[keep]).mean()),
                     "f1": float(f1_score(y[keep], yh, zero_division=0))})
    return rows


def error_detection_auroc(y: np.ndarray, p: np.ndarray, unc: np.ndarray, thr: float) -> float:
    """AUROC of the uncertainty score for separating wrong from correct predictions."""
    err = ((p >= thr).astype(int) != y).astype(int)
    return float(roc_auc_score(err, unc)) if 0 < err.sum() < len(err) else float("nan")
