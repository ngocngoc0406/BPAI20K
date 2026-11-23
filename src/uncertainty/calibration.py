"""Calibration: temperature scaling (fit on validation only), ECE, Brier score, reliability data."""
from __future__ import annotations

import numpy as np


def nll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(logits: np.ndarray, y: np.ndarray, grid: np.ndarray | None = None) -> float:
    """logits: (S, N) sampled logits. Picks T minimising NLL of the MC-averaged probability."""
    grid = grid if grid is not None else np.exp(np.linspace(np.log(0.05), np.log(20.0), 300))
    best_t, best = 1.0, np.inf
    for t in grid:
        p = (1.0 / (1.0 + np.exp(-logits / t))).mean(0)
        v = nll(p, y)
        if v < best:
            best_t, best = float(t), v
    return best_t


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def reliability(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict[str, list[float]]:
    edges = np.linspace(0, 1, n_bins + 1)
    conf, acc, cnt = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        cnt.append(int(m.sum()))
        conf.append(float(p[m].mean()) if m.any() else float((lo + hi) / 2))
        acc.append(float(y[m].mean()) if m.any() else float("nan"))
    return {"bin_lower": edges[:-1].tolist(), "confidence": conf, "frequency": acc, "count": cnt}


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error of the positive-class probability (equal-width bins)."""
    r = reliability(p, y, n_bins)
    n = len(p)
    return float(sum(c / n * abs(f - cf) for cf, f, c in zip(r["confidence"], r["frequency"], r["count"]) if c > 0))
