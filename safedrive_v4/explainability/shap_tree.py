"""Exact Tree SHAP for the LightGBM ensemble (LightGBM's built-in `pred_contrib=True`, i.e. the TreeSHAP algorithm of
Lundberg et al., Nature Machine Intelligence 2020). No extra dependency.

Attributions are in LOG-ODDS of the RAW (uncalibrated) member output; for an ensemble of M members the explanation is the mean of
the members' attributions (exact for the mean of member log-odds). The Platt calibration is a monotone map applied afterwards,
so the ranking and signs of contributions are unchanged; the report states both spaces.

Consistency checks (computed, not assumed):
  local_accuracy_error   |sum(phi) + phi0 - raw logit| for each member (must be ~0)
  member_rank_stability  mean Spearman correlation of |phi| rankings between members (attribution stability)
  deletion_fidelity      drop in predicted logit when the top-k positive features are replaced by their training median,
                         compared with replacing k random features (a faithful explanation removes more risk)
"""
from __future__ import annotations

import numpy as np


def _contrib(member, X: np.ndarray) -> np.ndarray:
    booster = member.booster_ if hasattr(member, "booster_") else member
    return np.asarray(booster.predict(X, pred_contrib=True))   # (N, d + 1), last column = expected value (bias)


def _raw(member, X: np.ndarray) -> np.ndarray:
    booster = member.booster_ if hasattr(member, "booster_") else member
    return np.asarray(booster.predict(X, raw_score=True))


def explain(members: list, X: np.ndarray) -> dict:
    X = np.atleast_2d(np.asarray(X, np.float64))
    C = np.stack([_contrib(m, X) for m in members])            # (M, N, d+1)
    raw = np.stack([_raw(m, X) for m in members])              # (M, N)
    phi, base = C[:, :, :-1], C[:, :, -1]
    la_err = float(np.max(np.abs(phi.sum(-1) + base - raw)))
    return {"phi": phi.mean(0), "base": base.mean(0), "raw_logit": raw.mean(0), "phi_members": phi, "local_accuracy_error": la_err}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 1.0


def member_rank_stability(phi_members: np.ndarray, top: int = 30) -> float:
    """Mean pairwise Spearman correlation of |phi| over the union of each sample's top features."""
    M, N, d = phi_members.shape
    vals = []
    for n in range(N):
        A = np.abs(phi_members[:, n, :])
        idx = np.unique(np.argsort(-A.mean(0))[:top])
        for i in range(M):
            for j in range(i + 1, M):
                vals.append(_spearman(A[i, idx], A[j, idx]))
    return float(np.mean(vals)) if vals else float("nan")


def deletion_fidelity(members: list, X: np.ndarray, phi: np.ndarray, median: np.ndarray, k: int = 10, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    X = np.atleast_2d(np.asarray(X, np.float64))
    base = np.mean([_raw(m, X) for m in members], 0)
    Xt, Xr = X.copy(), X.copy()
    for n in range(len(X)):
        top = np.argsort(-phi[n])[:k]
        Xt[n, top] = median[top]
        rnd = rng.choice(X.shape[1], k, replace=False)
        Xr[n, rnd] = median[rnd]
    dt = base - np.mean([_raw(m, Xt) for m in members], 0)
    dr = base - np.mean([_raw(m, Xr) for m in members], 0)
    return {"k": k, "mean_logit_drop_top": float(dt.mean()), "mean_logit_drop_random": float(dr.mean()),
            "fraction_top_beats_random": float((dt > dr).mean())}


def group_of(name: str) -> str:
    base = name.split("__")[0]
    if base.startswith("sg_"):
        return "scene_graph"
    if base.startswith("kg_"):
        return "knowledge_graph"
    if base.startswith(("res_", "corridor_", "lead_gap", "lead_brake", "cutin_", "pair_", "ego_ttr", "lead_res")):
        return "v3_interaction"
    if base.startswith(("fd_", "diff")):
        return "v3_frame_difference"
    return "v3_motion"


def top_contributions(phi_row: np.ndarray, names: list[str], x_row: np.ndarray, k: int = 8) -> list[dict]:
    idx = np.argsort(-np.abs(phi_row))[:k]
    return [{"feature": names[i], "group": group_of(names[i]), "value": float(x_row[i]), "shap_logit": float(phi_row[i]),
             "direction": "raises risk" if phi_row[i] > 0 else "lowers risk"} for i in idx]


def global_importance(phi: np.ndarray, names: list[str], k: int = 20) -> list[dict]:
    m = np.abs(phi).mean(0)
    idx = np.argsort(-m)[:k]
    grp: dict[str, float] = {}
    for i, n in enumerate(names):
        grp[group_of(n)] = grp.get(group_of(n), 0.0) + float(m[i])
    tot = sum(grp.values()) or 1.0
    return {"top_features": [{"feature": names[i], "group": group_of(names[i]), "mean_abs_shap": float(m[i])} for i in idx],
            "group_share": {g: v / tot for g, v in sorted(grp.items(), key=lambda kv: -kv[1])}}


def group_stability(phi_members: np.ndarray, names: list[str]) -> dict:
    """Agreement between members at the FEATURE-GROUP level (sum of |phi| per group): mean pairwise Spearman over groups and the
    share of windows where all members name the same top group."""
    groups = sorted({group_of(n) for n in names})
    gi = np.array([groups.index(group_of(n)) for n in names])
    M, N, _ = phi_members.shape
    G = np.zeros((M, N, len(groups)))
    for g in range(len(groups)):
        G[:, :, g] = np.abs(phi_members[:, :, gi == g]).sum(-1)
    rho = [_spearman(G[i, n], G[j, n]) for n in range(N) for i in range(M) for j in range(i + 1, M)] if len(groups) > 1 else [1.0]
    top_same = float(np.mean([len(set(G[:, n].argmax(-1))) == 1 for n in range(N)]))
    return {"group_spearman": float(np.mean(rho)), "same_top_group_all_members": top_same, "groups": groups}


def group_contributions(phi_row: np.ndarray, names: list[str]) -> list[dict]:
    """Signed SHAP sum per feature group (stable across ensemble members: group Spearman 0.90-0.95 on held-out windows,
    vs 0.06 for single-feature rankings, see experiments/v4/shap_global.json). Shown FIRST in explanations."""
    agg: dict[str, float] = {}
    for i, n in enumerate(names):
        agg[group_of(n)] = agg.get(group_of(n), 0.0) + float(phi_row[i])
    return [{"group": g, "shap_logit": v, "direction": "raises risk" if v > 0 else "lowers risk"}
            for g, v in sorted(agg.items(), key=lambda kv: -abs(kv[1]))]
