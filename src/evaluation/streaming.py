"""Streaming early-warning metrics at fixed false-alarm budgets (defined before the v3 experiments).

valid alarm      an alarm onset in [t_event - MAX_LEAD, t_event)   (MAX_LEAD = 5 s: the longest annotated alert lead is 4.5 s;
                 earlier onsets are not anticipations of this event)
recall@B         fraction of positive videos with a valid alarm, at the threshold whose false-alarm rate on negative videos
                 is <= B per hour
EDR@X            early-detection rate: fraction of positive videos with a valid alarm onset at or before t_event - X
FA/h             alarm onsets on negative (normal-driving) videos per hour (onset rule: k consecutive steps, re-arm after k below)
Post-processing  optional EMA smoothing of the score (alpha=1 means none) before the onset rule; causal.
"""
from __future__ import annotations

import numpy as np

from src.evaluation.early_warning import alarm_onsets

MAX_LEAD = 5.0


def ema(x: np.ndarray, alpha: float) -> np.ndarray:
    if alpha >= 1.0:
        return x
    out = np.empty_like(x)
    acc = x[0] if len(x) else 0.0
    for i, v in enumerate(x):
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def _prep(streams: dict, recs: dict, warmup: float, alpha: float):
    pos, neg = [], []
    for vid, s in streams.items():
        t, sc = np.asarray(s["times"]), ema(np.asarray(s["score"], float), alpha)
        keep = t >= warmup - 1e-6
        t, sc = t[keep], sc[keep]
        r = recs[vid]
        if r.label == 1 and r.time_of_event is not None:
            m = t < r.time_of_event
            pos.append((t[m], sc[m], r.time_of_event))
        elif r.label == 0:
            neg.append((t, sc))
    hours = sum(float(t[-1] - t[0]) for t, _ in neg if len(t) > 1) / 3600.0
    return pos, neg, hours


def evaluate_at(streams: dict, recs: dict, thr: float, k: int = 3, warmup: float = 1.0, alpha: float = 1.0,
                leads=(1.0, 2.0, 3.0), prepared=None) -> dict:
    pos, neg, hours = prepared or _prep(streams, recs, warmup, alpha)
    fa = sum(len(alarm_onsets(t, s, thr, k)) for t, s in neg)
    hit, early, first_leads = 0, {x: 0 for x in leads}, []
    for t, s, te in pos:
        ons = [o for o in alarm_onsets(t, s, thr, k) if o >= te - MAX_LEAD]
        if ons:
            hit += 1
            first_leads.append(te - ons[0])
            for x in leads:
                early[x] += int(ons[0] <= te - x)
    n = max(len(pos), 1)
    return {"threshold": float(thr), "fa_per_hour": fa / hours if hours > 0 else float("nan"), "false_alarms": fa,
            "recall": hit / n, **{f"edr@{x:g}s": early[x] / n for x in leads},
            "median_lead_s": float(np.median(first_leads)) if first_leads else float("nan"), "n_pos": len(pos), "neg_hours": hours}


def operating_points(streams: dict, recs: dict, budgets=(5, 10, 20), k: int = 3, warmup: float = 1.0, alpha: float = 1.0,
                     grid: np.ndarray | None = None) -> dict:
    """For each FA budget, the lowest threshold whose FA/h <= budget on these streams, and the metrics there."""
    prep = _prep(streams, recs, warmup, alpha)
    if grid is None:
        allneg = np.concatenate([s for _, s in prep[1]]) if prep[1] else np.array([0.5])
        grid = np.unique(np.r_[np.quantile(allneg, np.linspace(0.80, 1.0, 81)), [1.0 + 1e-6]])
    rows = [evaluate_at(streams, recs, float(th), k, warmup, alpha, prepared=prep) for th in grid]
    out = {}
    for b in budgets:
        ok = [r for r in rows if r["fa_per_hour"] <= b]
        out[f"@{b}FA"] = min(ok, key=lambda r: r["threshold"]) if ok else rows[-1]
    out["curve"] = [{k2: r[k2] for k2 in ("threshold", "fa_per_hour", "recall", "edr@1s", "edr@2s", "edr@3s")} for r in rows]
    return out


def per_video_table(streams: dict, recs: dict, grid: np.ndarray, k: int = 3, warmup: float = 1.0, alpha: float = 1.0) -> dict:
    """Per-video, per-threshold counts so that any resample of videos can be scored by weighted sums (fast bootstrap).
    Returns ids, is_pos, hours (neg), fa[v, j], hit[v, j] (valid onset), and the scores are identical to evaluate_at."""
    ids, is_pos, hours, fa, hit = [], [], [], [], []
    for vid, s in streams.items():
        r = recs[vid]
        t, sc = np.asarray(s["times"]), ema(np.asarray(s["score"], float), alpha)
        keep = t >= warmup - 1e-6
        t, sc = t[keep], sc[keep]
        if r.label == 1 and r.time_of_event is not None:
            m = t < r.time_of_event
            row = [any(o >= r.time_of_event - MAX_LEAD for o in alarm_onsets(t[m], sc[m], th, k)) for th in grid]
            ids.append(vid), is_pos.append(True), hours.append(0.0), fa.append(np.zeros(len(grid))), hit.append(np.array(row, float))
        elif r.label == 0:
            ids.append(vid), is_pos.append(False), hours.append(float(t[-1] - t[0]) / 3600 if len(t) > 1 else 0.0)
            fa.append(np.array([len(alarm_onsets(t, sc, th, k)) for th in grid], float)), hit.append(np.zeros(len(grid)))
    return {"ids": ids, "is_pos": np.array(is_pos), "hours": np.array(hours), "fa": np.stack(fa), "hit": np.stack(hit), "grid": grid}


def score_from_table(tab: dict, weights: np.ndarray | None = None, budgets=(5, 10, 20)) -> float:
    """Streaming score S (mean recall at the budgets) for a weighted set of videos (weights = bootstrap counts)."""
    w = np.ones(len(tab["ids"])) if weights is None else weights
    pos, neg = tab["is_pos"], ~tab["is_pos"]
    hours = float((w[neg] * tab["hours"][neg]).sum())
    fah = (w[neg, None] * tab["fa"][neg]).sum(0) / max(hours, 1e-9)
    rec = (w[pos, None] * tab["hit"][pos]).sum(0) / max(float(w[pos].sum()), 1e-9)
    out = []
    for b in budgets:
        ok = np.where(fah <= b)[0]
        out.append(rec[ok[0]] if len(ok) else 0.0)  # lowest admissible threshold (grid is increasing)
    return float(np.mean(out))


def streaming_score(op: dict, budgets=(5, 10, 20)) -> float:
    """Pre-declared primary selection metric: mean recall at 5, 10 and 20 false alarms per hour."""
    return float(np.mean([op[f"@{b}FA"]["recall"] for b in budgets]))
