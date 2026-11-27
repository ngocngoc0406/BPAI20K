"""Active annotation: which videos to annotate first. Scores never use labels; the model never writes labels.

Signals (per video, label-free):
  uncertainty    mean of the 10 highest ensemble entropies along the video (frozen v3 or the current research model)
  disagreement   mean of the 10 highest standard deviations between ensemble members
  rarity         inverse frequency of the video's (light, weather, scene) metadata combination + rare rule patterns
                 (cut-in R01, pedestrian R04, lead deceleration R05) from the precomputed KG channels
  annotation     number of open inter-annotator disagreements (once two annotators worked on it)
hybrid = mean of the percentile ranks of uncertainty and rarity.
The annotator's queue shows only the ORDER; the reasons (which reveal model uncertainty) are shown to the coordinator only, so the
annotators stay blind to model output.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

STRATEGIES = ("random", "uncertainty", "disagreement", "rarity", "hybrid")


def _rank(x: np.ndarray) -> np.ndarray:
    o = np.argsort(np.argsort(x))
    return o / max(len(x) - 1, 1)


def rarity_scores(vids: list[str], recs: dict, kg_rule_rate: dict | None = None) -> np.ndarray:
    keys = [(recs[v].meta.get("light_conditions"), recs[v].meta.get("weather"), recs[v].meta.get("scene")) for v in vids]
    cnt = Counter(keys)
    r = np.array([1.0 / cnt[k] for k in keys])
    if kg_rule_rate:
        rr = np.array([kg_rule_rate.get(v, 0.0) for v in vids])
        r = 0.5 * _rank(r) + 0.5 * _rank(rr)
    return r


def model_scores(members: list, matrix_fn, vids: list[str], times_of: dict, stride: int = 5) -> tuple[np.ndarray, np.ndarray]:
    from src.prediction.light_models import summarize_probs
    unc, dis = [], []
    for v in vids:
        t = times_of[v][::stride]
        X = matrix_fn(v, t)
        ps = np.stack([m.predict_proba(X)[:, 1] for m in members])
        s = summarize_probs(ps)
        unc.append(float(np.mean(np.sort(s["entropy"])[-10:])))
        dis.append(float(np.mean(np.sort(s["p_std"])[-10:])))
    return np.array(unc), np.array(dis)


def select(strategy: str, pool: list[str], k: int, rng: np.random.Generator, unc=None, dis=None, rar=None) -> list[str]:
    if strategy == "random":
        return list(rng.choice(pool, size=min(k, len(pool)), replace=False))
    score = {"uncertainty": unc, "disagreement": dis, "rarity": rar,
             "hybrid": None if unc is None or rar is None else 0.5 * _rank(unc) + 0.5 * _rank(rar)}[strategy]
    order = np.argsort(-(score + 1e-9 * rng.random(len(score))))   # tiny random tie-break
    return [pool[i] for i in order[:k]]


def queue(store, vids: list[str], recs: dict, unc: np.ndarray | None = None, kg_rule_rate: dict | None = None) -> list[dict]:
    """Product queue (no labels): hybrid order + reasons for the coordinator; already double-submitted videos go last."""
    rar = rarity_scores(vids, recs, kg_rule_rate)
    sc = 0.5 * _rank(unc) + 0.5 * _rank(rar) if unc is not None else _rank(rar)
    out = []
    for i, v in enumerate(vids):
        subs = [a for a in store.annotators(v) if store.submitted(v, a)] if store else []
        dis = 0
        if len(subs) >= 2:
            from safedrive_v4.annotation.agreement import video_agreement
            dis = len(video_agreement(store, v, subs[0], subs[1])["disagreements"])
        done = len(subs) >= 2 and dis == 0
        out.append({"video_id": v, "priority": float(sc[i]) + (0.5 if dis else 0.0) - (1.0 if done else 0.0),
                    "reasons_coordinator_only": {"model_uncertainty_rank": None if unc is None else float(_rank(unc)[i]),
                                                 "rarity_rank": float(_rank(rar)[i]), "open_disagreements": dis, "double_submitted": len(subs) >= 2}})
    return sorted(out, key=lambda r: -r["priority"])
