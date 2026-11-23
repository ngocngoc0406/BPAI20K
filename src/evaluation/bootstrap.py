"""Video-level bootstrap confidence intervals for window-level metrics (videos are the independent unit)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import nexar_map


def bootstrap_map(df: pd.DataFrame, cols: list[str], horizons: list[float], n: int = 1000, seed: int = 0,
                  pairs: list[tuple[str, str]] | None = None) -> dict:
    """df: test_predictions.csv (columns y, group, video_id, <model columns>). Returns point, 95 % CI per model and
    paired differences (a - b) with CI and the fraction of resamples where a > b."""
    rng = np.random.default_rng(seed)
    vids = df.video_id.astype(str).unique()
    idx_by_vid = {v: np.where(df.video_id.astype(str).values == v)[0] for v in vids}
    y, g = df.y.values, df.group.values
    point = {c: nexar_map(y, df[c].values, g, horizons)["mAP"] for c in cols}
    samples = {c: [] for c in cols}
    for _ in range(n):
        pick = rng.choice(vids, size=len(vids), replace=True)
        ii = np.concatenate([idx_by_vid[v] for v in pick])
        for c in cols:
            samples[c].append(nexar_map(y[ii], df[c].values[ii], g[ii], horizons)["mAP"])
    out = {"n_boot": n, "models": {c: {"mAP": point[c], "ci95": [float(np.nanpercentile(samples[c], 2.5)),
                                                                  float(np.nanpercentile(samples[c], 97.5))]} for c in cols}}
    out["paired"] = {}
    for a, b in pairs or []:
        d = np.array(samples[a]) - np.array(samples[b])
        out["paired"][f"{a} - {b}"] = {"diff": point[a] - point[b], "ci95": [float(np.nanpercentile(d, 2.5)),
                                                                            float(np.nanpercentile(d, 97.5))],
                                       "p_a_better": float(np.nanmean(d > 0))}
    return out
