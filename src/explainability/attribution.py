"""Gradient-based and perturbation-based attributions for the temporal model.

* Integrated Gradients (Sundararajan et al., 2017) w.r.t. the standardised inputs, baseline = 0
  (i.e. the training-set mean). Gives per-timestep and per-feature attributions.
* Permutation importance by feature group (visual block, each motion feature) on a held-out set.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from src.features.motion_features import MOTION_FEATURES


def integrated_gradients(models: list[torch.nn.Module], v: np.ndarray, m: np.ndarray, device: str = "cpu",
                         steps: int = 32) -> dict[str, np.ndarray]:
    """v: (T, Dv), m: (T, Dm) standardised window. Returns attributions averaged over ensemble members."""
    xv = torch.from_numpy(v).to(device)[None]
    xm = torch.from_numpy(m).to(device)[None]
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=device)[:, None, None]
    av, am = torch.zeros_like(xv[0]), torch.zeros_like(xm[0])
    for model in models:
        model.eval()
        iv = (alphas * xv).requires_grad_(True)
        im = (alphas * xm).requires_grad_(True)
        with torch.backends.cudnn.flags(enabled=False):  # cuDNN RNN backward requires train mode
            out = model(iv, im).sum()
            gv, gm = torch.autograd.grad(out, [iv, im])
        av += (xv[0] * gv.mean(0)).detach()
        am += (xm[0] * gm.mean(0)).detach()
    av, am = (av / len(models)).cpu().numpy(), (am / len(models)).cpu().numpy()
    return {
        "visual": av, "motion": am,
        "temporal": np.abs(av).sum(1) + np.abs(am).sum(1),
        "visual_total": float(av.sum()),
        "motion_by_feature": dict(zip(MOTION_FEATURES, am.sum(0).astype(float).tolist())),
    }


def group_contributions(ig: dict) -> dict[str, float]:
    """Share of absolute attribution: visual appearance vs motion cues."""
    vis = float(np.abs(ig["visual"]).sum())
    mot = float(np.abs(ig["motion"]).sum())
    tot = max(vis + mot, 1e-12)
    return {"visual": vis / tot, "motion": mot / tot}


def permutation_importance(predict_fn, V: np.ndarray, M: np.ndarray, y: np.ndarray, n_repeats: int = 3,
                           seed: int = 0, use_visual: bool = True) -> dict[str, float]:
    """Drop in average precision when one feature group is shuffled across windows."""
    rng = np.random.default_rng(seed)
    base = average_precision_score(y, predict_fn(V, M))
    out: dict[str, float] = {}
    groups = ([("visual (CNN appearance)", None)] if use_visual else []) + [(n, i) for i, n in enumerate(MOTION_FEATURES)]
    for name, j in groups:
        drops = []
        for _ in range(n_repeats):
            perm = rng.permutation(len(y))
            if j is None:
                Vp, Mp = V[perm], M
            else:
                Vp, Mp = V, M.copy()
                Mp[:, :, j] = M[perm][:, :, j]
            drops.append(base - average_precision_score(y, predict_fn(Vp, Mp)))
        out[name] = float(np.mean(drops))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
