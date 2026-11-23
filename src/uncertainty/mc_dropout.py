"""Monte-Carlo Dropout and Deep-Ensemble predictive distributions.

For each input we collect S = n_members * mc_samples Bernoulli probabilities and report:
    p_mean          predictive mean (the model's probability)
    p_std           spread of the sampled probabilities
    entropy         predictive (total) entropy H[p_mean]
    aleatoric       expected entropy E[H[p_s]]
    mutual_info     epistemic uncertainty = entropy - aleatoric (BALD)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _enable_dropout(model: nn.Module) -> None:
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


@torch.no_grad()
def sample_logits(models: list[nn.Module], V: np.ndarray, M: np.ndarray, device: str, mc_samples: int = 20,
                  bs: int = 512, seed: int = 0) -> np.ndarray:
    """Return (S, N) logits. mc_samples=1 -> deterministic (dropout off) pass per member."""
    torch.manual_seed(seed)
    outs = []
    for model in models:
        if mc_samples > 1:
            _enable_dropout(model)
        else:
            model.eval()
        for _ in range(mc_samples):
            chunk = []
            for i in range(0, len(V), bs):
                chunk.append(model(torch.from_numpy(V[i:i + bs]).to(device),
                                   torch.from_numpy(M[i:i + bs]).to(device)).cpu())
            outs.append(torch.cat(chunk).numpy())
        model.eval()
    return np.stack(outs)


def summarize(logits: np.ndarray, temperature: float = 1.0) -> dict[str, np.ndarray]:
    ps = 1.0 / (1.0 + np.exp(-logits / temperature))
    p = ps.mean(0)
    ent = binary_entropy(p)
    ale = binary_entropy(ps).mean(0)
    return {"p_mean": p, "p_std": ps.std(0), "entropy": ent, "aleatoric": ale,
            "mutual_info": np.maximum(ent - ale, 0.0)}
