"""Training with per-epoch window re-sampling and early stopping on validation average precision."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from src.data.dataset_loader import VideoRecord
from src.prediction.models import CollisionModel, build_model
from src.prediction.windows import Standardizer, materialize, train_samples
from src.preprocessing.feature_cache import VideoFeatures
from src.utils.io import set_seed
from src.utils.logging_utils import get_logger

log = get_logger("trainer")


@torch.no_grad()
def predict_logits(model: nn.Module, V: np.ndarray, M: np.ndarray, device: str, bs: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(V), bs):
        out.append(model(torch.from_numpy(V[i:i + bs]).to(device), torch.from_numpy(M[i:i + bs]).to(device)).cpu())
    return torch.cat(out).numpy()


def train_model(train_recs: list[VideoRecord], val_data: tuple[np.ndarray, np.ndarray, np.ndarray],
                feats: dict[str, VideoFeatures], scaler: Standardizer, cfg: dict, hp: dict, seed: int,
                window_s: float, device: str = "cpu") -> tuple[CollisionModel, dict]:
    mc = cfg["model"]
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(hp).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=mc["lr"], weight_decay=mc["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    Vv, Mv, Yv = val_data
    best_ap, best_state, best_ep, bad = -1.0, None, 0, 0
    hist: list[dict] = []
    t0 = time.perf_counter()
    for ep in range(mc["epochs"]):
        samples = train_samples(train_recs, feats, cfg, rng)  # fresh random windows every epoch
        V, M, Y = materialize(samples, feats, window_s, scaler)
        perm = rng.permutation(len(Y))
        model.train()
        tot = 0.0
        for i in range(0, len(perm), mc["batch_size"]):
            b = perm[i:i + mc["batch_size"]]
            xv, xm = torch.from_numpy(V[b]).to(device), torch.from_numpy(M[b]).to(device)
            y = torch.from_numpy(Y[b]).to(device)
            loss = loss_fn(model(xv, xm), y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(b)
        val_logits = predict_logits(model, Vv, Mv, device)
        ap = float(average_precision_score(Yv, val_logits))
        vloss = float(loss_fn(torch.from_numpy(val_logits), torch.from_numpy(Yv)))
        hist.append({"epoch": ep, "train_loss": tot / len(Y), "val_loss": vloss, "val_ap": ap})
        if ap > best_ap + 1e-4:
            best_ap, best_state, best_ep, bad = ap, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= mc["patience"]:
                break
    model.load_state_dict(best_state)
    info = {"best_epoch": best_ep, "best_val_ap": best_ap, "history": hist, "train_seconds": time.perf_counter() - t0,
            "seed": seed, "hparams": hp, "window_s": window_s}
    log.info("trained %s seed=%d best_ep=%d val_AP=%.4f (%.1fs)", hp["kind"], seed, best_ep, best_ap,
             info["train_seconds"])
    return model, info
