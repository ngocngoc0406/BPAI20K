"""Model bundle (ensemble members + scaler + calibration + thresholds) and streaming inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.prediction.models import CollisionModel, build_model
from src.prediction.windows import Standardizer, window_indices
from src.preprocessing.feature_cache import VideoFeatures
from src.uncertainty.mc_dropout import sample_logits, summarize
from src.utils.io import load_json, save_json


class ModelNotFoundError(FileNotFoundError):
    pass


def save_bundle(out_dir: str | Path, models: list[CollisionModel], scaler: Standardizer, meta: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(models):
        torch.save({"state_dict": m.state_dict(), "hparams": m.hparams}, out_dir / f"member_{i}.pt")
    np.savez(out_dir / "scaler.npz", **scaler.state())
    save_json({**meta, "n_members": len(models)}, out_dir / "bundle.json")


class EnsemblePredictor:
    kind = "neural"
    uses_cnn = True
    def __init__(self, models: list[CollisionModel], scaler: Standardizer, meta: dict, device: str = "cpu"):
        self.models = [m.to(device).eval() for m in models]
        self.scaler, self.meta, self.device = scaler, meta, device
        self.window_s = float(meta["window_s"])
        self.temperature = float(meta.get("temperature", 1.0))
        self.mc_samples = int(meta.get("mc_samples", 20))

    @classmethod
    def load(cls, model_dir: str | Path, device: str = "cpu") -> "EnsemblePredictor":
        model_dir = Path(model_dir)
        bj = model_dir / "bundle.json"
        if not bj.exists():
            raise ModelNotFoundError(f"No trained model bundle at {model_dir}. Run scripts/run_experiments.py first.")
        meta = load_json(bj)
        models = []
        for i in range(meta["n_members"]):
            ck = torch.load(model_dir / f"member_{i}.pt", map_location=device, weights_only=True)
            m = build_model(ck["hparams"])
            m.load_state_dict(ck["state_dict"])
            models.append(m)
        with np.load(model_dir / "scaler.npz") as z:
            scaler = Standardizer.from_state({k: z[k] for k in z.files})
        return cls(models, scaler, meta, device)

    def windows_for(self, feat: VideoFeatures, t_ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = int(round(self.window_s * feat.fps))
        V, M = [], []
        for t in t_ends:
            idx = window_indices(feat.times, float(t), n)
            v, m = self.scaler.transform(feat.visual[idx].astype(np.float32), feat.motion[idx].astype(np.float32))
            V.append(v)
            M.append(m)
        return np.stack(V), np.stack(M)

    def predict(self, V: np.ndarray, M: np.ndarray, mc_samples: int | None = None) -> dict[str, np.ndarray]:
        logits = sample_logits(self.models, V, M, self.device, mc_samples or self.mc_samples)
        return summarize(logits, self.temperature)

    def predict_at(self, feat: VideoFeatures, t_ends: np.ndarray) -> dict[str, np.ndarray]:
        V, M = self.windows_for(feat, np.asarray(t_ends))
        return self.predict(V, M)

    def predict_video(self, feat: VideoFeatures, stride_steps: int = 2) -> dict[str, np.ndarray]:
        """Causal sliding-window timeline from t=0 (windows left-padded at the start): the prediction at time t
        only uses frames <= t, so looking up the last timeline point <= t_end can never see the future."""
        idx = np.arange(0, len(feat.times), stride_steps)
        t_ends = feat.times[idx]
        V, M = self.windows_for(feat, t_ends)
        out = self.predict(V, M)
        out["times"] = t_ends
        out["step"] = idx
        return out
