"""Streaming predictor for the lightweight (tabular) SafeDrive v2 bundle. Same interface as EnsemblePredictor
(predict_video -> times, step, p_mean, p_std, entropy, aleatoric, mutual_info), plus model-agnostic explanations."""
from __future__ import annotations

import dataclasses
import pickle
from pathlib import Path

import numpy as np

from src.features.window_features import FEATURE_VERSION, channel_names, video_window_matrix
from src.prediction.light_models import summarize_probs
from src.prediction.predictor import ModelNotFoundError
from src.prediction.windows import window_indices
from src.preprocessing.feature_cache import VideoFeatures
from src.utils.io import load_json


class TabularPredictor:
    kind = "tabular"

    def __init__(self, payload: dict, meta: dict):
        self.members = payload["members"]
        self.calibrator = payload["calibrator"]
        self.pca = payload.get("pca")
        self.col_median = payload.get("col_median")
        self.meta = meta
        self.window_s = float(meta["window_s"])
        self.groups = list(meta["groups"])
        self.names = channel_names(self.groups)
        self.device = "cpu"
        self.models = self.members  # attribute name used by generic code

    @classmethod
    def load(cls, model_dir: str | Path, device: str = "cpu") -> "TabularPredictor":
        model_dir = Path(model_dir)
        if not (model_dir / "bundle.json").exists() or not (model_dir / "model.pkl").exists():
            raise ModelNotFoundError(f"No lightweight model bundle at {model_dir}. Run scripts/run_light_study.py first.")
        meta = load_json(model_dir / "bundle.json")
        if int(meta.get("feature_version", 1)) != FEATURE_VERSION:
            raise ValueError(f"{model_dir}: trained with feature version {meta.get('feature_version', 1)}, code has {FEATURE_VERSION}. "
                             "Re-train the bundle or check out the matching code version.")
        with open(model_dir / "model.pkl", "rb") as fh:  # locally produced artefact (trusted)
            payload = pickle.load(fh)
        return cls(payload, meta)

    @property
    def needs_ego(self) -> bool:
        return any(g in ("ego_h", "ego_f", "rel_h", "rel_f") for g in self.groups)

    @property
    def uses_cnn(self) -> bool:
        return bool(self.meta.get("uses_cnn", False))

    def features(self, feat: VideoFeatures, t_ends: np.ndarray) -> np.ndarray:
        return video_window_matrix(feat, np.asarray(t_ends, float), self.window_s, self.groups, self.pca)

    def predict_matrix(self, X: np.ndarray) -> dict[str, np.ndarray]:
        ps = np.stack([m.predict_proba(X)[:, 1] for m in self.members])
        return summarize_probs(self.calibrator(ps))

    def predict_at(self, feat: VideoFeatures, t_ends: np.ndarray) -> dict[str, np.ndarray]:
        return self.predict_matrix(self.features(feat, t_ends))

    def predict_video(self, feat: VideoFeatures, stride_steps: int = 2) -> dict[str, np.ndarray]:
        idx = np.arange(0, len(feat.times), stride_steps)
        out = self.predict_matrix(self.features(feat, feat.times[idx]))
        out["times"], out["step"] = feat.times[idx], idx
        return out

    # ---------------- explanations (model-agnostic, exact for this model) ----------------
    def _p(self, X: np.ndarray) -> np.ndarray:
        return self.predict_matrix(X)["p_mean"]

    def channel_contributions(self, feat: VideoFeatures, t_end: float) -> dict[str, float]:
        """Occlusion: p(x) - p(x with all aggregates of one channel set to the training median). Positive = pushes toward collision."""
        x = self.features(feat, np.array([t_end]))
        p0 = float(self._p(x)[0])
        base = [n.split("__")[0] for n in self.names]
        chans = list(dict.fromkeys(base))
        Xo = np.repeat(x, len(chans), 0)
        for j, ch in enumerate(chans):
            cols = [i for i, b in enumerate(base) if b == ch]
            Xo[j, cols] = self.col_median[cols]
        return dict(zip(chans, (p0 - self._p(Xo)).astype(float).tolist()))

    def temporal_contributions(self, feat: VideoFeatures, t_end: float) -> tuple[np.ndarray, np.ndarray]:
        """Occlusion over time: for each step in the window, hold the previous step's values (as if that frame carried no
        new information) and measure the drop in p. Returns (times, contribution)."""
        n = int(round(self.window_s * feat.fps))
        idx = window_indices(feat.times, t_end, n)
        p0 = float(self._p(self.features(feat, np.array([t_end])))[0])
        contrib = np.zeros(n)
        for j in range(1, n):
            a, b = idx[j], idx[j - 1]
            # copy every per-step array (incl. ego and memoised interaction cues) and hold step a at step b's values
            f2 = dataclasses.replace(feat, motion=feat.motion.copy(), diff=None if feat.diff is None else feat.diff.copy(),
                                     ego=None if feat.ego is None else feat.ego.copy(),
                                     derived={k: v.copy() for k, v in feat.derived.items()})
            f2.motion[a] = f2.motion[b]
            if f2.diff is not None:
                f2.diff[a] = 0.0  # a repeated frame has zero frame difference
            if f2.ego is not None:
                f2.ego[a] = f2.ego[b]
            for arr in f2.derived.values():
                arr[a] = arr[b]
            contrib[j] = p0 - float(self._p(self.features(f2, np.array([t_end])))[0])
        return feat.times[idx], contrib
