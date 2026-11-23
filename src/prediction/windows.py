"""Observation-window construction and labelling.

Task definition (anticipation, following the Nexar challenge test protocol):
    Given only the frames in an observation window (t_end - W, t_end], predict whether a
    collision / near-collision event occurs shortly after t_end.

    * Positive windows come from positive videos and end strictly BEFORE the annotated
      time_of_event: t_end = time_of_event - h, with horizon h > 0. No post-event frame is ever used.
    * Negative windows come from negative (normal-driving) videos.
    * Evaluation horizons h in {0.5, 1.0, 1.5} s mirror the Nexar test set.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.dataset_loader import VideoRecord
from src.preprocessing.feature_cache import VideoFeatures


@dataclass
class WindowSample:
    video_id: str
    t_end: float
    label: int
    horizon: float     # seconds before event (NaN for negatives)
    group: str = ""    # e.g. "h0.5" or "neg"


def window_indices(times: np.ndarray, t_end: float, n_steps: int) -> np.ndarray:
    """Indices of the n_steps grid points ending at the last time <= t_end (left-padded by repeating
    the first frame if the window starts before the video)."""
    last = int(np.searchsorted(times, t_end + 1e-4, side="right") - 1)
    last = max(last, 0)
    idx = np.arange(last - n_steps + 1, last + 1)
    return np.clip(idx, 0, len(times) - 1)


def get_window(feat: VideoFeatures, t_end: float, window_s: float) -> tuple[np.ndarray, np.ndarray]:
    n = int(round(window_s * feat.fps))
    idx = window_indices(feat.times, t_end, n)
    return feat.visual[idx].astype(np.float32), feat.motion[idx].astype(np.float32)


def train_samples(recs: list[VideoRecord], feats: dict[str, VideoFeatures], cfg: dict,
                  rng: np.random.Generator) -> list[WindowSample]:
    w = cfg["windows"]
    lo, hi = w["train_pos_horizon_s"]
    k = int(w["train_samples_per_video"])
    out: list[WindowSample] = []
    for r in recs:
        f = feats.get(r.video_id)
        if f is None:
            continue
        dur = float(f.times[-1])
        if r.label == 1 and r.time_of_event is not None:
            if w.get("train_pos_mode") == "alert" and r.time_of_alert is not None:
                # alert-aligned labels: positives end between the annotated earliest-alert time and the event
                ends = rng.uniform(r.time_of_alert, max(r.time_of_event - 0.1, r.time_of_alert + 1e-3), size=k)
            else:
                ends = r.time_of_event - rng.uniform(lo, hi, size=k)
            for t in ends:
                if t > 0.5:
                    out.append(WindowSample(r.video_id, float(t), 1, float(r.time_of_event - t), "pos"))
            # Hard negatives: windows that end >= margin s before the annotated earliest alert time.
            # Derived from the ground-truth timestamps (normal driving preceding the conflict), not invented labels.
            margin = w.get("hard_negative_margin_s")
            if margin is not None and r.time_of_alert is not None:
                t_hi = r.time_of_alert - float(margin)
                if t_hi > 2.0:
                    for t in rng.uniform(1.0, t_hi, size=max(k // 2, 1)):
                        out.append(WindowSample(r.video_id, float(t), 0, float("nan"), "hardneg"))
        elif r.label == 0:
            # dense negatives (streaming-matched): many windows per normal-driving video
            for t in rng.uniform(min(2.0, dur), dur, size=int(w.get("neg_samples_per_video") or k)):
                out.append(WindowSample(r.video_id, float(t), 0, float("nan"), "neg"))
    return out


def eval_samples(recs: list[VideoRecord], feats: dict[str, VideoFeatures], cfg: dict,
                 seed: int = 1234) -> list[WindowSample]:
    """Deterministic evaluation windows: positives at fixed horizons; negatives at one random
    anchor per video with the same 0/0.5/1.0 s offsets (so both classes share the clip geometry)."""
    hs = sorted(cfg["windows"]["eval_horizons_s"])
    rng = np.random.default_rng(seed)
    out: list[WindowSample] = []
    for r in sorted(recs, key=lambda x: x.video_id):
        f = feats.get(r.video_id)
        if f is None:
            continue
        dur = float(f.times[-1])
        if r.label == 1 and r.time_of_event is not None:
            for h in hs:
                t = r.time_of_event - h
                if t > 0.5:
                    out.append(WindowSample(r.video_id, float(t), 1, float(h), f"h{h}"))
        elif r.label == 0:
            anchor = float(rng.uniform(min(6.0, dur), dur))
            for h in hs:
                out.append(WindowSample(r.video_id, anchor - (h - hs[0]), 0, float("nan"), "neg"))
    return out


class Standardizer:
    """Signed-log + z-score for motion features; z-score for visual embeddings (fit on train only)."""

    def __init__(self):
        self.mv = self.sv = self.mm = self.sm = None

    @staticmethod
    def slog(x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.log1p(np.abs(x))

    def fit(self, feats: list[VideoFeatures]) -> "Standardizer":
        V = np.concatenate([f.visual.astype(np.float32) for f in feats])
        M = self.slog(np.concatenate([f.motion for f in feats]))
        self.mv, self.sv = V.mean(0), V.std(0) + 1e-5
        self.mm, self.sm = M.mean(0), M.std(0) + 1e-5
        return self

    def transform(self, v: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return ((v - self.mv) / self.sv).astype(np.float32), ((self.slog(m) - self.mm) / self.sm).astype(np.float32)

    def state(self) -> dict[str, np.ndarray]:
        return {"mv": self.mv, "sv": self.sv, "mm": self.mm, "sm": self.sm}

    @classmethod
    def from_state(cls, st: dict[str, np.ndarray]) -> "Standardizer":
        s = cls()
        s.mv, s.sv, s.mm, s.sm = (np.asarray(st[k], np.float32) for k in ("mv", "sv", "mm", "sm"))
        return s


def materialize(samples: list[WindowSample], feats: dict[str, VideoFeatures], window_s: float,
                scaler: Standardizer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    V, M, Y = [], [], []
    for s in samples:
        v, m = get_window(feats[s.video_id], s.t_end, window_s)
        v, m = scaler.transform(v, m)
        V.append(v)
        M.append(m)
        Y.append(s.label)
    return np.stack(V), np.stack(M), np.array(Y, np.float32)
