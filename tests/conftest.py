from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.motion_features import MOTION_FEATURES  # noqa: E402
from src.preprocessing.feature_cache import VideoFeatures  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from tests.fixtures import make_blank_video, make_zoom_video  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    c = load_config()
    c["model"].update(epochs=3, ensemble_size=2, mc_samples=3, patience=2, batch_size=32)
    c["windows"].update(train_samples_per_video=4)
    return c


@pytest.fixture(scope="session")
def zoom_video(tmp_path_factory):
    return make_zoom_video(tmp_path_factory.mktemp("vid") / "zoom.mp4")


@pytest.fixture(scope="session")
def blank_video(tmp_path_factory):
    return make_blank_video(tmp_path_factory.mktemp("vid") / "blank.mp4")


def toy_features(vid: str, label: int, T: int = 120, seed: int = 0) -> VideoFeatures:
    """Random features for software tests only. Positives get a rising motion cue."""
    rng = np.random.default_rng(seed)
    m = rng.normal(0, 1, (T, len(MOTION_FEATURES))).astype(np.float32)
    if label:
        m[:, MOTION_FEATURES.index("looming_center")] += np.linspace(0, 4, T)
    return VideoFeatures(vid, np.arange(T, dtype=np.float32) / 10.0, rng.normal(0, 1, (T, 512)).astype(np.float16), m,
                         np.zeros((0, 8), np.float32), {"target_fps": 10.0},
                         np.abs(rng.normal(0, 0.05, (T, 20))).astype(np.float32))


@pytest.fixture()
def toy_dataset():
    from src.data.dataset_loader import VideoRecord
    recs, feats = [], {}
    for i in range(16):
        lab = i % 2
        vid = f"{i:05d}"
        recs.append(VideoRecord(vid, None, lab, 10.0 if lab else None, 8.5 if lab else None))
        feats[vid] = toy_features(vid, lab, seed=i)
    return recs, feats


@pytest.fixture(autouse=True)
def _campaign_logs_to_tmp(tmp_path, monkeypatch):
    """v3.4.4: no test may append to the real campaign logs in data/campaign or data/annotator_training."""
    from src.realworld import privacy, readiness, training
    monkeypatch.setattr(privacy, "REVIEW_LOG", tmp_path / "_privacy_review_log.jsonl")
    monkeypatch.setattr(training, "SUBMISSIONS", tmp_path / "_training_submissions.jsonl")
    monkeypatch.setattr(readiness, "ENCRYPTION_PATH", tmp_path / "_encryption_status.jsonl")
