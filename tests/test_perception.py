"""Video processing, detection, tracking, motion features, corruptions, feature cache."""
from __future__ import annotations

import numpy as np
import pytest

from src.features.extractor import FeatureExtractor, ProcessingCancelled
from src.features.motion_features import MOTION_FEATURES, MotionFeatureComputer
from src.preprocessing.feature_cache import load_features, save_features
from src.preprocessing.frame_sampler import sample_on_grid
from src.preprocessing.video_reader import VideoReadError, get_meta, iter_frames
from src.preprocessing.video_transform import CORRUPTIONS, make_transform
from src.tracking.tracker import track_statistics


@pytest.fixture(scope="module")
def extractor(cfg):
    return FeatureExtractor(cfg)


def test_video_meta_and_grid(zoom_video):
    m = get_meta(zoom_video)
    assert abs(m.fps - 30) < 1 and m.n_frames == 120
    steps = list(sample_on_grid(iter_frames(zoom_video), 10.0))
    assert len(steps) == 40
    assert np.allclose([s[1] for s in steps[:3]], [0.0, 0.1, 0.2])


def test_unreadable_video(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"garbage")
    with pytest.raises(VideoReadError):
        get_meta(bad)
    with pytest.raises(VideoReadError):
        get_meta(tmp_path / "missing.mp4")


def test_extract_real_detections(extractor, zoom_video):
    f, frames = extractor.extract(zoom_video, keep_frames=True)
    assert f.visual.shape == (40, 512) and f.motion.shape == (40, len(MOTION_FEATURES))
    assert len(frames) == 40
    assert len(f.tracks) > 0, "detector found nothing in the street photo"
    assert set(np.unique(f.tracks[:, 2]).astype(int)) <= {0, 1, 2, 3, 5, 7}
    assert (f.tracks[:, 4:] >= -0.05).all() and (f.tracks[:, 4:] <= 1.05).all()
    st = track_statistics(f.tracks)
    assert st["n_tracks"] >= 1
    # the zoom makes objects grow -> positive looming somewhere
    assert f.motion[:, MOTION_FEATURES.index("looming_all")].max() > 0


def test_no_objects(extractor, blank_video):
    f, _ = extractor.extract(blank_video)
    assert len(f.tracks) == 0
    assert np.isfinite(f.motion).all()
    assert f.motion[:, MOTION_FEATURES.index("n_objects")].sum() == 0


def test_cancel(extractor, zoom_video):
    with pytest.raises(ProcessingCancelled):
        extractor.extract(zoom_video, should_cancel=lambda: True)


def test_long_video_truncated(extractor, zoom_video):
    f, _ = extractor.extract(zoom_video, max_duration_s=1.0)
    assert f.times[-1] <= 1.0 + 1e-6


def test_cache_roundtrip(tmp_path, extractor, zoom_video):
    f, _ = extractor.extract(zoom_video, end_s=1.0)
    p = tmp_path / "x.npz"
    save_features(f, p)
    g = load_features(p)
    assert np.allclose(f.motion, g.motion) and g.video_id == f.video_id and len(g) == len(f)


def test_motion_looming_and_tau():
    mc = MotionFeatureComputer(fps=10, k=3)
    flow = np.zeros(4, np.float32)
    for step in range(8):
        s = 0.1 * (1.1 ** step)  # object growing 10 % per step in the corridor
        box = np.array([[0.5 - s / 2, 0.6 - s / 2, 0.5 + s / 2, 0.6 + s / 2]], np.float32)
        v = mc.update(step, box, np.array([1]), np.array([2]), np.array([0.9]), flow, 0.5)
    loom = v[MOTION_FEATURES.index("looming_center")]
    tau = v[MOTION_FEATURES.index("tau_center")]
    assert loom > 0.5 and 0 < tau < 10


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_corruptions(name):
    img = np.full((90, 160, 3), 128, np.uint8)
    img[30:60, 60:100] = 255
    t = make_transform(name, 2, seed=3)
    outs = [t(img) for _ in range(5)]
    assert all(o.shape == img.shape and o.dtype == np.uint8 for o in outs)
    if name != "frame_drop":
        assert not np.array_equal(outs[0], img)
