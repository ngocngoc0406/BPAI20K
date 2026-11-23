"""v3: ego-motion, relational / interaction features, streaming metrics, shake corruption, alert-aligned labels."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.evaluation.streaming import MAX_LEAD, ema, evaluate_at, operating_points, streaming_score
from src.features.ego_motion import EGO_DIM, EGO_LAYOUT, EgoMotionEstimator, background_mask
from src.features.relational import REL_FEATURES, TTR_CAP, derive_relational
from src.preprocessing.video_transform import make_transform


def _textured(h=360, w=640, seed=0):
    rng = np.random.default_rng(seed)
    img = cv2.resize(rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(img, (3, 3), 0)


def test_ego_recovers_known_camera_translation():
    img = _textured()
    e = EgoMotionEstimator()
    M = np.float32([[1, 0, 12], [0, 1, 0]])  # camera pans: scene moves +12 px (640 width) per step
    e(img, np.zeros((0, 4)), 0.1)
    v = e(cv2.warpAffine(img, M, (640, 360), borderMode=cv2.BORDER_REFLECT), np.zeros((0, 4)), 0.1)
    assert v.shape == (EGO_DIM,) and v[EGO_LAYOUT["h_valid"]] == 1
    dx = v[EGO_LAYOUT["h_stats"]][0]
    assert dx == pytest.approx(12 / 640 / 0.1, rel=0.15)
    H = v[EGO_LAYOUT["H"]].reshape(3, 3)
    p = H @ np.array([0.5, 0.5, 1.0])
    assert p[0] / p[2] == pytest.approx(0.5 + 12 / 640, abs=0.005)


def test_background_mask_excludes_boxes_and_hood():
    m = background_mask((90, 160), np.array([[0.4, 0.4, 0.6, 0.6]]))
    assert m[45, 80] == 0 and m[89, 10] == 0 and m[10, 10] == 255


def _track_rows(T, boxes_fn, tid=1, cls=2):
    return np.array([[t, tid, cls, 0.9, *boxes_fn(t)] for t in range(T)], np.float32)


def test_relational_compensation_removes_camera_motion():
    """A static object seen by a camera that zooms in grows in the image: raw looming > 0, compensated ~ 0."""
    T, z = 20, 1.03
    ego = np.zeros((T, EGO_DIM), np.float32)
    S = np.array([[z, 0, 0.5 * (1 - z)], [0, z, 0.5 * (1 - z)], [0, 0, 1]])  # zoom about the image centre
    ego[:, EGO_LAYOUT["H"]] = S.ravel()
    ego[:, EGO_LAYOUT["F"]] = S.ravel()

    def box(t):
        s = 0.1 * z ** t
        return [0.5 - s / 2, 0.6 - s / 2 + (0.1 * (z ** t - 1)), 0.5 + s / 2, 0.6 + s / 2 + (0.1 * (z ** t - 1))]

    tr = _track_rows(T, box)
    times = np.arange(T) / 10.0
    raw = derive_relational(times, tr, None, 10, mode="none")
    comp = derive_relational(times, tr, ego, 10, mode="h")
    i = REL_FEATURES.index("res_loom_max")
    assert raw[10, i] > 0.2 and abs(comp[10, i]) < 0.05
    assert raw.shape == (T, len(REL_FEATURES)) and np.isfinite(raw).all()


def test_relational_cut_in_and_pairs():
    T = 15
    a = _track_rows(T, lambda t: [0.75 - 0.02 * t, 0.5, 0.85 - 0.02 * t, 0.65], tid=1)  # moves left toward the corridor
    b = _track_rows(T, lambda t: [0.45, 0.5, 0.55, 0.65], tid=2)
    r = derive_relational(np.arange(T) / 10.0, np.concatenate([a, b]), None, 10, mode="none")
    assert r[8, REL_FEATURES.index("cutin_score")] > 0
    assert r[8, REL_FEATURES.index("pair_ttr_min")] < TTR_CAP
    assert r[8, REL_FEATURES.index("pair_converging")] >= 1
    empty = derive_relational(np.arange(5) / 10.0, np.zeros((0, 8), np.float32), None)
    assert (empty[:, REL_FEATURES.index("pair_ttr_min")] == TTR_CAP).all()


def test_streaming_metrics():
    class R:
        def __init__(self, lab, te=None):
            self.label, self.time_of_event, self.time_of_alert = lab, te, te

    t = np.arange(0, 30, 0.2)
    pos = np.zeros_like(t)
    pos[(t > 16.5) & (t < 20)] = 0.9            # alarm ~3.5 s before event at 20 s
    far = np.zeros_like(t)
    far[(t > 5) & (t < 6)] = 0.9                 # alarm 15 s before the event: not a valid anticipation
    neg = np.zeros_like(t)
    neg[(t > 10) & (t < 11)] = 0.9
    streams = {"p": {"times": t, "score": pos}, "f": {"times": t, "score": far}, "n": {"times": t, "score": neg}}
    recs = {"p": R(1, 20.0), "f": R(1, 20.0), "n": R(0)}
    e = evaluate_at(streams, recs, 0.5, k=3, warmup=1.0)
    assert e["recall"] == 0.5 and e["edr@3s"] == 0.5 and e["false_alarms"] == 1
    op = operating_points(streams, recs, (5, 10, 20), 3, 1.0)
    assert 0 <= streaming_score(op) <= 1
    assert MAX_LEAD == 5.0
    assert np.allclose(ema(np.array([1.0, 0, 0]), 0.5), [1.0, 0.5, 0.25])


def test_shake_corruption_moves_content():
    img = _textured(seed=3)
    sh = make_transform("shake", 2, seed=1)
    outs = [sh(img) for _ in range(4)]
    assert all(o.shape == img.shape for o in outs)
    assert any(np.abs(o.astype(int) - img.astype(int)).mean() > 2 for o in outs)


def test_alert_aligned_labels(toy_dataset, cfg):
    import copy

    from src.prediction.windows import train_samples
    recs, feats = toy_dataset
    c = copy.deepcopy(cfg)
    c["windows"]["train_pos_mode"] = "alert"
    for s in train_samples(recs, feats, c, np.random.default_rng(0)):
        if s.label == 1:
            r = next(x for x in recs if x.video_id == s.video_id)
            assert r.time_of_alert - 1e-6 <= s.t_end < r.time_of_event


def test_tabular_explanations_with_ego_and_interactions(toy_dataset, cfg):
    """Regression: temporal occlusion must keep ego / derived interaction arrays (bug fixed in v3)."""
    from src.features.ego_motion import IDENTITY
    from src.features.window_features import build_matrix
    from src.prediction.light_models import BaggedEnsemble, Calibrator
    from src.prediction.tabular_predictor import TabularPredictor
    from src.prediction.windows import train_samples
    recs, feats = toy_dataset
    for f in feats.values():
        f.ego = np.zeros((len(f.times), EGO_DIM), np.float32)
        f.ego[:, EGO_LAYOUT["H"]] = IDENTITY
        f.ego[:, EGO_LAYOUT["F"]] = IDENTITY
        f.tracks = np.array([[t, 1, 2, 0.9, 0.4, 0.5, 0.5 + 0.001 * t, 0.6] for t in range(len(f.times))], np.float32)
    groups = ["motion", "diff", "ego_h", "rel_h", "rel_none"]
    tr = train_samples(recs, feats, cfg, np.random.default_rng(3))
    X, y = build_matrix(tr, feats, 1.0, groups)
    ens = BaggedEnsemble("lgbm", 2, seed=0).fit(X, y, np.array([s.video_id for s in tr]))
    pred = TabularPredictor({"members": ens.members, "calibrator": Calibrator("none"), "col_median": np.median(X, 0)},
                            {"window_s": 1.0, "groups": groups})
    assert pred.needs_ego
    f = feats["00001"]
    t, c = pred.temporal_contributions(f, 6.0)
    assert len(t) == len(c) == 10 and np.isfinite(c).all()
    assert len(pred.channel_contributions(f, 6.0)) == 28 + 20 + 6 + 18 + 18
