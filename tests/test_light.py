"""Lightweight (v2) path: frame difference, window features, tabular models, calibration, predictor, pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from src.features.frame_diff import FRAME_DIFF_FEATURES, FrameDiff
from src.features.window_features import AGGS, build_matrix, channel_names, video_window_matrix
from src.prediction.light_models import BaggedEnsemble, Calibrator, choose_calibration, make_model, summarize_probs
from src.prediction.windows import eval_samples, train_samples


def test_frame_diff():
    fd = FrameDiff()
    a = np.zeros((90, 160, 3), np.uint8)
    b = a.copy()
    b[:45, :80] = 255
    assert fd(a).sum() == 0                      # first frame -> zeros
    v = fd(b)
    assert v.shape == (len(FRAME_DIFF_FEATURES),) and v[0] > 0.9 and v[15] == 0  # top-left cell changed, bottom-right not


def test_window_features_shapes_and_causality(toy_dataset):
    recs, feats = toy_dataset
    f = feats["00001"]
    X = video_window_matrix(f, np.array([5.0, 8.0]), 2.0, ["motion", "diff"])
    assert X.shape == (2, len(AGGS) * (28 + 20)) == (2, len(channel_names(["motion", "diff"])))
    f2 = type(f)(f.video_id, f.times, f.visual, f.motion.copy(), f.tracks, f.meta, f.diff.copy())
    f2.motion[f.times > 5.0] += 100.0              # changing the future must not change the window ending at 5 s
    f2.diff[f.times > 5.0] += 1.0
    assert np.allclose(X[0], video_window_matrix(f2, np.array([5.0]), 2.0, ["motion", "diff"])[0])


@pytest.mark.parametrize("kind", ["logreg", "lgbm", "xgb", "catboost"])
def test_models_learn_toy_signal(kind, toy_dataset, cfg):
    recs, feats = toy_dataset
    tr = train_samples(recs, feats, cfg, np.random.default_rng(0))
    X, y = build_matrix(tr, feats, 2.0, ["motion"])
    p = make_model(kind, 0).fit(X, y).predict_proba(X)[:, 1]
    assert p.shape == y.shape and ((p >= 0) & (p <= 1)).all()


def test_bagging_calibration_summary(toy_dataset, cfg):
    recs, feats = toy_dataset
    tr = train_samples(recs, feats, cfg, np.random.default_rng(1))
    X, y = build_matrix(tr, feats, 2.0, ["motion", "diff"])
    ens = BaggedEnsemble("lgbm", 3, seed=0).fit(X, y, np.array([s.video_id for s in tr]))
    ps = ens.predict_samples(X)
    assert ps.shape == (3, len(y))
    s = summarize_probs(ps)
    assert (s["mutual_info"] >= 0).all() and (s["entropy"] >= s["mutual_info"] - 1e-9).all()
    method, scores = choose_calibration(ps.mean(0), y, np.array([s.video_id for s in tr]))
    assert method in scores
    for m in ("none", "platt", "isotonic"):
        q = Calibrator(m).fit(ps.mean(0), y)(ps)
        assert q.shape == ps.shape and ((q >= 0) & (q <= 1)).all()


def test_tabular_bundle_pipeline(tmp_path, toy_dataset, cfg, zoom_video):
    """Integration: train a tabular bundle -> TabularPredictor -> explanations -> SafeDrivePipeline (no CNN) on a real clip."""
    import pickle

    from src.pipeline import SafeDrivePipeline, load_predictor
    from src.risk.risk_engine import RiskConfig, cue_reference
    from src.utils.io import save_json
    recs, feats = toy_dataset
    tr = train_samples(recs, feats, cfg, np.random.default_rng(2))
    X, y = build_matrix(tr, feats, 2.0, ["motion", "diff"])
    ens = BaggedEnsemble("logreg", 2, seed=0).fit(X, y, np.array([s.video_id for s in tr]))
    b = tmp_path / "v2"
    b.mkdir()
    with open(b / "model.pkl", "wb") as fh:
        pickle.dump({"members": ens.members, "calibrator": Calibrator("none"), "pca": None, "col_median": np.median(X, 0)}, fh)
    save_json({"type": "tabular", "window_s": 2.0, "warmup_s": 2.0, "groups": ["motion", "diff"], "n_members": 2, "uses_cnn": False,
               "threshold_f1_val": 0.5, "risk": RiskConfig(cue_reference=cue_reference(np.concatenate([f.motion for f in feats.values()]))).to_dict()},
              b / "bundle.json")
    pred = load_predictor(b)
    assert pred.kind == "tabular" and not pred.uses_cnn
    tl = pred.predict_video(feats["00001"], 2)
    assert {"p_mean", "mutual_info", "times", "step"} <= set(tl)
    contrib = pred.channel_contributions(feats["00001"], 6.0)
    assert len(contrib) == 48
    t, c = pred.temporal_contributions(feats["00001"], 6.0)
    assert len(t) == len(c) == 20
    ev = eval_samples(recs, feats, cfg)
    assert pred.predict_at(feats["00001"], np.array([s.t_end for s in ev if s.video_id == "00001"]))["p_mean"].ndim == 1
    pipe = SafeDrivePipeline(b, cfg=cfg, device="cpu")
    assert pipe.extractor.encoder is None          # light mode: ResNet-18 is not loaded
    res = pipe.analyze(zoom_video)
    assert res.features.visual.shape[1] == 0 and res.features.diff.shape[1] == 20
    assert len(res.timeline) > 0 and res.model_dir == str(b)
    assert pipe.render(res, tmp_path / "a.mp4").exists()
