"""Windows, models, training, uncertainty, calibration, risk engine, metrics, explainability, bundle, pipeline."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation.early_warning import alarm_onsets, evaluate_early_warning
from src.evaluation.metrics import best_f1_threshold, classification_metrics, nexar_map, selective_risk
from src.explainability.attribution import group_contributions, integrated_gradients, permutation_importance
from src.features.motion_features import MOTION_FEATURES
from src.prediction.models import CollisionModel
from src.prediction.predictor import EnsemblePredictor, ModelNotFoundError, save_bundle
from src.prediction.trainer import train_model
from src.prediction.windows import Standardizer, eval_samples, get_window, materialize, train_samples, window_indices
from src.risk.risk_engine import LEVELS, RiskConfig, RiskEngine, cue_reference
from src.uncertainty.calibration import brier, ece, fit_temperature, reliability
from src.uncertainty.mc_dropout import sample_logits, summarize


def test_window_is_causal():
    times = np.arange(100) / 10.0
    idx = window_indices(times, 5.0, 30)
    assert times[idx].max() <= 5.0 + 1e-6 and len(idx) == 30
    idx0 = window_indices(times, 0.5, 30)  # padded at the start
    assert idx0.min() == 0 and len(idx0) == 30


def test_samples_never_after_event(toy_dataset, cfg):
    recs, feats = toy_dataset
    rng = np.random.default_rng(0)
    for s in train_samples(recs, feats, cfg, rng) + eval_samples(recs, feats, cfg):
        r = next(x for x in recs if x.video_id == s.video_id)
        if r.label == 1:
            assert s.t_end < r.time_of_event
            v, m = get_window(feats[s.video_id], s.t_end, 3.0)
            assert v.shape == (30, 512)


@pytest.mark.parametrize("kind", ["frame", "pool", "gru", "lstm", "tcn"])
def test_model_shapes(kind):
    m = CollisionModel(kind, d_motion=len(MOTION_FEATURES))
    y = m(torch.randn(4, 30, 512), torch.randn(4, 30, len(MOTION_FEATURES)))
    assert y.shape == (4,)


def test_mc_dropout_and_ensemble():
    ms = [CollisionModel("gru", d_motion=len(MOTION_FEATURES)) for _ in range(2)]
    V = np.random.randn(5, 30, 512).astype(np.float32)
    M = np.random.randn(5, 30, len(MOTION_FEATURES)).astype(np.float32)
    lg = sample_logits(ms, V, M, "cpu", mc_samples=4)
    assert lg.shape == (8, 5)
    s = summarize(lg)
    assert (s["mutual_info"] >= 0).all() and (s["entropy"] >= s["mutual_info"] - 1e-9).all()
    assert (s["p_std"] > 0).any()  # dropout active -> stochastic
    det = sample_logits(ms[:1], V, M, "cpu", mc_samples=1)
    assert np.allclose(det, sample_logits(ms[:1], V, M, "cpu", mc_samples=1))


def test_calibration_metrics():
    y = np.array([0, 0, 1, 1], float)
    assert brier(np.array([0, 0, 1, 1.0]), y) == 0
    assert ece(np.array([0.0, 0.0, 1.0, 1.0]), y) == 0
    assert ece(np.full(4, 0.9), y) == pytest.approx(0.4)
    r = reliability(np.array([0.05, 0.95]), np.array([0.0, 1.0]))
    assert sum(r["count"]) == 2
    # over-confident logits -> temperature > 1
    rng = np.random.default_rng(0)
    yy = rng.integers(0, 2, 400).astype(float)
    logits = (yy * 2 - 1) * 6 + rng.normal(0, 6, 400)
    assert fit_temperature(logits[None], yy) > 1.0


def test_metrics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    m = classification_metrics(y, p, 0.5)
    assert m["precision"] == 1.0 and m["recall"] == 0.5 and m["roc_auc"] == 0.75
    g = np.array(["neg", "neg", "h0.5", "h1.0"])
    nm = nexar_map(y, p, g, [0.5, 1.0])
    assert set(nm) == {"ap_h0.5", "ap_h1.0", "mAP"}
    assert 0 <= best_f1_threshold(y, p) <= 1
    rows = selective_risk(y, p, np.array([0.1, 0.9, 0.5, 0.2]), 0.5)
    assert rows[0]["coverage"] == 1.0


def test_early_warning():
    t = np.arange(50) / 10.0
    s = np.zeros(50)
    s[20:30] = 0.9
    assert alarm_onsets(t, s, 0.5, k=3) == [pytest.approx(2.2)]
    s2 = s.copy()
    s2[40:45] = 0.9
    assert len(alarm_onsets(t, s2, 0.5, k=3)) == 2

    class R:
        def __init__(self, lab, te):
            self.label, self.time_of_event, self.time_of_alert = lab, te, (te - 1 if te else None)

    res = evaluate_early_warning({"a": {"times": t, "score": s}, "b": {"times": t, "score": s2}},
                                 {"a": R(1, 4.0), "b": R(0, None)}, 0.5, 3)
    assert res["detection_rate"] == 1.0 and res["mean_tta_s"] == pytest.approx(1.8)
    assert res["false_alarms"] == 2


def test_risk_engine_levels():
    motion = np.random.default_rng(0).normal(0, 1, (500, len(MOTION_FEATURES)))
    rc = RiskConfig(caution=0.3, warning=0.5, high=0.8, mi_threshold=0.05, cue_reference=cue_reference(motion))
    eng = RiskEngine(rc)
    assert eng.level_for(0.1, 0.0) == "NORMAL"
    assert eng.level_for(0.4, 0.0) == "CAUTION"
    assert eng.level_for(0.6, 0.0) == "WARNING"
    assert eng.level_for(0.9, 0.5) == "HIGH RISK"
    assert eng.level_for(0.6, 0.2) == "UNCERTAIN"
    times = np.arange(20) / 5.0
    p = np.linspace(0.1, 0.95, 20)
    pred = {"p_mean": p, "mutual_info": np.zeros(20), "entropy": np.zeros(20)}
    rows = motion[:20].copy()
    rows[-1, MOTION_FEATURES.index("looming_center")] = 10.0
    ass = eng.assess(times, pred, rows)
    assert all(a.level in LEVELS for a in ass)
    assert ass[-1].level == "HIGH RISK" and ass[-1].risk_score >= ass[-1].prediction_score
    assert any(f["feature"] == "looming_center" for f in ass[-1].triggering_factors)
    assert "probability" in ass[-1].explanation


def test_train_bundle_pipeline(toy_dataset, cfg, tmp_path, zoom_video):
    """Integration: train -> MC/ensemble -> bundle save/load -> attribution -> pipeline on a real clip."""
    recs, feats = toy_dataset
    scaler = Standardizer().fit(list(feats.values()))
    ev = eval_samples(recs, feats, cfg)
    V, M, Y = materialize(ev, feats, 3.0, scaler)
    hp = dict(kind="gru", use_visual=True, use_motion=True, d_visual=512, d_motion=len(MOTION_FEATURES),
              visual_proj=16, motion_proj=8, hidden=16, dropout=0.2)
    members = [train_model(recs, (V, M, Y), feats, scaler, cfg, hp, s, 3.0)[0] for s in (0, 1)]
    T = fit_temperature(sample_logits(members, V, M, "cpu", 3), Y)
    meta = {"window_s": 3.0, "temperature": T, "mc_samples": 3, "threshold_f1_val": 0.5,
            "risk": RiskConfig(cue_reference=cue_reference(np.concatenate([f.motion for f in feats.values()]))).to_dict()}
    save_bundle(tmp_path / "bundle", members, scaler, meta)
    pred = EnsemblePredictor.load(tmp_path / "bundle")
    tl = pred.predict_video(feats["00001"], 2)
    assert len(tl["times"]) > 0 and (tl["p_mean"] >= 0).all() and (tl["p_mean"] <= 1).all()
    ig = integrated_gradients(pred.models, V[0], M[0], steps=8)
    assert ig["visual"].shape == V[0].shape and abs(sum(group_contributions(ig).values()) - 1) < 1e-6
    imp = permutation_importance(lambda a, b: summarize(sample_logits(pred.models, a, b, "cpu", 1))["p_mean"], V, M, Y, 1)
    assert len(imp) == 1 + len(MOTION_FEATURES)

    from src.explainability.gradcam import GradCAM
    from src.pipeline import SafeDrivePipeline
    from src.reporting.exports import result_json, result_pdf, save_all, timeline_csv
    pipe = SafeDrivePipeline(tmp_path / "bundle", cfg=cfg, device="cpu")
    res = pipe.analyze(zoom_video)
    assert len(res.timeline) > 0 and res.max_level in LEVELS
    assert set(res.timeline.columns) >= {"time_s", "p_mean", "mutual_info", "risk_score", "level"}
    out = pipe.render(res, tmp_path / "annot.mp4")
    assert out.exists() and out.stat().st_size > 1000
    assert timeline_csv(res).startswith(b"time_s")
    assert b"summary" in result_json(res)
    assert result_pdf(res, meta)[:4] == b"%PDF"
    paths = save_all(res, tmp_path / "exp", meta)
    assert all(p.exists() for p in paths.values())
    cam = GradCAM(pipe.predictor, "cpu")
    v, m = pipe.predictor.windows_for(res.features, np.array([res.features.times[-1]]))
    heat = cam(res.frames[-1], v[0], m[0])
    assert heat.shape == res.frames[-1].shape[:2] and heat.max() <= 1.0 + 1e-6


def test_missing_checkpoint(tmp_path):
    with pytest.raises(ModelNotFoundError):
        EnsemblePredictor.load(tmp_path / "none")


def test_cpu_fallback(monkeypatch):
    from src.utils.device import get_device
    monkeypatch.setenv("SAFEDRIVE_DEVICE", "cpu")
    assert get_device() == "cpu"
