"""v4 pipeline integration: streaming replay on a real cached clip with a TEMPORARY research bundle (synthetic LightGBM in tmp),
model compatibility, offline/streaming feature agreement, bounded memory, latency, trace integrity, and v3 isolation."""
from __future__ import annotations

import glob
import json
import pickle

import numpy as np
import pytest

from safedrive_v4 import FEATURE_SCHEMA
from safedrive_v4.config import load_config
from safedrive_v4.features.graph_features import KG_FEATURES, SG_FEATURES, channel_names, video_channels, window_matrix
from src.preprocessing.feature_cache import load_features
from src.utils.config import resolve

CACHES = sorted(glob.glob(str(resolve("data/processed/features/clean/*.npz"))))
pytestmark = pytest.mark.skipif(not CACHES, reason="no v3 feature caches")
GROUPS = ["motion", "diff", "rel_none", "sg", "kg"]


def _bundle(tmp_path, schema=FEATURE_SCHEMA):
    from lightgbm import LGBMClassifier

    from src.features.window_features import channel_names as v3n
    from src.prediction.light_models import Calibrator
    names = v3n(GROUPS[:3]) + channel_names(SG_FEATURES) + channel_names(KG_FEATURES)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, len(names)))
    y = (X[:, 0] > 0).astype(int)
    ms = [LGBMClassifier(n_estimators=20, num_leaves=5, random_state=s, verbose=-1).fit(X, y) for s in range(2)]
    cal = Calibrator("platt").fit(ms[0].predict_proba(X)[:, 1], y)
    d = tmp_path / "bundle"
    d.mkdir()
    with open(d / "model.pkl", "wb") as fh:
        pickle.dump({"members": ms, "calibrator": cal, "col_median": np.median(X, 0)}, fh)
    (d / "bundle.json").write_text(json.dumps({"version": "test", "feature_schema": schema, "groups": GROUPS, "window_s": 1.0,
                                               "feature_names": names, "thresholds_from_dev_oof": {"@20FA": 0.55, "@10FA": 0.6}}))
    return d


def test_model_compatibility_rejects_other_feature_schema(tmp_path):
    from safedrive_v4.pipeline import ResearchModel
    with pytest.raises(ValueError):
        ResearchModel(_bundle(tmp_path, "v4feat-0.9"))


def test_streaming_pipeline_matches_offline_features_latency_and_traces(tmp_path):
    from safedrive_v4.pipeline import ResearchModel, V4Pipeline
    cfg = load_config()
    f = load_features(CACHES[0])
    p = V4Pipeline(ResearchModel(_bundle(tmp_path)), cfg)
    out = p.process_video(f, explain=True, trace_dir=tmp_path / "traces")
    tl = out["timeline"]
    assert len(tl) == len(f.times) and {r["level"] for r in tl} <= {"LOW", "MODERATE", "HIGH", "INSUFFICIENT_EVIDENCE"}
    assert all(0 <= r["p"] <= 1 and 0 <= r["reliability"] <= 1 for r in tl)
    lat = out["latency"]
    assert lat["total"]["mean_ms"] > 0 and lat["graph"]["mean_ms"] < 50          # measured, bounded
    # streaming SG/KG window features == offline features (same causal code, bounded buffer)
    SG, KG, _ = video_channels(f, cfg)
    t = f.times[len(f.times) // 2]
    step = int(np.searchsorted(f.times, t + 1e-4) - 1)
    p.reset()
    for s, tt, d, M, sc in __import__("safedrive_v4.scene_graph.builder", fromlist=["video_steps"]).video_steps(f):
        snap = p.builder.update(s, tt, d, M, sc)
        res = p.engine.evaluate(snap)
        from safedrive_v4.features.graph_features import kg_vector, sg_vector
        p.buf.append((sg_vector(snap), kg_vector(res.base if hasattr(res, "base") else res), None, None, None))
        if s == step:
            break
    x_stream = p._v4_window()
    x_off = np.concatenate([window_matrix(f.times, SG, np.array([t]), 1.0, f.fps), window_matrix(f.times, KG, np.array([t]), 1.0, f.fps)], 1)
    assert np.allclose(x_stream, x_off, atol=1e-6)
    assert len(p.buf) <= p.buffer_steps                                            # bounded memory
    for tr in out["traces"]:
        v = tr.verify()
        assert v["ok"], v
        assert tr.unknowns and tr.prediction["risk_probability"] == pytest.approx(next(r["p"] for r in tl if r.get("event_id") == tr.event_id))


def test_v4_does_not_touch_v3_artefacts():
    """v4 lives outside the hashed v3 software directories and never writes v3 models / frozen hashes."""
    from src.realworld import provenance
    import safedrive_v4
    root = resolve(".")
    assert not str(resolve(safedrive_v4.__file__)).startswith(str(root / "src"))
    assert "safedrive_v4" not in provenance.SOFTWARE_DIRS
    chk = provenance.check(provenance.compute("models/safedrive_v3"))
    assert chk["eligible"], chk


def test_a3_v3_readonly_predictions_equal_v3_and_files_untouched():
    """A3: the frozen v3 model predicts through the v4 pipeline with exactly v3's output; no v3 file changes."""
    from safedrive_v4.pipeline import V3ReadOnlyModel, V4Pipeline
    from src.prediction.tabular_predictor import TabularPredictor
    from src.realworld.provenance import sha256_file
    before = {p: sha256_file(resolve(p)) for p in ("models/safedrive_v3/model.pkl", "models/safedrive_v3/bundle.json",
                                                    "configs/frozen_v3_hashes.json")}
    f = load_features(CACHES[0])
    p = V4Pipeline(V3ReadOnlyModel(), load_config())
    out = p.process_video(f, explain=True)
    v3 = TabularPredictor.load(resolve("models/safedrive_v3"))
    idx = np.arange(10, len(f.times), 25)
    ref = v3.predict_matrix(v3.features(f, f.times[idx]))["p_mean"]
    got = np.array([out["timeline"][i]["p"] for i in idx])
    assert np.allclose(got, ref, atol=1e-9)
    assert all(tr.verify()["ok"] for tr in out["traces"])
    after = {p: sha256_file(resolve(p)) for p in before}
    assert before == after
