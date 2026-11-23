"""v4 explainability, counterfactual validity, uncertainty / warning policy, evidence-trace integrity and leakage checks."""
from __future__ import annotations

import glob

import numpy as np
import pytest

from safedrive_v4.config import load_config
from safedrive_v4.event_memory.trace import EvidenceTrace
from safedrive_v4.explainability import counterfactual as cf
from safedrive_v4.explainability import graph_explain, shap_tree
from safedrive_v4.features.graph_features import KG_FEATURES, SG_FEATURES, video_channels, window_matrix
from safedrive_v4.uncertainty.reliability import OODScorer, perception_reliability
from safedrive_v4.warning.policy import WarningPolicy
from src.preprocessing.feature_cache import load_features
from src.utils.config import resolve

CFG = load_config()
CACHES = sorted(glob.glob(str(resolve("data/processed/features/clean/*.npz"))))
needs_cache = pytest.mark.skipif(not CACHES, reason="no v3 feature caches")


def _lgbm(n=600, d=12, seed=0):
    from lightgbm import LGBMClassifier
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    ms = [LGBMClassifier(n_estimators=60, num_leaves=7, random_state=s, verbose=-1).fit(X, y) for s in range(3)]
    return ms, X, y


def test_tree_shap_local_accuracy_stability_and_fidelity():
    ms, X, _ = _lgbm()
    ex = shap_tree.explain(ms, X[:40])
    assert ex["local_accuracy_error"] < 1e-6                       # phi sums to the model logit
    assert shap_tree.member_rank_stability(ex["phi_members"], top=6) > 0.5
    fid = shap_tree.deletion_fidelity(ms, X[:40], ex["phi"], np.median(X, 0), k=2)
    assert fid["mean_logit_drop_top"] > fid["mean_logit_drop_random"]
    names = [f"f{i}" for i in range(X.shape[1])]
    g = shap_tree.global_importance(ex["phi"], names)
    assert g["top_features"][0]["feature"] in ("f0", "f1")


def test_warning_policy_separates_uncertainty_from_risk():
    p = WarningPolicy(0.5, 0.8, consecutive=3, reliability_min=0.45)
    lv = [p.decide(0.9, 0.0, 0.9).level for _ in range(3)]
    assert lv == ["LOW", "LOW", "HIGH"]                              # onset needs k consecutive steps
    p.reset()
    lv = [p.decide(0.1, 0.0, 0.2).level for _ in range(3)]
    assert lv[-1] == "INSUFFICIENT_EVIDENCE"                         # degraded input is not reported as risk
    p.reset()
    d = [p.decide(0.95, 0.0, 0.2) for _ in range(3)][-1]
    assert d.level == "HIGH" and d.perception_reliability == 0.2    # high risk is never hidden by low reliability
    p.reset()
    assert [p.decide(0.6, 0.0, 0.9).level for _ in range(3)][-1] == "MODERATE"


def test_perception_reliability_and_ood():
    good = perception_reliability(1.0, 0.8, 5, 0.1, 0)["score"]
    bad = perception_reliability(0.4, 0.3, 5, 0.8, 4)["score"]
    assert 0 <= bad < good <= 1
    rng = np.random.default_rng(0)
    o = OODScorer().fit(rng.normal(size=(2000, 10)))
    assert o.percentile(np.zeros((1, 10)))[0] < 0.5 and o.percentile(np.full((1, 10), 8.0))[0] >= 0.99


def test_evidence_trace_integrity_detects_tampering_and_dangling_refs():
    t = EvidenceTrace("e1", "v", 1.0, 10, [9, 10], [3], {"obj:3": {}, "ego": {}},
                      [{"edge_id": "E1", "source_node": "obj:3", "target_node": "ego"}], [{"edge_ids": ["E1"]}], [["obj:3", "APPROACHING", "ego"]],
                      {"risk_probability": 0.7}, [], {}, {"level": "HIGH"}, [], {}).seal()
    assert t.verify()["ok"]
    t.prediction["risk_probability"] = 0.1
    assert not t.verify()["hash_ok"]
    t2 = EvidenceTrace("e2", "v", 1.0, 10, [10], [3], {"ego": {}}, [{"edge_id": "E1", "source_node": "obj:3", "target_node": "ego"}],
                       [{"edge_ids": ["E9"]}], [], {}, [], {}, {}, [], {}).seal()
    v = t2.verify()
    assert not v["ok"] and v["missing_edges"] == ["E9"] and v["missing_nodes"] == ["obj:3"]


@needs_cache
def test_no_future_leakage_in_v4_window_features():
    """Changing everything AFTER t_end must not change the window features at t_end (causal graph + window)."""
    f = load_features(CACHES[0])
    t_end = float(f.times[len(f.times) // 2])
    SG, KG, _ = video_channels(f, CFG)
    f2 = load_features(CACHES[0])
    keep = f2.times[f2.tracks[:, 0].astype(int)] <= t_end + 1e-6
    f2.tracks = f2.tracks[keep]                                      # delete every future detection
    SG2, KG2, _ = video_channels(f2, CFG)
    for A, B in ((SG, SG2), (KG, KG2)):
        x1 = window_matrix(f.times, A, np.array([t_end]), 1.0, f.fps)
        x2 = window_matrix(f.times, B, np.array([t_end]), 1.0, f.fps)
        assert np.array_equal(x1, x2)
    assert SG.shape[1] == len(SG_FEATURES) and KG.shape[1] == len(KG_FEATURES)


@needs_cache
def test_counterfactual_recomputation_is_exact_and_edits_are_valid():
    f = load_features(CACHES[0])
    m = cf.recompute_motion(f)
    assert np.array_equal(m, f.motion)                               # factual recomputation reproduces the cache exactly
    tid = int(np.bincount(f.tracks[f.tracks[:, 1] >= 0, 1].astype(int)).argmax())
    steps = f.tracks[f.tracks[:, 1] == tid, 0].astype(int)
    t_end = float(f.times[steps[len(steps) // 2]])
    f_id, info = cf.edit_tracks(f, tid, t_end, 2.0, "identity")
    assert np.array_equal(f_id.motion, f.motion) and info["boxes_valid"]
    for kind in ("closing_speed", "no_cut_in"):
        f2, inf = cf.edit_tracks(f, tid, t_end, 2.0, kind, 0.5)
        assert inf["boxes_valid"] and inf["pixel_channels_kept"]
        assert np.array_equal(f2.motion[:, -6:], f.motion[:, -6:])  # pixel-derived columns kept (stated limitation)
        assert np.array_equal(f.tracks, load_features(CACHES[0]).tracks)  # the factual data is never modified


@needs_cache
def test_graph_explanation_chain_references_real_nodes_edges_and_times():
    f = load_features(CACHES[0])
    _, _, _, snaps, results = video_channels(f, CFG, keep=True)
    checked = 0
    for s, r in zip(snaps, results):
        for ch in graph_explain.chains(s, r):
            assert graph_explain.validate_chain(ch, s)["valid"], ch
            checked += 1
    assert checked > 0
