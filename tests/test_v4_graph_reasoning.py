"""v4 scene graph + knowledge-guided reasoning: synthetic tracks with known geometry (no model, no video)."""
from __future__ import annotations

import numpy as np
import pytest

from safedrive_v4.config import load_config
from safedrive_v4.knowledge_graph.ontology import add_rules, build_ontology, is_a, triples
from safedrive_v4.reasoning.engine import ReasoningEngine
from safedrive_v4.reasoning.rules import RULES
from safedrive_v4.scene_graph.builder import SceneGraphBuilder
from safedrive_v4.scene_graph.schema import EDGE_TYPES, EGO_ID, NODE_TYPES

CFG = load_config()


def box(cx, cy, s):
    return [cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2]


def run(tracks: dict, steps: int = 20, fps: float = 10.0, scene=None, cfg=CFG):
    """tracks: tid -> (coco_cls, conf, fn(step) -> (cx, cy, size) or None)."""
    b, e = SceneGraphBuilder(cfg, fps), ReasoningEngine(cfg)
    out = []
    for s in range(steps):
        rows = []
        for tid, (cls, conf, fn) in tracks.items():
            g = fn(s)
            if g is not None:
                rows.append([tid, cls, conf, *box(*g)])
        snap = b.update(s, s / fps, np.array(rows, float).reshape(-1, 7), None, scene or {"flow_mag_mean": 0.03, "flow_dx_median": 0.0,
                                                                                            "brightness": 0.45, "mean_det_conf": 0.8})
        out.append((snap, e.evaluate(snap)))
    return out, b


def rels(snap, rel, src=None, dst=None):
    return snap.edges_of(rel, source=src, target=dst)


def test_schema_and_ontology_are_consistent():
    assert "TrafficLight" in NODE_TYPES and "POTENTIAL_CONFLICT" in EDGE_TYPES
    g = add_rules(build_ontology(), RULES)
    assert is_a(g, "Motorcycle", "RoadUser") and is_a(g, "LowTTC", "RiskFactor") and not is_a(g, "Pedestrian", "Vehicle")
    t = triples(g)
    assert ("LowTTC", "INDICATES", "PotentialCollisionRisk") in t and ("Occlusion", "REDUCES", "ObservationReliability") in t
    for r in RULES:   # every rule documents trigger, inputs, KG relations, confidence, explanation, exceptions
        d = r.doc()
        assert d["rule_id"] and d["trigger"] and d["kg_relations"] and 0 < d["base_confidence"] <= 1 and d["explanation"]
        for s, p, o in r.kg_relations:
            assert s in g and o in g


def test_motorcycle_cut_in_creates_edge_and_finding_with_evidence():
    # motorcycle in the right adjacent region moving left (inward) at 0.25 frame/s
    out, _ = run({7: (3, 0.85, lambda s: (0.85 - 0.025 * s, 0.6, 0.08))}, 12)
    snap, res = out[5]   # x = 0.725: still in the right adjacent region, moving inward
    e = rels(snap, "CUTTING_IN_FRONT_OF", "obj:7", EGO_ID)
    assert e and e[0].evidence_features["lateral_in"] > 0.08 and e[0].temporal_validity["support"] >= 2
    f = [x for x in res.findings if x.rule_id == "R01"]
    assert f and f[0].subjects == ["obj:7"] and e[0].edge_id in f[0].edge_ids and "CUTTING_IN_FRONT_OF" in f[0].kg_path
    assert "moves toward the ego path" in f[0].explanation and f[0].timestamp == snap.timestamp


def test_approaching_car_low_ttc_and_high_closing():
    # car in the corridor whose size grows 12 % per step (scale growth ~1.2/s): image TTC < 1.5 s
    out, _ = run({3: (2, 0.9, lambda s: (0.5, 0.6, 0.05 * 1.12 ** s))}, 12)
    snap, res = out[10]
    a = rels(snap, "APPROACHING", "obj:3", EGO_ID)
    assert a and a[0].evidence_features["tau_s"] < 1.5
    ids = {x.rule_id for x in res.findings}
    assert {"R02", "R03"} <= ids and res.risk_evidence > 0
    assert not rels(snap, "MOVING_AWAY", "obj:3", EGO_ID)                  # contradictory relation never co-exists


def test_pedestrian_crossing_vru_rule():
    out, _ = run({11: (0, 0.8, lambda s: (0.30 + 0.03 * s, 0.6, 0.06))}, 12)
    snap, res = out[9]
    assert rels(snap, "CROSSING_PATH", "obj:11", EGO_ID)
    assert any(x.rule_id == "R04" for x in res.findings)


def test_stationary_object_and_short_tracks_create_no_motion_relations():
    out, _ = run({1: (2, 0.9, lambda s: (0.5, 0.6, 0.08)), 2: (2, 0.9, lambda s: (0.7, 0.6, 0.08) if s >= 9 else None)}, 11)
    snap, res = out[10]
    motion = {"APPROACHING", "MOVING_AWAY", "CUTTING_IN_FRONT_OF", "CROSSING_PATH", "POTENTIAL_CONFLICT"}
    assert not [e for e in snap.edges if e.relation_type in motion]
    assert snap.nodes["obj:2"].velocity_estimate is None and snap.nodes["obj:2"].visibility_status == "new"
    assert not [x for x in res.findings if x.kind == "risk"]


def test_low_confidence_detection_gives_no_relation():
    out, _ = run({3: (2, 0.2, lambda s: (0.5, 0.6, 0.05 * 1.12 ** s))}, 12)
    assert not [e for s, _ in out for e in s.edges if e.relation_type == "APPROACHING"]


def test_temporal_validity_requires_persistence():
    # growth only at a single step (one-frame jump) -> no APPROACHING edge
    out, _ = run({3: (2, 0.9, lambda s: (0.5, 0.6, 0.08 if s != 8 else 0.12))}, 12)
    assert not [e for s, _ in out for e in s.edges if e.relation_type == "APPROACHING"]


def test_track_id_change_and_disappearance():
    out, b = run({5: (2, 0.9, lambda s: (0.85 - 0.025 * s, 0.6, 0.08) if s < 8 else None),
                  6: (2, 0.9, lambda s: (0.85 - 0.025 * s, 0.6, 0.08) if s >= 8 else None)}, 14)
    snap, _ = out[8]
    assert "obj:5" not in snap.nodes and snap.nodes["obj:6"].velocity_estimate is None   # new id starts without history
    assert not rels(snap, "CUTTING_IN_FRONT_OF", "obj:6", EGO_ID)                         # no relation copied to the new id
    for s in range(14, 40):   # bounded memory: the lost track is forgotten
        b.update(s, s / 10, np.zeros((0, 7)), None, {})
    assert 5 not in b.hist and 6 not in b.hist and all(len(d) <= CFG["scene_graph"]["persistence_n"] for d in b.persist.values())


def test_occlusion_marks_node_and_lowers_reliability():
    out, _ = run({1: (2, 0.9, lambda s: (0.5, 0.7, 0.2)), 2: (2, 0.9, lambda s: (0.52, 0.62, 0.12))}, 6)
    snap, res = out[5]
    assert rels(snap, "OCCLUDING", "obj:1", "obj:2")
    assert snap.nodes["obj:2"].visibility_status == "partially_occluded"
    assert any(f.rule_id == "R07" for f in res.findings) and res.reliability < 1.0


def test_rule_conflict_risk_on_occluded_object_is_flagged_and_downweighted():
    out, _ = run({1: (2, 0.9, lambda s: (0.5, 0.75, 0.25)), 2: (2, 0.9, lambda s: (0.52, 0.62, 0.08 * 1.12 ** s))}, 12)
    _, res = out[10]
    c = [x for x in res.conflicts if x["type"] == "risk_on_partially_observed_object"]
    assert c and all("conflict" in " ".join(f.exceptions_applied) for f in res.findings if f.kind == "risk" and f.subjects[0] == c[0]["subject"])


def test_insufficient_evidence_and_perception_degradation():
    out, _ = run({1: (2, 0.9, lambda s: (0.5, 0.6, 0.08) if s >= 9 else None)}, 10,
                 scene={"flow_mag_mean": 0.03, "flow_dx_median": 0.0, "brightness": 0.08, "mean_det_conf": 0.3})
    _, res = out[9]
    assert res.insufficient and any(f.rule_id == "R08" for f in res.findings) and res.reliability < 1.0
    assert res.risk_evidence == 0.0   # degraded input is not risk


def test_ego_turning_halves_cut_in_confidence():
    def go(turn):
        o, _ = run({7: (3, 0.85, lambda s: (0.85 - 0.025 * s, 0.6, 0.08))}, 10,
                   scene={"flow_mag_mean": 0.03, "flow_dx_median": turn, "brightness": 0.45, "mean_det_conf": 0.8})
        return max(f.confidence for f in o[5][1].findings if f.rule_id == "R01")
    assert go(0.05) == pytest.approx(0.5 * go(0.0), rel=1e-6)


def test_opposite_direction_disabled_by_default():
    out, _ = run({4: (2, 0.9, lambda s: (0.25 - 0.01 * s, 0.6, 0.05 * 1.15 ** s))}, 12)
    assert not [e for s, _ in out for e in s.edges if e.relation_type == "OPPOSITE_DIRECTION"]
