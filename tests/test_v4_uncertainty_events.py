"""v4 uncertainty propagation (subjective logic), uncertainty-aware reasoning, input-quality monitor, 5-state warning policy and the
Near-Miss Intelligence event store (categories, review rules, encrypted-store requirement, export)."""
from __future__ import annotations

import numpy as np
import pytest

from safedrive_v4.config import load_config
from safedrive_v4.event_memory.store import EventStore, V4Event, auto_category, events_from_timeline
from safedrive_v4.reasoning.uncertain_engine import UncertainReasoningEngine, kgu_vector
from safedrive_v4.scene_graph.builder import SceneGraphBuilder
from safedrive_v4.uncertainty.input_quality import InputQualityMonitor
from safedrive_v4.uncertainty.subjective_logic import Opinion, conjunction, cumulative_fusion, discount, weighted_observations
from safedrive_v4.warning.policy import WarningPolicy

CFG = load_config()


def test_subjective_logic_operators_are_consistent():
    rng = np.random.default_rng(1)
    for _ in range(200):
        x, y = Opinion(*rng.dirichlet([1, 1, 1]), rng.uniform(0.05, 0.95)), Opinion(*rng.dirichlet([1, 1, 1]), rng.uniform(0.05, 0.95))
        z = conjunction(x, y)
        assert abs(z.expectation - x.expectation * y.expectation) < 1e-9 and abs(z.b + z.d + z.u - 1) < 1e-9
    f = cumulative_fusion(Opinion.from_evidence(3, 1), Opinion.from_evidence(2, 2))
    assert np.allclose(f.evidence(), (5, 3))                                 # fusion adds independent evidence
    assert discount(Opinion.from_evidence(4, 0), 0.5).u > Opinion.from_evidence(4, 0).u   # aging raises uncertainty
    lo, hi = weighted_observations([(True, 0.2)] * 5), weighted_observations([(True, 1.0)] * 5)
    assert lo.u > hi.u and lo.d == hi.d == 0.0                               # unreliable evidence -> uncertainty, not disbelief
    with pytest.raises(ValueError):
        Opinion(0.6, 0.6, 0.1)


def _run(track_fn, conf=0.9, steps=12):
    b, e = SceneGraphBuilder(CFG), UncertainReasoningEngine(CFG)
    out = []
    for s in range(steps):
        cx, cy, sz = track_fn(s)
        row = np.array([[3, 2, conf, cx - sz / 2, cy - sz / 2, cx + sz / 2, cy + sz / 2]])
        snap = b.update(s, s / 10, row, None, {"flow_mag_mean": 0.03, "flow_dx_median": 0.0, "brightness": 0.45, "mean_det_conf": conf})
        out.append((snap, e.evaluate(snap)))
    return out


def test_edges_and_nodes_carry_opinions_and_low_confidence_raises_uncertainty():
    hi = _run(lambda s: (0.5, 0.6, 0.05 * 1.12 ** s), 0.9)
    lo = _run(lambda s: (0.5, 0.6, 0.05 * 1.12 ** s), 0.35)
    e_hi = [e for e in hi[10][0].edges if e.relation_type == "APPROACHING"]
    e_lo = [e for e in lo[10][0].edges if e.relation_type == "APPROACHING"]
    assert e_hi and e_hi[0].opinion and e_hi[0].uncertainty is not None
    assert e_lo and e_lo[0].uncertainty > e_hi[0].uncertainty                 # weaker detections = less evidence
    r_hi, r_lo = hi[10][1], lo[10][1]
    # a low detector confidence is evidence that the detection may be wrong: risk belief and expectation drop, disbelief rises
    assert r_lo.temporal_risk.b < r_hi.temporal_risk.b and r_lo.temporal_risk.d > r_hi.temporal_risk.d
    assert r_lo.temporal_risk.expectation < r_hi.temporal_risk.expectation and r_lo.reliability.expectation < r_hi.reliability.expectation
    v = kgu_vector(r_hi)
    assert v.shape == (17,) and np.all(np.isfinite(v))


def test_temporal_risk_opinion_accumulates_and_ages():
    out = _run(lambda s: (0.5, 0.6, 0.05 * 1.12 ** min(s, 8)), 0.9, 30)    # approaching until step 8, then constant size
    u = [r.temporal_risk.u for _, r in out]
    e = [r.temporal_risk.expectation for _, r in out]
    assert max(e[5:10]) > e[2]                                               # evidence builds up while the approach lasts
    assert u[-1] > min(u[5:12])                                              # and ages into uncertainty afterwards


def test_input_quality_monitor_detects_degraded_input():
    rng = np.random.default_rng(0)
    m = InputQualityMonitor()
    tex = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
    assert not m.update(tex, 0.0).degraded
    assert m.update(np.zeros((360, 640, 3), np.uint8), 0.1).degraded           # black frame
    m.reset()
    for i in range(12):
        q = m.update(tex, 0.1 * i)
    assert q.degraded and any("frozen" in r for r in q.reasons)              # frozen image
    m.reset()
    assert m.update(np.full((360, 640, 3), 128, np.uint8), 0.0).degraded       # flat grey = no texture = blurred
    m.reset()
    m.update(tex, 0.0)
    assert m.update(None, 2.0).degraded                                      # camera lost


def test_warning_policy_five_states():
    p = WarningPolicy(0.5, 0.8, 3, 0.45)
    assert [p.decide(0.9, 0, 0.9).level for _ in range(3)][-1] == "HIGH"
    d = p.decide(0.9, 0, 0.2)
    assert d.level == "HIGH" and d.cautious                                  # high risk, unreliable perception -> cautious HIGH
    d = p.decide(0.9, 0, 0.9, input_degraded=True, input_reasons=["image black"])
    assert d.level == "DEGRADED_INPUT" and d.input_degraded
    assert [p.decide(0.9, 0, 0.9).level for _ in range(3)][-1] == "HIGH"      # onset counter restarted after degraded input
    p.reset()
    assert [p.decide(0.1, 0, 0.2).level for _ in range(3)][-1] == "INSUFFICIENT_EVIDENCE"
    g = WarningPolicy(0.5, 0.8, 3, 0.45, evidence_min=0.3)                   # optional evidence gate (H4 policy P1)
    assert [g.decide(0.9, 0, 0.9, kg_evidence=0.1).level for _ in range(3)][-1] == "LOW"
    g.reset()
    assert [g.decide(0.9, 0, 0.9, kg_evidence=0.5).level for _ in range(3)][-1] == "HIGH"


def test_event_store_categories_review_rules_and_export(tmp_path):
    st = EventStore(tmp_path / "store")
    assert auto_category("HIGH", False, 0.4) == "potential_near_miss" and auto_category("HIGH", True, 0.9) == "uncertain"
    assert auto_category("MODERATE", False, 0.9) == "risk_event"
    tl = [{"t": i / 10, "p": 0.2, "level": "LOW", "reliability": 0.8, "kg_risk_expectation": 0.1} for i in range(60)]
    for i in range(20, 26):
        tl[i].update(level="HIGH", p=0.95, kg_risk_expectation=0.5)
    evs = events_from_timeline(tl, {}, "trip1", "clip.mp4")
    assert len(evs) == 1 and evs[0].category_auto == "potential_near_miss" and evs[0].status == "unreviewed"
    assert evs[0].category != "collision"                                     # never automatic
    st.save(evs[0])
    eid = evs[0].event_id
    with pytest.raises(ValueError):
        st.review(eid, "relabel", "COORD-01", "collision", evidence="")       # collision needs written evidence
    ev = st.review(eid, "confirm", "COORD-01")
    assert ev.status == "confirmed" and ev.category == "near_miss" and ev.review_log
    ev = st.review(eid, "reject", "COORD-01", notes="lead car was parked")
    assert ev.category == "normal" and ev.status == "rejected"
    out = st.export("trip1", tmp_path / "x.json")
    assert out.exists() and "trip1" in out.read_text()


def test_clips_are_never_written_outside_the_encrypted_store(tmp_path):
    st = EventStore(tmp_path / "s")
    st.clips_allowed = False                                                 # e.g. VeraCrypt vault not mounted
    clip = tmp_path / "tmp.mp4"
    clip.write_bytes(b"x")
    ev = st.attach_clip(st.save(V4Event("e1", "t", "src", "risk_event", 1, 1, 1, "MODERATE", 0.6, 0.1, 0.8, False, [], [])), clip)
    assert ev.clip is None and "never written unencrypted" in ev.clip_status and not clip.exists()
