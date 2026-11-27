"""Annotation schema ann-v4.0 + append-only store: validation rules, evidence capture, history / restore, hash chain, submission
lock, blindness, adjudication (originals untouched)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from safedrive_v4.annotation.schema import Actor, Counterfactual, Interaction, TemporalEvent, ValidationError
from safedrive_v4.annotation.store import AccessError, AnnotationStore


@pytest.fixture()
def st(tmp_path):
    acc = tmp_path / "access.json"
    acc.write_text(json.dumps({"raw_access": ["ANN-A", "ANN-B"], "coordinators": ["COORD-01"]}))
    return AnnotationStore(tmp_path / "ann", access_path=acc)


def frame(v=0):
    rng = np.random.default_rng(v)
    return rng.integers(0, 255, (90, 160, 3), dtype=np.uint8)


def ev(st, ann, cat="near_miss", a=10.0, p=11.0, b=12.0, actors=("A",), video="v1"):
    refs = [st.save_evidence(video, frame(int(p * 10)), p, int(p * 10))]
    return TemporalEvent(video, ann, cat, a, b, p if cat in ("risk_event", "near_miss", "collision") else None, list(actors), refs,
                         "lead car brakes hard, gap closes", "observed")


def test_validation_rules(st):
    with pytest.raises(ValidationError):   # event without peak
        TemporalEvent("v1", "ANN-A", "near_miss", 1, 2, None, [], [], "x x x x x").validate()
    with pytest.raises(ValidationError):   # event without evidence
        TemporalEvent("v1", "ANN-A", "risk_event", 1, 3, 2, [], [], "lead brakes").validate()
    with pytest.raises(ValidationError):   # uncertain without a reason
        TemporalEvent("v1", "ANN-A", "uncertain", 1, 3, None, [], [], "too dark to see").validate()
    TemporalEvent("v1", "ANN-A", "uncertain", 1, 3, None, [], [], "too dark to see", "insufficient_evidence",
                  {"visibility_level": "poor", "occlusion_level": "none", "motion_blur": "none", "lighting_quality": "very_low",
                   "annotation_confidence": "low", "uncertainty_reason": "night, no street light"}).validate()
    with pytest.raises(ValidationError):   # insufficient evidence cannot carry an event category
        e = ev(st, "ANN-A")
        e.observation = "insufficient_evidence"
        e.validate()
    one = [st.save_evidence("v1", frame(1), 5.0, 50)]
    with pytest.raises(ValidationError):   # relation from a single frame
        Interaction("v1", "ANN-A", "A", "APPROACHING", 4.0, 6.0, "ego", one).validate()
    two = one + [st.save_evidence("v1", frame(2), 5.5, 55)]
    Interaction("v1", "ANN-A", "A", "APPROACHING", 4.0, 6.0, "ego", two).validate()
    with pytest.raises(ValidationError):   # actor that cannot be found again
        Actor("v1", "ANN-A", "A", "car").validate()
    with pytest.raises(ValidationError):   # counterfactual must stay research-only and point to observed facts
        Counterfactual("v1", "ANN-A", "evt_x", [], "x" * 6, "y" * 6).validate()


def test_evidence_is_content_addressed_and_verifiable(st):
    r = st.save_evidence("v1", frame(3), 2.0, 20)
    assert st.verify_evidence(r)
    p = st.evidence_path("v1", r["image_sha256"])
    p.write_bytes(b"tampered")
    assert not st.verify_evidence(r)


def test_history_restore_and_hash_chain(st):
    e = ev(st, "ANN-A")
    st.put("event", e)
    e.t_end = 13.0
    st.put("event", e)
    assert st.records("v1", "ANN-A", "event")[0].t_end == 13.0
    st.retract("v1", "ANN-A", e.record_id, "duplicate")
    assert st.records("v1", "ANN-A", "event") == []
    st.restore("v1", "ANN-A", e.record_id, 0, "retracted by mistake")
    assert st.records("v1", "ANN-A", "event")[0].t_end == 12.0           # version 0 restored, nothing deleted
    assert [o["op"] for o in st.history("v1", "ANN-A", e.record_id)] == ["create", "update", "retract", "restore"]
    assert st.verify_chain("v1", "ANN-A")["ok"]
    log = st._log("v1", "ANN-A")
    lines = log.read_text().splitlines()
    lines[1] = lines[1].replace("13.0", "14.0")
    log.write_text("\n".join(lines) + "\n")
    assert not st.verify_chain("v1", "ANN-A")["ok"]                     # tampering detected


def test_blindness_submission_lock_and_adjudication(st):
    st.put("event", ev(st, "ANN-A"))
    st.put("event", ev(st, "ANN-B", "risk_event", 9.5, 11.2, 12.4))
    with pytest.raises(AccessError):
        st.view_other("v1", "ANN-A", "ANN-B")                            # annotators never see each other
    st.submit("v1", "ANN-A")
    with pytest.raises(AccessError):
        st.view_other("v1", "COORD-01", "ANN-B")                         # not before both submitted
    with pytest.raises(ValidationError):
        st.put("event", ev(st, "ANN-A", "near_miss", 20, 21, 22))         # submitted: revision needs a reason
    st.submit("v1", "ANN-B")
    assert st.view_other("v1", "COORD-01", "ANN-B")
    with pytest.raises(AccessError):
        st.adjudicate("v1", "ANN-A", "accept_A", {}, None)               # only a coordinator adjudicates
    a = st.records("v1", "ANN-A", "event")[0]
    b = st.records("v1", "ANN-B", "event")[0]
    merged = ev(st, "COORD-01", "near_miss", 9.8, 11.1, 12.2)
    st.adjudicate("v1", "COORD-01", "merged", {"ANN-A": a.record_id, "ANN-B": b.record_id}, merged, "same situation")
    assert st.records("v1", "ANN-A", "event")[0].category == "near_miss" and st.records("v1", "ANN-B", "event")[0].category == "risk_event"
    assert st.adjudications("v1")[0]["decision"] == "merged"


def test_campaign_video_needs_raw_access(st):
    with pytest.raises(AccessError):
        st.check_annotator("Stranger", campaign_video=True)
    st.check_annotator("ANN-A", campaign_video=True)
    with pytest.raises(AccessError):
        st.check_annotator("real name")                                  # pseudonyms only


def test_store_refuses_to_run_unencrypted(monkeypatch, tmp_path):
    import safedrive_v4.annotation.store as S
    monkeypatch.setattr(S, "default_root", lambda: (tmp_path / "x", False))
    with pytest.raises(AccessError):
        S.AnnotationStore()
