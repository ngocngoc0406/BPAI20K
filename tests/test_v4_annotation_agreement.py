"""Agreement metrics on constructed annotations with known answers (events, boundaries, actors, relations, ordinal kappa)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from safedrive_v4.annotation.agreement import dataset_agreement, video_agreement, weighted_kappa
from safedrive_v4.annotation.schema import Actor, Interaction, TemporalEvent
from safedrive_v4.annotation.store import AnnotationStore


@pytest.fixture()
def st(tmp_path):
    acc = tmp_path / "access.json"
    acc.write_text(json.dumps({"raw_access": ["ANN-A", "ANN-B"], "coordinators": ["COORD-01"]}))
    return AnnotationStore(tmp_path / "ann", access_path=acc)


def _ev(st, ann, cat, a, p, b, actors=("A",), video="v1", unc=None):
    img = np.full((60, 80, 3), int(p * 10) % 255, np.uint8)
    refs = [st.save_evidence(video, img, p, int(p * 10))] if cat != "normal" else []
    e = TemporalEvent(video, ann, cat, a, b, p if cat in ("risk_event", "near_miss", "collision") else None, list(actors), refs,
                      "situation description", "observed")
    if unc:
        e.uncertainty.update(unc)
    st.put("event", e)
    return e


def _rel(st, ann, sub, rel, a, b, obj="ego", video="v1"):
    refs = [st.save_evidence(video, np.full((60, 80, 3), k, np.uint8), t, int(t * 10)) for k, t in ((1, a), (2, b))]
    st.put("interaction", Interaction(video, ann, sub, rel, a, b, obj, refs))


def test_identical_annotations_agree_perfectly(st):
    for ann in ("ANN-A", "ANN-B"):
        st.put("actor", Actor("v1", ann, "A", "car", track_id=7))
        _ev(st, ann, "near_miss", 10, 11, 12)
        _ev(st, ann, "risk_event", 20, 21, 23)
        _rel(st, ann, "A", "APPROACHING", 10, 11.5)
    r = video_agreement(st, "v1", "ANN-A", "ANN-B")
    assert r["matched"] == 2 and r["event_detection_f1"] == 1.0 and r["kappa_category_matched"] == 1.0
    assert r["boundary_deviation"]["t_start"]["mean_abs_s"] == 0.0 and r["actor_f1"] == 1.0 and r["relation_f1"] == 1.0
    assert r["disagreements"] == []


def test_disagreements_are_measured_not_resolved(st):
    st.put("actor", Actor("v1", "ANN-A", "A", "car", track_id=7))
    st.put("actor", Actor("v1", "ANN-B", "X", "car", track_id=7))           # different label, same track -> mapped
    _ev(st, "ANN-A", "near_miss", 10, 11, 12)
    _ev(st, "ANN-B", "risk_event", 10.5, 11.2, 12.5, actors=("X",))          # same situation, other category, shifted
    _ev(st, "ANN-A", "risk_event", 30, 31, 32)                               # only A
    _ev(st, "ANN-B", "uncertain", 40, 41, 42, actors=(), unc={"annotation_confidence": "low", "uncertainty_reason": "glare, cannot see"})
    _rel(st, "ANN-A", "A", "CUTTING_IN", 9, 11)
    _rel(st, "ANN-B", "X", "APPROACHING", 9.2, 11)
    r = video_agreement(st, "v1", "ANN-A", "ANN-B")
    assert r["matched"] == 1 and r["event_detection_f1"] == pytest.approx(2 / 4)
    assert r["boundary_deviation"]["t_start"]["mean_abs_s"] == pytest.approx(0.5)
    assert r["actor_f1"] == 1.0                                              # track id maps A <-> X
    assert r["relation_f1"] == 0.0 and r["kappa_relation_type_matched_pairs"] == 0.0   # CUTTING_IN vs APPROACHING on the same pair
    kinds = {d["type"] for d in r["disagreements"]}
    assert {"only_ANN-A", "only_ANN-B", "category"} <= kinds
    assert r["uncertainty"]["ANN-B"]["uncertain_share"] == pytest.approx(0.5)  # uncertain kept, never mapped to normal
    assert st.records("v1", "ANN-B", "event")[1].category in ("risk_event", "uncertain")


def test_weighted_kappa_penalises_far_disagreement_more():
    lv = ("clear", "partial", "poor", "not_visible")
    near = weighted_kappa(["clear", "partial", "poor", "clear"], ["partial", "partial", "poor", "clear"], lv)
    far = weighted_kappa(["clear", "partial", "poor", "clear"], ["not_visible", "partial", "poor", "clear"], lv)
    assert weighted_kappa(["clear", "poor"], ["clear", "poor"], lv) == 1.0 and near > far


def test_dataset_agreement_needs_both_submissions(st):
    _ev(st, "ANN-A", "near_miss", 10, 11, 12)
    _ev(st, "ANN-B", "near_miss", 10, 11, 12)
    assert dataset_agreement(st, "ANN-A", "ANN-B")["status"].startswith("NOT EVALUATED")
    st.submit("v1", "ANN-A"), st.submit("v1", "ANN-B")
    d = dataset_agreement(st, "ANN-A", "ANN-B")
    assert d["videos"] == 1 and d["kappa_category_pooled_with_unmatched"] == 1.0
