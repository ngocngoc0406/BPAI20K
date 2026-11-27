"""Annotation -> KG triples with evidence + provenance; evidence lookup and integrity; scene-graph validation (E6) on synthetic
snapshots; the three explanation views stay separate."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from safedrive_v4.annotation.kg_link import evidence_for, scene_graph_validation, to_graph, triples
from safedrive_v4.annotation.schema import Actor, Interaction, TemporalEvent
from safedrive_v4.annotation.store import AnnotationStore
from safedrive_v4.explainability.views import views


@pytest.fixture()
def st(tmp_path):
    acc = tmp_path / "access.json"
    acc.write_text(json.dumps({"raw_access": ["ANN-A", "ANN-B"], "coordinators": ["COORD-01"]}))
    return AnnotationStore(tmp_path / "ann", access_path=acc)


def _refs(st, ts, v="v1"):
    return [st.save_evidence(v, np.full((40, 60, 3), int(t * 10) % 255, np.uint8), t, int(t * 10)) for t in ts]


def _populate(st):
    for ann in ("ANN-A", "ANN-B"):
        st.put("actor", Actor("v1", ann, "A", "car", track_id=7))
        st.put("actor", Actor("v1", ann, "B", "car", track_id=9))
        st.put("event", TemporalEvent("v1", ann, "near_miss", 10, 13, 12, ["A", "B"], _refs(st, [12.0]), "A closes on B, B brakes"))
        st.put("interaction", Interaction("v1", ann, "A", "APPROACHING", 10, 12.5, "ego", _refs(st, [10.0, 12.0])))
        st.put("interaction", Interaction("v1", ann, "B", "BRAKING", 11, 12.5, None, _refs(st, [11.0, 12.0])))
        st.submit("v1", ann)
    e = st.records("v1", "ANN-A", "event")[0]
    st.adjudicate("v1", "COORD-01", "accept_A", {"ANN-A": e.record_id, "ANN-B": None}, e)
    i = next(x for x in st.records("v1", "ANN-A", "interaction") if x.relation == "APPROACHING")
    st.adjudicate("v1", "COORD-01", "accept_A", {"ANN-A": i.record_id}, i)
    return e


def test_triples_have_evidence_provenance_and_temporal_validity(st):
    e = _populate(st)
    trs = triples(st, "v1", "adjudicated")
    rels = {(t["source"], t["relation"], t["target"]) for t in trs}
    assert (f"event:{e.record_id}", "INVOLVES", "actor:A") in rels and ("actor:A", "APPROACHING", "ego") in rels
    assert any(t["relation"] == "SUPPORTED_BY" for t in trs)
    assert all(t["evidence"] and "adjudication" in t["provenance"] and t["valid_to"] >= t["valid_from"] for t in trs)
    single = triples(st, "v1", "ANN-B")
    assert any(t["relation"] == "BRAKING" and t["target"].startswith("evidence:") for t in single)   # unary grounded on frames
    assert all(t["provenance"]["status"].startswith("single annotator") for t in single)


def test_evidence_lookup_verifies_frames(st):
    _populate(st)
    g = to_graph(triples(st, "v1", "adjudicated"))
    ev = evidence_for(st, g, "actor:A", "APPROACHING", "ego")
    assert len(ev) == 2 and all(x["verified"] for x in ev)
    from pathlib import Path
    Path(ev[0]["path"]).write_bytes(b"tampered")
    assert not evidence_for(st, g, "actor:A", "APPROACHING", "ego")[0]["verified"]


def test_scene_graph_validation(st):
    assert scene_graph_validation(st, "v1", [])["status"].startswith("NOT EVALUATED")
    _populate(st)
    edge = lambda rel, src: SimpleNamespace(relation_type=rel, source_node=src, target_node="ego")  # noqa: E731
    snaps = [SimpleNamespace(timestamp=round(t, 1), edges=[edge("APPROACHING", "obj:7")] if 10.5 <= t <= 12.0 else
                             ([edge("APPROACHING", "obj:7")] if 14.0 <= t <= 14.3 else []))
             for t in np.arange(9.0, 15.0, 0.1)]
    r = scene_graph_validation(st, "v1", snaps)["per_relation"]["APPROACHING"]
    assert 0 < r["precision"] < 1 and 0 < r["recall"] < 1          # auto relation outside the human interval and partly missed


def test_three_explanation_views_are_separate():
    tr = SimpleNamespace(video_id="v1", timestamp=12.0, shap=[{"level": "group (stable)", "group": "v3_motion", "shap_logit": 0.4}],
                         nodes={"ego": {}, "obj:7": {"node_type": "Car", "confidence": 0.8, "tau_s": 1.2}},
                         kg_paths=[["obj:7", "APPROACHING", "ego"]], findings=[{"rule_id": "R02", "explanation": "x"}])
    v = views(tr)
    assert set(v) == {"model_explanation", "evidence_explanation", "rule_interpretation"}
    assert "NOT used by the v3 model" in v["rule_interpretation"]["note"] and v["model_explanation"]["available"]
    tr2 = SimpleNamespace(video_id="v1", timestamp=1.0, shap=[], nodes={"ego": {}}, kg_paths=[], findings=[])
    v2 = views(tr2)
    assert not v2["model_explanation"]["available"] and not v2["rule_interpretation"]["available"]   # nothing invented
