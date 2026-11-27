"""Active annotation (label-free scores, same budget, coordinator-only reasons), counterfactual rules (must rest on observed
evidence; never exported) and the privacy-gated export."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from safedrive_v4.annotation.active import queue, rarity_scores, select
from safedrive_v4.annotation.export import ExportRefused, export
from safedrive_v4.annotation.schema import Counterfactual, TemporalEvent, ValidationError
from safedrive_v4.annotation.store import AnnotationStore


@pytest.fixture()
def st(tmp_path):
    acc = tmp_path / "access.json"
    acc.write_text(json.dumps({"raw_access": ["ANN-A", "ANN-B"], "coordinators": ["COORD-01"]}))
    return AnnotationStore(tmp_path / "ann", access_path=acc)


def _ev(st, v, ann, cat="near_miss"):
    ref = st.save_evidence(v, np.full((30, 40, 3), 90, np.uint8), 5.0, 50)
    e = TemporalEvent(v, ann, cat, 4, 6, 5, [], [ref], "lead car brakes hard")
    st.put("event", e)
    return e


def test_select_same_budget_and_deterministic():
    pool = [f"v{i}" for i in range(20)]
    unc = np.arange(20, dtype=float)
    rar = np.zeros(20)
    for s in ("random", "uncertainty", "rarity", "hybrid"):
        out = select(s, pool, 5, np.random.default_rng(0), unc, unc, rar)
        assert len(out) == 5 == len(set(out))
    assert select("uncertainty", pool, 3, np.random.default_rng(0), unc) == ["v19", "v18", "v17"]
    assert select("random", pool, 5, np.random.default_rng(1)) == select("random", pool, 5, np.random.default_rng(1))


def test_rarity_prefers_rare_conditions():
    recs = {f"v{i}": SimpleNamespace(meta={"light_conditions": "Day", "weather": "Clear", "scene": "Urban"}) for i in range(9)}
    recs["v9"] = SimpleNamespace(meta={"light_conditions": "Night", "weather": "Rain", "scene": "Highway"})
    r = rarity_scores(list(recs), recs)
    assert int(np.argmax(r)) == 9


def test_queue_reasons_and_done_last(st):
    recs = {v: SimpleNamespace(meta={"light_conditions": "Day", "weather": "Clear", "scene": "Urban"}) for v in ("a", "b")}
    for ann in ("ANN-A", "ANN-B"):
        _ev(st, "a", ann)
        st.submit("a", ann)
    q = queue(st, ["a", "b"], recs)
    assert [r["video_id"] for r in q] == ["b", "a"]                      # agreed double-submitted video goes last
    assert all("reasons_coordinator_only" in r for r in q)


def test_counterfactual_needs_observed_evidence(st):
    e = _ev(st, "v1", "ANN-A")
    with pytest.raises(ValidationError):
        st.put("counterfactual", Counterfactual("v1", "ANN-A", e.record_id, ["missing"], "lead brakes", "if ego kept 2 s gap"))
    with pytest.raises(ValidationError):
        Counterfactual("v1", "ANN-A", e.record_id, [e.record_id], "lead brakes", "if ego kept 2 s gap", research_only=False).validate()
    with pytest.raises(ValidationError):   # another annotator's record cannot be used (blindness)
        st.put("counterfactual", Counterfactual("v1", "ANN-B", e.record_id, [e.record_id], "lead brakes", "if ego kept 2 s gap"))
    st.put("counterfactual", Counterfactual("v1", "ANN-A", e.record_id, [e.record_id], "lead brakes", "if ego kept a 2 s gap",
                                            "not_answerable_from_video"))
    assert len(st.records("v1", "ANN-A", "counterfactual")) == 1


def test_export_gate_and_counterfactual_exclusion(st, tmp_path):
    e = _ev(st, "v1", "ANN-A")
    st.put("counterfactual", Counterfactual("v1", "ANN-A", e.record_id, [e.record_id], "lead brakes", "if ego kept a 2 s gap"))
    _ev(st, "v1", "ANN-B", "risk_event")
    st.put("event", TemporalEvent("v2", "ANN-A", "normal", 0, 10, None, [], [], ""))   # draft only, never submitted
    for ann in ("ANN-A", "ANN-B"):
        st.submit("v1", ann)
    pol = tmp_path / "policy.json"
    with pytest.raises(ExportRefused):
        export(st, tmp_path / "out", "ANN-A", public={"v1"}, policy_path=pol)            # not a coordinator
    with pytest.raises(ExportRefused):
        export(st, tmp_path / "out", "COORD-01", policy_path=pol)                         # not public, not approved
    assert not (tmp_path / "out").exists()
    pol.write_text(json.dumps({"sharing_enabled": False}))
    with pytest.raises(ExportRefused):
        export(st, tmp_path / "out", "COORD-01", public={"v1"}, policy_path=pol)
    pol.unlink()
    r = export(st, tmp_path / "out", "COORD-01", public={"v1"}, split_of={"v1": "dev_fold_2"}, policy_path=pol)
    d = tmp_path / "out" / r["export_id"]
    blob = (d / "annotations.json").read_text(encoding="utf-8")
    assert "research_only" not in blob and "hypothetical" not in blob                # counterfactual excluded
    a = json.loads(blob)
    assert list(a["videos"]) == ["v1"] and a["videos"]["v1"]["status"] == "submitted_only"
    assert json.loads((d / "split_manifest.json").read_text())["v1"] == "dev_fold_2"
    man = json.loads((d / "evidence_manifest.json").read_text())["evidence"]
    assert len(man) == 2 and all(m["verified"] for m in man) and not list(d.glob("*.jpg"))
    prov = json.loads((d / "provenance.json").read_text())
    assert all(h["chain"] for h in prov["op_chains"].values())
    assert "near_miss" in (d / "summary.csv").read_text()


def test_nexar_split_marks_heldout():
    from safedrive_v4.annotation.export import nexar_split_of
    sp = nexar_split_of()
    if not sp:
        pytest.skip("no split file")
    assert set(sp.values()) <= {"dev", "held_out"} and "held_out" in sp.values()
