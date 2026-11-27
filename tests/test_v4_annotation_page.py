"""Page 8 (Annotation Intelligence) renders for an annotator and a coordinator on a temporary store; the annotator view shows
no model output and no coordinator tab; refused when the encrypted volume is absent."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.utils.config import resolve

PAGE = str(resolve("safedrive_v4/app/pages/8_Annotation_Intelligence.py"))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    import safedrive_v4.event_memory.store as E
    from src.realworld import privacy
    root = tmp_path / "sessions" / "v4"
    root.mkdir(parents=True)
    monkeypatch.setattr(E, "default_root", lambda: (root, True))
    acc = tmp_path / "access.json"
    acc.write_text(json.dumps({"raw_access": ["ANN-A", "ANN-B"], "coordinators": ["COORD-01"]}))
    monkeypatch.setattr(privacy, "ACCESS_PATH", acc)
    from safedrive_v4.annotation.schema import TemporalEvent
    from safedrive_v4.annotation.store import AnnotationStore
    S = AnnotationStore(access_path=acc)
    for ann, cat in (("ANN-A", "near_miss"), ("ANN-B", "risk_event")):
        ref = S.save_evidence("00063", np.full((30, 40, 3), 80, np.uint8), 5.0, 150)
        S.put("event", TemporalEvent("00063", ann, cat, 4, 6, 5, [], [ref], "lead brakes"))
        S.submit("00063", ann)
    return S


def _app():
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.cache_data.clear()
    at = AppTest.from_file(PAGE, default_timeout=300)
    at.session_state["ann_video"] = "00063"
    return at


def test_annotator_view(env):
    at = _app()
    at.run()
    assert not at.exception
    at.text_input(key="ann_who").set_value("ANN-A").run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [t.label for t in at.tabs]
    assert "Agreement" not in labels and "Export" not in labels and "Active queue" in labels
    text = " ".join(str(m.value) for m in at.markdown) + " ".join(str(c.value) for c in at.caption)
    assert "risk_event" not in text                     # never shows the other annotator's label
    at.button(key="p_tl").click().run()                 # +0.1 s step works (callback)
    assert not at.exception and at.session_state["t_tl"] == pytest.approx(0.1)


def test_coordinator_view(env):
    at = _app()
    at.run()
    at.radio[0].set_value("Coordinator")
    at.text_input(key="ann_who").set_value("COORD-01").run()
    assert not at.exception, [e.value for e in at.exception]
    assert {"Agreement", "Dataset quality", "Adjudication", "Export"} <= {t.label for t in at.tabs}


def test_non_coordinator_cannot_take_role(env):
    at = _app()
    at.run()
    at.radio[0].set_value("Coordinator")
    at.text_input(key="ann_who").set_value("ANN-A").run()
    assert any("coordinator list" in str(e.value) for e in at.error)


def test_refuses_without_encrypted_volume(tmp_path, monkeypatch):
    import safedrive_v4.event_memory.store as E
    monkeypatch.setattr(E, "default_root", lambda: (tmp_path, False))
    at = _app()
    at.run()
    assert any("never written unencrypted" in str(e.value) for e in at.error)
