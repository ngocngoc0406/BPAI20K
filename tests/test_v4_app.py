"""Single SafeDrive AI V4 app: every page renders without exception on a temporary store holding one analysed Nexar trip;
trip report PDF / JSON export works. Nothing is written to the real (encrypted) store."""
from __future__ import annotations

import glob

import pytest

from src.utils.config import resolve

pytestmark = pytest.mark.skipif(not glob.glob(str(resolve("data/processed/features/clean/00063.npz"))), reason="no caches")
PAGES = ["Home.py", "pages/1_Live_Driving_Assistant.py", "pages/2_Video_Analysis.py", "pages/3_Traffic_Scene_Graph.py",
         "pages/4_Risk_Explanation.py", "pages/5_Near_Miss_Memory.py", "pages/6_Trip_Report.py", "pages/7_Diagnostics.py"]


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    import safedrive_v4.event_memory.store as S
    root = tmp_path_factory.mktemp("v4store")
    mp = pytest.MonkeyPatch()
    mp.setattr(S, "default_root", lambda: (root, True))
    from safedrive_v4.app.core import analyze_cached
    ev = S.EventStore()
    r = analyze_cached("00063", ev)
    yield ev, r
    mp.undo()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(store, page):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.cache_resource.clear()
    st.cache_data.clear()
    at = AppTest.from_file(str(resolve(f"safedrive_v4/app/{page}")), default_timeout=300)
    at.session_state["trip_id"] = store[1]["trip_id"]
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_trip_report_exports_and_event_clip(store):
    from safedrive_v4.app.core import trip_report, trip_report_pdf
    ev, r = store
    rep = trip_report(r["trip_id"], ev)
    pdf = trip_report_pdf(rep)
    assert pdf[:4] == b"%PDF" and rep["events_total"] == len(r["events"]) >= 1
    assert any("No composite safety score" in x for x in rep["limitations"]) and rep["data_kind"] == "demo_video"
    e = r["events"][0]
    assert e.category != "collision" and e.clip_status == "saved" and e.clip_offsets["onset"] > 0
