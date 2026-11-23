from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from safedrive_v4.app.core import trip_report, trip_report_pdf  # noqa: E402
from safedrive_v4.app.ui import CATEGORY_TEXT, page, pick_trip, stores  # noqa: E402

page("Trip Report", "📄")
st.title("📄 Trip Report")
st.caption("Descriptive only. No composite 'safety score' is produced: none has been validated.")
ev, ts = stores()
tid = pick_trip(ts)
if not tid:
    st.stop()
rep = trip_report(tid, ev)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Events", rep["events_total"])
c2.metric("To review", rep["events_to_review"])
c3.metric("Confirmed near-miss", rep["confirmed_near_miss"])
c4.metric("Rejected (false alarms)", rep["rejected_false_alarms"])
st.write(f"Source **{rep.get('source')}** · {(rep.get('duration_s') or 0) / 60:.1f} min · data kind **{rep.get('data_kind')}**"
         + (" (does not count as real-world driving)" if not rep.get("counts_as_real_driving") else ""))
st.write("Time share by state: " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in rep["time_share_by_level"].items()))
if rep["frequent_risk_factors"]:
    st.write("Frequent risk factors (rule findings at warning onsets): " + ", ".join(f"{k} ({v})" for k, v in rep["frequent_risk_factors"]))
st.subheader("Event timeline")
if rep["events"]:
    st.dataframe(pd.DataFrame([{**e, "category": CATEGORY_TEXT.get(e["category"], e["category"]), "explanation": "; ".join(e["explanation"])}
                               for e in rep["events"]]), hide_index=True, use_container_width=True)
else:
    st.write("No warning events.")
st.subheader("Limitations")
for x in rep["limitations"]:
    st.write("• " + x)
st.caption(rep["disclaimer"])
d1, d2 = st.columns(2)
d1.download_button("⬇ PDF", trip_report_pdf(rep), f"safedrive_trip_{tid}.pdf", "application/pdf")
d2.download_button("⬇ JSON", json.dumps(rep, indent=1, default=float), f"safedrive_trip_{tid}.json", "application/json")
