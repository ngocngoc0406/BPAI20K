"""SafeDrive AI V4: one driving-safety copilot app.
Run: streamlit run safedrive_v4/app/Home.py --server.port 8502 --server.address localhost
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from safedrive_v4.app.ui import page, store_banner, stores  # noqa: E402

page("Home", "🛡️")
st.title("🛡️ SafeDrive AI V4: driving safety copilot")
st.markdown("""
Understands the traffic around your car from a dashcam, estimates collision risk, **tells you when its own evidence is weak**,
explains every warning with the objects, relations and frames behind it, keeps dangerous moments for review and summarises each
trip.

| Screen | What you do there |
|---|---|
| **1 · Live Driving Assistant** | Connect the camera (or play a demo video) and follow risk, warnings and input quality live |
| **2 · Video Analysis** | Analyse a recorded dashcam video end to end |
| **3 · Traffic Scene Graph** | See which road users interact with your car, since when, and how the evidence evolves |
| **4 · Risk Explanation** | Open a warning: road users, relations, reasoning path, contributing factors, evidence frames, uncertainty, unknowns |
| **5 · Near-Miss Memory** | Replay events (before / during / after), confirm or reject them, label and export |
| **6 · Trip Report** | Event timeline, near-miss records, explanations and limitations, as PDF / JSON |
| **7 · Model & System Diagnostics** | Model provenance, benchmark results, latency, resources, input-quality statistics |
""")
ev, ts = stores()
store_banner(ev)
trips = ts.list()
c1, c2, c3 = st.columns(3)
c1.metric("Trips", len(trips))
c2.metric("Events to review", sum(e.status == "unreviewed" for e in ev.all()))
c3.metric("Real-world driving recorded (h)", f"{sum(t.get('duration_s', 0) for t in trips if t.get('counts_as_real_driving')) / 3600:.1f}")
st.info("How good is the risk model? On 224 held-out Nexar clips it catches about 12 % of collisions / near-collisions at about "
        "9 false alarms per hour, and fewer than 5 % at least 2 s ahead. SafeDrive is a review and awareness aid, not a safety "
        "system: it never brakes or steers. Details: Model & System Diagnostics.")
