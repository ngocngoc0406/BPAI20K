"""Near-Miss Memory: browse stored safety events per trip, replay clips, read grounded explanations, export the trip report."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.safety.near_miss import NearMissMemory
from src.safety.trip_report import TripLog, build_report, report_pdf
from src.utils.config import resolve
from ui.common import card, pill, setup

setup("Near-Miss Memory", "🗂️")
st.title("🗂️ Near-Miss Memory & Trip Report")
st.caption("Events are model outputs on image-plane cues, not verified incidents. Offline: **alert** = an alarm was raised; "
           "**near-miss candidate** = probability above the CAUTION level (≈20 false alarms/h on dev) for ≥ 0.6 s without an alarm. "
           "Live: **risk_event** (CAUTION episode), **warning_event** (alarm), **potential_near_miss** (alarm episode reaching HIGH RISK, "
           "not verified). **confirmed_collision** is only ever set manually with evidence. Research prototype; it does not brake or steer.")

mem = NearMissMemory()
trips = mem.trips()
if not trips:
    st.info("No events stored yet. Analyse a video with **Save safety events to Near-Miss Memory** enabled.", icon="🎬")
    st.stop()

trip_id = st.selectbox("Trip", trips, index=len(trips) - 1)
events = mem.by_trip(trip_id)
trip = TripLog().get(trip_id)
rep = build_report(trip, events)
c = st.columns(4)
card(c[0], "Clips analysed", str(rep["n_clips"]), f"{rep['duration_min']:.1f} min")
card(c[1], "Alerts", str(rep["alerts"]), f"{(rep['alerts_per_hour'] or 0):.1f} per hour")
card(c[2], "Near-miss candidates", str(rep["near_miss_candidates"]), f"{(rep['near_miss_candidates_per_hour'] or 0):.1f} per hour")
card(c[3], "Top cue", rep["top_cues"][0][0] if rep["top_cues"] else "—", ", ".join(rep["profiles"]))
st.write("")

c1, c2 = st.columns(2)
c1.download_button("⬇ Trip report (PDF)", report_pdf(rep), f"{trip_id}_report.pdf", "application/pdf", use_container_width=True)
c2.download_button("⬇ Trip report (JSON)", json.dumps(rep, indent=2, ensure_ascii=False, default=str).encode("utf-8"),
                   f"{trip_id}_report.json", "application/json", use_container_width=True)

if not events:
    st.info("This trip has analysed clips but no stored events.")
    st.stop()
df = pd.DataFrame([{"type": e.get("category") or e["type"], "video": e["video"], "start s": e["t_start"], "peak s": e["t_peak"], "P peak": e["p_peak"],
                    "level": e["level_peak"], "lead object": (e.get("lead_object") or {}).get("class", "—"),
                    "cues": ", ".join(f["feature"] for f in e.get("factors", [])), "id": e["event_id"]} for e in events])
all_types = sorted(df.type.unique())
types = st.multiselect("Event types", all_types, default=all_types,
                       help="Offline: alert / near_miss_candidate. Live: risk_event / warning_event / potential_near_miss. "
                            "confirmed_collision is set only manually, with evidence.")
view = df[df.type.isin(types)]
st.dataframe(view.round(2), hide_index=True, use_container_width=True, height=260)

sel = st.selectbox("Inspect event", view.id.tolist(), format_func=lambda i: f"{i} · " + " · ".join(
    str(x) for x in view[view.id == i][["type", "video", "peak s"]].iloc[0].tolist()))
ev = next(e for e in events if e["event_id"] == sel)
l, r = st.columns([3, 2])
with l:
    if ev.get("clip") and Path(ev["clip"]).exists():
        st.video(ev["clip"])
    else:
        st.caption("No clip stored for this event (frames were not kept).")
with r:
    st.markdown(pill(ev["level_peak"]) + f" &nbsp; **{ev['type']}** at {ev['t_peak']:.1f} s in `{ev['video']}`", unsafe_allow_html=True)
    st.markdown(f"<div class='sd-card'>{ev['explanation']}</div>", unsafe_allow_html=True)
    for fc in ev.get("factors", []):
        st.markdown(f"- **{fc['description']}**: {fc['value']:.3g} (typical {fc['typical']:.3g})")
    lo = ev.get("lead_object")
    if lo:
        st.caption(f"Largest road user at peak: {lo['class']} #{lo['track_id']} (conf {lo['confidence']:.2f}, "
                   f"{lo['area_fraction']:.1%} of frame)")
    st.caption(f"Model {ev['model']} · profile {ev['profile']} · mean MI {ev['mutual_info_mean']:.3f} · stored {ev['created']}")
    if ev.get("clip_status") and ev["clip_status"] != "saved":
        st.error(f"Clip not available: {ev['clip_status']}")
    if ev.get("confirmed_evidence"):
        st.success(f"Confirmed collision (manual): {ev['confirmed_evidence']}")
    with st.expander("Mark as confirmed collision (manual, evidence required)"):
        evid = st.text_input("Evidence (e.g. visible contact in the clip, police / insurance reference)", key=f"evid_{sel}")
        if st.button("Confirm collision", key=f"conf_{sel}", disabled=len(evid.strip()) < 5):
            mem.confirm_collision(sel, evid)
            st.rerun()
    if st.button("🗑 Delete this event"):
        mem.delete(sel)
        st.rerun()
st.caption(f"Store: `{resolve('data/memory').relative_to(resolve('.'))}` (local only).")
