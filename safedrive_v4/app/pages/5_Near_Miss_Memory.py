from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from safedrive_v4.app.ui import CATEGORY_TEXT, page, store_banner, stores  # noqa: E402
from safedrive_v4.event_memory.store import HUMAN_CATEGORIES  # noqa: E402

page("Near-Miss Memory", "🧠")
st.title("🧠 Near-Miss Memory")
st.caption("Automatic categories are proposals. A collision is never recorded automatically: only by a reviewer with written evidence.")
ev, ts = stores()
store_banner(ev)
evs = ev.all()
if not evs:
    st.info("No events yet.")
    st.stop()
f1, f2 = st.columns(2)
cat = f1.multiselect("Category", sorted({e.category for e in evs}), default=sorted({e.category for e in evs}))
stat = f2.multiselect("Review status", ["unreviewed", "confirmed", "rejected", "relabelled"], default=["unreviewed", "confirmed", "relabelled"])
sel = [e for e in evs if e.category in cat and e.status in stat]
st.dataframe(pd.DataFrame([{"event": e.event_id, "trip": e.trip_id, "time (s)": f"{e.t_onset:.1f}–{e.t_end:.1f}", "category": CATEGORY_TEXT.get(e.category, e.category),
                            "status": e.status, "peak": e.level_peak, "p": round(e.p_peak, 2), "KG evidence": round(e.kg_evidence_peak, 2),
                            "cautious": e.cautious, "clip": e.clip_status} for e in sel]), hide_index=True, use_container_width=True)
if not sel:
    st.stop()
eid = st.selectbox("Open event", [e.event_id for e in sel], format_func=lambda k: f"{k} · {next(e for e in sel if e.event_id == k).category}")
e = ev.get(eid)
a, b = st.columns([3, 2])
if e.clip and Path(e.clip).exists():
    a.video(e.clip)
    o = e.clip_offsets
    a.caption(f"Clip: before {o.get('onset', 0):.1f} s · event {o.get('onset', 0):.1f}–{o.get('end', o.get('onset', 0)):.1f} s · "
              f"after until {o.get('post_end', 0):.1f} s (clip time)")
else:
    a.info(f"No clip: {e.clip_status}")
with b:
    st.markdown(f"**{CATEGORY_TEXT.get(e.category, e.category)}** · status **{e.status}**")
    st.write(f"Peak {e.level_peak} p={e.p_peak:.2f} at {e.t_peak:.1f} s · KG risk evidence {e.kg_evidence_peak:.2f} · "
             f"min. perception reliability {e.reliability_min:.2f}{' · cautious' if e.cautious else ''}")
    for x in e.explanation[:3]:
        st.code(x, language=None)
    if not e.explanation:
        st.write("No knowledge-graph rule fired (feature contributions only).")
if e.risk_timeline:
    rt = pd.DataFrame(e.risk_timeline, columns=["t", "p", "level", "kg_evidence", "reliability"])
    st.line_chart(rt.set_index("t")[["p", "kg_evidence", "reliability"]], height=180)
st.caption(f"Track history: {len(e.track_history)} observations of road users {e.track_ids}")
st.subheader("Review")
with st.form("review"):
    who = st.text_input("Reviewer pseudonym")
    dec = st.radio("Decision", ["confirm", "reject", "relabel"], horizontal=True,
                   format_func={"confirm": "Confirm category", "reject": "Reject (false alarm)", "relabel": "Relabel"}.get)
    newcat = st.selectbox("Category (for relabel)", HUMAN_CATEGORIES)
    notes = st.text_input("Notes")
    evidence = st.text_area("Evidence (required for a collision: what was seen, damage, report)")
    if st.form_submit_button("Save review"):
        try:
            ev.review(eid, dec, who, newcat if dec == "relabel" else None, notes, evidence)
            st.success("Saved.")
            st.rerun()
        except ValueError as ex:
            st.error(str(ex))
if e.review_log:
    st.dataframe(pd.DataFrame(e.review_log), hide_index=True, use_container_width=True)
st.download_button("Export this event (JSON)", json.dumps(asdict(e), indent=1, default=float), f"event_{eid}.json", "application/json")
