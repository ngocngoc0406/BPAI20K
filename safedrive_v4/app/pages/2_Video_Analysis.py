from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from safedrive_v4.app.ui import LEVEL_STYLE, page, pipeline, store_banner, stores  # noqa: E402

page("Video Analysis", "🎞️")
st.title("🎞️ Video Analysis")
ev, ts = stores()
store_banner(ev)
src = st.radio("Input", ["Nexar demo clip (fast: perception already cached)", "Video file on this computer (full perception: YOLO + ByteTrack)",
                         "Upload a video"], horizontal=False)
if src.startswith("Nexar"):
    vid = st.text_input("Nexar clip id", "00063")
    if st.button("Analyse", type="primary"):
        from safedrive_v4.app.core import analyze_cached
        with st.spinner("Scene graph · reasoning · risk · explanations…"):
            r = analyze_cached(vid.strip(), ev, pipeline())
        st.session_state["trip_id"] = r["trip_id"]
else:
    path = None
    if src.startswith("Video file"):
        path = st.text_input("Path to a video file", "")
    else:
        up = st.file_uploader("Dashcam video", type=["mp4", "avi", "mov", "mkv"], disabled=not ev.encrypted)
        if not ev.encrypted:
            st.error("Uploads are stored only inside the encrypted session volume, which is not mounted.")
        if up is not None and ev.encrypted:
            tmp = ev.root / "uploads"
            tmp.mkdir(parents=True, exist_ok=True)
            path = str(tmp / up.name)
            Path(path).write_bytes(up.getbuffer())
    kind = st.selectbox("Data kind", ["demo_video", "controlled_test", "real_driving"],
                        help="'real_driving' only for consented campaign footage; anything else never counts as real-world driving")
    if st.button("Analyse", type="primary", disabled=not path):
        if not Path(path).exists():
            st.error("File not found.")
            st.stop()
        from safedrive_v4.app.core import analyze_video
        bar = st.progress(0.0, "starting")
        r = analyze_video(path, ev, kind, pipeline(), lambda f, m: bar.progress(min(max(f, 0.0), 1.0), m))
        bar.progress(1.0, "done")
        st.session_state["trip_id"] = r["trip_id"]

tid = st.session_state.get("trip_id")
if tid:
    meta, tl = ts.meta(tid), pd.DataFrame(ts.timeline(tid))
    st.success(f"Trip {tid}: {meta['n_events'] if 'n_events' in meta else 0} event(s) · analysed in {meta.get('analysis_wall_s', 0):.1f} s "
               f"for {meta.get('duration_s', 0):.0f} s of video · data kind {meta.get('data_kind')}")
    st.line_chart(tl.set_index("t")[["p", "reliability", "kg_risk_expectation"]], height=220)
    st.caption("p = risk probability (frozen v3 model) · reliability = observation reliability (subjective-logic expectation) · "
               "kg_risk_expectation = knowledge-graph risk evidence (temporal opinion)")
    st.write(" · ".join(f"{LEVEL_STYLE[k][0]} {LEVEL_STYLE[k][1]}: {v}" for k, v in tl["level"].value_counts().items() if k in LEVEL_STYLE))
    evs = ev.all(tid)
    if evs:
        st.dataframe(pd.DataFrame([{"event": e.event_id, "from (s)": round(e.t_onset, 1), "to (s)": round(e.t_end, 1), "category": e.category,
                                    "peak level": e.level_peak, "p": round(e.p_peak, 2), "KG evidence": round(e.kg_evidence_peak, 2),
                                    "cautious": e.cautious, "clip": e.clip_status} for e in evs]), hide_index=True, use_container_width=True)
        st.caption("Open an event in Risk Explanation or Near-Miss Memory.")
