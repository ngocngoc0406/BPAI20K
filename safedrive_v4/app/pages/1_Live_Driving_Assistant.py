from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import cv2  # noqa: E402
import streamlit as st  # noqa: E402

from safedrive_v4.app.ui import level_badge, page, store_banner, stores  # noqa: E402

page("Live Driving Assistant", "🚗")
st.title("🚗 Live Driving Assistant")
st.caption("Set up while PARKED. The camera is opened only when you press Start. Never operate the screen while driving.")
ev, ts = stores()
store_banner(ev)


@st.cache_resource
def controller():
    from safedrive_v4.app.live import V4LiveController
    return V4LiveController(ev)


ctl = controller()
c1, c2, c3 = st.columns([2, 2, 1])
mode = c1.radio("Source", ["Demo video (recorded clip, played in real time)", "Camera"], horizontal=False)
if mode.startswith("Demo"):
    from safedrive_v4.app.core import _video_of
    vid = c2.text_input("Nexar clip id or video path", "00063")
    path = vid if Path(vid).exists() else _video_of(vid)
    kind = "demo_video"
else:
    idx = c2.number_input("Camera index", 0, 9, 0)
    kind = c2.selectbox("Data kind", ["live_camera_test", "real_driving"], help="'real_driving' only for a consented campaign session")
c3.write("")
if not ctl.running:
    if c3.button("▶ Start", type="primary"):
        from src.live.sources import CameraSource, VideoFileSource
        if mode.startswith("Demo"):
            if not path or not Path(path).exists():
                st.error("Video not found.")
                st.stop()
            src = VideoFileSource(path, realtime=True, loop=False)
        else:
            src = CameraSource(int(idx))
        ctl.start(src, trip_name="live" if not mode.startswith("Demo") else f"demo_{Path(path).stem}", data_kind=kind)
        st.rerun()
else:
    if c3.button("■ Stop"):
        s = ctl.stop()
        st.session_state["trip_id"] = s.get("trip_id")
        st.rerun()


@st.fragment(run_every=0.5)
def live_view():
    s, fr = ctl.snapshot()
    a, b = st.columns([3, 2])
    if fr is not None:
        a.image(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), caption=f"t = {s.get('t', 0):.1f} s · boxes and track ids", use_container_width=True)
    else:
        a.info(f"Status: {s.get('status')}" + (f" · {s.get('error')}" if s.get("error") else ""))
    b.markdown("### " + level_badge(s.get("level", "LOW"), s.get("cautious", False)))
    if s.get("status") == "running":
        b.metric("Risk probability (v3 model)", f"{s.get('p', 0):.2f}")
        b.metric("Perception reliability", f"{s.get('reliability', 0):.2f}")
        b.metric("Knowledge-graph risk evidence", f"{s.get('kg_evidence', 0):.2f}")
        b.caption(f"FPS {s.get('fps', 0):.1f} · objects {s.get('n_objects', 0)} · events {s.get('events', 0)}")
        if s.get("input_reasons"):
            b.warning("Input quality: " + "; ".join(s["input_reasons"]))
        if s.get("explanation"):
            b.markdown("**Why (evidence):** " + " | ".join(s["explanation"]))
        with b.expander("Latency (ms)"):
            st.json(s.get("latency_ms", {}))
    elif s.get("status") == "error":
        b.error(s.get("error"))
        with b.expander("details"):
            st.code(s.get("trace", ""))
    elif s.get("trip_id"):
        b.success(f"Session saved as trip {s['trip_id']}: see Near-Miss Memory and Trip Report.")


live_view()
