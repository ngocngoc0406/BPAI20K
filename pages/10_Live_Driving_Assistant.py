"""Live Driving Assistant (v3.2): camera or demo video -> live detection / tracking / risk level, audio warnings,
event recording with a rolling pre-event buffer, and driving sessions. Research prototype: no braking, no steering."""
from __future__ import annotations

from pathlib import Path

import cv2
import streamlit as st

from src.live.audio import AudioConfig
from src.live.capture import DeviceBusyError, active_devices
from src.live.controller import LiveController
from src.live.recorder import RecorderConfig
from src.live.sources import CameraSource, SourceError, VideoFileSource, list_cameras
from src.risk.risk_engine import LEVEL_COLORS
from src.utils.config import resolve
from src.utils.io import load_json
from ui.common import available_models, card, pill, setup

setup("Live Driving Assistant", "🚘")
st.title("🚘 Live Driving Assistant")
st.warning("**Research prototype.** Warnings are model estimates from image-plane cues. They can be missed or false "
           "(held-out: about 1 in 9 collisions warned at about 9 false alarms per hour with the default model). "
           "The system does not brake or steer and does not guarantee collision avoidance. Keep your eyes on the road.", icon="⚠️")


@st.cache_resource
def controller() -> LiveController:
    return LiveController()  # one per server process -> one camera stream


@st.cache_data(ttl=60)
def cameras() -> list[dict]:
    return list_cameras()


ctl = controller()
running = ctl.running

# ---------------- configuration ----------------
with st.expander("Settings", expanded=not running):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Source**")
        mode = st.radio("Input", ["Camera", "Demo video"], horizontal=True, disabled=running, key="live_mode")
        if mode == "Camera":
            cams = cameras() if not running else []
            if not cams and not running:
                st.error("No camera detected by the operating system (or it is in use by another application). "
                         "Use **Demo video** instead.")
            cam_idx = st.selectbox("Camera", [c["index"] for c in cams] or [0], disabled=running,
                                   format_func=lambda i: f"Camera {i}")
        else:
            sp = resolve("data/processed/splits_full.json")
            demo_ids = load_json(sp)["test"] if sp.exists() else []
            from src.data.dataset_loader import index_videos
            vids = index_videos(resolve("data/raw/nexar"))
            options = [str(vids[v]) for v in demo_ids if v in vids] + [str(p) for p in sorted(resolve("data/uploads").glob("*.mp4"))]
            demo = st.selectbox("Demo video (held-out test clips + uploads)", options, disabled=running,
                                format_func=lambda p: Path(p).name)
            loop = st.checkbox("Loop", value=True, disabled=running)
    with c2:
        st.markdown("**Model**")
        models = available_models()
        mname = st.selectbox("Prediction model", list(models), disabled=running, key="live_model")
        meta = load_json(models[mname] / "bundle.json") if models else {}
        profs = meta.get("risk", {}).get("profiles")
        profile = st.selectbox("Warning sensitivity", ["conservative", "balanced", "sensitive"], index=1, disabled=running or not profs,
                               help=None if profs else "This model has a single alarm threshold.")
        device = st.selectbox("Device", ["auto", "cpu"], disabled=running, help="cpu forces CPU-only inference")
    with c3:
        st.markdown("**Recording & audio**")
        rec_on = st.checkbox("Record events (clip + metadata)", value=True, disabled=running)
        pre_s = st.slider("Pre-event (s)", 1, 15, 5, disabled=running)
        post_s = st.slider("Post-event (s)", 1, 15, 5, disabled=running)
        quota = st.number_input("Clip storage quota (MB)", 100, 50000, 2048, step=100, disabled=running)
        risk_ev = st.checkbox("Also record risk events (CAUTION episodes)", value=True, disabled=running)
        audio_on = st.checkbox("Audio warnings", value=True)
        volume = st.slider("Volume", 0.0, 1.0, 0.6, 0.05)
        caution_snd = st.checkbox("Sound for CAUTION", value=True)
        if st.button("🔊 Test sound (WARNING tone)"):
            from src.live.audio import AudioAlerter
            tst = AudioAlerter(AudioConfig(enabled=True, volume=float(volume)))
            r = tst.alert("WARNING", lead_in_corridor=True)
            if r["played"]:
                st.success(f"Played: “{r['message']}” — if you heard nothing, check the Windows output device / volume.")
            else:
                st.error(f"Audio not available: {r['reason']}")
with st.expander("Trip / session metadata (v3.3 real-world validation) — fill in BEFORE driving", expanded=False):
    from src.realworld import trip_meta as tm
    st.caption("Conditions are **declared by you**, never inferred by the software; leave *unknown* if unsure. "
               "Real-driving and controlled sessions require consent, a supervisor and setup before the vehicle moves. "
               "Do not operate the screen while driving. This prototype is never the only safety aid; do not create dangerous situations.")
    m1, m2, m3 = st.columns(3)
    kind = m1.selectbox("Data kind", list(tm.DATA_KINDS), index=2, format_func=lambda k: tm.DATA_KINDS[k], key="tm_kind")
    lighting = m1.selectbox("Lighting", tm.LIGHTING, key="tm_light")
    weather = m1.selectbox("Weather", tm.WEATHER, key="tm_weather")
    roads = m2.multiselect("Road types", tm.ROAD, key="tm_road")
    traffic = m2.selectbox("Traffic density", tm.TRAFFIC, key="tm_traffic")
    mount = m2.selectbox("Camera mount", tm.MOUNTS, key="tm_mount")
    consent = m3.checkbox("Driver consents to recording", key="tm_consent")
    informed = m3.checkbox("All occupants informed", key="tm_informed")
    superv = m3.checkbox("Supervisor / second person present", key="tm_sup")
    before = m3.checkbox("Configured before the vehicle moves", key="tm_before")
    fullvid = st.checkbox("Record the FULL session video for annotation (local only; people blurred; licence plates are NOT blurred)",
                          key="tm_fullvid")
    blur = st.checkbox("Blur detected people in the full video", value=True, key="tm_blur")
    notes = st.text_input("Notes (no precise personal locations)", key="tm_notes")
    k1, k2, k3 = st.columns(3)
    camp = k1.text_input("Campaign id (v3.4)", value="v34-realworld", key="tm_camp")
    trip = k2.text_input("Trip id (one drive; same id for its sessions)", key="tm_trip")
    drv = k3.text_input("Driver pseudonym (e.g. D01, never a name)", key="tm_drv")
    regs = st.checkbox("Local recording rules were checked and are respected", key="tm_regs")
    share = st.checkbox("Driver allows sharing a privacy-reviewed BLURRED copy within the research team", key="tm_share")
    st.session_state["trip_meta"] = tm.TripMeta("pending", data_kind=kind, driver_consent=consent, occupants_informed=informed,
                                                supervisor_present=superv, configured_before_driving=before, lighting=lighting,
                                                weather=weather, road_types=roads, traffic=traffic, camera_mount=mount, notes=notes,
                                                record_full_video=fullvid, blur_people=blur, campaign_id=camp.strip(),
                                                trip_id=trip.strip(), driver_id=drv.strip(), regulations_checked=regs,
                                                share_consent=share)
if ctl.audio is not None:  # audio settings can change while running
    ctl.audio.cfg.enabled, ctl.audio.cfg.volume, ctl.audio.cfg.sound_caution = audio_on, float(volume), caution_snd

# ---------------- start / stop ----------------
b1, b2, b3, b4 = st.columns(4)
if b1.button("▶ Start camera", type="primary", disabled=running, use_container_width=True):
    try:
        src = CameraSource(int(cam_idx)) if mode == "Camera" else VideoFileSource(demo, realtime=True, loop=loop)
        with st.spinner("Loading models and opening the source…"):
            ctl.start(src, models[mname], profile, None if device == "auto" else "cpu",
                      RecorderConfig(pre_s=pre_s, post_s=post_s, max_store_mb=float(quota), record_risk_events=risk_ev),
                      AudioConfig(enabled=audio_on, volume=float(volume), sound_caution=caution_snd), record_events=rec_on)
        st.rerun()
    except (SourceError, DeviceBusyError) as e:
        st.error(f"Source not available: {e}  \nTip: close other apps using the camera, or switch to **Demo video**.")
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not start: {e}")
if b2.button("⏹ Stop camera", disabled=not running, use_container_width=True):
    rep = ctl.stop()
    if rep:
        st.session_state["last_session"] = rep
    st.rerun()
if b3.button("● Start session", disabled=not running or ctl.session is not None, use_container_width=True):
    from src.realworld.trip_meta import ConsentError
    try:
        ctl.start_session(st.session_state.get("trip_meta"))
        st.rerun()
    except (ConsentError, ValueError) as e:
        st.error(f"Session not started: {e}")
if b4.button("■ Stop session", disabled=ctl.session is None, use_container_width=True):
    st.session_state["last_session"] = ctl.stop_session()
    st.rerun()


# ---------------- live view (refreshed at most 4x per second) ----------------
@st.fragment(run_every=0.25)
def live_view() -> None:
    s, frame = ctl.snapshot()
    if not ctl.running and not s.get("error"):
        st.info("Stopped. Choose a source and press **Start camera**. Active devices: " + (", ".join(active_devices()) or "none"))
        return
    if s.get("error"):
        st.error(s["error"])
    if (s.get("stalled_s") or 0) > 2.0:
        st.warning(f"⏸ {s.get('status')}. The displayed level is not current.")
    left, right = st.columns([3, 2])
    with left:
        if frame is not None:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.caption(s.get("status", "starting…"))
    with right:
        lvl = s.get("level", "NORMAL")
        if s.get("degraded"):
            iv = s.get("input_interval_s") or 0
            st.error(f"**DEGRADED INPUT**: frames arrive every {iv:.2f} s, but the models need 0.1 s steps "
                     f"(≈{1 / max(iv, 1e-6):.1f} fps). Warnings and event recording are suspended. "
                     "Another application may be using the camera, or the computer is overloaded.")
            lvl = "UNCERTAIN"
        st.markdown(f"<div class='sd-card' style='border-color:{LEVEL_COLORS.get(lvl, '#64748b')}'>"
                    f"<div class='k'>Current level</div><div class='v'>{pill(lvl)}</div>"
                    f"<div class='s'>P(collision) {s.get('p', 0):.2f} · uncertainty (MI) {s.get('mi', 0):.3f}</div></div>",
                    unsafe_allow_html=True)
        a = s.get("last_audio")
        if a and a.get("message"):
            st.markdown(f"🔊 **{a['message']}**  \n<small>{a['level']} · {'played' if a['played'] else a['reason']}</small>",
                        unsafe_allow_html=True)
        r = st.columns(2)
        card(r[0], "Capture FPS", f"{s.get('capture_fps', 0):.1f}", s.get("camera", ""))
        card(r[1], "Objects", str(s.get("objects", 0)), f"step {s.get('step', 0)}")
        r = st.columns(2)
        card(r[0], "Step time", f"{s.get('step_ms', 0):.0f} ms", "detect+features+predict+render")
        card(r[1], "End-to-end", f"{s.get('e2e_ms', 0):.0f} ms", "capture → annotated frame")
        st.caption(f"Model {s.get('model')} · profile {s.get('profile')} · device {s.get('device')} · audio: {s.get('audio')} · "
                   f"camera interruptions {s.get('interruptions', 0)} · events being recorded {s.get('pending_events', 0)}")
        for err in s.get("recorder_errors", []):
            st.error(f"Recording error: {err}")
        if ctl.session is not None:
            ss = ctl.session.summary()
            st.markdown(f"**Session {ss['session_id']}** · {ss['duration_s'] / 60:.1f} min · alarms {ss['alarms']} · "
                        f"events {ss['events_total']} ({ss['events']})")
        else:
            st.caption("No session running: events are still recorded, but statistics and the session report need **Start session**.")


live_view()

rep = st.session_state.get("last_session")
if rep:
    st.subheader("Last session report")
    sm = rep["summary"]
    st.json({k: sm[k] for k in ("session_id", "duration_s", "alarms", "level_onsets", "events", "clips_saved", "clips_failed",
                                "processed_steps_per_s", "e2e_ms_p50", "e2e_ms_p95", "camera_interruptions")}, expanded=False)
    p = rep["paths"]
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Session report (PDF)", Path(p["pdf"]).read_bytes(), Path(p["pdf"]).name, "application/pdf", use_container_width=True)
    d2.download_button("⬇ Session report (JSON)", Path(p["json"]).read_bytes(), Path(p["json"]).name, "application/json", use_container_width=True)
st.caption("Recorded events appear in **Near-Miss Memory** (trip `live-<session id>`). Everything stays on this computer; nothing is uploaded.")
