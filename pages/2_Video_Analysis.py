"""Video Analysis: upload, process (detection + tracking + prediction), annotated playback, frame inspector."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from src.detection.detector import COCO_ROAD
from src.preprocessing.video_reader import VideoReadError, get_meta
from src.utils.config import resolve
from src.visualization.overlay import draw_frame
from ui.common import card, model_meta, pill, setup

cfg = setup("Video Analysis", "🎬")
st.title("🎬 Video Analysis")

if model_meta() is None:
    st.error("No trained model found in `models/safedrive_v1`. Train it first: `python scripts/run_experiments.py`.")
    st.stop()

MAX_MB, MAX_S = 300, 120.0
uploads = resolve(cfg["paths"]["uploads_dir"])
outputs = resolve(cfg["paths"]["outputs_dir"])

src_tab, ds_tab = st.tabs(["⬆️ Upload a video", "🗂️ Dataset test sample"])
video_path: Path | None = None
with src_tab:
    up = st.file_uploader(f"Dashcam video (mp4/avi/mov/mkv, ≤ {MAX_MB} MB; first {int(MAX_S)} s analysed)",
                          type=["mp4", "avi", "mov", "mkv"])
    if up is not None:
        data = up.getvalue()
        if len(data) > MAX_MB * 1e6:
            st.error("File too large.")
        else:
            h = hashlib.sha1(data).hexdigest()[:10]
            video_path = uploads / f"{h}__{Path(up.name).name}"
            if not video_path.exists():
                uploads.mkdir(parents=True, exist_ok=True)
                video_path.write_bytes(data)
with ds_tab:
    from src.utils.io import load_json
    sp = resolve(cfg["paths"]["splits"])
    if sp.exists():
        from ui.common import annotations
        ann = annotations()
        test_ids = load_json(sp)["test"]
        df = ann[ann.video_id.isin(test_ids)].copy() if ann is not None else pd.DataFrame()
        if len(df):
            df["desc"] = df.apply(lambda r: f"{r.video_id} · {'collision/near-miss' if r.label else 'normal'}"
                                  + (f" · event {r.time_of_event:.1f}s" if r.label else ""), axis=1)
            choice = st.selectbox("Held-out TEST video (never seen in training)", ["—"] + df.desc.tolist())
            if choice != "—":
                from src.data.dataset_loader import index_videos
                vid = choice.split(" · ")[0]
                video_path = index_videos(resolve(cfg["paths"]["raw_dir"])).get(vid)
    else:
        st.caption("Dataset not prepared.")

if video_path is None:
    st.info("Upload a clip or choose a dataset sample to begin.", icon="🎬")
    st.stop()

try:
    meta = get_meta(video_path)
except VideoReadError as e:
    st.error(f"Cannot read this video: {e}")
    st.stop()

c = st.columns(4)
card(c[0], "File", Path(video_path).name.split("__")[-1][:28])
card(c[1], "Duration", f"{meta.duration_s:.1f} s", f"{meta.n_frames} frames")
card(c[2], "Source FPS", f"{meta.fps:.1f}", "resampled to 10 Hz")
card(c[3], "Resolution", f"{meta.width}×{meta.height}")
if meta.duration_s > MAX_S:
    st.warning(f"Video is longer than {int(MAX_S)} s — only the first {int(MAX_S)} s will be analysed.")
st.write("")

key = str(video_path)
cur = st.session_state.get("result")
already = cur is not None and cur.video_path == key
from datetime import date  # noqa: E402
mc1, mc2 = st.columns(2)
save_mem = mc1.checkbox("Save safety events to Near-Miss Memory", value=True,
                        help="Alerts and near-miss candidates are stored with clip, measured cues and explanation.")
trip_id = mc2.text_input("Trip name", value=st.session_state.get("trip_id", f"trip-{date.today().isoformat()}"))
st.session_state["trip_id"] = trip_id
run = st.button("▶ Analyse video" if not already else "↻ Re-analyse", type="primary")
st.caption("Press **Stop** (top-right) or any other control to interrupt a running analysis.")

if run:
    from ui.common import get_pipeline
    pipe = get_pipeline()
    bar = st.progress(0.0, text="Starting…")
    t0 = time.perf_counter()
    try:
        res = pipe.analyze(video_path, progress=lambda f, m: bar.progress(min(f, 1.0), text=f"Detect · track · embed — {m}"),
                           max_duration_s=MAX_S, profile=st.session_state.get("warning_profile"))
        bar.progress(1.0, text="Rendering annotated video…")
        out_mp4 = outputs / f"{Path(video_path).stem}_annotated.mp4"
        pipe.render(res, out_mp4)
        st.session_state["result"] = res
        st.session_state["annotated"] = str(out_mp4)
        bar.empty()
        st.success(f"Analysis finished in {time.perf_counter() - t0:.1f} s on {pipe.device.upper()} "
                   f"(warning profile: {res.profile}, alarm threshold {res.alarm_threshold:.3f}).")
        if save_mem and trip_id.strip():
            from src.safety.near_miss import NearMissMemory
            from src.safety.trip_report import TripLog
            evs = NearMissMemory().add(pipe.safety_events(res, trip_id.strip()), res)
            TripLog().add_clip(trip_id.strip(), res)
            st.info(f"Near-Miss Memory: {sum(e.type == 'alert' for e in evs)} alert(s) and "
                    f"{sum(e.type == 'near_miss_candidate' for e in evs)} near-miss candidate(s) stored in trip '{trip_id.strip()}'.", icon="🗂️")
        already = True
    except VideoReadError as e:
        bar.empty()
        st.error(f"Video could not be decoded: {e}")
    except RuntimeError as e:
        bar.empty()
        st.error(f"Processing failed: {e}")

if not already:
    st.video(str(video_path))
    st.stop()

res = st.session_state["result"]
s = res.summary()
c = st.columns(5)
card(c[0], "Highest level", pill(s["max_level"]))
card(c[1], "Max P(collision)", f"{s['max_probability']:.2f}")
card(c[2], "First alarm", "none" if s["first_alarm_s"] is None else f"{s['first_alarm_s']:.1f} s", f"{s['n_alarms']} onset(s)")
card(c[3], "Tracks", str(s["tracking"]["n_tracks"]), f"median length {s['tracking'].get('median_length', 0):.0f} steps")
card(c[4], "Throughput", f"{s['timings'].get('extract_fps') or 0:.1f} fps", f"{s['timings']['latency_ms_per_window']:.1f} ms / window")
st.write("")

vcol, fcol = st.columns([3, 2])
with vcol:
    st.subheader("Annotated video")
    ann_path = st.session_state.get("annotated")
    if ann_path and Path(ann_path).exists():
        st.video(ann_path)
    else:
        st.video(str(video_path))
with fcol:
    st.subheader("Frame inspector")
    f = res.features
    step = st.slider("Time (s)", 0.0, float(f.times[-1]), float(s["first_alarm_s"] or 0.0), 0.1)
    i = int(min(round(step * f.fps), len(f) - 1))
    rows = f.tracks[f.tracks[:, 0] == i]
    tl = res.timeline
    j = max(int((tl.step <= i).sum()) - 1, 0)
    r = tl.iloc[j] if len(tl) else None
    img = draw_frame(res.frames[i], rows, r.level if r is not None else None, r.risk_score if r is not None else None,
                     r.p_mean if r is not None else None, float(f.times[i]))
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
    if len(rows):
        st.dataframe(pd.DataFrame({"ID": rows[:, 1].astype(int), "class": [COCO_ROAD.get(int(x), int(x)) for x in rows[:, 2]],
                                   "conf": rows[:, 3].round(2), "x1": rows[:, 4].round(3), "y1": rows[:, 5].round(3),
                                   "x2": rows[:, 6].round(3), "y2": rows[:, 7].round(3)}),
                     hide_index=True, use_container_width=True, height=180)
    else:
        st.caption("No road users detected in this frame.")

st.subheader("Detections over time")
tr = pd.DataFrame(f.tracks, columns=["step", "id", "cls", "conf", "x1", "y1", "x2", "y2"])
if len(tr):
    tr["class"] = tr.cls.astype(int).map(COCO_ROAD)
    tr["time_s"] = tr.step / f.fps
    cnt = tr.groupby(["time_s", "class"]).size().unstack(fill_value=0)
    st.area_chart(cnt, height=200)
else:
    st.caption("No detections in this video.")
