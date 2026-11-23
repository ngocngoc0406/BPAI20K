"""Dataset Explorer: statistics, label distribution, event timing, annotated sample videos."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.dataset_loader import index_videos
from src.preprocessing.frame_extractor import thumbnails
from src.utils.config import resolve
from src.utils.io import load_json
from ui.common import PLOT_LAYOUT, annotations, card, dataset_stats, setup

cfg = setup("Dataset Explorer", "🗂️")
st.title("🗂️ Dataset Explorer")
stats, ann = dataset_stats(), annotations()
if not stats or ann is None:
    st.warning("Dataset not prepared. Run `scripts/download_nexar.py` then `scripts/prepare_dataset.py`.")
    st.stop()
st.markdown("**Nexar Dashcam Collision Prediction** — real dashcam videos (1280×720, 30 fps, ~40 s) of collisions / "
            "near-misses and normal driving. Licence: Nexar Open Data License (attribution; no re-identification, "
            "no resale). Local copy: balanced random subset of the training split.")
a = stats["all"]
c = st.columns(5)
card(c[0], "Videos", str(a["n_videos"]))
card(c[1], "Positive", str(a["n_positive"]), "collision / near-miss")
card(c[2], "Negative", str(a["n_negative"]), "normal driving")
card(c[3], "Mean event time", f"{a.get('time_of_event_s', {}).get('mean', float('nan')):.1f} s")
card(c[4], "Mean alert lead", f"{a.get('alert_lead_s', {}).get('mean', float('nan')):.2f} s", "event − annotated alert")
st.write("")

splits = load_json(resolve(cfg["paths"]["splits"]))
vids = index_videos(resolve(cfg["paths"]["raw_dir"]))
df = ann[ann.video_id.isin(vids)].copy()
df["split"] = df.video_id.map({v: k for k, ids in splits.items() for v in ids}).fillna("excluded")
df["alert_lead_s"] = df.time_of_event - df.time_of_alert

l, r = st.columns(2)
with l:
    ct = df.groupby(["split", "label"]).size().unstack(fill_value=0).reindex(["train", "val", "test", "excluded"]).dropna(how="all")
    fig = go.Figure([go.Bar(x=ct.index, y=ct.get(0, 0), name="normal (0)", marker_color="#22c55e"),
                     go.Bar(x=ct.index, y=ct.get(1, 0), name="collision/near-miss (1)", marker_color="#ef4444")])
    fig.update_layout(**PLOT_LAYOUT, barmode="stack", height=330, title="Label distribution per split (video-level)")
    st.plotly_chart(fig, use_container_width=True)
with r:
    pos = df[df.label == 1]
    fig = go.Figure([go.Histogram(x=pos.time_of_event, name="time of event", marker_color="#ef4444", opacity=.7, nbinsx=25),
                     go.Histogram(x=pos.time_of_alert, name="time of alert", marker_color="#eab308", opacity=.7, nbinsx=25)])
    fig.update_layout(**PLOT_LAYOUT, barmode="overlay", height=330, title="Event / alert timestamps (positives)", xaxis_title="s")
    st.plotly_chart(fig, use_container_width=True)
fig = go.Figure(go.Histogram(x=pos.alert_lead_s, marker_color="#22d3ee", nbinsx=30))
fig.update_layout(**PLOT_LAYOUT, height=260, title="Annotated alert lead time = time_of_event − time_of_alert", xaxis_title="s")
st.plotly_chart(fig, use_container_width=True)
for k in ("light_conditions", "weather", "scene"):
    if k in a:
        st.markdown(f"**{k}**: " + ", ".join(f"{kk}: {vv}" for kk, vv in a[k].items()))

st.subheader("Browse videos")
f1, f2 = st.columns(2)
sp = f1.multiselect("Split", ["train", "val", "test"], default=["test"])
lb = f2.multiselect("Label", [0, 1], default=[0, 1])
view = df[df.split.isin(sp) & df.label.isin(lb)]
st.dataframe(view.drop(columns=[c for c in view.columns if c == "path"]).round(2), hide_index=True,
             use_container_width=True, height=260)
pick = st.selectbox("Preview", ["—"] + view.video_id.tolist())
if pick != "—":
    row = view[view.video_id == pick].iloc[0]
    p = vids[pick]
    if row.label == 1:
        st.info(f"Annotation: alert at **{row.time_of_alert:.2f} s**, event at **{row.time_of_event:.2f} s**.")
    st.video(str(p), start_time=int(max((row.time_of_event if row.label == 1 else 0) - 6, 0)))
    th = thumbnails(p, 6)
    cols = st.columns(6)
    for col, (t, im) in zip(cols, th):
        col.image(im, caption=f"{t:.1f} s", use_container_width=True)
_ = pd
