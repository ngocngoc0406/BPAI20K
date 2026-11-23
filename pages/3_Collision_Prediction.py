"""Collision Prediction: probability / risk timeline, levels, early-warning onset, ground truth."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.risk.risk_engine import LEVEL_COLORS
from ui.common import PLOT_LAYOUT, card, ground_truth_for, model_meta, pill, require_result, setup

setup("Collision Prediction", "📈")
st.title("📈 Collision Prediction")
res = require_result()
meta = model_meta() or {}
tl = res.timeline
rc = meta.get("risk", {})
gt = ground_truth_for(res.video_path)

s = res.summary()
c = st.columns(5)
card(c[0], "Risk category (max)", pill(s["max_level"]))
card(c[1], "Max P(collision)", f"{s['max_probability']:.2f}", "calibrated, ensemble mean")
card(c[2], "Early-warning onset", "none" if not res.alarms else f"{res.alarms[0]:.1f} s",
     f"≥ {rc.get('consecutive_steps', 3)} consecutive steps ≥ WARNING")
card(c[3], "Observation window", f"{meta.get('window_s', '?')} s", "causal — past frames only")
card(c[4], "Prediction horizon", "0.3 – 2.0 s", "trained to fire this far before an event")
st.write("")

if gt:
    if gt["label"] == 1:
        lead = (gt["time_of_event"] - res.alarms[0]) if res.alarms and res.alarms[0] < gt["time_of_event"] else None
        st.info(f"**Ground truth (Nexar annotation):** collision / near-miss at **{gt['time_of_event']:.2f} s**; "
                f"annotated earliest alert time **{gt['time_of_alert']:.2f} s**. "
                + (f"SafeDrive warned **{lead:.2f} s before the event**." if lead is not None else
                   "SafeDrive did not raise an alarm before the event."), icon="📌")
    else:
        st.info(f"**Ground truth (Nexar annotation):** normal driving (no event). Alarm onsets here are false alarms: "
                f"**{len(res.alarms)}**.", icon="📌")

fig = go.Figure()
fig.add_trace(go.Scatter(x=tl.time_s, y=(tl.p_mean + tl.p_std).clip(upper=1), line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=tl.time_s, y=(tl.p_mean - tl.p_std).clip(lower=0), fill="tonexty", fillcolor="rgba(34,211,238,.15)",
                         line=dict(width=0), name="±1 sd (MC/ensemble)"))
fig.add_trace(go.Scatter(x=tl.time_s, y=tl.p_mean, name="P(collision)", line=dict(color="#22d3ee", width=2.4)))
fig.add_trace(go.Scatter(x=tl.time_s, y=tl.risk_score, name="risk score", line=dict(color="#f97316", width=1.6, dash="dot")))
for name, key in [("CAUTION", "caution"), ("WARNING", "warning"), ("HIGH RISK", "high")]:
    if key in rc:
        fig.add_hline(y=rc[key], line=dict(color=LEVEL_COLORS[name], width=1, dash="dash"),
                      annotation_text=name, annotation_position="right", annotation_font_color=LEVEL_COLORS[name])
for a in res.alarms:
    fig.add_vline(x=a, line=dict(color="#ef4444", width=2), annotation_text="ALARM", annotation_font_color="#ef4444")
if gt and gt["label"] == 1:
    fig.add_vline(x=gt["time_of_event"], line=dict(color="#e5e9f5", width=2), annotation_text="EVENT (GT)")
    if gt["time_of_alert"] is not None:
        fig.add_vrect(x0=gt["time_of_alert"], x1=gt["time_of_event"], fillcolor="rgba(239,68,68,.10)", line_width=0,
                      annotation_text="GT alert window", annotation_position="top left")
fig.update_layout(**PLOT_LAYOUT, height=420, yaxis=dict(range=[0, 1.02], title="score"), xaxis=dict(title="time (s)"),
                  title="Collision probability and risk over time")
st.plotly_chart(fig, use_container_width=True)

# level strip
strip = go.Figure()
for lvl, col in LEVEL_COLORS.items():
    m = tl.level == lvl
    if m.any():
        strip.add_trace(go.Bar(x=tl.time_s[m], y=[1] * int(m.sum()), marker_color=col, name=lvl, width=0.2,
                               hovertext=tl.explanation[m], hoverinfo="text+x"))
strip.update_layout(**{**PLOT_LAYOUT, "margin": dict(l=10, r=10, t=30, b=10)}, height=150, barmode="stack",
                    yaxis=dict(visible=False), xaxis=dict(title="time (s)"), title="Risk timeline (hover for reasons)", bargap=0)
st.plotly_chart(strip, use_container_width=True)

st.subheader("Timeline details")
lvl = st.multiselect("Show levels", list(LEVEL_COLORS), default=["CAUTION", "WARNING", "HIGH RISK", "UNCERTAIN"])
view = tl[tl.level.isin(lvl)][["time_s", "level", "p_mean", "risk_score", "mutual_info", "trend", "factors", "explanation"]]
st.dataframe(view.round(3), hide_index=True, use_container_width=True, height=320)
if view.empty:
    st.caption("No moments at the selected levels.")
st.caption("risk score = calibrated probability + positive 1-s trend + image-plane looming cue (weights and thresholds "
           "fixed on the validation split). It is not the same as the collision probability.")
_ = pd
