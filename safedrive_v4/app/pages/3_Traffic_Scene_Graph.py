from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from safedrive_v4.app.ui import draw, frame_at, graph_figure, page, pick_trip, replay_graph, stores  # noqa: E402

page("Traffic Scene Graph", "🕸️")
st.title("🕸️ Traffic Scene Graph")
st.caption("Which road users interact with your car, since when, how the evidence evolves. Relations are emitted only with "
           "evidence (persistent in >= 2 of 3 steps); each carries a subjective-logic opinion (belief / uncertainty).")
ev, ts = stores()
tid = pick_trip(ts)
if not tid:
    st.stop()
meta, tl = ts.meta(tid), pd.DataFrame(ts.timeline(tid))
t = st.slider("Time (s)", 0.0, float(tl["t"].max()), float(tl.loc[tl["p"].idxmax(), "t"]), 0.1)
step = int(round(t * 10))
snap, res = replay_graph(tid, step)
if snap is None:
    st.warning("No replay data for this trip.")
    st.stop()
hl = {f.subjects[0] for f in res.base.findings if f.kind == "risk" and f.subjects}
a, b = st.columns(2)
img = frame_at(meta.get("source"), t)
if img is not None:
    a.image(draw(img, snap, hl), caption="red = road users in risk findings · yellow = occluded / new", use_container_width=True)
b.plotly_chart(graph_figure(snap, hl), use_container_width=True)
ego_edges = [e for e in snap.edges if "ego" in (e.source_node, e.target_node) and e.relation_type != "LOCATED_IN"]
st.subheader("Road users interacting with your car")
if ego_edges:
    st.dataframe(pd.DataFrame([{"road user": e.source_node if e.target_node == "ego" else e.target_node,
                                "type": snap.nodes[e.source_node if e.target_node == "ego" else e.target_node].node_type,
                                "relation": e.relation_type, "since (s)": round(e.temporal_validity.get("first_seen", 0), 1),
                                "edge confidence": round(e.confidence, 2), "belief": e.opinion.get("belief") if e.opinion else None,
                                "uncertainty": e.uncertainty, "evidence": {k: round(v, 3) for k, v in e.evidence_features.items()}}
                               for e in ego_edges]).astype(str), hide_index=True, use_container_width=True)
else:
    st.write("No relation with the ego vehicle at this time.")
st.subheader("Is the situation getting worse?")
w = tl[(tl["t"] >= t - 3) & (tl["t"] <= t + 0.01)]
if len(w) > 2:
    st.line_chart(w.set_index("t")[["p", "kg_risk_expectation", "kg_risk_uncertainty"]], height=180)
    trend = w["kg_risk_expectation"].iloc[-1] - w["kg_risk_expectation"].iloc[0]
    st.write(f"Knowledge-graph risk evidence over the last 3 s: {'rising' if trend > 0.05 else ('falling' if trend < -0.05 else 'stable')} "
             f"({w['kg_risk_expectation'].iloc[0]:.2f} → {w['kg_risk_expectation'].iloc[-1]:.2f}); uncertainty now "
             f"{w['kg_risk_uncertainty'].iloc[-1]:.2f}.")
