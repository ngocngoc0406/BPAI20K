from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from safedrive_v4.app.ui import draw, frame_at, level_badge, page, pick_trip, replay_graph, stores  # noqa: E402

page("Risk Explanation", "🔍")
st.title("🔍 Risk Explanation")
st.caption("Every element below comes from stored pipeline data (graph, rules, model). No language model writes causes.")
ev, ts = stores()
tid = pick_trip(ts)
if not tid:
    st.stop()
traces = ts.traces(tid)
if not traces:
    st.info("No warning in this trip.")
    st.stop()
meta = ts.meta(tid)
key = st.selectbox("Warning", list(traces), format_func=lambda k: f"{traces[k].timestamp:.1f} s · {traces[k].warning['level']}"
                   f"{' (cautious)' if traces[k].warning.get('cautious') else ''} · p={traces[k].prediction['risk_probability']:.2f}")
tr = traces[key]
v = tr.verify()
st.markdown(level_badge(tr.warning["level"], tr.warning.get("cautious", False)) + f" at **{tr.timestamp:.1f} s** · "
            f"trace integrity: {'✅ verified' if v['ok'] else '❌ ' + str(v)}")
st.write(" · ".join(tr.warning.get("reasons", [])))
snap, res = replay_graph(tid, tr.step)
a, b = st.columns(2)
img = frame_at(meta.get("source"), tr.timestamp)
if img is not None and snap is not None:
    a.image(draw(img, snap, {f"obj:{i}" for i in tr.track_ids}), caption=f"evidence frame at {tr.timestamp:.1f} s (frames {tr.frame_refs[0]}–{tr.frame_refs[-1]})",
            use_container_width=True)
with b:
    st.subheader("Road users involved")
    rows = [{"road user": k, "type": n.get("node_type"), "confidence": n.get("confidence"), "visibility": n.get("visibility_status"),
             "image TTC proxy (s)": n.get("tau_s"), "track age (steps)": n.get("track_age_steps")}
            for k, n in tr.nodes.items() if k != "ego"]
    if rows:
        st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True, use_container_width=True)
    else:
        st.write("None (no rule fired).")
st.subheader("Reasoning path (knowledge graph)")
if tr.kg_paths:
    for p in tr.kg_paths:
        st.code(" → ".join(p), language=None)
    for f in tr.findings:
        op = tr.uncertainty.get("finding_opinions", {}).get(f"{f['rule_id']}:{f['subjects'][0]}", {})
        st.markdown(f"- **{f['rule_id']}** {f['explanation']} · belief {op.get('belief', 0):.2f}, uncertainty {op.get('uncertainty', 1):.2f}"
                    + (f" · exceptions: {'; '.join(f['exceptions_applied'])}" if f["exceptions_applied"] else ""))
else:
    st.write("No knowledge-graph rule fired at this warning: the model's warning is explained only by feature contributions below.")
st.subheader("Contributing factors (exact Tree SHAP)")
sh = pd.DataFrame(tr.shap)
if len(sh):
    st.markdown("**By feature group** (stable across ensemble members)")
    st.dataframe(sh[sh["level"].str.startswith("group")].drop(columns=["level"]), hide_index=True, use_container_width=True)
    with st.expander("Individual features (indicative only: unstable ranking across ensemble members)"):
        st.dataframe(sh[sh["level"].str.startswith("feature")].drop(columns=["level"]), hide_index=True, use_container_width=True)
else:
    st.write("Not available (live session: SHAP is computed only in video analysis).")
st.subheader("Uncertainty")
u = tr.uncertainty
c1, c2, c3 = st.columns(3)
c1.metric("Model uncertainty (mutual information)", f"{u.get('mutual_info', 0):.3f}")
c2.metric("Perception reliability", f"{u.get('perception_reliability', {}).get('score', 0):.2f}")
kgo = u.get("kg_risk_opinion_temporal") or {}
c3.metric("KG risk evidence (belief / uncertainty)", f"{kgo.get('belief', 0):.2f} / {kgo.get('uncertainty', 1):.2f}")
st.subheader("What the system could not determine")
for x in tr.unknowns:
    st.write("• " + x)
if not meta.get("video_id", "").startswith("live") and tr.track_ids:
    st.subheader("What if? (physically consistent counterfactuals)")
    if st.button("Compute (recomputes every feature from edited trajectories; a few seconds)"):
        import safedrive_v4.evaluation.xai_eval as xe
        xe._KIND, xe._MODEL = "a3", None
        from safedrive_v4.config import resolve
        if resolve(f"data/processed/features/clean/{meta.get('video_id')}.npz").exists():
            st.dataframe(pd.DataFrame(xe.counterfactual_suite(meta["video_id"], tr.timestamp, tr.track_ids[:1])), hide_index=True, use_container_width=True)
        else:
            st.info("Counterfactuals need the v3 feature cache of the clip (available for Nexar clips).")
with st.expander("Full evidence trace (sealed JSON)"):
    st.json(tr.__dict__)
