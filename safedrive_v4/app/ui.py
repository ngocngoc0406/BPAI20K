"""Shared UI helpers of the single SafeDrive AI V4 app (state, drawing, scene-graph figure, wording)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from safedrive_v4.app.core import DISCLAIMER, TripStore
from safedrive_v4.event_memory.store import EventStore

LEVEL_STYLE = {"LOW": ("🟢", "Low risk"), "MODERATE": ("🟠", "Moderate risk"), "HIGH": ("🔴", "High risk"),
               "INSUFFICIENT_EVIDENCE": ("⚪", "Insufficient evidence"), "DEGRADED_INPUT": ("⚫", "Degraded input")}
CATEGORY_TEXT = {"potential_near_miss": "Potential near-miss (needs review)", "risk_event": "Risk event", "uncertain": "Uncertain",
                 "near_miss": "Near-miss (confirmed)", "collision": "Collision (confirmed with evidence)", "normal": "Normal (false alarm)"}


def page(title: str, icon: str = "🛡️") -> None:
    st.set_page_config(page_title=f"SafeDrive AI V4 · {title}", page_icon=icon, layout="wide")
    st.sidebar.caption("SafeDrive AI V4: driving safety copilot (research prototype)")
    st.sidebar.caption("⚠️ " + DISCLAIMER)


@st.cache_resource
def stores() -> tuple[EventStore, TripStore]:
    ev = EventStore()
    return ev, TripStore(ev)


@st.cache_resource
def pipeline():
    from safedrive_v4.pipeline import V3ReadOnlyModel, V4Pipeline
    return V4Pipeline(V3ReadOnlyModel())


def store_banner(ev: EventStore) -> None:
    if not ev.encrypted:
        st.warning("The encrypted session volume (VeraCrypt V:) is not mounted: event clips will NOT be saved (only metadata). "
                   "Mount D:\\SafeDriveVault.hc as V: in VeraCrypt and reload.")


def level_badge(level: str, cautious: bool = False) -> str:
    icon, txt = LEVEL_STYLE.get(level, ("❔", level))
    return f"{icon} **{txt}**" + (" · *low confidence (perception unreliable)*" if cautious else "")


def pick_trip(ts: TripStore, key: str = "trip") -> str | None:
    trips = ts.list()
    if not trips:
        st.info("No trip yet: analyse a video (Video Analysis) or run the Live Driving Assistant.")
        return None
    labels = {t["trip_id"]: f"{t.get('name')} · {t.get('created', '')[:16]} · {t.get('data_kind')} · {t.get('n_events', 0)} events" for t in trips}
    default = st.session_state.get("trip_id")
    ids = list(labels)
    return st.selectbox("Trip", ids, index=ids.index(default) if default in ids else 0, format_func=labels.get, key=key)


def frame_at(video: str | None, t: float, width: int = 640):
    if not video or not Path(video).exists():
        return None
    from src.preprocessing.frame_extractor import resize_width
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, fr = cap.read()
    cap.release()
    return resize_width(fr, width) if ok else None


def draw(img, snap, highlight: set[str]):
    h, w = img.shape[:2]
    out = img.copy()
    cv2.rectangle(out, (int(0.3 * w), int(0.35 * h)), (int(0.7 * w), h - 1), (255, 160, 80), 1)
    for n in snap.nodes.values():
        if n.bounding_box is None:
            continue
        a, b, c, d = n.bounding_box
        col = (60, 60, 255) if n.node_id in highlight else ((0, 200, 255) if n.visibility_status != "visible" else (60, 220, 60))
        cv2.rectangle(out, (int(a * w), int(b * h)), (int(c * w), int(d * h)), col, 2)
        cv2.putText(out, f"{n.node_type} #{n.object_id}", (int(a * w), max(12, int(b * h) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def graph_figure(snap, highlight: set[str]):
    import plotly.graph_objects as go
    pos = {"ego": (0.5, 1.05)}
    for n in snap.nodes.values():
        if n.center_point is not None:
            pos[n.node_id] = tuple(n.center_point)
    fig = go.Figure()
    for e in snap.edges:
        if e.relation_type == "LOCATED_IN" or e.source_node not in pos or e.target_node not in pos:
            continue
        (xa, ya), (xb, yb) = pos[e.source_node], pos[e.target_node]
        hot = e.source_node in highlight or e.target_node in highlight
        op = e.opinion or {}
        fig.add_trace(go.Scatter(x=[xa, xb], y=[ya, yb], mode="lines", line=dict(width=1 + 4 * e.confidence, color="crimson" if hot else "gray"),
                                 hoverinfo="text", showlegend=False,
                                 text=f"{e.relation_type} · since {e.temporal_validity.get('first_seen', 0):.1f}s · belief {op.get('belief', 0):.2f} "
                                      f"uncertainty {op.get('uncertainty', 1):.2f} · {e.evidence_features}"))
        fig.add_annotation(x=(xa + xb) / 2, y=(ya + yb) / 2, text=e.relation_type, showarrow=False, font=dict(size=9))
    xs, ys, tx, cs = [], [], [], []
    for nid, (x, y) in pos.items():
        n = snap.nodes.get(nid)
        xs.append(x), ys.append(y)
        tx.append("Ego" if nid == "ego" else f"{n.node_type} #{n.object_id}")
        cs.append("crimson" if nid in highlight else ("royalblue" if nid == "ego" else "seagreen"))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=tx, textposition="top center", marker=dict(size=12, color=cs), showlegend=False))
    fig.update_yaxes(autorange="reversed", visible=False)
    fig.update_xaxes(visible=False, range=[-0.05, 1.05])
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
    return fig


@st.cache_data(show_spinner=False)
def replay_graph(trip_id: str, step: int, _ts_key: str = ""):
    """Rebuild the scene graph of a stored trip up to `step` (tracks + cache-free replay: camera transforms not stored -> identity
    when no cache exists; for Nexar trips the v3 cache is used so the replay is identical to the analysis)."""
    from safedrive_v4.config import load_config
    from safedrive_v4.reasoning.uncertain_engine import UncertainReasoningEngine
    from safedrive_v4.scene_graph.builder import SceneGraphBuilder, video_steps
    ev, ts = stores()
    meta = ts.meta(trip_id)
    cfg = load_config()
    b, e = SceneGraphBuilder(cfg), UncertainReasoningEngine(cfg)
    feat = ts.features(trip_id)
    if feat is not None:
        gen = video_steps(feat)        # identical to the analysis
    else:                              # live trips: tracks + per-step camera transform + scene channels (replay.npz)
        tr = ts.tracks(trip_id)
        rp = ts.root / trip_id / "replay.npz"
        if tr is None or not rp.exists():
            return None, None
        z = np.load(rp)
        H, S, cols = z["H"], z["scene"], [str(c) for c in z["scene_cols"]]
        gen = ((s, float(z["times"][s]), tr[tr[:, 0] == s][:, 1:8], H[s].reshape(3, 3), dict(zip(cols, map(float, S[s]))))
               for s in range(len(H)))
    snap = res = None
    for s, t, d, M, sc in gen:
        snap = b.update(s, t, d, M, sc)
        res = e.evaluate(snap)
        if s >= step:
            break
    return snap, res
