"""Shared Streamlit helpers: theme, cached resources, status sidebar, result access."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.risk.risk_engine import LEVEL_COLORS  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.io import load_json  # noqa: E402

MODELS = {"v3 (default: v2 + vehicle-interaction cues, dense-negative training)": resolve("models/safedrive_v3"),
          "v3-stabilized (+ camera ego-motion compensation; for shaky cameras)": resolve("models/safedrive_v3_stabilized"),
          "v2-light (motion + frame-difference, gradient boosting, no CNN)": resolve("models/safedrive_v2_light"),
          "v1 (LSTM ensemble + ResNet-18, archived baseline)": resolve("models/safedrive_v1")}
MODEL_DIR = next((p for p in MODELS.values() if (p / "bundle.json").exists()), resolve("models/safedrive_v1"))
RESULTS = resolve("experiments/latest/results.json")
RESULTS_LIGHT = resolve("experiments/latest/results_light.json")
ROBUST = resolve("experiments/latest/robustness.json")
STATS = resolve("data/processed/dataset_stats.json")

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1350px;}
.sd-hero {padding: 1.4rem 1.6rem; border-radius: 16px; border: 1px solid #1f2b47;
  background: radial-gradient(1200px 300px at 0% 0%, rgba(34,211,238,.14), transparent),
              linear-gradient(135deg, #0f172a 0%, #111a33 100%);}
.sd-hero h1 {margin: 0; font-size: 2.1rem; letter-spacing: .5px;}
.sd-hero p {color: #a9b4d0; margin: .4rem 0 0 0;}
.sd-card {border: 1px solid #1f2b47; border-radius: 14px; padding: .9rem 1rem; background: #0f172a; height: 100%;}
.sd-card .k {color: #8b98b8; font-size: .78rem; text-transform: uppercase; letter-spacing: .8px;}
.sd-card .v {font-size: 1.45rem; font-weight: 700; margin-top: .15rem;}
.sd-card .s {color: #8b98b8; font-size: .78rem;}
.sd-pill {display: inline-block; padding: .18rem .6rem; border-radius: 999px; font-weight: 700; font-size: .8rem;}
.sd-dot {display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;}
</style>
"""


def setup(title: str, icon: str = "🛡️") -> dict:
    st.set_page_config(page_title=f"SafeDrive AI · {title}", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    cfg = load_config()
    sidebar_status()
    return cfg


def card(col, k: str, v: str, s: str = "") -> None:
    col.markdown(f'<div class="sd-card"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div></div>',
                 unsafe_allow_html=True)


def pill(level: str) -> str:
    c = LEVEL_COLORS.get(level, "#64748b")
    return f'<span class="sd-pill" style="background:{c}22;color:{c};border:1px solid {c}">{level}</span>'


def available_models() -> dict:
    return {k: p for k, p in MODELS.items() if (p / "bundle.json").exists()}


def current_model_dir() -> Path:
    avail = available_models()
    choice = st.session_state.get("model_choice")
    return avail.get(choice, MODEL_DIR) if avail else MODEL_DIR


def model_meta() -> dict | None:
    p = current_model_dir() / "bundle.json"
    return load_json(p) if p.exists() else None


def results_light() -> dict | None:
    return load_json(RESULTS_LIGHT) if RESULTS_LIGHT.exists() else None


def results() -> dict | None:
    return load_json(RESULTS) if RESULTS.exists() else None


def robustness() -> dict | None:
    return load_json(ROBUST) if ROBUST.exists() else None


def dataset_stats() -> dict | None:
    return load_json(STATS) if STATS.exists() else None


def sidebar_status() -> None:
    meta = model_meta()
    stats = dataset_stats()
    with st.sidebar:
        st.markdown("### 🛡️ SafeDrive AI")
        st.caption("Explainable & uncertainty-aware early collision prediction")
        ok = '<span class="sd-dot" style="background:#22c55e"></span>'
        bad = '<span class="sd-dot" style="background:#ef4444"></span>'
        avail = available_models()
        if avail:
            names = list(avail)
            cur = st.session_state.get("model_choice", names[0])
            st.selectbox("Prediction model", names, index=names.index(cur) if cur in names else 0, key="model_choice")
            meta = model_meta()
            prof = (meta or {}).get("risk", {}).get("profiles")
            if prof:
                opts = ["conservative", "balanced", "sensitive"]
                lab = {"conservative": "Conservative (~5 false alarms/h on dev)", "balanced": "Balanced (~10/h)",
                       "sensitive": "Sensitive (~20/h)"}
                curp = st.session_state.get("warning_profile", "balanced")
                st.selectbox("Warning sensitivity", opts, index=opts.index(curp) if curp in opts else 1, key="warning_profile",
                             format_func=lambda o: lab[o])
        st.markdown((ok if meta else bad) + ("Model: **trained** " + f"(x{meta['n_members']}, {meta.get('type', 'neural')})" if meta else "Model: **not trained**"),
                    unsafe_allow_html=True)
        st.markdown((ok if stats else bad) + (f"Dataset: **{stats['all']['n_videos']} videos**" if stats else "Dataset: **not prepared**"),
                    unsafe_allow_html=True)
        try:
            import torch
            dev = f"CUDA · {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
        except Exception:  # noqa: BLE001
            dev = "CPU"
        st.markdown(ok + f"Device: **{dev}**", unsafe_allow_html=True)
        if "result" in st.session_state:
            r = st.session_state["result"]
            st.markdown("---")
            st.markdown(f"**Current video**  \n`{Path(r.video_path).name}`", unsafe_allow_html=True)
            st.markdown(pill(r.max_level), unsafe_allow_html=True)
        st.markdown("---")
        st.caption("Research prototype — not a certified safety system.")


@st.cache_resource(show_spinner="Loading detector, tracker and prediction model…")
def _pipeline_for(model_dir: str):
    from src.pipeline import SafeDrivePipeline
    return SafeDrivePipeline(model_dir)


def get_pipeline(model_dir: str | None = None):
    return _pipeline_for(str(model_dir or current_model_dir()))


def require_result():
    if "result" not in st.session_state:
        st.info("No video analysed yet. Go to **Video Analysis**, upload a dashcam clip or pick a dataset sample.", icon="🎬")
        try:
            st.page_link("pages/2_Video_Analysis.py", label="Open Video Analysis", icon="🎬")
        except Exception:  # noqa: BLE001 — page registry unavailable when a page runs standalone
            pass
        st.stop()
    return st.session_state["result"]


def annotations() -> pd.DataFrame | None:
    p = resolve("data/annotations/nexar_train.csv")
    if not p.exists():
        return None
    from src.data.annotation_parser import parse_annotations
    return parse_annotations(p)


def ground_truth_for(video_path: str) -> dict | None:
    ann = annotations()
    if ann is None:
        return None
    vid = Path(video_path).stem.split("__")[-1]
    row = ann[ann.video_id == vid]
    if row.empty:
        return None
    r = row.iloc[0]
    return {"video_id": vid, "label": int(r.label), "time_of_event": None if pd.isna(r.time_of_event) else float(r.time_of_event),
            "time_of_alert": None if pd.isna(r.time_of_alert) else float(r.time_of_alert)}


PLOT_LAYOUT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,.6)",
                   margin=dict(l=10, r=10, t=40, b=10), font=dict(size=12),
                   legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
