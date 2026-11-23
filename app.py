"""SafeDrive AI — Overview page.  Run:  streamlit run app.py"""
from __future__ import annotations

import streamlit as st

from ui.common import card, dataset_stats, model_meta, results, results_light, setup

cfg = setup("Overview")
meta, res, stats, rl = model_meta(), results(), dataset_stats(), results_light()

st.markdown("""
<div class="sd-hero">
  <h1>🛡️ SafeDrive AI</h1>
  <p>Explainable and uncertainty-aware <b>early traffic-collision prediction</b> from dashcam video.
  Detects and tracks road users, models their motion over time, predicts collision / near-miss risk
  <i>before</i> it happens, quantifies how much the prediction can be trusted, and shows why it warned.</p>
</div>""", unsafe_allow_html=True)
st.write("")

c = st.columns(4)
from src.utils.config import resolve as _res  # noqa: E402
from src.utils.io import load_json as _lj  # noqa: E402
_cmp = _res("experiments/latest/ew_heldout_compare.json")
_ew = _res("experiments/latest/results_ew_full.json")
if _cmp.exists() and _ew.exists():
    m3 = _lj(_cmp)["models"]["v3_selected"]
    f3 = _lj(_ew)["final"]["selected_v3"]["test_all"]
    card(c[0], "Held-out mAP / AUC", f"{f3['mAP']:.3f} / {f3['roc_auc']:.3f}", "v3 · 224 videos never used for choices")
    card(c[1], "Recall @10 false alarms/h", f"{m3['recall@10']:.0%}", f"@5: {m3['recall@5']:.0%} · @20: {m3['recall@20']:.0%} (held-out)")
    card(c[2], "Calibration (ECE)", f"{f3['ece']:.3f}", "Platt, fitted on dev out-of-fold")
    _bm = _res("experiments/latest/pipeline_benchmark_v3.json")
    if _bm.exists():
        b3 = _lj(_bm)
        card(c[3], "Throughput", f"{b3['v3@cuda']['end_to_end_fps']:.0f} / {b3['v3@cpu']['end_to_end_fps']:.0f} fps", "GPU / CPU-only, end-to-end")
elif rl:
    fin, ewl = rl["final"], rl["early_warning_test"]["FINAL_prob_only"]
    card(c[0], "Test mAP (Nexar protocol)", f"{fin['test']['mAP']:.3f}", "v2-light; v1 LSTM 0.582 (difference n.s.)")
    card(c[1], "Prediction latency", f"{rl['efficiency']['latency_ms_per_window_ensemble']:.1f} ms", "per window, CPU only, 34 KB model")
    card(c[2], "Timely detection", f"{ewl['timely_detection_rate']:.0%}", f"≤3 s before event · {ewl['false_alarms_per_hour']:.0f} false alarms/h")
    card(c[3], "Perception speed", f"{rl['efficiency'].get('perception_fps_light_est', 0):.0f} fps", "YOLOv8n + ByteTrack + flow + frame diff")
elif res:
    e = res["models"]["E_ensemble_mc_dropout_ts"]["test"]
    ew = res["early_warning_test"]["E_full_risk_engine"]
    card(c[0], "Test mAP (Nexar protocol)", f"{e['mAP']:.3f}", "mean AP @ 0.5 / 1.0 / 1.5 s before event")
    card(c[1], "Test ROC-AUC", f"{e['roc_auc']:.3f}", f"F1 {e['f1']:.3f} at validation threshold")
    card(c[2], "Mean warning time", f"{ew['mean_tta_s']:.2f} s", f"detection rate {ew['detection_rate']:.0%}")
    card(c[3], "Calibration (ECE)", f"{e['ece']:.3f}", f"Brier {e['brier']:.3f} after temperature scaling")
else:
    for col, k in zip(c, ["Test mAP", "ROC-AUC", "Warning time", "ECE"]):
        card(col, k, "—", "run scripts/run_experiments.py")

st.write("")
left, right = st.columns([3, 2])
with left:
    st.subheader("System architecture")
    st.graphviz_chart("""
digraph G { rankdir=LR; bgcolor="transparent";
 node [shape=box style="rounded,filled" fillcolor="#111a33" color="#22d3ee" fontcolor="#e5e9f5" fontname="Helvetica" fontsize=11];
 edge [color="#64748b"];
 V [label="Dashcam video\\n(any FPS → 10 Hz grid)"];
 D [label="YOLOv8n detection\\nperson · bicycle · car\\nmotorcycle · bus · truck"];
 T [label="ByteTrack\\nmulti-object tracking"];
 M [label="Motion features (28)\\nlooming · apparent τ · lateral\\npairwise closing · optical flow"];
 C [label="Frame difference (20)\\n4×4 change grid + stats\\n(v1: ResNet-18)"];
 G [label="Window aggregates (8 per cue)\\nlast · mean · max · slope · trend\\n→ bagged classifier ×5 (v2-light)\\n(v1: LSTM ensemble)"];
 U [label="Calibrated P +\\nensemble entropy / disagreement"];
 R [label="Risk engine\\nNORMAL · CAUTION · WARNING\\nHIGH RISK · UNCERTAIN"];
 X [label="Explainability\\nocclusion per cue & time step\\n(v1: Grad-CAM · IG)"];
 V->D->T->M->G; V->C->G; G->U->R; G->X; M->R;
}""", use_container_width=True)
with right:
    st.subheader("Status")
    if meta:
        calib = f"T = {meta['temperature']:.2f}" if "temperature" in meta else f"calibration: {meta.get('calibration', 'none')}"
        st.success(f"**Model trained** · {meta['description']}  \nwindow {meta['window_s']} s · {calib} · "
                   f"trained {meta['trained']}")
    else:
        st.warning("No trained model bundle in `models/safedrive_v1`. Run the experiment pipeline (see README).")
    if stats:
        a = stats["all"]
        st.info(f"**Nexar Dashcam Collision Prediction** (subset)  \n{a['n_videos']} videos · {a['n_positive']} positive "
                f"(collision / near-miss) · {a['n_negative']} normal driving  \nsplit by video: "
                f"{stats['train']['n_videos']} / {stats['val']['n_videos']} / {stats['test']['n_videos']} (train/val/test)")
    else:
        st.warning("Dataset not prepared yet (scripts/download_nexar.py → prepare_dataset.py).")

st.subheader("What each page does")
feats = [
    ("🎬 Video Analysis", "Upload a clip, run detection + tracking, watch the annotated video and inspect frames."),
    ("📈 Collision Prediction", "Probability and risk timeline, early-warning onset, ground truth when available."),
    ("🧠 Explainable AI", "Grad-CAM on the frame, temporal & motion attributions, grounded triggering factors."),
    ("🎯 Uncertainty", "MC-Dropout/ensemble uncertainty, calibration and the UNCERTAIN abstention state."),
    ("📊 Evaluation", "Baselines, ablation A–E, early-warning metrics, robustness and efficiency — from experiment files."),
    ("🗂️ Dataset Explorer", "Label distribution, event timing, browse annotated videos."),
    ("📄 Reports", "Export CSV / JSON / PDF and the annotated video."),
]
cols = st.columns(4)
for i, (k, v) in enumerate(feats):
    with cols[i % 4]:
        st.markdown(f'<div class="sd-card"><div class="v" style="font-size:1.05rem">{k}</div><div class="s">{v}</div></div>',
                    unsafe_allow_html=True)
        st.write("")
st.caption("All metrics shown in the app are read from experiment output files (experiments/latest). "
           "Motion quantities are image-plane (apparent) motion, not physical speed.")
