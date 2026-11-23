"""Uncertainty: per-video confidence vs uncertainty, UNCERTAIN state, calibration from experiments."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.common import PLOT_LAYOUT, card, model_meta, results, results_light, setup

setup("Uncertainty", "🎯")
st.title("🎯 Uncertainty & Calibration")
st.markdown("""
- **Model confidence** — the calibrated probability itself (how sure the model *says* it is).
- **Predictive uncertainty** — spread across ensemble members (v2-light: 5 bootstrap-bagged classifiers; v1: 5 networks ×
  MC-Dropout): total entropy, and its disagreement part, mutual information (MI), high when members disagree.
- **Calibration quality** — whether a stated 70 % really happens ~70 % of the time (ECE, Brier, reliability diagram).
""")
meta = model_meta() or {}
rc = meta.get("risk", {})

res = st.session_state.get("result")
if res is not None:
    tl = res.timeline
    st.subheader("Current video")
    c = st.columns(4)
    card(c[0], "Mean confidence", f"{(tl.p_mean.where(tl.p_mean > .5, 1 - tl.p_mean)).mean():.2f}", "max(p, 1-p)")
    card(c[1], "Mean entropy", f"{tl.entropy.mean():.3f}", "nats")
    card(c[2], "Max MI", f"{tl.mutual_info.max():.3f}", f"UNCERTAIN threshold {rc.get('mi_threshold', float('nan')):.3f}")
    card(c[3], "UNCERTAIN steps", f"{(tl.level == 'UNCERTAIN').mean():.0%}", "system abstains")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tl.time_s, y=tl.p_mean, name="P(collision)", line=dict(color="#22d3ee")))
    fig.add_trace(go.Scatter(x=tl.time_s, y=tl.entropy, name="entropy (total)", line=dict(color="#eab308")))
    fig.add_trace(go.Scatter(x=tl.time_s, y=tl.mutual_info, name="mutual information (epistemic)", line=dict(color="#a855f7")))
    if "mi_threshold" in rc:
        fig.add_hline(y=rc["mi_threshold"], line=dict(color="#a855f7", dash="dash"), annotation_text="MI threshold")
    unc = tl[tl.level == "UNCERTAIN"]
    if len(unc):
        fig.add_trace(go.Scatter(x=unc.time_s, y=[1.0] * len(unc), mode="markers", name="UNCERTAIN",
                                 marker=dict(color="#a855f7", symbol="diamond", size=8)))
        st.warning("**UNCERTAIN — more information is needed.** At these moments the score is in the ambiguous band "
                   "and ensemble members disagree, so SafeDrive abstains rather than forcing a verdict.")
    fig.update_layout(**PLOT_LAYOUT, height=360, xaxis_title="time (s)", title="Confidence vs uncertainty over time")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Analyse a video to see its uncertainty timeline. Calibration results below come from the test set.")

rl = results_light()
if rl:
    F = rl["final"]
    st.subheader("v2-light (deployed) — calibration and uncertainty on the held-out test set")
    c = st.columns(4)
    card(c[0], "ECE / Brier", f"{F['test']['ece']:.3f} / {F['test']['brier']:.3f}",
         f"calibration chosen by CV inside val: {F['config']['calibration']}")
    card(c[1], "Error detection AUROC (entropy)", f"{F['error_auroc_entropy']:.3f}", "does uncertainty flag wrong predictions?")
    card(c[2], "Error detection AUROC (MI)", f"{F['error_auroc_mutual_info']:.3f}", "member disagreement")
    sr = F["selective_risk_entropy"]
    card(c[3], "Accuracy @50 % coverage", f"{sr[0]['accuracy']:.3f} → {sr[-1]['accuracy']:.3f}", "abstain on highest entropy")
    l2, r2 = st.columns(2)
    with l2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color="#64748b", dash="dash"), name="perfect"))
        rel = pd.DataFrame(F["reliability_test_calibrated"])
        rel = rel[rel["count"] > 0]
        fig.add_trace(go.Scatter(x=rel.confidence, y=rel.frequency, mode="lines+markers", name="v2-light", line=dict(color="#22d3ee"),
                                 marker=dict(size=(rel["count"] / rel["count"].max() * 16 + 5))))
        fig.update_layout(**PLOT_LAYOUT, height=340, title="Reliability (v2-light, test)", xaxis_title="predicted probability",
                          yaxis_title="observed frequency")
        st.plotly_chart(fig, use_container_width=True)
    with r2:
        fig = go.Figure()
        for u, ls in [("entropy", "solid"), ("mutual_info", "dot")]:
            d = pd.DataFrame(F[f"selective_risk_{u}"])
            fig.add_trace(go.Scatter(x=d.coverage, y=d.accuracy, mode="lines+markers", name=f"abstain by {u}", line=dict(color="#22d3ee", dash=ls)))
        fig.update_layout(**PLOT_LAYOUT, height=340, title="Risk–coverage (v2-light, test)", xaxis_title="coverage",
                          yaxis_title="accuracy", xaxis_autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

r = results()
if not r:
    st.warning("No experiment results yet.")
    st.stop()
st.subheader("v1 (LSTM ensemble) — calibration on the held-out test set")
E = r["models"]["E_ensemble_mc_dropout_ts"]
D = r["models"]["D_single_mc_dropout_ts"]
c = st.columns(4)
card(c[0], "ECE before → after TS", f"{E['test_uncalibrated']['ece']:.3f} → {E['test']['ece']:.3f}", f"temperature T = {E['temperature']:.2f}")
card(c[1], "Brier before → after", f"{E['test_uncalibrated']['brier']:.3f} → {E['test']['brier']:.3f}")
card(c[2], "Error detection AUROC (MI)", f"{E['error_auroc_mutual_info']:.3f}", "does MI flag wrong predictions?")
card(c[3], "Error detection AUROC (entropy)", f"{E['error_auroc_entropy']:.3f}")
st.write("")
l, rr = st.columns(2)
with l:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color="#64748b", dash="dash"), name="perfect"))
    for k, col, nm in [("reliability_test_uncalibrated", "#f97316", "before TS"), ("reliability_test_calibrated", "#22d3ee", "after TS")]:
        rel = pd.DataFrame(E[k])
        rel = rel[rel["count"] > 0]
        fig.add_trace(go.Scatter(x=rel.confidence, y=rel.frequency, mode="lines+markers", name=nm, line=dict(color=col),
                                 marker=dict(size=(rel["count"] / rel["count"].max() * 16 + 5))))
    fig.update_layout(**PLOT_LAYOUT, height=380, title="Reliability diagram (ensemble, test)",
                      xaxis_title="predicted probability", yaxis_title="observed frequency")
    st.plotly_chart(fig, use_container_width=True)
with rr:
    fig = go.Figure()
    for tag, M_, col in [("ensemble", E, "#22d3ee"), ("single model", D, "#eab308")]:
        sr = pd.DataFrame(M_["selective_risk_mutual_info"])
        fig.add_trace(go.Scatter(x=sr.coverage, y=sr.accuracy, mode="lines+markers", name=f"{tag} (abstain by MI)", line=dict(color=col)))
        sr2 = pd.DataFrame(M_["selective_risk_entropy"])
        fig.add_trace(go.Scatter(x=sr2.coverage, y=sr2.accuracy, mode="lines+markers", name=f"{tag} (abstain by entropy)",
                                 line=dict(color=col, dash="dot")))
    fig.update_layout(**PLOT_LAYOUT, height=380, title="Risk–coverage: accuracy when abstaining on uncertain cases",
                      xaxis_title="coverage (fraction answered)", yaxis_title="accuracy", xaxis_autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
st.caption("Temperature fitted on the validation split only. Equal-width 10-bin ECE on the positive-class probability.")
