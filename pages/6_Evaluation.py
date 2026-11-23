"""Evaluation: every number here is read from experiments/latest/*.json."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.utils.config import resolve
from src.utils.io import load_json
from ui.common import PLOT_LAYOUT, card, results, results_light, robustness, setup

setup("Evaluation", "📊")
st.title("📊 Evaluation")
r = results()
if not r:
    st.warning("No experiment results found. Run `python scripts/run_experiments.py`.")
    st.stop()
st.caption(f"Run {r['timestamp']} · device {r['device']} · videos train/val/test = "
           f"{r['n_videos']['train']}/{r['n_videos']['val']}/{r['n_videos']['test']} · test windows {r['n_eval_windows']['test']} · "
           f"seeds {r['seeds']} · observation window {r['selected_window_s']} s (selected on val)")


def g(d, k):
    v = d.get(k)
    return v["mean"] if isinstance(v, dict) else v


def gs(d, k):
    v = d.get(k)
    return v["std"] if isinstance(v, dict) else None


def light_tab() -> None:
    rl = results_light()
    if not rl:
        st.info("Run `python scripts/run_light_study.py` to populate the lightweight study.")
        return
    fin, ewl = rl["final"], rl["early_warning_test"]
    ewv1 = r["early_warning_test"]
    E = r["models"]["E_ensemble_mc_dropout_ts"]
    st.markdown(f"**Deployed v2-light:** {fin['config']['classifier']} (bagged ×{fin['config']['n_members']}) on "
                f"`{fin['config']['features']}` window features, W = {fin['config']['W']} s, {fin['config']['n_features']} inputs, "
                f"calibration: {fin['config']['calibration']}. Selection rule (validation only): *{rl['selection_rule']}*.")
    effl, effv1 = rl["efficiency"], r["efficiency"]
    rob1 = robustness()
    rbl = load_json(resolve("experiments/latest/robustness_light.json")) if resolve("experiments/latest/robustness_light.json").exists() else None
    rows = [
        ("Test mAP (Nexar)", fin["test"]["mAP"], E["test"]["mAP"]),
        ("ROC-AUC", fin["test"]["roc_auc"], E["test"]["roc_auc"]),
        ("Precision / Recall", f"{fin['test']['precision']:.3f} / {fin['test']['recall']:.3f}", f"{E['test']['precision']:.3f} / {E['test']['recall']:.3f}"),
        ("F1", fin["test"]["f1"], E["test"]["f1"]),
        ("ECE / Brier", f"{fin['test']['ece']:.3f} / {fin['test']['brier']:.3f}", f"{E['test']['ece']:.3f} / {E['test']['brier']:.3f}"),
        ("Detection (prob. alarms, ≤30 FA/h on val)", ewl["FINAL_prob_only"]["detection_rate"], ewv1["E_ensemble_prob_only"]["detection_rate"]),
        ("Timely detection (≤3 s before event)", ewl["FINAL_prob_only"]["timely_detection_rate"], ewv1["E_ensemble_prob_only"]["timely_detection_rate"]),
        ("False alarms / h (test)", ewl["FINAL_prob_only"]["false_alarms_per_hour"], ewv1["E_ensemble_prob_only"]["false_alarms_per_hour"]),
        ("Median time-to-alert (s)", ewl["FINAL_prob_only"]["median_tta_s"], ewv1["E_ensemble_prob_only"]["median_tta_s"]),
        ("Prediction latency (ms / window)", effl["latency_ms_per_window_ensemble"], effv1["latency_ms_per_window_ensemble_mc"]),
        ("Prediction device", "CPU", "GPU"),
        ("Model size (MB)", effl["model_size_mb"], 1.5),
        ("Perception (est. frames/s)", effl.get("perception_fps_light_est"), effl.get("perception_fps_v1_est")),
    ]
    if rbl and rob1:
        worst = lambda rb: min(v["mAP"] for k, v in rb["conditions"].items() if not k.startswith("clean"))
        rows.append(("Robustness: clean → worst-case mAP", f"{rbl['conditions']['clean_0']['mAP']:.3f} → {worst(rbl):.3f}",
                     f"{rob1['conditions']['clean_0']['mAP']:.3f} → {worst(rob1):.3f}"))
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else ("—" if v is None else str(v))
    st.dataframe(pd.DataFrame([{"metric": a, "v2-light": fmt(b), "v1 LSTM ensemble": fmt(c)} for a, b, c in rows]),
                 hide_index=True, use_container_width=True)
    bt = rl.get("bootstrap_vs_v1", {}).get("paired", {})
    for k, v in bt.items():
        st.caption(f"Paired video-bootstrap {k}: Δ mAP {v['diff']:+.3f}, 95 % CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}], "
                   f"P(v2 better) = {v['p_a_better']:.2f}")
    if "cv_grid" in rl:
        g = pd.DataFrame(rl["cv_grid"])
        st.markdown(f"**Selection protocol:** {rl.get('protocol', '')}. Selected `{rl['selected']['config']}` "
                    f"(OOF mAP {rl['selected']['oof_mAP']:.3f}); best `{rl['best']['config']}` ({rl['best']['oof_mAP']:.3f}); "
                    f"tolerance {rl['margin']:.3f} = one fold standard error.")
        fig = go.Figure()
        for cnn, col in [(False, "#22d3ee"), (True, "#f97316")]:
            d = g[g.needs_cnn == cnn]
            fig.add_trace(go.Scatter(x=d.fit_s_per_fold, y=d.oof_mAP, mode="markers", name="needs CNN" if cnn else "no CNN",
                                     error_y=dict(type="data", array=d.fold_se), marker=dict(color=col, size=9), text=d.config,
                                     hovertemplate="%{text}<br>OOF mAP %{y:.3f}<br>fit %{x:.1f}s"))
        fig.update_layout(**PLOT_LAYOUT, height=380, title="5-fold CV on train+val: OOF mAP (± fold SE) vs training cost",
                          xaxis_title="fit time per fold (s, log)", xaxis_type="log", yaxis_title="OOF mAP")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(g.sort_values("oof_mAP", ascending=False)[["config", "needs_cnn", "oof_mAP", "fold_mean", "fold_se", "roc_auc",
                                                                 "fit_s_per_fold", "n_features"]].round(3),
                     hide_index=True, use_container_width=True, height=300)
    old = resolve("backups/v2_val_selected/results_light.json")
    if old.exists():
        go_ = pd.DataFrame(load_json(old)["grid_summary"])
        go_["config"] = go_.model + " | " + go_.features + " | W" + go_.W.astype(str)
        fig = go.Figure(go.Scatter(x=go_.val_mAP, y=go_.test_mAP, mode="markers", text=go_.config,
                                   marker=dict(color=go_.needs_cnn.map({True: "#f97316", False: "#22d3ee"}), size=8)))
        fig.update_layout(**PLOT_LAYOUT, height=340, title="Why the protocol changed: first (single 60-video validation) study — val vs test mAP",
                          xaxis_title="validation mAP", yaxis_title="test mAP")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Validation ranking barely predicts test ranking; the model it selected (LogReg) was later shown by cross-validation "
                   "to be significantly worse than gradient boosting. Selection therefore moved to 5-fold CV on train+val.")
    cvp = resolve("experiments/latest/cv_study.json")
    if cvp.exists():
        cv = load_json(cvp)
        cvd = pd.DataFrame([{"configuration": k, "OOF mAP": v["oof_mAP"], "fold mean": v["fold_mAP_mean"], "fold std": v["fold_mAP_std"],
                             "ROC-AUC": v["roc_auc"], "ECE": v["ece"]} for k, v in cv["summary"].items()]).sort_values("OOF mAP", ascending=False)
        st.markdown(f"**5-fold grouped cross-validation on all {cv['n_videos']} videos ({cv['n_oof_windows']} out-of-fold windows)** "
                    "— corroborating analysis, not used for selection.")
        st.dataframe(cvd.round(3), hide_index=True, use_container_width=True)


rows = []
names = {"majority": "Majority class", "logreg_pooled": "Logistic regression (pooled)", "xgb_pooled": "XGBoost (pooled)",
         "xgb_motion_only": "XGBoost (motion only)", "A_frame_visual_motion": "A · Frame-level MLP (no temporal)",
         "B_gru_visual": "B · GRU visual only", "gru_motion_only": "GRU motion only",
         "C_gru_visual_motion": "C · GRU visual(abs) + motion", "C2_gru_vdelta_motion": "C2 · GRU visual-delta + motion",
         "gru_visual_delta_only": "GRU visual-delta only", "lstm_vdelta_motion": "LSTM visual-delta + motion",
         "tcn_vdelta_motion": "TCN visual-delta + motion", "pool_vdelta_motion": "Temporal pooling visual-delta + motion",
         "D_single_mc_dropout_ts": "D · C + MC-Dropout + temp. scaling", "E_ensemble_mc_dropout_ts": "E · Ensemble×5 + MC-Dropout + TS",
         "E_full_with_risk_engine": "E + risk engine score"}
for k, m in r["models"].items():
    t = m["test"]
    rows.append({"model": names.get(k, k), "mAP": g(t, "mAP"), "±": gs(t, "mAP"), "AP@0.5s": g(t, "ap_h0.5"),
                 "AP@1.0s": g(t, "ap_h1.0"), "AP@1.5s": g(t, "ap_h1.5"), "ROC-AUC": g(t, "roc_auc"), "PR-AUC": g(t, "pr_auc"),
                 "Precision": g(t, "precision"), "Recall": g(t, "recall"), "F1": g(t, "f1"), "Accuracy": g(t, "accuracy"),
                 "Brier": g(t, "brier"), "ECE": g(t, "ece")})
df = pd.DataFrame(rows)
best = df.iloc[df["mAP"].idxmax()]
c = st.columns(4)
card(c[0], "Best test mAP", f"{best['mAP']:.3f}", best["model"])
E = r["models"]["E_ensemble_mc_dropout_ts"]["test"]
card(c[1], "Proposed (E) F1", f"{E['f1']:.3f}", f"P {E['precision']:.2f} · R {E['recall']:.2f}")
ew = r["early_warning_test"]["E_full_risk_engine"]
card(c[2], "False alarms / hour", f"{ew['false_alarms_per_hour']:.1f}", f"{ew['false_alarms']} onsets in {ew['negative_hours']*60:.0f} min normal driving")
card(c[3], "Mean time-to-alert", f"{ew['mean_tta_s']:.2f} s", f"median {ew['median_tta_s']:.2f} s · detected {ew['detection_rate']:.0%}")
st.write("")

def v3_tab() -> None:
    p = resolve("experiments/latest/results_ew_full.json")
    if not p.exists():
        st.info("Run `python scripts/run_ew_study.py --dev full` to populate the early-warning study.")
        return
    ew = load_json(p)
    st.markdown(f"**Early-warning study (v3)** on {ew['n_dev']} dev videos (5-fold CV, out-of-fold streams); held-out {ew['n_test']} + "
                f"{ew['n_test_ext']} videos. Primary metric S = mean recall at 5/10/20 false alarms per hour. Rule: *{ew['rule']}*.")
    tests = {**ew["stageA_tests"], **ew["stageB_tests"]}
    d = pd.DataFrame([{"change": k, "ΔS": v["mean"], "CI90 low": v["ci90"][0], "CI90 high": v["ci90"][1], "kept": v["ci90"][0] > 0} for k, v in tests.items()])
    fig = go.Figure(go.Scatter(x=d["ΔS"], y=d.change, mode="markers", marker=dict(color=d.kept.map({True: "#16a34a", False: "#94a3b8"}), size=10),
                               error_x=dict(type="data", symmetric=False, array=d["CI90 high"] - d["ΔS"], arrayminus=d["ΔS"] - d["CI90 low"])))
    fig.add_vline(x=0, line=dict(color="#e5e9f5", width=1))
    fig.update_layout(**PLOT_LAYOUT, height=480, title="Δ streaming score vs reference (paired bootstrap, 90 % CI; green = kept)",
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
    cp = resolve("experiments/latest/ew_heldout_compare.json")
    if cp.exists():
        c = load_json(cp)
        st.markdown(f"**Held-out at matched false-alarm rates** ({c['n_videos']} videos, {c['neg_hours']:.2f} h normal driving)")
        st.dataframe(pd.DataFrame([{"model": k, **{x: v[x] for x in ("recall@5", "recall@10", "recall@20", "S", "edr@1s@10", "edr@2s@10", "edr@3s@10")}}
                                   for k, v in c["models"].items()]).round(3), hide_index=True, use_container_width=True)
        for k, v in c["paired"].items():
            st.caption(f"{k}: ΔS {v['S']['mean']:+.3f} [{v['S']['ci90'][0]:+.3f}, {v['S']['ci90'][1]:+.3f}], P(better) = {v['S']['p_better']:.2f}")
    rows = []
    for name, f in [("v2-light", "robustness_v2_v3caches.json"), ("v3", "robustness_v3.json"), ("v3-stabilized", "robustness_v3_stabilized.json")]:
        q = resolve(f"experiments/latest/{f}")
        if q.exists():
            for cond, v in load_json(q)["conditions"].items():
                rows.append({"model": name, "condition": cond, "mAP": v["mAP"]})
    if rows:
        rr = pd.DataFrame(rows)
        fig = go.Figure([go.Bar(x=g.condition, y=g.mAP, name=m) for m, g in rr.groupby("model", sort=False)])
        fig.update_layout(**PLOT_LAYOUT, barmode="group", height=380, yaxis=dict(range=[0, 1], title="test mAP"),
                          title="Robustness incl. camera shake (identical corrupted test clips)")
        st.plotly_chart(fig, use_container_width=True)


tabv3, tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["🚨 Early warning (v3)", "⚡ Lightweight study (v2 vs v1)", "Baselines & models (v1 run)", "Ablation A–E", "Early warning", "Robustness", "Efficiency"])
with tabv3:
    v3_tab()
with tab0:
    light_tab()
with tab1:
    st.dataframe(df.style.format({c: "{:.3f}" for c in df.columns if c != "model"}, na_rep="—")
                 .highlight_max(subset=["mAP", "ROC-AUC", "F1"], color="#164e63"),
                 hide_index=True, use_container_width=True, height=540)
    st.caption("Neural rows: mean (± std) over 3 seeds. Thresholds for P/R/F1 are F1-optimal on validation. "
               "mAP = mean of AP at 0.5 / 1.0 / 1.5 s before the event (positives at that horizon vs all negatives).")
with tab2:
    keys = ["A_frame_visual_motion", "B_gru_visual", "C_gru_visual_motion", "C2_gru_vdelta_motion", "D_single_mc_dropout_ts", "E_ensemble_mc_dropout_ts"]
    ab = df.set_index(df.model).loc[[names[k] for k in keys]]
    fig = go.Figure()
    for met, col in [("mAP", "#22d3ee"), ("ROC-AUC", "#a855f7"), ("F1", "#f97316")]:
        fig.add_trace(go.Bar(x=ab.index, y=ab[met], name=met, marker_color=col,
                             error_y=dict(type="data", array=ab["±"].fillna(0)) if met == "mAP" else None))
    fig.update_layout(**PLOT_LAYOUT, barmode="group", height=420, yaxis=dict(range=[0, 1]), title="Ablation (test)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(ab[["mAP", "±", "ROC-AUC", "F1", "Brier", "ECE"]].round(3), use_container_width=True)
    ws = pd.DataFrame(r["window_selection_rows"]).groupby("window_s").mAP.agg(["mean", "std"]).reset_index()
    fig = go.Figure(go.Scatter(x=ws.window_s, y=ws["mean"], error_y=dict(array=ws["std"]), mode="lines+markers", line=dict(color="#22d3ee")))
    fig.update_layout(**PLOT_LAYOUT, height=300, title="Observation window length (validation mAP, GRU visual+motion)",
                      xaxis_title="window (s)", yaxis_title="val mAP")
    st.plotly_chart(fig, use_container_width=True)
with tab3:
    ewr = []
    for k, v in r["early_warning_test"].items():
        if k == "operating_curve_E":
            continue
        ewr.append({"system": k, "threshold": v["threshold"], "detection rate": v["detection_rate"], "mean TTA (s)": v["mean_tta_s"],
                    "median TTA (s)": v["median_tta_s"], "false alarms / h": v["false_alarms_per_hour"],
                    "alarm after annotated alert time": v["frac_alarm_after_annotated_alert"],
                    "median (alarm − annotated alert) s": v["median_alarm_minus_annotated_alert_s"]})
    st.dataframe(pd.DataFrame(ewr).round(3), hide_index=True, use_container_width=True)
    oc = pd.DataFrame(r["early_warning_test"]["operating_curve_E"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=oc.false_alarms_per_hour, y=oc.detection_rate, mode="lines+markers", name="detection rate",
                             text=oc.threshold.round(2), line=dict(color="#22d3ee")))
    fig.add_trace(go.Scatter(x=oc.false_alarms_per_hour, y=oc.mean_tta_s, mode="lines+markers", name="mean TTA (s)", yaxis="y2",
                             text=oc.threshold.round(2), line=dict(color="#f97316")))
    fig.update_layout(**PLOT_LAYOUT, height=380, title="Operating curve over alarm threshold (test, reported only)",
                      xaxis_title="false alarms per hour", yaxis=dict(title="detection rate", range=[0, 1.05]),
                      yaxis2=dict(title="mean TTA (s)", overlaying="y", side="right"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Streaming evaluation on full held-out videos: alarm onset = ≥3 consecutive steps above threshold. "
               "TTA = event time − first onset before the event. FA/h measured on normal-driving videos.")
with tab4:
    rb = robustness()
    if not rb:
        st.info("Run `python scripts/run_robustness.py` to populate robustness results.")
    else:
        rr = pd.DataFrame([{"condition": k, **{m: v[m] for m in ("mAP", "roc_auc", "f1", "ece", "mean_mutual_info")}}
                           for k, v in rb["conditions"].items()])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=rr.condition, y=rr.mAP, name="mAP", marker_color="#22d3ee"))
        fig.add_trace(go.Scatter(x=rr.condition, y=rr.mean_mutual_info, name="mean MI", yaxis="y2", mode="lines+markers",
                                 line=dict(color="#a855f7")))
        fig.update_layout(**PLOT_LAYOUT, height=400, title="Robustness under controlled corruptions (test clips, full pipeline re-run)",
                          yaxis=dict(range=[0, 1], title="mAP"), yaxis2=dict(overlaying="y", side="right", title="mean MI"))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(rr.round(4), hide_index=True, use_container_width=True)
with tab5:
    ef = r["efficiency"]
    c = st.columns(4)
    card(c[0], "Perception throughput", f"{ef.get('feature_extraction_fps') or 0:.1f} fps", "detect+track+flow+CNN @10 Hz grid")
    card(c[1], "Latency (single, det.)", f"{ef['latency_ms_per_window_single_deterministic']:.2f} ms", "per window")
    card(c[2], "Latency (ensemble+MC)", f"{ef['latency_ms_per_window_ensemble_mc']:.1f} ms", "per window")
    card(c[3], "Peak VRAM (training)", f"{ef.get('peak_vram_gb_training', 0):.2f} GB", f"RSS {ef['process_rss_gb']:.1f} GB")
    st.json(r["environment"])
