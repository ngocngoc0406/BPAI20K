"""Reports: export CSV / JSON / PDF / annotated video; experiment summary."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.reporting.exports import result_json, result_pdf, timeline_csv
from src.utils.config import resolve
from ui.common import model_meta, results, robustness, setup

setup("Reports", "📄")
st.title("📄 Reports & Exports")

res = st.session_state.get("result")
if res is None:
    st.info("Analyse a video first to export its report. Experiment summaries are below.")
else:
    stem = Path(res.video_path).stem.split("__")[-1]
    st.subheader(f"Analysis export — {stem}")
    c = st.columns(4)
    c[0].download_button("⬇ Timeline CSV", timeline_csv(res), f"{stem}_timeline.csv", "text/csv", use_container_width=True)
    c[1].download_button("⬇ Result JSON", result_json(res), f"{stem}_result.json", "application/json", use_container_width=True)
    try:
        pdf = result_pdf(res, model_meta())
        c[2].download_button("⬇ PDF report", pdf, f"{stem}_report.pdf", "application/pdf", use_container_width=True)
    except Exception as e:  # noqa: BLE001
        c[2].error(f"PDF failed: {e}")
    ann = st.session_state.get("annotated")
    if ann and Path(ann).exists():
        c[3].download_button("⬇ Annotated video", Path(ann).read_bytes(), f"{stem}_annotated.mp4", "video/mp4",
                             use_container_width=True)
    st.json(res.summary(), expanded=False)

r = results()
st.subheader("Experiment summary")
if not r:
    st.warning("No experiment results.")
    st.stop()
E = r["models"]["E_ensemble_mc_dropout_ts"]
ew = r["early_warning_test"]["E_full_risk_engine"]
st.markdown(f"""
| Item | Value |
|---|---|
| Run | `{r['timestamp']}` on {r['device']} ({r['environment'].get('gpu', 'CPU')}) |
| Videos (train / val / test) | {r['n_videos']['train']} / {r['n_videos']['val']} / {r['n_videos']['test']} |
| Selected window / architecture (val) | {r['selected_window_s']} s / {r['selected_architecture']['name']} |
| v1 E (LSTM ensemble) — test mAP / ROC-AUC / F1 | {E['test']['mAP']:.3f} / {E['test']['roc_auc']:.3f} / {E['test']['f1']:.3f} |
| ECE before → after temperature scaling | {E['test_uncalibrated']['ece']:.3f} → {E['test']['ece']:.3f} |
| Early warning — detection rate / mean TTA / FA per h | {ew['detection_rate']:.2f} / {ew['mean_tta_s']:.2f} s / {ew['false_alarms_per_hour']:.1f} |
| Runtime | {r['runtime_minutes']:.1f} min |
""")
from ui.common import results_light  # noqa: E402
rl = results_light()
if rl:
    F, ewl = rl["final"], rl["early_warning_test"]["FINAL_prob_only"]
    st.markdown(f"""
| v2-light (deployed) | Value |
|---|---|
| Model | {F['config']['classifier']} ×{F['config']['n_members']} on {F['config']['features']} (W = {F['config']['W']} s, {F['config']['n_features']} inputs) |
| Test mAP / ROC-AUC / F1 / ECE | {F['test']['mAP']:.3f} / {F['test']['roc_auc']:.3f} / {F['test']['f1']:.3f} / {F['test']['ece']:.3f} |
| Early warning — detection / timely / FA per h | {ewl['detection_rate']:.2f} / {ewl['timely_detection_rate']:.2f} / {ewl['false_alarms_per_hour']:.1f} |
| Prediction latency / model size | {rl['efficiency']['latency_ms_per_window_ensemble']:.2f} ms (CPU) / {rl['efficiency']['model_size_mb'] * 1000:.0f} KB |
""")
latest = resolve("experiments/latest")
for name in ("results.json", "robustness.json", "results_light.json", "robustness_light.json", "cv_study.json"):
    p = latest / name
    if p.exists():
        st.download_button(f"⬇ {name}", p.read_bytes(), name, "application/json")
if robustness() is None:
    st.caption("Robustness results not available yet.")
docs = resolve("docs")
st.subheader("Documents")
for p in sorted(docs.glob("*.md")) + sorted(resolve("reports").glob("*.pdf")) + sorted(resolve("reports").glob("*.pptx")):
    st.markdown(f"- `{p.relative_to(resolve('.'))}`")
