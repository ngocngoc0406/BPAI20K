"""Generate figures, result tables (Markdown), PDF technical report and PPTX slides from experiment files.

Every number is read from experiments/latest/results.json and robustness.json — nothing is typed by hand.
Usage:  python scripts/make_reports.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.config import resolve  # noqa: E402
from src.utils.io import load_json  # noqa: E402

FIG = resolve("reports/figures")
NAMES = {"majority": "Majority class", "logreg_pooled": "Logistic regression (pooled)", "xgb_pooled": "XGBoost (pooled)",
         "xgb_motion_only": "XGBoost (motion only)", "A_frame_visual_motion": "A: Frame-level MLP (no temporal)",
         "B_gru_visual": "B: GRU visual only", "gru_motion_only": "GRU motion only",
         "C_gru_visual_motion": "C: GRU visual(abs)+motion", "C2_gru_vdelta_motion": "C2: GRU visual-delta+motion",
         "gru_visual_delta_only": "GRU visual-delta only", "lstm_vdelta_motion": "LSTM visual-delta+motion",
         "tcn_vdelta_motion": "TCN visual-delta+motion", "pool_vdelta_motion": "Temporal pooling visual-delta+motion",
         "D_single_mc_dropout_ts": "D: single + MC-Dropout + TS", "E_ensemble_mc_dropout_ts": "E: ensemble x5 + MC-Dropout + TS",
         "E_full_with_risk_engine": "E + risk-engine score"}
plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def m(d: dict, k: str):
    v = d.get(k)
    return (v["mean"], v["std"]) if isinstance(v, dict) else (v, None)


def fmt(v, s=None, p=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{p}f}" + (f" ± {s:.{p}f}" if s is not None else "")


def model_table(r: dict) -> pd.DataFrame:
    rows = []
    for k, d in r["models"].items():
        t = d["test"]
        rows.append({"key": k, "Model": NAMES.get(k, k), **{c: m(t, mk) for c, mk in [
            ("mAP", "mAP"), ("AP@0.5s", "ap_h0.5"), ("AP@1.0s", "ap_h1.0"), ("AP@1.5s", "ap_h1.5"), ("ROC-AUC", "roc_auc"),
            ("PR-AUC", "pr_auc"), ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"), ("Accuracy", "accuracy"),
            ("Brier", "brier"), ("ECE", "ece")]}})
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, cols: list[str], std_cols=("mAP",)) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, tuple):
                cells.append(fmt(v[0], v[1] if c in std_cols else None))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def figures(r: dict, rb: dict | None) -> list[Path]:
    FIG.mkdir(parents=True, exist_ok=True)
    paths = []
    df = model_table(r)
    # 1 model comparison
    d = df[df.key != "E_full_with_risk_engine"].copy()
    d["v"] = d.mAP.map(lambda x: x[0])
    d["s"] = d.mAP.map(lambda x: x[1] or 0)
    d = d.sort_values("v")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    cols = ["#94a3b8" if not k[0] in "ABCDE" else "#0891b2" for k in d.key]
    ax.barh(d.Model, d.v, xerr=d.s, color=cols)
    ax.set_xlabel("test mAP (Nexar protocol)")
    ax.set_xlim(0, 1)
    ax.set_title("Model comparison (held-out test)")
    fig.tight_layout()
    paths.append(FIG / "model_comparison.png")
    fig.savefig(paths[-1])
    plt.close(fig)
    # 2 ablation
    keys = ["A_frame_visual_motion", "B_gru_visual", "C_gru_visual_motion", "C2_gru_vdelta_motion", "D_single_mc_dropout_ts", "E_ensemble_mc_dropout_ts"]
    ab = df.set_index("key").loc[keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, (met, c) in enumerate([("mAP", "#0891b2"), ("ROC-AUC", "#7c3aed"), ("F1", "#ea580c")]):
        ax.bar(x + (i - 1) * 0.26, [v[0] for v in ab[met]], 0.26, yerr=[v[1] or 0 for v in ab[met]], label=met, color=c)
    ax.set_xticks(x, ["A\nno temporal", "B\nGRU visual", "C\n+ motion", "C2\nvisual delta", "D\n+ MC/TS", "E\nensemble"])
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=3)
    ax.set_title("Ablation A–E (test)")
    fig.tight_layout()
    paths.append(FIG / "ablation.png")
    fig.savefig(paths[-1])
    plt.close(fig)
    # 3 reliability
    E = r["models"]["E_ensemble_mc_dropout_ts"]
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8")
    for k, c, lab in [("reliability_test_uncalibrated", "#ea580c", f"before TS (ECE {E['test_uncalibrated']['ece']:.3f})"),
                      ("reliability_test_calibrated", "#0891b2", f"after TS (ECE {E['test']['ece']:.3f})")]:
        rel = pd.DataFrame(E[k])
        rel = rel[rel["count"] > 0]
        ax.plot(rel.confidence, rel.frequency, "o-", color=c, label=lab)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Reliability (ensemble, test)")
    fig.tight_layout()
    paths.append(FIG / "reliability.png")
    fig.savefig(paths[-1])
    plt.close(fig)
    # 4 risk-coverage
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    for tag, c in [("E_ensemble_mc_dropout_ts", "#0891b2"), ("D_single_mc_dropout_ts", "#ca8a04")]:
        for u, ls in [("mutual_info", "-"), ("entropy", ":")]:
            sr = pd.DataFrame(r["models"][tag][f"selective_risk_{u}"])
            ax.plot(sr.coverage, sr.accuracy, ls, marker="o", color=c, label=f"{tag[0]} / {u}")
    ax.invert_xaxis()
    ax.set_xlabel("coverage")
    ax.set_ylabel("accuracy on retained")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Risk–coverage (test)")
    fig.tight_layout()
    paths.append(FIG / "risk_coverage.png")
    fig.savefig(paths[-1])
    plt.close(fig)
    # 5 window selection
    ws = pd.DataFrame(r["window_selection_rows"]).groupby("window_s").mAP.agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    ax.errorbar(ws.index, ws["mean"], yerr=ws["std"], marker="o", color="#0891b2", capsize=3)
    ax.set_xlabel("observation window (s)")
    ax.set_ylabel("validation mAP")
    ax.set_title("Window length (GRU vis+mot)")
    fig.tight_layout()
    paths.append(FIG / "window_selection.png")
    fig.savefig(paths[-1])
    plt.close(fig)
    # 6 operating curve
    oc = pd.DataFrame(r["early_warning_test"]["operating_curve_E"]).sort_values("false_alarms_per_hour")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(oc.false_alarms_per_hour, oc.detection_rate, "o-", color="#0891b2", label="detection rate (any time before event)")
    if "timely_detection_rate" in oc:
        ax.plot(oc.false_alarms_per_hour, oc.timely_detection_rate, "^-", color="#16a34a", label="timely detection (≤3 s before)")
    ax2 = ax.twinx()
    ax2.plot(oc.false_alarms_per_hour, oc.mean_tta_s, "s--", color="#ea580c", label="mean TTA (s)")
    ax.set_xlabel("false alarms per hour (normal driving)")
    ax.set_ylabel("detection rate")
    ax2.set_ylabel("mean TTA (s)")
    ax.set_title("Early-warning operating curve (test)")
    fig.legend(loc="lower right", frameon=False, fontsize=7)
    fig.tight_layout()
    paths.append(FIG / "operating_curve.png")
    fig.savefig(paths[-1])
    plt.close(fig)
    # 7 permutation importance
    pi = list(r["permutation_importance_test_ap_drop"].items())[:12][::-1]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.barh([k for k, _ in pi], [v for _, v in pi], color="#7c3aed")
    ax.set_xlabel("drop in test AP when shuffled")
    ax.set_title("Permutation feature importance")
    fig.tight_layout()
    paths.append(FIG / "feature_importance.png")
    fig.savefig(paths[-1])
    plt.close(fig)
    # 8 robustness
    if rb:
        rr = pd.DataFrame([{"c": k, **v} for k, v in rb["conditions"].items()])
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.bar(rr.c, rr.mAP, color=["#16a34a" if c.startswith("clean") else "#0891b2" for c in rr.c])
        ax.set_ylim(0, 1)
        ax.set_ylabel("test mAP")
        ax2 = ax.twinx()
        ax2.plot(rr.c, rr.mean_mutual_info, "o-", color="#7c3aed")
        ax2.set_ylabel("mean MI", color="#7c3aed")
        ax.set_xticks(range(len(rr)), rr.c, rotation=35, ha="right")
        ax.set_title("Robustness under corruptions (full pipeline re-run)")
        fig.tight_layout()
        paths.append(FIG / "robustness.png")
        fig.savefig(paths[-1])
        plt.close(fig)
    return paths


def tables_md(r: dict, rb: dict | None) -> str:
    df = model_table(r)
    main_cols = ["Model", "mAP", "AP@0.5s", "AP@1.0s", "AP@1.5s", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1", "Brier", "ECE"]
    s = [f"<!-- AUTO-GENERATED by scripts/make_reports.py from experiments/latest (run {r['timestamp']}). Do not edit by hand. -->",
         "## Table 1 — Test results, all models", md_table(df, main_cols), "",
         "*Neural models A–C, the LSTM/TCN/pool variants and GRU motion-only: mean ± std over seeds " + str(r["seeds"]) +
         " (std shown for mAP). P/R/F1 are at the validation F1-optimal threshold.*", ""]
    keys = ["A_frame_visual_motion", "B_gru_visual", "gru_visual_delta_only", "gru_motion_only", "C_gru_visual_motion", "C2_gru_vdelta_motion", "D_single_mc_dropout_ts",
            "E_ensemble_mc_dropout_ts", "E_full_with_risk_engine"]
    s += ["## Table 2 — Ablation", md_table(df.set_index("key").loc[keys].reset_index(), ["Model", "mAP", "ROC-AUC", "F1", "Brier", "ECE"]), ""]
    ws = r["window_selection_val"]
    s += ["## Table 3 — Observation window (validation mAP, GRU visual+motion)", "| Window (s) | val mAP |", "|---|---|"]
    s += [f"| {w} | {fmt(v['mean'], v['std'])} |" for w, v in ws.items()]
    s += [f"\nSelected window: **{r['selected_window_s']} s**. Selected architecture: **{r['selected_architecture']['name']}** "
          f"(val mAP: " + ", ".join(f"{k} {v:.3f}" for k, v in r["selected_architecture"]["val_mAP"].items()) + ").", ""]
    E, D = r["models"]["E_ensemble_mc_dropout_ts"], r["models"]["D_single_mc_dropout_ts"]
    s += ["## Table 4 — Uncertainty and calibration (test)", "| | D (single + MC) | E (ensemble + MC) |", "|---|---|---|",
          f"| Temperature T (fit on val) | {D['temperature']:.3f} | {E['temperature']:.3f} |",
          f"| ECE before → after TS | {D['test_uncalibrated']['ece']:.3f} → {D['test']['ece']:.3f} | {E['test_uncalibrated']['ece']:.3f} → {E['test']['ece']:.3f} |",
          f"| Brier before → after TS | {D['test_uncalibrated']['brier']:.3f} → {D['test']['brier']:.3f} | {E['test_uncalibrated']['brier']:.3f} → {E['test']['brier']:.3f} |",
          f"| Error-detection AUROC (MI) | {fmt(D['error_auroc_mutual_info'])} | {fmt(E['error_auroc_mutual_info'])} |",
          f"| Error-detection AUROC (entropy) | {fmt(D['error_auroc_entropy'])} | {fmt(E['error_auroc_entropy'])} |",
          f"| Error-detection AUROC (p std) | {fmt(D['error_auroc_p_std'])} | {fmt(E['error_auroc_p_std'])} |", ""]
    s += ["Risk–coverage (E): accuracy when answering only the least-uncertain fraction", "| coverage | by MI | by entropy |", "|---|---|---|"]
    for a, b in zip(E["selective_risk_mutual_info"], E["selective_risk_entropy"]):
        s.append(f"| {a['coverage']:.1f} | {a['accuracy']:.3f} | {b['accuracy']:.3f} |")
    s.append("")
    s += ["## Table 5 — Streaming early warning on full test videos", "| System | thr (val) | detection rate | timely (≤3 s) | mean TTA (s) | median TTA (s) | FA / h | alarm after annotated alert |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in r["early_warning_test"].items():
        if k == "operating_curve_E":
            continue
        s.append(f"| {k} | {v['threshold']:.3f} | {fmt(v['detection_rate'])} | {fmt(v['timely_detection_rate'])} | {fmt(v['mean_tta_s'], p=2)} | {fmt(v['median_tta_s'], p=2)} | "
                 f"{fmt(v['false_alarms_per_hour'], p=1)} | {fmt(v['frac_alarm_after_annotated_alert'])} |")
    ew = r["early_warning_test"]["E_full_risk_engine"]
    s += [f"\nNormal-driving test time: {ew['negative_hours'] * 60:.1f} min. Positive test videos: {ew['n_positive_videos']}.",
          f"Risk thresholds (val): " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r["risk_thresholds_val"].items()),
          f"Risk-level counts on test timelines: {r['risk_level_counts_test']}", ""]
    s += ["## Table 6 — Permutation importance (test AP drop, top 10)", "| Feature group | ΔAP |", "|---|---|"]
    s += [f"| {k} | {v:.4f} |" for k, v in list(r["permutation_importance_test_ap_drop"].items())[:10]]
    s.append("")
    ef = r["efficiency"]
    s += ["## Table 7 — Efficiency", "| Item | Value |", "|---|---|",
          f"| Perception throughput (detect + track + flow + CNN) | {fmt(ef.get('feature_extraction_fps'), p=1)} frames/s on {r['device']} |",
          f"| Model latency, single deterministic | {ef['latency_ms_per_window_single_deterministic']:.2f} ms / window |",
          f"| Model latency, ensemble ×{r['config']['model']['ensemble_size']} × MC {r['config']['model']['mc_samples']} | {ef['latency_ms_per_window_ensemble_mc']:.2f} ms / window |",
          f"| Peak VRAM during training | {fmt(ef.get('peak_vram_gb_training'), p=2)} GB |",
          f"| Process RSS | {ef['process_rss_gb']:.2f} GB |", f"| Total experiment runtime | {r['runtime_minutes']:.1f} min |", ""]
    if rb:
        s += ["## Table 8 — Robustness (test clips, bundle E, full pipeline re-run)",
              "| Condition | mAP | ROC-AUC | F1 | ECE | mean MI |", "|---|---|---|---|---|---|"]
        for k, v in rb["conditions"].items():
            s.append(f"| {k} | {v['mAP']:.3f} | {v['roc_auc']:.3f} | {v['f1']:.3f} | {v['ece']:.3f} | {v['mean_mutual_info']:.4f} |")
    return "\n".join(s) + "\n"


def pdf_report(r: dict, rb: dict | None, figs: list[Path], tables: str) -> Path:
    from fpdf import FPDF

    def L(t):
        return t.encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(True, 14)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 17)
    pdf.multi_cell(0, 8, "SafeDrive AI: Explainable and Uncertainty-Aware Early Traffic Collision Prediction", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, L(f"Technical report - experiment run {r['timestamp']} on {r['environment'].get('gpu', r['device'])}. "
                         f"Nexar subset: {sum(r['n_videos'].values())} videos (train/val/test {r['n_videos']['train']}/"
                         f"{r['n_videos']['val']}/{r['n_videos']['test']}). All numbers generated from experiment files."))
    pdf.ln(3)
    for line in tables.splitlines():
        if line.startswith("<!--"):
            continue
        if line.startswith("## "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(0, 6, L(line[3:]), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Courier", "", 6.5)
        elif line.startswith("|---"):
            continue
        elif line.startswith("|"):
            pdf.multi_cell(0, 3.4, L(" ".join(c.strip()[:34].ljust(12) for c in line.strip("|").split("|"))), new_x="LMARGIN", new_y="NEXT")
        elif line.strip():
            pdf.set_font("Helvetica", "I", 7.5)
            pdf.multi_cell(0, 3.8, L(line.replace("**", "").replace("*", "")), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Courier", "", 6.5)
    for p in figs:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, p.stem.replace("_", " ").title(), new_x="LMARGIN", new_y="NEXT")
        pdf.image(str(p), w=170)
    out = resolve("reports/SafeDrive_AI_Technical_Report.pdf")
    pdf.output(str(out))
    return out


def slides(r: dict, rb: dict | None, figs: list[Path]) -> Path:
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.util import Inches, Pt

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    BG, FG, AC = RGBColor(0x0B, 0x10, 0x20), RGBColor(0xE5, 0xE9, 0xF5), RGBColor(0x22, 0xD3, 0xEE)

    def slide(title: str, bullets: list[str] | None = None, img: Path | None = None):
        s = prs.slides.add_slide(prs.slide_layouts[6])
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = BG
        tb = s.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12), Inches(0.9)).text_frame
        tb.text = title
        tb.paragraphs[0].font.size, tb.paragraphs[0].font.bold, tb.paragraphs[0].font.color.rgb = Pt(30), True, AC
        if bullets:
            w = Inches(6.0) if img else Inches(12)
            tf = s.shapes.add_textbox(Inches(0.6), Inches(1.4), w, Inches(5.6)).text_frame
            tf.word_wrap = True
            for i, b in enumerate(bullets):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                p.text = "• " + b
                p.font.size, p.font.color.rgb = Pt(17), FG
                p.space_after = Pt(8)
        if img is not None and img.exists():
            s.shapes.add_picture(str(img), Inches(6.9) if bullets else Inches(2.2), Inches(1.4),
                                 width=Inches(6.0) if bullets else Inches(8.8))
        return s

    E = r["models"]["E_ensemble_mc_dropout_ts"]
    ew = r["early_warning_test"]["E_full_risk_engine"]
    C = r["models"]["C_gru_visual_motion"]["test"]
    A = r["models"]["A_frame_visual_motion"]["test"]
    f = {p.stem: p for p in figs}
    slide("SafeDrive AI", ["Explainable and uncertainty-aware early traffic-collision prediction from dashcam video",
                           "Hệ thống AI dự đoán sớm nguy cơ va chạm giao thông, có khả năng giải thích và định lượng độ tin cậy",
                           "Vingroup applied-AI programme — personal project"])
    slide("Problem", ["Collisions develop in 1–3 s; warning earlier gives the driver time to react",
                      "False alarms destroy trust; black-box scores give no reason to act",
                      "Goal: predict risk BEFORE the event, say how sure we are, and show why",
                      "Constraint: one laptop, RTX 3050 4 GB"])
    slide("Data: Nexar Dashcam Collision Prediction", [
        f"Real dashcam videos, 1280×720 @30 fps, ~40 s; annotated event and alert times",
        f"Balanced subset used: {sum(r['n_videos'].values())} videos, split by video {r['n_videos']['train']}/{r['n_videos']['val']}/{r['n_videos']['test']}",
        "Protocol: observation ends 0.5 / 1.0 / 1.5 s before the event → mAP over horizons",
        "Nexar Open Data License (attribution, no re-identification)"])
    slide("Method", ["YOLOv8n + ByteTrack → 28 image-plane motion cues (looming, apparent τ, lateral, closing, flow)",
                     "ResNet-18 appearance embedding per frame (10 Hz)",
                     f"Temporal model selected on val: {r['selected_architecture']['name']}, window {r['selected_window_s']} s",
                     "Deep ensemble ×5 + MC-Dropout ×20 + temperature scaling",
                     "Risk engine: NORMAL / CAUTION / WARNING / HIGH RISK / UNCERTAIN",
                     "Grad-CAM, Integrated Gradients, permutation importance"])
    slide("Results: model comparison", [f"Proposed E: test mAP {E['test']['mAP']:.3f}, ROC-AUC {E['test']['roc_auc']:.3f}, F1 {E['test']['f1']:.3f}",
                                        f"C (GRU visual+motion): mAP {C['mAP']['mean']:.3f} ± {C['mAP']['std']:.3f}",
                                        f"A (no temporal): mAP {A['mAP']['mean']:.3f} ± {A['mAP']['std']:.3f}"], f.get("model_comparison"))
    slide("Ablation A–E", None, f.get("ablation"))
    slide("Early warning", [f"Detection rate {ew['detection_rate']:.0%} of collision videos before the event",
                            f"Mean warning time {ew['mean_tta_s']:.2f} s (median {ew['median_tta_s']:.2f} s)",
                            f"{ew['false_alarms_per_hour']:.1f} false alarms per hour on normal driving ({ew['negative_hours'] * 60:.0f} min)"],
          f.get("operating_curve"))
    slide("Uncertainty & calibration", [f"ECE {E['test_uncalibrated']['ece']:.3f} → {E['test']['ece']:.3f} after temperature scaling (T={E['temperature']:.2f})",
                                        f"Error-detection AUROC: MI {E['error_auroc_mutual_info']:.3f}, entropy {E['error_auroc_entropy']:.3f}",
                                        "UNCERTAIN state: abstain when ensemble members disagree"], f.get("reliability"))
    slide("Explainability", ["Grad-CAM through the full temporal model", "Integrated Gradients over time and per motion cue",
                             "Grounded factors: measured cue values vs training-set typical values (no LLM)"], f.get("feature_importance"))
    if rb:
        slide("Robustness", None, f.get("robustness"))
    rl = getattr(slides, "light", None)
    if rl:
        F, bm = rl["final"], (load_json(resolve("experiments/latest/pipeline_benchmark.json"))
                              if resolve("experiments/latest/pipeline_benchmark.json").exists() else {})
        slide("Optimisation (v2-light): selection", [
            "No algorithm kept by default: XGBoost, LightGBM, CatBoost, LogReg, light GRU/TCN, 4 feature sets, 3 windows",
            "Single 60-video validation ranked models unreliably → selection by 5-fold CV on train+val (340 videos)",
            f"Selected (cheapest within 1 SE of best): {rl['selected']['config']} — no CNN",
            "Absolute ResNet features add nothing under CV (XGB motion 0.614 vs + CNN 0.612, 400-video CV)"], f.get("v2_cv_accuracy_vs_cost"))
        slide("Optimisation (v2-light): results", [
            f"Test mAP {F['test']['mAP']:.3f} (v1 0.582; Δ n.s.), ROC-AUC {F['test']['roc_auc']:.3f}, F1 {F['test']['f1']:.3f}",
            f"Calibrated: ECE {F['test_uncalibrated']['ece']:.3f} → {F['test']['ece']:.3f} (Platt, chosen by CV)",
            f"Prediction {rl['efficiency']['latency_ms_per_window_ensemble']:.1f} ms/window on CPU; no GPU needed for prediction",
            (f"End-to-end {bm['v2_light@cuda']['end_to_end_fps']:.0f} fps GPU, {bm['v2_light@cpu']['end_to_end_fps']:.0f} fps CPU-only"
             if "v2_light@cpu" in bm else ""),
            "Early warning remains the weak point (few timely alarms at ≤10 FA/h)"], f.get("v2_throughput"))
        if f.get("v2_robustness_vs_v1"):
            slide("Robustness: v1 vs v2-light", None, f.get("v2_robustness_vs_v1"))
    ew3 = getattr(slides, "v3", None)
    if ew3:
        cp3 = resolve("experiments/latest/ew_heldout_compare.json")
        cmp3 = load_json(cp3) if cp3.exists() else None
        b3 = ["All 1,500 Nexar videos: dev 1,276 (5-fold CV), held-out 224 (test 60 + test_ext 164)",
              "Kept on dev (paired bootstrap): vehicle-interaction cues (ΔS +0.025) and dense-negative training (ΔS +0.023)",
              "Not kept: longer horizons (significantly worse), alert-aligned labels, hard negatives, 2 s window, EMA / k ≠ 3",
              "Ego-motion compensation: no streaming gain on stable footage, but +0.1–0.2 mAP under camera shake (v3-stabilized)"]
        if cmp3:
            m = cmp3["models"]
            b3.append(f"Held-out recall @5/10/20 FA/h: v3 {m['v3_selected']['recall@5']:.2f}/{m['v3_selected']['recall@10']:.2f}/"
                      f"{m['v3_selected']['recall@20']:.2f} vs v2 {m['v2light_deployed_400']['recall@5']:.2f}/"
                      f"{m['v2light_deployed_400']['recall@10']:.2f}/{m['v2light_deployed_400']['recall@20']:.2f} (not significant)")
        slide("v3: early-warning optimisation", b3, f.get("v3_stage_tests"))
        if f.get("v3_robustness"):
            slide("v3: robustness incl. camera shake", None, f.get("v3_robustness"))
    slide("Limitations & future work", ["Subset of Nexar; small test set → wide confidence intervals",
                                        "Image-plane motion, not physical speed or TTC; no camera calibration",
                                        "Detector not fine-tuned; no MOT ground truth for ID switches",
                                        "Next: full dataset, depth / ego-motion, video backbones, on-vehicle latency study"])
    slide("Demo", ["streamlit run app.py → http://localhost:8501", "Video Analysis → Collision Prediction → Explainable AI → Uncertainty → Reports"])
    out = resolve("reports/SafeDrive_AI_Slides.pptx")
    prs.save(str(out))
    return out


def main() -> int:
    r = load_json(resolve("experiments/latest/results.json"))
    rbp = resolve("experiments/latest/robustness.json")
    rb = load_json(rbp) if rbp.exists() else None
    figs = figures(r, rb)
    tables = tables_md(r, rb)
    from report_light import light_artifacts
    lf, lt, rl = light_artifacts(r, rb)
    figs += lf
    slides.light = rl
    from report_v3 import v3_artifacts
    vf, vt, ew = v3_artifacts()
    figs += vf
    slides.v3 = ew
    tables = "# Part A — v1 study (LSTM ensemble)\n" + tables + lt + vt
    resolve("docs/RESULTS_TABLES.md").write_text(tables, encoding="utf-8")
    pdf = pdf_report(r, rb, figs, tables)
    ppt = slides(r, rb, figs)
    print("figures:", [p.name for p in figs])
    print("wrote docs/RESULTS_TABLES.md,", pdf.name, ppt.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
