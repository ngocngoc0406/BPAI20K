"""v2 lightweight-study figures + markdown tables for make_reports.py (all numbers read from experiment files)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.config import resolve  # noqa: E402
from src.utils.io import load_json  # noqa: E402

FIG = resolve("reports/figures")


def _opt(name: str):
    p = resolve(f"experiments/latest/{name}")
    return load_json(p) if p.exists() else None


def light_artifacts(r: dict, rb: dict | None) -> tuple[list[Path], str, dict | None]:
    rl = _opt("results_light.json")
    if not rl or "cv_grid" not in rl:
        return [], "", None
    rbl, bm, cvs, pp = _opt("robustness_light.json"), _opt("pipeline_benchmark.json") or {}, _opt("cv_study.json"), _opt("perception_profile.json")
    FIG.mkdir(parents=True, exist_ok=True)
    figs: list[Path] = []
    g = pd.DataFrame(rl["cv_grid"])
    sel = rl["selected"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for cnn, c, lab in [(False, "#0891b2", "no CNN"), (True, "#ea580c", "needs CNN")]:
        d = g[g.needs_cnn == cnn]
        ax.errorbar(d.fit_s_per_fold, d.oof_mAP, yerr=d.fold_se, fmt="o", color=c, label=lab, alpha=.8, capsize=2)
    ax.scatter([sel["fit_s_per_fold"]], [sel["oof_mAP"]], s=170, facecolors="none", edgecolors="k", label="selected")
    ax.set_xscale("log")
    ax.set_xlabel("training time per fold (s)")
    ax.set_ylabel("OOF mAP (train+val 5-fold CV)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("v2 selection: accuracy vs cost")
    fig.tight_layout()
    figs.append(FIG / "v2_cv_accuracy_vs_cost.png")
    fig.savefig(figs[-1], dpi=140)
    plt.close(fig)
    if rbl and rb:
        cond = list(rbl["conditions"])
        x = np.arange(len(cond))
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        ax.bar(x - .2, [rb["conditions"][c]["mAP"] for c in cond], .4, label="v1 LSTM + ResNet", color="#94a3b8")
        ax.bar(x + .2, [rbl["conditions"][c]["mAP"] for c in cond], .4, label="v2-light", color="#0891b2")
        ax.set_xticks(x, cond, rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("test mAP")
        ax.legend(frameon=False)
        ax.set_title("Robustness on identical corrupted clips")
        fig.tight_layout()
        figs.append(FIG / "v2_robustness_vs_v1.png")
        fig.savefig(figs[-1], dpi=140)
        plt.close(fig)
    if bm:
        keys = [k for k in ("v1_lstm@cuda", "v2_light@cuda", "v2_light@cpu") if k in bm]
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        ax.bar(keys, [bm[k]["end_to_end_fps"] for k in keys], color=["#94a3b8", "#0891b2", "#16a34a"][: len(keys)])
        ax.axhline(10, color="#ef4444", ls="--", lw=1)
        ax.text(len(keys) - .5, 10.8, "real time (10 Hz)", color="#ef4444", fontsize=7, ha="right")
        ax.set_ylabel("end-to-end frames/s")
        ax.set_title("Pipeline throughput (5 test videos)")
        fig.tight_layout()
        figs.append(FIG / "v2_throughput.png")
        fig.savefig(figs[-1], dpi=140)
        plt.close(fig)

    F, E = rl["final"], r["models"]["E_ensemble_mc_dropout_ts"]
    ewl, ew1 = rl["early_warning_test"], r["early_warning_test"]
    bt = rl.get("bootstrap_vs_v1", {})
    L = ["", "# Part B — Lightweight optimisation (v2)", f"<!-- from experiments/latest/results_light.json (run {rl['timestamp']}) -->", "",
         f"Protocol: {rl.get('protocol', '')}.", f"Selection rule: {rl['selection_rule']}.", "",
         "## Table B1 — 5-fold CV on train+val (340 videos, out-of-fold windows), top 15",
         "| Config | needs CNN | OOF mAP | fold mean ± SE | ROC-AUC | fit s/fold |", "|---|---|---|---|---|---|"]
    for row in g.sort_values("oof_mAP", ascending=False).head(15).itertuples():
        L.append(f"| {row.config} | {row.needs_cnn} | {row.oof_mAP:.3f} | {row.fold_mean:.3f} ± {row.fold_se:.3f} | {row.roc_auc:.3f} | {row.fit_s_per_fold:.2f} |")
    L += ["", f"Best: {rl['best']['config']} ({rl['best']['oof_mAP']:.3f}); tolerance {rl['margin']:.3f}; "
          f"**selected: {sel['config']}** (OOF mAP {sel['oof_mAP']:.3f}).", ""]
    if cvs:
        L += ["## Table B2 — Corroborating 5-fold CV on all 400 videos (not used for selection)",
              "| Config | OOF mAP | fold mean ± sd | ROC-AUC |", "|---|---|---|---|"]
        for k, v in sorted(cvs["summary"].items(), key=lambda kv: -kv[1]["oof_mAP"]):
            L.append(f"| {k} | {v['oof_mAP']:.3f} | {v['fold_mAP_mean']:.3f} ± {v['fold_mAP_std']:.3f} | {v['roc_auc']:.3f} |")
        L.append("")
    m = bt.get("models", {})

    def ci(k):
        return f" [{m[k]['ci95'][0]:.3f}, {m[k]['ci95'][1]:.3f}]" if k in m else ""

    L += ["## Table B3 — Held-out test (identical 180 windows): v2.1-light vs v1", "| Metric | v2.1-light | v1 E (LSTM ensemble) |", "|---|---|---|",
          f"| mAP [95 % CI] | {F['test']['mAP']:.3f}{ci('V21_FINAL')} | {E['test']['mAP']:.3f}{ci('E_ensemble_mc_dropout_ts')} |",
          f"| AP @0.5 / 1.0 / 1.5 s | {F['test']['ap_h0.5']:.3f} / {F['test']['ap_h1.0']:.3f} / {F['test']['ap_h1.5']:.3f} | "
          f"{E['test']['ap_h0.5']:.3f} / {E['test']['ap_h1.0']:.3f} / {E['test']['ap_h1.5']:.3f} |",
          f"| ROC-AUC / PR-AUC | {F['test']['roc_auc']:.3f} / {F['test']['pr_auc']:.3f} | {E['test']['roc_auc']:.3f} / {E['test']['pr_auc']:.3f} |",
          f"| Precision / Recall / F1 | {F['test']['precision']:.3f} / {F['test']['recall']:.3f} / {F['test']['f1']:.3f} | "
          f"{E['test']['precision']:.3f} / {E['test']['recall']:.3f} / {E['test']['f1']:.3f} |",
          f"| ECE before → after calibration | {F['test_uncalibrated']['ece']:.3f} → {F['test']['ece']:.3f} ({F['config']['calibration']}) | "
          f"{E['test_uncalibrated']['ece']:.3f} → {E['test']['ece']:.3f} (temperature) |",
          f"| Brier | {F['test']['brier']:.3f} | {E['test']['brier']:.3f} |",
          f"| Error-detection AUROC (entropy / MI) | {F['error_auroc_entropy']:.3f} / {F['error_auroc_mutual_info']:.3f} | "
          f"{E['error_auroc_entropy']:.3f} / {E['error_auroc_mutual_info']:.3f} |",
          f"| Accuracy at 100 % → 50 % coverage (abstain by entropy) | {F['selective_risk_entropy'][0]['accuracy']:.3f} → "
          f"{F['selective_risk_entropy'][-1]['accuracy']:.3f} | {E['selective_risk_entropy'][0]['accuracy']:.3f} → {E['selective_risk_entropy'][-1]['accuracy']:.3f} |"]
    for a1, a2, lab in [("FINAL_prob_only", "E_ensemble_prob_only", "probability alarms"), ("FINAL_risk_engine", "E_full_risk_engine", "risk-engine alarms")]:
        x, y = ewl[a1], ew1[a2]
        L.append(f"| Early warning, {lab}: detection / timely (≤3 s) / FA per h / median TTA | {x['detection_rate']:.3f} / "
                 f"{x['timely_detection_rate']:.3f} / {x['false_alarms_per_hour']:.1f} / {x['median_tta_s']:.2f} s | {y['detection_rate']:.3f} / "
                 f"{y['timely_detection_rate']:.3f} / {y['false_alarms_per_hour']:.1f} / {y['median_tta_s']:.2f} s |")
    L.append("")
    for k, v in bt.get("paired", {}).items():
        L.append(f"Paired video bootstrap {k}: Δ mAP {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}], P(first better) = {v['p_a_better']:.2f}.  ")
    ef = rl["efficiency"]
    L += ["", "## Table B4 — Efficiency (same laptop: RTX 3050 4 GB / CPU)", "| Item | v2-light | v1 |", "|---|---|---|",
          f"| Prediction latency per window | {ef['latency_ms_per_window_ensemble']:.2f} ms CPU (×{F['config']['n_members']}); "
          f"{ef['latency_ms_per_window_single']:.2f} ms single member | {r['efficiency']['latency_ms_per_window_ensemble_mc']:.1f} ms GPU (×5 × MC 20) |",
          f"| Prediction model size | {ef['model_size_mb']:.2f} MB | 1.5 MB + ResNet-18 weights (~45 MB) |"]
    for k in ("v2_light@cuda", "v2_light@cpu", "v1_lstm@cuda"):
        if k in bm:
            b = bm[k]
            L.append(f"| End-to-end {k} | {b['end_to_end_fps']:.1f} fps = {b['realtime_factor_10hz']:.1f}× real time; peak VRAM {b['peak_vram_gb']:.2f} GB | |")
    if pp:
        d = pp["detectors"]
        L += ["", f"Perception profile (ms/frame): decode+resize {pp['decode_resize_ms_per_frame']:.1f}; YOLOv8n@640+ByteTrack "
              f"{d['yolov8n@640']['ms_per_frame']:.1f} (@480 {d['yolov8n@480']['ms_per_frame']:.1f}, @320 {d['yolov8n@320']['ms_per_frame']:.1f}; "
              f"YOLO11n@640 {d['yolo11n@640']['ms_per_frame']:.1f}); Farneback flow {pp['flow_ms_per_frame']:.1f}; ResNet-18 {pp['resnet18_ms_per_frame']:.1f}; "
              f"frame difference {pp['frame_diff_ms_per_frame']:.2f}. Smaller YOLO inputs were not faster (overhead-bound) but lost "
              f"{1 - d['yolov8n@480']['recall_vs_v8n640']:.0%} (480) / {1 - d['yolov8n@320']['recall_vs_v8n640']:.0%} (320) of the boxes found at 640."]
    if rbl and rb:
        L += ["", "## Table B5 — Robustness (identical corrupted test clips, full perception re-run)",
              "| Condition | v2-light mAP / F1 / ECE | v1 mAP / F1 / ECE |", "|---|---|---|"]
        for c in rbl["conditions"]:
            a, b = rbl["conditions"][c], rb["conditions"][c]
            L.append(f"| {c} | {a['mAP']:.3f} / {a['f1']:.3f} / {a['ece']:.3f} | {b['mAP']:.3f} / {b['f1']:.3f} / {b['ece']:.3f} |")
    cr = rl.get("caution_recalibration")
    if cr:
        L += ["", f"CAUTION recalibration (dev OOF streams, {cr['normal_driving_hours_oof']:.2f} h of normal driving): threshold "
              f"{cr['old']:.3f} → {cr['new']:.3f}; normal-driving time at CAUTION-or-above {cr['fraction_of_normal_time_at_or_above_old']:.0%} → "
              f"{cr['fraction_of_normal_time_at_or_above_new']:.0%}. Test level fractions afterwards: {cr.get('test_level_fractions_after')}."]
    L += ["", "## Table B6 — Permutation importance, v2 (test AP drop, top 10)", "| Channel | ΔAP |", "|---|---|"]
    L += [f"| {k} | {v:.4f} |" for k, v in list(rl["permutation_importance_test_ap_drop"].items())[:10]]
    return figs, "\n".join(L) + "\n", rl
