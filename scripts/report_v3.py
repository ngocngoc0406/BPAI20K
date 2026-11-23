"""v3 early-warning study: figures + markdown tables for make_reports.py (all numbers read from experiment files)."""
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


def v3_artifacts() -> tuple[list[Path], str, dict | None]:
    ew = _opt("results_ew_full.json")
    if not ew:
        return [], "", None
    cmp_, rb2, rb3, rbs = _opt("ew_heldout_compare.json"), _opt("robustness_v2_v3caches.json"), _opt("robustness_v3.json"), _opt("robustness_v3_stabilized.json")
    bm, probe = _opt("pipeline_benchmark_v3.json") or {}, _opt("shake_dev_probe.json")
    FIG.mkdir(parents=True, exist_ok=True)
    figs: list[Path] = []
    rows = pd.DataFrame(ew["rows"])
    # fig 1: stage A/B streaming score with paired CIs
    tests = {**{k: v for k, v in ew["stageA_tests"].items()}, **{k.split("|W")[0] + (" (W2)" if k.endswith("W2") else ""): v for k, v in ew["stageB_tests"].items()}}
    names = list(tests)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = np.arange(len(names))
    mean = [tests[n]["mean"] for n in names]
    lo = [tests[n]["mean"] - tests[n]["ci90"][0] for n in names]
    hi = [tests[n]["ci90"][1] - tests[n]["mean"] for n in names]
    col = ["#16a34a" if tests[n]["ci90"][0] > 0 else "#94a3b8" for n in names]
    for i in range(len(names)):
        ax.errorbar([mean[i]], [y[i]], xerr=[[lo[i]], [hi[i]]], fmt="o", color=col[i], capsize=3)
    ax.axvline(0, color="k", lw=.8)
    ax.set_yticks(y, [n[:46] for n in names], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Δ streaming score S vs reference (paired video bootstrap, 90 % CI)")
    ax.set_title("v3 study on 1,276 dev videos (green = kept)")
    fig.tight_layout()
    figs.append(FIG / "v3_stage_tests.png")
    fig.savefig(figs[-1], dpi=140)
    plt.close(fig)
    if rb2 and rb3:
        cond = list(rb2["conditions"])
        x = np.arange(len(cond))
        fig, ax = plt.subplots(figsize=(8, 3.4))
        series = [("v2-light", rb2, "#94a3b8"), ("v3", rb3, "#0891b2")] + ([("v3-stabilized", rbs, "#16a34a")] if rbs else [])
        wdt = 0.8 / len(series)
        for i, (lab, rb, c) in enumerate(series):
            ax.bar(x + (i - (len(series) - 1) / 2) * wdt, [rb["conditions"][k]["mAP"] for k in cond], wdt, label=lab, color=c)
        ax.set_xticks(x, cond, rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("test mAP (60 videos)")
        ax.legend(frameon=False, ncol=3)
        ax.set_title("Robustness incl. camera shake (identical corrupted clips)")
        fig.tight_layout()
        figs.append(FIG / "v3_robustness.png")
        fig.savefig(figs[-1], dpi=140)
        plt.close(fig)
    F = ew["final"]
    L = ["", "# Part C — Early-warning optimisation (v3)", f"<!-- from experiments/latest/results_ew_full.json (run {ew['timestamp']}) -->", "",
         f"Dev = {ew['n_dev']} videos (5-fold grouped CV, OOF streams); held-out = original test {ew['n_test']} + test_ext {ew['n_test_ext']} videos.",
         f"Primary metric S = mean recall at 5 / 10 / 20 false alarms per hour (valid alarm: onset within 5 s before the event). Rule: {ew['rule']}.", "",
         "## Table C1 — Stage A (features) and B (labels / horizon / window), OOF dev",
         "| Stage | Variant | S | R@5 | R@10 | R@20 | EDR@1/2/3 s (10 FA/h) | OOF mAP |", "|---|---|---|---|---|---|---|---|"]
    for _, r in rows[rows.stage.isin(["A", "B"])].iterrows():
        L.append(f"| {r['stage']} | {r['name']} | {r['S']:.3f} | {r['recall@5']:.3f} | {r['recall@10']:.3f} | {r['recall@20']:.3f} | "
                 f"{r['edr@1s@10FA']:.3f} / {r['edr@2s@10FA']:.3f} / {r['edr@3s@10FA']:.3f} | {r['oof_mAP']:.3f} |")
    L += ["", "Paired tests (Δ S, 90 % CI; Stage A also Δ OOF mAP, 95 % CI):", ""]
    for k, v in ew["stageA_tests"].items():
        L.append(f"- {k}: ΔS {v['mean']:+.3f} [{v['ci90'][0]:+.3f}, {v['ci90'][1]:+.3f}]; ΔmAP {v['map']['mean']:+.3f} [{v['map']['ci95'][0]:+.3f}, {v['map']['ci95'][1]:+.3f}]")
    for k, v in ew["stageB_tests"].items():
        L.append(f"- {k}: ΔS {v['mean']:+.3f} [{v['ci90'][0]:+.3f}, {v['ci90'][1]:+.3f}]")
    s = ew["selected"]
    L += ["", f"**Selected:** {s['features']} · labels {s['labels']} · W {s['W']:g} s · k {s['k']} · EMA α {s['alpha']}.", "",
          "## Table C2 — Held-out, thresholds fixed on dev OOF streams", "| Model | Set | mAP | ROC-AUC | F1 | ECE | R@5 (FA/h) | R@10 (FA/h) | R@20 (FA/h) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m in ("baseline_v2config", "selected_v3"):
        for h in ("test", "test_ext", "test_all"):
            if h in F.get(m, {}):
                x = F[m][h]
                sr = x["streaming"]
                L.append(f"| {m} | {h} ({x['n_videos']}) | {x['mAP']:.3f} | {x['roc_auc']:.3f} | {x['f1']:.3f} | {x['ece']:.3f} | "
                         + " | ".join(f"{sr[f'@{b}FA']['recall']:.3f} ({sr[f'@{b}FA']['fa_per_hour']:.1f})" for b in (5, 10, 20)) + " |")
    if cmp_:
        L += ["", f"## Table C3 — Held-out at matched false-alarm rates ({cmp_['n_videos']} videos, {cmp_['neg_hours']:.2f} h normal driving)",
              "| Model | R@5 | R@10 | R@20 | S | EDR@1/2/3 s at 10 FA/h |", "|---|---|---|---|---|---|"]
        for k, v in cmp_["models"].items():
            L.append(f"| {k} | {v['recall@5']:.3f} | {v['recall@10']:.3f} | {v['recall@20']:.3f} | {v['S']:.3f} | "
                     f"{v.get('edr@1s@10', np.nan):.3f} / {v.get('edr@2s@10', np.nan):.3f} / {v.get('edr@3s@10', np.nan):.3f} |")
        for k, v in cmp_["paired"].items():
            L.append(f"\nPaired bootstrap {k}: ΔS {v['S']['mean']:+.3f} [{v['S']['ci90'][0]:+.3f}, {v['S']['ci90'][1]:+.3f}], P = {v['S']['p_better']:.2f}.")
    if rb2 and rb3:
        L += ["", "## Table C4 — Robustness, test clips (mAP)", "| Condition | v2-light | v3 | v3-stabilized |", "|---|---|---|---|"]
        for c in rb2["conditions"]:
            L.append(f"| {c} | {rb2['conditions'][c]['mAP']:.3f} | {rb3['conditions'][c]['mAP']:.3f} | "
                     f"{rbs['conditions'][c]['mAP']:.3f} |" if rbs else f"| {c} | {rb2['conditions'][c]['mAP']:.3f} | {rb3['conditions'][c]['mAP']:.3f} | — |")
    if probe:
        L += ["", "## Table C5 — Dev camera-shake probe (train 280 → val 60 clips, mAP)", "| Condition | " + " | ".join(next(iter(probe.values())).keys()) + " |",
              "|---" * (1 + len(next(iter(probe.values())))) + "|"]
        for c, row in probe.items():
            L.append(f"| {c} | " + " | ".join(f"{v['mAP']:.3f}" for v in row.values()) + " |")
    if bm:
        L += ["", "## Table C6 — End-to-end efficiency (5 test videos)", "| Pipeline | FPS | × real time | prediction ms/frame | peak VRAM GB | RSS GB |", "|---|---|---|---|---|---|"]
        for k, v in bm.items():
            if isinstance(v, dict):
                L.append(f"| {k} | {v['end_to_end_fps']:.1f} | {v['realtime_factor_10hz']:.1f} | {v['predict_ms_per_frame']:.2f} | {v['peak_vram_gb']:.2f} | {v['process_rss_gb']:.2f} |")
    return figs, "\n".join(L) + "\n", ew
