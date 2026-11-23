"""v3.3 failure analysis: score-distribution overlap and temporal consistency of the deployed v3 output on the 224 held-out
videos (clean caches). Descriptive only. Output: experiments/latest/v33_temporal_overlap.json"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.data.dataset_loader import load_dataset  # noqa: E402
from src.evaluation.streaming import MAX_LEAD  # noqa: E402
from src.pipeline import SafeDrivePipeline  # noqa: E402
from src.preprocessing.feature_cache import cache_path, load_features  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.io import load_json, save_json  # noqa: E402


def runs(mask: np.ndarray) -> list[int]:
    out, n = [], 0
    for m in mask:
        if m:
            n += 1
        elif n:
            out.append(n)
            n = 0
    return out + ([n] if n else [])


def main() -> int:
    cfg = load_config(None)
    sp = load_json(resolve("data/processed/splits_full.json"))
    recs = {r.video_id: r for r in load_dataset(resolve(cfg["paths"]["annotations"]), resolve(cfg["paths"]["raw_dir"]))}
    fdir = resolve(cfg["paths"]["features_dir"])
    pipe = SafeDrivePipeline(resolve("models/safedrive_v3"), device="cpu")
    _, thr = pipe.alarm_threshold("balanced")
    warm = float(pipe.predictor.meta.get("warmup_s", pipe.predictor.window_s))
    rows = []
    for v in sp["test"] + sp["test_ext"]:
        f = load_features(cache_path(fdir, v))
        r = recs[v]
        tl = pipe.analyze_features(f, profile="balanced").timeline
        t, p = tl.time_s.to_numpy(), tl.p_mean.to_numpy()
        pos = r.label == 1
        m = (t >= warm) & ((t < r.time_of_event) if pos else True)
        pe = (t >= r.time_of_event - MAX_LEAD) & (t < r.time_of_event) if pos else m
        pp = p[m]
        rr = runs(pp >= thr)
        rows.append({"pos": pos, "peak": float(p[pe].max()) if pe.any() else np.nan, "mean": float(pp.mean()),
                     "dp": float(np.mean(np.abs(np.diff(pp)))), "minutes": len(pp) / 600,
                     "crossings": int(np.sum(np.diff((pp >= thr).astype(int)) == 1) + int(pp[0] >= thr)),
                     "runs": rr, "early_peak": float(p[m & (t < r.time_of_event - MAX_LEAD)].max()) if pos and (m & (t < r.time_of_event - MAX_LEAD)).any() else np.nan})
    y = np.array([r["pos"] for r in rows])
    peak = np.array([r["peak"] for r in rows])
    ok = ~np.isnan(peak)
    out = {"threshold_balanced": thr,
           "auc_video_peak_pre_event_vs_negative_peak": float(roc_auc_score(y[ok], peak[ok])),
           "auc_video_mean_p": float(roc_auc_score(y, [r["mean"] for r in rows])),
           "peak_quantiles": {lab: np.nanpercentile(peak[y == (lab == "pos")], [10, 25, 50, 75, 90]).round(3).tolist() for lab in ("pos", "neg")},
           "pos_early_peak_median (before the 5 s window)": float(np.nanmedian([r["early_peak"] for r in rows if r["pos"]])),
           "mean_abs_step_change_p": {lab: float(np.mean([r["dp"] for r in rows if r["pos"] == (lab == "pos")])) for lab in ("pos", "neg")},
           "threshold_crossings_per_min_neg": float(sum(r["crossings"] for r in rows if not r["pos"]) / sum(r["minutes"] for r in rows if not r["pos"])),
           "above_threshold_runs_neg": {"n": int(sum(len(r["runs"]) for r in rows if not r["pos"])),
                                        "share_shorter_than_3_steps": float(np.mean([x < 3 for r in rows if not r["pos"] for x in r["runs"]] or [np.nan]))},
           "above_threshold_runs_pos_pre_event": {"n": int(sum(len(r["runs"]) for r in rows if r["pos"])),
                                                  "share_shorter_than_3_steps": float(np.mean([x < 3 for r in rows if r["pos"] for x in r["runs"]] or [np.nan]))}}
    save_json(out, resolve("experiments/latest/v33_temporal_overlap.json"))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
