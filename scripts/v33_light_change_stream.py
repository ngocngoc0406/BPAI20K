"""v3.3 rapid-illumination-change stress test, FULL STREAMS, deployed alarm logic, paired with the clean baseline.

The 60 original test videos are re-processed end to end (detection, tracking, flow, frame difference, ego-motion) with the
`light_change` corruption (src/preprocessing/video_transform.py, STRESS_V33; seeded per video) and scored with the deployed v3
pipeline (SafeDrivePipeline.analyze_features, all three profiles). The baseline is the clean full-stream cache of the SAME
videos (same extractor). Nothing is selected or tuned here.

Reported per condition: per-video valid alarm (<= 5 s before the event), recall with Wilson CI, false alarms and FA/h
(exact Poisson CI) on the normal videos, lead times, P distributions, detections per step; paired: McNemar exact test on
recall, paired bootstrap over videos for delta recall and delta FA/h, mean per-video shift of P.
Output: experiments/latest/v33_light_change_stream.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from realworld_benchmark_strata import poisson_ci, wilson  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from src.data.dataset_loader import load_dataset  # noqa: E402
from src.evaluation.streaming import MAX_LEAD  # noqa: E402
from src.features.extractor import FeatureExtractor  # noqa: E402
from src.pipeline import SafeDrivePipeline  # noqa: E402
from src.preprocessing.feature_cache import cache_path, load_features, save_features  # noqa: E402
from src.preprocessing.video_transform import make_transform  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.io import load_json, save_json  # noqa: E402

PROFILES = ("conservative", "balanced", "sensitive")


def score(pipe, f, r, warm):
    out = {}
    pos = r.label == 1 and r.time_of_event is not None
    for prof in PROFILES:
        res = pipe.analyze_features(f, profile=prof)
        ons = np.asarray(res.alarms, float)
        if pos:
            v = ons[(ons >= r.time_of_event - MAX_LEAD) & (ons < r.time_of_event)]
            out[prof] = {"hit": bool(len(v)), "lead": float(r.time_of_event - v.min()) if len(v) else None}
        else:
            out[prof] = {"fa": int(len(ons))}
        if prof == "balanced":
            tl = res.timeline
            m = (tl.time_s < r.time_of_event) if pos else np.ones(len(tl), bool)
            out["p"] = tl.p_mean.to_numpy()[m]
            out["mi_mean"] = float(tl.mutual_info.to_numpy()[m].mean())
    n_det = np.bincount(f.tracks[:, 0].astype(int), minlength=len(f.times)) if len(f.tracks) else np.zeros(len(f.times))
    out["det_per_step"] = float(n_det.mean())
    out["zero_det_share"] = float(np.mean(n_det == 0))
    out["neg_hours"] = 0.0 if pos else max(0.0, float(f.times[-1]) - warm) / 3600
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--severities", default="1,2")
    ap.add_argument("--keep-awake", action="store_true")
    a = ap.parse_args()
    if a.keep_awake and sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    cfg = load_config(None)
    sp = load_json(resolve("data/processed/splits_full.json"))
    recs = {r.video_id: r for r in load_dataset(resolve(cfg["paths"]["annotations"]), resolve(cfg["paths"]["raw_dir"]))}
    fdir = resolve(cfg["paths"]["features_dir"])
    vids = [v for v in sp["test"] if cache_path(fdir, v).exists()]
    pipe = SafeDrivePipeline(resolve("models/safedrive_v3"), device="cpu")
    warm = float(pipe.predictor.meta.get("warmup_s", pipe.predictor.window_s))
    fx = FeatureExtractor(cfg, get_device(), use_cnn=False)
    # the baseline is re-extracted with the SAME extractor / device as the corrupted runs; the original cache is a sanity check
    conds = {"clean_original_cache": {v: score(pipe, load_features(cache_path(fdir, v)), recs[v], warm) for v in vids}}
    extract_s = {}
    for name, (tf, sev) in [("clean", ("clean", 0))] + [(f"light_change_{s}", ("light_change", int(s))) for s in a.severities.split(",")]:
        conds[name], t0 = {}, time.perf_counter()
        for i, v in enumerate(vids):
            cp = cache_path(fdir, v, f"stream_v33_{name}")
            if cp.exists():
                f = load_features(cp)
            else:
                f, _ = fx.extract(recs[v].path, v, transform=make_transform(tf, sev, int(v) if v.isdigit() else 0))
                save_features(f, cp)
            conds[name][v] = score(pipe, f, recs[v], warm)
            if i % 10 == 0:
                print(name, i, len(vids), flush=True)
        extract_s[name] = time.perf_counter() - t0

    pos = [v for v in vids if recs[v].label == 1]
    neg = [v for v in vids if recs[v].label == 0]
    rng = np.random.default_rng(33)
    result = {"description": __doc__, "n_videos": len(vids), "n_pos": len(pos), "n_neg": len(neg),
              "neg_hours": round(sum(conds["clean"][v]["neg_hours"] for v in neg), 4), "seconds": extract_s, "conditions": {}}
    base = conds["clean"]
    for name, c in conds.items():
        hrs = sum(c[v]["neg_hours"] for v in neg)
        o = {"det_per_step": float(np.mean([c[v]["det_per_step"] for v in vids])),
             "zero_detection_step_share": float(np.mean([c[v]["zero_det_share"] for v in vids])),
             "p_pos_pre_event_mean": float(np.mean(np.concatenate([c[v]["p"] for v in pos]))),
             "p_neg_mean": float(np.mean(np.concatenate([c[v]["p"] for v in neg]))),
             "p_pos_peak_median": float(np.median([c[v]["p"].max() for v in pos])),
             "p_neg_peak_median": float(np.median([c[v]["p"].max() for v in neg])),
             "mi_mean": float(np.mean([c[v]["mi_mean"] for v in vids]))}
        if name != "clean":
            o["p_shift_mean_per_video"] = float(np.mean([c[v]["p"].mean() - base[v]["p"].mean() for v in vids]))
            o["p_shift_ci95"] = [round(float(x), 4) for x in np.percentile(
                [np.mean([c[v]["p"].mean() - base[v]["p"].mean() for v in rng.choice(vids, len(vids))]) for _ in range(2000)], [2.5, 97.5])]
            o["det_shift_per_step"] = float(np.mean([c[v]["det_per_step"] - base[v]["det_per_step"] for v in vids]))
        for prof in PROFILES:
            hits = np.array([c[v][prof]["hit"] for v in pos])
            fa = np.array([c[v][prof]["fa"] for v in neg])
            leads = [c[v][prof]["lead"] for v in pos if c[v][prof]["lead"] is not None]
            po = {"recall": round(float(hits.mean()), 3), "hits": int(hits.sum()), "recall_ci95": wilson(int(hits.sum()), len(pos)),
                  "false_alarms": int(fa.sum()), "fa_h": round(float(fa.sum() / hrs), 2), "fa_h_ci95": poisson_ci(int(fa.sum()), hrs),
                  "warned_ge_2s": round(float(np.sum(np.array(leads) >= 2) / len(pos)), 3), "median_lead_s": float(np.median(leads)) if leads else None}
            if name != "clean":
                bh = np.array([base[v][prof]["hit"] for v in pos])
                bfa = np.array([base[v][prof]["fa"] for v in neg])
                b01, b10 = int(np.sum(~bh & hits)), int(np.sum(bh & ~hits))
                po["paired"] = {"gained": b01, "lost": b10,
                                "mcnemar_exact_p": float(binomtest(b01, b01 + b10, 0.5).pvalue) if b01 + b10 else 1.0,
                                "delta_recall": round(float(hits.mean() - bh.mean()), 3),
                                "delta_fa": int(fa.sum() - bfa.sum())}
                dr, df = [], []
                nh = np.array([c[v]["neg_hours"] for v in neg])
                for _ in range(2000):
                    ip, ineg = rng.integers(0, len(pos), len(pos)), rng.integers(0, len(neg), len(neg))
                    dr.append(hits[ip].mean() - bh[ip].mean())
                    df.append((fa[ineg].sum() - bfa[ineg].sum()) / max(nh[ineg].sum(), 1e-9))
                po["paired"]["delta_recall_ci95"] = [round(float(x), 3) for x in np.percentile(dr, [2.5, 97.5])]
                po["paired"]["delta_fa_h_ci95"] = [round(float(x), 1) for x in np.percentile(df, [2.5, 97.5])]
            o[prof] = po
        result["conditions"][name] = o
    result["per_video"] = {name: {v: {k: (val if k != "p" else None) for k, val in d.items() if k != "p"} for v, d in c.items()}
                           for name, c in conds.items()}
    save_json(result, resolve("experiments/latest/v33_light_change_stream.json"))
    for name, o in result["conditions"].items():
        print(name, {k: o[k] for k in ("det_per_step", "p_neg_mean", "p_pos_pre_event_mean")}, o["balanced"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
