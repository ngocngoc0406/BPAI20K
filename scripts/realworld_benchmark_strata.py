"""v3.3: held-out benchmark (Nexar, 224 videos) with the DEPLOYED pipeline logic, stratified by the dataset's own verified
metadata (light_conditions, weather, scene) and by two MEASURED H3 proxies. Descriptive only: nothing is selected here.

H3 = the fixed image-plane driving corridor (|cx - 0.5| < 0.2, cy > 0.35) used by the corridor cues / messages.
Proxies (no map / no lane data, so these are proxies, not road geometry):
  turn proxy   median |egoh_dx| (horizontal background shift, normalised frame widths / s) over the evaluated span
               (positives: the 5 s before the event; negatives: the whole stream). Strata = tertiles on DEV videos.
  off-corridor positives whose largest road user in the last 2 s before the event has its centre OUTSIDE the corridor.

Output: experiments/latest/v33_benchmark_strata.json (+ per-video CSV for the failure-case gallery).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.dataset_loader import load_dataset  # noqa: E402
from src.evaluation.streaming import MAX_LEAD  # noqa: E402
from src.pipeline import SafeDrivePipeline  # noqa: E402
from src.preprocessing.feature_cache import cache_path, load_features  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.io import load_json, save_json  # noqa: E402

PROFILES = ("conservative", "balanced", "sensitive")


def wilson(k: int, n: int) -> list:
    if n == 0:
        return [None, None]
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(max(0.0, c - h)), 3), round(float(min(1.0, c + h)), 3)]


def poisson_ci(k: int, hours: float) -> list:
    from scipy.stats import chi2
    if hours <= 0:
        return [None, None]
    lo = chi2.ppf(0.025, 2 * k) / 2 if k > 0 else 0.0
    hi = chi2.ppf(0.975, 2 * (k + 1)) / 2
    return [round(float(lo / hours), 2), round(float(hi / hours), 2)]


def turn_proxy(f, t0: float, t1: float) -> float:
    m = (f.times >= t0) & (f.times < t1) & (f.ego[:, 28] > 0.5)
    return float(np.median(np.abs(f.ego[m, 0]))) if m.sum() >= 5 else float("nan")


def off_corridor(f, te: float) -> bool | None:
    tr = f.tracks
    steps = np.flatnonzero((f.times >= te - 2.0) & (f.times < te))
    rows = tr[np.isin(tr[:, 0].astype(int), steps)]
    if not len(rows):
        return None
    area = (rows[:, 6] - rows[:, 4]) * (rows[:, 7] - rows[:, 5])
    r = rows[int(np.argmax(area))]
    cx, cy = (r[4] + r[6]) / 2, (r[5] + r[7]) / 2
    return not (abs(cx - 0.5) < 0.2 and cy > 0.35)


def main() -> int:
    cfg = load_config(None)
    sp = load_json(resolve("data/processed/splits_full.json"))  # 1,500-video split: dev 1,276 / held-out 224
    recs = {r.video_id: r for r in load_dataset(resolve(cfg["paths"]["annotations"]), resolve(cfg["paths"]["raw_dir"]))}
    meta = pd.read_csv(resolve(cfg["paths"]["annotations"]), dtype={"video_id": str}).set_index("video_id")
    fdir = resolve(cfg["paths"]["features_dir"])
    held = [v for v in sp["test"] + sp.get("test_ext", []) if cache_path(fdir, v).exists()]
    dev = [v for v in sp["train"] + sp["val"] + sp.get("dev_new", []) if cache_path(fdir, v).exists()]
    pipe = SafeDrivePipeline(resolve("models/safedrive_v3"), device="cpu")
    warm = float(pipe.predictor.meta.get("warmup_s", pipe.predictor.window_s))

    # turn-proxy tertile edges from DEV videos only
    dev_tp = []
    for v in dev:
        f = load_features(cache_path(fdir, v))
        r = recs[v]
        dev_tp.append(turn_proxy(f, r.time_of_event - MAX_LEAD, r.time_of_event) if r.label == 1 and r.time_of_event
                      else turn_proxy(f, warm, 1e9))
    edges = [float(x) for x in np.nanpercentile(dev_tp, [100 / 3, 200 / 3])]
    print("dev turn-proxy tertile edges", edges, flush=True)

    rows = []
    for i, v in enumerate(held):
        f = load_features(cache_path(fdir, v))
        r = recs[v]
        m = meta.loc[v]
        row = {"video_id": v, "label": int(r.label), "time_of_event": r.time_of_event, "light": m["light_conditions"],
               "weather": m["weather"], "scene": m["scene"], "duration_s": float(f.times[-1])}
        pos = r.label == 1 and r.time_of_event is not None
        row["turn_proxy"] = turn_proxy(f, r.time_of_event - MAX_LEAD, r.time_of_event) if pos else turn_proxy(f, warm, 1e9)
        row["off_corridor"] = off_corridor(f, r.time_of_event) if pos else None
        for prof in PROFILES:
            res = pipe.analyze_features(f, profile=prof)
            ons = np.asarray(res.alarms, float)
            if pos:
                pre = ons[ons < r.time_of_event]
                valid = pre[pre >= r.time_of_event - MAX_LEAD]
                row[f"{prof}_hit"] = bool(len(valid))
                row[f"{prof}_lead"] = float(r.time_of_event - valid.min()) if len(valid) else None
                row[f"{prof}_early_alarms"] = int(np.sum(pre < r.time_of_event - MAX_LEAD))
            else:
                row[f"{prof}_fa"] = int(len(ons))
            if prof == "balanced":
                tl = res.timeline
                row["p_max"] = float(tl.p_mean.max())
                if pos:
                    w = tl[(tl.time_s >= r.time_of_event - MAX_LEAD) & (tl.time_s < r.time_of_event)]
                    row["p_max_pre_event"] = float(w.p_mean.max()) if len(w) else None
                    row["mi_pre_event"] = float(w.mutual_info.mean()) if len(w) else None
                row["alarms_balanced"] = [round(float(x), 2) for x in ons]
        row["neg_hours"] = 0.0 if pos else max(0.0, float(f.times[-1]) - warm) / 3600
        rows.append(row)
        if i % 40 == 0:
            print(i, len(held), flush=True)
    df = pd.DataFrame(rows)
    df["turn_stratum"] = pd.cut(df.turn_proxy, [-np.inf, edges[0], edges[1], np.inf], labels=["low", "mid", "high"]).astype(str)
    out_csv = resolve("experiments/latest/v33_benchmark_per_video.csv")
    df.to_csv(out_csv, index=False)

    def summarise(sub: pd.DataFrame) -> dict:
        pos, neg = sub[sub.label == 1], sub[sub.label == 0]
        hrs = float(neg.neg_hours.sum())
        o = {"n_pos": len(pos), "n_neg": len(neg), "neg_hours": round(hrs, 3)}
        for prof in PROFILES:
            k = int(pos[f"{prof}_hit"].sum()) if len(pos) else 0
            fa = int(neg[f"{prof}_fa"].sum()) if len(neg) else 0
            leads = pos[f"{prof}_lead"].dropna().to_numpy()
            o[prof] = {"recall": round(k / len(pos), 3) if len(pos) else None, "recall_ci95": wilson(k, len(pos)), "hits": k,
                       "false_alarms": fa, "fa_h": round(fa / hrs, 2) if hrs > 0 else None, "fa_h_ci95": poisson_ci(fa, hrs),
                       "warned_ge_1s": round(float(np.sum(leads >= 1)) / len(pos), 3) if len(pos) else None,
                       "warned_ge_2s": round(float(np.sum(leads >= 2)) / len(pos), 3) if len(pos) else None,
                       "warned_ge_3s": round(float(np.sum(leads >= 3)) / len(pos), 3) if len(pos) else None,
                       "median_lead_s": round(float(np.median(leads)), 2) if len(leads) else None}
        return o

    result = {"description": __doc__, "model": "models/safedrive_v3", "n_videos": len(df), "turn_proxy_dev_edges": edges,
              "overall": summarise(df), "strata": {}}
    for col in ("light", "weather", "scene", "turn_stratum"):
        result["strata"][col] = {str(g): summarise(sub) for g, sub in df.groupby(col)}
    posd = df[df.label == 1]
    result["strata"]["off_corridor_positives"] = {
        str(g): {"n_pos": len(sub), **{p: {"recall": round(float(sub[f"{p}_hit"].mean()), 3), "recall_ci95": wilson(int(sub[f"{p}_hit"].sum()), len(sub))}
                                        for p in PROFILES}}
        for g, sub in posd.groupby(posd.off_corridor.map({True: "outside_corridor", False: "inside_corridor", None: "no_road_user"}).fillna("no_road_user"))}
    save_json(result, resolve("experiments/latest/v33_benchmark_strata.json"))
    print(result["overall"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
