"""Held-out comparison at MATCHED false-alarm rates (descriptive; no choice is made here).

Retrains the baseline (v2 config) and the selected v3 config on the 1,276 dev videos exactly as run_ew_study.py does
(bagged LightGBM x5, same seeds), loads the deployed v2-light bundle (400-video model), and scores every held-out video
(original test 60 + test_ext 164). Recall / EDR are read off each model's own held-out FA curve at 5 / 10 / 20 FA/h, so models
are compared at equal false-alarm rates; a paired video bootstrap gives CIs. Output: experiments/latest/ew_heldout_compare.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from run_ew_study import LABELS, Study  # noqa: E402

from src.evaluation.streaming import MAX_LEAD, per_video_table  # noqa: E402
from src.features.window_features import build_matrix, video_window_matrix  # noqa: E402
from src.prediction.light_models import BaggedEnsemble  # noqa: E402
from src.prediction.tabular_predictor import TabularPredictor  # noqa: E402
from src.prediction.windows import train_samples  # noqa: E402
from src.utils.config import resolve  # noqa: E402
from src.utils.io import load_json, save_json  # noqa: E402

BUDGETS = (5, 10, 20)


def curve_metrics(tab: dict, w: np.ndarray | None = None, leads_tabs: dict | None = None) -> dict:
    w = np.ones(len(tab["ids"])) if w is None else w
    pos, neg = tab["is_pos"], ~tab["is_pos"]
    hours = float((w[neg] * tab["hours"][neg]).sum())
    fah = (w[neg, None] * tab["fa"][neg]).sum(0) / max(hours, 1e-9)
    npos = max(float(w[pos].sum()), 1e-9)
    out = {}
    for b in BUDGETS:
        ok = np.where(fah <= b)[0]
        j = ok[0] if len(ok) else len(fah) - 1
        out[f"recall@{b}"] = float((w[pos] * tab["hit"][pos, j]).sum() / npos)
        if leads_tabs:
            for x, lt in leads_tabs.items():
                out[f"edr@{x}s@{b}"] = float((w[pos] * lt["hit"][pos, j]).sum() / npos)
    out["S"] = float(np.mean([out[f"recall@{b}"] for b in BUDGETS]))
    return out


def lead_table(streams, recs, grid, k, warm, lead):
    """Like per_video_table but a hit requires a valid onset at or before t_event - lead."""
    from src.evaluation.early_warning import alarm_onsets
    from src.evaluation.streaming import ema
    tab = per_video_table(streams, recs, grid, k, warm)
    for i, vid in enumerate(tab["ids"]):
        if tab["is_pos"][i]:
            r, s = recs[vid], streams[vid]
            t, sc = np.asarray(s["times"]), ema(np.asarray(s["score"], float), 1.0)
            m = (t >= warm - 1e-6) & (t < r.time_of_event)
            tab["hit"][i] = [any(r.time_of_event - MAX_LEAD <= o <= r.time_of_event - lead for o in alarm_onsets(t[m], sc[m], th, k)) for th in grid]
    return tab


def main() -> int:
    st = Study("full", False)
    res = load_json(resolve("experiments/latest/results_ew_full.json"))
    held = st.test + st.test_ext
    models = {}
    for name, (groups, W, lab) in {"v2config_full_dev": (["motion", "diff"], 1.0, "L0"),
                                   "v3_selected": (res["selected"]["groups"], res["selected"]["W"], res["selected"]["labels"])}.items():
        cfg = st.cfg_for(lab)
        tr = sum((train_samples([st.recs[v] for v in st.dev], st.feats, cfg, np.random.default_rng(900)) for _ in range(3)), [])
        X, y = build_matrix(tr, st.feats, W, groups)
        ens = BaggedEnsemble("lgbm", 5, seed=300).fit(X, y, np.array([s.video_id for s in tr]))
        models[name] = (lambda f, t, e=ens, g=groups, w=W: e.predict_proba(video_window_matrix(f, t, w, g)), W)
        print("trained", name, flush=True)
    v2 = TabularPredictor.load(resolve("models/safedrive_v2_light"))
    models["v2light_deployed_400"] = (lambda f, t: v2.predict_matrix(v2.features(f, t))["p_mean"], v2.window_s)
    grid = np.linspace(0.05, 1.0, 96)
    tabs, ltabs = {}, {}
    for name, (fn, W) in models.items():
        streams = {}
        for v in held:
            f = st.feats[v]
            idx = np.arange(0, len(f.times), st.stride)
            streams[v] = {"times": f.times[idx], "score": fn(f, f.times[idx])}
        tabs[name] = per_video_table(streams, st.recs, grid, 3, W)
        ltabs[name] = {x: lead_table(streams, st.recs, grid, 3, W, x) for x in (1.0, 2.0, 3.0)}
    out = {"n_videos": len(held), "neg_hours": float(tabs["v3_selected"]["hours"].sum()), "models": {}, "paired": {}}
    for name in models:
        out["models"][name] = curve_metrics(tabs[name], None, {f"{x:g}": t for x, t in ltabs[name].items()})
        print(name, {k: round(v, 3) for k, v in out["models"][name].items() if k.startswith(("recall", "S", "edr@1s@10", "edr@2s@10", "edr@3s@10"))}, flush=True)
    rng = np.random.default_rng(0)
    nv = len(held)
    for a, b in [("v3_selected", "v2config_full_dev"), ("v3_selected", "v2light_deployed_400"), ("v2config_full_dev", "v2light_deployed_400")]:
        d = {k: [] for k in ["S", *[f"recall@{x}" for x in BUDGETS]]}
        for _ in range(2000):
            w = np.bincount(rng.integers(0, nv, nv), minlength=nv).astype(float)
            ma, mb = curve_metrics(tabs[a], w), curve_metrics(tabs[b], w)
            for k in d:
                d[k].append(ma[k] - mb[k])
        out["paired"][f"{a} - {b}"] = {k: {"mean": float(np.mean(v)), "ci90": [float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))],
                                            "p_better": float(np.mean(np.array(v) > 0))} for k, v in d.items()}
        p = out["paired"][f"{a} - {b}"]["S"]
        print(f"{a} - {b}: dS={p['mean']:+.3f} CI90 [{p['ci90'][0]:+.3f},{p['ci90'][1]:+.3f}] P={p['p_better']:.2f}", flush=True)
    save_json(out, resolve("experiments/latest/ew_heldout_compare.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
