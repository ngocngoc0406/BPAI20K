"""Lightweight-architecture study (v2). Same video split and the SAME evaluation windows as v1
(eval_samples seeds 1000/1001), so results are directly comparable. Test is used once, after selection.

1. Grid on VALIDATION: window W x feature set x classifier (3 seeds each) + light temporal NNs (no CNN).
2. Pre-declared selection rule: among configs with val mAP >= best - 0.01, take the cheapest
   (no-CNN before CNN; LogReg < LightGBM < XGBoost < CatBoost < temporal NN; shorter window).
3. Final: 5-member bagged ensemble of the selected config; calibration method chosen by grouped CV inside val;
   alarm threshold chosen on validation streams (same rule as v1); risk engine; test evaluation; efficiency;
   bootstrap comparison with v1 on identical windows.
Usage: python scripts/run_light_study.py [--quick]
"""
from __future__ import annotations

import argparse
import copy
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402

from src.data.dataset_loader import load_dataset  # noqa: E402
from src.evaluation.bootstrap import bootstrap_map  # noqa: E402
from src.evaluation.early_warning import evaluate_early_warning  # noqa: E402
from src.evaluation.metrics import (best_f1_threshold, classification_metrics, error_detection_auroc, nexar_map,  # noqa: E402
                                    selective_risk, threshold_for_precision, threshold_for_recall)
from src.features.frame_diff import FRAME_DIFF_FEATURES  # noqa: E402
from src.features.motion_features import MOTION_FEATURES  # noqa: E402
from src.features.window_features import NEEDS_CNN, build_matrix, channel_names, fit_vabs_pca, video_window_matrix  # noqa: E402
from src.prediction.light_models import COST_RANK, BaggedEnsemble, Calibrator, choose_calibration, make_model, summarize_probs  # noqa: E402
from src.prediction.trainer import train_model  # noqa: E402
from src.prediction.windows import Standardizer, eval_samples, materialize, train_samples  # noqa: E402
from src.preprocessing.feature_cache import VideoFeatures, cache_path, load_features  # noqa: E402
from src.risk.risk_engine import RiskConfig, RiskEngine, cue_reference  # noqa: E402
from src.uncertainty.calibration import brier, ece, reliability  # noqa: E402
from src.uncertainty.mc_dropout import sample_logits, summarize  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.device import environment_report, get_device  # noqa: E402
from src.utils.io import load_json, save_json, set_seed  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
FEATURE_SETS = {"motion": ["motion"], "motion+diff": ["motion", "diff"], "motion+diff+vdelta": ["motion", "diff", "vdelta"],
                "motion+vabs": ["motion", "vabs"]}
CLASSIFIERS = ["logreg", "lgbm", "xgb", "catboost"]


def select_alarm_threshold(signals, recs, k, warmup, timely_s, budget):
    best, bk, fb, fk = None, None, None, None
    for thr in np.round(np.arange(0.05, 1.0, 0.01), 2):
        e = evaluate_early_warning(signals, recs, float(thr), k, warmup, timely_s)
        key = (e["timely_detection_rate"], -e["false_alarms_per_hour"])
        if e["false_alarms_per_hour"] <= budget and (bk is None or key > bk):
            best, bk = (float(thr), e), key
        f2 = (-e["false_alarms_per_hour"], e["timely_detection_rate"])
        if fk is None or f2 > fk:
            fb, fk = (float(thr), e), f2
    thr, e = best if best is not None else fb
    return thr, {k2: e[k2] for k2 in ("timely_detection_rate", "detection_rate", "false_alarms_per_hour", "mean_tta_s")} | {
        "met_budget": best is not None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--from-grid", default=None, help="reuse grid_rows.csv of an earlier run (skips step 1)")
    a = ap.parse_args()
    cfg = load_config()
    out = resolve("experiments") / f"light_{STAMP}"
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger("light_study", out / "study.log")
    device = get_device()
    seeds = [42, 43, 44] if not a.quick else [42]
    windows_s = [1.0, 2.0, 3.0] if not a.quick else [1.0]
    horizons = cfg["windows"]["eval_horizons_s"]
    set_seed(42)
    t_start = time.perf_counter()

    splits = load_json(resolve(cfg["paths"]["splits"]))
    recs = {r.video_id: r for r in load_dataset(resolve(cfg["paths"]["annotations"]), resolve(cfg["paths"]["raw_dir"]))}
    fdir = resolve(cfg["paths"]["features_dir"])
    R = {sp: [recs[v] for v in splits[sp] if cache_path(fdir, v).exists()] for sp in ("train", "val", "test")}
    feats = {r.video_id: load_features(cache_path(fdir, r.video_id)) for sp in R for r in R[sp]}
    missing = [v for v, f in feats.items() if f.diff is None]
    assert not missing, f"{len(missing)} caches lack frame-diff features; run scripts/add_framediff.py"
    ev = {sp: eval_samples(R[sp], feats, cfg, seed=1000 + i) for i, sp in enumerate(["val", "test"])}
    Y = {sp: np.array([s.label for s in ev[sp]], np.float32) for sp in ev}
    G = {sp: np.array([s.group for s in ev[sp]]) for sp in ev}
    pca = fit_vabs_pca([feats[r.video_id] for r in R["train"]])
    log.info("videos %s, eval windows val=%d test=%d", {k: len(v) for k, v in R.items()}, len(Y["val"]), len(Y["test"]))

    def train_set(seed):
        rng = np.random.default_rng(seed)
        return sum((train_samples(R["train"], feats, cfg, rng) for _ in range(3)), [])

    # ---------------- 1. grid on validation ----------------
    rows = []
    preds_test: dict[str, np.ndarray] = {}
    if a.from_grid:
        windows_s = []  # skip both grid loops
        rows = pd.read_csv(resolve(a.from_grid) / "grid_rows.csv").to_dict("records")
        log.info("reusing %d grid rows from %s", len(rows), a.from_grid)
    for W in windows_s:
        Xev = {fs: {sp: build_matrix(ev[sp], feats, W, g, pca)[0] for sp in ev} for fs, g in FEATURE_SETS.items()}
        for seed in seeds:
            tr = train_set(seed)
            for fs, g in FEATURE_SETS.items():
                Xtr, ytr = build_matrix(tr, feats, W, g, pca)
                for clf in CLASSIFIERS:
                    t0 = time.perf_counter()
                    m = make_model(clf, seed).fit(Xtr, ytr)
                    fit_s = time.perf_counter() - t0
                    pv = m.predict_proba(Xev[fs]["val"])[:, 1]
                    pt = m.predict_proba(Xev[fs]["test"])[:, 1]
                    thr = best_f1_threshold(Y["val"], pv)
                    rows.append({"family": "tabular", "W": W, "features": fs, "model": clf, "seed": seed, "needs_cnn": bool(set(g) & NEEDS_CNN),
                                 "n_features": Xtr.shape[1], "fit_s": fit_s,
                                 "val_mAP": nexar_map(Y["val"], pv, G["val"], horizons)["mAP"],
                                 **{f"test_{k}": v for k, v in classification_metrics(Y["test"], pt, thr).items()},
                                 **{f"test_{k}": v for k, v in nexar_map(Y["test"], pt, G["test"], horizons).items()}})
                    if seed == seeds[0]:
                        preds_test[f"W{W}|{fs}|{clf}"] = pt
            log.info("W=%.0f seed=%d done (%.1f min)", W, seed, (time.perf_counter() - t_start) / 60)
    # light temporal NNs on motion + frame-diff (no CNN): diff is fed through the 'visual' stream
    feats_nn = {v: VideoFeatures(f.video_id, f.times, f.diff, f.motion, f.tracks, f.meta) for v, f in feats.items()}
    sc_nn = Standardizer().fit([feats_nn[r.video_id] for r in R["train"]])
    cfg_nn = copy.deepcopy(cfg)
    cfg_nn["model"].update(epochs=40 if not a.quick else 3)
    for W in windows_s:  # (empty when --from-grid)
        Vv, Mv, Yv = materialize(ev["val"], feats_nn, W, sc_nn)
        Vt, Mt, Yt = materialize(ev["test"], feats_nn, W, sc_nn)
        for kind in ("gru", "tcn"):
            hp = dict(kind=kind, use_visual=True, use_motion=True, d_visual=len(FRAME_DIFF_FEATURES), d_motion=len(MOTION_FEATURES),
                      visual_proj=16, motion_proj=32, hidden=32, dropout=0.3, in_dropout=0.3, visual_mode="abs")
            for seed in seeds:
                t0 = time.perf_counter()
                m, info = train_model(R["train"], (Vv, Mv, Yv), feats_nn, sc_nn, cfg_nn, hp, seed, W, device)
                fit_s = time.perf_counter() - t0
                pv = summarize(sample_logits([m], Vv, Mv, device, 1))["p_mean"]
                pt = summarize(sample_logits([m], Vt, Mt, device, 1))["p_mean"]
                thr = best_f1_threshold(Yv, pv)
                rows.append({"family": "temporal_nn", "W": W, "features": "motion+diff", "model": kind, "seed": seed, "needs_cnn": False,
                             "n_features": int(sum(p.numel() for p in m.parameters())), "fit_s": fit_s,
                             "val_mAP": nexar_map(Yv, pv, G["val"], horizons)["mAP"],
                             **{f"test_{k}": v for k, v in classification_metrics(Yt, pt, thr).items()},
                             **{f"test_{k}": v for k, v in nexar_map(Yt, pt, G["test"], horizons).items()}})
                if seed == seeds[0]:
                    preds_test[f"W{W}|motion+diff|{kind}"] = pt
    grid = pd.DataFrame(rows)
    grid.to_csv(out / "grid_rows.csv", index=False)
    agg = grid.groupby(["family", "W", "features", "model", "needs_cnn"]).agg(
        val_mAP=("val_mAP", "mean"), val_mAP_std=("val_mAP", "std"), test_mAP=("test_mAP", "mean"), test_mAP_std=("test_mAP", "std"),
        test_f1=("test_f1", "mean"), test_roc_auc=("test_roc_auc", "mean"), fit_s=("fit_s", "mean"), n_features=("n_features", "first")).reset_index()
    best_row = agg.loc[agg.val_mAP.idxmax()]
    best_val = float(best_row.val_mAP)
    margin = max(0.01, float(np.nan_to_num(best_row.val_mAP_std)))  # a costlier model must beat cheaper ones by more than seed noise
    cand = agg[agg.val_mAP >= best_val - margin].copy()
    cand["cost"] = cand.apply(lambda r: (int(r.needs_cnn), COST_RANK.get(r.model, 4), r.W), axis=1)
    sel = cand.sort_values("cost").iloc[0]
    log.info("best val mAP %.4f (%s/%s/W%.0f); margin %.4f; %d candidates; selected %s", best_val, best_row.features, best_row.model,
             best_row.W, margin, len(cand), sel[["W", "features", "model", "val_mAP"]].to_dict())
    agg.to_csv(out / "grid_summary.csv", index=False)
    if sel.family != "tabular":
        save_json({"grid_summary": agg.to_dict("records"), "selected": sel.to_dict(), "margin": margin}, out / "grid_only.json")
        log.error("Selection rule picked a temporal NN; integrate it before running the final stage. Grid saved to %s", out)
        return 3

    # ---------------- 2. final model ----------------
    W, fs, clf = float(sel.W), sel.features, sel.model
    groups = FEATURE_SETS[fs]
    tr = train_set(100)
    Xtr, ytr = build_matrix(tr, feats, W, groups, pca)
    gtr = np.array([s.video_id for s in tr])
    t0 = time.perf_counter()
    ens = BaggedEnsemble(clf, n_members=5 if not a.quick else 2, seed=100).fit(Xtr, ytr, gtr)
    fit_final_s = time.perf_counter() - t0
    Xv = build_matrix(ev["val"], feats, W, groups, pca)[0]
    Xt = build_matrix(ev["test"], feats, W, groups, pca)[0]
    psv, pst = ens.predict_samples(Xv), ens.predict_samples(Xt)
    cal_method, cal_scores = choose_calibration(psv.mean(0), Y["val"], np.array([s.video_id for s in ev["val"]]))
    cal = Calibrator(cal_method).fit(psv.mean(0), Y["val"])
    sv, st = summarize_probs(cal(psv)), summarize_probs(cal(pst))
    st_raw = summarize_probs(pst)
    thr = best_f1_threshold(Y["val"], sv["p_mean"])
    final = {"config": {"W": W, "features": fs, "groups": groups, "classifier": clf, "n_members": len(ens.members),
                        "n_features": int(Xtr.shape[1]), "calibration": cal_method, "calibration_cv_nll": cal_scores},
             "test": {**classification_metrics(Y["test"], st["p_mean"], thr), **nexar_map(Y["test"], st["p_mean"], G["test"], horizons)},
             "val": {**classification_metrics(Y["val"], sv["p_mean"], thr), **nexar_map(Y["val"], sv["p_mean"], G["val"], horizons)},
             "test_uncalibrated": {"ece": ece(st_raw["p_mean"], Y["test"]), "brier": brier(st_raw["p_mean"], Y["test"]),
                                   **nexar_map(Y["test"], st_raw["p_mean"], G["test"], horizons)},
             "reliability_test_calibrated": reliability(st["p_mean"], Y["test"]),
             "reliability_test_uncalibrated": reliability(st_raw["p_mean"], Y["test"]),
             "fit_seconds": fit_final_s}
    for u in ("mutual_info", "entropy", "p_std"):
        final[f"error_auroc_{u}"] = error_detection_auroc(Y["test"], st["p_mean"], st[u], thr)
    final["selective_risk_entropy"] = selective_risk(Y["test"], st["p_mean"], st["entropy"], thr)
    final["selective_risk_mutual_info"] = selective_risk(Y["test"], st["p_mean"], st["mutual_info"], thr)
    preds_test["FINAL"] = st["p_mean"]
    log.info("final %s test mAP=%.4f F1=%.4f ECE %.4f->%.4f (cal=%s)", final["config"], final["test"]["mAP"], final["test"]["f1"],
             final["test_uncalibrated"]["ece"], final["test"]["ece"], cal_method)

    # ---------------- 3. streaming + risk engine ----------------
    stride = cfg["windows"]["inference_stride_steps"]

    def timeline(vid):
        f = feats[vid]
        idx = np.arange(0, len(f.times), stride)
        X = video_window_matrix(f, f.times[idx], W, groups, pca)
        s = summarize_probs(cal(ens.predict_samples(X)))
        return {**s, "times": f.times[idx], "step": idx}

    tl = {sp: {r.video_id: timeline(r.video_id) for r in R[sp]} for sp in ("val", "test")}
    rc = RiskConfig(trend_weight=cfg["risk"]["trend_weight"], looming_weight=cfg["risk"]["looming_weight"],
                    consecutive_steps=cfg["risk"]["alarm_consecutive_steps"],
                    cue_reference=cue_reference(np.concatenate([feats[r.video_id].motion for r in R["train"]])))
    eng = RiskEngine(rc)
    k, budget, timely_s = rc.consecutive_steps, cfg["risk"]["fa_budget_per_hour"], cfg["risk"]["timely_window_s"]
    li = MOTION_FEATURES.index("looming_center")

    def risk_stream(tls):
        return {v: {"times": t["times"], "score": eng.risk_scores(t["times"], t["p_mean"], feats[v].motion[t["step"], li])[0]} for v, t in tls.items()}

    def prob_stream(tls):
        return {v: {"times": t["times"], "score": t["p_mean"]} for v, t in tls.items()}

    recv, rect = {r.video_id: r for r in R["val"]}, {r.video_id: r for r in R["test"]}
    rs_val = risk_stream(tl["val"])
    rv = np.array([rs_val[s.video_id]["score"][int(np.searchsorted(rs_val[s.video_id]["times"], s.t_end + 1e-4) - 1)] for s in ev["val"]])
    rc.warning, sel_e = select_alarm_threshold(rs_val, recv, k, W, timely_s, budget)
    rc.caution = min(threshold_for_recall(Y["val"], rv, 0.90), 0.8 * rc.warning)
    rc.high = max(threshold_for_precision(Y["val"], rv, 0.90), rc.warning + 0.5 * (1 - rc.warning))
    rc.mi_threshold = float(np.quantile(np.concatenate([t["mutual_info"] for t in tl["val"].values()]), cfg["risk"]["uncertainty_quantile"]))
    sig_test = {}
    levels: dict[str, int] = {}
    for v, t in tl["test"].items():
        ass = eng.assess(t["times"], t, feats[v].motion[t["step"]])
        sig_test[v] = {"times": t["times"], "score": np.array([x.risk_score if x.level != "UNCERTAIN" else 0.0 for x in ass])}
        for x in ass:
            levels[x.level] = levels.get(x.level, 0) + 1
    thr_p, _ = select_alarm_threshold(prob_stream(tl["val"]), recv, k, W, timely_s, budget)
    ew = {"FINAL_risk_engine": evaluate_early_warning(sig_test, rect, rc.warning, k, W, timely_s),
          "FINAL_prob_only": evaluate_early_warning(prob_stream(tl["test"]), rect, thr_p, k, W, timely_s)}
    curve = []
    for th in np.round(np.arange(0.05, 1.0, 0.05), 2):
        e = evaluate_early_warning(sig_test, rect, float(th), k, W, timely_s)
        curve.append({kk: e[kk] for kk in ("threshold", "detection_rate", "timely_detection_rate", "mean_tta_s", "false_alarms_per_hour")})
    ew["operating_curve_FINAL"] = curve
    for kk in ("FINAL_risk_engine", "FINAL_prob_only"):
        e = ew[kk]
        log.info("early warning %s thr=%.2f det=%.3f timely=%.3f FA/h=%.1f", kk, e["threshold"], e["detection_rate"],
                 e["timely_detection_rate"], e["false_alarms_per_hour"])

    # ---------------- 4. importance + efficiency ----------------
    names = channel_names(groups)
    base = [n.split("__")[0] for n in names]
    from sklearn.metrics import average_precision_score
    ap0 = average_precision_score(Y["test"], cal(pst.mean(0)))
    rng = np.random.default_rng(0)
    imp = {}
    for ch in dict.fromkeys(base):
        cols = [i for i, b in enumerate(base) if b == ch]
        drops = []
        for _ in range(3):
            Xp = Xt.copy()
            Xp[:, cols] = Xt[rng.permutation(len(Xt))][:, cols]
            drops.append(ap0 - average_precision_score(Y["test"], cal(ens.predict_proba(Xp))))
        imp[ch] = float(np.mean(drops))
    imp = dict(sorted(imp.items(), key=lambda kv: -kv[1]))
    f0 = feats[R["test"][0].video_id]
    t_one = f0.times[len(f0.times) // 2: len(f0.times) // 2 + 1]
    for _ in range(5):
        ens.predict_samples(video_window_matrix(f0, t_one, W, groups, pca))
    t0 = time.perf_counter()
    for _ in range(100):
        x1 = video_window_matrix(f0, t_one, W, groups, pca)
        ens.predict_samples(x1)
    lat_ens = 10 * (time.perf_counter() - t0)
    t0 = time.perf_counter()
    for _ in range(100):
        ens.members[0].predict_proba(video_window_matrix(f0, t_one, W, groups, pca))
    lat_single = 10 * (time.perf_counter() - t0)
    model_bytes = len(pickle.dumps(ens.members))
    eff = {"latency_ms_per_window_ensemble": lat_ens, "latency_ms_per_window_single": lat_single,
           "model_size_mb": model_bytes / 1e6, "process_rss_gb": psutil.Process().memory_info().rss / 1e9, "uses_gpu": False}
    perf = resolve("experiments/latest/perception_profile.json")
    if perf.exists():
        p = load_json(perf)
        d = p["detectors"]["yolov8n@640"]["ms_per_frame"]
        eff["perception_ms_per_frame_light"] = p["decode_resize_ms_per_frame"] + d + p["flow_ms_per_frame"] + p["frame_diff_ms_per_frame"]
        eff["perception_ms_per_frame_v1"] = p["decode_resize_ms_per_frame"] + d + p["flow_ms_per_frame"] + p["resnet18_ms_per_frame"]
        eff["perception_fps_light_est"] = 1000 / eff["perception_ms_per_frame_light"]
        eff["perception_fps_v1_est"] = 1000 / eff["perception_ms_per_frame_v1"]

    # ---------------- 5. comparison with v1 on identical windows ----------------
    comp = {}
    v1_dir = resolve("backups/v1_lstm_ensemble/20261119_020750")
    if (v1_dir / "test_predictions.csv").exists():
        v1 = pd.read_csv(v1_dir / "test_predictions.csv", dtype={"video_id": str})
        df = pd.DataFrame({"video_id": [s.video_id for s in ev["test"]], "t_end": np.round([s.t_end for s in ev["test"]], 3),
                           "y": Y["test"], "group": G["test"], "FINAL": preds_test["FINAL"]})
        v1["t_end"] = v1.t_end.round(3)
        m = df.merge(v1[["video_id", "t_end", "E_ensemble_mc_dropout_ts", "xgb_pooled"]], on=["video_id", "t_end"], how="inner")
        assert len(m) == len(df), f"window mismatch with v1: {len(m)} vs {len(df)}"
        comp = bootstrap_map(m, ["FINAL", "E_ensemble_mc_dropout_ts", "xgb_pooled"], horizons, n=1000,
                             pairs=[("FINAL", "E_ensemble_mc_dropout_ts"), ("FINAL", "xgb_pooled")])
        log.info("bootstrap: %s", {k2: v["diff"] for k2, v in comp["paired"].items()})

    results = {"timestamp": STAMP, "environment": environment_report(), "selection_rule": "cheapest config with val mAP >= best - 0.01; "
               "cost order: no-CNN < CNN, logreg < lgbm < xgb < catboost < temporal NN, shorter W",
               "best_val_mAP": float(best_val), "selected": sel.to_dict(), "grid_summary": agg.to_dict("records"),
               "final": final, "early_warning_test": ew, "risk_thresholds_val": {**{kk: v for kk, v in rc.to_dict().items() if kk != "cue_reference"},
                                                                                 "warning_selection_on_val": sel_e},
               "risk_level_counts_test": levels, "permutation_importance_test_ap_drop": imp, "efficiency": eff,
               "bootstrap_vs_v1": comp, "runtime_minutes": (time.perf_counter() - t_start) / 60, "seeds": seeds}
    save_json(results, out / "results_light.json")
    save_json(results, resolve("experiments/latest/results_light.json"))
    pd.DataFrame({"video_id": [s.video_id for s in ev["test"]], "t_end": [s.t_end for s in ev["test"]], "y": Y["test"], "group": G["test"],
                  **preds_test}).to_csv(out / "test_predictions.csv", index=False)
    # bundle
    bdir = resolve("models/safedrive_v2_light")
    bdir.mkdir(parents=True, exist_ok=True)
    with open(bdir / "model.pkl", "wb") as fh:
        pickle.dump({"members": ens.members, "calibrator": cal, "pca": pca if "vabs" in groups else None,
                     "col_median": np.median(Xtr, 0).astype(np.float32)}, fh)
    save_json({"type": "tabular", "description": f"SafeDrive v2-light: {clf.upper()} bagged x{len(ens.members)} on {fs} window features "
               f"(W={W}s, {cal_method} calibration)", "window_s": W, "warmup_s": W, "groups": groups, "classifier": clf,
               "n_members": len(ens.members), "calibration": cal_method, "threshold_f1_val": thr, "fps": 10.0, "uses_cnn": bool(set(groups) & NEEDS_CNN),
               "feature_names": names, "motion_features": MOTION_FEATURES, "trained": STAMP, "experiment_dir": str(out),
               "risk": rc.to_dict()}, bdir / "bundle.json")
    log.info("saved %s and bundle %s (%.1f min)", out, bdir, results["runtime_minutes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
