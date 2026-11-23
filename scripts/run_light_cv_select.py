"""v2.1 selection protocol: grouped 5-fold cross-validation on the 340 TRAIN+VAL videos only (test untouched).

Why: in run_light_study.py the 60-video validation ranking disagreed with cross-validated rankings, so a single small
validation split is not a reliable selector. Here every choice uses out-of-fold (OOF) predictions on train+val:
  * model/feature/window selection: cheapest config with OOF mAP >= best - max(0.01, SE_best)   (SE over folds)
  * calibration method: grouped CV on the OOF probabilities
  * alarm / risk thresholds: OOF *streaming* timelines of all 340 videos (~2 h of normal driving instead of 18 min)
Then the selected config is retrained on all 340 videos (5 bagged members) and evaluated ONCE on the 60 test videos.
Usage: python scripts/run_light_cv_select.py [--quick]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from src.data.dataset_loader import load_dataset  # noqa: E402
from src.evaluation.bootstrap import bootstrap_map  # noqa: E402
from src.evaluation.early_warning import evaluate_early_warning  # noqa: E402
from src.evaluation.metrics import (best_f1_threshold, classification_metrics, error_detection_auroc, nexar_map,  # noqa: E402
                                    selective_risk, threshold_for_precision, threshold_for_recall)
from src.features.motion_features import MOTION_FEATURES  # noqa: E402
from src.features.window_features import NEEDS_CNN, build_matrix, channel_names, fit_vabs_pca, video_window_matrix  # noqa: E402
from src.prediction.light_models import COST_RANK, BaggedEnsemble, Calibrator, choose_calibration, make_model, summarize_probs  # noqa: E402
from src.prediction.windows import eval_samples, train_samples  # noqa: E402
from src.preprocessing.feature_cache import cache_path, load_features  # noqa: E402
from src.risk.risk_engine import RiskConfig, RiskEngine, cue_reference  # noqa: E402
from src.uncertainty.calibration import brier, ece, reliability  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.device import environment_report  # noqa: E402
from src.utils.io import load_json, save_json  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
FEATURE_SETS = {"motion": ["motion"], "motion+diff": ["motion", "diff"], "motion+diff+vdelta": ["motion", "diff", "vdelta"],
                "motion+vabs": ["motion", "vabs"]}
CLASSIFIERS = ["logreg", "lgbm", "xgb", "catboost"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_light_study import select_alarm_threshold  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    out = resolve("experiments") / f"light_cv_{STAMP}"
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger("light_cv", out / "study.log")
    horizons = cfg["windows"]["eval_horizons_s"]
    windows_s = [1.0, 2.0, 3.0] if not a.quick else [2.0]
    classifiers = CLASSIFIERS if not a.quick else ["logreg", "lgbm"]
    t_start = time.perf_counter()
    splits = load_json(resolve(cfg["paths"]["splits"]))
    recs = {r.video_id: r for r in load_dataset(resolve(cfg["paths"]["annotations"]), resolve(cfg["paths"]["raw_dir"]))}
    fdir = resolve(cfg["paths"]["features_dir"])
    dev_ids = sorted(v for v in splits["train"] + splits["val"] if cache_path(fdir, v).exists())   # 340 videos
    test_ids = sorted(v for v in splits["test"] if cache_path(fdir, v).exists())
    feats = {v: load_features(cache_path(fdir, v)) for v in dev_ids + test_ids}
    rng = np.random.default_rng(11)
    fold = {}
    for lab in (0, 1):
        vs = [v for v in dev_ids if recs[v].label == lab]
        rng.shuffle(vs)
        fold.update({v: i % 5 for i, v in enumerate(vs)})
    # fixed OOF evaluation windows (Nexar protocol) for every dev video
    oof_ev = {k: eval_samples([recs[v] for v in dev_ids if fold[v] == k], feats, cfg, seed=4000 + k) for k in range(5)}
    meta = pd.DataFrame([{"video_id": s.video_id, "t_end": s.t_end, "y": s.label, "group": s.group, "fold": k}
                         for k in range(5) for s in oof_ev[k]])
    y_oof, g_oof = meta.y.values.astype(float), meta.group.values
    tr_samples, pcas = {}, {}
    for k in range(5):
        tr_recs = [recs[v] for v in dev_ids if fold[v] != k]
        r2 = np.random.default_rng(500 + k)
        tr_samples[k] = sum((train_samples(tr_recs, feats, cfg, r2) for _ in range(3)), [])
        pcas[k] = fit_vabs_pca([feats[r.video_id] for r in tr_recs])
    log.info("dev videos %d (5 folds), OOF windows %d, test videos %d", len(dev_ids), len(meta), len(test_ids))

    # ---------------- 1. CV grid ----------------
    rows, oof_store = [], {}
    for W in windows_s:
        for fs, groups in FEATURE_SETS.items():
            mats = {k: (build_matrix(tr_samples[k], feats, W, groups, pcas[k]), build_matrix(oof_ev[k], feats, W, groups, pcas[k])[0])
                    for k in range(5)}
            for clf in classifiers:
                p = np.zeros(len(meta))
                fold_map, fit_s = [], 0.0
                for k in range(5):
                    (Xtr, ytr), Xte = mats[k]
                    t0 = time.perf_counter()
                    m = make_model(clf, 42).fit(Xtr, ytr)
                    fit_s += time.perf_counter() - t0
                    pk = m.predict_proba(Xte)[:, 1]
                    p[(meta.fold == k).values] = pk
                    mk = (meta.fold == k).values
                    fold_map.append(nexar_map(y_oof[mk], pk, g_oof[mk], horizons)["mAP"])
                name = f"{clf}|{fs}|W{W:g}"
                oof_store[name] = p
                rows.append({"config": name, "W": W, "features": fs, "model": clf, "needs_cnn": bool(set(groups) & NEEDS_CNN),
                             "oof_mAP": nexar_map(y_oof, p, g_oof, horizons)["mAP"], "fold_mean": float(np.mean(fold_map)),
                             "fold_se": float(np.std(fold_map, ddof=1) / np.sqrt(5)), "roc_auc": classification_metrics(y_oof, p, 0.5)["roc_auc"],
                             "fit_s_per_fold": fit_s / 5, "n_features": int(mats[0][1].shape[1])})
                log.info("%-34s OOF mAP=%.3f (folds %.3f ± SE %.3f)", name, rows[-1]["oof_mAP"], rows[-1]["fold_mean"], rows[-1]["fold_se"])
    grid = pd.DataFrame(rows)
    grid.to_csv(out / "cv_grid.csv", index=False)
    best = grid.loc[grid.oof_mAP.idxmax()]
    margin = max(0.01, float(best.fold_se))
    cand = grid[grid.oof_mAP >= best.oof_mAP - margin].copy()
    cand["cost"] = cand.apply(lambda r: (int(r.needs_cnn), COST_RANK[r.model], r.W), axis=1)
    sel = cand.sort_values("cost").iloc[0]
    log.info("best %s %.3f; margin %.3f; %d candidates; SELECTED %s (OOF mAP %.3f)", best.config, best.oof_mAP, margin, len(cand),
             sel.config, sel.oof_mAP)
    W, fs, clf = float(sel.W), sel.features, sel.model
    groups = FEATURE_SETS[fs]

    # ---------------- 2. OOF bagged ensembles -> calibration + streaming thresholds ----------------
    stride = cfg["windows"]["inference_stride_steps"]
    n_mem = 5 if not a.quick else 2
    oof_ps = np.zeros((n_mem, len(meta)))
    oof_tl = {}
    for k in range(5):
        Xtr, ytr = build_matrix(tr_samples[k], feats, W, groups, pcas[k])
        ens_k = BaggedEnsemble(clf, n_mem, seed=200 + k).fit(Xtr, ytr, np.array([s.video_id for s in tr_samples[k]]))
        mk = (meta.fold == k).values
        oof_ps[:, mk] = ens_k.predict_samples(build_matrix(oof_ev[k], feats, W, groups, pcas[k])[0])
        for v in [v for v in dev_ids if fold[v] == k]:
            f = feats[v]
            idx = np.arange(0, len(f.times), stride)
            oof_tl[v] = {"ps": ens_k.predict_samples(video_window_matrix(f, f.times[idx], W, groups, pcas[k])), "times": f.times[idx], "step": idx}
    cal_method, cal_scores = choose_calibration(oof_ps.mean(0), y_oof, meta.video_id.values)
    cal = Calibrator(cal_method).fit(oof_ps.mean(0), y_oof)
    s_oof = summarize_probs(cal(oof_ps))
    thr_f1 = best_f1_threshold(y_oof, s_oof["p_mean"])
    for v, t in oof_tl.items():
        t.update(summarize_probs(cal(t.pop("ps"))))
    rc = RiskConfig(trend_weight=cfg["risk"]["trend_weight"], looming_weight=cfg["risk"]["looming_weight"],
                    consecutive_steps=cfg["risk"]["alarm_consecutive_steps"],
                    cue_reference=cue_reference(np.concatenate([feats[v].motion for v in dev_ids])))
    eng = RiskEngine(rc)
    li = MOTION_FEATURES.index("looming_center")
    k_steps, budget, timely_s = rc.consecutive_steps, cfg["risk"]["fa_budget_per_hour"], cfg["risk"]["timely_window_s"]

    def risk_stream(tls):
        return {v: {"times": t["times"], "score": eng.risk_scores(t["times"], t["p_mean"], feats[v].motion[t["step"], li])[0]} for v, t in tls.items()}

    def prob_stream(tls):
        return {v: {"times": t["times"], "score": t["p_mean"]} for v, t in tls.items()}

    recd = {v: recs[v] for v in dev_ids}
    rs_oof = risk_stream(oof_tl)
    rv = np.array([rs_oof[r.video_id]["score"][int(np.searchsorted(rs_oof[r.video_id]["times"], r.t_end + 1e-4) - 1)]
                   for r in meta.itertuples()])
    rc.warning, sel_e = select_alarm_threshold(rs_oof, recd, k_steps, W, timely_s, budget)
    rc.caution = min(threshold_for_recall(y_oof, rv, 0.90), 0.8 * rc.warning)
    rc.high = max(threshold_for_precision(y_oof, rv, 0.90), rc.warning + 0.5 * (1 - rc.warning))
    rc.mi_threshold = float(np.quantile(np.concatenate([t["mutual_info"] for t in oof_tl.values()]), cfg["risk"]["uncertainty_quantile"]))
    thr_prob, sel_p = select_alarm_threshold(prob_stream(oof_tl), recd, k_steps, W, timely_s, budget)
    log.info("calibration=%s; alarm thresholds (OOF streams): risk %.2f %s | prob %.2f %s", cal_method, rc.warning, sel_e, thr_prob, sel_p)

    # ---------------- 3. final model on all 340 dev videos -> single test evaluation ----------------
    dev_recs = [recs[v] for v in dev_ids]
    r3 = np.random.default_rng(900)
    tr_all = sum((train_samples(dev_recs, feats, cfg, r3) for _ in range(3)), [])
    pca = fit_vabs_pca([feats[v] for v in dev_ids])
    Xtr, ytr = build_matrix(tr_all, feats, W, groups, pca)
    t0 = time.perf_counter()
    ens = BaggedEnsemble(clf, n_mem, seed=300).fit(Xtr, ytr, np.array([s.video_id for s in tr_all]))
    fit_final = time.perf_counter() - t0
    test_recs = [recs[v] for v in test_ids]
    ev_t = eval_samples(test_recs, feats, cfg, seed=1001)  # identical test windows to v1 and run_light_study
    Xt = build_matrix(ev_t, feats, W, groups, pca)[0]
    yt = np.array([s.label for s in ev_t], float)
    gt = np.array([s.group for s in ev_t])
    pst = ens.predict_samples(Xt)
    st_, st_raw = summarize_probs(cal(pst)), summarize_probs(pst)
    final = {"config": {"W": W, "features": fs, "groups": groups, "classifier": clf, "n_members": n_mem, "n_features": int(Xtr.shape[1]),
                        "calibration": cal_method, "calibration_cv_nll": cal_scores, "trained_on": "train+val (340 videos)"},
             "test": {**classification_metrics(yt, st_["p_mean"], thr_f1), **nexar_map(yt, st_["p_mean"], gt, horizons)},
             "oof_dev": {**classification_metrics(y_oof, s_oof["p_mean"], thr_f1), **nexar_map(y_oof, s_oof["p_mean"], g_oof, horizons)},
             "test_uncalibrated": {"ece": ece(st_raw["p_mean"], yt), "brier": brier(st_raw["p_mean"], yt), **nexar_map(yt, st_raw["p_mean"], gt, horizons)},
             "reliability_test_calibrated": reliability(st_["p_mean"], yt), "reliability_test_uncalibrated": reliability(st_raw["p_mean"], yt),
             "fit_seconds": fit_final}
    for u in ("mutual_info", "entropy", "p_std"):
        final[f"error_auroc_{u}"] = error_detection_auroc(yt, st_["p_mean"], st_[u], thr_f1)
    final["selective_risk_entropy"] = selective_risk(yt, st_["p_mean"], st_["entropy"], thr_f1)
    final["selective_risk_mutual_info"] = selective_risk(yt, st_["p_mean"], st_["mutual_info"], thr_f1)
    tl_t = {}
    for v in test_ids:
        f = feats[v]
        idx = np.arange(0, len(f.times), stride)
        tl_t[v] = {**summarize_probs(cal(ens.predict_samples(video_window_matrix(f, f.times[idx], W, groups, pca)))), "times": f.times[idx], "step": idx}
    rect = {v: recs[v] for v in test_ids}
    sig, levels = {}, {}
    for v, t in tl_t.items():
        ass = eng.assess(t["times"], t, feats[v].motion[t["step"]])
        sig[v] = {"times": t["times"], "score": np.array([x.risk_score if x.level != "UNCERTAIN" else 0.0 for x in ass])}
        for x in ass:
            levels[x.level] = levels.get(x.level, 0) + 1
    ew = {"FINAL_risk_engine": evaluate_early_warning(sig, rect, rc.warning, k_steps, W, timely_s),
          "FINAL_prob_only": evaluate_early_warning(prob_stream(tl_t), rect, thr_prob, k_steps, W, timely_s),
          "operating_curve_FINAL": [{kk: e[kk] for kk in ("threshold", "detection_rate", "timely_detection_rate", "mean_tta_s", "false_alarms_per_hour")}
                                    for e in (evaluate_early_warning(prob_stream(tl_t), rect, float(th), k_steps, W, timely_s)
                                              for th in np.round(np.arange(0.05, 1.0, 0.05), 2))]}
    for kk in ("FINAL_risk_engine", "FINAL_prob_only"):
        e = ew[kk]
        log.info("TEST early warning %s thr=%.2f det=%.3f timely=%.3f TTA(med)=%.2f FA/h=%.1f", kk, e["threshold"], e["detection_rate"],
                 e["timely_detection_rate"], e["median_tta_s"], e["false_alarms_per_hour"])
    log.info("TEST final %s mAP=%.4f AUC=%.4f F1=%.4f ECE=%.4f", sel.config, final["test"]["mAP"], final["test"]["roc_auc"],
             final["test"]["f1"], final["test"]["ece"])

    # ---------------- 4. importance, efficiency, comparison ----------------
    names = channel_names(groups)
    base = [n.split("__")[0] for n in names]
    ap0 = average_precision_score(yt, cal(pst.mean(0)))
    rng2 = np.random.default_rng(0)
    imp = {}
    for ch in dict.fromkeys(base):
        cols = [i for i, b in enumerate(base) if b == ch]
        d = []
        for _ in range(3):
            Xp = Xt.copy()
            Xp[:, cols] = Xt[rng2.permutation(len(Xt))][:, cols]
            d.append(ap0 - average_precision_score(yt, cal(ens.predict_proba(Xp))))
        imp[ch] = float(np.mean(d))
    imp = dict(sorted(imp.items(), key=lambda kv: -kv[1]))
    f0 = feats[test_ids[0]]
    t1 = f0.times[len(f0.times) // 2: len(f0.times) // 2 + 1]
    for _ in range(5):
        ens.predict_samples(video_window_matrix(f0, t1, W, groups, pca))
    t0 = time.perf_counter()
    for _ in range(100):
        ens.predict_samples(video_window_matrix(f0, t1, W, groups, pca))
    lat = 10 * (time.perf_counter() - t0)
    t0 = time.perf_counter()
    for _ in range(100):
        ens.members[0].predict_proba(video_window_matrix(f0, t1, W, groups, pca))
    lat1 = 10 * (time.perf_counter() - t0)
    eff = {"latency_ms_per_window_ensemble": lat, "latency_ms_per_window_single": lat1, "model_size_mb": len(pickle.dumps(ens.members)) / 1e6,
           "process_rss_gb": psutil.Process().memory_info().rss / 1e9, "uses_gpu": False}
    pp = resolve("experiments/latest/perception_profile.json")
    if pp.exists():
        p = load_json(pp)
        d = p["detectors"]["yolov8n@640"]["ms_per_frame"]
        eff["perception_ms_per_frame_light"] = p["decode_resize_ms_per_frame"] + d + p["flow_ms_per_frame"] + p["frame_diff_ms_per_frame"]
        eff["perception_ms_per_frame_v1"] = p["decode_resize_ms_per_frame"] + d + p["flow_ms_per_frame"] + p["resnet18_ms_per_frame"]
        eff["perception_fps_light_est"] = 1000 / eff["perception_ms_per_frame_light"]
        eff["perception_fps_v1_est"] = 1000 / eff["perception_ms_per_frame_v1"]
    comp = {}
    v1p = resolve("backups/v1_lstm_ensemble/20261119_020750/test_predictions.csv")
    v2p = sorted(resolve("experiments").glob("light_2*/test_predictions.csv"))
    if v1p.exists():
        df = pd.DataFrame({"video_id": [s.video_id for s in ev_t], "t_end": np.round([s.t_end for s in ev_t], 3), "y": yt, "group": gt,
                           "V21_FINAL": st_["p_mean"]})
        v1 = pd.read_csv(v1p, dtype={"video_id": str})
        v1["t_end"] = v1.t_end.round(3)
        m = df.merge(v1[["video_id", "t_end", "E_ensemble_mc_dropout_ts", "xgb_pooled"]], on=["video_id", "t_end"])
        cols = ["V21_FINAL", "E_ensemble_mc_dropout_ts", "xgb_pooled"]
        if v2p:
            v2 = pd.read_csv(v2p[-1], dtype={"video_id": str})
            v2["t_end"] = v2.t_end.round(3)
            m = m.merge(v2[["video_id", "t_end", "FINAL"]].rename(columns={"FINAL": "V2_val_selected"}), on=["video_id", "t_end"])
            cols.append("V2_val_selected")
        assert len(m) == len(df), "test windows differ from v1"
        comp = bootstrap_map(m, cols, horizons, n=1000, pairs=[("V21_FINAL", c) for c in cols[1:]])
        log.info("bootstrap %s", {k2: (round(v["diff"], 3), [round(x, 3) for x in v["ci95"]]) for k2, v in comp["paired"].items()})
    results = {"timestamp": STAMP, "protocol": "v2.1: grouped 5-fold CV on train+val (340 videos) for every choice; test evaluated once",
               "environment": environment_report(), "selection_rule": "cheapest config with OOF mAP >= best - max(0.01, SE_best); "
               "cost: no-CNN < CNN, logreg < lgbm < xgb < catboost, shorter W", "best": best.to_dict(), "margin": margin,
               "selected": sel.to_dict(), "cv_grid": grid.to_dict("records"), "final": final, "early_warning_test": ew,
               "alarm_selection_oof": {"risk": sel_e, "prob": sel_p, "thr_prob": thr_prob},
               "risk_thresholds_val": {kk: v for kk, v in rc.to_dict().items() if kk != "cue_reference"},
               "risk_level_counts_test": levels, "permutation_importance_test_ap_drop": imp, "efficiency": eff,
               "bootstrap_vs_v1": comp, "runtime_minutes": (time.perf_counter() - t_start) / 60}
    save_json(results, out / "results_light.json")
    save_json(results, resolve("experiments/latest/results_light.json"))
    pd.DataFrame({"video_id": [s.video_id for s in ev_t], "t_end": [s.t_end for s in ev_t], "y": yt, "group": gt, "FINAL": st_["p_mean"]}).to_csv(
        out / "test_predictions.csv", index=False)
    bdir = resolve("models/safedrive_v2_light")
    bdir.mkdir(parents=True, exist_ok=True)
    with open(bdir / "model.pkl", "wb") as fh:
        pickle.dump({"members": ens.members, "calibrator": cal, "pca": pca if "vabs" in groups else None,
                     "col_median": np.median(Xtr, 0).astype(np.float32)}, fh)
    save_json({"type": "tabular", "feature_version": 1, "description": f"SafeDrive v2-light: {clf.upper()} bagged x{n_mem} on {fs} window features (W={W:g}s, "
               f"{cal_method} calibration; selected by 5-fold CV on train+val)", "window_s": W, "warmup_s": W, "groups": groups,
               "classifier": clf, "n_members": n_mem, "calibration": cal_method, "threshold_f1_val": thr_f1, "fps": 10.0,
               "uses_cnn": bool(set(groups) & NEEDS_CNN), "feature_names": names, "motion_features": MOTION_FEATURES, "trained": STAMP,
               "experiment_dir": str(out), "risk": rc.to_dict()}, bdir / "bundle.json")
    log.info("saved %s and bundle (%.1f min)", out, results["runtime_minutes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
