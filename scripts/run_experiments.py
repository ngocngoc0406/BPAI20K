"""Full, reproducible experiment pipeline (all model selection on VALIDATION; test used once for reporting).

Steps
  1. observation-window selection (GRU visual+motion, 3 seeds) on val mAP
  2. baselines: majority, logistic regression, XGBoost (pooled features)
  3. neural variants x 3 seeds: frame-level, GRU-visual, GRU-motion, GRU/LSTM/TCN/pool visual+motion
  4. proposed: deep ensemble (5) of the selected architecture + MC-Dropout + temperature scaling
  5. risk-engine thresholds on val timelines; early-warning evaluation on full test videos
  6. permutation feature importance, efficiency, figures, model bundle

Usage:  python scripts/run_experiments.py [--quick]
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402

from src.data.dataset_loader import load_dataset  # noqa: E402
from src.evaluation.early_warning import evaluate_early_warning  # noqa: E402
from src.evaluation.metrics import (best_f1_threshold, classification_metrics, error_detection_auroc,  # noqa: E402
                                    nexar_map, selective_risk, threshold_for_precision, threshold_for_recall)
from src.explainability.attribution import permutation_importance  # noqa: E402
from src.features.motion_features import MOTION_FEATURES  # noqa: E402
from src.prediction.baselines import MajorityBaseline, PooledFeaturizer, logistic_regression, xgboost  # noqa: E402
from src.prediction.predictor import EnsemblePredictor, save_bundle  # noqa: E402
from src.prediction.trainer import train_model  # noqa: E402
from src.prediction.windows import Standardizer, eval_samples, materialize, train_samples  # noqa: E402
from src.preprocessing.feature_cache import cache_path, load_features  # noqa: E402
from src.risk.risk_engine import RiskConfig, RiskEngine, cue_reference  # noqa: E402
from src.uncertainty.calibration import ece, fit_temperature, reliability  # noqa: E402
from src.uncertainty.mc_dropout import sample_logits, summarize  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.device import environment_report, get_device  # noqa: E402
from src.utils.io import load_json, save_json, set_seed  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def hp_for(cfg: dict, kind: str, vis: bool, mot: bool, vmode: str | None = None) -> dict:
    m = cfg["model"]
    return dict(kind=kind, use_visual=vis, use_motion=mot, d_visual=512, d_motion=len(MOTION_FEATURES),
                visual_proj=m["visual_proj"], motion_proj=m["motion_proj"], hidden=m["hidden"], dropout=m["dropout"],
                in_dropout=m.get("in_dropout", 0.0), visual_mode=vmode or m.get("visual_mode", "abs"))


def select_alarm_threshold(signals: dict, recs: dict, k: int, warmup: float, timely_s: float, budget: float) -> tuple[float, dict]:
    """On VALIDATION streams: maximise timely detection subject to FA/h <= budget (tie -> fewer FA).
    If no threshold meets the budget, take the one with the fewest false alarms."""
    best, best_key, fallback, fb_key = None, None, None, None
    for thr in np.round(np.arange(0.05, 1.0, 0.01), 2):
        e = evaluate_early_warning(signals, recs, float(thr), k, warmup, timely_s)
        key = (e["timely_detection_rate"], -e["false_alarms_per_hour"])
        if e["false_alarms_per_hour"] <= budget and (best_key is None or key > best_key):
            best, best_key = (float(thr), e), key
        fkey = (-e["false_alarms_per_hour"], e["timely_detection_rate"])
        if fb_key is None or fkey > fb_key:
            fallback, fb_key = (float(thr), e), fkey
    thr, e = best if best is not None else fallback
    return thr, {k2: e[k2] for k2 in ("timely_detection_rate", "detection_rate", "false_alarms_per_hour", "mean_tta_s")} | \
        {"met_budget": best is not None}


def evaluate(p: np.ndarray, y: np.ndarray, groups: np.ndarray, thr: float, horizons: list[float]) -> dict:
    return {**classification_metrics(y, p, thr), **nexar_map(y, p, groups, horizons)}


def mean_std(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    return {k: {"mean": float(np.nanmean([r[k] for r in rows])), "std": float(np.nanstd([r[k] for r in rows]))}
            for k in keys}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true", help="fewer seeds/epochs (smoke run)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.quick:
        cfg["model"].update(epochs=4, ensemble_size=2, mc_samples=5)
    out = resolve(cfg["paths"]["experiments_dir"]) / STAMP
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger("experiments", out / "train.log")
    device = get_device()
    seeds = [cfg["seed"], cfg["seed"] + 1, cfg["seed"] + 2] if not a.quick else [cfg["seed"]]
    horizons = cfg["windows"]["eval_horizons_s"]
    set_seed(cfg["seed"])
    t_start = time.perf_counter()

    # ---------------- data ----------------
    splits = load_json(resolve(cfg["paths"]["splits"]))
    recs_all = {r.video_id: r for r in load_dataset(resolve(cfg["paths"]["annotations"]), resolve(cfg["paths"]["raw_dir"]))}
    fdir = resolve(cfg["paths"]["features_dir"])
    feats, R = {}, {}
    for sp, ids in splits.items():
        R[sp] = []
        for vid in ids:
            p = cache_path(fdir, vid)
            if p.exists() and vid in recs_all:
                feats[vid] = load_features(p)
                R[sp].append(recs_all[vid])
    log.info("videos with features: %s", {k: len(v) for k, v in R.items()})
    scaler = Standardizer().fit([feats[r.video_id] for r in R["train"]])
    ev = {sp: eval_samples(R[sp], feats, cfg, seed=1000 + i) for i, sp in enumerate(["val", "test"])}
    results: dict = {"timestamp": STAMP, "environment": environment_report(), "device": device, "config": cfg,
                     "n_videos": {k: len(v) for k, v in R.items()},
                     "n_eval_windows": {k: len(v) for k, v in ev.items()}, "seeds": seeds}

    def data_for(sp: str, W: float):
        V, M, Y = materialize(ev[sp], feats, W, scaler)
        return V, M, Y, np.array([s.group for s in ev[sp]])

    # ---------------- 1. window selection ----------------
    win_rows = []
    for W in cfg["windows"]["candidates_s"]:
        Vv, Mv, Yv, Gv = data_for("val", W)
        for s in seeds:
            m, info = train_model(R["train"], (Vv, Mv, Yv), feats, scaler, cfg, hp_for(cfg, "gru", True, True), s, W, device)
            p = summarize(sample_logits([m], Vv, Mv, device, 1))["p_mean"]
            win_rows.append({"window_s": W, "seed": s, **nexar_map(Yv, p, Gv, horizons), "best_epoch": info["best_epoch"]})
    win_df = pd.DataFrame(win_rows)
    wm = win_df.groupby("window_s").mAP.mean()
    W = float(wm.idxmax()) if wm.notna().any() else float(cfg["windows"]["observation_s"])
    agg = win_df.groupby("window_s").mAP.agg(["mean", "std"])
    results["window_selection_val"] = {str(w): {"mean": float(r["mean"]), "std": float(r["std"])} for w, r in agg.iterrows()}
    results["window_selection_rows"] = win_rows
    results["selected_window_s"] = W
    log.info("selected window %.1fs (val mAP by window: %s)", W, win_df.groupby("window_s").mAP.mean().round(4).to_dict())

    Vv, Mv, Yv, Gv = data_for("val", W)
    Vt, Mt, Yt, Gt = data_for("test", W)
    rng = np.random.default_rng(cfg["seed"])
    tr_s = train_samples(R["train"], feats, cfg, rng) + train_samples(R["train"], feats, cfg, rng)
    Vtr, Mtr, Ytr = materialize(tr_s, feats, W, scaler)
    preds = {"y": Yt.tolist(), "group": Gt.tolist(), "video_id": [s.video_id for s in ev["test"]],
             "t_end": [s.t_end for s in ev["test"]]}
    models_res: dict = {}

    # ---------------- 2. classical baselines ----------------
    maj = MajorityBaseline().fit(Ytr)
    models_res["majority"] = {"test": evaluate(maj.predict_proba(len(Yt)), Yt, Gt, 0.5, horizons)}
    for name, vis, mot, mk in [("logreg_pooled", True, True, logistic_regression), ("xgb_pooled", True, True, xgboost),
                               ("xgb_motion_only", False, True, xgboost)]:
        fz = PooledFeaturizer(use_visual=vis, use_motion=mot).fit(Vtr, Mtr)
        clf = mk(cfg["seed"]).fit(fz.transform(Vtr, Mtr), Ytr)
        pv = clf.predict_proba(fz.transform(Vv, Mv))[:, 1]
        pt = clf.predict_proba(fz.transform(Vt, Mt))[:, 1]
        thr = best_f1_threshold(Yv, pv)
        models_res[name] = {"val": evaluate(pv, Yv, Gv, thr, horizons), "test": evaluate(pt, Yt, Gt, thr, horizons)}
        preds[name] = pt.tolist()
        log.info("%s test mAP=%.4f F1=%.4f", name, models_res[name]["test"]["mAP"], models_res[name]["test"]["f1"])

    # ---------------- 3. neural variants ----------------
    variants = {  # name: (kind, use_visual, use_motion, visual_mode)
        "A_frame_visual_motion": ("frame", True, True, "abs"),
        "B_gru_visual": ("gru", True, False, "abs"),
        "gru_visual_delta_only": ("gru", True, False, "delta"),
        "gru_motion_only": ("gru", False, True, "abs"),
        "C_gru_visual_motion": ("gru", True, True, "abs"),
        "C2_gru_vdelta_motion": ("gru", True, True, "delta"),
        "lstm_vdelta_motion": ("lstm", True, True, "delta"),
        "tcn_vdelta_motion": ("tcn", True, True, "delta"),
        "pool_vdelta_motion": ("pool", True, True, "delta"),
    }
    histories = {}
    for name, (kind, vis, mot, vmode) in variants.items():
        val_rows, test_rows = [], []
        for s in seeds:
            m, info = train_model(R["train"], (Vv, Mv, Yv), feats, scaler, cfg, hp_for(cfg, kind, vis, mot, vmode), s, W, device)
            histories[f"{name}_seed{s}"] = info["history"]
            pv = summarize(sample_logits([m], Vv, Mv, device, 1))["p_mean"]
            pt = summarize(sample_logits([m], Vt, Mt, device, 1))["p_mean"]
            thr = best_f1_threshold(Yv, pv)
            val_rows.append(evaluate(pv, Yv, Gv, thr, horizons))
            test_rows.append(evaluate(pt, Yt, Gt, thr, horizons))
            if s == seeds[0]:
                preds[name] = pt.tolist()
        models_res[name] = {"val": mean_std(val_rows), "test": mean_std(test_rows), "per_seed_test": test_rows}
        log.info("%s test mAP=%.4f±%.4f", name, models_res[name]["test"]["mAP"]["mean"], models_res[name]["test"]["mAP"]["std"])
    temporal = ["C2_gru_vdelta_motion", "lstm_vdelta_motion", "tcn_vdelta_motion", "pool_vdelta_motion"]
    best_name = max(temporal, key=lambda n: models_res[n]["val"]["mAP"]["mean"])
    kind, _, _, vmode = variants[best_name]
    results["selected_architecture"] = {"name": best_name, "kind": kind,
                                        "val_mAP": {n: models_res[n]["val"]["mAP"]["mean"] for n in temporal}}
    log.info("selected architecture on val: %s", best_name)

    # ---------------- 4. proposed: ensemble + MC dropout + temperature ----------------
    hp = hp_for(cfg, kind, True, True, vmode)
    members, member_info = [], []
    for i in range(cfg["model"]["ensemble_size"]):
        m, info = train_model(R["train"], (Vv, Mv, Yv), feats, scaler, cfg, hp, 100 + i, W, device)
        members.append(m)
        member_info.append({k: info[k] for k in ("seed", "best_epoch", "best_val_ap", "train_seconds")})
        histories[f"ensemble_member{i}"] = info["history"]
    S = cfg["model"]["mc_samples"]
    unc = {}
    for tag, mods, mc in [("D_single_mc_dropout_ts", members[:1], S), ("E_ensemble_mc_dropout_ts", members, S)]:
        lv = sample_logits(mods, Vv, Mv, device, mc)
        lt = sample_logits(mods, Vt, Mt, device, mc)
        T = fit_temperature(lv, Yv)
        sv, st, st_raw = summarize(lv, T), summarize(lt, T), summarize(lt, 1.0)
        thr = best_f1_threshold(Yv, sv["p_mean"])
        res = {"val": evaluate(sv["p_mean"], Yv, Gv, thr, horizons), "test": evaluate(st["p_mean"], Yt, Gt, thr, horizons),
               "temperature": T, "test_uncalibrated": {"ece": ece(st_raw["p_mean"], Yt),
                                                       "brier": float(np.mean((st_raw["p_mean"] - Yt) ** 2))}}
        for uname in ("mutual_info", "entropy", "p_std"):
            res[f"error_auroc_{uname}"] = error_detection_auroc(Yt, st["p_mean"], st[uname], thr)
        res["selective_risk_mutual_info"] = selective_risk(Yt, st["p_mean"], st["mutual_info"], thr)
        res["selective_risk_entropy"] = selective_risk(Yt, st["p_mean"], st["entropy"], thr)
        res["reliability_test_calibrated"] = reliability(st["p_mean"], Yt)
        res["reliability_test_uncalibrated"] = reliability(st_raw["p_mean"], Yt)
        models_res[tag] = res
        preds[tag] = st["p_mean"].tolist()
        preds[f"{tag}_mutual_info"] = st["mutual_info"].tolist()
        unc[tag] = (T, thr)
        log.info("%s T=%.2f test mAP=%.4f ECE %.4f->%.4f", tag, T, res["test"]["mAP"], res["test_uncalibrated"]["ece"],
                 res["test"]["ece"])

    T_e, thr_e = unc["E_ensemble_mc_dropout_ts"]
    meta = {"description": f"SafeDrive ensemble x{len(members)} {kind.upper()} (visual-{vmode} + motion), MC-Dropout x{S}, temperature scaling",
            "window_s": W, "warmup_s": W, "temperature": T_e, "mc_samples": S, "threshold_f1_val": thr_e, "fps": 10.0,
            "motion_features": MOTION_FEATURES, "trained": STAMP, "experiment_dir": str(out), "members": member_info}
    bundle_dir = resolve(cfg["paths"]["models_dir"]) / "safedrive_v1"

    # ---------------- 5. risk engine + early warning ----------------
    predictor = EnsemblePredictor(members, scaler, meta, device)
    stride = cfg["windows"]["inference_stride_steps"]

    def timelines(recs, pred_obj):
        return {r.video_id: {**pred_obj.predict_video(feats[r.video_id], stride)} for r in recs}

    tl_val = timelines(R["val"], predictor)
    all_train_motion = np.concatenate([feats[r.video_id].motion for r in R["train"]])
    rc = RiskConfig(trend_weight=cfg["risk"]["trend_weight"], looming_weight=cfg["risk"]["looming_weight"],
                    consecutive_steps=cfg["risk"]["alarm_consecutive_steps"], cue_reference=cue_reference(all_train_motion))
    eng = RiskEngine(rc)

    def risk_at_samples(tls, samples):
        out_s = []
        for s in samples:
            tl = tls[s.video_id]
            i = int(np.searchsorted(tl["times"], s.t_end + 1e-4) - 1)  # last point <= t_end (timeline starts at 0)
            assert i >= 0 and tl["times"][i] <= s.t_end + 1e-4, "causality violated"
            sc, _ = eng.risk_scores(tl["times"], tl["p_mean"], feats[s.video_id].motion[tl["step"], MOTION_FEATURES.index("looming_center")])
            out_s.append(sc[i])
        return np.array(out_s)

    k_steps, budget, timely_s = rc.consecutive_steps, cfg["risk"]["fa_budget_per_hour"], cfg["risk"]["timely_window_s"]
    recv = {r.video_id: r for r in R["val"]}

    def risk_stream(tls):
        out_s = {}
        for vid, tl in tls.items():
            sc, _ = eng.risk_scores(tl["times"], tl["p_mean"], feats[vid].motion[tl["step"], MOTION_FEATURES.index("looming_center")])
            out_s[vid] = {"times": tl["times"], "score": sc}
        return out_s

    rv = risk_at_samples(tl_val, ev["val"])
    # WARNING: chosen on validation STREAMS (timely detection s.t. FA/h budget) - the window-level F1 threshold
    # is degenerate for streaming (it fires on almost every window).
    rc.warning, sel_e = select_alarm_threshold(risk_stream(tl_val), recv, k_steps, W, timely_s, budget)
    rc.caution = min(threshold_for_recall(Yv, rv, 0.90), 0.8 * rc.warning)  # always strictly below WARNING
    rc.high = max(threshold_for_precision(Yv, rv, 0.90), rc.warning + 0.5 * (1 - rc.warning))
    rc.mi_threshold = float(np.quantile(np.concatenate([t["mutual_info"] for t in tl_val.values()]),
                                        cfg["risk"]["uncertainty_quantile"]))
    meta["risk"] = rc.to_dict()
    results["risk_thresholds_val"] = {k: v for k, v in rc.to_dict().items() if k != "cue_reference"}
    results["risk_thresholds_val"]["warning_selection_on_val"] = sel_e
    log.info("risk thresholds: %s", results["risk_thresholds_val"])

    tl_test = timelines(R["test"], predictor)
    rt = risk_at_samples(tl_test, ev["test"])
    models_res["E_full_with_risk_engine"] = {"test": evaluate(rt, Yt, Gt, rc.warning, horizons)}
    preds["E_full_risk_score"] = rt.tolist()

    def alarm_signal(tl, vid):
        ass = eng.assess(tl["times"], tl, feats[vid].motion[tl["step"]])
        return np.array([x.risk_score if x.level != "UNCERTAIN" else 0.0 for x in ass]), [x.level for x in ass]

    ew = {}
    sig_test, level_counts = {}, {}
    for vid, tl in tl_test.items():
        sig, lv = alarm_signal(tl, vid)
        sig_test[vid] = {"times": tl["times"], "score": sig}
        for l in lv:
            level_counts[l] = level_counts.get(l, 0) + 1
    recd = {r.video_id: r for r in R["test"]}

    def prob_stream(tls):
        return {v: {"times": t["times"], "score": t["p_mean"]} for v, t in tls.items()}

    ew = {"E_full_risk_engine": evaluate_early_warning(sig_test, recd, rc.warning, k_steps, W, timely_s)}
    # same threshold-selection rule (on val) for every compared system -> fair comparison
    thr_ep, _ = select_alarm_threshold(prob_stream(tl_val), recv, k_steps, W, timely_s, budget)
    ew["E_ensemble_prob_only"] = evaluate_early_warning(prob_stream(tl_test), recd, thr_ep, k_steps, W, timely_s)
    single = EnsemblePredictor(members[:1], scaler, {**meta, "temperature": 1.0, "mc_samples": 1}, device)
    thr_c, _ = select_alarm_threshold(prob_stream(timelines(R["val"], single)), recv, k_steps, W, timely_s, budget)
    ew["C2_single_deterministic"] = evaluate_early_warning(prob_stream(timelines(R["test"], single)), recd, thr_c, k_steps,
                                                          W, timely_s)
    # operating curve over thresholds on TEST (reported only, never used for selection)
    curve = []
    for thr in np.round(np.arange(0.05, 1.0, 0.05), 2):
        e = evaluate_early_warning(sig_test, recd, float(thr), k_steps, W, timely_s)
        curve.append({k: e[k] for k in ("threshold", "detection_rate", "timely_detection_rate", "mean_tta_s",
                                        "false_alarms_per_hour")})
    ew["operating_curve_E"] = curve
    for k in ("E_full_risk_engine", "E_ensemble_prob_only", "C2_single_deterministic"):
        log.info("early warning %s: thr=%.2f det=%.3f timely=%.3f TTA=%.2fs FA/h=%.1f", k, ew[k]["threshold"],
                 ew[k]["detection_rate"], ew[k]["timely_detection_rate"], ew[k]["mean_tta_s"], ew[k]["false_alarms_per_hour"])
    results["early_warning_test"] = ew
    results["risk_level_counts_test"] = level_counts

    # ---------------- 6. importance, efficiency ----------------
    def det_predict(V, M):
        return summarize(sample_logits(members, V, M, device, 1), T_e)["p_mean"]

    results["permutation_importance_test_ap_drop"] = permutation_importance(det_predict, Vt, Mt, Yt, n_repeats=3)
    eff = {}
    one_v, one_m = Vt[:1], Mt[:1]
    for tag, mods, mc in [("single_deterministic", members[:1], 1), ("ensemble_mc", members, S)]:
        sample_logits(mods, one_v, one_m, device, mc)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            sample_logits(mods, one_v, one_m, device, mc)
        if device == "cuda":
            torch.cuda.synchronize()
        eff[f"latency_ms_per_window_{tag}"] = 1000 * (time.perf_counter() - t0) / 20
    ex = load_json(resolve("data/processed/extract_summary.json")) if resolve("data/processed/extract_summary.json").exists() else {}
    eff["feature_extraction_fps"] = ex.get("mean_extract_fps")
    eff["process_rss_gb"] = psutil.Process().memory_info().rss / 1e9
    if device == "cuda":
        eff["peak_vram_gb_training"] = torch.cuda.max_memory_allocated() / 1e9
    results["efficiency"] = eff
    results["models"] = models_res
    results["runtime_minutes"] = (time.perf_counter() - t_start) / 60

    save_bundle(bundle_dir, members, scaler, meta)
    save_json(results, out / "results.json")
    save_json(histories, out / "training_histories.json")
    pd.DataFrame(preds).to_csv(out / "test_predictions.csv", index=False)
    pd.DataFrame(ew["E_full_risk_engine"]["per_video"]).to_csv(out / "early_warning_per_video.csv", index=False)
    latest = resolve(cfg["paths"]["experiments_dir"]) / "latest"
    latest.mkdir(exist_ok=True)
    save_json(results, latest / "results.json")
    (latest / "RUN").write_text(STAMP)
    log.info("saved %s and bundle %s (%.1f min)", out, bundle_dir, results["runtime_minutes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
