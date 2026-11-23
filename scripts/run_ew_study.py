"""v3 early-warning study. All choices on out-of-fold (OOF) streams of the DEV videos; test sets evaluated once at the end.

Pre-declared:
  baseline         = deployed v2-light config: motion+diff, W=1 s, labels L0 (positives end 0.3-2.0 s before event), k=3, no EMA
  primary metric   = S = mean(recall@5, recall@10, recall@20 false alarms / hour) on OOF dev streams (src/evaluation/streaming.py)
  acceptance rule  = a change is KEPT if the paired video-bootstrap 90 % CI of (S_candidate - S_reference) is > 0 (primary), OR
                     (secondary, declared after the 340-video dev run and before the full-dev run) if S is non-inferior
                     (90 % CI lower bound > -0.02) AND the OOF window mAP improves (paired video-bootstrap 90 % CI > 0)
  stages           = A feature groups (ego-motion, interactions, compensation method)  ->  B labels / horizons / window
                     -> C alarm post-processing (k, EMA)  ->  final (bagged x5, Platt, thresholds from OOF streams)
Usage: python scripts/run_ew_study.py --dev orig|full [--quick]
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
from src.evaluation.metrics import best_f1_threshold, classification_metrics, error_detection_auroc, nexar_map  # noqa: E402
from src.evaluation.streaming import evaluate_at, operating_points, per_video_table, score_from_table, streaming_score  # noqa: E402
from src.features.motion_features import MOTION_FEATURES  # noqa: E402
from src.features.window_features import build_matrix, channel_names, video_window_matrix  # noqa: E402
from src.prediction.light_models import BaggedEnsemble, Calibrator, make_model, summarize_probs  # noqa: E402
from src.prediction.windows import eval_samples, train_samples  # noqa: E402
from src.preprocessing.feature_cache import cache_path, load_features  # noqa: E402
from src.risk.risk_engine import RiskConfig, cue_reference  # noqa: E402
from src.uncertainty.calibration import brier, ece  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.device import environment_report  # noqa: E402
from src.utils.io import load_json, save_json  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE = {"groups": ["motion", "diff"], "W": 1.0, "labels": "L0", "k": 3, "alpha": 1.0}
FEATURES_A = {
    "A0 base (motion+diff)": ["motion", "diff"],
    "A1 +ego stats (homography)": ["motion", "diff", "ego_h"],
    "A2 +interactions, no compensation": ["motion", "diff", "rel_none"],
    "A3 +interactions, homography-compensated": ["motion", "diff", "rel_h"],
    "A4 +interactions, flow-compensated": ["motion", "diff", "rel_f"],
    "A5 +ego(H) +interactions(H)": ["motion", "diff", "ego_h", "rel_h"],
    "A6 +ego(F) +interactions(F)": ["motion", "diff", "ego_f", "rel_f"],
}
LABELS = {  # windows-config overrides
    "L0": {"train_pos_horizon_s": [0.3, 2.0], "hard_negative_margin_s": None},
    "L1 horizon 0.3-1.5": {"train_pos_horizon_s": [0.3, 1.5], "hard_negative_margin_s": None},
    "L2 horizon 0.3-3.0": {"train_pos_horizon_s": [0.3, 3.0], "hard_negative_margin_s": None},
    "L3 horizon 0.3-4.0": {"train_pos_horizon_s": [0.3, 4.0], "hard_negative_margin_s": None},
    "L4 alert-aligned": {"train_pos_mode": "alert", "hard_negative_margin_s": None},
    "L5 L0 + hard negatives (>=2 s before alert)": {"train_pos_horizon_s": [0.3, 2.0], "hard_negative_margin_s": 2.0},
    "L6 L2 + hard negatives (>=2 s before alert)": {"train_pos_horizon_s": [0.3, 3.0], "hard_negative_margin_s": 2.0},
    "L7 dense negatives (40/video)": {"train_pos_horizon_s": [0.3, 2.0], "hard_negative_margin_s": None, "neg_samples_per_video": 40},
    "L8 dense negatives (40/video) + mined": {"train_pos_horizon_s": [0.3, 2.0], "hard_negative_margin_s": None,
                                               "neg_samples_per_video": 40, "mine_negatives": True},
}


def mine_hard_negatives(m, tr_recs, feats, W, groups, X, y, stride, per_video: int = 10):
    """Round-2 training set: add the highest-scoring windows of each TRAINING normal-driving video (fold-internal only)."""
    extra = []
    for r in tr_recs:
        if r.label != 0:
            continue
        f = feats[r.video_id]
        idx = np.arange(int(W * f.fps), len(f.times), stride)
        Xv = video_window_matrix(f, f.times[idx], W, groups)
        top = np.argsort(-m.predict_proba(Xv)[:, 1])[:per_video]
        extra.append(Xv[top])
    Xn = np.concatenate(extra) if extra else np.zeros((0, X.shape[1]), np.float32)
    return np.concatenate([X, Xn]), np.concatenate([y, np.zeros(len(Xn), np.float32)])


class Study:
    def __init__(self, dev: str, quick: bool):
        self.cfg = load_config()
        self.quick = quick
        sp = load_json(resolve("data/processed/splits_full.json")) if dev == "full" else load_json(resolve(self.cfg["paths"]["splits"]))
        self.recs = {r.video_id: r for r in load_dataset(resolve(self.cfg["paths"]["annotations"]), resolve(self.cfg["paths"]["raw_dir"]))}
        fdir = resolve(self.cfg["paths"]["features_dir"])
        have = lambda ids: sorted(v for v in ids if cache_path(fdir, v).exists())  # noqa: E731
        dev_ids = sp["train"] + sp["val"] + (sp.get("dev_new", []) if dev == "full" else [])
        self.dev = have(dev_ids)
        self.test = have(sp["test"])
        self.test_ext = have(sp.get("test_ext", [])) if dev == "full" else []
        self.feats = {v: load_features(cache_path(fdir, v)) for v in self.dev + self.test + self.test_ext}
        missing = [v for v, f in self.feats.items() if f.ego is None or f.diff is None]
        assert not missing, f"{len(missing)} caches lack ego/diff features"
        rng = np.random.default_rng(13)
        self.fold = {}
        for lab in (0, 1):
            vs = [v for v in self.dev if self.recs[v].label == lab]
            rng.shuffle(vs)
            self.fold.update({v: i % 5 for i, v in enumerate(vs)})
        self.oof_ev = {k: eval_samples([self.recs[v] for v in self.dev if self.fold[v] == k], self.feats, self.cfg, seed=5000 + k) for k in range(5)}
        self.stride = self.cfg["windows"]["inference_stride_steps"]
        self.cache: dict = {}

    def cfg_for(self, labels: str) -> dict:
        c = copy.deepcopy(self.cfg)
        c["windows"].update(LABELS[labels])
        return c

    def run(self, groups: list[str], W: float, labels: str, model: str = "lgbm") -> dict:
        """OOF window predictions + OOF per-video streams for one configuration (single model per fold)."""
        key = (tuple(groups), W, labels, model)
        if key in self.cache:
            return self.cache[key]
        cfg = self.cfg_for(labels)
        y, g, p = [], [], []
        streams = {}
        t0 = time.perf_counter()
        for k in range(5):
            tr_recs = [self.recs[v] for v in self.dev if self.fold[v] != k]
            r2 = np.random.default_rng(600 + k)
            tr = sum((train_samples(tr_recs, self.feats, cfg, r2) for _ in range(3 if not self.quick else 1)), [])
            X, yt = build_matrix(tr, self.feats, W, groups)
            m = make_model(model, 42).fit(X, yt)
            if cfg["windows"].get("mine_negatives"):
                X, yt = mine_hard_negatives(m, tr_recs, self.feats, W, groups, X, yt, self.stride)
                m = make_model(model, 42).fit(X, yt)
            ev = self.oof_ev[k]
            Xe = build_matrix(ev, self.feats, W, groups)[0]
            p.append(m.predict_proba(Xe)[:, 1])
            y.append([s.label for s in ev])
            g.append([s.group for s in ev])
            for v in [v for v in self.dev if self.fold[v] == k]:
                f = self.feats[v]
                idx = np.arange(0, len(f.times), self.stride)
                streams[v] = {"times": f.times[idx], "score": m.predict_proba(video_window_matrix(f, f.times[idx], W, groups))[:, 1]}
        y, g, p = np.concatenate(y).astype(float), np.concatenate(g), np.concatenate(p)
        res = {"y": y, "g": g, "p": p, "streams": streams, "seconds": time.perf_counter() - t0,
               "oof_mAP": nexar_map(y, p, g, self.cfg["windows"]["eval_horizons_s"])["mAP"], "roc_auc": classification_metrics(y, p, 0.5)["roc_auc"]}
        self.cache[key] = res
        return res

    def score(self, res: dict, W: float, k: int = 3, alpha: float = 1.0) -> dict:
        op = operating_points(res["streams"], self.recs, (5, 10, 20), k, W, alpha)
        return {"S": streaming_score(op), **{f"recall@{b}": op[f"@{b}FA"]["recall"] for b in (5, 10, 20)},
                **{f"edr@{x}s@10FA": op["@10FA"][f"edr@{x}s"] for x in (1, 2, 3)}, "thr@10FA": op["@10FA"]["threshold"], "op": op}

    def paired_map(self, ra: dict, rb: dict, n: int = 500) -> dict:
        """Paired video bootstrap of OOF window mAP (a - b); both use the same OOF windows."""
        df = pd.DataFrame({"video_id": [s.video_id for k in range(5) for s in self.oof_ev[k]], "y": ra["y"], "group": ra["g"],
                           "a": ra["p"], "b": rb["p"]})
        bt = bootstrap_map(df, ["a", "b"], self.cfg["windows"]["eval_horizons_s"], n=n, pairs=[("a", "b")])["paired"]["a - b"]
        return {"mean": bt["diff"], "ci95": bt["ci95"], "p_better": bt["p_a_better"]}

    def paired(self, ra: dict, rb: dict, Wa: float, Wb: float, ka=3, kb=3, aa=1.0, ab=1.0, n: int = 1000) -> dict:
        """Video-bootstrap of S_a - S_b. Thresholds are re-derived on every resample (per-video tables, src/evaluation/streaming.py)."""
        grid = np.linspace(0.2, 1.0, 81)
        ta = per_video_table(ra["streams"], self.recs, grid, ka, Wa, aa)
        tb = per_video_table(rb["streams"], self.recs, grid, kb, Wb, ab)
        assert ta["ids"] == tb["ids"]
        rng = np.random.default_rng(0)
        nv = len(ta["ids"])
        diffs = []
        for _ in range(n if not self.quick else 50):
            w = np.bincount(rng.integers(0, nv, nv), minlength=nv).astype(float)
            diffs.append(score_from_table(ta, w) - score_from_table(tb, w))
        d = np.array(diffs)
        return {"mean": float(d.mean()), "point": score_from_table(ta) - score_from_table(tb),
                "ci90": [float(np.quantile(d, 0.05)), float(np.quantile(d, 0.95))], "p_better": float((d > 0).mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="orig", choices=["orig", "full"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    st = Study(a.dev, a.quick)
    out_dir = resolve("experiments") / f"ew_{a.dev}_{STAMP}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger("ew_study", out_dir / "study.log")
    log.info("dev=%d videos (5 folds), test=%d, test_ext=%d", len(st.dev), len(st.test), len(st.test_ext))
    t_start = time.perf_counter()
    rows = []

    def record(stage, name, groups, W, labels, k=3, alpha=1.0):
        r = st.run(groups, W, labels)
        s = st.score(r, W, k, alpha)
        row = {"stage": stage, "name": name, "groups": "+".join(groups), "W": W, "labels": labels, "k": k, "alpha": alpha,
               "oof_mAP": r["oof_mAP"], "roc_auc": r["roc_auc"], **{kk: v for kk, v in s.items() if kk != "op"}, "fit_seconds": r["seconds"]}
        rows.append(row)
        log.info("%s %-45s S=%.3f R@5/10/20=%.3f/%.3f/%.3f EDR@1/2/3=%.3f/%.3f/%.3f mAP=%.3f", stage, name, s["S"], s["recall@5"],
                 s["recall@10"], s["recall@20"], s["edr@1s@10FA"], s["edr@2s@10FA"], s["edr@3s@10FA"], r["oof_mAP"])
        return r, s

    # ---------- A: feature groups ----------
    base_r, base_s = record("A", "A0 base (motion+diff)", BASE["groups"], 1.0, "L0")
    resA = {}
    for name, groups in list(FEATURES_A.items())[1:]:
        resA[name] = record("A", name, groups, 1.0, "L0")
    tests = {}
    for name, (r, s) in resA.items():
        tests[name] = st.paired(r, base_r, 1.0, 1.0)
        tests[name]["map"] = st.paired_map(r, base_r)
        log.info("   paired vs base %-40s dS=%+.3f CI90 [%+.3f, %+.3f] | dmAP=%+.3f CI95 [%+.3f, %+.3f]", name, tests[name]["mean"],
                 *tests[name]["ci90"], tests[name]["map"]["mean"], *tests[name]["map"]["ci95"])

    def accept(t):
        return t["ci90"][0] > 0 or (t["ci90"][0] > -0.02 and t["map"]["ci95"][0] > 0)

    kept = [n for n, t in tests.items() if accept(t)]
    best_A = max(kept, key=lambda n: (tests[n]["ci90"][0] > 0, resA[n][1]["S"] + tests[n]["map"]["mean"])) if kept else "A0 base (motion+diff)"
    groups_sel = FEATURES_A[best_A]
    ref_r, ref_s = (resA[best_A] if best_A in resA else (base_r, base_s))
    log.info("Stage A: kept %s -> selected features %s", kept, best_A)

    # ---------- B: labels / horizons / window ----------
    resB = {}
    for lab in list(LABELS)[1:]:
        resB[(lab, 1.0)] = record("B", lab, groups_sel, 1.0, lab)
    resB[("L0", 2.0)] = record("B", "L0, window 2 s", groups_sel, 2.0, "L0")
    testsB = {}
    for (lab, W), (r, s) in resB.items():
        testsB[f"{lab}|W{W:g}"] = st.paired(r, ref_r, W, 1.0)
        log.info("   paired vs ref %-40s dS=%+.3f CI90 [%+.3f, %+.3f]", f"{lab}|W{W:g}", testsB[f"{lab}|W{W:g}"]["mean"], *testsB[f"{lab}|W{W:g}"]["ci90"])
    keptB = [k2 for k2, t in testsB.items() if t["ci90"][0] > 0]
    lab_sel, W_sel = "L0", 1.0
    if keptB:
        bestB = max(keptB, key=lambda k2: resB[(k2.split("|W")[0], float(k2.split("|W")[1]))][1]["S"])
        lab_sel, W_sel = bestB.split("|W")[0], float(bestB.split("|W")[1])
        ref_r, ref_s = resB[(lab_sel, W_sel)]
    log.info("Stage B: kept %s -> labels %s, W %.0f", keptB, lab_sel, W_sel)

    # ---------- C: alarm post-processing ----------
    resC = {}
    for kk in (1, 2, 3, 5):
        for al in (1.0, 0.6, 0.3):
            s = st.score(ref_r, W_sel, kk, al)
            resC[(kk, al)] = s
            rows.append({"stage": "C", "name": f"k={kk}, EMA alpha={al}", "groups": "+".join(groups_sel), "W": W_sel, "labels": lab_sel,
                         "k": kk, "alpha": al, "oof_mAP": ref_r["oof_mAP"], "roc_auc": ref_r["roc_auc"], **{x: v for x, v in s.items() if x != "op"}})
            log.info("C k=%d alpha=%.1f S=%.3f R@5/10/20=%.3f/%.3f/%.3f", kk, al, s["S"], s["recall@5"], s["recall@10"], s["recall@20"])
    k_sel, a_sel = 3, 1.0
    bestC = max(resC, key=lambda x: resC[x]["S"])
    if bestC != (3, 1.0):
        t = st.paired(ref_r, ref_r, W_sel, W_sel, bestC[0], 3, bestC[1], 1.0)
        log.info("   paired post-proc %s vs k3/none dS=%+.3f CI90 [%+.3f, %+.3f]", bestC, t["mean"], *t["ci90"])
        if t["ci90"][0] > 0:
            k_sel, a_sel = bestC
    log.info("Stage C: selected k=%d alpha=%.1f", k_sel, a_sel)
    pd.DataFrame(rows).to_csv(out_dir / "study_rows.csv", index=False)

    # ---------- final: selected vs baseline, both retrained on all dev ----------
    final = {}
    configs = {"baseline_v2config": (BASE["groups"], 1.0, "L0", 3, 1.0), "selected_v3": (groups_sel, W_sel, lab_sel, k_sel, a_sel)}
    held = {"test": st.test, "test_ext": st.test_ext, "test_all": st.test + st.test_ext if st.test_ext else []}
    preds_store = {}
    for cname, (groups, W, lab, kk, al) in configs.items():
        if cname == "selected_v3" and configs["selected_v3"] == configs["baseline_v2config"]:
            final[cname] = final["baseline_v2config"]
            continue
        cfg = st.cfg_for(lab)
        oof = st.run(groups, W, lab)
        cal = Calibrator("platt").fit(oof["p"], oof["y"])
        oof_streams = {v: {"times": s["times"], "score": cal(s["score"])} for v, s in oof["streams"].items()}
        ops = operating_points(oof_streams, st.recs, (5, 10, 20), kk, W, al)
        thr_f1 = best_f1_threshold(oof["y"], cal(oof["p"]))
        tr = sum((train_samples([st.recs[v] for v in st.dev], st.feats, cfg, np.random.default_rng(900)) for _ in range(3)), [])
        X, yt = build_matrix(tr, st.feats, W, groups)
        t0 = time.perf_counter()
        ens = BaggedEnsemble("lgbm", 5 if not a.quick else 2, seed=300).fit(X, yt, np.array([s.video_id for s in tr]))
        fit_s = time.perf_counter() - t0
        res = {"config": {"groups": groups, "W": W, "labels": lab, "k": kk, "alpha": al, "n_features": int(X.shape[1])},
               "oof_dev": {"mAP": oof["oof_mAP"], **{f"recall@{b}": ops[f"@{b}FA"]["recall"] for b in (5, 10, 20)}},
               "thresholds_from_dev": {f"@{b}FA": ops[f"@{b}FA"]["threshold"] for b in (5, 10, 20)}, "fit_seconds": fit_s}
        for hname, ids in held.items():
            if not ids:
                continue
            ev = eval_samples([st.recs[v] for v in ids], st.feats, st.cfg, seed=1001 if hname == "test" else 7001)
            Xh = build_matrix(ev, st.feats, W, groups)[0]
            yh, gh = np.array([s.label for s in ev], float), np.array([s.group for s in ev])
            ps = cal(ens.predict_samples(Xh))
            sm = summarize_probs(ps)
            streams = {}
            for v in ids:
                f = st.feats[v]
                idx = np.arange(0, len(f.times), st.stride)
                # same aggregation as the product (TabularPredictor): mean of calibrated member probabilities (audit M1)
                streams[v] = {"times": f.times[idx], "score": summarize_probs(cal(ens.predict_samples(video_window_matrix(f, f.times[idx], W, groups))))["p_mean"]}
            sr = {f"@{b}FA": evaluate_at(streams, st.recs, res["thresholds_from_dev"][f"@{b}FA"], kk, W, al) for b in (5, 10, 20)}
            res[hname] = {**classification_metrics(yh, sm["p_mean"], thr_f1), **nexar_map(yh, sm["p_mean"], gh, st.cfg["windows"]["eval_horizons_s"]),
                          "brier": brier(sm["p_mean"], yh), "ece": ece(sm["p_mean"], yh),
                          "error_auroc_entropy": error_detection_auroc(yh, sm["p_mean"], sm["entropy"], thr_f1),
                          "streaming": sr, "n_videos": len(ids)}
            preds_store[(cname, hname)] = (ev, sm["p_mean"], streams)
            log.info("FINAL %-18s %-8s mAP=%.3f AUC=%.3f F1=%.3f ECE=%.3f | R@5=%.3f(FA %.1f) R@10=%.3f(FA %.1f) R@20=%.3f(FA %.1f) EDR@1/2/3(10FA)=%.2f/%.2f/%.2f",
                     cname, hname, res[hname]["mAP"], res[hname]["roc_auc"], res[hname]["f1"], res[hname]["ece"],
                     sr["@5FA"]["recall"], sr["@5FA"]["fa_per_hour"], sr["@10FA"]["recall"], sr["@10FA"]["fa_per_hour"],
                     sr["@20FA"]["recall"], sr["@20FA"]["fa_per_hour"], sr["@10FA"]["edr@1s"], sr["@10FA"]["edr@2s"], sr["@10FA"]["edr@3s"])
        # efficiency of the prediction stage
        f0 = st.feats[st.test[0]]
        t1 = f0.times[len(f0.times) // 2: len(f0.times) // 2 + 1]
        for _ in range(3):
            ens.predict_samples(video_window_matrix(f0, t1, W, groups))
        f0.derived.clear()
        t0 = time.perf_counter()
        for _ in range(50):
            f0.derived.clear()  # include the relational-feature derivation cost
            ens.predict_samples(video_window_matrix(f0, t1, W, groups))
        res["latency_ms_per_window_incl_features"] = 20 * (time.perf_counter() - t0)
        res["model_size_mb"] = len(pickle.dumps(ens.members)) / 1e6
        final[cname] = res
        if cname == "selected_v3" or configs["selected_v3"] == configs["baseline_v2config"]:
            sel_ens, sel_cal, sel_X, sel_ops = ens, cal, X, ops
    # paired comparison on held-out (if different)
    comp = {}
    if configs["selected_v3"] != configs["baseline_v2config"] and st.test_ext:
        for hname in ("test_all",):
            evb, pb, sb = preds_store[("baseline_v2config", hname)]
            evs, ps_, ss = preds_store[("selected_v3", hname)]
            df = pd.DataFrame({"video_id": [s.video_id for s in evb], "y": [s.label for s in evb], "group": [s.group for s in evb], "base": pb, "v3": ps_})
            comp[hname] = bootstrap_map(df, ["base", "v3"], st.cfg["windows"]["eval_horizons_s"], n=1000, pairs=[("v3", "base")])
    results = {"timestamp": STAMP, "dev": a.dev, "n_dev": len(st.dev), "n_test": len(st.test), "n_test_ext": len(st.test_ext),
               "environment": environment_report(), "baseline": BASE, "rule": "keep a change only if paired video-bootstrap 90% CI of ΔS > 0",
               "stageA_tests": tests, "stageB_tests": testsB, "selected": {"features": best_A, "groups": groups_sel, "labels": lab_sel, "W": W_sel,
                                                                            "k": k_sel, "alpha": a_sel},
               "rows": rows, "final": final, "map_comparison_heldout": comp, "runtime_minutes": (time.perf_counter() - t_start) / 60,
               "process_rss_gb": psutil.Process().memory_info().rss / 1e9}
    save_json(results, out_dir / "results_ew.json")
    save_json(results, resolve(f"experiments/latest/results_ew_{a.dev}.json"))
    # bundle (deployable) for the selected config
    bdir = resolve(f"models/safedrive_v3_candidate_{a.dev}")
    bdir.mkdir(parents=True, exist_ok=True)
    with open(bdir / "model.pkl", "wb") as fh:
        pickle.dump({"members": sel_ens.members, "calibrator": sel_cal, "pca": None, "col_median": np.median(sel_X, 0).astype(np.float32)}, fh)
    g_sel, W_s, lab_s, k_s, a_s = configs["selected_v3"]
    rc = RiskConfig(trend_weight=st.cfg["risk"]["trend_weight"], looming_weight=st.cfg["risk"]["looming_weight"], consecutive_steps=k_s,
                    cue_reference=cue_reference(np.concatenate([st.feats[v].motion for v in st.dev])))
    rc.warning = float(sel_ops["@10FA"]["threshold"])
    rc.high = float(sel_ops["@5FA"]["threshold"]) if sel_ops["@5FA"]["threshold"] > rc.warning else min(0.99, rc.warning + 0.5 * (1 - rc.warning))
    rc.caution = float(sel_ops["@20FA"]["threshold"]) if sel_ops["@20FA"]["threshold"] < rc.warning else 0.8 * rc.warning
    rc.mi_threshold = 1.0  # set by recalibration script if used
    save_json({"type": "tabular", "feature_version": 1, "description": f"SafeDrive v3 candidate: LightGBM x5 on {'+'.join(g_sel)} (W={W_s:g}s, labels {lab_s}, k={k_s}, EMA {a_s})",
               "window_s": W_s, "warmup_s": W_s, "groups": g_sel, "classifier": "lgbm", "n_members": len(sel_ens.members), "calibration": "platt",
               "threshold_f1_val": 0.5, "fps": 10.0, "uses_cnn": False, "feature_names": channel_names(g_sel), "motion_features": MOTION_FEATURES,
               "trained": STAMP, "experiment_dir": str(out_dir), "alarm_ema_alpha": a_s,
               "risk": {**rc.to_dict(), "level_rule": "WARNING = threshold at 10 FA/h, HIGH = 5 FA/h, CAUTION = 20 FA/h (dev OOF streams)"}},
              bdir / "bundle.json")
    log.info("saved %s (%.1f min)", out_dir, results["runtime_minutes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
