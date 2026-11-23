"""Hypothesis tests H2 (uncertainty-aware reasoning vs perception errors) and H4 (adaptive warning), plus the new ablation
configurations A3 (v3 + uncertainty-aware KG reasoning) and A5 (v3 + scene graph + uncertainty-aware KG = KG-guided prediction).
Declared BEFORE running (same v3 protocol, folds, labels, windows; held-out evaluated once, never used for a choice):

H2  Models are trained on CLEAN dev folds and their OOF windows / streams are evaluated on clean AND perception-error versions
    of the held-out folds (track-level noise, safedrive_v4/evaluation/precompute_u.py). Degradation D = metric(noisy) - metric(clean).
    H2 is SUPPORTED only if D(A3) > D(A2) (less degradation) with a paired video-bootstrap 90 % CI of D(A3) - D(A2) > 0, for S or
    for mAP; otherwise NOT SUPPORTED. A0 (v3) is reported as the reference.
H4  Warning policies on the v3 predictor (A0): every policy re-derives its thresholds on dev OOF streams for the SAME false-alarm
    budgets (5 / 10 / 20 FA/h), so recall is compared at equal false alarms. Policy parameters are chosen on dev OOF only (max S),
    then the held-out streams of the deployed v3 model are evaluated once with the dev thresholds.
      P0  v3 alarm rule (calibrated probability, k = 3)
      P1  evidence-gated: an alarm step needs uncertainty-aware KG risk evidence E_t >= e (temporal opinion expectation)
      P2  reliability-aware: steps whose observation reliability expectation < r become CAUTIOUS (shown, not a full warning);
          reported: full-warning recall / FA, coverage (share of steps eligible for full warnings) and selective FA/h
    H4 is SUPPORTED only if P1 or P2 reaches a higher S than P0 on held-out with recall at 10 FA/h not lower than P0's.
Usage: python -m safedrive_v4.evaluation.hypotheses [--part h2,a35,h4]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from safedrive_v4.config import resolve  # noqa: E402
from safedrive_v4.evaluation.ablation import LABELS, V3_GROUPS, V4Study, crossfit_platt, heldout, model, window_metrics  # noqa: E402
from safedrive_v4.evaluation.precompute_u import load_u  # noqa: E402
from safedrive_v4.features.graph_features import window_matrix  # noqa: E402
from safedrive_v4.reasoning.uncertain_engine import KGU_FEATURES  # noqa: E402
from src.evaluation.metrics import classification_metrics, nexar_map  # noqa: E402
from src.evaluation.streaming import evaluate_at, operating_points, per_video_table, score_from_table  # noqa: E402
from src.features.window_features import video_window_matrix  # noqa: E402
from src.prediction.light_models import Calibrator  # noqa: E402
from src.prediction.windows import train_samples  # noqa: E402


class HStudy(V4Study):
    def __init__(self):
        super().__init__(False)
        for v in self.feats:
            self.ch[v]["kgu"] = load_u(v, "clean")["kgu"]
        self.noisy: dict = {}

    def load_noisy(self):
        import copy
        for v in self.dev:
            z = load_u(v, "noisy")
            f2 = copy.copy(self.feats[v])
            f2.tracks, f2.motion, f2.derived = z["tracks"], z["motion"], {}
            self.noisy[v] = (f2, {"sg": z["sg"], "kg": z["kg"], "kgu": z["kgu"], "times": self.ch[v]["times"]})

    def matrix_video(self, vid, t_ends, W, groups, noisy: bool = False):
        v3, v4 = self._split(groups)
        f, ch = (self.noisy[vid] if noisy else (self.feats[vid], self.ch[vid]))
        parts = [video_window_matrix(f, t_ends, W, v3)] if v3 else []
        for name, w in v4:
            parts.append(window_matrix(f.times, ch[name], t_ends, w or W, f.fps))
        return np.concatenate(parts, 1).astype(np.float32)

    def _split(self, groups):
        v3 = [g for g in groups if not g.startswith(("sg", "kg"))]
        v4 = [(g.partition("@")[0], float(g.partition("@")[2]) if g.partition("@")[2] else None) for g in groups if g.startswith(("sg", "kg"))]
        return v3, v4

    def run2(self, groups, kind="lgbm", with_noisy=False) -> tuple[dict, dict | None]:
        """OOF on clean and (optionally) noisy evaluation inputs with the SAME fold models."""
        cfg = self.cfg_for(LABELS)
        outs = {"clean": {"y": [], "g": [], "p": [], "streams": {}}, "noisy": {"y": [], "g": [], "p": [], "streams": {}}}
        for k in range(5):
            tr_recs = [self.recs[v] for v in self.dev if self.fold[v] != k]
            r2 = np.random.default_rng(600 + k)   # ONE generator for the 3 draws, exactly as the v3 Study.run
            tr = sum((train_samples(tr_recs, self.feats, cfg, r2) for _ in range(3)), [])
            X, yt = self.matrix(tr, 1.0, groups)
            m = model(kind, 42).fit(X, yt)
            ev = self.oof_ev[k]
            for mode in (["clean", "noisy"] if with_noisy else ["clean"]):
                nz = mode == "noisy"
                by = {}
                for i, s in enumerate(ev):
                    by.setdefault(s.video_id, []).append(i)
                Xe = np.zeros((len(ev), X.shape[1]), np.float32)
                for v, ii in by.items():
                    Xe[ii] = self.matrix_video(v, np.array([ev[i].t_end for i in ii]), 1.0, groups, nz)
                o = outs[mode]
                o["p"].append(m.predict_proba(Xe)[:, 1]), o["y"].append([s.label for s in ev]), o["g"].append([s.group for s in ev])
                for v in [v for v in self.dev if self.fold[v] == k]:
                    f = self.feats[v]
                    idx = np.arange(0, len(f.times), self.stride)
                    o["streams"][v] = {"times": f.times[idx], "score": m.predict_proba(self.matrix_video(v, f.times[idx], 1.0, groups, nz))[:, 1]}
        res = {}
        for mode, o in outs.items():
            if not o["p"]:
                continue
            y, g, p = np.concatenate(o["y"]).astype(float), np.concatenate(o["g"]), np.concatenate(o["p"])
            res[mode] = {"y": y, "g": g, "p": p, "streams": o["streams"], "oof_mAP": nexar_map(y, p, g, self.cfg["windows"]["eval_horizons_s"])["mAP"],
                         "roc_auc": classification_metrics(y, p, 0.5)["roc_auc"], "vids": np.array([s.video_id for k_ in range(5) for s in self.oof_ev[k_]])}
        return res["clean"], res.get("noisy")


def boot_degradation(st, a_clean, a_noisy, b_clean, b_noisy, n=1000):
    """Paired video bootstrap of [D(a) - D(b)] for S (thresholds re-derived per resample) and for window mAP."""
    import pandas as pd

    from src.evaluation.metrics import nexar_map as nm
    grid = np.linspace(0.2, 1.0, 81)
    tabs = [per_video_table(r["streams"], st.recs, grid, 3, 1.0, 1.0) for r in (a_clean, a_noisy, b_clean, b_noisy)]
    nv = len(tabs[0]["ids"])
    rng = np.random.default_rng(0)
    ds, dm = [], []
    df = pd.DataFrame({"v": a_clean["vids"]})
    vids = df["v"].values
    uv = np.unique(vids)
    pos = {v: np.where(vids == v)[0] for v in uv}
    hz = st.cfg["windows"]["eval_horizons_s"]
    for _ in range(n):
        w = np.bincount(rng.integers(0, nv, nv), minlength=nv).astype(float)
        sc = [score_from_table(t, w) for t in tabs]
        ds.append((sc[1] - sc[0]) - (sc[3] - sc[2]))
        pick = rng.choice(uv, len(uv), replace=True)
        ii = np.concatenate([pos[v] for v in pick])
        m = [nm(r["y"][ii], r["p"][ii], r["g"][ii], hz)["mAP"] for r in (a_clean, a_noisy, b_clean, b_noisy)]
        dm.append((m[1] - m[0]) - (m[3] - m[2]))
    q = lambda x: [float(np.quantile(x, 0.05)), float(np.quantile(x, 0.95))]  # noqa: E731
    return {"dS_diff_mean": float(np.mean(ds)), "dS_diff_ci90": q(ds), "dmAP_diff_mean": float(np.mean(dm)), "dmAP_diff_ci90": q(dm)}


def step_values(st, vid, times, col, which="kgu"):
    C = st.ch[vid][which][:, KGU_FEATURES.index(col)]
    idx = np.clip(np.round(np.asarray(times) * 10).astype(int), 0, len(C) - 1)
    return C[idx]


def policy_streams(st, streams, policy, param):
    out = {}
    for v, s in streams.items():
        sc = np.asarray(s["score"], float)
        if policy == "P1":
            ev = step_values(st, v, s["times"], "kgu_temporal_expectation")
            sc = np.where(ev >= param, sc, 0.0)
        elif policy == "P2":
            rel = step_values(st, v, s["times"], "kgu_reliability_expectation")
            sc = np.where(rel >= param, sc, 0.0)    # full warnings only on reliable steps; the rest are CAUTIOUS
        out[v] = {"times": s["times"], "score": sc}
    return out


def coverage(st, streams, param):
    vals = np.concatenate([step_values(st, v, s["times"], "kgu_reliability_expectation") for v, s in streams.items()])
    return float((vals >= param).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="h2,a35,h4")
    a = ap.parse_args()
    parts = a.part.split(",")
    out_dir = resolve("experiments/v4") / f"hypotheses_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    st = HStudy()
    hz = st.cfg["windows"]["eval_horizons_s"]
    R: dict = {"protocol": __doc__}
    save = lambda: (out_dir / "results.json").write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")  # noqa: E731
    t0 = time.perf_counter()
    if "h2" in parts or "a35" in parts:
        st.load_noisy()
        runs = {}
        # reproduction guard: A0 must reproduce the v3 OOF numbers before anything else is compared
        c0, _ = st.run2(V3_GROUPS)
        assert abs(c0["oof_mAP"] - 0.6170491281919167) < 1e-9, f"A0 does not reproduce v3: {c0['oof_mAP']}"
        for name, groups in (("A0 v3", V3_GROUPS), ("A2 v3 + KG (plain)", V3_GROUPS + ["kg"]), ("A3 v3 + uncertainty-aware KG", V3_GROUPS + ["kgu"]),
                             ("A5 v3 + SG + uncertainty-aware KG", V3_GROUPS + ["sg", "kgu"])):
            c, n = st.run2(groups, with_noisy=not name.startswith("A5"))
            runs[name] = (c, n)
            row = {"groups": groups, "clean": {"oof": window_metrics(c["y"], c["p"], c["g"], hz),
                                               "oof_calibrated": window_metrics(c["y"], crossfit_platt(c["p"], c["y"], c["vids"]), c["g"], hz),
                                               **{k: v for k, v in st.score(c, 1.0).items() if k != "op"}}}
            if n is not None:
                row["noisy"] = {"oof": window_metrics(n["y"], n["p"], n["g"], hz), **{k: v for k, v in st.score(n, 1.0).items() if k != "op"}}
                row["degradation"] = {"S": row["noisy"]["S"] - row["clean"]["S"], "mAP": row["noisy"]["oof"]["mAP"] - row["clean"]["oof"]["mAP"]}
            if not name.startswith("A0"):
                ref = runs["A0 v3"][0]
                dS, dM = st.paired(c, ref, 1.0, 1.0), st.paired_map(c, ref)
                row.update(dS_vs_A0=dS, dmAP_vs_A0=dM,
                           verdict="better than A0" if (dS["ci90"][0] > 0 or (dS["ci90"][0] > -0.02 and dM["ci95"][0] > 0)) else "no evidence of improvement over A0")
            R.setdefault("ablation_oof", {})[name] = row
            print(name, json.dumps({"S": row["clean"]["S"], "mAP": row["clean"]["oof"]["mAP"], **({"deg": row["degradation"]} if "degradation" in row else {}),
                                    "verdict": row.get("verdict", "reference")}), f"{time.perf_counter() - t0:.0f}s", flush=True)
            save()
        a3, a2 = runs["A3 v3 + uncertainty-aware KG"], runs["A2 v3 + KG (plain)"]
        bd = boot_degradation(st, a3[0], a3[1], a2[0], a2[1])
        sup = bd["dS_diff_ci90"][0] > 0 or bd["dmAP_diff_ci90"][0] > 0
        R["H2"] = {"degradation_A3_minus_A2": bd, "verdict": "SUPPORTED" if sup else "NOT SUPPORTED",
                   "degradation_A0_minus_A2": boot_degradation(st, runs["A0 v3"][0], runs["A0 v3"][1], a2[0], a2[1])}
        print("H2", json.dumps(R["H2"]), flush=True)
        save()
        # held-out (once) for A3 and A5
        held = {}
        for name in ("A0 v3", "A3 v3 + uncertainty-aware KG", "A5 v3 + SG + uncertainty-aware KG"):
            groups = R["ablation_oof"][name]["groups"]
            h = heldout(st, groups, "lgbm", runs[name][0] | {"y": runs[name][0]["y"]})
            held[name] = h
            R.setdefault("heldout", {})[name] = {"window": h["window"], "streaming": h["streaming"]}
            print("HELDOUT", name, json.dumps({"mAP": h["window"]["mAP"], **{b: (h["streaming"][b]["recall"], h["streaming"][b]["fa_per_hour"]) for b in h["streaming"]}}), flush=True)
        import pandas as pd

        from src.evaluation.bootstrap import bootstrap_map
        base = held["A0 v3"]
        for name in ("A3 v3 + uncertainty-aware KG", "A5 v3 + SG + uncertainty-aware KG"):
            df = pd.DataFrame({"video_id": [s.video_id for s in base["_ev"]], "y": [s.label for s in base["_ev"]], "group": [s.group for s in base["_ev"]],
                               "a0": base["_p"], "cand": held[name]["_p"]})
            R.setdefault("heldout_paired_mAP", {})[name] = bootstrap_map(df, ["a0", "cand"], hz, n=1000, pairs=[("cand", "a0")])["paired"]["cand - a0"]
        save()
    if "h4" in parts:
        c = st.run2(V3_GROUPS)[0]
        cal = Calibrator("platt").fit(c["p"], c["y"])
        dev_streams = {v: {"times": s["times"], "score": cal(s["score"])} for v, s in c["streams"].items()}
        grid = {"P1": [0.12, 0.15, 0.2, 0.25, 0.3], "P2": [0.40, 0.45, 0.50, 0.55]}
        dev = {"P0": {"param": None, "op": operating_points(dev_streams, st.recs, (5, 10, 20), 3, 1.0, 1.0)}}
        for pol, vals in grid.items():
            best = None
            for v in vals:
                op = operating_points(policy_streams(st, dev_streams, pol, v), st.recs, (5, 10, 20), 3, 1.0, 1.0)
                S = float(np.mean([op[f"@{b}FA"]["recall"] for b in (5, 10, 20)]))
                if best is None or S > best["S"]:
                    best = {"param": v, "op": op, "S": S}
            dev[pol] = best
        # held-out: deployed v3 model streams (calibrated ensemble mean), dev thresholds per policy
        from src.prediction.tabular_predictor import TabularPredictor
        v3 = TabularPredictor.load(resolve("models/safedrive_v3"))
        ids = st.test + st.test_ext
        hs = {}
        for v in ids:
            f = st.feats[v]
            idx = np.arange(0, len(f.times), st.stride)
            hs[v] = {"times": f.times[idx], "score": v3.predict_matrix(v3.features(f, f.times[idx]))["p_mean"]}
        H4 = {}
        for pol, d in dev.items():
            streams = hs if pol == "P0" else policy_streams(st, hs, pol, d["param"])
            ev = {f"@{b}FA": evaluate_at(streams, st.recs, d["op"][f"@{b}FA"]["threshold"], 3, 1.0, 1.0) for b in (5, 10, 20)}
            H4[pol] = {"param_from_dev": d["param"], "dev_S": float(np.mean([d["op"][f"@{b}FA"]["recall"] for b in (5, 10, 20)])),
                       "heldout": ev, "heldout_S": float(np.mean([ev[b]["recall"] for b in ev])),
                       "coverage_heldout": coverage(st, hs, d["param"]) if pol == "P2" else 1.0}
            print("H4", pol, json.dumps({"param": d["param"], "dev_S": H4[pol]["dev_S"], "heldout_S": H4[pol]["heldout_S"],
                                         **{b: (round(ev[b]["recall"], 3), round(ev[b]["fa_per_hour"], 2), round(ev[b]["edr@2s"], 3)) for b in ev}}), flush=True)
        p0 = H4["P0"]
        sup = any(H4[p]["heldout_S"] > p0["heldout_S"] and H4[p]["heldout"]["@10FA"]["recall"] >= p0["heldout"]["@10FA"]["recall"] for p in ("P1", "P2"))
        R["H4"] = {**H4, "verdict": "SUPPORTED" if sup else "NOT SUPPORTED"}
        save()
    (resolve("experiments/v4") / "latest_hypotheses.json").write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")
    print("saved", out_dir, f"{(time.perf_counter() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
