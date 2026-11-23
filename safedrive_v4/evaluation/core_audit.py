"""V4 AI-core audit (Phase A): measured diagnostics per module. Nothing here changes a model, a threshold or a label.

Data use:
  * perception / scene graph / KG / decision diagnostics: DEV videos (1,276) only;
  * risk temporal behaviour: dev OOF streams of A0 (= v3 protocol, reproduced exactly);
  * uncertainty calibration / risk-coverage / XAI temporal consistency: the frozen v3 ensemble on HELD-OUT windows, descriptive only
    (no parameter is chosen on it; v3 reported error-detection AUROC on held-out in the same way).
No ground truth exists for boxes, tracks or relations, so perception and relation accuracy are NOT computed; proxies are reported
and labelled as proxies.
Usage: python -m safedrive_v4.evaluation.core_audit [--graph-videos 300]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from safedrive_v4.config import load_config, resolve  # noqa: E402


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def perception(feats, recs, vids):
    per = []
    for v in vids:
        f = feats[v]
        tr = f.tracks
        mins = max(float(f.times[-1]) / 60.0, 1e-6)
        T = len(f.times)
        conf = tr[:, 3] if len(tr) else np.zeros(0)
        tracks = defaultdict(list)
        for r in tr:
            if r[1] >= 0:
                tracks[int(r[1])].append(r)
        lens = np.array([len(x) for x in tracks.values()]) if tracks else np.zeros(0)
        cls_changes = sum(int(np.sum(np.diff(np.array([x[2] for x in rows])) != 0)) for rows in tracks.values())
        ends = {tid: (int(rows[-1][0]), rows[-1]) for tid, rows in tracks.items()}
        starts = {tid: (int(rows[0][0]), rows[0]) for tid, rows in tracks.items()}
        switch = loss = 0
        for tid, (se, re) in ends.items():
            b = re[4:8]
            border = b[0] < 0.02 or b[2] > 0.98 or b[3] > 0.98
            area = (b[2] - b[0]) * (b[3] - b[1])
            if se >= T - 2:
                continue
            cand = [t2 for t2, (ss, rs) in starts.items() if t2 != tid and 0 < ss - se <= 3 and rs[2] == re[2] and _iou(rs[4:8], b) > 0.3]
            if cand:
                switch += 1
            elif not border and area > 0.01:
                loss += 1
        m = recs[v].meta
        per.append({"vid": v, "light": m.get("light_conditions"), "weather": m.get("weather"), "dets_per_frame": len(tr) / max(T, 1),
                    "conf_mean": float(conf.mean()) if len(conf) else np.nan, "conf_lt_0_4": float((conf < 0.4).mean()) if len(conf) else np.nan,
                    "untracked_share": float((tr[:, 1] < 0).mean()) if len(tr) else np.nan,
                    "tracks_per_min": len(tracks) / mins, "track_len_median": float(np.median(lens)) if len(lens) else np.nan,
                    "short_tracks_share": float((lens < 3).mean()) if len(lens) else np.nan,
                    "id_switch_proxy_per_min": switch / mins, "midframe_loss_per_min": loss / mins,
                    "class_changes_per_track": cls_changes / max(len(tracks), 1)})
    import pandas as pd
    df = pd.DataFrame(per)
    cols = [c for c in df.columns if c not in ("vid", "light", "weather")]
    out = {"all": df[cols].median().to_dict(), "by_light": df.groupby("light")[cols].median().to_dict("index"),
           "by_weather": df.groupby("weather")[cols].median().to_dict("index"), "n_videos": len(df),
           "videos_by_light": df["light"].value_counts().to_dict()}
    return out


def scene_graph(feats, vids, cfg):
    from safedrive_v4.reasoning.uncertain_engine import UncertainReasoningEngine
    from safedrive_v4.scene_graph.builder import build_video
    life = defaultdict(list)
    flick = Counter()
    seg = Counter()
    per_min = Counter()
    inv = Counter()
    rule_fire = defaultdict(int)
    co = Counter()
    steps = 0
    minutes = 0.0
    t0 = time.perf_counter()
    for v in vids:
        f = feats[v]
        minutes += float(f.times[-1]) / 60.0
        e = UncertainReasoningEngine(cfg)
        on: dict = {}
        last_off: dict = {}
        for s in build_video(f, cfg):
            steps += 1
            r = e.evaluate(s)
            keys = set()
            leads = 0
            pairs = defaultdict(set)
            for ed in s.edges:
                if ed.relation_type == "LOCATED_IN":
                    continue
                k = (ed.source_node, ed.target_node, ed.relation_type)
                keys.add(k)
                per_min[ed.relation_type] += 1
                if ed.source_node not in s.nodes or ed.target_node not in s.nodes:
                    inv["edge_references_missing_node"] += 1
                if abs(ed.timestamp - s.timestamp) > 1e-9:
                    inv["edge_timestamp_mismatch"] += 1
                if ed.relation_type == "FOLLOWING":
                    leads += 1
                pairs[(ed.source_node, ed.target_node)].add(ed.relation_type)
                for n in (ed.source_node, ed.target_node):
                    if n.startswith("obj:") and ed.confidence > s.nodes[n].attrs.get("reliability", 1) + 1e-6:
                        inv["edge_conf_exceeds_node_reliability"] += 1
            if leads > 1:
                inv["more_than_one_lead"] += 1
            for rel in pairs.values():
                if {"APPROACHING", "MOVING_AWAY"} <= rel:
                    inv["approaching_and_moving_away"] += 1
                if {"SAME_DIRECTION", "APPROACHING"} <= rel:
                    inv["same_direction_and_approaching (geometric tension)"] += 1
            for k in keys - set(on):
                if k in last_off and s.step - last_off[k] <= 3:
                    flick[k[2]] += 1          # re-appeared within 0.3 s of disappearing
                on[k] = s.step
            for k in list(on):
                if k not in keys:
                    life[k[2]].append(s.step - on.pop(k))
                    seg[k[2]] += 1
                    last_off[k] = s.step
            fired = sorted({fd.rule_id for fd in r.base.findings})
            for a in fired:
                rule_fire[a] += 1
                for b in fired:
                    if a < b:
                        co[(a, b)] += 1
            for fd in r.base.findings:
                if fd.kind == "risk" and not fd.edge_ids:
                    inv["risk_finding_without_edge"] += 1
    rel = {}
    for k in per_min:
        L = np.array(life[k]) if life[k] else np.zeros(1)
        rel[k] = {"per_min": per_min[k] / minutes, "segments": seg[k], "lifetime_median_s": float(np.median(L)) / 10.0,
                  "lifetime_p10_s": float(np.percentile(L, 10)) / 10.0, "flicker_share": flick[k] / max(seg[k], 1)}
    jac = {f"{a}&{b}": c / max(rule_fire[a] + rule_fire[b] - c, 1) for (a, b), c in co.items() if c > 0}
    return {"videos": len(vids), "steps": steps, "minutes": minutes, "seconds": time.perf_counter() - t0, "relations": rel,
            "invariant_violations": dict(inv), "rule_fire_steps": dict(rule_fire),
            "rule_jaccard_top": dict(sorted(jac.items(), key=lambda kv: -kv[1])[:10]),
            "r03_subset_of_r02": co.get(("R02", "R03"), 0) / max(rule_fire.get("R03", 1), 1)}


def risk_and_decision(st):
    from safedrive_v4.evaluation.ablation import LABELS, V3_GROUPS
    from src.evaluation.early_warning import alarm_onsets
    from src.evaluation.streaming import operating_points
    from src.prediction.light_models import Calibrator
    from src.uncertainty.calibration import brier, ece
    r = st.run(V3_GROUPS, 1.0, LABELS, "lgbm")
    assert abs(r["oof_mAP"] - 0.6170491281919167) < 1e-9, "A0 does not reproduce v3"
    cal = Calibrator("platt").fit(r["p"], r["y"])
    streams = {v: {"times": s["times"], "score": cal(s["score"])} for v, s in r["streams"].items()}
    op = operating_points(streams, st.recs, (5, 10, 20), 3, 1.0, 1.0)
    thr10, thr20 = op["@10FA"]["threshold"], op["@20FA"]["threshold"]
    late = miss = hit = 0
    leads, p_before = [], defaultdict(list)
    chatter, episodes, transitions, minutes = 0, 0, 0, 0.0
    for v, s in streams.items():
        t, sc = np.asarray(s["times"]), np.asarray(s["score"])
        rec = st.recs[v]
        ons = alarm_onsets(t, sc, thr10, 3)
        if rec.label == 1 and rec.time_of_event is not None:
            te = rec.time_of_event
            pre = [o for o in ons if te - 5.0 <= o < te]
            if pre:
                hit += 1
                leads.append(te - pre[0])
            elif any(o >= te for o in ons):
                late += 1
            else:
                miss += 1
            for d in (3.0, 2.0, 1.0, 0.5):
                i = np.searchsorted(t, te - d)
                if 0 <= i < len(sc):
                    p_before[d].append(float(sc[i]))
        elif rec.label == 0:
            minutes += float(t[-1] - t[0]) / 60.0
            lv = np.where(sc >= thr10, 2, np.where(sc >= thr20, 1, 0))
            # k = 3 onset rule per level (as the policy): level = highest level held for the last 3 steps
            held = np.array([min(lv[max(0, i - 2): i + 1]) if i >= 2 else 0 for i in range(len(lv))])
            transitions += int(np.sum(np.diff(held) != 0))
            ep = alarm_onsets(t, sc, thr20, 3)
            episodes += len(ep)
            chatter += int(np.sum(np.diff(ep) < 2.0)) if len(ep) > 1 else 0
    n_pos = hit + late + miss
    win = {"oof_ece_raw": ece(r["p"], r["y"]), "oof_brier_raw": brier(r["p"], r["y"]),
           "oof_ece_platt": ece(cal(r["p"]), r["y"]), "oof_brier_platt": brier(cal(r["p"]), r["y"])}
    return {"thresholds_dev": {"@10FA": thr10, "@20FA": thr20}, "positives": n_pos, "hit_before_event": hit, "late_only_after_event": late,
            "missed": miss, "lead_time_median_s": float(np.median(leads)) if leads else None,
            "lead_time_quartiles_s": [float(np.percentile(leads, q)) for q in (25, 75)] if leads else None,
            "median_p_before_event": {f"-{d}s": float(np.median(v)) for d, v in p_before.items()},
            "normal_driving": {"level_transitions_per_min": transitions / max(minutes, 1e-6),
                               "moderate_or_high_episodes_per_hour": episodes / max(minutes / 60, 1e-6),
                               "re_onsets_within_2s_share": chatter / max(episodes, 1)},
            "calibration_dev_oof": win}, r, cal


def single_feature_auroc(st, r):
    """Which kinematic cues separate pre-event windows from negatives (dev OOF windows; univariate AUROC, no model)."""
    from sklearn.metrics import roc_auc_score

    from src.features.window_features import channel_names
    ev = [s for k in range(5) for s in st.oof_ev[k]]
    X, y = st.matrix(ev, 1.0, ["motion", "rel_none"])
    names = channel_names(["motion", "rel_none"])
    want = ["tau_center__last", "looming_center__last", "looming_all__max", "max_lateral_in__last", "lead_bottom_y__last",
            "res_tau_center@none__min", "res_loom_center@none__max", "ego_ttr_min@none__min", "cutin_score@none__max",
            "lead_gap_rate@none__min", "pair_ttr_min@none__min", "max_pair_closing__max"]
    out = {}
    for w in want:
        if w in names:
            a = roc_auc_score(y, X[:, names.index(w)])
            out[w] = round(float(max(a, 1 - a)), 3)
    return out


def uncertainty_heldout(st, cfg):
    from sklearn.metrics import roc_auc_score

    from safedrive_v4.evaluation.precompute_u import load_u
    from safedrive_v4.reasoning.uncertain_engine import KGU_FEATURES
    from src.prediction.tabular_predictor import TabularPredictor
    from src.prediction.windows import eval_samples
    from src.uncertainty.calibration import brier, ece
    v3 = TabularPredictor.load(resolve("models/safedrive_v3"))
    ids = st.test + st.test_ext
    ev = eval_samples([st.recs[v] for v in ids], st.feats, st.cfg, seed=7001)
    by = defaultdict(list)
    for i, s in enumerate(ev):
        by[s.video_id].append(i)
    P, MI, ENT, REL = (np.zeros(len(ev)) for _ in range(4))
    for v, ii in by.items():
        o = v3.predict_matrix(v3.features(st.feats[v], np.array([ev[i].t_end for i in ii])))
        P[ii], MI[ii], ENT[ii] = o["p_mean"], o["mutual_info"], o["entropy"]
        k = load_u(v, "clean")["kgu"][:, KGU_FEATURES.index("kgu_reliability_expectation")]
        REL[ii] = [k[min(int(round(ev[i].t_end * 10)), len(k) - 1)] for i in ii]
    y = np.array([s.label for s in ev], float)
    thr = 0.5
    err = ((P >= thr) != (y == 1)).astype(int)
    res = {"n_windows": len(ev), "ece": ece(P, y), "brier": brier(P, y),
           "error_detection_auroc": {"mutual_info (epistemic)": float(roc_auc_score(err, MI)), "entropy (total)": float(roc_auc_score(err, ENT)),
                                     "aleatoric (entropy - MI)": float(roc_auc_score(err, ENT - MI)),
                                     "1 - perception reliability": float(roc_auc_score(err, 1 - REL))}}
    curves = {}
    for name, u in (("entropy", ENT), ("mutual_info", MI), ("1-reliability", 1 - REL)):
        order = np.argsort(u)
        cov, risk = [], []
        for c in (1.0, 0.9, 0.8, 0.7, 0.5):
            keep = order[: int(round(c * len(order)))]
            cov.append(c), risk.append(float(err[keep].mean()))
        curves[name] = dict(zip([f"cov{c:.1f}" for c in cov], risk))
    res["risk_coverage_error_rate"] = curves
    # does reliability react to injected perception errors? (dev, clean vs noisy tracks, same steps)
    d = []
    for v in st.dev[:300]:
        c, n = load_u(v, "clean")["kgu"], load_u(v, "noisy")["kgu"]
        i = KGU_FEATURES.index("kgu_reliability_expectation")
        m = min(len(c), len(n))
        d.append((c[:m, i], n[:m, i]))
    cl, nz = np.concatenate([a for a, _ in d]), np.concatenate([b for _, b in d])
    res["reliability_vs_injected_errors_dev"] = {"mean_clean": float(cl.mean()), "mean_noisy": float(nz.mean()),
                                                 "auroc_noisy_vs_clean_steps": float(roc_auc_score(np.r_[np.zeros(len(cl)), np.ones(len(nz))], -np.r_[cl, nz]))}
    return res


def xai_temporal(st):
    """Stability of the SHAP group ranking over consecutive steps (held-out positive videos, frozen v3 model, descriptive)."""
    from safedrive_v4.explainability import shap_tree
    from safedrive_v4.pipeline import V3ReadOnlyModel
    m = V3ReadOnlyModel()
    agree, rho = [], []
    for v in [v for v in st.test + st.test_ext if st.recs[v].label == 1][:40]:
        f = st.feats[v]
        te = st.recs[v].time_of_event
        ts = f.times[(f.times >= te - 3) & (f.times < te)]
        if len(ts) < 3:
            continue
        X = st.matrix_video(v, ts, 1.0, m.groups)
        ex = shap_tree.explain(m.members, X)
        G = np.array([[g["shap_logit"] for g in sorted(shap_tree.group_contributions(ex["phi"][i], m.names), key=lambda g: g["group"])]
                      for i in range(len(ts))])
        top = np.abs(G).argmax(1)
        agree.append(float(np.mean(top[1:] == top[:-1])))
        for i in range(1, len(G)):
            rho.append(shap_tree._spearman(np.abs(G[i]), np.abs(G[i - 1])))
    return {"videos": len(agree), "top_group_same_as_previous_step": float(np.mean(agree)), "group_rank_spearman_consecutive": float(np.mean(rho))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-videos", type=int, default=300)
    a = ap.parse_args()
    from safedrive_v4.evaluation.ablation import V4Study
    cfg = load_config()
    st = V4Study(False)
    out = {}
    t0 = time.perf_counter()
    out["A1_perception_proxies_dev"] = perception(st.feats, st.recs, st.dev)
    print("A1 done", f"{time.perf_counter() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(3)
    gv = list(rng.choice(st.dev, a.graph_videos, replace=False))
    out["A2_scene_graph_dev"] = scene_graph(st.feats, gv, cfg)
    print("A2 done", f"{time.perf_counter() - t0:.0f}s", flush=True)
    rd, r, cal = risk_and_decision(st)
    out["A3_A7_risk_and_decision_dev_oof"] = rd
    out["A3_single_feature_auroc_dev_oof"] = single_feature_auroc(st, r)
    print("A3 done", f"{time.perf_counter() - t0:.0f}s", flush=True)
    out["A4_uncertainty_heldout_descriptive"] = uncertainty_heldout(st, cfg)
    print("A4 done", f"{time.perf_counter() - t0:.0f}s", flush=True)
    out["A6_xai_temporal_consistency_heldout"] = xai_temporal(st)
    out["runtime_min"] = (time.perf_counter() - t0) / 60
    p = resolve("experiments/v4/core_audit.json")
    p.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
