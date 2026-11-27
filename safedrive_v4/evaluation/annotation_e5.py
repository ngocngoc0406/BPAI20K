"""E5: active annotation vs random sampling under the same annotation budget (DEV only; the oracle is the EXISTING human Nexar
clip label; no new label is created and the held-out set is not used).

Setup: evaluation set = dev fold 0 (its v3-protocol evaluation windows); pool = dev folds 1-4. Each strategy starts from the same
random seed set (40 videos), then selects 40 videos per round up to 360, "annotates" them by revealing their existing Nexar label,
retrains a research LightGBM (3 bagged members, v3 features, v3 labels L7) and evaluates window mAP on fold 0.
Reported per strategy and budget (5 seeds): eval mAP, share of positive (event) videos selected, number of distinct condition
combinations covered; paired difference to random (mean and 90 % interval over seeds) for the area under the mAP curve.
Usage: python -m safedrive_v4.evaluation.annotation_e5 [--seeds 5]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from safedrive_v4.annotation.active import STRATEGIES, model_scores, rarity_scores, select  # noqa: E402
from safedrive_v4.config import resolve  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-size", type=int, default=40)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--budget", type=int, default=360)
    a = ap.parse_args()
    from safedrive_v4.evaluation.ablation import LABELS, V3_GROUPS, V4Study
    from safedrive_v4.evaluation.precompute import load_channels
    from safedrive_v4.features.graph_features import KG_FEATURES
    from src.evaluation.metrics import nexar_map
    from src.prediction.light_models import BaggedEnsemble
    from src.prediction.windows import train_samples
    st = V4Study(False)
    cfg = st.cfg_for(LABELS)
    ev = st.oof_ev[0]
    Xe, ye = st.matrix(ev, 1.0, V3_GROUPS)
    ge = np.array([s.group for s in ev])
    pool_all = [v for v in st.dev if st.fold[v] != 0]
    times_of = {v: st.feats[v].times for v in pool_all}
    cols = [KG_FEATURES.index(f"kg_rule_{r}") for r in ("r01", "r04", "r05")]
    rule_rate = {v: float((load_channels(v, st.v4cfg)["kg"][:, cols] > 0).mean()) for v in pool_all}
    rar_all = rarity_scores(pool_all, st.recs, rule_rate)
    combo = {v: (st.recs[v].meta.get("light_conditions"), st.recs[v].meta.get("weather"), st.recs[v].meta.get("scene")) for v in pool_all}
    budgets = list(range(a.seed_size, a.budget + 1, a.batch))
    res = {s: [] for s in STRATEGIES}
    t0 = time.perf_counter()
    for seed in range(a.seeds):
        rng0 = np.random.default_rng(1000 + seed)
        seed_set = list(rng0.choice(pool_all, a.seed_size, replace=False))
        for strat in STRATEGIES:
            rng = np.random.default_rng(2000 + seed)
            labelled = list(seed_set)
            curve = []
            while True:
                recs = [st.recs[v] for v in labelled]
                tr = sum((train_samples(recs, st.feats, cfg, np.random.default_rng(600 + seed)) for _ in range(2)), [])
                X, y = st.matrix(tr, 1.0, V3_GROUPS)
                ens = BaggedEnsemble("lgbm", 3, seed=300 + seed).fit(X, y, np.array([s.video_id for s in tr]))
                m = nexar_map(ye, ens.predict_proba(Xe), ge, st.cfg["windows"]["eval_horizons_s"])["mAP"]
                curve.append({"budget": len(labelled), "map": float(m), "positive_share": float(np.mean([st.recs[v].label == 1 for v in labelled])),
                              "condition_combos": len({combo[v] for v in labelled})})
                if len(labelled) >= a.budget:
                    break
                pool = [v for v in pool_all if v not in set(labelled)]
                idx = [pool_all.index(v) for v in pool]
                unc = dis = None
                if strat in ("uncertainty", "disagreement", "hybrid"):
                    unc, dis = model_scores(ens.members, lambda v, t: st.matrix_video(v, t, 1.0, V3_GROUPS), pool, times_of, stride=10)
                labelled += select(strat, pool, a.batch, rng, unc, dis, rar_all[idx])
            res[strat].append(curve)
            print(f"seed {seed} {strat:<12} final mAP {curve[-1]['map']:.3f} pos {curve[-1]['positive_share']:.2f} "
                  f"combos {curve[-1]['condition_combos']}  {time.perf_counter() - t0:.0f}s", flush=True)
    summ = {}
    auc = {s: np.array([np.trapz([c["map"] for c in cv], budgets) / (budgets[-1] - budgets[0]) for cv in res[s]]) for s in STRATEGIES}
    for s in STRATEGIES:
        d = auc[s] - auc["random"]
        summ[s] = {"auc_map_mean": float(auc[s].mean()),
                   "final_map_mean": float(np.mean([cv[-1]["map"] for cv in res[s]])),
                   "final_positive_share_mean": float(np.mean([cv[-1]["positive_share"] for cv in res[s]])),
                   "final_condition_combos_mean": float(np.mean([cv[-1]["condition_combos"] for cv in res[s]])),
                   "auc_diff_vs_random_mean": float(d.mean()), "auc_diff_vs_random_range_over_seeds": [float(d.min()), float(d.max())],
                   "seeds_better_than_random": int((d > 0).sum())}
    out = {"protocol": __doc__, "budgets": budgets, "seeds": a.seeds, "pool": len(pool_all), "eval_windows": len(ev),
           "summary": summ, "curves": res, "minutes": (time.perf_counter() - t0) / 60}
    resolve("experiments/v4/annotation_e5.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
