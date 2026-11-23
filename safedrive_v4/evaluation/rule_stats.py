"""Per-rule evidence statistics on DEV videos only (held-out untouched):
firing rate in the 2 s before annotated events (positive videos) vs firing rate in normal-driving videos (per step), the lift,
and the alarm-relevant rate: share of normal videos in which the rule fires at least once per minute on average.
Usage: python -m safedrive_v4.evaluation.rule_stats
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from safedrive_v4.config import load_config, resolve  # noqa: E402
from safedrive_v4.evaluation.precompute import load_channels  # noqa: E402
from safedrive_v4.features.graph_features import KG_FEATURES, SG_FEATURES  # noqa: E402
from safedrive_v4.reasoning.rules import RULES  # noqa: E402


def main() -> int:
    from src.data.dataset_loader import load_dataset
    from src.utils.config import load_config as v3cfg
    from src.utils.io import load_json
    c = v3cfg()
    recs = {r.video_id: r for r in load_dataset(resolve(c["paths"]["annotations"]), resolve(c["paths"]["raw_dir"]))}
    sp = load_json(resolve("data/processed/splits_full.json"))
    dev = sp["train"] + sp["val"] + sp.get("dev_new", [])
    cfg = load_config()
    pre, neg, neg_per_min = {r.rule_id: [] for r in RULES}, {r.rule_id: [] for r in RULES}, {r.rule_id: [] for r in RULES}
    edges_pre, edges_neg = [], []
    for v in dev:
        r = recs.get(v)
        try:
            ch = load_channels(v, cfg)
        except FileNotFoundError:
            continue
        t, K, S = ch["times"], ch["kg"], ch["sg"]
        for i, rule in enumerate(RULES):
            col = K[:, KG_FEATURES.index(f"kg_rule_{rule.rule_id.lower()}")] > 0
            if r.label == 1 and r.time_of_event is not None:
                m = (t >= r.time_of_event - 2.0) & (t < r.time_of_event)
                if m.any():
                    pre[rule.rule_id].append(col[m].mean())
            elif r.label == 0:
                neg[rule.rule_id].append(col.mean())
                onsets = int(np.sum(np.diff(col.astype(int)) == 1))
                neg_per_min[rule.rule_id].append(onsets / max((t[-1] - t[0]) / 60.0, 1e-6))
    out = []
    for rule in RULES:
        a, b = float(np.mean(pre[rule.rule_id])), float(np.mean(neg[rule.rule_id]))
        out.append({"rule": rule.rule_id, "name": rule.name, "kind": rule.kind, "rate_2s_before_event": a, "rate_normal_driving": b,
                    "lift": a / b if b > 0 else None, "onsets_per_min_normal": float(np.mean(neg_per_min[rule.rule_id]))})
    res = {"dev_videos": len(dev), "rules": out,
           "note": "dev only; lift > 1 means the rule fires more often just before events than in normal driving"}
    (resolve("experiments/v4") / "rule_stats_dev.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for r in out:
        print(f"{r['rule']} {r['name']:<38} pre {r['rate_2s_before_event']:.3f} normal {r['rate_normal_driving']:.3f} "
              f"lift {r['lift'] if r['lift'] is None else round(r['lift'], 2)}  onsets/min(normal) {r['onsets_per_min_normal']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
