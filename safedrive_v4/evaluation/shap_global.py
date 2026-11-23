"""Global Tree-SHAP importance (feature and group shares) of the A3 (v3) and A6 (v4 research) models on the held-out evaluation
windows (descriptive only; nothing is selected on it). Output: experiments/v4/shap_global.json"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from safedrive_v4.config import resolve  # noqa: E402


def main() -> int:
    from safedrive_v4.evaluation.ablation import V4Study
    from safedrive_v4.explainability import shap_tree
    from safedrive_v4.pipeline import ResearchModel, V3ReadOnlyModel
    from src.prediction.windows import eval_samples
    st = V4Study(False)
    ids = st.test + st.test_ext
    ev = eval_samples([st.recs[v] for v in ids], st.feats, st.cfg, seed=7001)
    out = {}
    for name, m in (("A3 (v3 model)", V3ReadOnlyModel()), ("A6 (v4 research model)", ResearchModel())):
        X = st.matrix(ev, 1.0, m.groups)[0]
        ex = shap_tree.explain(m.members, X)
        g = shap_tree.global_importance(ex["phi"], m.names, 20)
        pos = np.array([s.label for s in ev]) == 1
        g_pos = shap_tree.global_importance(ex["phi"][pos], m.names, 10)
        out[name] = {"n_windows": int(len(ev)), "local_accuracy_max_error": ex["local_accuracy_error"],
                     "member_rank_stability": shap_tree.member_rank_stability(ex["phi_members"][:, ::25]),
                     "group_stability": shap_tree.group_stability(ex["phi_members"], m.names),
                     "group_share_all": g["group_share"], "group_share_positive_windows": g_pos["group_share"],
                     "top_features": g["top_features"]}
        print(name, json.dumps(g["group_share"]), "feature stability", round(out[name]["member_rank_stability"], 3), "group", out[name]["group_stability"])
    resolve("experiments/v4/shap_global.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
