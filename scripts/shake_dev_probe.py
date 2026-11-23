"""Dev-only robustness probe for ego-motion compensation under camera shake (no test data).
Models trained on the 280 TRAIN videos; evaluated on clips of the 60 VAL videos re-extracted (full perception) under
clean / shake-1 / shake-2 / frame-drop-2 / low-light-2. Feature variants: base, +ego, +interactions (none / H / F), +ego+interactions.
Output: experiments/latest/shake_dev_probe.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.data.dataset_loader import load_dataset  # noqa: E402
from src.evaluation.metrics import classification_metrics, nexar_map  # noqa: E402
from src.features.extractor import FeatureExtractor  # noqa: E402
from src.features.window_features import build_matrix  # noqa: E402
from src.prediction.light_models import make_model  # noqa: E402
from src.prediction.windows import eval_samples, train_samples  # noqa: E402
from src.preprocessing.feature_cache import cache_path, load_features, save_features  # noqa: E402
from src.preprocessing.video_transform import make_transform  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.io import load_json, save_json  # noqa: E402

VARIANTS = {"base": ["motion", "diff"], "+ego(H)": ["motion", "diff", "ego_h"], "+rel(none)": ["motion", "diff", "rel_none"],
            "+rel(H)": ["motion", "diff", "rel_h"], "+rel(F)": ["motion", "diff", "rel_f"],
            "+ego(H)+rel(H)": ["motion", "diff", "ego_h", "rel_h"], "+ego(F)+rel(F)": ["motion", "diff", "ego_f", "rel_f"]}
CONDS = [("clean", 0), ("shake", 1), ("shake", 2), ("frame_drop", 2), ("low_light", 2)]


def main() -> int:
    cfg = load_config()
    sp = load_json(resolve(cfg["paths"]["splits"]))
    recs = {r.video_id: r for r in load_dataset(resolve(cfg["paths"]["annotations"]), resolve(cfg["paths"]["raw_dir"]))}
    fdir = resolve(cfg["paths"]["features_dir"])
    feats = {v: load_features(cache_path(fdir, v)) for v in sp["train"] + sp["val"]}
    tr = sum((train_samples([recs[v] for v in sp["train"]], feats, cfg, np.random.default_rng(77)) for _ in range(3)), [])
    models = {}
    for name, g in VARIANTS.items():
        X, y = build_matrix(tr, feats, 1.0, g)
        models[name] = make_model("lgbm", 42).fit(X, y)
    ev = eval_samples([recs[v] for v in sp["val"]], feats, cfg, seed=1000)
    by_vid: dict[str, list] = {}
    for s in ev:
        by_vid.setdefault(s.video_id, []).append(s)
    fx = FeatureExtractor(cfg, use_cnn=False)
    out = {}
    for cname, sev in CONDS:
        tag = f"devprobe_{cname}_{sev}"
        cf = {}
        for vid, ss in by_vid.items():
            cp = cache_path(fdir, vid, tag)
            if cp.exists():
                cf[vid] = load_features(cp)
                continue
            t_hi, t_lo = max(s.t_end for s in ss), max(min(s.t_end for s in ss) - 1.0 - 2.0, 0.0)
            f, _ = fx.extract(recs[vid].path, vid, start_s=t_lo, end_s=t_hi + 0.05,
                              transform=make_transform(cname, sev, int(vid) if vid.isdigit() else 0))
            save_features(f, cp)
            cf[vid] = f
        y = np.array([s.label for s in ev], float)
        g = np.array([s.group for s in ev])
        row = {}
        for name, grp in VARIANTS.items():
            p = models[name].predict_proba(build_matrix(ev, cf, 1.0, grp)[0])[:, 1]
            row[name] = {"mAP": nexar_map(y, p, g, cfg["windows"]["eval_horizons_s"])["mAP"], "roc_auc": classification_metrics(y, p, 0.5)["roc_auc"]}
        out[f"{cname}_{sev}"] = row
        print(f"{cname}_{sev}: " + "  ".join(f"{n} {v['mAP']:.3f}" for n, v in row.items()), flush=True)
    save_json(out, resolve("experiments/latest/shake_dev_probe.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
