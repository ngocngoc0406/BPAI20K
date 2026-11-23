"""Full-dataset split (pre-declared before any v3 experiment):
- the original 400-video split is kept unchanged (train 280 / val 60 / test 60)
- every additional valid Nexar video is assigned by label-stratified random split (seed 2026): 85 % 'dev_new', 15 % 'test_ext'
- development data for v3 = train + val + dev_new; held-out = test (original) and test_ext (never used for any choice)
Output: data/processed/splits_full.json, data/processed/validation_full.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.data.dataset_loader import load_dataset  # noqa: E402
from src.data.dataset_validator import validate  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.io import load_json, save_json  # noqa: E402


def main() -> int:
    cfg = load_config()
    old = load_json(resolve(cfg["paths"]["splits"]))
    used = set(old["train"] + old["val"] + old["test"])
    recs = [r for r in load_dataset(resolve(cfg["paths"]["annotations"]), resolve(cfg["paths"]["raw_dir"])) if r.video_id not in used]
    rep = validate(recs, probe=True)
    good = [r for r in recs if r.video_id in set(rep.ok)]
    rng = np.random.default_rng(2026)
    new = {"dev_new": [], "test_ext": []}
    for lab in (0, 1):
        ids = sorted(r.video_id for r in good if r.label == lab)
        rng.shuffle(ids)
        n_te = int(round(0.15 * len(ids)))
        new["test_ext"] += ids[:n_te]
        new["dev_new"] += ids[n_te:]
    out = {**old, **{k: sorted(v) for k, v in new.items()}}
    allv = sum(out.values(), [])
    assert len(allv) == len(set(allv)), "leakage between splits"
    save_json(out, resolve("data/processed/splits_full.json"))
    save_json({"issues": rep.issues, "n_checked": len(recs), "n_ok": len(rep.ok)}, resolve("data/processed/validation_full.json"))
    print({k: len(v) for k, v in out.items()}, "issues:", len(rep.issues))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
