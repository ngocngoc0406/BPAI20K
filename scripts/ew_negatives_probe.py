"""Dev-only probe (OOF streams, no test data): do streaming-matched negatives (dense / mined) improve early warning?
Compares L0 vs L7 (dense negatives) vs L8 (dense + mined) for the base and the ego+interaction(F) feature sets.
Usage: python scripts/ew_negatives_probe.py --dev orig|full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_ew_study import Study  # noqa: E402

from src.utils.config import resolve  # noqa: E402
from src.utils.io import save_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="orig")
    a = ap.parse_args()
    st = Study(a.dev, False)
    out = {}
    feats = {"base": ["motion", "diff"], "ego+rel(F)": ["motion", "diff", "ego_f", "rel_f"]}
    for fname, groups in feats.items():
        ref = st.run(groups, 1.0, "L0")
        out[f"{fname}|L0"] = {k: v for k, v in st.score(ref, 1.0).items() if k != "op"} | {"mAP": ref["oof_mAP"]}
        print(fname, "L0", round(out[f"{fname}|L0"]["S"], 3), flush=True)
        for lab in ("L7 dense negatives (40/video)", "L8 dense negatives (40/video) + mined"):
            r = st.run(groups, 1.0, lab)
            s = st.score(r, 1.0)
            t = st.paired(r, ref, 1.0, 1.0)
            out[f"{fname}|{lab}"] = {k: v for k, v in s.items() if k != "op"} | {"mAP": r["oof_mAP"], "paired_vs_L0": t}
            print(fname, lab, f"S={s['S']:.3f} R@5/10/20={s['recall@5']:.3f}/{s['recall@10']:.3f}/{s['recall@20']:.3f} mAP={r['oof_mAP']:.3f}",
                  f"dS={t['mean']:+.3f} CI90 [{t['ci90'][0]:+.3f},{t['ci90'][1]:+.3f}]", flush=True)
    save_json(out, resolve(f"experiments/latest/ew_negatives_probe_{a.dev}.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
