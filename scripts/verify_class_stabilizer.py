"""v3.4.2 check: per-track class stabilisation must leave every model input unchanged while removing label flicker.

Re-extracts N held-out videos with the CURRENT tracker (stabiliser on) and compares with the cached features (extracted
before the change, same extractor otherwise): motion, frame-difference and ego-motion arrays, track boxes / ids / confidences,
the deployed v3 probabilities and alarms. Reports per-track class flips before (cache) and after (re-extraction).
Output: experiments/latest/v342_class_stabilizer_check.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.data.dataset_loader import load_dataset  # noqa: E402
from src.detection.detector import COCO_ROAD  # noqa: E402
from src.features.extractor import FeatureExtractor  # noqa: E402
from src.pipeline import SafeDrivePipeline  # noqa: E402
from src.preprocessing.feature_cache import cache_path, load_features  # noqa: E402
from src.tracking.tracker import class_group  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.io import save_json  # noqa: E402


def flips(tracks: np.ndarray) -> dict:
    n_tr = n_flip_tracks = within = cross = 0
    pairs: dict = {}
    for tid in np.unique(tracks[:, 1]):
        if tid < 0:
            continue
        r = tracks[tracks[:, 1] == tid]
        c = r[np.argsort(r[:, 0]), 2].astype(int)
        n_tr += 1
        ch = np.flatnonzero(np.diff(c) != 0)
        n_flip_tracks += bool(len(ch))
        for i in ch:
            same = class_group(c[i]) == class_group(c[i + 1])
            within += same
            cross += not same
            k = "/".join(sorted((COCO_ROAD.get(c[i], str(c[i])), COCO_ROAD.get(c[i + 1], str(c[i + 1])))))
            pairs[k] = pairs.get(k, 0) + 1
    return {"tracks": n_tr, "tracks_with_class_change": n_flip_tracks, "changes_within_group": within,
            "changes_across_groups": cross, "pairs": dict(sorted(pairs.items(), key=lambda x: -x[1])[:8])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    cfg = load_config(None)
    sp = json.load(open(resolve("data/processed/splits_full.json")))
    recs = {r.video_id: r for r in load_dataset(resolve(cfg["paths"]["annotations"]), resolve(cfg["paths"]["raw_dir"]))}
    fdir = resolve(cfg["paths"]["features_dir"])
    vids = [v for v in sp["test"] if cache_path(fdir, v).exists()][: a.n]
    fx = FeatureExtractor(cfg, get_device(), use_cnn=False)
    pipe = SafeDrivePipeline(resolve("models/safedrive_v3"), device="cpu")
    out = {"videos": vids, "per_video": {}, "all_inputs_identical": True}
    before_t, after_t = [], []
    for v in vids:
        old = load_features(cache_path(fdir, v))
        new, _ = fx.extract(recs[v].path, v)
        same = {
            "motion": bool(np.array_equal(old.motion, new.motion)),
            "diff": bool(np.array_equal(old.diff, new.diff)),
            "ego": bool(np.array_equal(old.ego, new.ego)),
            "track_boxes_ids_conf": bool(np.array_equal(old.tracks[:, [0, 1, 3, 4, 5, 6, 7]], new.tracks[:, [0, 1, 3, 4, 5, 6, 7]])),
            "track_class_groups": bool(np.array_equal([class_group(int(c)) for c in old.tracks[:, 2]],
                                                      [class_group(int(c)) for c in new.tracks[:, 2]])),
        }
        ro, rn = pipe.analyze_features(old, profile="balanced"), pipe.analyze_features(new, profile="balanced")
        same["p_mean"] = bool(np.array_equal(ro.timeline.p_mean.to_numpy(), rn.timeline.p_mean.to_numpy()))
        same["alarms"] = list(ro.alarms) == list(rn.alarms)
        same["fine_labels_changed_rows"] = int(np.sum(old.tracks[:, 2] != new.tracks[:, 2]))
        out["per_video"][v] = same
        out["all_inputs_identical"] &= all(val for k, val in same.items() if k != "fine_labels_changed_rows")
        before_t.append(old.tracks)
        after_t.append(new.tracks)
        print(v, same, flush=True)
    out["flips_before_cache"] = flips(np.vstack([t + np.array([0, 1e6 * i, 0, 0, 0, 0, 0, 0]) for i, t in enumerate(before_t)]))
    out["flips_after_stabilizer"] = flips(np.vstack([t + np.array([0, 1e6 * i, 0, 0, 0, 0, 0, 0]) for i, t in enumerate(after_t)]))
    save_json(out, resolve("experiments/latest/v342_class_stabilizer_check.json"))
    print(json.dumps({k: out[k] for k in ("all_inputs_identical", "flips_before_cache", "flips_after_stabilizer")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
