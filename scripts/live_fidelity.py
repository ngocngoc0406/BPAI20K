"""Real-time fidelity: does the live assistant (real-time playback, newest-frame sampling, wall-clock jitter) reproduce the
offline (evaluated) predictions? For N held-out videos, play each in real time through the LiveController and compare with
the offline pipeline on the same file: valid-alarm agreement (alarm in [event - 5 s, event)), alarm counts on normal videos,
and |P_live - P_offline| at matching grid times. Output: experiments/live/fidelity.json
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.data.dataset_loader import index_videos, load_dataset  # noqa: E402
from src.live.audio import AudioConfig  # noqa: E402
from src.live.controller import LiveController  # noqa: E402
from src.live.sources import VideoFileSource  # noqa: E402
from src.pipeline import SafeDrivePipeline  # noqa: E402
from src.utils.config import resolve  # noqa: E402
from src.utils.io import load_json, save_json  # noqa: E402


def main() -> int:
    n_pos, n_neg = int(sys.argv[1]) if len(sys.argv) > 1 else 6, int(sys.argv[2]) if len(sys.argv) > 2 else 4
    profile = sys.argv[3] if len(sys.argv) > 3 else "balanced"
    sp = load_json(resolve("data/processed/splits_full.json"))
    recs = {r.video_id: r for r in load_dataset(resolve("data/annotations/nexar_train.csv"), resolve("data/raw/nexar"))}
    held = sp["test"] + sp["test_ext"]
    vids = index_videos(resolve("data/raw/nexar"))
    off_pipe = SafeDrivePipeline(resolve("models/safedrive_v3"))
    # half of the positives: videos where the offline model raises a valid alarm (from cached features), so that
    # alarm reproduction can be checked; the other half and the negatives are taken in split order
    from src.preprocessing.feature_cache import cache_path, load_features
    hit, miss = [], []
    for v in [v for v in held if recs[v].label == 1]:
        if len(hit) >= n_pos // 2 and len(miss) >= n_pos - n_pos // 2:
            break  # enough candidates: stop scanning early
        a = off_pipe.analyze_features(load_features(cache_path(resolve("data/processed/features"), v)), v, [], profile=profile).alarms
        (hit if any(recs[v].time_of_event - 5 <= x < recs[v].time_of_event for x in a) else miss).append(v)
    pos = hit[: n_pos // 2] + miss[: n_pos - min(len(hit), n_pos // 2)]
    neg = [v for v in held if recs[v].label == 0][:n_neg]
    rows = []
    for v in pos + neg:
        r = recs[v]
        off = off_pipe.analyze(vids[v], keep_frames=False, profile=profile)
        ctl = LiveController()
        ctl.trace = []
        ctl.start(VideoFileSource(str(vids[v]), realtime=True, loop=False), resolve("models/safedrive_v3"), profile,
                  audio_backend="none", audio_cfg=AudioConfig(enabled=False), record_events=False)
        while ctl.running:
            time.sleep(0.5)
        ctl.stop()
        tr = np.array([(t, p, a) for t, p, a, _ in ctl.trace]) if ctl.trace else np.zeros((0, 3))
        tl = off.timeline
        dp = [abs(p - float(np.interp(t, tl.time_s, tl.p_mean))) for t, p, _ in tr]
        live_alarms = [float(t) for t, _, a in tr if a]
        valid = lambda al: any(r.time_of_event - 5 <= x < r.time_of_event for x in al) if r.label == 1 else None  # noqa: E731
        rows.append({"video": v, "label": r.label, "live_alarms": live_alarms, "offline_alarms": off.alarms,
                     "live_valid": valid(live_alarms), "offline_valid": valid(off.alarms),
                     "mean_abs_dp": float(np.mean(dp)) if dp else None, "max_abs_dp": float(np.max(dp)) if dp else None,
                     "live_steps": len(tr), "offline_steps": len(tl)})
        print("ROW", rows[-1], flush=True)
    posr = [x for x in rows if x["label"] == 1]
    out = {"profile": profile, "rows": rows,
           "valid_alarm_agreement": float(np.mean([x["live_valid"] == x["offline_valid"] for x in posr])) if posr else None,
           "live_valid_alarms": sum(bool(x["live_valid"]) for x in posr), "offline_valid_alarms": sum(bool(x["offline_valid"]) for x in posr),
           "normal_video_alarms_live": sum(len(x["live_alarms"]) for x in rows if x["label"] == 0),
           "normal_video_alarms_offline": sum(len(x["offline_alarms"]) for x in rows if x["label"] == 0),
           "mean_abs_dp": float(np.mean([x["mean_abs_dp"] for x in rows if x["mean_abs_dp"] is not None]))}
    save_json(out, resolve("experiments/live/fidelity.json"))
    print("FIDELITY", {k: v for k, v in out.items() if k != "rows"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
