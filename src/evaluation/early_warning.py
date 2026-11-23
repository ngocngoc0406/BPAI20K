"""Early-warning (video-level, streaming) evaluation.

Definitions
-----------
alarm onset     first step of a run of >= k consecutive steps with score >= thr; after an onset the
                alarm re-arms only once the score has been below thr for k consecutive steps.
warm-up         no onset is possible before one full observation window (t < W) has been seen.
timely          a positive video has at least one onset within [time_of_event - 3 s, time_of_event).
time-to-alert   for a positive video, the first onset strictly before time_of_event, t_a.
TTA             time_of_event - t_a (seconds of warning; larger = earlier). Undefined if no onset.
detection rate  fraction of positive videos with an onset before the event.
FA/hour         onsets on negative (normal-driving) videos divided by their total duration in hours.
"""
from __future__ import annotations

import numpy as np


def alarm_onsets(times: np.ndarray, score: np.ndarray, thr: float, k: int = 3) -> list[float]:
    onsets, run, below, armed = [], 0, k, True
    for t, s in zip(times, score):
        if s >= thr:
            run += 1
            below = 0
            if armed and run >= k:
                onsets.append(float(t))
                armed = False
        else:
            run = 0
            below += 1
            if below >= k:
                armed = True
    return onsets


def evaluate_early_warning(timelines: dict[str, dict], recs: dict, thr: float, k: int = 3, warmup_s: float = 0.0,
                           timely_s: float = 3.0) -> dict:
    """timelines: video_id -> {"times": array, "score": array}; recs: video_id -> VideoRecord.
    warmup_s: no alarm can be raised before a full observation window is available (t < warmup_s).
    timely: at least one onset within [event - timely_s, event)."""
    tta, rel_alert, detected, n_pos, timely = [], [], 0, 0, 0
    fa, neg_hours, pos_early_fa = 0, 0.0, 0
    per_video = []
    for vid, tl in timelines.items():
        r = recs[vid]
        times, score = np.asarray(tl["times"]), np.asarray(tl["score"])
        keep = times >= warmup_s - 1e-6
        times, score = times[keep], score[keep]
        if len(times) == 0:
            continue
        if r.label == 1 and r.time_of_event is not None:
            n_pos += 1
            m = times < r.time_of_event
            ons = alarm_onsets(times[m], score[m], thr, k)
            is_timely = any(o >= r.time_of_event - timely_s for o in ons)
            timely += int(is_timely)
            row = {"video_id": vid, "label": 1, "time_of_event": r.time_of_event, "time_of_alert": r.time_of_alert,
                   "first_alarm": ons[0] if ons else None, "n_onsets": len(ons), "timely": is_timely}
            if ons:
                detected += 1
                tta.append(r.time_of_event - ons[0])
                if r.time_of_alert is not None:
                    rel_alert.append(ons[0] - r.time_of_alert)
                    pos_early_fa += sum(o < r.time_of_alert - 3.0 for o in ons)
            per_video.append(row)
        elif r.label == 0:
            ons = alarm_onsets(times, score, thr, k)
            fa += len(ons)
            dur = float(times[-1] - times[0]) if len(times) > 1 else 0.0
            neg_hours += dur / 3600.0
            per_video.append({"video_id": vid, "label": 0, "n_onsets": len(ons), "duration_s": dur})
    return _summary(thr, k, n_pos, detected, timely, timely_s, warmup_s, tta, rel_alert, pos_early_fa, neg_hours, fa,
                    per_video)


def _summary(thr, k, n_pos, detected, timely, timely_s, warmup_s, tta, rel_alert, pos_early_fa, neg_hours, fa, per_video):
    tta_a = np.array(tta)
    rel = np.array(rel_alert)
    return {
        "threshold": thr, "consecutive_steps": k,
        "n_positive_videos": n_pos, "detection_rate": detected / n_pos if n_pos else float("nan"),
        "timely_detection_rate": timely / n_pos if n_pos else float("nan"), "timely_window_s": timely_s,
        "warmup_s": warmup_s,
        "mean_tta_s": float(tta_a.mean()) if len(tta_a) else float("nan"),
        "median_tta_s": float(np.median(tta_a)) if len(tta_a) else float("nan"),
        "frac_alarm_after_annotated_alert": float((rel >= 0).mean()) if len(rel) else float("nan"),
        "median_alarm_minus_annotated_alert_s": float(np.median(rel)) if len(rel) else float("nan"),
        "early_onsets_pos_videos_gt3s_before_alert": int(pos_early_fa),
        "negative_hours": neg_hours, "false_alarms": fa,
        "false_alarms_per_hour": fa / neg_hours if neg_hours > 0 else float("nan"),
        "per_video": per_video,
    }
