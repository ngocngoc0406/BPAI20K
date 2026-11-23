"""Real-world session metrics computed ONLY from recorded data (timeline.csv, session_report.json, annotations, trip_meta).

Definitions (aligned with the offline benchmark, see docs/SAFEDRIVE_V33_EVALUATION_PROTOCOL.md):
- alarm onset: a 0 -> 1 transition of the logged `alarm` column (non-degraded steps).
- hazard reference time of an annotation: critical_t if given, else risk_end.
- valid warning for a warning-worthy annotation (warning_worthy / potential_near_miss / confirmed_collision):
  an alarm onset in [ref - MAX_LEAD, ref)  (MAX_LEAD = 5 s, as in the benchmark).
- false alarm: an onset that is valid for no warning-worthy annotation, and does not fall in an 'uncertain' annotation
  (uncertain onsets are reported separately and excluded from both FA and recall).
- warning without hazard: an onset that overlaps NO annotated situation at all (any category except 'normal',
  window [risk_onset - MAX_LEAD, risk_end]).
- repeated warning: an onset within REPEAT_S of the previous onset.
- recall is only reported for sessions marked as fully reviewed by an annotator (otherwise unannotated hazards are unknown).
- CAUTION-level "event recall": a warning-worthy annotation for which p >= the CAUTION threshold occurred in the window.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.streaming import MAX_LEAD
from src.realworld import annotation as ann
from src.realworld.trip_meta import load_meta

REPEAT_S = 30.0
STEP_S = 0.1


def load_timeline(session_dir: Path) -> pd.DataFrame | None:
    p = Path(session_dir) / "timeline.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df if len(df) else None


def alarm_onsets(tl: pd.DataFrame) -> np.ndarray:
    a = tl["alarm"].fillna(0).astype(int).to_numpy()
    if "degraded" in tl:
        a = a * (1 - tl["degraded"].fillna(0).astype(int).to_numpy())
    on = np.flatnonzero(np.diff(np.r_[0, a]) == 1)
    return tl["t_stream"].to_numpy()[on]


def caution_times(tl: pd.DataFrame, caution: float | None) -> np.ndarray:
    if caution is None or "p" not in tl:
        lv = tl["level"].astype(str)
        return tl["t_stream"].to_numpy()[lv.isin(["CAUTION", "WARNING", "HIGH RISK"]).to_numpy()]
    p = pd.to_numeric(tl["p"], errors="coerce").to_numpy()
    return tl["t_stream"].to_numpy()[np.nan_to_num(p) >= caution]


def frame_integrity(tl: pd.DataFrame) -> dict:
    """Missing / duplicate 10 Hz steps. A stream reset (step counter restarts after a capture gap > 1 s, see controller)
    starts a new segment and is counted separately, not as a duplicate."""
    t = tl["t_stream"].to_numpy(float)
    st = tl["step"].to_numpy()
    resets = np.flatnonzero((np.diff(st) < 0) | (st[1:] == 0)) + 1  # step counter restarted (engine.reset after a gap)
    missing = dup = expected = 0
    for seg_t in np.split(t, resets):
        if not len(seg_t):
            continue
        d = np.diff(seg_t)
        missing += int(np.sum(np.clip(np.round(d / STEP_S) - 1, 0, None)))
        dup += int(np.sum(d <= 0))
        expected += int(round((seg_t[-1] - seg_t[0]) / STEP_S)) + 1
    return {"steps_logged": int(len(t)), "steps_expected": expected, "missing_steps": missing, "duplicate_steps": dup,
            "stream_resets": int(len(resets)), "completeness": float((len(t) - dup) / expected) if expected else float("nan")}


def runtime(tl: pd.DataFrame) -> dict:
    out = {}
    for c in ("step_ms", "e2e_ms", "capture_fps"):
        v = pd.to_numeric(tl[c], errors="coerce").dropna().to_numpy()
        if len(v):
            out[c] = {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "mean": float(v.mean())}
    dur = float(tl["wall"].iloc[-1] - tl["wall"].iloc[0]) if len(tl) > 1 else 0.0
    out["processed_rate_hz"] = float(len(tl) / dur) if dur > 0 else float("nan")
    out["degraded_share"] = float(tl["degraded"].mean()) if "degraded" in tl else float("nan")
    rss = pd.to_numeric(tl["rss_mb"], errors="coerce")
    ok = rss.notna().to_numpy()
    if ok.sum() >= 3:
        w = tl["wall"].to_numpy()[ok]
        slope = np.polyfit((w - w[0]) / 60.0, rss.to_numpy()[ok], 1)[0]
        out["rss_mb"] = {"first": float(rss[ok].iloc[0]), "last": float(rss[ok].iloc[-1]), "max": float(rss[ok].max()),
                         "slope_mb_per_min": float(slope)}
    return out


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6 if Path(path).exists() else 0.0


def session_metrics(session_dir: Path, annotator: str | None = None, caution: float | None = None) -> dict:
    """All metrics for one session. `annotator`: whose annotations are the reference (default: the first fully-reviewed one)."""
    d = Path(session_dir)
    sid = d.name
    meta = load_meta(sid, d.parent)
    tl = load_timeline(d)
    rep_p = d / "session_report.json"
    rep = json.loads(rep_p.read_text(encoding="utf-8")) if rep_p.exists() else {}
    out: dict = {"session_id": sid, "data_kind": meta.data_kind if meta else "legacy_unknown",
                 "lighting": meta.lighting if meta else "unknown", "weather": meta.weather if meta else "unknown",
                 "road_types": meta.road_types if meta else [], "traffic": meta.traffic if meta else "unknown",
                 "model": rep.get("summary", {}).get("model", ""), "profile": rep.get("summary", {}).get("profile", ""),
                 "has_timeline": tl is not None}
    evs = rep.get("events", [])
    out["recording"] = {"events": len(evs), "clips_saved": sum(e.get("clip_status") == "saved" for e in evs),
                        "clip_success": (sum(e.get("clip_status") == "saved" for e in evs) / len(evs)) if evs else float("nan"),
                        "clip_completeness": float(np.mean([e.get("clip_frames", 0) / max(1, round((e.get("pre_s", 5) + e.get("post_s", 5)) * 10))
                                                            for e in evs if e.get("clip_status") == "saved"]))
                        if any(e.get("clip_status") == "saved" for e in evs) else float("nan")}
    if tl is None:
        out["note"] = "no per-step timeline (session recorded before v3.3): only report-level counts available"
        out["hours"] = rep.get("summary", {}).get("duration_s", 0) / 3600 if rep else 0.0
        return out
    valid = tl[tl["degraded"].fillna(0).astype(int) == 0] if "degraded" in tl else tl
    hours = len(valid) * STEP_S / 3600
    out["hours"] = hours
    out["hours_wall"] = float(tl["wall"].iloc[-1] - tl["wall"].iloc[0]) / 3600 if len(tl) > 1 else 0.0
    out["integrity"] = frame_integrity(tl)
    out["runtime"] = runtime(tl)
    clip_mb = sum(Path(e["clip"]).stat().st_size for e in evs if e.get("clip") and Path(e["clip"]).exists()) / 1e6
    out["recording"]["storage_mb"] = dir_size_mb(d) + clip_mb
    out["recording"]["storage_mb_per_hour"] = out["recording"]["storage_mb"] / out["hours_wall"] if out["hours_wall"] > 0 else float("nan")
    seg = d / "video" / "segments.json"
    if seg.exists():
        s = json.loads(seg.read_text(encoding="utf-8"))
        out["recording"]["full_video"] = {"frames_written": s["frames_written"], "frames_dropped": s["frames_dropped"],
                                          "segments": len(s["segments"]), "errors": len(s["errors"])}
    on = alarm_onsets(tl)
    out["alarms"] = len(on)
    out["alarms_per_hour"] = len(on) / hours if hours > 0 else float("nan")
    out["repeated_warnings"] = int(np.sum(np.diff(on) < REPEAT_S)) if len(on) > 1 else 0

    reviewers = (meta.fully_reviewed_by if meta else []) or []
    ref = annotator or (reviewers[0] if reviewers else None)
    anns = ann.load(d, ref) if ref else []
    out["reference_annotator"] = ref
    out["fully_reviewed"] = bool(ref and ref in reviewers)
    out["annotations"] = {c: sum(a.category == c for a in anns) for c in ann.CATEGORIES}
    if not anns and not out["fully_reviewed"]:
        out["fa_per_hour"] = None
        out["note"] = "not annotated: false alarms / recall cannot be computed (alarms are not assumed false or true)"
        return out
    ww = [a for a in anns if a.category in ann.WARNING_WORTHY]
    unc = [a for a in anns if a.category == "uncertain"]
    situ = [a for a in anns if a.category != "normal"]
    full = out["fully_reviewed"]
    covered = [a for a in anns if a.category not in ann.WARNING_WORTHY and a.category != "uncertain"]  # normal / potential_risk
    status = []
    for t in on:
        if any(a.critical - MAX_LEAD <= t < a.critical for a in ww):
            status.append("valid")
        elif any(a.risk_onset - MAX_LEAD <= t <= a.risk_end for a in unc):
            status.append("uncertain")
        elif full or any(a.risk_onset - MAX_LEAD <= t <= a.risk_end for a in covered + ww):
            status.append("false")
        else:
            status.append("unreviewed")  # partial (clip-only) review: an alarm nobody looked at is neither true nor false
    status = np.array(status, dtype=object)
    no_hazard = [not any(a.risk_onset - MAX_LEAD <= t <= a.risk_end for a in situ) for t in on]
    out["alarm_status"] = {k: int((status == k).sum()) for k in ("valid", "false", "uncertain")}
    if (status == "unreviewed").any():
        out["alarm_status"]["unreviewed"] = int((status == "unreviewed").sum())
    # FA/h is only a rate when every alarm was reviewed
    out["fa_per_hour"] = float((status == "false").sum() / hours) if hours > 0 and not (status == "unreviewed").any() else None
    out["warnings_without_hazard"] = int(np.sum(no_hazard))
    if out["fully_reviewed"]:
        ct = caution_times(tl, caution)
        leads = []
        hit = 0
        ev_hit = 0
        for a in ww:
            w = on[(on >= a.critical - MAX_LEAD) & (on < a.critical)]
            if len(w):
                hit += 1
                leads.append(float(a.critical - w.min()))
            if np.any((ct >= a.critical - MAX_LEAD) & (ct < a.critical)):
                ev_hit += 1
        n = len(ww)
        out["warning_worthy_events"] = n
        out["warning_recall"] = hit / n if n else None
        out["caution_event_recall"] = ev_hit / n if n else None
        out["lead_times_s"] = leads
        out["warned_ge"] = {f"{x}s": (sum(lt >= x for lt in leads) / n if n else None) for x in (1, 2, 3)}
    else:
        out["warning_recall"] = None
        out["note"] = "recall needs a FULL review of the session (mark it as fully reviewed on the annotation page)"
    return out


def aggregate(rows: list[dict], key: str = "data_kind") -> pd.DataFrame:
    """Aggregate per group (never across data kinds): pooled hours, alarms, false alarms and recall over annotated events."""
    recs = []
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for g, sub in df.groupby(key):
        hrs = float(sub["hours"].sum())
        fa = sub.dropna(subset=["fa_per_hour"]) if "fa_per_hour" in sub else sub.iloc[0:0]
        fa_n = sum(r["alarm_status"]["false"] for _, r in fa.iterrows()) if len(fa) else None
        fa_h = float(fa["hours"].sum()) if len(fa) else 0.0
        rev = sub[sub.get("fully_reviewed", False) == True] if "fully_reviewed" in sub else sub.iloc[0:0]  # noqa: E712
        n_ev = int(rev["warning_worthy_events"].fillna(0).sum()) if "warning_worthy_events" in rev else 0
        n_hit = sum(len(x) for x in rev["lead_times_s"]) if "lead_times_s" in rev else 0
        recs.append({key: g, "sessions": len(sub), "hours": hrs, "alarms": int(sub.get("alarms", pd.Series(dtype=float)).fillna(0).sum()),
                     "annotated_hours": fa_h, "false_alarms": fa_n, "fa_per_hour": (fa_n / fa_h) if fa_n is not None and fa_h > 0 else None,
                     "reviewed_sessions": len(rev), "warning_worthy_events": n_ev,
                     "warning_recall": (n_hit / n_ev) if n_ev else None})
    return pd.DataFrame(recs)
