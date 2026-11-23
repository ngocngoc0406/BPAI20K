"""v3.4 real-world data-collection campaign: dataset manifest, group-wise partitions, leakage and duplicate checks, data
validation, legacy-session audit and campaign metrics computed on CONSENSUS labels of two independent annotators.

Nothing here trains or changes a model. Partitions are assigned once per group (driver, else trip, else session) and frozen in
data/campaign/partitions.json; the final_test partition is evaluated only on explicit request and every such access is logged.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.evaluation.streaming import MAX_LEAD
from src.realworld import annotation as ann
from src.realworld import metrics as rwm
from src.realworld import provenance
from src.realworld.trip_meta import SESSIONS_DIR, list_sessions, load_meta
from src.utils.config import resolve

CAMPAIGN_DIR = resolve("data/campaign")
PARTITIONS = {"train": 0.5, "dev": 0.2, "final_test": 0.3}
MIN_EVENTS_FOR_RECALL = 10   # confirmed independent events of a type below which recall is "insufficient"
OFFICIAL_KINDS = ("real_driving",)
PHASH_DUP_BITS = 6           # mean Hamming distance (of 64) at or below which two segments are flagged as near-duplicates


# ---------------- hashing ----------------
def dhash(frame: np.ndarray) -> int:
    g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def video_phash(path: Path, n: int = 8) -> list[int]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    out = []
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) * total / n))
        ok, fr = cap.read()
        if ok:
            out.append(dhash(fr))
    cap.release()
    return out


def phash_distance(a: list[int], b: list[int]) -> float:
    if not a or not b or len(a) != len(b):
        return float("inf")
    return float(np.mean([bin(x ^ y).count("1") for x, y in zip(a, b)]))


# ---------------- partitions ----------------
def group_key(meta) -> str:
    if meta is None:
        return ""
    return f"driver:{meta.driver_id}" if meta.driver_id else (f"trip:{meta.trip_id}" if meta.trip_id else f"session:{meta.session_id}")


def load_partitions(cdir: Path = CAMPAIGN_DIR) -> dict:
    p = Path(cdir) / "partitions.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"salt": "safedrive-v34", "groups": {}}


def assign_partition(group: str, cdir: Path = CAMPAIGN_DIR, manual: str | None = None, reason: str = "") -> str:
    """Assign a group once (deterministic hash of salt+group, or a manual choice with a reason) and freeze it."""
    import hashlib
    parts = load_partitions(cdir)
    if group in parts["groups"]:
        return parts["groups"][group]["partition"]
    if manual:
        if manual not in PARTITIONS or not reason.strip():
            raise ValueError("manual partition needs a valid partition and a reason")
        part = manual
    else:
        u = int(hashlib.sha256((parts["salt"] + group).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        acc, part = 0.0, "final_test"
        for k, w in PARTITIONS.items():
            acc += w
            if u < acc:
                part = k
                break
    parts["groups"][group] = {"partition": part, "assigned": datetime.now().isoformat(timespec="seconds"),
                              "method": "manual" if manual else "hash", "reason": reason}
    Path(cdir).mkdir(parents=True, exist_ok=True)
    (Path(cdir) / "partitions.json").write_text(json.dumps(parts, indent=2), encoding="utf-8")
    return part


# ---------------- legacy audit ----------------
def observed_origin(source: str) -> str:
    s = source.replace("\\", "/").lower()
    if "pytest-of-" in s or s.startswith("slow:"):
        return "automated test (pytest fixture source)"
    if s.startswith("camera:"):
        return "camera source (no record of where / what was filmed)"
    if "/data/raw/nexar/" in s or s.startswith("file:"):
        return "video file replay (recorded dataset clip)"
    return "unknown"


def legacy_audit(root: Path = SESSIONS_DIR) -> list[dict]:
    """Sessions without trip metadata: classification legacy_unknown, with ONLY what was recorded (no metadata assigned)."""
    rows = []
    for sid in list_sessions(root):
        d = Path(root) / sid
        if (d / "trip_meta.json").exists():
            continue
        rep = json.loads((d / "session_report.json").read_text(encoding="utf-8")) if (d / "session_report.json").exists() else {}
        s = rep.get("summary", {})
        rows.append({"session_id": sid, "classification": "legacy_unknown", "recorded_source": s.get("source", ""),
                     "observed_origin": observed_origin(s.get("source", "")), "model": s.get("model"), "profile": s.get("profile"),
                     "duration_s": s.get("duration_s"), "started": s.get("started"), "events": len(rep.get("events", [])),
                     "files": sorted(f.name for f in d.iterdir()), "official_evaluation": False,
                     "decision": "kept unchanged; excluded from every campaign partition and from official metrics"})
    return rows


# ---------------- manifest + validation ----------------
def session_entry(d: Path, with_phash: bool = True) -> dict:
    sid = d.name
    meta = load_meta(sid, d.parent)
    prov = provenance.load_session(d)
    chk = provenance.check(prov) if prov else {"eligible": False, "frozen_reference": False}
    tl = rwm.load_timeline(d)
    segs = []
    seg_info = json.loads((d / "video" / "segments.json").read_text(encoding="utf-8")) if (d / "video" / "segments.json").exists() else {}
    for f in sorted((d / "video").glob("segment_*.mp4")):
        segs.append({"file": f"video/{f.name}", "sha256": provenance.sha256_file(f), "bytes": f.stat().st_size,
                     "phash": video_phash(f) if with_phash else []})
    integ = rwm.frame_integrity(tl) if tl is not None else {}
    frames = seg_info.get("frames_written")
    steps = integ.get("steps_logged")
    issues = []
    if meta is None:
        issues.append("no trip metadata (legacy_unknown)")
    else:
        if meta.data_kind not in OFFICIAL_KINDS:
            issues.append(f"data kind {meta.data_kind} is not real driving")
        if meta.data_kind == "real_driving" and not (meta.driver_consent and meta.supervisor_present and meta.configured_before_driving):
            issues.append("consent / supervisor / setup flags incomplete")
        if not meta.regulations_checked and meta.data_kind == "real_driving":
            issues.append("recording rules not confirmed")
        if not meta.record_full_video:
            issues.append("no full-session video (cannot be fully annotated)")
    if prov is None:
        issues.append("no provenance (recorded before v3.4)")
    elif not chk.get("eligible"):
        issues.append("model / feature / threshold hash differs from the frozen reference" if chk.get("frozen_reference")
                      else "no frozen reference for this model")
    if tl is None:
        issues.append("no timeline")
    elif integ.get("missing_steps", 0) > 0:
        issues.append(f"{integ['missing_steps']} missing 10 Hz steps")
    sync = None
    if frames is not None and steps:
        sync = abs(frames - steps) / steps
        if sync > 0.01:
            issues.append(f"video/timeline desynchronised ({frames} frames vs {steps} steps)")
    if seg_info.get("frames_dropped"):
        issues.append(f"{seg_info['frames_dropped']} video frames dropped by the writer")
    pv = d / "shared" / "privacy_review.json"
    privacy = json.loads(pv.read_text(encoding="utf-8")) if pv.exists() else None
    names = ann.annotators(d)
    reviewed = list(meta.fully_reviewed_by) if meta else []
    group = group_key(meta)
    eligible = meta is not None and meta.data_kind in OFFICIAL_KINDS and not any(
        i for i in issues if not i.startswith("no full-session video"))
    return {"session_id": sid, "trip_id": meta.trip_id if meta else "", "driver_id": meta.driver_id if meta else "",
            "campaign_id": meta.campaign_id if meta else "", "data_kind": meta.data_kind if meta else "legacy_unknown",
            "group": group, "lighting": meta.lighting if meta else "unknown", "weather": meta.weather if meta else "unknown",
            "road_types": meta.road_types if meta else [], "traffic": meta.traffic if meta else "unknown",
            "hours": len(tl) * rwm.STEP_S / 3600 if tl is not None else 0.0, "timeline_steps": steps, "video_frames": frames,
            "sync_rel_diff": sync, "missing_steps": integ.get("missing_steps"), "video_segments": segs,
            "model_sha256": prov.get("model_sha256") if prov else None, "feature_config_sha256": prov.get("feature_config_sha256") if prov else None,
            "threshold_config_sha256": prov.get("threshold_config_sha256") if prov else None,
            "software_sha256": prov.get("software_sha256") if prov else None, "provenance_check": chk,
            "annotators": names, "fully_reviewed_by": reviewed, "double_annotated": len(set(names) & set(reviewed)) >= 2,
            "privacy_shared_copy": bool((d / "shared").exists() and any((d / "shared").glob("*.mp4"))),
            "privacy_review": privacy, "issues": issues, "eligible": bool(eligible)}


def build_manifest(root: Path = SESSIONS_DIR, cdir: Path = CAMPAIGN_DIR, with_phash: bool = True, assign: bool = True) -> dict:
    entries = []
    for sid in list_sessions(root):
        e = session_entry(Path(root) / sid, with_phash)
        if not e["eligible"]:
            e["partition"] = "excluded"
        elif assign:
            e["partition"] = assign_partition(e["group"], cdir)
        else:
            e["partition"] = load_partitions(cdir)["groups"].get(e["group"], {}).get("partition", "unassigned")
        entries.append(e)
    man = {"created": datetime.now().isoformat(timespec="seconds"), "schema": "safedrive-v3.4-manifest",
           "frozen_reference": provenance.frozen(), "sessions": entries, "leakage": leakage_check(entries)}
    Path(cdir).mkdir(parents=True, exist_ok=True)
    (Path(cdir) / "manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
    return man


def leakage_check(entries: list[dict]) -> dict:
    part = [e for e in entries if e.get("partition") in PARTITIONS]
    problems = []
    for key in ("trip_id", "driver_id", "group"):
        seen: dict = {}
        for e in part:
            if e[key]:
                seen.setdefault(e[key], set()).add(e["partition"])
        problems += [f"{key} {k} appears in partitions {sorted(v)}" for k, v in seen.items() if len(v) > 1]
    sha: dict = {}
    for e in entries:
        for s in e["video_segments"]:
            sha.setdefault(s["sha256"], []).append(f"{e['session_id']}/{s['file']}")
    exact = {k: v for k, v in sha.items() if len(v) > 1}
    near = []
    segs = [(e, s) for e in entries for s in e["video_segments"] if s.get("phash")]
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            (e1, s1), (e2, s2) = segs[i], segs[j]
            if e1["session_id"] == e2["session_id"]:
                continue
            dist = phash_distance(s1["phash"], s2["phash"])
            if dist <= PHASH_DUP_BITS:
                near.append({"a": f"{e1['session_id']}/{s1['file']}", "b": f"{e2['session_id']}/{s2['file']}", "mean_hamming": dist,
                             "cross_partition": e1.get("partition") != e2.get("partition")})
    return {"group_leaks": problems, "exact_duplicate_videos": exact, "near_duplicate_segments": near,
            "ok": not problems and not exact and not any(n["cross_partition"] for n in near)}


# ---------------- consensus metrics ----------------
def session_consensus(d: Path, pair: tuple[str, str] | None = None) -> tuple[list[dict], tuple[str, str] | None]:
    meta = load_meta(d.name, d.parent)
    reviewed = [n for n in (meta.fully_reviewed_by if meta else []) if n in ann.annotators(d)]
    if pair is None:
        if len(reviewed) < 2:
            return [], None
        pair = (reviewed[0], reviewed[1])
    return ann.consensus(ann.load(d, pair[0]), ann.load(d, pair[1]), ann.load_disagreements(d)), pair


def session_event_metrics(d: Path) -> dict | None:
    """Alarm classification and per-type hits for one double-annotated, fully reviewed session."""
    cons, pair = session_consensus(d)
    tl = rwm.load_timeline(d)
    if pair is None or tl is None:
        return None
    valid_tl = tl[tl["degraded"].fillna(0).astype(int) == 0]
    hours = len(valid_tl) * rwm.STEP_S / 3600
    on = rwm.alarm_onsets(tl)
    conf_ev = [e for e in cons if e["status"] == "confirmed" and e["category"] in ann.EVENT_CATEGORIES]
    excl = [e for e in cons if e["status"] != "confirmed" or e["category"] == "uncertain"]
    status = []
    for t in on:
        if any(e["critical_t"] is not None and e["critical_t"] - MAX_LEAD <= t < e["critical_t"] for e in conf_ev):
            status.append("valid")
        elif any(e["risk_onset"] - MAX_LEAD <= t <= e["risk_end"] for e in excl):
            status.append("excluded")
        else:
            status.append("false")
    events = []
    for e in conf_ev:
        w = on[(on >= e["critical_t"] - MAX_LEAD) & (on < e["critical_t"])]
        events.append({"category": e["category"], "hit": bool(len(w)), "lead_s": float(e["critical_t"] - w.min()) if len(w) else None,
                       "tags": e["tags"]})
    return {"session_id": d.name, "pair": pair, "hours": hours, "alarms": len(on), "false_alarms": status.count("false"),
            "valid_alarms": status.count("valid"), "excluded_alarms": status.count("excluded"),
            "repeated_warnings": int(np.sum(np.diff(on) < rwm.REPEAT_S)) if len(on) > 1 else 0, "events": events,
            "n_confirmed": len(conf_ev), "n_disputed": sum(e["status"] == "disputed" for e in cons),
            "n_single": sum(e["status"] == "single" for e in cons)}


def campaign_metrics(manifest: dict, partitions=("train", "dev"), root: Path = SESSIONS_DIR, cdir: Path = CAMPAIGN_DIR,
                     include_final_test: bool = False, requested_by: str = "") -> dict:
    """Pooled metrics per partition set, by event type and by declared condition. final_test only on explicit request (logged)."""
    from scipy.stats import chi2
    parts = list(partitions) + (["final_test"] if include_final_test else [])
    if include_final_test:
        Path(cdir).mkdir(parents=True, exist_ok=True)
        with open(Path(cdir) / "final_test_access_log.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), "by": requested_by}) + "\n")

    def wilson(k, n):
        if n == 0:
            return [None, None]
        z, p = 1.96, k / n
        den = 1 + z * z / n
        c, h = (p + z * z / (2 * n)) / den, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return [round(float(max(0, c - h)), 3), round(float(min(1, c + h)), 3)]

    def poisson(k, hrs):
        if hrs <= 0:
            return [None, None]
        lo = chi2.ppf(0.025, 2 * k) / 2 if k else 0.0
        return [round(float(lo / hrs), 2), round(float(chi2.ppf(0.975, 2 * (k + 1)) / 2 / hrs), 2)]

    sess = [e for e in manifest["sessions"] if e["eligible"] and e.get("partition") in parts]
    rows = []
    for e in sess:
        m = session_event_metrics(Path(root) / e["session_id"])
        if m is not None:
            m.update({k: e[k] for k in ("lighting", "weather", "road_types", "traffic", "partition")})
            rows.append(m)

    def summarise(rs: list[dict]) -> dict:
        hrs = sum(r["hours"] for r in rs)
        fa = sum(r["false_alarms"] for r in rs)
        out = {"sessions": len(rs), "hours": round(hrs, 3), "alarms": sum(r["alarms"] for r in rs), "false_alarms": fa,
               "fa_per_hour": round(fa / hrs, 2) if hrs > 0 else None, "fa_per_hour_ci95": poisson(fa, hrs),
               "excluded_alarms": sum(r["excluded_alarms"] for r in rs), "repeated_warnings": sum(r["repeated_warnings"] for r in rs),
               "by_event_type": {}}
        for cat in ["collision", "near_miss", "risk_event", "all_events"]:
            evs = [ev for r in rs for ev in r["events"] if cat == "all_events" or ev["category"] == cat]
            n, k = len(evs), sum(ev["hit"] for ev in evs)
            leads = [ev["lead_s"] for ev in evs if ev["lead_s"] is not None]
            o = {"confirmed_events": n}
            if n < MIN_EVENTS_FOR_RECALL:
                o["recall"] = f"insufficient (n={n} < {MIN_EVENTS_FOR_RECALL} confirmed events)"
            else:
                o.update({"recall": round(k / n, 3), "recall_ci95": wilson(k, n),
                          "warned_ge_1s": round(sum(x >= 1 for x in leads) / n, 3), "warned_ge_1s_ci95": wilson(sum(x >= 1 for x in leads), n),
                          "warned_ge_2s": round(sum(x >= 2 for x in leads) / n, 3), "warned_ge_2s_ci95": wilson(sum(x >= 2 for x in leads), n),
                          "lead_s_median": float(np.median(leads)) if leads else None, "lead_s_all": leads})
            out["by_event_type"][cat] = o
        if len(rs) >= 5 and hrs > 0:  # cluster (session) bootstrap for FA/h: sessions are not independent of their drivers
            rng = np.random.default_rng(34)
            bs = []
            for _ in range(2000):
                idx = rng.integers(0, len(rs), len(rs))
                h = sum(rs[i]["hours"] for i in idx)
                bs.append(sum(rs[i]["false_alarms"] for i in idx) / h if h > 0 else np.nan)
            out["fa_per_hour_session_bootstrap_ci95"] = [round(float(x), 2) for x in np.nanpercentile(bs, [2.5, 97.5])]
        return out

    res = {"partitions": parts, "eligible_sessions": len(sess), "double_annotated_sessions": len(rows), "overall": summarise(rows),
           "by_lighting": {}, "by_weather": {}, "by_road_type": {}}
    for key, field_ in (("by_lighting", "lighting"), ("by_weather", "weather")):
        for g in sorted({r[field_] for r in rows}):
            res[key][g] = summarise([r for r in rows if r[field_] == g])
    for g in sorted({x for r in rows for x in r["road_types"]}):
        res["by_road_type"][g] = summarise([r for r in rows if g in r["road_types"]])
    return res


def agreement_report(manifest: dict, root: Path = SESSIONS_DIR) -> dict:
    per, all_a, all_b = [], [], []
    for e in manifest["sessions"]:
        d = Path(root) / e["session_id"]
        _, pair = session_consensus(d)
        if pair is None:
            continue
        A, B = ann.load(d, pair[0]), ann.load(d, pair[1])
        r = ann.agreement(A, B)
        cons = ann.consensus(A, B, ann.load_disagreements(d))
        per.append({"session_id": e["session_id"], "pair": pair, **{k: r[k] for k in ("matched", "only_first", "only_second",
                    "kappa_category", "kappa_with_unmatched", "kappa_event_vs_not", "offset_start", "offset_event_time", "offset_end")},
                    "confirmed": sum(c["status"] == "confirmed" for c in cons), "disputed": sum(c["status"] == "disputed" for c in cons),
                    "single": sum(c["status"] == "single" for c in cons),
                    "disagreement_reasons": [d_.reason for d_ in ann.load_disagreements(d)]})
        all_a += [(e["session_id"], a) for a in A]
        all_b += [(e["session_id"], b) for b in B]
    pooled = None
    if per:
        pa, pb = [], []
        for s in {x[0] for x in all_a} | {x[0] for x in all_b}:
            pairs, ua, ub = ann.match([a for sid, a in all_a if sid == s], [b for sid, b in all_b if sid == s])
            pa += [p[0].category for p in pairs] + [a.category for a in ua] + ["none"] * len(ub)
            pb += [p[1].category for p in pairs] + ["none"] * len(ua) + [b.category for b in ub]
        ev = lambda c: "event" if c in ann.EVENT_CATEGORIES else "no_event"  # noqa: E731
        pooled = {"situations": len(pa), "kappa_with_unmatched": ann.cohen_kappa(pa, pb),
                  "kappa_event_vs_not": ann.cohen_kappa([ev(c) for c in pa], [ev(c) for c in pb])}
    return {"sessions": per, "pooled": pooled, "note": None if per else "no double-annotated, fully reviewed session yet"}


def validation_report(manifest: dict) -> dict:
    ss = manifest["sessions"]
    real = [e for e in ss if e["data_kind"] == "real_driving"]
    return {"sessions_total": len(ss), "legacy_unknown": sum(e["data_kind"] == "legacy_unknown" for e in ss),
            "real_driving_sessions": len(real), "real_driving_hours": round(sum(e["hours"] for e in real), 3),
            "eligible_sessions": sum(e["eligible"] for e in ss), "eligible_hours": round(sum(e["hours"] for e in ss if e["eligible"]), 3),
            "trips": len({e["trip_id"] for e in real if e["trip_id"]}), "drivers": len({e["driver_id"] for e in real if e["driver_id"]}),
            "sessions_with_missing_steps": sum(bool(e["missing_steps"]) for e in real),
            "sessions_desynchronised": sum((e["sync_rel_diff"] or 0) > 0.01 for e in real),
            "share_missing_or_desync": (sum(bool(e["missing_steps"]) or (e["sync_rel_diff"] or 0) > 0.01 for e in real) / len(real)) if real else None,
            "double_annotated_sessions": sum(e["double_annotated"] for e in real),
            "privacy_reviewed_shared_copies": sum(bool(e["privacy_review"] and e["privacy_review"].get("approved")) for e in real),
            "partitions": {p: {"sessions": sum(e.get("partition") == p for e in ss), "hours": round(sum(e["hours"] for e in ss if e.get("partition") == p), 3)}
                           for p in list(PARTITIONS) + ["excluded"]},
            "leakage_ok": manifest["leakage"]["ok"], "leakage": manifest["leakage"],
            "issues": {e["session_id"]: e["issues"] for e in ss if e["issues"]}}


def campaign_status(manifest: dict | None, root: Path = SESSIONS_DIR, cdir: Path = CAMPAIGN_DIR) -> dict:
    """v3.4.3 dashboard summary (read-only). Real-driving hours come only from sessions declared as real driving."""
    from src.realworld import privacy, readiness, training
    ss = manifest["sessions"] if manifest else []
    real = [e for e in ss if e["data_kind"] == "real_driving"]
    out = {"real_driving_hours": round(sum(e["hours"] for e in real), 3), "eligible_sessions": sum(e["eligible"] for e in ss),
           "excluded_sessions": sum(not e["eligible"] for e in ss),
           "missing_provenance": sum(1 for e in ss if e["data_kind"] != "legacy_unknown" and not e.get("model_sha256"))}
    recs = readiness.backup_records(Path(cdir) / "backup_status.jsonl")
    lb, lv = readiness._latest(recs, "backup"), readiness._latest(recs, "verify")
    if lb is None:
        out["backup"] = {"label": "none", "detail": "no backup yet (needs an external drive)"}
    else:
        good = lb.get("ok") and lb.get("external") and not lb.get("test_mode")
        ver = lv is not None and lv.get("ok") and lv["at"] >= lb["at"]
        out["backup"] = {"label": "verified" if good and ver else ("unverified" if good else "not valid"),
                         "detail": f"last {lb['at']} · {'external' if lb.get('external') else 'NOT external'} · verify {lv and lv.get('ok')}"}
    st_ = {}
    for s in list_sessions(root):
        if (Path(root) / s / "trip_meta.json").exists():
            k = privacy.shared_status(Path(root) / s)
            st_[k] = st_.get(k, 0) + 1
    pol = privacy.sharing_policy()
    out["privacy"] = {"label": "sharing disabled" if not pol.get("sharing_enabled", True) else
                      f"{st_.get('approved', 0)} approved", "detail": ", ".join(f"{k} {v}" for k, v in sorted(st_.items())) or "no shared copies"}
    da = sum(e["double_annotated"] for e in real)
    out["annotation"] = {"label": f"{da} / {len(real)}", "detail": "real-driving sessions double-annotated (two full reviews)"}
    tr = training.report_if_available()
    out["training"] = {"label": "not built" if tr is None else ("PASSED" if tr.get("passed") else ("complete, not passed" if tr.get("complete")
                       else "in progress / not started")), "detail": "practice pack; never evaluation data"}
    parts = {p: sum(e.get("partition") == p for e in ss) for p in PARTITIONS}
    lk = (manifest or {}).get("leakage", {}).get("ok")
    out["split"] = {"label": "leakage OK" if lk else ("no manifest" if manifest is None else "LEAKAGE"),
                    "detail": " / ".join(f"{p} {n}" for p, n in parts.items()) + " sessions"}
    fl = Path(cdir) / "final_test_access_log.jsonl"
    n_acc = len([x for x in fl.read_text(encoding="utf-8").splitlines() if x.strip()]) if fl.exists() else 0
    out["final_test"] = {"label": "locked" if n_acc == 0 else f"opened {n_acc}×", "detail": f"{parts['final_test']} sessions; accesses logged"}
    out["readiness"] = readiness.load_latest(Path(cdir) / "readiness_latest.json")
    return out
