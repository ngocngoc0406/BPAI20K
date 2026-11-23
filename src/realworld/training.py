"""v3.4.3 annotator training: a TRAINING PACK of recorded demo clips (Nexar train split, pre-anonymised) and degraded variants,
kept OUTSIDE data/sessions (data/annotator_training/sessions/) so it can never enter the campaign manifest, partitions,
agreement report or metrics. Results on this pack measure annotator readiness only; they are never evaluation results.

The pack is one session with one video. A hidden key (training_key.json; never shown on the annotator view) lists each clip's
window, the accepted categories and, for event clips, the reference event time (Nexar time_of_event; Nexar does not separate
collisions from near misses, so collision / near_miss / risk_event are all accepted and the difference is discussed with the
coordinator). Degraded clips (dark + blurred, or heavily occluded) practise 'uncertain'.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from src.realworld import annotation as ann
from src.utils.config import resolve

TRAINING_ROOT = resolve("data/annotator_training")
TRAINING_SESSIONS = TRAINING_ROOT / "sessions"
PACK_ID = "TRAINING_PACK_01"
REPORT_PATH = TRAINING_ROOT / "training_report.json"
SUBMISSIONS = TRAINING_ROOT / "submissions.jsonl"   # v3.4.4: one locked submission per annotator (labels SHA-256)
RESULTS_DIR = TRAINING_ROOT / "results"             # v3.4.4: one result file per annotator
MARK = "ANNOTATOR TRAINING DATA - NOT REAL DRIVING - NEVER EVALUATION DATA"
CRITERIA = {
    "fully_reviewed": True,
    "event_detection_min": 0.80,          # share of reference event clips with an event label inside the clip window
    "timing_median_abs_max_s": 1.0,       # median |labelled event time - reference event time|
    "false_events_on_normal_max": 1,      # event labels inside clips whose key is 'normal'
    "categories_practised": sorted(ann.CATEGORIES),  # every category used at least once in the pack
    "evidence_complete": True,            # every event label has evidence text and evidence frames (enforced at entry)
    "pair_kappa_event_vs_not_min": 0.60,  # between the two annotators
}


def build_pack(n_event: int = 4, n_normal: int = 3, n_uncertain: int = 2, seed: int = 7, out: Path | None = None,
               fps: float = 10.0, size=(640, 360)) -> dict:
    """Build the pack from the Nexar TRAIN split (never held-out). Deterministic for a given seed."""
    import cv2
    import imageio.v2 as imageio

    from src.data.dataset_loader import load_dataset
    from src.preprocessing.video_transform import make_transform
    from src.realworld.trip_meta import TripMeta, save_meta
    from src.utils.config import load_config
    cfg = load_config(None)
    recs = load_dataset(resolve(cfg["paths"]["annotations"]), resolve(cfg["paths"]["raw_dir"]))
    split = json.loads(resolve("data/processed/splits_full.json").read_text(encoding="utf-8"))
    train = set(split["train"])
    rng = np.random.default_rng(seed)
    pos = [r for r in recs if r.video_id in train and r.label == 1 and r.time_of_event is not None and r.time_of_event == r.time_of_event and r.time_of_event > 9]
    neg = [r for r in recs if r.video_id in train and r.label == 0]
    pos = [pos[i] for i in rng.permutation(len(pos))[: n_event + n_uncertain]]
    neg = [neg[i] for i in rng.permutation(len(neg))[:n_normal]]
    plan = [("event", r) for r in pos[:n_event]] + [("normal", r) for r in neg] + [("uncertain", r) for r in pos[n_event:]]
    plan = [plan[i] for i in rng.permutation(len(plan))]
    out = Path(out) if out else TRAINING_SESSIONS / PACK_ID
    if out.exists():
        raise FileExistsError(f"{out} exists; the pack is built once (delete it by hand to rebuild)")
    (out / "video").mkdir(parents=True)
    path = out / "video" / "segment_000.mp4"
    wr = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=6, macro_block_size=8, ffmpeg_log_level="error")
    t, key = 0.0, []
    sep = np.zeros((size[1], size[0], 3), np.uint8)
    for ci, (kind, r) in enumerate(plan):
        cap = cv2.VideoCapture(str(r.path))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if kind == "normal":
            t0, t1 = 5.0, 15.0
        else:
            t0, t1 = float(r.time_of_event) - 8.0, float(r.time_of_event) + 2.0
        tf = make_transform("low_light", 2, seed + ci) if kind == "uncertain" and ci % 2 == 0 else \
            (make_transform("occlusion", 2, seed + ci) if kind == "uncertain" else None)
        blur = make_transform("motion_blur", 2, seed + ci) if kind == "uncertain" and ci % 2 == 0 else None
        cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
        start = t
        n_target = int(round((t1 - t0) * fps))
        k, step = 0, src_fps / fps
        idx_next, fi = 0.0, 0
        while k < n_target:
            ok, fr = cap.read()
            if not ok:
                break
            if fi >= idx_next:
                fr = cv2.resize(fr, size, interpolation=cv2.INTER_AREA)
                if tf is not None:
                    fr = tf(fr)
                if blur is not None:
                    fr = blur(fr)
                wr.append_data(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                k += 1
                t += 1.0 / fps
                idx_next += step
            fi += 1
        cap.release()
        entry = {"clip": ci, "kind": kind, "source": f"nexar:{r.video_id}", "t_start": round(start, 2), "t_end": round(t, 2)}
        if kind == "event":
            entry.update(accepted=["collision", "near_miss", "risk_event"], reference_event_time=round(start + 8.0, 2))
        elif kind == "normal":
            entry.update(accepted=["normal"], reference_event_time=None)
        else:
            entry.update(accepted=["uncertain", "collision", "near_miss", "risk_event"], reference_event_time=round(start + 8.0, 2),
                         degradation="low_light sev2 + motion_blur sev2" if blur is not None else "occlusion sev2",
                         note="evidence deliberately degraded; 'uncertain' is the expected answer unless the event is clearly visible")
        key.append(entry)
        for _ in range(int(fps)):  # 1 s black separator between clips
            wr.append_data(sep)
            t += 1.0 / fps
    wr.close()
    (out / "video" / "segments.json").write_text(json.dumps({"segments": [{"file": str(path), "t_start": 0.0, "t_end": round(t, 2),
                                                                          "frames": int(round(t * fps))}], "frames_dropped": 0,
                                                            "training": True}, indent=2), encoding="utf-8")
    save_meta(TripMeta(PACK_ID if out.name == PACK_ID else out.name, data_kind="demo_video", notes=MARK, record_full_video=False,
                       blur_people=False), out.parent)
    (out / "training_key.json").write_text(json.dumps({"pack": out.name, "mark": MARK, "created": datetime.now().isoformat(timespec="seconds"),
                                                       "source_split": "nexar train (never held-out)", "clips": key}, indent=2),
                                           encoding="utf-8")
    (out / "TRAINING_DATA_NOT_EVALUATION.txt").write_text(MARK + "\n", encoding="utf-8")
    return {"pack": str(out), "clips": len(key), "duration_s": round(t, 1)}


class TrainingError(RuntimeError):
    pass


def labels_sha256(pack_dir: Path, name: str) -> str:
    from src.realworld.provenance import sha256_file
    p = ann.ann_path(Path(pack_dir), name)
    return sha256_file(p) if p.exists() else ""


def submissions(pack: str = PACK_ID, path: Path = SUBMISSIONS) -> dict:
    """annotator -> FIRST submission record for this pack (a submission cannot be replaced)."""
    out = {}
    p = Path(path)
    for line in (p.read_text(encoding="utf-8").splitlines() if p.exists() else []):
        if line.strip():
            r = json.loads(line)
            if r.get("pack") == pack and r.get("annotator") not in out:
                out[r["annotator"]] = r
    return out


def submit(pack_dir: Path, name: str, path: Path = SUBMISSIONS, access_path: Path | None = None) -> dict:
    """The annotator declares the pack finished. Labels are locked by their SHA-256: any later change invalidates the result.
    Only people on the raw-access list (the campaign annotators) can submit; each annotator submits once."""
    from src.realworld import privacy
    from src.realworld.trip_meta import load_meta, save_meta
    pack_dir = Path(pack_dir)
    if not privacy.can_view_raw(name, access_path or privacy.ACCESS_PATH):
        raise TrainingError("only annotators on the raw-access list (data/campaign/access.json) take the training")
    if name in submissions(pack_dir.name, path):
        raise TrainingError(f"{name} has already submitted {pack_dir.name}; a submission cannot be replaced")
    labels = ann.load(pack_dir, name)
    if not labels:
        raise TrainingError("no labels to submit")
    rec = {"pack": pack_dir.name, "annotator": name, "at": datetime.now().isoformat(timespec="seconds"),
           "labels": len(labels), "labels_sha256": labels_sha256(pack_dir, name)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    meta = load_meta(pack_dir.name, pack_dir.parent)
    if meta and name not in meta.fully_reviewed_by:
        meta.fully_reviewed_by.append(name)
        save_meta(meta, pack_dir.parent)
    return rec


def is_locked(pack_dir: Path, name: str, path: Path = SUBMISSIONS) -> bool:
    return name in submissions(Path(pack_dir).name, path)


def key_visible(pack_dir: Path, names: list[str], path: Path = SUBMISSIONS) -> bool:
    """The hidden key / results are shown only after BOTH annotators have submitted (independence)."""
    subs = submissions(Path(pack_dir).name, path)
    return len(names) >= 2 and all(n in subs for n in names)


def load_key(pack_dir: Path) -> dict:
    return json.loads((Path(pack_dir) / "training_key.json").read_text(encoding="utf-8"))


def _inside(a: ann.Annotation, c: dict) -> bool:
    return c["t_start"] - 0.5 <= a.critical <= c["t_end"] + 0.5


def report(pack_dir: Path, names: list[str] | None = None, criteria: dict = CRITERIA, submissions_path: Path | None = None) -> dict:
    """submissions_path given (campaign use): 'complete' requires a locked submission by each annotator whose labels are unchanged
    since submission. Without it (unit tests of the scoring only) 'fully reviewed' in the pack metadata is used."""
    from src.realworld.trip_meta import load_meta
    pack_dir = Path(pack_dir)
    key = load_key(pack_dir)
    meta = load_meta(pack_dir.name, pack_dir.parent)
    names = names or ann.annotators(pack_dir)
    per = {}
    for n in names:
        A = ann.load(pack_dir, n)
        ev_clips = [c for c in key["clips"] if c["kind"] == "event"]
        hits, offs = 0, []
        for c in ev_clips:
            xs = [a for a in A if a.category in ann.EVENT_CATEGORIES and _inside(a, c)]
            if xs:
                hits += 1
                offs.append(min((a.critical - c["reference_event_time"] for a in xs), key=abs))
        false_ev = sum(1 for c in key["clips"] if c["kind"] == "normal" for a in A if a.category in ann.EVENT_CATEGORIES and _inside(a, c))
        unc = [c for c in key["clips"] if c["kind"] == "uncertain"]
        unc_used = sum(any(a.category == "uncertain" and _inside(a, c) for a in A) for c in unc)
        used = sorted({a.category for a in A})
        # discrimination by clip kind: which categories this annotator used inside each kind of clip
        disc = {}
        for kind in ("event", "normal", "uncertain"):
            cs = [c for c in key["clips"] if c["kind"] == kind]
            cnt: dict[str, int] = {}
            for c in cs:
                for a in A:
                    if _inside(a, c):
                        cnt[a.category] = cnt.get(a.category, 0) + 1
            good = sum(any(_inside(a, c) and a.category in c["accepted"] for a in A)
                       and not (kind == "normal" and any(_inside(a, c) and a.category in ann.EVENT_CATEGORIES for a in A)) for c in cs)
            disc[kind] = {"clips": len(cs), "clips_with_accepted_label": good, "categories_used": cnt}
        evid = all(len(a.evidence.strip()) >= 5 and a.evidence_frames for a in A if a.category in ann.EVENT_CATEGORIES)
        r = {"labels": len(A), "fully_reviewed": bool(meta and n in meta.fully_reviewed_by), "categories_used": used,
             "event_detection": round(hits / len(ev_clips), 3) if ev_clips else None,
             "timing_offsets_s": [round(o, 2) for o in offs],
             "timing_median_abs_s": round(float(np.median(np.abs(offs))), 2) if offs else None,
             "false_events_on_normal": false_ev, "uncertain_used_on_degraded": f"{unc_used}/{len(unc)}", "evidence_complete": evid,
             "discrimination_by_clip_kind": disc,
             "collision_vs_near_miss_vs_risk_event": "not scorable on this pack (Nexar does not separate them); discussed at debrief"}
        r["criteria"] = {
            "fully_reviewed": r["fully_reviewed"],
            "event_detection": r["event_detection"] is not None and r["event_detection"] >= criteria["event_detection_min"],
            "timing": r["timing_median_abs_s"] is not None and r["timing_median_abs_s"] <= criteria["timing_median_abs_max_s"],
            "false_events": false_ev <= criteria["false_events_on_normal_max"],
            "categories_practised": set(criteria["categories_practised"]) <= set(used),
            "evidence_complete": evid,
        }
        per[n] = r
    pair = None
    if len(names) >= 2:
        g = ann.agreement(ann.load(pack_dir, names[0]), ann.load(pack_dir, names[1]))
        pair = {"annotators": names[:2], **{k: g[k] for k in ("matched", "only_first", "only_second", "kappa_category",
                                                               "kappa_with_unmatched", "kappa_event_vs_not", "offset_event_time",
                                                               "offset_start", "offset_end")}}
        kv = g["kappa_event_vs_not"]
        pair["criterion_kappa"] = bool(kv == kv and kv >= criteria["pair_kappa_event_vs_not_min"])
    invalid: dict[str, str] = {}
    if submissions_path is not None:
        subs = submissions(pack_dir.name, submissions_path)
        for n in names:
            if n not in subs:
                invalid[n] = "not submitted"
            elif subs[n]["labels_sha256"] != labels_sha256(pack_dir, n):
                invalid[n] = "labels changed after submission"
            per[n]["submitted_at"] = subs.get(n, {}).get("at")
        complete = len(names) >= 2 and not invalid
    else:
        complete = len(names) >= 2 and all(per[n]["fully_reviewed"] for n in names)
    crit = {n: per[n]["criteria"] for n in names}
    passed = bool(complete and pair and pair["criterion_kappa"] and all(all(v.values()) for v in crit.values()))
    return {"pack": pack_dir.name, "mark": MARK, "at": datetime.now().isoformat(timespec="seconds"), "annotators": names,
            "complete": complete, "passed": passed, "per_annotator": per, "pair": pair, "criteria": criteria,
            "submitted": sorted(set(names) - set(invalid)) if submissions_path is not None else None, "invalid": invalid,
            "criteria_results": {"per_annotator": crit, "pair_kappa": pair and pair["criterion_kappa"]}}


def report_if_available(root: Path = TRAINING_SESSIONS, submissions_path: Path = SUBMISSIONS,
                        access_path: Path | None = None) -> dict | None:
    """Readiness view: every raw-access annotator must have submitted. Before that only the submission state is returned
    (no score, so nothing about the key leaks to an annotator who has not finished)."""
    from src.realworld import privacy
    d = Path(root) / PACK_ID
    if not (d / "training_key.json").exists():
        return None
    names = list(privacy.load_access(access_path or privacy.ACCESS_PATH).get("raw_access", []))
    subs = submissions(PACK_ID, submissions_path)
    if not key_visible(d, names, submissions_path):
        return {"pack": PACK_ID, "annotators": names, "submitted": sorted(n for n in names if n in subs), "complete": False,
                "passed": False, "criteria_results": None, "invalid": {n: "not submitted" for n in names if n not in subs}}
    return report(d, names, submissions_path=submissions_path)


def save_report(rep: dict, path: Path = REPORT_PATH, results_dir: Path | None = None) -> None:
    """Pair report + one file per annotator (results/<annotator>.json)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    rd = Path(results_dir) if results_dir else Path(path).parent / "results"
    rd.mkdir(parents=True, exist_ok=True)
    for n, r in (rep.get("per_annotator") or {}).items():
        (rd / f"{n}.json").write_text(json.dumps({"pack": rep["pack"], "mark": MARK, "annotator": n, "at": rep["at"],
                                                  "passed_individual": all(r["criteria"].values()),
                                                  "invalid": (rep.get("invalid") or {}).get(n), **r},
                                                 indent=2, ensure_ascii=False, default=str), encoding="utf-8")


__all__ = ["build_pack", "report", "report_if_available", "load_key", "save_report", "submit", "submissions", "is_locked",
           "key_visible", "TrainingError", "asdict"]
