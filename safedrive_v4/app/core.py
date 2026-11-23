"""SafeDrive AI v4 application core (used by the single Streamlit app and by tests).

Trips: one analysed video or one live session = one trip. Stored next to the event store (inside the encrypted session volume):
  <store>/trips/<trip_id>/meta.json        source, data kind, model, created, duration, latency
  <store>/trips/<trip_id>/timeline.json    per-step risk / level / reliability / KG evidence
  <store>/trips/<trip_id>/traces/*.json    sealed evidence traces of warning onsets
  <store>/trips/<trip_id>/tracks.npy       tracker output (for scene-graph replay and event track histories)
Data kinds: 'real_driving' only for a consented campaign recording; Nexar / demo videos are 'demo_video' and never count as
real-world driving.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

from safedrive_v4.event_memory.store import EventStore, events_from_timeline
from safedrive_v4.event_memory.trace import EvidenceTrace

DISCLAIMER = ("SafeDrive AI is a research prototype that helps the driver recognise and review risk. It does not brake, steer or "
              "control the vehicle, is not a certified ADAS and cannot prevent accidents. Events are model outputs on image-plane "
              "cues and are not verified incidents unless a reviewer confirmed them. Distances, speeds and TTC are image-based "
              "estimates, not physical measurements.")


class TripStore:
    def __init__(self, events: EventStore):
        self.events = events
        self.root = events.root / "trips"
        self.root.mkdir(parents=True, exist_ok=True)

    def new_id(self, name: str) -> str:
        safe = "".join(c if c.isalnum() else "_" for c in name)[:40]
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe}_{uuid.uuid4().hex[:4]}"

    def save(self, trip_id: str, meta: dict, timeline: list, traces: list[EvidenceTrace], tracks: np.ndarray | None) -> Path:
        d = self.root / trip_id
        (d / "traces").mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(meta, indent=1, default=float), encoding="utf-8")
        (d / "timeline.json").write_text(json.dumps(timeline, default=float), encoding="utf-8")
        for tr in traces:
            tr.save(d / "traces")
        if tracks is not None:
            np.save(d / "tracks.npy", tracks)
        return d

    def list(self) -> list[dict]:
        out = []
        for d in sorted(self.root.iterdir(), reverse=True) if self.root.exists() else []:
            if (d / "meta.json").exists():
                out.append({"trip_id": d.name, **json.loads((d / "meta.json").read_text(encoding="utf-8"))})
        return out

    def meta(self, trip_id: str) -> dict:
        return json.loads((self.root / trip_id / "meta.json").read_text(encoding="utf-8"))

    def timeline(self, trip_id: str) -> list[dict]:
        return json.loads((self.root / trip_id / "timeline.json").read_text(encoding="utf-8"))

    def traces(self, trip_id: str) -> dict[str, EvidenceTrace]:
        return {p.stem.replace("trace_", ""): EvidenceTrace.load(p) for p in sorted((self.root / trip_id / "traces").glob("trace_*.json"))}

    def features(self, trip_id: str):
        p = self.root / trip_id / "features.npz"
        if not p.exists():
            return None
        from src.preprocessing.feature_cache import load_features
        return load_features(p)

    def tracks(self, trip_id: str) -> np.ndarray | None:
        p = self.root / trip_id / "tracks.npy"
        return np.load(p) if p.exists() else None


# --------------------------------------------------------------------------------------------- clips
def make_clip(video_path: str | Path, t0: float, t1: float, tracks: np.ndarray | None, timeline: list[dict], out: Path,
              width: int = 640, fps: float = 10.0) -> Path | None:
    """Annotated clip [t0, t1] re-read from the source video (boxes, track ids, level banner). None if the source is unavailable."""
    import cv2

    from src.preprocessing.frame_extractor import resize_width
    from src.visualization.overlay import draw_frame, write_video
    if not video_path or not Path(video_path).exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    lv = {round(r["t"], 1): r for r in timeline}
    t = max(0.0, t0)
    while t <= t1 + 1e-6:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            break
        fr = resize_width(fr, width)
        rows = tracks[np.isclose(tracks[:, 0] / fps, t, atol=0.051)] if tracks is not None and len(tracks) else np.zeros((0, 8))
        r = lv.get(round(t, 1), {})
        frames.append(draw_frame(fr, rows, r.get("level", "LOW").replace("_", " "), r.get("p"), r.get("p"), t))
        t = round(t + 1.0 / fps, 3)
    cap.release()
    if not frames:
        return None
    return write_video(frames, out, fps)


def _video_of(video_id: str) -> str | None:
    from src.data.dataset_loader import load_dataset
    from src.utils.config import load_config as v3cfg
    from src.utils.config import resolve as v3res
    c = v3cfg()
    for r in load_dataset(v3res(c["paths"]["annotations"]), v3res(c["paths"]["raw_dir"])):
        if r.video_id == video_id:
            return str(r.path)
    return None


# --------------------------------------------------------------------------------------------- analysis
def run_trip(feat, source_video: str | None, name: str, data_kind: str, store: EventStore, pipeline=None,
             progress=None) -> dict:
    from safedrive_v4.pipeline import V3ReadOnlyModel, V4Pipeline
    p = pipeline or V4Pipeline(V3ReadOnlyModel())
    ts = TripStore(store)
    trip_id = ts.new_id(name)
    t0 = time.perf_counter()
    out = p.process_video(feat, explain=True)
    wall = time.perf_counter() - t0
    traces = {tr.event_id: tr for tr in out["traces"]}
    evs = events_from_timeline(out["timeline"], traces, trip_id, source_video or feat.video_id, feat.tracks)
    for i, ev in enumerate(evs):
        if progress:
            progress(0.8 + 0.2 * i / max(len(evs), 1), f"saving event {i + 1}/{len(evs)}")
        store.save(ev)
        clip = None
        if store.clips_allowed:
            pre, post = 5.0, 5.0
            a, b = max(0.0, ev.t_onset - pre), ev.t_end + post
            clip = make_clip(source_video, a, b, feat.tracks, out["timeline"], store.root / "clips" / f"{ev.event_id}.mp4")
            ev.clip_offsets = {"pre_start": 0.0, "onset": ev.t_onset - a, "end": ev.t_end - a, "post_end": b - a}
        store.attach_clip(ev, clip, "" if store.clips_allowed else "")
    lv = {}
    for r in out["timeline"]:
        lv[r["level"]] = lv.get(r["level"], 0) + 1
    meta = {"name": name, "source": source_video or feat.video_id, "video_id": feat.video_id, "data_kind": data_kind,
            "created": datetime.now().isoformat(timespec="seconds"), "duration_s": float(feat.times[-1]) if len(feat.times) else 0.0,
            "model": p.model.meta.get("version"), "steps": len(out["timeline"]), "levels": lv, "n_events": len(evs),
            "analysis_wall_s": round(wall, 2), "latency": out["latency"], "encrypted_store": store.encrypted,
            "counts_as_real_driving": data_kind == "real_driving"}
    d = ts.save(trip_id, meta, out["timeline"], out["traces"], feat.tracks)
    from src.preprocessing.feature_cache import save_features
    save_features(feat, d / "features.npz")   # exact replay of the scene graph (tracks, camera transforms, scene channels)
    return {"trip_id": trip_id, "meta": meta, "events": evs}


def analyze_cached(video_id: str, store: EventStore, pipeline=None) -> dict:
    """Nexar clip with an existing v3 feature cache (demo data: never real driving)."""
    from safedrive_v4.config import resolve
    from src.preprocessing.feature_cache import load_features
    f = load_features(resolve(f"data/processed/features/clean/{video_id}.npz"))
    return run_trip(f, _video_of(video_id), f"nexar_{video_id}", "demo_video", store, pipeline)


def analyze_video(path: str | Path, store: EventStore, data_kind: str = "demo_video", pipeline=None, progress=None,
                  max_duration_s: float | None = 600.0) -> dict:
    """Any video file: v3 perception (YOLO + ByteTrack + flow + frame difference + ego motion) -> v4 pipeline."""
    from src.features.extractor import FeatureExtractor
    from src.utils.config import load_config as v3cfg
    from src.utils.device import get_device
    fx = FeatureExtractor(v3cfg(), get_device(), use_cnn=False, use_ego=True)
    feat, _ = fx.extract(path, Path(path).stem, progress=(lambda f, m: progress(0.8 * f, m)) if progress else None,
                         max_duration_s=max_duration_s)
    return run_trip(feat, str(path), Path(path).stem, data_kind, store, pipeline, progress)


# --------------------------------------------------------------------------------------------- trip report
def trip_report(trip_id: str, store: EventStore) -> dict:
    from collections import Counter
    ts = TripStore(store)
    meta, tl = ts.meta(trip_id), ts.timeline(trip_id)
    evs = store.all(trip_id)
    factors = Counter()
    for tr in ts.traces(trip_id).values():
        for f in tr.findings:
            factors[f"{f['rule_id']} {f['conclusion']}"] += 1
    n = max(len(tl), 1)
    rep = {"trip_id": trip_id, "generated": datetime.now().isoformat(timespec="seconds"), **{k: meta.get(k) for k in
           ("name", "source", "data_kind", "duration_s", "model", "counts_as_real_driving", "encrypted_store")},
           "events_total": len(evs), "events_to_review": sum(e.status == "unreviewed" for e in evs),
           "confirmed_near_miss": sum(e.category == "near_miss" and e.status != "unreviewed" for e in evs),
           "confirmed_collision": sum(e.category == "collision" for e in evs),
           "rejected_false_alarms": sum(e.status == "rejected" for e in evs),
           "events_by_category": dict(Counter(e.category for e in evs)),
           "time_share_by_level": {k: v / n for k, v in Counter(r["level"] for r in tl).items()},
           "frequent_risk_factors": factors.most_common(6),
           "events": [{"event_id": e.event_id, "t_onset": e.t_onset, "t_end": e.t_end, "category": e.category, "status": e.status,
                       "level": e.level_peak, "p_peak": e.p_peak, "kg_evidence": e.kg_evidence_peak, "reliability_min": e.reliability_min,
                       "cautious": e.cautious, "explanation": e.explanation[:2], "clip_status": e.clip_status} for e in sorted(evs, key=lambda e: e.t_onset)],
           "limitations": ["No composite safety score is produced (none has been validated).",
                           "Risk estimates come from the frozen v3 model (held-out recall about 12 % at about 9 false alarms per hour); "
                           "most dangerous situations are NOT detected, and fewer than 5 % are warned 2 s ahead.",
                           "Explanations: about half of the warnings have a knowledge-graph reasoning chain; the rest are explained "
                           "only by feature contributions.",
                           "Image-based geometry; no lanes, traffic lights or physical distances.",
                           "Data kind: " + str(meta.get("data_kind")) + (" (does not count as real-world driving)" if meta.get("data_kind") != "real_driving" else "")],
           "disclaimer": DISCLAIMER}
    return rep


def _latin(s) -> str:
    return str(s).encode("latin-1", "replace").decode("latin-1")


def trip_report_pdf(rep: dict) -> bytes:
    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(True, 14)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _latin(f"SafeDrive AI V4 - Trip report: {rep.get('name') or rep['trip_id']}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9.5)
    dur = rep.get("duration_s") or 0
    for ln in [f"Generated {rep['generated']} | source {rep.get('source')} | duration {dur / 60:.1f} min | data kind {rep.get('data_kind')}",
               f"Model: {rep.get('model')} | encrypted event store: {rep.get('encrypted_store')}",
               f"Events: {rep['events_total']} (to review: {rep['events_to_review']}; confirmed near-miss: {rep['confirmed_near_miss']}; "
               f"confirmed collision: {rep['confirmed_collision']}; rejected false alarms: {rep['rejected_false_alarms']})",
               "By category: " + ", ".join(f"{k} {v}" for k, v in rep["events_by_category"].items()),
               "Time share by state: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in rep["time_share_by_level"].items()),
               "Frequent risk factors (rule findings at warning onsets): " + (", ".join(f"{k} ({v})" for k, v in rep["frequent_risk_factors"]) or "none")]:
        pdf.multi_cell(0, 5.2, _latin(ln), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Event timeline", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    for e in rep["events"][:200]:
        pdf.multi_cell(0, 4.2, _latin(f"t={e['t_onset']:.1f}-{e['t_end']:.1f}s [{e['category']} / {e['status']}] {e['level']} "
                                      f"p={e['p_peak']:.2f} KG-evidence={e['kg_evidence']:.2f} reliability>={e['reliability_min']:.2f}"
                                      f"{' (cautious)' if e['cautious'] else ''} - {'; '.join(e['explanation']) or 'no rule fired (feature contributions only)'}"),
                       new_x="LMARGIN", new_y="NEXT")
    if not rep["events"]:
        pdf.cell(0, 6, "No warning events in this trip.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Limitations", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    for x in rep["limitations"]:
        pdf.multi_cell(0, 4.2, _latin("- " + x), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _latin(rep["disclaimer"]), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
