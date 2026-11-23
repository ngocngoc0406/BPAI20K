"""Near-Miss Intelligence store (v4).

Automatic categories (honest naming; nothing is confirmed automatically):
  potential_near_miss   HIGH warning (not cautious) with knowledge-graph risk evidence  -> "potential": needs human review
  risk_event            MODERATE warning, or HIGH without KG risk evidence
  uncertain             HIGH flagged cautious (low perception reliability / model disagreement)
Human-only categories (set by review, with evidence text): collision, near_miss (confirmed), risk_event, normal (false alarm).

Each event stores: pre / event / post clip (one file + offsets), track history of the involved road users over the clip span,
risk timeline (probability, level, KG risk opinion), the sealed evidence trace, prediction and uncertainty, annotation status
(unreviewed / confirmed / rejected / relabelled) and the review log.

Privacy: clips contain faces and plates. The store lives INSIDE the encrypted session volume (next to data/sessions, i.e. on V:
when the VeraCrypt vault is mounted). If the encrypted location is not available, events are stored WITHOUT clips
(clip_status says why); nothing is written unencrypted.
"""
from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

AUTO_CATEGORIES = ("potential_near_miss", "risk_event", "uncertain")
HUMAN_CATEGORIES = ("collision", "near_miss", "risk_event", "normal", "uncertain")
STATUSES = ("unreviewed", "confirmed", "rejected", "relabelled")


def default_root() -> tuple[Path, bool]:
    """(store root, encrypted?) - next to the session store; encrypted when data/sessions resolves into a VeraCrypt volume."""
    from safedrive_v4.config import resolve
    sess = resolve("data/sessions")
    try:
        real = sess.resolve(strict=True)
    except OSError:
        return resolve("data/v4_events_unavailable"), False
    try:
        from src.realworld.readiness import veracrypt_volume
        enc = veracrypt_volume(real.drive.rstrip(":")) is not None
    except Exception:  # noqa: BLE001
        enc = False
    return real.parent / "safedrive_v4_events", enc


def auto_category(level: str, cautious: bool, kg_evidence: float, evidence_min: float = 0.2) -> str:
    if level == "HIGH" and cautious:
        return "uncertain"
    if level == "HIGH" and kg_evidence >= evidence_min:
        return "potential_near_miss"
    return "risk_event"


@dataclass
class V4Event:
    event_id: str
    trip_id: str
    source: str
    category_auto: str
    t_onset: float
    t_peak: float
    t_end: float
    level_peak: str
    p_peak: float
    kg_evidence_peak: float
    reliability_min: float
    cautious: bool
    track_ids: list[int]
    explanation: list[str]
    trace_file: str | None = None
    clip: str | None = None
    clip_status: str = "pending"
    clip_offsets: dict = field(default_factory=dict)   # pre_start, onset, end, post_end (s, relative to clip start)
    track_history: list = field(default_factory=list)  # [t, track_id, cls, conf, x1, y1, x2, y2]
    risk_timeline: list = field(default_factory=list)  # [t, p, level, kg_expectation, reliability]
    status: str = "unreviewed"
    category_human: str | None = None
    review_log: list = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def category(self) -> str:
        return self.category_human or self.category_auto


class EventStore:
    def __init__(self, root: str | Path | None = None, require_encrypted: bool = True):
        if root is None:
            root, enc = default_root()
        else:
            enc = True   # explicit roots are used by tests / exports
        self.root = Path(root)
        self.encrypted = enc
        self.clips_allowed = enc or not require_encrypted
        (self.root / "events").mkdir(parents=True, exist_ok=True)
        (self.root / "clips").mkdir(parents=True, exist_ok=True)

    def path(self, event_id: str) -> Path:
        return self.root / "events" / f"{event_id}.json"

    def save(self, ev: V4Event) -> V4Event:
        tmp = self.path(ev.event_id).with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(ev), indent=1, default=float), encoding="utf-8")
        tmp.replace(self.path(ev.event_id))
        return ev

    def get(self, event_id: str) -> V4Event:
        return V4Event(**json.loads(self.path(event_id).read_text(encoding="utf-8")))

    def all(self, trip_id: str | None = None) -> list[V4Event]:
        out = [V4Event(**json.loads(p.read_text(encoding="utf-8"))) for p in sorted((self.root / "events").glob("*.json"))]
        return [e for e in out if trip_id is None or e.trip_id == trip_id]

    def trips(self) -> list[str]:
        return sorted({e.trip_id for e in self.all()})

    def review(self, event_id: str, decision: str, reviewer: str, category: str | None = None, notes: str = "",
               evidence: str = "") -> V4Event:
        """decision: confirm | reject | relabel. 'collision' needs written evidence; rejecting = false alarm (category normal)."""
        if not reviewer.strip():
            raise ValueError("reviewer name / pseudonym required")
        ev = self.get(event_id)
        if decision == "confirm":
            cat = category or {"potential_near_miss": "near_miss"}.get(ev.category_auto, ev.category_auto)
            ev.status = "confirmed"
        elif decision == "reject":
            cat, ev.status = "normal", "rejected"
        elif decision == "relabel":
            if category not in HUMAN_CATEGORIES:
                raise ValueError(f"category must be one of {HUMAN_CATEGORIES}")
            cat, ev.status = category, "relabelled"
        else:
            raise ValueError("decision must be confirm / reject / relabel")
        if cat not in HUMAN_CATEGORIES:
            raise ValueError(f"category must be one of {HUMAN_CATEGORIES}")
        if cat == "collision" and len(evidence.strip()) < 10:
            raise ValueError("a collision can only be recorded with written evidence (what was seen, damage, report)")
        ev.category_human = cat
        ev.review_log.append({"at": datetime.now().isoformat(timespec="seconds"), "reviewer": reviewer, "decision": decision,
                              "category": cat, "notes": notes, "evidence": evidence})
        return self.save(ev)

    def delete(self, event_id: str) -> None:
        ev = self.get(event_id)
        if ev.clip and Path(ev.clip).exists():
            Path(ev.clip).unlink()
        self.path(event_id).unlink()

    def export(self, trip_id: str, out: str | Path) -> Path:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([asdict(e) for e in self.all(trip_id)], indent=1, default=float), encoding="utf-8")
        return out

    def attach_clip(self, ev: V4Event, src_path: str | Path | None, reason_if_none: str = "") -> V4Event:
        if src_path is None:
            ev.clip_status = reason_if_none or "no frames"
        elif not self.clips_allowed:
            ev.clip_status = "not saved: encrypted store (VeraCrypt V:) not available - clips are never written unencrypted"
            Path(src_path).unlink(missing_ok=True)
        else:
            dst = self.root / "clips" / f"{ev.event_id}.mp4"
            if Path(src_path) != dst:
                shutil.move(str(src_path), dst)
            ev.clip, ev.clip_status = str(dst), "saved"
        return self.save(ev)


def new_event_id() -> str:
    return uuid.uuid4().hex[:12]


def events_from_timeline(timeline: list[dict], traces: dict, trip_id: str, source: str, tracks=None, k_post: int = 10) -> list[V4Event]:
    """Episodes of MODERATE / HIGH warnings in a processed timeline -> events (onset .. last warning step)."""
    evs, i = [], 0
    n = len(timeline)
    while i < n:
        if timeline[i]["level"] not in ("MODERATE", "HIGH"):
            i += 1
            continue
        j = i
        while j + 1 < n and timeline[j + 1]["level"] in ("MODERATE", "HIGH"):
            j += 1
        seg = timeline[i:j + 1]
        pk = max(seg, key=lambda r: r["p"])
        lvl = "HIGH" if any(r["level"] == "HIGH" for r in seg) else "MODERATE"
        cautious = any(r.get("cautious") for r in seg if r["level"] == "HIGH")
        kge = max(r.get("kg_risk_expectation", 0.0) for r in seg)
        trs = [traces[r["event_id"]] for r in seg if r.get("event_id") in traces]
        tids = sorted({t for tr in trs for t in tr.track_ids})
        expl = [p for tr in trs for p in (" -> ".join(path) for path in tr.kg_paths)]
        ev = V4Event(new_event_id(), trip_id, source, auto_category(lvl, cautious, kge), seg[0]["t"], pk["t"], seg[-1]["t"], lvl, pk["p"],
                     kge, min(r["reliability"] for r in seg), cautious, tids, expl[:5],
                     trace_file=trs[0].event_id if trs else None)
        lo, hi = max(0, i - 50), min(n, j + 1 + 50)
        ev.risk_timeline = [[r["t"], r["p"], r["level"], r.get("kg_risk_expectation", 0.0), r["reliability"]] for r in timeline[lo:hi]]
        if tracks is not None and len(tracks):
            t0, t1 = timeline[lo]["t"], timeline[hi - 1]["t"]
            st = tracks[:, 0] / 10.0
            sel = tracks[(st >= t0) & (st <= t1)]
            if tids:
                sel = sel[[int(x) in tids for x in sel[:, 1]]]
            ev.track_history = [[float(r[0]) / 10.0, *map(float, r[1:])] for r in sel]
        evs.append(ev)
        i = j + 1
    return evs
