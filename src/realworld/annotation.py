"""Manual ground-truth annotation of real-world sessions + inter-annotator agreement (schema v3.4).

Each annotation describes a traffic situation observed by a human reviewer from the video (NOT what the model said):
  category     collision | near_miss | risk_event | normal | uncertain
  risk_onset   start: when the risk became visible (s, session stream time)
  critical_t   EVENT TIME: impact (collision), closest approach / evasive action (near_miss), the moment a warning was needed at
               the latest (risk_event). Required for the three event categories.
  risk_end     end of the situation
  confidence   low | medium | high
  evidence     visual evidence: a written description AND >= 1 evidence frame time (evidence_frames) for event categories
Rules: a near-miss is never labelled only because the model warned (annotators work blind). If evidence is insufficient
-> 'uncertain'. Annotators are never forced to agree; disagreements are recorded with each annotator's reason.
v3.3 labels are mapped on load: confirmed_collision->collision, potential_near_miss->near_miss,
warning_worthy->risk_event, potential_risk->risk_event (flagged legacy_label).
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "v3.4"
CATEGORIES = ["collision", "near_miss", "risk_event", "normal", "uncertain"]
EVENT_CATEGORIES = {"collision", "near_miss", "risk_event"}
WARNING_WORTHY = EVENT_CATEGORIES  # every annotated event is a situation a warning should precede
LEGACY_MAP = {"confirmed_collision": "collision", "potential_near_miss": "near_miss", "warning_worthy": "risk_event",
              "potential_risk": "risk_event"}
TAGS = ["motorcycle_cut_in", "vehicle_cut_in", "lead_braking", "lane_change_ego", "lane_change_other", "curve", "intersection",
        "pedestrian", "cyclist", "dense_traffic", "low_light", "backlight", "glare", "rain", "camera_shake", "occlusion", "other"]
CONFIDENCE = ["low", "medium", "high"]


class AnnotationError(ValueError):
    pass


@dataclass
class Annotation:
    session_id: str
    annotator: str
    category: str
    risk_onset: float
    risk_end: float
    critical_t: float | None = None
    tags: list = field(default_factory=list)
    confidence: str = "medium"
    note: str = ""
    evidence: str = ""
    evidence_frames: list = field(default_factory=list)  # session times (s) of frames that show the evidence
    legacy_label: str = ""
    schema: str = SCHEMA_VERSION
    annot_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def __post_init__(self):
        if self.category in LEGACY_MAP:
            self.legacy_label, self.category = self.category, LEGACY_MAP[self.category]

    def validate(self) -> None:
        if self.category not in CATEGORIES:
            raise AnnotationError(f"invalid category {self.category}")
        if self.confidence not in CONFIDENCE:
            raise AnnotationError(f"invalid confidence {self.confidence}")
        if not self.annotator.strip():
            raise AnnotationError("annotator name required")
        if self.risk_end < self.risk_onset:
            raise AnnotationError("risk_end must be >= risk_onset")
        if self.critical_t is not None and not (self.risk_onset <= self.critical_t <= self.risk_end):
            raise AnnotationError("critical_t must lie inside [risk_onset, risk_end]")
        if self.category in EVENT_CATEGORIES:
            if self.critical_t is None:
                raise AnnotationError(f"{self.category} needs the event time")
            if len(self.evidence.strip()) < 5 or not self.evidence_frames:
                raise AnnotationError(f"{self.category} needs visual evidence: a description and at least one evidence frame time")
            if any(not (self.risk_onset - 5 <= float(t) <= self.risk_end + 5) for t in self.evidence_frames):
                raise AnnotationError("evidence frames must lie within (or <= 5 s around) the situation")
        bad = [t for t in self.tags if t not in TAGS]
        if bad:
            raise AnnotationError(f"unknown tags {bad}")

    @property
    def critical(self) -> float:
        return self.critical_t if self.critical_t is not None else self.risk_end


def ann_path(session_dir: Path, annotator: str) -> Path:
    safe = "".join(c for c in annotator if c.isalnum() or c in "-_") or "anon"
    return Path(session_dir) / f"annotations_{safe}.jsonl"


def add(session_dir: Path, a: Annotation) -> Annotation:
    a.validate()
    p = ann_path(session_dir, a.annotator)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(a), ensure_ascii=False) + "\n")
    return a


def load(session_dir: Path, annotator: str | None = None) -> list[Annotation]:
    out = []
    files = [ann_path(session_dir, annotator)] if annotator else sorted(Path(session_dir).glob("annotations_*.jsonl"))
    for f in files:
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    out.append(Annotation(**{k: v for k, v in d.items() if k in Annotation.__dataclass_fields__}))
    return out


def annotators(session_dir: Path) -> list[str]:
    return sorted({a.annotator for a in load(session_dir, None)})


def delete(session_dir: Path, annotator: str, annot_id: str) -> bool:
    p = ann_path(session_dir, annotator)
    if not p.exists():
        return False
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    keep = [r for r in rows if r["annot_id"] != annot_id]
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
    return len(keep) < len(rows)


def _tiou(a: Annotation, b: Annotation) -> float:
    inter = max(0.0, min(a.risk_end, b.risk_end) - max(a.risk_onset, b.risk_onset))
    union = max(a.risk_end, b.risk_end) - min(a.risk_onset, b.risk_onset)
    return inter / union if union > 0 else float(a.risk_onset == b.risk_onset)


def cohen_kappa(x: list[str], y: list[str]) -> float:
    if not x:
        return float("nan")
    cats = sorted(set(x) | set(y))
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)))
    for a, b in zip(x, y):
        m[idx[a], idx[b]] += 1
    n = m.sum()
    po = np.trace(m) / n
    pe = float((m.sum(0) * m.sum(1)).sum()) / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def match(a_list: list[Annotation], b_list: list[Annotation], min_tiou: float = 0.3):
    """Greedy one-to-one matching by temporal IoU (highest first). Returns (pairs, unmatched_a, unmatched_b)."""
    cand = sorted(((_tiou(a, b), i, j) for i, a in enumerate(a_list) for j, b in enumerate(b_list)), reverse=True)
    ua, ub, pairs = set(range(len(a_list))), set(range(len(b_list))), []
    for t, i, j in cand:
        if t < min_tiou:
            break
        if i in ua and j in ub:
            ua.discard(i)
            ub.discard(j)
            pairs.append((a_list[i], b_list[j], t))
    pairs.sort(key=lambda p: p[0].risk_onset)
    return pairs, [a_list[i] for i in sorted(ua)], [b_list[j] for j in sorted(ub)]


def agreement(a_list: list[Annotation], b_list: list[Annotation], min_tiou: float = 0.3) -> dict:
    """Agreement between two independent annotators.
    - kappa_category: Cohen's kappa on the categories of MATCHED situations;
    - kappa_with_unmatched: kappa where a situation found by only one annotator is paired with 'none' (stricter);
    - kappa_event_vs_not: collision / near_miss / risk_event vs normal / uncertain / none;
    - time offsets (s) of start, event time and end for matched pairs (mean absolute and median)."""
    pairs, ua, ub = match(a_list, b_list, min_tiou)
    xa, xb = [p[0].category for p in pairs], [p[1].category for p in pairs]
    xa_u = xa + [a.category for a in ua] + ["none"] * len(ub)
    xb_u = xb + ["none"] * len(ua) + [b.category for b in ub]
    ev = lambda c: "event" if c in EVENT_CATEGORIES else "no_event"  # noqa: E731

    def offs(fn):
        d = [abs(fn(p[0]) - fn(p[1])) for p in pairs if fn(p[0]) is not None and fn(p[1]) is not None]
        return {"n": len(d), "mean_abs_s": float(np.mean(d)) if d else None, "median_abs_s": float(np.median(d)) if d else None}
    cats = CATEGORIES + ["none"]
    conf = {a: {b: 0 for b in cats} for a in cats}
    for a, b in zip(xa_u, xb_u):
        conf[a][b] += 1
    return {"matched": len(pairs), "only_first": len(ua), "only_second": len(ub),
            "kappa_category": cohen_kappa(xa, xb), "kappa_with_unmatched": cohen_kappa(xa_u, xb_u),
            "kappa_event_vs_not": cohen_kappa([ev(c) for c in xa_u], [ev(c) for c in xb_u]),
            "exact_category_agreement": float(np.mean([a == b for a, b in zip(xa, xb)])) if pairs else float("nan"),
            "onset_abs_diff_s_mean": offs(lambda z: z.risk_onset)["mean_abs_s"] if pairs else float("nan"),
            "offset_start": offs(lambda z: z.risk_onset), "offset_event_time": offs(lambda z: z.critical_t),
            "offset_end": offs(lambda z: z.risk_end), "mean_tiou": float(np.mean([p[2] for p in pairs])) if pairs else float("nan"),
            "confusion": conf}


# ---------------- disagreements and consensus (no forced agreement) ----------------
@dataclass
class Disagreement:
    session_id: str
    key: str                 # "<annot_id_A>|<annot_id_B>" (either side may be '-': found by one annotator only)
    annotator: str
    reason: str
    status: str = "open"     # open | agreed_after_discussion | unresolved
    agreed_category: str = ""
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


def disagreements_path(session_dir: Path) -> Path:
    return Path(session_dir) / "disagreements.jsonl"


def add_disagreement(session_dir: Path, d: Disagreement) -> None:
    if not d.reason.strip():
        raise AnnotationError("a reason is required")
    if d.status not in ("open", "agreed_after_discussion", "unresolved"):
        raise AnnotationError("invalid status")
    if d.status == "agreed_after_discussion" and d.agreed_category not in CATEGORIES:
        raise AnnotationError("agreed_after_discussion needs the agreed category")
    with open(disagreements_path(session_dir), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")


def load_disagreements(session_dir: Path) -> list[Disagreement]:
    p = disagreements_path(session_dir)
    if not p.exists():
        return []
    return [Disagreement(**json.loads(x)) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def pair_key(a: Annotation | None, b: Annotation | None) -> str:
    return f"{a.annot_id if a else '-'}|{b.annot_id if b else '-'}"


def consensus(a_list: list[Annotation], b_list: list[Annotation], disputes: list[Disagreement] | None = None,
              min_tiou: float = 0.3) -> list[dict]:
    """Reference events from two independent annotators.
    status: confirmed      both found it with the SAME category (times: mean of the two), or both agreed after discussion
            disputed       matched with different categories and not agreed (an 'unresolved' or 'open' record, or none)
            single         found by one annotator only (not agreed by the other)
    Only 'confirmed' events with an event category count for recall; disputed / single / uncertain windows are excluded
    zones for false alarms (an alarm there is neither true nor false)."""
    latest = {}
    for d in disputes or []:
        latest[d.key] = d  # last record wins
    pairs, ua, ub = match(a_list, b_list, min_tiou)
    out = []
    for a, b, t in pairs:
        d = latest.get(pair_key(a, b))
        cat = a.category if a.category == b.category else (d.agreed_category if d and d.status == "agreed_after_discussion" else None)
        crit = [x.critical_t for x in (a, b) if x.critical_t is not None]
        out.append({"status": "confirmed" if cat else "disputed", "category": cat or f"{a.category}/{b.category}",
                    "risk_onset": (a.risk_onset + b.risk_onset) / 2, "risk_end": (a.risk_end + b.risk_end) / 2,
                    "critical_t": float(np.mean(crit)) if crit else None, "tiou": t, "key": pair_key(a, b),
                    "tags": sorted(set(a.tags) | set(b.tags)), "reason": d.reason if d else ""})
    for a in ua + ub:
        k = pair_key(a, None) if a in ua else pair_key(None, a)
        d = latest.get(k)
        agreed = d is not None and d.status == "agreed_after_discussion"
        out.append({"status": "confirmed" if agreed else "single", "category": d.agreed_category if agreed else a.category,
                    "risk_onset": a.risk_onset, "risk_end": a.risk_end, "critical_t": a.critical_t, "tiou": None, "key": k,
                    "tags": list(a.tags), "reason": d.reason if d else ""})
    return sorted(out, key=lambda e: e["risk_onset"])
