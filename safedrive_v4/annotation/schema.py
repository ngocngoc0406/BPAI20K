"""Annotation schema ann-v4.0 (see docs/SAFEDRIVE_V4_ANNOTATION_DATA_SCHEMA.md). Validation is strict: a label that cannot be
supported by evidence is rejected, an uncertain label needs a reason, and nothing is converted into 'normal'."""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime

SCHEMA = "ann-v4.0"
CATEGORIES = ("normal", "risk_event", "near_miss", "collision", "uncertain")
EVENT_CATEGORIES = ("risk_event", "near_miss", "collision")
RELATIONS = ("APPROACHING", "MOVING_AWAY", "CUTTING_IN", "FOLLOWING", "CROSSING", "BRAKING", "YIELDING", "BLOCKING",
             "COLLISION_INTERACTION")
UNARY_OK = ("BRAKING",)           # a relation that may have no object other than the ego context (e.g. lead braking)
OBSERVATION = ("observed", "inferred_from_sequence", "insufficient_evidence")
ACTOR_CLASSES = ("ego", "car", "truck", "bus", "motorcycle", "bicycle", "pedestrian", "other")
VISIBILITY = ("clear", "partial", "poor", "not_visible")
OCCLUSION = ("none", "partial", "heavy")
BLUR = ("none", "mild", "severe")
LIGHTING = ("good", "low", "very_low", "glare")
CONFIDENCE = ("low", "medium", "high")
ORDINAL = {"visibility_level": VISIBILITY, "occlusion_level": OCCLUSION, "motion_blur": BLUR, "lighting_quality": LIGHTING,
           "annotation_confidence": CONFIDENCE}


class ValidationError(ValueError):
    pass


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class EvidenceRef:
    video_id: str
    t: float
    frame_index: int
    image_sha256: str | None = None      # JPEG of the frame stored in the encrypted store (evidence/<video>/<sha>.jpg)
    track_id: int | None = None
    box: list | None = None              # [x1, y1, x2, y2] normalised (optional)
    note: str = ""


@dataclass
class Uncertainty:
    visibility_level: str = "clear"
    occlusion_level: str = "none"
    motion_blur: str = "none"
    lighting_quality: str = "good"
    annotation_confidence: str = "medium"
    uncertainty_reason: str = ""

    def validate(self, category: str | None = None) -> None:
        for k, allowed in ORDINAL.items():
            if getattr(self, k) not in allowed:
                raise ValidationError(f"{k} must be one of {allowed}")
        needs = (category == "uncertain" or self.annotation_confidence == "low" or self.visibility_level in ("poor", "not_visible"))
        if needs and len(self.uncertainty_reason.strip()) < 5:
            raise ValidationError("an uncertainty reason is required (uncertain category, low confidence or poor / no visibility)")


@dataclass
class Actor:
    video_id: str
    annotator: str
    label: str
    cls: str
    track_id: int | None = None
    ref_frame_t: float | None = None
    box: list | None = None
    note: str = ""
    record_id: str = field(default_factory=lambda: new_id("act"))
    schema: str = SCHEMA

    def validate(self) -> None:
        if not self.label.strip():
            raise ValidationError("actor label required")
        if self.cls not in ACTOR_CLASSES:
            raise ValidationError(f"actor class must be one of {ACTOR_CLASSES}")
        if self.cls != "ego" and self.track_id is None and (self.box is None or self.ref_frame_t is None):
            raise ValidationError("an actor needs a tracker id or a box at a reference frame (so it can be found again)")


@dataclass
class TemporalEvent:
    video_id: str
    annotator: str
    category: str
    t_start: float
    t_end: float
    t_peak: float | None = None
    actors: list = field(default_factory=list)          # actor labels
    evidence: list = field(default_factory=list)        # EvidenceRef dicts
    rationale: str = ""
    observation: str = "observed"
    uncertainty: dict = field(default_factory=lambda: asdict(Uncertainty()))
    record_id: str = field(default_factory=lambda: new_id("evt"))
    schema: str = SCHEMA

    @property
    def duration(self) -> float:
        return float(self.t_end - self.t_start)

    def validate(self) -> None:
        if self.category not in CATEGORIES:
            raise ValidationError(f"category must be one of {CATEGORIES}")
        if self.observation not in OBSERVATION:
            raise ValidationError(f"observation must be one of {OBSERVATION}")
        if not (self.t_end >= self.t_start >= 0):
            raise ValidationError("need 0 <= t_start <= t_end")
        Uncertainty(**self.uncertainty).validate(self.category)
        if self.category in EVENT_CATEGORIES:
            if self.t_peak is None or not (self.t_start <= self.t_peak <= self.t_end):
                raise ValidationError("risk_event / near_miss / collision need a peak time inside [start, end]")
            if self.t_start == self.t_peak and self.category != "collision":
                pass  # allowed: an abrupt situation
            if not self.evidence:
                raise ValidationError("an event needs at least one evidence frame")
            if self.observation == "insufficient_evidence":
                raise ValidationError("insufficient evidence -> use the 'uncertain' category, not an event category")
        if self.category != "normal" and len(self.rationale.strip()) < 5:
            raise ValidationError("a rationale (what is seen) is required")
        for e in self.evidence:
            ev = EvidenceRef(**e)
            if not (self.t_start - 5.0 <= ev.t <= self.t_end + 5.0):
                raise ValidationError("evidence frames must lie within the event (± 5 s)")
            if ev.image_sha256 is None:
                raise ValidationError("evidence must reference a stored frame image (capture it in the evidence viewer)")


@dataclass
class Interaction:
    video_id: str
    annotator: str
    subject: str                     # actor label
    relation: str
    t_start: float
    t_end: float
    object: str | None = "ego"       # actor label, 'ego', or None for unary relations (BRAKING)
    evidence: list = field(default_factory=list)
    annotator_confidence: str = "medium"
    observation: str = "observed"
    event_id: str | None = None
    rationale: str = ""
    record_id: str = field(default_factory=lambda: new_id("int"))
    schema: str = SCHEMA

    def validate(self) -> None:
        if self.relation not in RELATIONS:
            raise ValidationError(f"relation must be one of {RELATIONS}")
        if self.observation not in OBSERVATION:
            raise ValidationError(f"observation must be one of {OBSERVATION}")
        if self.annotator_confidence not in CONFIDENCE:
            raise ValidationError(f"confidence must be one of {CONFIDENCE}")
        if not (self.t_end >= self.t_start >= 0):
            raise ValidationError("need 0 <= t_start <= t_end")
        if self.object is None and self.relation not in UNARY_OK:
            raise ValidationError(f"{self.relation} needs an object actor (or ego)")
        if self.object == self.subject:
            raise ValidationError("subject and object must differ")
        if self.observation != "insufficient_evidence":
            frames = {round(EvidenceRef(**e).t, 2) for e in self.evidence}
            if len(frames) < 2:
                raise ValidationError("a relation needs evidence at >= 2 different times (never from a single frame)")
            if self.t_end - self.t_start < 0.2:
                raise ValidationError("a relation needs a duration >= 0.2 s")
        for e in self.evidence:
            if EvidenceRef(**e).image_sha256 is None:
                raise ValidationError("evidence must reference a stored frame image")


@dataclass
class Counterfactual:
    """RESEARCH ONLY: never ground truth, never exported with labels, never used for training."""
    video_id: str
    annotator: str
    event_id: str
    observed_facts: list           # record ids of evidence-bearing events / interactions (what was SEEN)
    interpretation: str            # the annotator's reading of the observed facts
    hypothetical: str              # the what-if
    answerable: str = "partly"     # yes | partly | not_answerable_from_video
    research_only: bool = True
    record_id: str = field(default_factory=lambda: new_id("cf"))
    schema: str = SCHEMA

    def validate(self) -> None:
        if not self.observed_facts:
            raise ValidationError("a counterfactual must point to observed facts (records with evidence)")
        if len(self.interpretation.strip()) < 5 or len(self.hypothetical.strip()) < 5:
            raise ValidationError("interpretation and hypothetical are both required and kept separate")
        if self.answerable not in ("yes", "partly", "not_answerable_from_video"):
            raise ValidationError("answerable must be yes / partly / not_answerable_from_video")
        if not self.research_only:
            raise ValidationError("counterfactual records are research-only")


RECORD_TYPES = {"actor": Actor, "event": TemporalEvent, "interaction": Interaction, "counterfactual": Counterfactual}
