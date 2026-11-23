"""Trip / session metadata DECLARED by the person running the session (never inferred automatically).

Conditions such as weather or road type are only what the user records. "unknown" is always allowed and is the default.
Real-driving sessions require explicit consent and a supervisor, and must be configured BEFORE the vehicle moves.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from src.utils.config import resolve

SESSIONS_DIR = resolve("data/sessions")

DATA_KINDS = {
    "real_driving": "Real driving on public roads (supervised)",
    "controlled_test": "Controlled test (closed area / staged, no public-road risk)",
    "live_camera_test": "Live camera test (not driving, e.g. indoor / desk)",
    "demo_video": "Demo video replay (recorded footage, not live)",
}
LIGHTING = ["unknown", "day", "night", "low_light", "backlight", "dawn_dusk", "tunnel_changing"]
WEATHER = ["unknown", "clear", "cloudy", "rain", "fog", "snow"]
ROAD = ["straight", "curves", "urban", "highway", "rural", "intersection_heavy", "roundabout"]
TRAFFIC = ["unknown", "sparse", "moderate", "dense"]
SITUATIONS = ["motorcycle_cut_in", "lead_braking", "lane_change", "pedestrians", "parking_lot", "other"]
MOUNTS = ["unknown", "fixed_windshield", "fixed_dashboard", "handheld", "vibrating_loose"]


class ConsentError(ValueError):
    pass


@dataclass
class TripMeta:
    session_id: str
    data_kind: str = "live_camera_test"
    driver_consent: bool = False
    occupants_informed: bool = False
    supervisor_present: bool = False
    configured_before_driving: bool = False
    lighting: str = "unknown"
    weather: str = "unknown"
    road_types: list = field(default_factory=list)
    traffic: str = "unknown"
    situations_expected: list = field(default_factory=list)
    camera_mount: str = "unknown"
    location_note: str = ""   # free text; avoid precise personal locations
    notes: str = ""
    record_full_video: bool = False
    blur_people: bool = True
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    fully_reviewed_by: list = field(default_factory=list)  # annotators who reviewed the WHOLE session (needed for recall)
    # ---- v3.4 campaign fields ----
    campaign_id: str = ""
    trip_id: str = ""            # one drive (a trip may span several sessions); split unit together with driver_id
    driver_id: str = ""          # pseudonym only (e.g. D01), never a real name
    regulations_checked: bool = False   # the operator confirmed local recording rules were checked before recording
    share_consent: bool = False         # the driver allows sharing a BLURRED copy within the research team

    def validate(self) -> None:
        if self.data_kind not in DATA_KINDS:
            raise ValueError(f"unknown data kind {self.data_kind}")
        if self.data_kind in ("real_driving", "controlled_test"):
            missing = [k for k in ("driver_consent", "supervisor_present", "configured_before_driving") if not getattr(self, k)]
            if missing:
                raise ConsentError("Real-driving / controlled sessions need: " + ", ".join(missing))
        if self.record_full_video and not (self.driver_consent and self.occupants_informed):
            raise ConsentError("Full-session video needs the driver's consent and informed occupants.")
        if self.data_kind == "real_driving" and self.campaign_id and not self.regulations_checked:
            raise ConsentError("Campaign real-driving sessions need confirmation that local recording rules were checked.")
        if self.driver_id and any(c.isspace() for c in self.driver_id):
            raise ValueError("driver_id must be a pseudonym code such as D01 (no names)")
        for v, allowed in ((self.lighting, LIGHTING), (self.weather, WEATHER), (self.traffic, TRAFFIC), (self.camera_mount, MOUNTS)):
            if v not in allowed:
                raise ValueError(f"invalid value {v}")


def meta_path(session_id: str, root: Path = SESSIONS_DIR) -> Path:
    return Path(root) / session_id / "trip_meta.json"


def save_meta(m: TripMeta, root: Path = SESSIONS_DIR) -> Path:
    m.validate()
    p = meta_path(m.session_id, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(m), indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def load_meta(session_id: str, root: Path = SESSIONS_DIR) -> TripMeta | None:
    p = meta_path(session_id, root)
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return TripMeta(**{k: v for k, v in d.items() if k in TripMeta.__dataclass_fields__})


def list_sessions(root: Path = SESSIONS_DIR) -> list[str]:
    root = Path(root)
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []


def delete_session(session_id: str, root: Path = SESSIONS_DIR, memory_root: Path | None = None, requested_by: str = "",
                   reason: str = "", log_path: Path | None = None) -> dict:
    """Delete everything recorded for a session: its folder (timeline, video, annotations, reports) and its event clips +
    metadata in the Near-Miss Memory store."""
    import shutil
    root = Path(root)
    removed = {"session_dir": False, "events": 0, "clips": 0}
    d = root / session_id
    if d.exists():
        shutil.rmtree(d)
        removed["session_dir"] = True
    mem = Path(memory_root) if memory_root else resolve("data/memory")
    ev = mem / "events.jsonl"
    if ev.exists():
        keep, gone = [], []
        for line in ev.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            (gone if e.get("session_id") == session_id or e.get("trip_id") == f"live-{session_id}" else keep).append(e)
        for e in gone:
            if e.get("clip") and Path(e["clip"]).exists():
                Path(e["clip"]).unlink()
                removed["clips"] += 1
        ev.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in keep), encoding="utf-8")
        removed["events"] = len(gone)
    # audit trail of the deletion itself (ids and counts only, no content)
    lp = Path(log_path) if log_path else resolve("data/campaign/deletion_log.jsonl")
    lp.parent.mkdir(parents=True, exist_ok=True)
    with open(lp, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"session_id": session_id, "deleted_at": datetime.now().isoformat(timespec="seconds"),
                             "requested_by": requested_by, "reason": reason, **removed}, ensure_ascii=False) + "\n")
    return removed
