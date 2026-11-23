"""v3.4.3 operator checklist before EVERY real-driving session (done while PARKED; never while the vehicle moves).

Automatic items are measured by the software at the moment the checklist is saved; manual items are confirmations by the
operator and the supervisor (named). A real-driving session can start only with a complete checklist saved within the last
CHECKLIST_VALID_H hours whose automatic items all passed; at session start the cheap automatic items are re-checked.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from src.realworld import readiness
from src.realworld.trip_meta import ConsentError
from src.utils.config import resolve

CHECKLIST_DIR = resolve("data/campaign/checklists")
CHECKLIST_VALID_H = 2.0
CAMERA_CHECK_VALID_MIN = 60.0
READINESS_VALID_DAYS = 7.0

MANUAL_ITEMS = {
    "vehicle_parked": "The vehicle is parked; nobody operates the screen while driving",
    "camera_fixed": "The camera is fixed firmly (does not move when touched)",
    "camera_view": "The view shows the road ahead, horizon roughly level, hood at most about 1/3 of the image",
    "timeline_camera_ok": "On the Live page, the camera runs, state is not DEGRADED, the timeline is updating",
    "supervisor_present": "A supervisor is present and knows the protocol (watches the road, not the screen)",
    "consent_recording": "Driver consent obtained; occupants informed; local recording rules respected",
    "participants_briefed": "Everyone in the vehicle was told what is recorded, how it is stored and how to request deletion",
    # v3.4.4 safety points (repeated before every session; the coordinator's one-off camera / safety checklist is item 12)
    "no_view_obstruction": "Camera, mount and cable do not obstruct the driver's view or the mirrors",
    "airbag_clear": "Nothing is mounted on or in front of an airbag zone",
    "cable_safe": "Cables are fixed away from pedals, gear lever, steering wheel and door",
    "no_laptop_while_moving": "Recording is started now, while parked; nobody will touch the laptop until the vehicle is parked again",
    "stop_procedure": "Everyone knows the stop procedure: problems are handled only after parking; the driver ignores the prototype",
}


class ChecklistError(ConsentError):
    pass


def auto_checks(model_dir: str = "models/safedrive_v3", sessions_root: Path | None = None, access_path: Path | None = None,
                backup_path: Path = readiness.BACKUP_STATUS_PATH, camera_path: Path = readiness.CAMERA_CHECK_PATH,
                readiness_path: Path = readiness.READINESS_LATEST, data_dir: Path | None = None, now: datetime | None = None) -> dict:
    from src.realworld import privacy, provenance
    from src.realworld.trip_meta import SESSIONS_DIR
    now = now or datetime.now()
    out = {}
    acc = privacy.load_access(access_path or privacy.ACCESS_PATH)
    out["access"] = {"ok": bool(acc.get("raw_access")) and bool(acc.get("coordinators")), "detail": acc}
    free = shutil.disk_usage(readiness.sessions_location(Path(data_dir or resolve("data")))).free / 1e9
    out["disk_space"] = {"ok": free >= readiness.MIN_FREE_GB_PER_SESSION, "detail": f"{free:.1f} GB free (≥ {readiness.MIN_FREE_GB_PER_SESSION} GB)"}
    # every earlier real-driving session must be in a verified external backup
    root = Path(sessions_root or SESSIONS_DIR)
    real = []
    for s in readiness.campaign_sessions(root):
        m = json.loads((root / s / "trip_meta.json").read_text(encoding="utf-8"))
        if m.get("data_kind") == "real_driving":
            real.append(s)
    recs = readiness.backup_records(backup_path)
    lb, lv = readiness._latest(recs, "backup"), readiness._latest(recs, "verify")
    ok_b = lb is not None and lb.get("ok") and lb.get("external") and not lb.get("test_mode") and \
        lv is not None and lv.get("ok") and lv["at"] >= lb["at"] and set(lb.get("sessions", [])) >= set(real)
    out["backup"] = {"ok": bool(ok_b), "detail": f"{len(real)} earlier real-driving session(s); last backup {lb and lb.get('at')}, "
                                                 f"last verify {lv and lv.get('at')} ok={lv and lv.get('ok')}"}
    cc = readiness.latest_camera_check(camera_path)
    fresh = cc is not None and (now - datetime.fromisoformat(cc["at"])).total_seconds() <= CAMERA_CHECK_VALID_MIN * 60
    out["camera_check"] = {"ok": bool(fresh and cc.get("ok")), "detail": cc if cc else "no camera check (press 'Test camera' while parked)"}
    prov = provenance.compute(model_dir)
    chk = provenance.check(prov)
    out["model_hash"] = {"ok": bool(chk.get("eligible")), "detail": chk}
    rd = readiness.load_latest(readiness_path)
    rd_ok = rd is not None and rd["decision"]["verdict"] == "GO" and (now - datetime.fromisoformat(rd["at"])).days < READINESS_VALID_DAYS
    out["precampaign_go"] = {"ok": bool(rd_ok), "detail": f"latest readiness {rd['at'] if rd else 'none'}: "
                                                          f"{rd['decision']['verdict'] if rd else 'not run'}"}
    return out


def save(operator: str, supervisor: str, manual: dict, auto: dict, path_dir: Path = CHECKLIST_DIR) -> dict:
    if not operator.strip() or not supervisor.strip():
        raise ValueError("operator and supervisor pseudonyms are required")
    if operator.strip() == supervisor.strip():
        raise ValueError("the supervisor must be a different person from the operator / driver")
    missing_manual = [k for k in MANUAL_ITEMS if not manual.get(k)]
    failed_auto = [k for k, v in auto.items() if not v["ok"]]
    rec = {"at": datetime.now().isoformat(timespec="seconds"), "operator": operator.strip(), "supervisor": supervisor.strip(),
           "manual": {k: bool(manual.get(k)) for k in MANUAL_ITEMS}, "auto": auto, "missing_manual": missing_manual,
           "failed_auto": failed_auto, "complete": not missing_manual and not failed_auto}
    Path(path_dir).mkdir(parents=True, exist_ok=True)
    (Path(path_dir) / f"checklist_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json").write_text(
        json.dumps(rec, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return rec


def latest(path_dir: Path = CHECKLIST_DIR) -> dict | None:
    fs = sorted(Path(path_dir).glob("checklist_*.json"))
    return json.loads(fs[-1].read_text(encoding="utf-8")) if fs else None


def require_ready(model_dir: str = "models/safedrive_v3", path_dir: Path | None = None, now: datetime | None = None,
                  data_dir: Path | None = None) -> dict:
    """Raise ChecklistError unless a complete checklist was saved within CHECKLIST_VALID_H hours and the cheap automatic items
    (free disk space, model / feature / threshold hashes) still pass now."""
    from src.realworld import provenance
    now = now or datetime.now()
    rec = latest(Path(path_dir) if path_dir else CHECKLIST_DIR)
    if rec is None:
        raise ChecklistError("no pre-session checklist: complete it on the 'Campaign Readiness' page while parked")
    if not rec.get("complete"):
        raise ChecklistError("the latest checklist is incomplete: " + ", ".join(rec.get("missing_manual", []) + rec.get("failed_auto", [])))
    if (now - datetime.fromisoformat(rec["at"])).total_seconds() > CHECKLIST_VALID_H * 3600:
        raise ChecklistError(f"the latest checklist is older than {CHECKLIST_VALID_H:g} h; repeat it while parked")
    free = shutil.disk_usage(readiness.sessions_location(Path(data_dir or resolve("data")))).free / 1e9
    if free < readiness.MIN_FREE_GB_PER_SESSION:
        raise ChecklistError(f"only {free:.1f} GB free on the data drive")
    chk = provenance.check(provenance.compute(model_dir))
    if not chk.get("eligible"):
        raise ChecklistError(f"model / feature / threshold hash differs from the frozen reference: {chk}")
    return rec
