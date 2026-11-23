"""v3.4 privacy tooling: blurred SHARED copies of full-session video, a mandatory human privacy review, and access tiers.

Raw video (`<session>/video/`) never leaves this computer and is visible only to people on the raw-access list.
A shared copy (`<session>/shared/`) is produced OFFLINE after the trip and may be shared within the research team only after a
human reviewer has checked it (privacy_review.json, approved = true). Nothing is uploaded by this software.

Blurring (no dedicated face or licence-plate detector is available in this project, so blurring is conservative and
region-based on the COCO detector, run at a LOW confidence threshold, with boxes kept for +-2 frames to bridge detector flicker):
  person                      whole box blurred (covers faces)
  bicycle / motorcycle        whole box blurred (riders, small plates)
  car / bus / truck           whole box blurred (v3.4 audit fix: a lower 45 % band missed rear plates mounted at mid-height)
  every box                   enlarged by BOX_PAD of its width / height on each side (detector boxes are not tight)
The blur recall has NOT been measured; the human review is therefore mandatory, and reviewers must reject copies with visible
faces or plates. Annotators who need brake lights / plates for labelling use the raw video (raw-access list).
"""
from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import resolve

ACCESS_PATH = resolve("data/campaign/access.json")
DET_CONF = 0.15
PLATE_BAND = 1.0   # fraction of a vehicle box (from its bottom) that is blurred; 1.0 = whole box (audit 2026-11-22)
BOX_PAD = 0.08
KEEP_FRAMES = 2
EXPORT_LOG = resolve("data/campaign/export_log.jsonl")
PERSON, TWO_WHEEL, VEHICLE = {0}, {1, 3}, {2, 5, 7}


def blur_regions(frame: np.ndarray, boxes: list[tuple[int, float, float, float, float]]) -> np.ndarray:
    """boxes: (cls, x1, y1, x2, y2) in pixels."""
    out = frame.copy()
    h, w = out.shape[:2]
    for c, x1, y1, x2, y2 in boxes:
        if c in VEHICLE and PLATE_BAND < 1.0:
            y1 = y2 - PLATE_BAND * (y2 - y1)
        px, py = BOX_PAD * (x2 - x1), BOX_PAD * (y2 - y1)
        x1, y1, x2, y2 = x1 - px, y1 - py, x2 + px, y2 + py
        x1, y1, x2, y2 = max(int(x1), 0), max(int(y1), 0), min(int(x2), w), min(int(y2), h)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = out[y1:y2, x1:x2]
        small = cv2.resize(roi, (max(1, (x2 - x1) // 12), max(1, (y2 - y1) // 12)), interpolation=cv2.INTER_AREA)
        out[y1:y2, x1:x2] = cv2.GaussianBlur(cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST), (0, 0), 3)
    return out


def make_shared_copy(session_dir: Path, device: str | None = None, weights: str | None = None, progress=None) -> dict:
    """Blur every full-session segment into <session>/shared/. Returns counts; writes shared/blur_log.json."""
    import imageio.v2 as imageio
    from src.detection.detector import Detector
    from src.utils.config import load_config
    from src.utils.device import get_device
    cfg = load_config(None)
    d = Path(session_dir)
    segs = sorted((d / "video").glob("segment_*.mp4"))
    if not segs:
        raise FileNotFoundError("no full-session video in this session")
    out_dir = d / "shared"
    out_dir.mkdir(exist_ok=True)
    w = resolve(weights or cfg["detection"]["weights"])
    det = Detector(str(w) if w.exists() else Path(cfg["detection"]["weights"]).name, conf=DET_CONF, iou=0.5,
                   classes=sorted(PERSON | TWO_WHEEL | VEHICLE), device=device or get_device(), imgsz=640)
    log = {"created": datetime.now().isoformat(timespec="seconds"), "method": __doc__.split("Blurring")[1].strip()[:900],
           "det_conf": DET_CONF, "plate_band": PLATE_BAND, "box_pad": BOX_PAD, "keep_frames": KEEP_FRAMES, "segments": []}
    for k, seg in enumerate(segs):
        rd = imageio.get_reader(str(seg))
        fps = rd.get_meta_data().get("fps", 10)
        wr = imageio.get_writer(str(out_dir / seg.name), fps=fps, codec="libx264", quality=6, macro_block_size=8, ffmpeg_log_level="error")
        hist: deque = deque(maxlen=2 * KEEP_FRAMES + 1)
        frames: deque = deque()
        n, nb = 0, 0

        def emit():
            nonlocal nb
            fr = frames.popleft()
            boxes = [b for hb in hist for b in hb]
            nb += len(boxes)
            wr.append_data(cv2.cvtColor(blur_regions(fr, boxes), cv2.COLOR_BGR2RGB))
        for rgb in rd:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            hist.append([(x.cls, *x.xyxy) for x in det.detect(bgr, n)])
            frames.append(bgr)
            n += 1
            if len(frames) > KEEP_FRAMES:  # the frame KEEP_FRAMES back now has detections from +-KEEP_FRAMES frames
                emit()
            if progress and n % 100 == 0:
                progress(k, n)
        while frames:
            hist.append([])
            emit()
        rd.close()
        wr.close()
        log["segments"].append({"file": seg.name, "frames": n, "blurred_boxes": nb})
    (out_dir / "blur_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    rv = out_dir / "privacy_review.json"
    if rv.exists():
        rv.unlink()  # a new copy invalidates any earlier review
    if (out_dir / "revocation.json").exists():  # a new copy starts again at pending_review (history stays in revocation_log)
        (out_dir / "revocation.json").unlink()
    return log


REVIEW_SCHEMA = "v3.4.3"
CAMPAIGN_CAMERA_KINDS = ("real_driving", "controlled_test", "live_camera_test")  # demo_video / Nexar never validates the blur
SHARING_POLICY_PATH = resolve("data/campaign/sharing_policy.json")
REVOCATION_LOG = resolve("data/campaign/revocation_log.jsonl")
REVIEW_LOG = resolve("data/campaign/privacy_review_log.jsonl")  # v3.4.4: append-only log of every review decision


def _seg_hashes(shared_dir: Path) -> dict:
    from src.realworld.provenance import sha256_file
    return {p.name: sha256_file(p) for p in sorted(Path(shared_dir).glob("segment_*.mp4"))}


def review(session_dir: Path, reviewer: str, approved: bool, segments_checked: list[str], notes: str = "",
           identifiable_found: bool | None = None, access_path: Path | None = None, recorded_parked: bool | None = None,
           log_path: Path | None = None) -> dict:
    """Record the HUMAN privacy review of a shared copy (v3.4.3).
    * only a coordinator may review;
    * approval requires every segment to be checked AND the explicit finding that no face / plate is readable
      (identifiable_found=False); a copy with readable faces or plates is recorded as REJECTED and can never be exported;
    * the SHA-256 of every reviewed segment is stored, so any later change of the copy makes the review stale."""
    d = Path(session_dir) / "shared"
    segs = sorted(p.name for p in d.glob("segment_*.mp4"))
    if not segs:
        raise FileNotFoundError("no shared copy to review")
    if not reviewer.strip():
        raise ValueError("reviewer name required")
    if not is_coordinator(reviewer, access_path or ACCESS_PATH):
        raise PermissionError("only a coordinator listed in data/campaign/access.json can review a shared copy")
    if identifiable_found is None and approved:
        raise ValueError("approval needs the explicit finding that no face or licence plate is readable")
    if identifiable_found:
        approved = False  # readable faces / plates -> rejected, whatever else was ticked
    if approved and set(segments_checked) != set(segs):
        raise ValueError("approval requires every shared segment to be checked")
    r = {"schema": REVIEW_SCHEMA, "reviewer": reviewer, "approved": bool(approved),
         "decision": "approved" if approved else "rejected", "identifiable_found": identifiable_found,
         "segments_checked": sorted(segments_checked), "segments": segs, "segments_sha256": _seg_hashes(d),
         "notes": notes, "reviewed_at": datetime.now().isoformat(timespec="seconds"), "recorded_parked": recorded_parked}
    (d / "privacy_review.json").write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    lp = Path(log_path) if log_path else REVIEW_LOG
    lp.parent.mkdir(parents=True, exist_ok=True)
    with open(lp, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"session": Path(session_dir).name, **{k: r[k] for k in ("reviewer", "reviewed_at", "decision",
                            "identifiable_found", "segments_sha256", "recorded_parked", "notes")}}, ensure_ascii=False) + "\n")
    return r


def load_review(session_dir: Path) -> dict:
    p = Path(session_dir) / "shared" / "privacy_review.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except json.JSONDecodeError:
        return {}


def review_log(log_path: Path | None = None) -> list[dict]:
    lp = Path(log_path) if log_path else REVIEW_LOG
    return [json.loads(x) for x in lp.read_text(encoding="utf-8").splitlines() if x.strip()] if lp.exists() else []


def shared_status(session_dir: Path, access_path: Path | None = None) -> str:
    """no_shared_copy | pending_review | approved | rejected | stale (copy changed after the review, old review format, or the
    reviewer is no longer a coordinator) | revoked"""
    d = Path(session_dir) / "shared"
    segs = sorted(p.name for p in d.glob("segment_*.mp4")) if d.exists() else []
    if not segs:
        return "no_shared_copy"
    if (d / "revocation.json").exists():
        return "revoked"
    rp = d / "privacy_review.json"
    if not rp.exists():
        return "pending_review"
    r = json.loads(rp.read_text(encoding="utf-8"))
    if not r.get("approved"):
        return "rejected"
    if r.get("segments") != segs or r.get("segments_sha256") != _seg_hashes(d):
        return "stale"
    if not is_coordinator(r.get("reviewer", ""), access_path or ACCESS_PATH):
        return "stale"
    return "approved"


def sharing_policy(path: Path | None = None) -> dict:
    p = Path(path) if path else SHARING_POLICY_PATH
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"sharing_enabled": True, "decided_by": None}


def set_sharing_policy(enabled: bool, coordinator: str, reason: str, path: Path | None = None,
                       access_path: Path | None = None) -> dict:
    """Coordinator decision. sharing_enabled=False is the written 'nothing is shared' decision: export_shared refuses everything."""
    if not is_coordinator(coordinator, access_path or ACCESS_PATH):
        raise PermissionError("only a coordinator can change the sharing policy")
    if not reason.strip():
        raise ValueError("a reason is required")
    p = Path(path) if path else SHARING_POLICY_PATH
    old = sharing_policy(p)
    hist = old.get("history", []) + ([{k: old.get(k) for k in ("sharing_enabled", "decided_by", "decided_at", "reason")}]
                                     if old.get("decided_by") else [])
    rec = {"sharing_enabled": bool(enabled), "decided_by": coordinator, "decided_at": datetime.now().isoformat(timespec="seconds"),
           "reason": reason.strip(), "history": hist}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    return rec


def revoke_shared(session_dir: Path, coordinator: str, reason: str, delete_exports: bool = False, log_path: Path | None = None,
                  export_log: Path | None = None, access_path: Path | None = None) -> dict:
    """Withdraw a shared copy (e.g. the blur was found insufficient after export). Blocks any further export of this copy, lists
    every earlier export of the session and, with delete_exports=True, deletes the exported folders that are still reachable and
    carry a matching export_manifest.json. Raw video is never touched. Copies already passed on to other people cannot be recalled
    by software: the log tells the coordinator whom to contact."""
    import shutil
    if not is_coordinator(coordinator, access_path or ACCESS_PATH):
        raise PermissionError("only a coordinator can revoke a shared copy")
    if not reason.strip():
        raise ValueError("a reason is required")
    d = Path(session_dir)
    sh = d / "shared"
    if not sh.exists():
        raise FileNotFoundError("no shared copy")
    el = Path(export_log) if export_log else EXPORT_LOG
    exports = [json.loads(x) for x in el.read_text(encoding="utf-8").splitlines() if x.strip()] if el.exists() else []
    exports = [e for e in exports if e.get("session_id") == d.name]
    deleted, unreachable = [], []
    if delete_exports:
        for e in exports:
            out = Path(e["destination"])
            mf = out / "export_manifest.json"
            if mf.exists() and json.loads(mf.read_text(encoding="utf-8")).get("session_id") == d.name:
                shutil.rmtree(out)
                deleted.append(str(out))
            else:
                unreachable.append(e["destination"])
    rec = {"session_id": d.name, "revoked_at": datetime.now().isoformat(timespec="seconds"), "by": coordinator, "reason": reason.strip(),
           "earlier_exports": [e["destination"] for e in exports], "deleted_exports": deleted, "unreachable_exports": unreachable}
    (sh / "revocation.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    lp = Path(log_path) if log_path else REVOCATION_LOG
    lp.parent.mkdir(parents=True, exist_ok=True)
    with open(lp, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + chr(10))
    return rec


def load_access(path: Path = ACCESS_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {"raw_access": [], "coordinators": []}


def can_view_raw(name: str, path: Path = ACCESS_PATH) -> bool:
    return bool(name) and name in load_access(path).get("raw_access", [])


def is_coordinator(name: str, path: Path = ACCESS_PATH) -> bool:
    return bool(name) and name in load_access(path).get("coordinators", [])


def shareable(session_dir: Path, access_path: Path | None = None, policy_path: Path | None = None) -> bool:
    """True only for an approved, unchanged, unrevoked copy while sharing is enabled."""
    return sharing_policy(policy_path).get("sharing_enabled", True) and shared_status(session_dir, access_path) == "approved"


class ExportRefused(PermissionError):
    pass


def export_shared(session_dir: Path, dest: Path, coordinator: str, log_path: Path | None = None,
                  policy_path: Path | None = None) -> dict:
    """The ONLY sanctioned way to take video out of a session for sharing: copies the APPROVED blurred copy (never raw video),
    verifies SHA-256 of every copied file and logs the export (ids, files, hashes; no content).
    Refused when: the caller is not a coordinator, the session has no shared copy, the review is missing / rejected, or the shared
    files changed after the review (hash mismatch with the reviewed segments list)."""
    import shutil
    from src.realworld.provenance import sha256_file
    d = Path(session_dir)
    if not is_coordinator(coordinator):
        raise ExportRefused("only a coordinator listed in data/campaign/access.json can export")
    if not sharing_policy(policy_path).get("sharing_enabled", True):
        raise ExportRefused("sharing is disabled by the coordinator's no-sharing decision (data/campaign/sharing_policy.json)")
    status = shared_status(d)
    if status == "stale":
        raise ExportRefused("the shared copy changed after the review; review again")
    if status != "approved":
        raise ExportRefused({"no_shared_copy": "no shared copy", "pending_review": "no approved privacy review for this session's "
                             "shared copy (pending_review)", "rejected": "the privacy review REJECTED this copy (readable faces / "
                             "plates or incomplete)", "revoked": "this shared copy was revoked"}.get(status, status))
    rv = json.loads((d / "shared" / "privacy_review.json").read_text(encoding="utf-8"))
    segs = sorted(p.name for p in (d / "shared").glob("segment_*.mp4"))
    out = Path(dest) / d.name
    if (Path(dest).resolve() / "").as_posix().startswith((d.resolve() / "").as_posix()):
        raise ExportRefused("destination inside the session folder")
    out.mkdir(parents=True, exist_ok=False)
    files = {}
    for name in segs + ["privacy_review.json", "blur_log.json"]:
        src = d / "shared" / name
        if src.exists():
            shutil.copy2(src, out / name)
            h1, h2 = sha256_file(src), sha256_file(out / name)
            if h1 != h2:
                raise ExportRefused(f"copy verification failed for {name}")
            files[name] = h1
    rec = {"session_id": d.name, "exported_at": datetime.now().isoformat(timespec="seconds"), "by": coordinator,
           "destination": str(out), "files_sha256": files, "reviewer": rv["reviewer"]}
    (out / "export_manifest.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    lp = Path(log_path) if log_path else EXPORT_LOG
    lp.parent.mkdir(parents=True, exist_ok=True)
    with open(lp, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec
