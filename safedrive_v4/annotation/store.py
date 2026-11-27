"""Append-only annotation store with a hash-chained operation log, restore, submission lock, blindness and adjudication.

Layout (inside the encrypted session volume, next to the event store):
  <root>/ops/<video>/<annotator>.jsonl        create | update | retract | restore operations (hash chain)
  <root>/evidence/<video>/<sha256>.jpg        evidence frames (content-addressed)
  <root>/submissions.jsonl                    submissions (annotator, video, records hash)
  <root>/adjudication/<video>.jsonl           coordinator decisions (originals are never changed)
Nothing is ever deleted or rewritten: a retract marks a record inactive, a restore appends the earlier version again.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from safedrive_v4.annotation.schema import RECORD_TYPES, SCHEMA, ValidationError, new_id, now


class AccessError(PermissionError):
    pass


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def default_root() -> tuple[Path, bool]:
    from safedrive_v4.event_memory.store import default_root as ev_root
    root, enc = ev_root()
    return root.parent / "safedrive_v4_annotation", enc


class AnnotationStore:
    def __init__(self, root: str | Path | None = None, require_encrypted: bool = True, access_path: Path | None = None):
        if root is None:
            root, enc = default_root()
            if require_encrypted and not enc:
                raise AccessError("the encrypted session volume is not mounted: annotation data is never written unencrypted")
        self.root = Path(root)
        self.access_path = access_path
        (self.root / "ops").mkdir(parents=True, exist_ok=True)
        (self.root / "evidence").mkdir(parents=True, exist_ok=True)
        (self.root / "adjudication").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ roles
    def _access(self):
        from src.realworld import privacy
        return privacy.load_access(self.access_path or privacy.ACCESS_PATH)

    def is_coordinator(self, name: str) -> bool:
        return name in self._access().get("coordinators", [])

    def check_annotator(self, name: str, campaign_video: bool = False) -> None:
        if not name.strip() or any(ch.isspace() for ch in name.strip()):
            raise AccessError("annotator pseudonym required (no spaces, no real names)")
        if campaign_video and name not in self._access().get("raw_access", []):
            raise AccessError("campaign (raw) video: only annotators on the raw-access list")

    # ------------------------------------------------------------------ op log
    def _log(self, video: str, annotator: str) -> Path:
        safe = "".join(c for c in annotator if c.isalnum() or c in "-_")
        p = self.root / "ops" / video / f"{safe}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def ops(self, video: str, annotator: str) -> list[dict]:
        p = self._log(video, annotator)
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []

    def _append(self, video: str, annotator: str, op: dict) -> dict:
        prev = self.ops(video, annotator)
        op = {**op, "at": now(), "annotator": annotator, "schema": SCHEMA, "seq": len(prev),
              "prev_hash": prev[-1]["hash"] if prev else None}
        op["hash"] = _h({k: v for k, v in op.items() if k != "hash"})
        with open(self._log(video, annotator), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(op, ensure_ascii=False, default=str) + "\n")
        return op

    def verify_chain(self, video: str, annotator: str) -> dict:
        prev, bad = None, []
        for o in self.ops(video, annotator):
            if o.get("prev_hash") != prev or o["hash"] != _h({k: v for k, v in o.items() if k != "hash"}):
                bad.append(o.get("seq"))
            prev = o["hash"]
        return {"ok": not bad, "bad_seq": bad}

    # ------------------------------------------------------------------ records
    def put(self, rtype: str, record, reason: str = "") -> dict:
        record.validate()
        if self.submitted(record.video_id, record.annotator) and not reason.strip():
            raise ValidationError("this video is submitted: a revision needs a reason (it is recorded in the history)")
        cur = self.state(record.video_id, record.annotator)
        op = "update" if record.record_id in cur else "create"
        return self._append(record.video_id, record.annotator, {"op": op, "type": rtype, "record_id": record.record_id,
                                                                "record": asdict(record), "reason": reason})

    def retract(self, video: str, annotator: str, record_id: str, reason: str) -> dict:
        if len(reason.strip()) < 3:
            raise ValidationError("a reason is required to retract a record")
        if record_id not in self.state(video, annotator):
            raise ValidationError("unknown or already retracted record")
        return self._append(video, annotator, {"op": "retract", "record_id": record_id, "reason": reason})

    def restore(self, video: str, annotator: str, record_id: str, seq: int, reason: str) -> dict:
        """Re-append the version of `record_id` written at op `seq` (history is kept; nothing is overwritten)."""
        old = next((o for o in self.ops(video, annotator) if o["seq"] == seq and o.get("record_id") == record_id and "record" in o), None)
        if old is None:
            raise ValidationError("no such version")
        return self._append(video, annotator, {"op": "restore", "type": old["type"], "record_id": record_id, "record": old["record"],
                                               "reason": reason or f"restore of seq {seq}"})

    def state(self, video: str, annotator: str, upto_seq: int | None = None) -> dict:
        cur: dict = {}
        for o in self.ops(video, annotator):
            if upto_seq is not None and o["seq"] > upto_seq:
                break
            if o["op"] in ("create", "update", "restore"):
                cur[o["record_id"]] = {"type": o["type"], **o["record"]}
            elif o["op"] == "retract":
                cur.pop(o["record_id"], None)
        return cur

    def history(self, video: str, annotator: str, record_id: str) -> list[dict]:
        return [o for o in self.ops(video, annotator) if o.get("record_id") == record_id]

    def records(self, video: str, annotator: str, rtype: str | None = None) -> list:
        out = []
        for r in self.state(video, annotator).values():
            if rtype and r["type"] != rtype:
                continue
            cls = RECORD_TYPES[r["type"]]
            out.append(cls(**{k: v for k, v in r.items() if k in cls.__dataclass_fields__}))
        return out

    def annotators(self, video: str) -> list[str]:
        d = self.root / "ops" / video
        return sorted(p.stem for p in d.glob("*.jsonl")) if d.exists() else []

    def videos(self) -> list[str]:
        return sorted(p.name for p in (self.root / "ops").iterdir() if p.is_dir())

    # ------------------------------------------------------------------ evidence
    def save_evidence(self, video: str, frame_bgr, t: float, frame_index: int, track_id=None, box=None, note="") -> dict:
        import cv2
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValidationError("could not encode the evidence frame")
        b = buf.tobytes()
        sha = hashlib.sha256(b).hexdigest()
        p = self.root / "evidence" / video / f"{sha}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_bytes(b)
        return {"video_id": video, "t": float(t), "frame_index": int(frame_index), "image_sha256": sha,
                "track_id": track_id, "box": box, "note": note}

    def evidence_path(self, video: str, sha: str) -> Path:
        return self.root / "evidence" / video / f"{sha}.jpg"

    def verify_evidence(self, ref: dict) -> bool:
        p = self.evidence_path(ref["video_id"], ref["image_sha256"])
        return p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == ref["image_sha256"]

    # ------------------------------------------------------------------ submission / blindness
    def submit(self, video: str, annotator: str) -> dict:
        st = self.state(video, annotator)
        if not any(r["type"] == "event" for r in st.values()):
            raise ValidationError("submit at least one temporal record (use 'normal' for a reviewed clip without events)")
        rec = {"video_id": video, "annotator": annotator, "at": now(), "records_sha256": _h(st), "n_records": len(st)}
        with open(self.root / "submissions.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec

    def submissions(self) -> list[dict]:
        p = self.root / "submissions.jsonl"
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []

    def submitted(self, video: str, annotator: str) -> bool:
        return any(s["video_id"] == video and s["annotator"] == annotator for s in self.submissions())

    def view_other(self, video: str, viewer: str, other: str) -> dict:
        """Blindness: another annotator's records are visible only to a coordinator, and only after BOTH submitted."""
        if not self.is_coordinator(viewer):
            raise AccessError("annotators never see another annotator's labels")
        if not (self.submitted(video, other) and len([a for a in self.annotators(video) if self.submitted(video, a)]) >= 2):
            raise AccessError("visible only after both annotators submitted")
        return self.state(video, other)

    # ------------------------------------------------------------------ adjudication
    def adjudicate(self, video: str, coordinator: str, decision: str, refs: dict, record=None, note: str = "") -> dict:
        """decision: accept_A | accept_B | merged | unresolved. refs: {annotator: record_id or None}. Originals are untouched."""
        if not self.is_coordinator(coordinator):
            raise AccessError("only a coordinator adjudicates")
        subs = [a for a in self.annotators(video) if self.submitted(video, a)]
        if len(subs) < 2:
            raise AccessError("adjudication needs two submitted annotators")
        if decision not in ("accept_A", "accept_B", "merged", "unresolved"):
            raise ValidationError("invalid decision")
        if decision != "unresolved":
            if record is None:
                raise ValidationError("an adjudicated record is required unless unresolved")
            record.validate()
        rec = {"adj_id": new_id("adj"), "video_id": video, "coordinator": coordinator, "at": now(), "decision": decision,
               "refs": refs, "record": asdict(record) if record is not None else None, "record_type": type(record).__name__ if record else None,
               "note": note}
        with open(self.root / "adjudication" / f"{video}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        return rec

    def adjudications(self, video: str) -> list[dict]:
        p = self.root / "adjudication" / f"{video}.jsonl"
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []
