"""Multi-object tracking (ByteTrack / BoT-SORT through Ultralytics) + track-quality statistics."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.detection.detector import COCO_ROAD, TWO_WHEEL, VEHICLE, Detection, load_yolo, parse_boxes

# ---- per-track class stabilisation (v3.4.2) ----
# YOLO re-classifies the same object every frame, so a tracked truck can flicker truck -> bus -> car. Each track keeps a
# confidence-weighted, exponentially decayed vote per class; the shown class changes only when another class clearly wins.
# Mode "within_group" (default) only re-labels INSIDE the detection's feature group (vehicle {car, bus, truck}, two-wheeler
# {bicycle, motorcycle}, person): the model's features use only group counts, so they are provably unchanged and the frozen
# model / features / thresholds stay valid. Cross-group flips (e.g. car <-> person, 2.2 % of flips on held-out data) are left
# as detected: changing them would change the model input and needs a new evaluation + frozen reference.
CLASS_DECAY = 0.90          # per update; vote half-life ~6.6 steps (0.66 s at 10 Hz); chosen on 150 DEV videos
CLASS_SWITCH_MARGIN = 2.0   # a new class must have >= 2x the current vote: a persistent true change is accepted after ~1.0 s
CLASS_FORGET_STEPS = 50     # drop a track's votes after 5 s unseen


def class_group(c: int) -> str:
    return "vehicle" if c in VEHICLE else ("two_wheeler" if c in TWO_WHEEL else ("person" if c == 0 else f"other{c}"))


class ClassStabilizer:
    def __init__(self, decay: float = CLASS_DECAY, margin: float = CLASS_SWITCH_MARGIN, forget: int = CLASS_FORGET_STEPS,
                 mode: str = "within_group"):
        if mode not in ("within_group", "off"):
            raise ValueError("mode must be within_group or off (cross-group smoothing changes model features)")
        self.decay, self.margin, self.forget, self.mode = decay, margin, forget, mode
        self.state: dict[int, dict] = {}

    def reset(self) -> None:
        self.state.clear()

    def __call__(self, dets: list[Detection], frame_index: int) -> list[Detection]:
        if self.mode == "off":
            return dets
        for d in dets:
            if d.track_id < 0:
                continue
            st = self.state.get(d.track_id)
            g = class_group(d.cls)
            if st is None or st["group"] != g:  # new track, or a cross-group flip: restart votes in the new group
                st = {"group": g, "votes": {}, "shown": d.cls, "last": frame_index}
                self.state[d.track_id] = st
            gap = max(1, frame_index - st["last"])
            for k in st["votes"]:
                st["votes"][k] *= self.decay ** gap
            st["votes"][d.cls] = st["votes"].get(d.cls, 0.0) + max(d.conf, 1e-3)
            st["last"] = frame_index
            best = max(st["votes"], key=st["votes"].get)
            if best != st["shown"] and st["votes"][best] >= self.margin * st["votes"].get(st["shown"], 0.0):
                st["shown"] = best
            if st["shown"] != d.cls:
                d.cls, d.name = st["shown"], COCO_ROAD.get(st["shown"], str(st["shown"]))
        stale = [k for k, v in self.state.items() if frame_index - v["last"] > self.forget]
        for k in stale:
            del self.state[k]
        return dets


class Tracker:
    """Detect + track frame by frame. Call `reset()` between videos."""

    def __init__(self, weights: str = "yolov8n.pt", tracker: str = "bytetrack.yaml", conf: float = 0.25,
                 iou: float = 0.5, classes: list[int] | None = None, device: str = "cpu", imgsz: int = 640):
        self.model = load_yolo(weights)
        self.tracker_cfg, self.conf, self.iou, self.device, self.imgsz = tracker, conf, iou, device, imgsz
        self.classes = classes or list(COCO_ROAD)
        self.stabilizer = ClassStabilizer()

    def reset(self) -> None:
        self.stabilizer.reset()
        pred = getattr(self.model, "predictor", None)
        if pred is not None and getattr(pred, "trackers", None):
            for t in pred.trackers:
                t.reset()

    def update(self, frame: np.ndarray, frame_index: int) -> list[Detection]:
        res = self.model.track(frame, persist=True, tracker=self.tracker_cfg, conf=self.conf, iou=self.iou,
                               classes=self.classes, device=self.device, imgsz=self.imgsz, verbose=False)[0]
        return self.stabilizer(parse_boxes(res, frame_index), frame_index)


def track_statistics(tracks: np.ndarray) -> dict:
    """Quality statistics from a (N, 8) [step, id, cls, conf, x1, y1, x2, y2] track array.

    Without MOT ground truth, ID switches cannot be measured; we report fragmentation proxies:
    track length distribution and the number of gaps (track re-appears after missing steps)."""
    if len(tracks) == 0:
        return {"n_tracks": 0, "mean_length": 0.0, "n_gaps": 0, "untracked_fraction": 0.0}
    by_id: dict[int, list[int]] = defaultdict(list)
    untracked = 0
    for row in tracks:
        if row[1] < 0:
            untracked += 1
            continue
        by_id[int(row[1])].append(int(row[0]))
    lengths = [len(v) for v in by_id.values()]
    gaps = sum(int(np.sum(np.diff(sorted(v)) > 1)) for v in by_id.values())
    return {
        "n_tracks": len(by_id),
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "median_length": float(np.median(lengths)) if lengths else 0.0,
        "n_gaps": int(gaps),
        "untracked_fraction": float(untracked / len(tracks)),
    }
