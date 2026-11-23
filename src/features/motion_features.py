"""Track-based motion features on the image plane.

IMPORTANT: all quantities are *apparent* (image-plane) motion in normalised image units
(fraction of frame width/height per second). They are not real-world speeds. Without camera
calibration and depth, no physical TTC is computed. `tau_center` is the scale-based
"apparent time-to-contact" tau = s / (ds/dt) (Lee, 1976), which is only valid under
constant closing speed and is used purely as a learned input feature.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from src.detection.detector import TWO_WHEEL, VEHICLE
from src.features.optical_flow import FLOW_FEATURES

OBJECT_FEATURES = [
    "n_objects", "n_vehicles", "n_persons", "n_two_wheelers",
    "max_area", "sum_area",
    "looming_center", "looming_all", "tau_center",
    "lead_offset_x", "lead_bottom_y", "lead_area", "lead_lateral_in",
    "max_lateral_in", "max_speed", "mean_speed", "max_accel",
    "min_pair_dist", "max_pair_iou", "max_pair_closing",
    "n_new_tracks", "n_lost_tracks",
]
SCENE_FEATURES = ["brightness", "mean_det_conf"]
MOTION_FEATURES = OBJECT_FEATURES + FLOW_FEATURES + SCENE_FEATURES

FEATURE_DESCRIPTIONS = {
    "n_objects": "number of tracked road users", "n_vehicles": "number of cars/buses/trucks",
    "n_persons": "number of pedestrians", "n_two_wheelers": "number of bicycles/motorcycles",
    "max_area": "largest object size (fraction of frame)", "sum_area": "total object area",
    "looming_center": "fastest growth of an object in the driving corridor (1/s)",
    "looming_all": "fastest object growth anywhere in view (1/s)",
    "tau_center": "apparent time-to-contact of the corridor object (s, image-based)",
    "lead_offset_x": "horizontal offset of the largest object from image centre",
    "lead_bottom_y": "vertical position of the largest object's bottom edge (proximity proxy)",
    "lead_area": "size of the largest object", "lead_lateral_in": "largest object's lateral motion toward the centre",
    "max_lateral_in": "fastest lateral motion toward the image centre",
    "max_speed": "fastest apparent object motion", "mean_speed": "mean apparent object motion",
    "max_accel": "largest apparent acceleration", "min_pair_dist": "smallest distance between two objects",
    "max_pair_iou": "largest overlap between two objects", "max_pair_closing": "fastest closing rate between two objects",
    "n_new_tracks": "objects entering view", "n_lost_tracks": "objects leaving view / lost",
    "flow_mag_mean": "global optical-flow magnitude (ego-motion proxy)",
    "flow_mag_center": "optical-flow magnitude in the road ahead",
    "flow_div_center": "flow expansion in the road ahead (approach proxy)",
    "flow_dx_median": "median horizontal flow (turning proxy)",
    "brightness": "mean frame brightness", "mean_det_conf": "mean detector confidence",
}

TAU_CAP = 10.0


@dataclass
class _Obs:
    step: int
    cx: float
    cy: float
    s: float      # linear size sqrt(w*h)
    y2: float
    cls: int


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


class MotionFeatureComputer:
    def __init__(self, fps: float = 10.0, k: int = 3, min_area: float = 0.002):
        self.fps, self.k, self.min_area = fps, k, min_area
        self.hist: dict[int, deque[_Obs]] = defaultdict(lambda: deque(maxlen=2 * k + 1))
        self.prev_ids: set[int] = set()
        self.prev_pair_dist: dict[tuple[int, int], float] = {}

    def reset(self) -> None:
        self.hist.clear()
        self.prev_ids = set()
        self.prev_pair_dist = {}

    def _vel(self, h: deque[_Obs], lag: int) -> tuple[float, float, float] | None:
        """(vx, vy, scale growth rate) over `lag` steps, or None if not enough history."""
        if len(h) <= lag or h[-1].step - h[-1 - lag].step > 2 * lag:
            return None
        a, b = h[-1 - lag], h[-1]
        dt = (b.step - a.step) / self.fps
        if dt <= 0:
            return None
        growth = (b.s - a.s) / max(a.s, 1e-4) / dt
        return (b.cx - a.cx) / dt, (b.cy - a.cy) / dt, growth

    def update(self, step: int, boxes: np.ndarray, ids: np.ndarray, cls: np.ndarray, conf: np.ndarray,
               flow: np.ndarray, brightness: float) -> np.ndarray:
        """boxes: (n,4) normalised xyxy. Returns a feature vector ordered as MOTION_FEATURES."""
        f = dict.fromkeys(OBJECT_FEATURES, 0.0)
        f["min_pair_dist"], f["tau_center"] = 1.5, TAU_CAP
        n = len(boxes)
        cur_ids: set[int] = set()
        objs = []
        for i in range(n):
            x1, y1, x2, y2 = boxes[i]
            w, hgt = max(x2 - x1, 1e-4), max(y2 - y1, 1e-4)
            o = _Obs(step, (x1 + x2) / 2, (y1 + y2) / 2, float(np.sqrt(w * hgt)), y2, int(cls[i]))
            tid = int(ids[i])
            if tid >= 0:
                self.hist[tid].append(o)
                cur_ids.add(tid)
            objs.append((tid, o, w * hgt, boxes[i]))
        f["n_objects"] = float(n)
        f["n_vehicles"] = float(sum(c in VEHICLE for c in cls))
        f["n_persons"] = float(sum(c == 0 for c in cls))
        f["n_two_wheelers"] = float(sum(c in TWO_WHEEL for c in cls))
        if n:
            areas = np.array([a for _, _, a, _ in objs])
            f["max_area"], f["sum_area"] = float(areas.max()), float(areas.sum())
            speeds, lat_in, loom_all, loom_c, accels = [], [], [], [], []
            for tid, o, area, _ in objs:
                if tid < 0:
                    continue
                v = self._vel(self.hist[tid], self.k)
                if v is None:
                    continue
                vx, vy, g = v
                speeds.append(np.hypot(vx, vy))
                if area >= self.min_area:
                    lat_in.append(-np.sign(o.cx - 0.5) * vx)
                loom_all.append(g)
                if abs(o.cx - 0.5) < 0.2 and o.cy > 0.35:
                    loom_c.append(g)
                h = self.hist[tid]
                if len(h) > 2 * self.k:
                    v0 = self._vel(deque(list(h)[:-self.k], maxlen=h.maxlen), self.k)
                    if v0 is not None:
                        accels.append(np.hypot(vx - v0[0], vy - v0[1]) / (self.k / self.fps))
            if speeds:
                f["max_speed"], f["mean_speed"] = float(np.max(speeds)), float(np.mean(speeds))
            if lat_in:
                f["max_lateral_in"] = float(np.max(lat_in))
            if loom_all:
                f["looming_all"] = float(np.clip(np.max(loom_all), -5, 5))
            if loom_c:
                g = float(np.clip(np.max(loom_c), -5, 5))
                f["looming_center"] = g
                f["tau_center"] = float(min(1.0 / g, TAU_CAP)) if g > 1.0 / TAU_CAP else TAU_CAP
            if accels:
                f["max_accel"] = float(np.clip(np.max(accels), 0, 50))
            li = int(np.argmax(areas))
            tid, o, area, _ = objs[li]
            f["lead_offset_x"], f["lead_bottom_y"], f["lead_area"] = abs(o.cx - 0.5), o.y2, area
            if tid >= 0:
                v = self._vel(self.hist[tid], self.k)
                if v is not None:
                    f["lead_lateral_in"] = float(-np.sign(o.cx - 0.5) * v[0])
            # pairwise relations between road users
            pair_dist: dict[tuple[int, int], float] = {}
            closing = []
            for a in range(n):
                for b in range(a + 1, n):
                    d = float(np.hypot(objs[a][1].cx - objs[b][1].cx, objs[a][1].cy - objs[b][1].cy))
                    f["min_pair_dist"] = min(f["min_pair_dist"], d)
                    f["max_pair_iou"] = max(f["max_pair_iou"], _iou(objs[a][3], objs[b][3]))
                    ta, tb = objs[a][0], objs[b][0]
                    if ta >= 0 and tb >= 0:
                        key = (min(ta, tb), max(ta, tb))
                        pair_dist[key] = d
                        if key in self.prev_pair_dist:
                            closing.append((self.prev_pair_dist[key] - d) * self.fps)
            if closing:
                f["max_pair_closing"] = float(np.clip(np.max(closing), -10, 10))
            self.prev_pair_dist = pair_dist
        f["n_new_tracks"] = float(len(cur_ids - self.prev_ids))
        f["n_lost_tracks"] = float(len(self.prev_ids - cur_ids))
        self.prev_ids = cur_ids
        obj_vec = np.array([f[k] for k in OBJECT_FEATURES], np.float32)
        scene = np.array([brightness, float(np.mean(conf)) if len(conf) else 0.0], np.float32)
        return np.concatenate([obj_vec, flow.astype(np.float32), scene])
