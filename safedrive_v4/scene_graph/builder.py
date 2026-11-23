"""Incremental Dynamic Traffic Scene Graph builder.

At each step the builder receives the tracked detections of that step (from the v3 YOLO + ByteTrack pipeline or its cache)
and the camera transform of that step, and returns a Snapshot. State is bounded (per-track deque of `history_steps`, per-edge
persistence deque of `persistence_n`), so memory does not grow with video length, and only the current step's tracks are
processed (no full rebuild).

Relations are emitted only with sufficient evidence:
  * the involved track must be older than k steps (a velocity estimate exists) and above `min_conf`;
  * the triggering condition must hold in >= m of the last n steps (temporal validity), including the current step;
  * every edge stores the numeric evidence that triggered it.
OPPOSITE_DIRECTION is disabled by default: a monocular image gives no reliable cue for heading (documented in the scene-graph doc).
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterator

import numpy as np

from safedrive_v4.scene_graph.schema import (COCO_TO_TYPE, EGO_ID, REGION_IDS, TWO_WHEELER_TYPES, VEHICLE_TYPES, Edge, Node,
                                              Snapshot)

SCENE_COLUMNS = ("flow_mag_mean", "flow_dx_median", "brightness", "mean_det_conf")


def _apply(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ M.T
    return p[:, :2] / np.where(np.abs(p[:, 2:3]) < 1e-9, 1e-9, p[:, 2:3])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / u) if u > 0 else 0.0


class SceneGraphBuilder:
    def __init__(self, cfg: dict, fps: float = 10.0):
        self.c = cfg["scene_graph"] if "scene_graph" in cfg else cfg
        self.fps = float(fps)
        self.k = int(self.c["k_steps"])
        self.reset()

    def reset(self) -> None:
        self.hist: dict[int, deque] = {}                 # tid -> deque[(step, t, box, conf, cls)]
        self.vel_hist: dict[int, deque] = {}             # tid -> deque[(step, v, g)]
        self.transforms: deque = deque(maxlen=int(self.c["history_steps"]) + 1)  # (step, M_{step-1->step})
        self.persist: dict[tuple[str, str, str], deque] = {}
        self.evid: dict[tuple[str, str, str], deque] = {}   # (hit, reliability weight) over opinion_window steps
        self.first_seen: dict[tuple[str, str, str], float] = {}
        self.last_step: int | None = None

    # ------------------------------------------------------------------ kinematics
    def _accumulated(self, step: int, lag: int) -> np.ndarray:
        A = np.eye(3)
        for s, M in self.transforms:
            if step - lag < s <= step:
                A = M @ A
        return A

    def _kinematics(self, tid: int, step: int, box: np.ndarray) -> tuple[np.ndarray | None, float | None]:
        h = self.hist.get(tid)
        past = next((x for x in h if x[0] == step - self.k), None) if h else None
        if past is None:
            return None, None
        pb = past[2]
        quad = np.array([[pb[0], pb[1]], [pb[2], pb[1]], [pb[2], pb[3]], [pb[0], pb[3]]], float)
        q = _apply(self._accumulated(step, self.k), quad)
        c_hat = q.mean(0)
        a_hat = 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], 1)) - np.dot(q[:, 1], np.roll(q[:, 0], 1)))
        c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        area = max(box[2] - box[0], 1e-4) * max(box[3] - box[1], 1e-4)
        dt = self.k / self.fps
        v = (c - c_hat) / dt
        g = (np.sqrt(area) / max(np.sqrt(max(a_hat, 1e-8)), 1e-4) - 1.0) / dt
        return v, float(np.clip(g, -5.0, 5.0))

    # ------------------------------------------------------------------ persistence
    def _persist(self, key: tuple[str, str, str], hit: bool, t: float) -> tuple[bool, dict]:
        n, m = int(self.c["persistence_n"]), int(self.c["persistence_m"])
        d = self.persist.setdefault(key, deque(maxlen=n))
        d.append(bool(hit))
        if hit and key not in self.first_seen:
            self.first_seen[key] = t
        support = sum(d)
        valid = hit and support >= m
        return valid, {"first_seen": self.first_seen.get(key, t), "last_seen": t, "support": float(support), "window_n": float(n)}

    def _region(self, cx: float, cy: float) -> str | None:
        c = self.c
        dx = cx - 0.5
        if abs(dx) < c["corridor_half_width"] and cy > c["corridor_top_y"]:
            return REGION_IDS[0]
        if c["corridor_half_width"] <= abs(dx) < c["adjacent_outer"] and cy > c["corridor_top_y"] - 0.05:
            return REGION_IDS[1] if dx < 0 else REGION_IDS[2]
        return None

    # ------------------------------------------------------------------ main update
    def update(self, step: int, t: float, dets: np.ndarray, M: np.ndarray | None = None, scene: dict | None = None) -> Snapshot:
        """dets: (n, 7) [track_id, coco_cls, conf, x1, y1, x2, y2] normalised. M: camera transform (step-1 -> step) or None."""
        c = self.c
        self.transforms.append((step, np.eye(3) if M is None else np.asarray(M, float)))
        scene = dict(scene or {})
        nodes: dict[str, Node] = {}
        ego = Node(EGO_ID, "EgoVehicle", t, attrs={"forward_motion_proxy": scene.get("flow_mag_mean"),
                                                   "turn_proxy": scene.get("flow_dx_median"), "image_based": True})
        nodes[EGO_ID] = ego
        for rid in REGION_IDS:
            nodes[rid] = Node(rid, "RoadRegion", t, attrs={"definition": "fixed image region (no lane detection)"})
        live = set()
        objs = []
        for row in np.asarray(dets, float).reshape(-1, 7):
            tid, cls, conf = int(row[0]), int(row[1]), float(row[2])
            if tid < 0 or cls not in COCO_TO_TYPE:
                continue
            box = row[3:7].copy()
            live.add(tid)
            h = self.hist.setdefault(tid, deque(maxlen=int(c["history_steps"])))
            v, g = self._kinematics(tid, step, box)
            h.append((step, t, box, conf, cls))
            vh = self.vel_hist.setdefault(tid, deque(maxlen=int(c["history_steps"])))
            acc = None
            if v is not None:
                prev = next((x for x in vh if x[0] == step - self.k), None)
                if prev is not None:
                    acc = (v - prev[1]) / (self.k / self.fps)
                vh.append((step, v, g))
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            area = max(box[2] - box[0], 1e-4) * max(box[3] - box[1], 1e-4)
            age = len(h)
            speeds = [float(np.hypot(*x[1])) for x in vh][-5:]
            stability = 1.0 / (1.0 + 5.0 * float(np.std(speeds))) if len(speeds) >= 2 else 0.5
            truncated = box[0] < 0.01 or box[2] > 0.99 or box[3] > 0.995
            conf_term = float(np.clip((conf - 0.25) / 0.5, 0, 1))
            age_term = min(age / (self.k + 2), 1.0)
            reliability = conf_term * age_term * stability * (0.8 if truncated else 1.0)
            tau = None
            if g is not None and g > c["approach_growth_min"]:
                tau = float(min(1.0 / g, c["tau_cap_s"]))
            typ = COCO_TO_TYPE[cls]
            n = Node(f"obj:{tid}", typ, t, object_id=tid, class_label=typ.lower(), confidence=conf,
                     bounding_box=tuple(float(x) for x in box), center_point=(float(cx), float(cy)),
                     velocity_estimate=None if v is None else (float(v[0]), float(v[1])), scale_growth=g,
                     acceleration_estimate=None if acc is None else (float(acc[0]), float(acc[1])), tau_s=tau,
                     track_history=[(float(x[1]), float((x[2][0] + x[2][2]) / 2), float((x[2][1] + x[2][3]) / 2)) for x in h],
                     track_age_steps=age, visibility_status="new" if age <= self.k else ("truncated" if truncated else "visible"),
                     uncertainty=float(1.0 - reliability), region=self._region(cx, cy),
                     attrs={"area": float(area), "y2": float(box[3]), "reliability": float(reliability)})
            nodes[n.node_id] = n
            objs.append(n)
        # forget tracks not seen for history_steps (bounded memory)
        for tid in [k for k, hh in self.hist.items() if hh and step - hh[-1][0] > int(c["history_steps"])]:
            self.hist.pop(tid, None)
            self.vel_hist.pop(tid, None)
        # occlusion status (nearer box = lower bottom edge)
        occl_pairs = []
        for i, a in enumerate(objs):
            for b in objs[i + 1:]:
                iou = _iou(np.array(a.bounding_box), np.array(b.bounding_box))
                if iou > c["occlusion_iou"]:
                    front, back = (a, b) if a.attrs["y2"] >= b.attrs["y2"] else (b, a)
                    if back.visibility_status == "visible":
                        back.visibility_status = "partially_occluded"
                        back.uncertainty = float(min(1.0, back.uncertainty + 0.2))
                    occl_pairs.append((front, back, iou))
        candidates: list[tuple[tuple[str, str, str], bool, float, dict]] = []

        def cand(src: str, dst: str, rel: str, hit: bool, strength: float, ev: dict) -> None:
            candidates.append(((src, dst, rel), hit, strength, ev))

        corridor = [o for o in objs if o.region == REGION_IDS[0]]
        lead = max(corridor, key=lambda o: o.attrs["y2"]) if corridor else None
        for o in objs:
            ok = o.velocity_estimate is not None and (o.confidence or 0) >= c["min_conf"] and o.attrs["area"] >= c["min_area"]
            if o.region:
                cand(o.node_id, o.region, "LOCATED_IN", True, 1.0, {"cx": o.center_point[0], "cy": o.center_point[1]})
            if not ok:
                continue
            vx, vy = o.velocity_estimate
            g = o.scale_growth or 0.0
            dx = o.center_point[0] - 0.5
            lat_in = float(-np.sign(dx) * vx)
            in_corr = o.region == REGION_IDS[0]
            in_adj = o.region in REGION_IDS[1:]
            cand(o.node_id, EGO_ID, "APPROACHING", (in_corr or in_adj) and g > c["approach_growth_min"],
                 g / (2 * c["approach_growth_min"]), {"scale_growth": g, "tau_s": o.tau_s or c["tau_cap_s"]})
            cand(o.node_id, EGO_ID, "MOVING_AWAY", (in_corr or in_adj) and g < -c["away_growth_min"],
                 -g / (2 * c["away_growth_min"]), {"scale_growth": g})
            gap_to_corr = abs(dx) - c["corridor_half_width"]
            cut = (in_adj and o.node_type in VEHICLE_TYPES | {"Motorcycle"} and lat_in > c["cutin_lateral_min"]
                   and gap_to_corr < lat_in * c["cutin_horizon_s"])
            cand(o.node_id, EGO_ID, "CUTTING_IN_FRONT_OF", cut, lat_in / (2 * c["cutin_lateral_min"]),
                 {"lateral_in": lat_in, "gap_to_corridor": float(gap_to_corr), "time_to_corridor_s": float(gap_to_corr / max(lat_in, 1e-6))})
            crossing = (o.node_type in {"Pedestrian", "Bicycle"} and (in_corr or in_adj) and abs(vx) > c["crossing_lateral_min"]
                        and (in_corr or lat_in > 0))
            cand(o.node_id, EGO_ID, "CROSSING_PATH", crossing, abs(vx) / (2 * c["crossing_lateral_min"]), {"lateral_speed": float(vx), "lateral_in": lat_in})
            near_ego = o.attrs["y2"] > c["ego_near_bottom_y"] and (in_corr or in_adj)
            cand(o.node_id, EGO_ID, "NEAR", near_ego, o.attrs["y2"], {"bottom_y": o.attrs["y2"], "area": o.attrs["area"]})
            if o is lead:
                cand(EGO_ID, o.node_id, "FOLLOWING", abs(lat_in) < 0.1, 1.0, {"lateral_in": lat_in, "bottom_y": o.attrs["y2"]})
                moving = (scene.get("flow_mag_mean") or 0.0) > c.get("ego_moving_flow_min", 0.01)
                cand(o.node_id, EGO_ID, "SAME_DIRECTION", moving and abs(g) < c["same_direction_growth_max"], 1.0,
                     {"scale_growth": g, "ego_flow": float(scene.get("flow_mag_mean") or 0.0)})
            if c.get("enable_opposite_direction"):
                cand(o.node_id, EGO_ID, "OPPOSITE_DIRECTION", in_adj and dx < 0 and g > 3 * c["approach_growth_min"] and lat_in < 0, 0.5,
                     {"scale_growth": g, "lateral_in": lat_in})
            # trajectory conflict: constant-velocity image prediction reaches the ego zone, or low tau in the corridor
            H = c["conflict_horizon_s"]
            fx, fy = o.center_point[0] + vx * H, o.center_point[1] + vy * H
            reaches = abs(fx - 0.5) < c["corridor_half_width"] and fy > 0.75
            low_tau = in_corr and o.tau_s is not None and o.tau_s < 2.0
            cand(o.node_id, EGO_ID, "POTENTIAL_CONFLICT", (reaches and (in_corr or in_adj)) or low_tau, 1.0,
                 {"pred_x": float(fx), "pred_y": float(fy), "tau_s": o.tau_s or c["tau_cap_s"], "horizon_s": H})
        moving_objs = sorted([o for o in objs if o.velocity_estimate is not None and (o.confidence or 0) >= c["min_conf"]],
                             key=lambda o: -o.attrs["area"])[:6]
        for i, a in enumerate(moving_objs):
            for b in moving_objs[i + 1:]:
                d = np.subtract(b.center_point, a.center_point)
                dist = float(np.linalg.norm(d))
                size = 0.5 * (np.sqrt(a.attrs["area"]) + np.sqrt(b.attrs["area"]))
                rel = dist / max(size, 1e-4)
                closing = float(-np.dot(d, np.subtract(b.velocity_estimate, a.velocity_estimate)) / max(dist, 1e-6))
                ttr = dist / closing if closing > 1e-3 else c["tau_cap_s"]
                cand(a.node_id, b.node_id, "NEAR", rel < c["near_rel_dist"], 1.0, {"relative_distance": rel})
                cand(a.node_id, b.node_id, "INTERACTING_WITH", closing > 0.05 and ttr < c["interaction_ttr_s"], 1.0,
                     {"closing_rate": closing, "time_to_meet_s": float(min(ttr, c["tau_cap_s"]))})
        for front, back, iou in occl_pairs:
            cand(front.node_id, back.node_id, "OCCLUDING", True, iou / (2 * c["occlusion_iou"]), {"iou": iou})
        # node reliability opinions: detector confidence as weighted evidence for "this observation is correct",
        # discounted for unstable motion, partial occlusion and truncation (lost evidence -> uncertainty)
        from safedrive_v4.uncertainty.subjective_logic import Opinion, discount, weighted_observations
        for o in objs:
            hh = list(self.hist[o.object_id])[-int(c.get("opinion_window", 10)):]
            op = Opinion.from_evidence(sum(x[3] for x in hh), sum(1 - x[3] for x in hh), a=0.5)
            speeds = [float(np.hypot(*x[1])) for x in self.vel_hist.get(o.object_id, [])][-5:]
            if len(speeds) >= 2:
                op = discount(op, 1.0 / (1.0 + 5.0 * float(np.std(speeds))))
            if o.visibility_status in ("partially_occluded", "truncated"):
                op = discount(op, 0.7)
            o.attrs["reliability_opinion"] = op.to_dict()
        # persistence filter (keys not evaluated this step count as a miss) and confidence
        seen = set()
        edges: list[Edge] = []
        win = int(c.get("opinion_window", 10))
        base_rate = float(c.get("relation_base_rate", 0.1))
        for key, hit, strength, ev in candidates:
            seen.add(key)
            rel_nodes = [nodes[n].attrs.get("reliability", 1.0) for n in key[:2] if n in nodes and n.startswith("obj:")]
            base = min(rel_nodes) if rel_nodes else 1.0
            self.evid.setdefault(key, deque(maxlen=win)).append((bool(hit), float(base)))
            valid, tv = self._persist(key, hit, t)
            if not valid:
                continue
            conf = float(np.clip(base * (tv["support"] / tv["window_n"]) * np.clip(strength, 0.0, 1.0), 0.0, 1.0))
            if key[2] == "LOCATED_IN":
                conf = float(base)
            tv["valid_since"] = tv["first_seen"]
            op = weighted_observations(list(self.evid[key]), a=base_rate, decay=float(c.get("evidence_decay", 0.9)))
            edges.append(Edge(f"{key[2]}:{key[0]}->{key[1]}@{step}", key[0], key[1], key[2], conf, t,
                              {k: float(v) for k, v in ev.items()}, tv, float(op.u), op.to_dict()))
        for key in list(self.persist):
            if key not in seen:
                self.persist[key].append(False)
                if not any(self.persist[key]):
                    self.persist.pop(key)
                    self.first_seen.pop(key, None)
                    self.evid.pop(key, None)   # object no longer evaluated: its evidence is dropped (bounded memory)
        scene.update(n_objects=float(len(objs)), n_corridor=float(len(corridor)),
                     n_occluded=float(sum(o.visibility_status == "partially_occluded" for o in objs)),
                     n_new=float(sum(o.visibility_status == "new" for o in objs)))
        self.last_step = step
        return Snapshot(step, float(t), nodes, edges, scene)


def video_steps(feat) -> Iterator[tuple[int, float, np.ndarray, np.ndarray | None, dict]]:
    """Yield (step, t, dets(n,7), M, scene) from a v3 VideoFeatures cache (read-only)."""
    from src.features.ego_motion import EGO_LAYOUT
    from src.features.motion_features import MOTION_FEATURES
    tr = feat.tracks
    by_step = defaultdict(list)
    for r in tr:
        by_step[int(r[0])].append([r[1], r[2], r[3], r[4], r[5], r[6], r[7]])
    cols = {n: MOTION_FEATURES.index(n) for n in SCENE_COLUMNS if n in MOTION_FEATURES}
    for s, t in enumerate(feat.times):
        M = feat.ego[s, EGO_LAYOUT["H"]].astype(float).reshape(3, 3) if feat.ego is not None else None
        scene = {n: float(feat.motion[s, i]) for n, i in cols.items()}
        yield s, float(t), np.asarray(by_step.get(s, []), float).reshape(-1, 7), M, scene


def build_video(feat, cfg: dict) -> Iterator[Snapshot]:
    b = SceneGraphBuilder(cfg, feat.fps)
    for s, t, d, M, sc in video_steps(feat):
        yield b.update(s, t, d, M, sc)
