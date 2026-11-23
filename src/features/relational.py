"""Ego-compensated object motion + vehicle-interaction features, derived from cached tracks and per-step camera transforms.

For a track seen at steps t-k and t, its box at t-k is mapped through the accumulated camera transform T_{t-k->t}
(product of per-step transforms). The mapped box is where the object would appear if it were part of the static scene.
The difference between the observed and the mapped box is the object's motion RELATIVE TO THE SCENE (residual motion):
  residual centre velocity  (c_t - c_hat) / (k dt)                      [frame units / s]
  residual scale growth     (s_t / s_hat - 1) / (k dt)                   [1/s]   (looming not explained by ego zoom)
With an identity transform (mode 'none') these reduce to raw image-plane motion, giving the uncompensated ablation.

All quantities are image-plane. No physical distance, speed or time-to-collision is claimed (no calibration / depth).
`*_ttr` values are image-based "time to reach" proxies (distance / closing rate, capped at 10 s), used only as learned inputs.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.features.ego_motion import EGO_LAYOUT

REL_FEATURES = [
    # ego-compensated object motion
    "res_speed_max", "res_speed_mean", "res_lat_in_max", "res_loom_center", "res_loom_max", "res_tau_center",
    "lead_res_loom", "lead_res_lat_in",
    # interactions with the ego path / other road users
    "corridor_occupancy", "lead_gap", "lead_gap_rate", "lead_brake_proxy",
    "cutin_score", "cutin_count", "pair_ttr_min", "pair_converging", "pair_heading_cos", "ego_ttr_min",
]
TTR_CAP = 10.0
FEATURE_DESCRIPTIONS_REL = {
    "res_speed_max": "fastest object motion relative to the scene (camera motion removed)",
    "res_speed_mean": "mean object motion relative to the scene",
    "res_lat_in_max": "fastest scene-relative lateral motion toward the ego path",
    "res_loom_center": "fastest scene-relative growth of a corridor object (approach not explained by ego motion)",
    "res_loom_max": "fastest scene-relative growth of any object",
    "res_tau_center": "image-based time-to-reach of the corridor object after ego compensation (s, capped)",
    "lead_res_loom": "lead object's scene-relative growth", "lead_res_lat_in": "lead object's scene-relative lateral motion toward centre",
    "corridor_occupancy": "area occupied by road users in the driving corridor",
    "lead_gap": "image gap between the corridor lead object's bottom edge and the frame bottom",
    "lead_gap_rate": "rate of change of that gap (negative = closing in)",
    "lead_brake_proxy": "increase in the lead object's growth rate (sudden approach, e.g. lead braking)",
    "cutin_score": "strongest cut-in cue: object beside the corridor moving inward (weighted by size)",
    "cutin_count": "number of objects beside the corridor moving inward",
    "pair_ttr_min": "shortest image-based time for two road users to meet (s, capped)",
    "pair_converging": "number of converging road-user pairs",
    "pair_heading_cos": "cosine between the motion directions of the most converging pair (-1 = head-on)",
    "ego_ttr_min": "shortest image-based time for an object to reach the ego bottom-centre point (s, capped)",
}


def _transform_points(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ M.T
    return p[:, :2] / np.where(np.abs(p[:, 2:3]) < 1e-9, 1e-9, p[:, 2:3])


def step_transforms(ego: np.ndarray | None, mode: str, T: int) -> list[np.ndarray]:
    """Per-step 3x3 transforms (t-1 -> t). mode: 'h' homography, 'f' flow-affine, 'none' identity."""
    if mode == "none" or ego is None:
        return [np.eye(3)] * T
    key = "H" if mode == "h" else "F"
    return [ego[t, EGO_LAYOUT[key]].astype(np.float64).reshape(3, 3) for t in range(T)]


def derive_relational(times: np.ndarray, tracks: np.ndarray, ego: np.ndarray | None, fps: float = 10.0, k: int = 3,
                      mode: str = "h", min_area: float = 0.002) -> np.ndarray:
    """Returns (T, len(REL_FEATURES)) float32. tracks: (N,8) [step, id, cls, conf, x1, y1, x2, y2] normalised."""
    T = len(times)
    out = np.zeros((T, len(REL_FEATURES)), np.float32)
    out[:, REL_FEATURES.index("res_tau_center")] = TTR_CAP
    out[:, REL_FEATURES.index("pair_ttr_min")] = TTR_CAP
    out[:, REL_FEATURES.index("ego_ttr_min")] = TTR_CAP
    out[:, REL_FEATURES.index("lead_gap")] = 1.0
    out[:, REL_FEATURES.index("pair_heading_cos")] = 1.0
    if len(tracks) == 0:
        return out
    Ms = step_transforms(ego, mode, T)
    # accumulated transforms A_t = M_t @ ... @ M_{t-k+1}  (maps t-k -> t)
    acc = []
    for t in range(T):
        A = np.eye(3)
        for j in range(max(t - k + 1, 1), t + 1):
            A = Ms[j] @ A
        acc.append(A)
    by_step: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for r in tracks:
        if r[1] >= 0:
            by_step[int(r[0])][int(r[1])] = r[4:8].astype(np.float64)
    dtk = k / fps
    prev_lead_growth, prev_gap = None, None
    lead_hist: dict[int, float] = {}
    for t in range(T):
        cur = by_step.get(t, {})
        if not cur:
            continue
        past = by_step.get(t - k, {})
        objs = []
        for tid, b in cur.items():
            c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            area = max(b[2] - b[0], 1e-4) * max(b[3] - b[1], 1e-4)
            v = np.zeros(2)
            g = 0.0
            has = tid in past and t - k >= 0
            if has:
                pb = past[tid]
                quad = np.array([[pb[0], pb[1]], [pb[2], pb[1]], [pb[2], pb[3]], [pb[0], pb[3]]])
                q = _transform_points(acc[t], quad)
                c_hat = q.mean(0)
                a_hat = 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], 1)) - np.dot(q[:, 1], np.roll(q[:, 0], 1)))
                v = (c - c_hat) / dtk
                g = (np.sqrt(area) / max(np.sqrt(max(a_hat, 1e-8)), 1e-4) - 1.0) / dtk
            objs.append({"id": tid, "c": c, "area": area, "v": v, "g": float(np.clip(g, -5, 5)), "has": has, "y2": b[3]})
        f = dict.fromkeys(REL_FEATURES, 0.0)
        f.update(res_tau_center=TTR_CAP, pair_ttr_min=TTR_CAP, ego_ttr_min=TTR_CAP, lead_gap=1.0, pair_heading_cos=1.0)
        moving = [o for o in objs if o["has"]]
        if moving:
            sp = [np.hypot(*o["v"]) for o in moving]
            f["res_speed_max"], f["res_speed_mean"] = float(max(sp)), float(np.mean(sp))
            lat = [-np.sign(o["c"][0] - 0.5) * o["v"][0] for o in moving if o["area"] >= min_area]
            f["res_lat_in_max"] = float(max(lat)) if lat else 0.0
            f["res_loom_max"] = float(max(o["g"] for o in moving))
        corridor = [o for o in objs if abs(o["c"][0] - 0.5) < 0.2 and o["c"][1] > 0.35]
        f["corridor_occupancy"] = float(sum(o["area"] for o in corridor))
        cm = [o for o in corridor if o["has"]]
        if cm:
            gmax = max(o["g"] for o in cm)
            f["res_loom_center"] = gmax
            f["res_tau_center"] = float(min(1.0 / gmax, TTR_CAP)) if gmax > 1.0 / TTR_CAP else TTR_CAP
        if corridor:
            lead = max(corridor, key=lambda o: o["y2"])  # lowest bottom edge in the corridor = nearest in the image
            gap = 1.0 - lead["y2"]
            f["lead_gap"] = float(gap)
            if prev_gap is not None and prev_gap[0] == lead["id"]:
                f["lead_gap_rate"] = float((gap - prev_gap[1]) * fps)
            prev_gap = (lead["id"], gap)
            if lead["has"]:
                f["lead_res_loom"] = lead["g"]
                f["lead_res_lat_in"] = float(-np.sign(lead["c"][0] - 0.5) * lead["v"][0])
                old = lead_hist.get(lead["id"])
                if old is not None:
                    f["lead_brake_proxy"] = float(np.clip((lead["g"] - old) / dtk, -20, 20))
                lead_hist[lead["id"]] = lead["g"]
        else:
            prev_gap = None
        side = [o for o in moving if 0.12 < abs(o["c"][0] - 0.5) < 0.4 and o["c"][1] > 0.3 and o["area"] >= min_area]
        inward = [(-np.sign(o["c"][0] - 0.5) * o["v"][0], o) for o in side]
        inward = [(vi, o) for vi, o in inward if vi > 0]
        if inward:
            f["cutin_score"] = float(max(vi * np.sqrt(o["area"]) for vi, o in inward))
            f["cutin_count"] = float(sum(vi > 0.05 for vi, _ in inward))
        # pairwise convergence (top-6 largest moving objects)
        top = sorted(moving, key=lambda o: -o["area"])[:6]
        best_ttr, best_cos, conv = TTR_CAP, 1.0, 0
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                a, b = top[i], top[j]
                d = b["c"] - a["c"]
                dist = float(np.linalg.norm(d))
                closing = float(-np.dot(d, b["v"] - a["v"]) / max(dist, 1e-6))
                if closing > 0.05:
                    conv += 1
                    ttr = min(dist / closing, TTR_CAP)
                    if ttr < best_ttr:
                        na, nb = np.linalg.norm(a["v"]), np.linalg.norm(b["v"])
                        best_ttr = ttr
                        best_cos = float(np.dot(a["v"], b["v"]) / (na * nb)) if na > 1e-6 and nb > 1e-6 else 1.0
        f["pair_ttr_min"], f["pair_heading_cos"], f["pair_converging"] = best_ttr, best_cos, float(conv)
        # convergence toward the ego reference point (bottom centre of the image)
        ego_pt = np.array([0.5, 1.0])
        for o in moving:
            d = ego_pt - o["c"]
            dist = float(np.linalg.norm(d))
            closing = float(np.dot(d, o["v"]) / max(dist, 1e-6))
            if closing > 0.05:
                f["ego_ttr_min"] = min(f["ego_ttr_min"], dist / closing)
        out[t] = [f[n] for n in REL_FEATURES]
    return out
