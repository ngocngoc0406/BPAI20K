"""Physically consistent counterfactuals at the TRACK level.

A counterfactual edits the observed trajectory of ONE road user inside the explanation window and then recomputes EVERY derived
quantity from the edited tracks (v3 object-motion features, v3 interaction features, scene graph, KG findings, window features)
with the same code as the factual pass. Features are therefore never changed independently of each other.

Edits (applied to steps in [t_end - span, t_end], continuous at the window start):
  closing_speed(alpha)  box scale s(t) = s(t0) * (s_obs(t) / s(t0)) ** alpha  (alpha < 1: slower approach; image TTC rises by ~1/alpha)
  higher_ttc(factor)    closing_speed with alpha = 1 / factor
  no_cut_in()           lateral centre frozen at x(t0) (size and vertical position as observed): the road user stays in its lane

Validity checks reported with every counterfactual:
  identity_reproduction  |p(identity edit) - p(factual)|  (must be ~0: the recomputation reproduces the factual prediction)
  boxes_valid            all edited boxes inside the frame with positive size
  max_step_change_ok     per-step box change of the edit not larger than the largest observed per-step change of that track
  pixel_channels_kept    optical-flow / frame-difference channels are computed from pixels and CANNOT be regenerated; they are
                         kept as observed (a stated limitation, not hidden)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from src.features.motion_features import MOTION_FEATURES, OBJECT_FEATURES, MotionFeatureComputer

N_OBJ = len(OBJECT_FEATURES)


@dataclass
class CFResult:
    kind: str
    params: dict
    track_id: int
    p_factual: float
    p_counterfactual: float
    delta: float
    validity: dict
    changed_findings: dict


def recompute_motion(feat) -> np.ndarray:
    """Object-motion columns recomputed from (possibly edited) tracks; pixel-derived columns kept from the cache."""
    mf = MotionFeatureComputer(feat.fps)
    out = feat.motion.copy()
    tr = feat.tracks
    steps = tr[:, 0].astype(int) if len(tr) else np.zeros(0, int)
    fl0 = MOTION_FEATURES.index("flow_mag_mean")
    for s in range(len(feat.times)):
        rows = tr[steps == s]
        vec = mf.update(s, rows[:, 4:8], rows[:, 1], rows[:, 2].astype(int), rows[:, 3], feat.motion[s, N_OBJ:fl0 + 4],
                        float(feat.motion[s, MOTION_FEATURES.index("brightness")]))
        out[s, :N_OBJ] = vec[:N_OBJ]
        out[s, MOTION_FEATURES.index("mean_det_conf")] = vec[MOTION_FEATURES.index("mean_det_conf")]
    return out


def edit_tracks(feat, track_id: int, t_end: float, span: float, kind: str, alpha: float = 0.5) -> tuple[object, dict]:
    f2 = copy.copy(feat)
    tr = feat.tracks.copy()
    sel = np.where((tr[:, 1] == track_id) & (feat.times[tr[:, 0].astype(int)] >= t_end - span - 1e-6)
                   & (feat.times[tr[:, 0].astype(int)] <= t_end + 1e-6))[0]
    sel = sel[np.argsort(tr[sel, 0])]
    info = {"edited_rows": int(len(sel))}
    if len(sel) >= 2 and kind != "identity":
        b0 = tr[sel[0], 4:8].copy()
        s0 = np.sqrt((b0[2] - b0[0]) * (b0[3] - b0[1]))
        x0 = (b0[0] + b0[2]) / 2
        for i in sel:
            b = tr[i, 4:8].copy()
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            w, h = b[2] - b[0], b[3] - b[1]
            if kind == "closing_speed":
                s = np.sqrt(w * h)
                k = (s0 * (s / s0) ** alpha) / max(s, 1e-6)
                w, h = w * k, h * k
            elif kind == "no_cut_in":
                cx = x0
            nb = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            if np.any(nb < 0) or np.any(nb > 1):
                info["clipped_rows"] = info.get("clipped_rows", 0) + 1   # leaves the view: observed truncated, as a detector would
            tr[i, 4:8] = np.clip(nb, 0.0, 1.0)
    obs = feat.tracks[sel, 4:8]
    new = tr[sel, 4:8]
    max_obs = float(np.max(np.abs(np.diff(obs, axis=0)))) if len(sel) > 1 else 0.0
    max_new = float(np.max(np.abs(np.diff(new, axis=0)))) if len(sel) > 1 else 0.0
    info.update(boxes_valid=bool(np.all(new[:, :2] >= -1e-6) and np.all(new[:, 2:] <= 1 + 1e-6) and np.all(new[:, 2:] > new[:, :2])) if len(sel) else True,
                max_step_change_ok=bool(max_new <= max_obs + 1e-6), pixel_channels_kept=True)
    f2.tracks = tr
    f2.derived = {}
    f2.motion = recompute_motion(f2)
    return f2, info


def track_candidates(snapshot, k: int = 3) -> list[int]:
    """Road users referenced by risk findings / ego edges at the explained step (most relevant first)."""
    ids = []
    for e in snapshot.edges:
        if e.target_node == "ego" and e.source_node.startswith("obj:") and e.relation_type not in ("LOCATED_IN", "MOVING_AWAY"):
            tid = int(e.source_node.split(":")[1])
            if tid not in ids:
                ids.append(tid)
    return ids[:k]


def run(predict_fn, feat, track_id: int, t_end: float, kind: str, span: float = 2.0, alpha: float = 0.5) -> CFResult:
    """predict_fn(feat, t_end) -> (p, reasoning summary dict). Returns factual vs counterfactual with validity checks."""
    p_f, r_f = predict_fn(feat, t_end)
    f_id, _ = edit_tracks(feat, track_id, t_end, span, "identity")
    p_id, _ = predict_fn(f_id, t_end)
    real_kind = "closing_speed" if kind in ("closing_speed", "higher_ttc") else kind
    # higher_ttc: `alpha` is the TTC factor (e.g. 2 = twice the image TTC) -> growth exponent 1 / factor
    a = 1.0 / max(alpha, 1e-6) if kind == "higher_ttc" else alpha
    f_cf, info = edit_tracks(feat, track_id, t_end, span, real_kind, a)
    p_cf, r_cf = predict_fn(f_cf, t_end)
    info["identity_reproduction"] = float(abs(p_id - p_f))
    info["valid"] = bool(info["boxes_valid"] and info["max_step_change_ok"] and info["identity_reproduction"] < 1e-4)
    changed = {k: (r_f.get(k), r_cf.get(k)) for k in set(r_f) | set(r_cf) if r_f.get(k) != r_cf.get(k)}
    return CFResult(kind, {"alpha": a, "span_s": span}, int(track_id), float(p_f), float(p_cf), float(p_cf - p_f), info, changed)
