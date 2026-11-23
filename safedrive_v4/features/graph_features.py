"""Graph-derived (SG) and KG-reasoning (KG) per-step channels, and window aggregation identical to v3 (8 statistics per channel).

Channels are computed causally: the value at step t uses only snapshots <= t (the builder and the engine are streaming).
Window features at t_end use only steps in (t_end - W, t_end] (v3 `window_indices`), so no future information enters.
"""
from __future__ import annotations

import time

import numpy as np

from safedrive_v4.reasoning.engine import ReasoningEngine, ReasoningResult
from safedrive_v4.reasoning.rules import RULE_IDS
from safedrive_v4.scene_graph.builder import SceneGraphBuilder, video_steps
from safedrive_v4.scene_graph.schema import EGO_ID, REGION_IDS, Snapshot

EGO_RELS = ["APPROACHING", "MOVING_AWAY", "CUTTING_IN_FRONT_OF", "CROSSING_PATH", "NEAR", "FOLLOWING", "SAME_DIRECTION", "POTENTIAL_CONFLICT"]
PAIR_RELS = ["NEAR", "INTERACTING_WITH", "OCCLUDING"]
TAU_CAP = 10.0
SG_FEATURES = ([f"sg_n_ego_{r.lower()}" for r in EGO_RELS] + [f"sg_conf_ego_{r.lower()}" for r in EGO_RELS]
               + [f"sg_n_pair_{r.lower()}" for r in PAIR_RELS] + [f"sg_conf_pair_{r.lower()}" for r in PAIR_RELS]
               + ["sg_min_tau_approach", "sg_max_growth_approach", "sg_max_lateral_cutin", "sg_min_time_to_corridor",
                  "sg_max_cross_speed", "sg_min_meet_time", "sg_ego_degree", "sg_density", "sg_mean_unc_corridor",
                  "sg_n_occluded", "sg_n_new", "sg_lead_growth", "sg_lead_bottom_y", "sg_lead_age"])
KG_FEATURES = [f"kg_rule_{r.lower()}" for r in RULE_IDS] + ["kg_risk_evidence", "kg_reliability", "kg_insufficient",
                                                            "kg_n_conflicts", "kg_n_risk_findings"]
AGG_STATS = ["last", "mean", "max", "min", "std", "slope", "delta", "trend"]  # same order as v3 _aggregate


def sg_vector(s: Snapshot) -> np.ndarray:
    f = dict.fromkeys(SG_FEATURES, 0.0)
    f.update(sg_min_tau_approach=TAU_CAP, sg_min_time_to_corridor=TAU_CAP, sg_min_meet_time=TAU_CAP)
    for e in s.edges:
        r = e.relation_type
        if r == "LOCATED_IN":
            continue
        ego_rel = EGO_ID in (e.source_node, e.target_node)
        if ego_rel and r in EGO_RELS:
            f[f"sg_n_ego_{r.lower()}"] += 1
            f[f"sg_conf_ego_{r.lower()}"] = max(f[f"sg_conf_ego_{r.lower()}"], e.confidence)
            f["sg_ego_degree"] += 1
        elif not ego_rel and r in PAIR_RELS:
            f[f"sg_n_pair_{r.lower()}"] += 1
            f[f"sg_conf_pair_{r.lower()}"] = max(f[f"sg_conf_pair_{r.lower()}"], e.confidence)
        ev = e.evidence_features
        if r == "APPROACHING":
            f["sg_min_tau_approach"] = min(f["sg_min_tau_approach"], ev.get("tau_s", TAU_CAP))
            f["sg_max_growth_approach"] = max(f["sg_max_growth_approach"], ev.get("scale_growth", 0.0))
        elif r == "CUTTING_IN_FRONT_OF":
            f["sg_max_lateral_cutin"] = max(f["sg_max_lateral_cutin"], ev.get("lateral_in", 0.0))
            f["sg_min_time_to_corridor"] = min(f["sg_min_time_to_corridor"], max(ev.get("time_to_corridor_s", TAU_CAP), 0.0))
        elif r == "CROSSING_PATH":
            f["sg_max_cross_speed"] = max(f["sg_max_cross_speed"], abs(ev.get("lateral_speed", 0.0)))
        elif r == "INTERACTING_WITH":
            f["sg_min_meet_time"] = min(f["sg_min_meet_time"], ev.get("time_to_meet_s", TAU_CAP))
    objs = [n for n in s.nodes.values() if n.object_id is not None]
    nonloc = sum(e.relation_type != "LOCATED_IN" for e in s.edges)
    f["sg_density"] = nonloc / max(len(objs), 1)
    corr = [n for n in objs if n.region == REGION_IDS[0]]
    f["sg_mean_unc_corridor"] = float(np.mean([n.uncertainty for n in corr])) if corr else 0.0
    f["sg_n_occluded"] = float(sum(n.visibility_status == "partially_occluded" for n in objs))
    f["sg_n_new"] = float(sum(n.visibility_status == "new" for n in objs))
    lead_e = s.edges_of("FOLLOWING", source=EGO_ID)
    if lead_e:
        n = s.nodes[lead_e[0].target_node]
        f["sg_lead_growth"] = n.scale_growth or 0.0
        f["sg_lead_bottom_y"] = n.attrs.get("y2", 0.0)
        f["sg_lead_age"] = float(min(n.track_age_steps, 20))
    return np.array([f[k] for k in SG_FEATURES], np.float32)


def kg_vector(r: ReasoningResult) -> np.ndarray:
    v = [r.per_rule.get(rid, 0.0) for rid in RULE_IDS]
    v += [r.risk_evidence, r.reliability, float(r.insufficient), float(len(r.conflicts)), float(sum(f.kind == "risk" for f in r.findings))]
    return np.array(v, np.float32)


def video_channels(feat, cfg: dict, keep: bool = False):
    """(T, |SG|), (T, |KG|), timing dict [, snapshots, results if keep]."""
    b, e = SceneGraphBuilder(cfg, feat.fps), ReasoningEngine(cfg)
    T = len(feat.times)
    SG, KG = np.zeros((T, len(SG_FEATURES)), np.float32), np.zeros((T, len(KG_FEATURES)), np.float32)
    tg = tr = 0.0
    snaps, results = [], []
    for s_, t, d, M, sc in video_steps(feat):
        t0 = time.perf_counter()
        snap = b.update(s_, t, d, M, sc)
        t1 = time.perf_counter()
        res = e.evaluate(snap)
        t2 = time.perf_counter()
        SG[s_], KG[s_] = sg_vector(snap), kg_vector(res)
        tg, tr = tg + t1 - t0, tr + t2 - t1
        if keep:
            snaps.append(snap)
            results.append(res)
    timing = {"graph_ms_per_step": 1000 * tg / max(T, 1), "reasoning_ms_per_step": 1000 * tr / max(T, 1), "steps": T}
    return (SG, KG, timing, snaps, results) if keep else (SG, KG, timing)


def slog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def window_matrix(times: np.ndarray, C: np.ndarray, t_ends: np.ndarray, W: float, fps: float) -> np.ndarray:
    from src.features.window_features import _aggregate
    from src.prediction.windows import window_indices
    n = int(round(W * fps))
    idx = np.stack([window_indices(times, float(t), n) for t in t_ends])
    return _aggregate(slog(C)[idx], fps).astype(np.float32)


def channel_names(prefix_names: list[str]) -> list[str]:
    return [f"{n}__{s}" for s in AGG_STATS for n in prefix_names]
