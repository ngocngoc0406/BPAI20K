"""Rule base for knowledge-guided reasoning over the dynamic scene graph (kg-1.0).

Every rule: id, trigger condition, inputs, KG relations used, base confidence, explanation template (filled ONLY with measured
values), exceptions. Rules produce FINDINGS (evidence statements), never warnings by themselves: whether a finding contributes to
a warning is decided by the learned predictor and the warning policy, which are evaluated (experiment plan).
Kinds: 'risk' (risk evidence), 'reliability' (lowers observation reliability), 'insufficient' (not enough evidence to conclude).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from safedrive_v4.scene_graph.schema import EGO_ID, REGION_IDS, VULNERABLE_TYPES, Snapshot


@dataclass
class Finding:
    rule_id: str
    kind: str
    conclusion: str
    confidence: float
    step: int
    timestamp: float
    subjects: list[str]
    edge_ids: list[str]
    evidence: dict[str, float]
    explanation: str
    exceptions_applied: list[str] = field(default_factory=list)
    kg_path: list[str] = field(default_factory=list)


@dataclass
class Rule:
    rule_id: str
    name: str
    kind: str
    trigger: str
    inputs: list[str]
    scene_relations: list[str]
    kg_relations: list[tuple[str, str, str]]
    concludes: str
    base_confidence: float
    explanation: str
    exceptions: list[str]
    fn: Callable[["Rule", Snapshot, dict, dict], list[Finding]]

    def doc(self) -> dict:
        return {k: getattr(self, k) for k in ("rule_id", "name", "kind", "trigger", "inputs", "scene_relations", "kg_relations",
                                               "concludes", "base_confidence", "explanation", "exceptions")}


def _f(rule: Rule, s: Snapshot, conf: float, subjects, edges, ev: dict, exc: list[str], path: list[str]) -> Finding:
    text = rule.explanation.format(**{k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in ev.items()}, subject=subjects[0] if subjects else "-")
    return Finding(rule.rule_id, rule.kind, rule.concludes, float(max(0.0, min(1.0, conf))), s.step, s.timestamp, list(subjects),
                   [e.edge_id for e in edges], {k: float(v) for k, v in ev.items() if isinstance(v, (int, float))}, text, exc, path)


def _common_exceptions(s: Snapshot, node_id: str, c: dict) -> tuple[float, list[str]]:
    n = s.nodes.get(node_id)
    mult, exc = 1.0, []
    if n is None:
        return mult, exc
    if n.visibility_status == "partially_occluded":
        mult *= 0.7
        exc.append("object partially occluded (x0.7)")
    if n.visibility_status == "truncated":
        mult *= 0.7
        exc.append("object truncated at the frame border (x0.7)")
    turn = s.nodes[EGO_ID].attrs.get("turn_proxy")
    if turn is not None and abs(turn) > c["ego_turn_flow_dx"]:
        mult *= 0.5
        exc.append("ego vehicle turning: lateral image motion may come from ego yaw (x0.5)")
    return mult, exc


# ---------------------------------------------------------------- rule functions
def r_cut_in(rule, s, ctx, c):
    out = []
    for e in s.edges_of("CUTTING_IN_FRONT_OF", target=EGO_ID):
        m, exc = _common_exceptions(s, e.source_node, c)
        n = s.nodes[e.source_node]
        ev = {**e.evidence_features, "class": n.node_type}
        out.append(_f(rule, s, rule.base_confidence * e.confidence * m, [e.source_node], [e], ev, exc,
                      [e.source_node, "CUTTING_IN_FRONT_OF", EGO_ID, "INVOLVES LateralIntrusion", "INCREASES PotentialCollisionRisk"]))
    return out


def r_low_ttc(rule, s, ctx, c):
    out = []
    for e in s.edges_of("APPROACHING", target=EGO_ID):
        tau = e.evidence_features.get("tau_s", 99.0)
        if tau >= c["low_tau_s"]:
            continue
        m, exc = _common_exceptions(s, e.source_node, c)
        sev = (c["low_tau_s"] - tau) / c["low_tau_s"]
        out.append(_f(rule, s, rule.base_confidence * e.confidence * m * (0.5 + 0.5 * sev), [e.source_node], [e],
                      {"tau_s": tau, "scale_growth": e.evidence_features.get("scale_growth", 0.0)}, exc,
                      [e.source_node, "APPROACHING", EGO_ID, f"tau={tau:.2f}s", "LowTTC INDICATES PotentialCollisionRisk"]))
    return out


def r_ttc_closing(rule, s, ctx, c):
    out = []
    for e in s.edges_of("APPROACHING", target=EGO_ID):
        tau, g = e.evidence_features.get("tau_s", 99.0), e.evidence_features.get("scale_growth", 0.0)
        if tau < c["critical_tau_s"] and g > c["high_closing_growth"]:
            m, exc = _common_exceptions(s, e.source_node, c)
            out.append(_f(rule, s, rule.base_confidence * e.confidence * m, [e.source_node], [e], {"tau_s": tau, "scale_growth": g}, exc,
                          [e.source_node, "APPROACHING", EGO_ID, "HighClosingSpeed + LowTTC", "INDICATES PotentialCollisionRisk"]))
    return out


def r_vru_crossing(rule, s, ctx, c):
    out = []
    for e in s.edges_of("CROSSING_PATH", target=EGO_ID):
        n = s.nodes[e.source_node]
        if n.node_type not in VULNERABLE_TYPES:
            continue
        m, exc = _common_exceptions(s, e.source_node, c)
        if n.attrs.get("y2", 1.0) < 0.5:
            m *= 0.5
            exc.append("road user far ahead (bottom edge above mid-frame, x0.5)")
        out.append(_f(rule, s, rule.base_confidence * e.confidence * m, [e.source_node], [e], {**e.evidence_features, "class": n.node_type}, exc,
                      [e.source_node, "CROSSING_PATH", EGO_ID, "VulnerableRoadUserInPath", "INCREASES PotentialCollisionRisk"]))
    return out


def r_lead_decel(rule, s, ctx, c):
    out = []
    hist = ctx.setdefault("lead_g", {})
    for e in s.edges_of("FOLLOWING", source=EGO_ID):
        n = s.nodes[e.target_node]
        g = n.scale_growth
        if g is None:
            continue
        prev = hist.get(e.target_node)
        hist[e.target_node] = (s.step, g)
        if prev is None or s.step - prev[0] > 5:
            continue
        dg = (g - prev[1]) / max((s.step - prev[0]) / 10.0, 0.1)
        if g > 0.1 and dg > 1.0:
            m, exc = _common_exceptions(s, e.target_node, c)
            out.append(_f(rule, s, rule.base_confidence * e.confidence * m, [e.target_node], [e], {"scale_growth": g, "growth_increase_per_s": dg}, exc,
                          [EGO_ID, "FOLLOWING", e.target_node, "growth rising (LeadDeceleration)", "INCREASES PotentialCollisionRisk"]))
    if len(hist) > 50:
        for k in sorted(hist, key=lambda x: hist[x][0])[:-20]:
            hist.pop(k)
    return out


def r_conflict(rule, s, ctx, c):
    out = []
    for e in s.edges_of("POTENTIAL_CONFLICT", target=EGO_ID):
        m, exc = _common_exceptions(s, e.source_node, c)
        out.append(_f(rule, s, rule.base_confidence * e.confidence * m, [e.source_node], [e], dict(e.evidence_features), exc,
                      [e.source_node, "POTENTIAL_CONFLICT", EGO_ID, "TrajectoryConflict", "INDICATES PotentialCollisionRisk"]))
    return out


def r_occlusion(rule, s, ctx, c):
    out = []
    for e in s.edges_of("OCCLUDING"):
        a, b = s.nodes.get(e.source_node), s.nodes.get(e.target_node)
        if a and b and (a.region in REGION_IDS or b.region in REGION_IDS):
            out.append(_f(rule, s, rule.base_confidence * e.confidence, [e.target_node, e.source_node], [e], dict(e.evidence_features), [],
                          [e.source_node, "OCCLUDING", e.target_node, "Occlusion", "REDUCES ObservationReliability"]))
    return out


def r_perception(rule, s, ctx, c):
    sc = s.scene
    reasons, sev = [], 0.0
    if sc.get("brightness") is not None and sc["brightness"] < c["dark_brightness"]:
        reasons.append("dark image")
        sev = max(sev, (c["dark_brightness"] - sc["brightness"]) / c["dark_brightness"])
    n = sc.get("n_objects", 0.0)
    if n >= 1 and sc.get("mean_det_conf") is not None and sc["mean_det_conf"] < c["low_detection_conf"]:
        reasons.append("low detector confidence")
        sev = max(sev, (c["low_detection_conf"] - sc["mean_det_conf"]) / c["low_detection_conf"])
    if n >= 3 and sc.get("n_new", 0.0) / n > c["churn_rate_max"]:
        reasons.append("unstable tracks (many new ids)")
        sev = max(sev, 0.5)
    if not reasons:
        return []
    return [_f(rule, s, rule.base_confidence * (0.5 + 0.5 * min(sev, 1.0)), [EGO_ID], [],
               {"brightness": sc.get("brightness", -1.0), "mean_det_conf": sc.get("mean_det_conf", -1.0), "reasons": "; ".join(reasons)}, [],
               ["scene", "; ".join(reasons), "PerceptionDegradation", "REDUCES ObservationReliability"])]


def r_insufficient(rule, s, ctx, c):
    corr = [n for n in s.nodes.values() if n.object_id is not None and n.region == REGION_IDS[0]]
    unknown = [n for n in corr if n.velocity_estimate is None or n.uncertainty > 0.7]
    if not unknown:
        return []
    return [_f(rule, s, rule.base_confidence * len(unknown) / max(len(corr), 1), [n.node_id for n in unknown], [],
               {"n_corridor": float(len(corr)), "n_unresolved": float(len(unknown))}, [],
               ["corridor objects without reliable motion estimate", "InsufficientEvidence"])]


def r_pair_ahead(rule, s, ctx, c):
    out = []
    for e in s.edges_of("INTERACTING_WITH"):
        a, b = s.nodes.get(e.source_node), s.nodes.get(e.target_node)
        if a and b and a.region in REGION_IDS and b.region in REGION_IDS:
            out.append(_f(rule, s, rule.base_confidence * e.confidence, [e.source_node, e.target_node], [e], dict(e.evidence_features), [],
                          [e.source_node, "INTERACTING_WITH", e.target_node, "TrafficInteraction ahead", "INCREASES ElevatedRisk"]))
    return out


RULES: list[Rule] = [
    Rule("R01", "cut-in", "risk", "CUTTING_IN_FRONT_OF(obj, ego) valid", ["lateral_in", "gap_to_corridor", "time_to_corridor_s"],
         ["CUTTING_IN_FRONT_OF"], [("Vehicle", "CAN_PERFORM", "CutInScenario"), ("LateralIntrusion", "INCREASES", "PotentialCollisionRisk")],
         "LateralIntrusion", 0.8, "{subject} ({class}) moves toward the ego path at {lateral_in} frame/s; reaches the corridor in about {time_to_corridor_s} s",
         ["ego turning halves confidence", "occluded / truncated object x0.7"], r_cut_in),
    Rule("R02", "fast approach / low image TTC", "risk", "APPROACHING(obj, ego) with tau < low_tau_s", ["tau_s", "scale_growth"],
         ["APPROACHING"], [("LowTTC", "INDICATES", "PotentialCollisionRisk")], "LowTTC", 0.7,
         "{subject} grows at {scale_growth}/s in the image: image-based time-to-contact {tau_s} s",
         ["tau is image-based (constant closing speed assumed)", "occluded / truncated object x0.7", "ego turning x0.5"], r_low_ttc),
    Rule("R03", "low TTC with high closing speed", "risk", "APPROACHING with tau < critical_tau_s and growth > high_closing_growth",
         ["tau_s", "scale_growth"], ["APPROACHING"], [("HighClosingSpeed", "INCREASES", "PotentialCollisionRisk"),
                                                      ("LowTTC", "INDICATES", "PotentialCollisionRisk")], "HighClosingSpeed", 0.85,
         "{subject}: image TTC {tau_s} s while growing at {scale_growth}/s", ["same as R02"], r_ttc_closing),
    Rule("R04", "vulnerable road user crossing", "risk", "CROSSING_PATH(pedestrian|bicycle, ego) valid", ["lateral_speed", "lateral_in"],
         ["CROSSING_PATH"], [("Pedestrian", "CAN_PERFORM", "CrossingScenario"), ("VulnerableRoadUserInPath", "INCREASES", "PotentialCollisionRisk")],
         "VulnerableRoadUserInPath", 0.75, "{subject} ({class}) crosses the image path at {lateral_speed} frame/s",
         ["far road user (upper half) x0.5", "occluded / truncated x0.7"], r_vru_crossing),
    Rule("R05", "lead deceleration proxy", "risk", "FOLLOWING(ego, lead) and lead growth rising > 1/s per s", ["scale_growth", "growth_increase_per_s"],
         ["FOLLOWING"], [("CarFollowingScenario", "CAN_INVOLVE", "LeadDeceleration"), ("LeadDeceleration", "INCREASES", "PotentialCollisionRisk")],
         "LeadDeceleration", 0.6, "lead {subject}: image growth rose to {scale_growth}/s (+{growth_increase_per_s}/s per s), consistent with braking",
         ["no brake-light detection; growth may also come from ego acceleration"], r_lead_decel),
    Rule("R06", "trajectory conflict", "risk", "POTENTIAL_CONFLICT(obj, ego) valid", ["pred_x", "pred_y", "tau_s"],
         ["POTENTIAL_CONFLICT"], [("TrajectoryConflict", "INDICATES", "PotentialCollisionRisk")], "TrajectoryConflict", 0.6,
         "{subject}: constant-velocity image prediction reaches the ego zone ({pred_x}, {pred_y}) within {horizon_s} s",
         ["constant image velocity assumed", "occluded / truncated x0.7", "ego turning x0.5"], r_conflict),
    Rule("R07", "occlusion near the path", "reliability", "OCCLUDING(a, b) with a or b in the corridor / adjacent regions", ["iou"],
         ["OCCLUDING"], [("Occlusion", "REDUCES", "ObservationReliability")], "Occlusion", 0.6,
         "{subject} is partly hidden (box overlap IoU {iou}); its motion estimate is less reliable", [], r_occlusion),
    Rule("R08", "perception degradation", "reliability", "dark frame, low detector confidence or unstable tracks",
         ["brightness", "mean_det_conf", "n_new"], [], [("PerceptionDegradation", "REDUCES", "ObservationReliability")],
         "PerceptionDegradation", 0.8, "perception degraded: {reasons} (brightness {brightness}, detector confidence {mean_det_conf})", [], r_perception),
    Rule("R09", "insufficient evidence", "insufficient", "corridor objects without a reliable motion estimate", ["track_age", "uncertainty"],
         [], [("ObservationReliability", "CONDITIONS", "Warning")], "InsufficientEvidence", 0.9,
         "{n_unresolved} of {n_corridor} objects in the path have no reliable motion estimate yet: risk cannot be concluded from them", [], r_insufficient),
    Rule("R10", "interaction between road users ahead", "risk", "INTERACTING_WITH(a, b) with both in the corridor / adjacent regions",
         ["closing_rate", "time_to_meet_s"], ["INTERACTING_WITH"], [("TrafficInteraction", "INCREASES", "ElevatedRisk")], "ElevatedRisk", 0.4,
         "{subject} and another road user converge (meet in about {time_to_meet_s} s in the image)", ["weak rule (no ego involvement)"], r_pair_ahead),
]
RULE_IDS = [r.rule_id for r in RULES]
