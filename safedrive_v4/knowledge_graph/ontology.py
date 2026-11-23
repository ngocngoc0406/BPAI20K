"""Traffic Knowledge Graph (kg-1.0): ontology + initial risk knowledge as a lightweight property graph (networkx MultiDiGraph).

Representation choice (see docs/SAFEDRIVE_V4_KNOWLEDGE_GRAPH.md): RDF/OWL would add a reasoner and serialisation stack that this
project does not need (rdflib is not installed; no OWL inference is used); Neo4j would add a database server for a graph of a few
dozen nodes. A property graph in networkx is enough, is inspectable, and is exported to subject-predicate-object triples (and
node-link JSON) for interoperability.

The domain relations below are INITIAL HYPOTHESES (status 'hypothesis'), not ground truth. Each carries `status` and `source`;
the evaluation (docs/SAFEDRIVE_V4_EXPERIMENT_PLAN.md) measures whether the rules that use them carry predictive signal.
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

CLASSES = {
    # class: parent
    "Thing": None, "RoadUser": "Thing", "Vehicle": "RoadUser", "EgoVehicle": "Vehicle", "Car": "Vehicle", "Bus": "Vehicle",
    "Truck": "Vehicle", "Motorcycle": "Vehicle", "Bicycle": "RoadUser", "Pedestrian": "RoadUser",
    "RoadScenario": "Thing", "CutInScenario": "RoadScenario", "CarFollowingScenario": "RoadScenario", "CrossingScenario": "RoadScenario",
    "TrafficInteraction": "Thing", "RiskFactor": "Thing", "LowTTC": "RiskFactor", "HighClosingSpeed": "RiskFactor",
    "LateralIntrusion": "RiskFactor", "TrajectoryConflict": "RiskFactor", "VulnerableRoadUserInPath": "RiskFactor",
    "LeadDeceleration": "RiskFactor", "Occlusion": "RiskFactor", "PerceptionDegradation": "RiskFactor",
    "RiskLevel": "Thing", "PotentialCollisionRisk": "RiskLevel", "ElevatedRisk": "RiskLevel", "LowRisk": "RiskLevel",
    "TrafficRule": "Thing", "KeepSafeDistance": "TrafficRule", "YieldToPedestrians": "TrafficRule", "SafeLaneChange": "TrafficRule",
    "Evidence": "Thing", "Warning": "Thing", "Uncertainty": "Thing", "ObservationReliability": "Uncertainty",
    "InsufficientEvidence": "Uncertainty",
}
# scene-graph relation type -> ontology interaction concept
SCENE_RELATIONS = {
    "FOLLOWING": "CarFollowingScenario", "CUTTING_IN_FRONT_OF": "CutInScenario", "APPROACHING": "TrafficInteraction",
    "MOVING_AWAY": "TrafficInteraction", "CROSSING_PATH": "CrossingScenario", "NEAR": "TrafficInteraction",
    "OCCLUDING": "Occlusion", "SAME_DIRECTION": "TrafficInteraction", "OPPOSITE_DIRECTION": "TrafficInteraction",
    "POTENTIAL_CONFLICT": "TrajectoryConflict", "INTERACTING_WITH": "TrafficInteraction",
}
DOMAIN_TRIPLES = [
    # (subject, predicate, object, status, source)
    ("Motorcycle", "CAN_PERFORM", "CutInScenario", "hypothesis", "domain knowledge; motorcycles change lanes in dense traffic"),
    ("Vehicle", "CAN_PERFORM", "CutInScenario", "hypothesis", "domain knowledge"),
    ("Pedestrian", "CAN_PERFORM", "CrossingScenario", "hypothesis", "domain knowledge"),
    ("CutInScenario", "INVOLVES", "LateralIntrusion", "hypothesis", "definition"),
    ("CarFollowingScenario", "CAN_INVOLVE", "LeadDeceleration", "hypothesis", "definition"),
    ("LowTTC", "INDICATES", "PotentialCollisionRisk", "hypothesis", "Lee (1976) tau; image-based proxy only"),
    ("HighClosingSpeed", "INCREASES", "PotentialCollisionRisk", "hypothesis", "domain knowledge"),
    ("LateralIntrusion", "INCREASES", "PotentialCollisionRisk", "hypothesis", "domain knowledge"),
    ("TrajectoryConflict", "INDICATES", "PotentialCollisionRisk", "hypothesis", "domain knowledge"),
    ("VulnerableRoadUserInPath", "INCREASES", "PotentialCollisionRisk", "hypothesis", "domain knowledge"),
    ("LeadDeceleration", "INCREASES", "PotentialCollisionRisk", "hypothesis", "domain knowledge"),
    ("Occlusion", "REDUCES", "ObservationReliability", "hypothesis", "perception: hidden objects are not observed"),
    ("PerceptionDegradation", "REDUCES", "ObservationReliability", "hypothesis", "perception"),
    ("ObservationReliability", "CONDITIONS", "Warning", "design", "low reliability -> 'insufficient evidence', not 'high risk'"),
    ("KeepSafeDistance", "RELATES_TO", "LowTTC", "hypothesis", "traffic rule"),
    ("YieldToPedestrians", "RELATES_TO", "VulnerableRoadUserInPath", "hypothesis", "traffic rule"),
    ("SafeLaneChange", "RELATES_TO", "LateralIntrusion", "hypothesis", "traffic rule"),
    ("PotentialCollisionRisk", "IS_A_LEVEL_ABOVE", "ElevatedRisk", "design", "ordering of risk levels"),
    ("ElevatedRisk", "IS_A_LEVEL_ABOVE", "LowRisk", "design", "ordering of risk levels"),
]


def build_ontology() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph(name="SafeDrive traffic KG", schema="kg-1.0")
    for cls, parent in CLASSES.items():
        g.add_node(cls, kind="class")
        if parent:
            g.add_edge(cls, parent, key="SUBCLASS_OF", predicate="SUBCLASS_OF", status="definition")
    for rel, concept in SCENE_RELATIONS.items():
        g.add_node(f"rel:{rel}", kind="scene_relation")
        g.add_edge(f"rel:{rel}", concept, key="INSTANTIATES", predicate="INSTANTIATES", status="definition")
    for s, p, o, status, src in DOMAIN_TRIPLES:
        g.add_edge(s, o, key=p, predicate=p, status=status, source=src)
    return g


def add_rules(g: nx.MultiDiGraph, rules) -> nx.MultiDiGraph:
    for r in rules:
        g.add_node(r.rule_id, kind="rule", name=r.name, confidence=r.base_confidence)
        for rel in r.scene_relations:
            g.add_edge(r.rule_id, f"rel:{rel}", key="USES", predicate="USES", status="definition")
        for s, p, o in r.kg_relations:
            g.add_edge(r.rule_id, s, key="SUPPORTED_BY", predicate="SUPPORTED_BY", via=f"{s} {p} {o}", status="definition")
        g.add_edge(r.rule_id, r.concludes, key="CONCLUDES", predicate="CONCLUDES", status="hypothesis")
    return g


def ancestors(g: nx.MultiDiGraph, cls: str) -> list[str]:
    out, cur = [], cls
    while True:
        nxt = [v for _, v, k in g.out_edges(cur, keys=True) if k == "SUBCLASS_OF"]
        if not nxt:
            return out
        cur = nxt[0]
        out.append(cur)


def is_a(g: nx.MultiDiGraph, cls: str, parent: str) -> bool:
    return cls == parent or parent in ancestors(g, cls)


def triples(g: nx.MultiDiGraph) -> list[tuple[str, str, str]]:
    return sorted((u, d.get("predicate", k), v) for u, v, k, d in g.edges(keys=True, data=True))


def export(g: nx.MultiDiGraph, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "traffic_kg.nodelink.json").write_text(json.dumps(nx.node_link_data(g, edges="links"), indent=1, default=str), encoding="utf-8")
    (out / "traffic_kg.triples.tsv").write_text("".join(f"{s}\t{p}\t{o}\n" for s, p, o in triples(g)), encoding="utf-8")
    return {"nodes": g.number_of_nodes(), "edges": g.number_of_edges()}
