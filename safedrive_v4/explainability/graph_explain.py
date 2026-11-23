"""Graph-level explanation: reasoning chains that link a prediction to concrete nodes, edges and timestamps.

A chain is built only from stored findings and their edges; every element carries the node / edge id and the time it was
observed. `validate_chain` checks that each referenced node and edge exists in the snapshot it claims to come from.
"""
from __future__ import annotations

from safedrive_v4.reasoning.engine import ReasoningResult
from safedrive_v4.scene_graph.schema import Snapshot


def chains(snapshot: Snapshot, result: ReasoningResult, k: int = 3) -> list[dict]:
    edges = {e.edge_id: e for e in snapshot.edges}
    out = []
    for f in sorted([f for f in result.findings if f.kind == "risk"], key=lambda f: -f.confidence)[:k]:
        steps = []
        for eid in f.edge_ids:
            e = edges.get(eid)
            if e is None:
                continue
            src, dst = snapshot.nodes.get(e.source_node), snapshot.nodes.get(e.target_node)
            steps.append({"node": e.source_node, "node_type": src.node_type if src else None, "relation": e.relation_type,
                          "target": e.target_node, "target_type": dst.node_type if dst else None, "edge_id": eid,
                          "confidence": round(e.confidence, 3), "observed_since_s": e.temporal_validity.get("first_seen"),
                          "timestamp_s": e.timestamp, "evidence": {k2: round(v, 3) for k2, v in e.evidence_features.items()}})
        out.append({"rule_id": f.rule_id, "conclusion": f.conclusion, "confidence": round(f.confidence, 3), "explanation": f.explanation,
                    "kg_path": f.kg_path, "graph_steps": steps, "exceptions": f.exceptions_applied, "timestamp_s": f.timestamp})
    return out


def validate_chain(chain: dict, snapshot: Snapshot) -> dict:
    eids = {e.edge_id for e in snapshot.edges}
    bad = [s["edge_id"] for s in chain["graph_steps"] if s["edge_id"] not in eids or s["node"] not in snapshot.nodes
           or s["target"] not in snapshot.nodes or abs(s["timestamp_s"] - snapshot.timestamp) > 1e-6]
    return {"valid": not bad and bool(chain["graph_steps"]), "invalid_steps": bad}


def reliability_notes(result: ReasoningResult) -> list[str]:
    return [f.explanation for f in result.findings if f.kind in ("reliability", "insufficient")]
