"""Evidence trace for every v4 warning (and on request for any step).

A trace stores ONLY values produced by the pipeline: track ids and frame references from the tracker, graph nodes / edges from the
scene-graph builder, rule findings from the reasoning engine, the model output, SHAP attributions and the warning decision.
`trace_sha256` is computed over the canonical JSON so that later edits are detectable; `verify()` re-checks it and checks that
every referenced edge / node exists in the stored graph excerpt (integrity).
No language model writes any part of a trace; an optional text rendering may only paraphrase the stored fields.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class EvidenceTrace:
    event_id: str
    video_id: str
    timestamp: float
    step: int
    frame_refs: list[int]
    track_ids: list[int]
    nodes: dict
    edges: list[dict]
    findings: list[dict]
    kg_paths: list[list[str]]
    prediction: dict
    shap: list[dict]
    uncertainty: dict
    warning: dict
    unknowns: list[str]
    provenance: dict
    counterfactuals: list[dict] = field(default_factory=list)
    trace_sha256: str = ""

    def canonical(self) -> str:
        d = asdict(self)
        d.pop("trace_sha256")
        return json.dumps(d, sort_keys=True, default=float)

    def seal(self) -> "EvidenceTrace":
        self.trace_sha256 = hashlib.sha256(self.canonical().encode()).hexdigest()
        return self

    def verify(self) -> dict:
        ok_hash = hashlib.sha256(self.canonical().encode()).hexdigest() == self.trace_sha256
        edge_ids = {e["edge_id"] for e in self.edges}
        node_ids = set(self.nodes)
        missing_edges = sorted({eid for f in self.findings for eid in f.get("edge_ids", []) if eid not in edge_ids})
        missing_nodes = sorted({n for e in self.edges for n in (e["source_node"], e["target_node"]) if n not in node_ids})
        return {"hash_ok": ok_hash, "missing_edges": missing_edges, "missing_nodes": missing_nodes,
                "ok": ok_hash and not missing_edges and not missing_nodes}

    def save(self, out_dir: str | Path) -> Path:
        p = Path(out_dir) / f"trace_{self.event_id}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=1, default=float), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceTrace":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def new_event_id() -> str:
    return uuid.uuid4().hex[:12]


def unknowns_for(snapshot, result) -> list[str]:
    """What the system could NOT determine at this step (shown to the user)."""
    out = ["physical distance, speed and real time-to-collision (monocular image, no calibration)",
           "traffic lights / signs (not detected by the v3 detector)", "lane markings (fixed image regions are used instead)"]
    if result is not None and result.insufficient:
        out.append("motion of some road users in the path (tracks too short or unreliable)")
    if snapshot is not None:
        occ = [n.node_id for n in snapshot.nodes.values() if n.visibility_status == "partially_occluded"]
        if occ:
            out.append(f"full extent of partially occluded road users: {', '.join(occ[:5])}")
    return out
