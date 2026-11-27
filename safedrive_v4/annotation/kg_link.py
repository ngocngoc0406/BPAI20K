"""Annotation -> knowledge-graph triples with evidence references and provenance, evidence lookup, and validation of the automatic
scene graph against human interactions (E6).

Triple: {source, relation, target, valid_from, valid_to, evidence: [EvidenceRef], provenance: {record_id, annotator | adjudication,
schema, op_seq}, confidence}. Examples:
  actor:A --APPROACHING--> ego            (interaction)
  actor:B --BRAKING--> evidence:<sha>@12.0  (unary relation: grounded on its evidence frames)
  event:<id> --INVOLVES--> actor:A          event:<id> --SUPPORTED_BY--> evidence:<sha>@12.0
Only human records (per annotator, marked as such, or adjudicated) become triples; model output never does.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

SG_RELATION = {"APPROACHING": "APPROACHING", "MOVING_AWAY": "MOVING_AWAY", "CUTTING_IN": "CUTTING_IN_FRONT_OF",
               "FOLLOWING": "FOLLOWING", "CROSSING": "CROSSING_PATH"}
NOT_MODELLED = ("BRAKING", "YIELDING", "BLOCKING", "COLLISION_INTERACTION")
CONF = {"low": 0.33, "medium": 0.66, "high": 1.0}


def _ev_node(e: dict) -> str:
    return f"evidence:{e['image_sha256'][:12]}@{e['t']:.2f}"


def _records(store, video: str, source: str) -> tuple[list, list, list, str]:
    """source: 'adjudicated' or an annotator pseudonym."""
    from safedrive_v4.annotation.schema import Interaction, TemporalEvent
    if source == "adjudicated":
        evs, ints = [], []
        for adj in store.adjudications(video):
            if adj["record"] is None:
                continue
            if adj["record_type"] == "TemporalEvent":
                evs.append((TemporalEvent(**adj["record"]), {"adjudication": adj["adj_id"], "coordinator": adj["coordinator"]}))
            elif adj["record_type"] == "Interaction":
                ints.append((Interaction(**adj["record"]), {"adjudication": adj["adj_id"], "coordinator": adj["coordinator"]}))
        actors = [a for ann in store.annotators(video) for a in store.records(video, ann, "actor")]
        return evs, ints, actors, "adjudicated"
    seq = {o["record_id"]: o["seq"] for o in store.ops(video, source) if "record_id" in o}
    prov = lambda r: {"annotator": source, "op_seq": seq.get(r.record_id), "status": "single annotator (not adjudicated)"}  # noqa: E731
    return ([(e, prov(e)) for e in store.records(video, source, "event")], [(i, prov(i)) for i in store.records(video, source, "interaction")],
            store.records(video, source, "actor"), source)


def triples(store, video: str, source: str = "adjudicated") -> list[dict]:
    evs, ints, actors, src = _records(store, video, source)
    out = []
    for e, pv in evs:
        base = {"provenance": {**pv, "record_id": e.record_id, "schema": e.schema}, "confidence": CONF[e.uncertainty["annotation_confidence"]],
                "valid_from": e.t_start, "valid_to": e.t_end, "category": e.category}
        for a in e.actors:
            out.append({**base, "source": f"event:{e.record_id}", "relation": "INVOLVES", "target": f"actor:{a}", "evidence": e.evidence})
        for ref in e.evidence:
            out.append({**base, "source": f"event:{e.record_id}", "relation": "SUPPORTED_BY", "target": _ev_node(ref), "evidence": [ref]})
    for i, pv in ints:
        base = {"provenance": {**pv, "record_id": i.record_id, "schema": i.schema}, "confidence": CONF[i.annotator_confidence],
                "valid_from": i.t_start, "valid_to": i.t_end, "observation": i.observation}
        if i.object is None:   # unary (BRAKING): grounded on each evidence frame
            for ref in i.evidence:
                out.append({**base, "source": f"actor:{i.subject}", "relation": i.relation, "target": _ev_node(ref), "evidence": [ref]})
        else:
            out.append({**base, "source": f"actor:{i.subject}", "relation": i.relation,
                        "target": "ego" if i.object == "ego" else f"actor:{i.object}", "evidence": i.evidence})
    return out


def to_graph(trs: list[dict]) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for t in trs:
        g.add_edge(t["source"], t["target"], key=f"{t['relation']}@{t['valid_from']:.2f}", **t)
    return g


def evidence_for(store, g: nx.MultiDiGraph, source: str, relation: str, target: str) -> list[dict]:
    """Every evidence frame behind (source, relation, target), with its file path and an integrity check."""
    out = []
    for u, v, d in g.out_edges(source, data=True):
        if v == target and d["relation"] == relation:
            for ref in d["evidence"]:
                out.append({**ref, "path": str(store.evidence_path(ref["video_id"], ref["image_sha256"])), "verified": store.verify_evidence(ref),
                            "valid_from": d["valid_from"], "valid_to": d["valid_to"], "provenance": d["provenance"]})
    return out


def scene_graph_validation(store, video: str, snapshots: list, source: str = "adjudicated") -> dict:
    """E6: automatic scene-graph relations vs human interactions on the same tracker ids (obj -> ego relations).
    precision = share of automatic relation steps that fall inside a human interval of the same relation and track;
    recall    = share of human interval steps where the scene graph emits that relation for that track."""
    _, ints, actors, _ = _records(store, video, source)
    tid = {a.label: a.track_id for a in actors if a.track_id is not None}
    human = [(i, tid[i.subject]) for i, _ in ints if i.relation in SG_RELATION and i.subject in tid and i.object in ("ego", None)]
    if not human:
        return {"status": "NOT EVALUATED: no human interaction with a tracker id for a modelled relation", "not_modelled": list(NOT_MODELLED)}
    res = {}
    for rel_h, rel_sg in SG_RELATION.items():
        H = [(i.t_start, i.t_end, t) for i, t in human if i.relation == rel_h]
        auto = [(s.timestamp, int(e.source_node.split(":")[1]) if e.source_node.startswith("obj:") else int(e.target_node.split(":")[1]))
                for s in snapshots for e in s.edges if e.relation_type == rel_sg and "ego" in (e.source_node, e.target_node)]
        if not H and not auto:
            continue
        inside = [any(a <= t <= b and tr == k for a, b, k in H) for t, tr in auto]
        steps = [(t, k) for a, b, k in H for t in np.arange(a, b + 1e-9, 0.1)]
        hit = [any(abs(t - s.timestamp) < 0.051 and any(e.relation_type == rel_sg and f"obj:{k}" in (e.source_node, e.target_node) for e in s.edges)
                   for s in snapshots) for t, k in steps]
        res[rel_h] = {"human_intervals": len(H), "auto_steps": len(auto), "precision": float(np.mean(inside)) if inside else None,
                      "recall": float(np.mean(hit)) if hit else None}
    return {"status": "evaluated", "source": source, "per_relation": res, "not_modelled": list(NOT_MODELLED)}
