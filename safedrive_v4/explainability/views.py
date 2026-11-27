"""Three separate explanation types for a warning (never merged into one 'model explanation'):

  model_explanation       TreeSHAP of the frozen v3 model: what the MODEL computed (feature-group contributions)
  evidence_explanation    what was OBSERVED: measured track values of the road users involved, and human evidence frames of
                          annotated events that overlap the warning (only adjudicated or submitted annotations)
  rule_interpretation     knowledge-graph rules over the scene graph: a semantic READING of the situation; the v3 model does
                          NOT use these rules, so they are not the cause of the model's output
Each view says when it is unavailable instead of filling it.
"""
from __future__ import annotations


def views(trace, store=None, annotator_sources: tuple[str, ...] = ("adjudicated",)) -> dict:
    shap = [s for s in (trace.shap or []) if str(s.get("level", "")).startswith("group")]
    model = ({"available": True, "method": "exact TreeSHAP of the frozen v3 LightGBM ensemble (log-odds)", "groups": shap,
              "note": "describes the model's computation"} if shap else
             {"available": False, "reason": "no SHAP for this warning (live session or no model matrix)"})
    measured = [{"road_user": k, "type": n.get("node_type"), "confidence": n.get("confidence"), "image_ttc_s": n.get("tau_s"),
                 "scale_growth_per_s": n.get("scale_growth"), "visibility": n.get("visibility_status")}
                for k, n in (trace.nodes or {}).items() if k != "ego"]
    human = []
    if store is not None:
        vid = trace.video_id
        for src in annotator_sources:
            try:
                from safedrive_v4.annotation.kg_link import triples
                for t in triples(store, vid, src):
                    if t["relation"] == "SUPPORTED_BY" and t["valid_from"] - 1.0 <= trace.timestamp <= t["valid_to"] + 1.0:
                        human.append({"event": t["source"], "category": t.get("category"), "frame_t": t["evidence"][0]["t"],
                                      "image_sha256": t["evidence"][0]["image_sha256"], "provenance": t["provenance"]})
            except Exception:  # noqa: BLE001
                continue
    evidence = {"available": bool(measured or human), "measured_tracks": measured, "human_evidence_frames": human,
                "note": "observations only; human evidence appears only for adjudicated annotations"}
    rules = ({"available": True, "paths": trace.kg_paths, "findings": [{"rule": f["rule_id"], "explanation": f["explanation"]} for f in trace.findings],
              "note": "rule-based interpretation; NOT used by the v3 model"} if trace.kg_paths else
             {"available": False, "reason": "no knowledge-graph rule fired at this warning"})
    return {"model_explanation": model, "evidence_explanation": evidence, "rule_interpretation": rules}
