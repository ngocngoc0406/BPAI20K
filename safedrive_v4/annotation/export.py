"""Versioned dataset export (coordinator only, privacy-gated). Counterfactual records are NEVER exported (research only).

Files written to <dest>/<export_id>/:
  annotations.json        schema + per-video ground-truth records: adjudicated records when they exist, otherwise the submitted
                          records of each annotator marked "single annotator (not adjudicated)"; never drafts, never model output
  summary.csv             one row per video: events per category, interactions, annotators, adjudicated?, uncertain share
  evidence_manifest.json  evidence references (video, t, frame, sha256, verified) -- hashes only, image files are NOT copied
  provenance.json         export id, schema, coordinator, time, store op-chain heads + verification, submission hashes
  split_manifest.json     video -> split (caller-provided; default 'unassigned'), so held-out membership travels with the data
Gate: coordinator; the written sharing policy must be enabled; every video must be public (e.g. Nexar) or privacy-approved
(`approved(video)`); otherwise the export is refused as a whole and nothing is written.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from safedrive_v4.annotation.schema import SCHEMA, new_id, now


class ExportRefused(PermissionError):
    pass


def _gt(store, v: str) -> tuple[list[dict], str]:
    adj = [a for a in store.adjudications(v) if a["record"] is not None and a["record_type"] in ("TemporalEvent", "Interaction")]
    if adj:
        return [{"record_type": a["record_type"], **a["record"], "provenance": {"adjudication": a["adj_id"], "decision": a["decision"]}}
                for a in adj], "adjudicated"
    out = []
    for ann in store.annotators(v):
        if not store.submitted(v, ann):
            continue
        for rt, cls in (("event", "TemporalEvent"), ("interaction", "Interaction"), ("actor", "Actor")):
            out += [{"record_type": cls, **asdict(r), "provenance": {"annotator": ann, "status": "single annotator (not adjudicated)"}}
                    for r in store.records(v, ann, rt)]
    return out, "submitted_only"


def nexar_split_of() -> dict:
    """video -> 'dev' | 'held_out' from the frozen split file (fold ids are not exported: they are label-stratified)."""
    from src.utils.config import resolve
    p = resolve("data/processed/splits_full.json")
    if not p.exists():
        return {}
    sp = json.loads(p.read_text(encoding="utf-8"))
    out = {v: "dev" for k in ("train", "val", "dev_new") for v in sp.get(k, [])}
    out.update({v: "held_out" for k in ("test", "test_ext") for v in sp.get(k, [])})
    return out


def export(store, dest: str | Path, coordinator: str, videos: list[str] | None = None, split_of: dict | None = None,
           public: set | None = None, approved=None, policy_path: Path | None = None) -> dict:
    from src.realworld import privacy
    if not store.is_coordinator(coordinator):
        raise ExportRefused("only a coordinator exports")
    if not privacy.sharing_policy(policy_path).get("sharing_enabled", True):
        raise ExportRefused("the written sharing policy disables sharing")
    vids = videos if videos is not None else [v for v in store.videos() if any(store.submitted(v, a) for a in store.annotators(v))]
    public, approved = public or set(), approved or (lambda v: False)
    blocked = [v for v in vids if v not in public and not approved(v)]
    if blocked:
        raise ExportRefused(f"not public and not privacy-approved: {blocked[:5]}{'...' if len(blocked) > 5 else ''}")
    eid = new_id("exp")
    out = Path(dest) / eid
    out.mkdir(parents=True, exist_ok=False)
    ann, ev_man, rows, heads = {}, [], [], {}
    for v in vids:
        recs, status = _gt(store, v)
        assert not any(r.get("research_only") for r in recs)          # counterfactuals can never get here
        ann[v] = {"status": status, "records": recs}
        for r in recs:
            for e in r.get("evidence") or []:
                ev_man.append({**{k: e.get(k) for k in ("video_id", "t", "frame_index", "image_sha256", "track_id")},
                               "record_id": r["record_id"], "verified": store.verify_evidence(e)})
        evs = [r for r in recs if r["record_type"] == "TemporalEvent"]
        cats = {c: sum(r["category"] == c for r in evs) for c in ("normal", "risk_event", "near_miss", "collision", "uncertain")}
        rows.append({"video_id": v, **cats, "interactions": sum(r["record_type"] == "Interaction" for r in recs),
                     "annotators": len([a for a in store.annotators(v) if store.submitted(v, a)]), "status": status,
                     "split": (split_of or {}).get(v, "unassigned")})
        for a in store.annotators(v):
            o = store.ops(v, a)
            heads[f"{v}/{a}"] = {"head": o[-1]["hash"] if o else None, "chain": store.verify_chain(v, a)["ok"]}
    (out / "annotations.json").write_text(json.dumps({"schema": SCHEMA, "export_id": eid, "videos": ann}, indent=1, default=str), encoding="utf-8")
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["video_id"])
        w.writeheader()
        w.writerows(rows)
    (out / "evidence_manifest.json").write_text(json.dumps({"note": "hashes only; images not copied", "evidence": ev_man}, indent=1), encoding="utf-8")
    prov = {"export_id": eid, "schema": SCHEMA, "coordinator": coordinator, "at": now(), "videos": len(vids), "op_chains": heads,
            "submissions": [s for s in store.submissions() if s["video_id"] in set(vids)], "counterfactuals": "excluded (research only)"}
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, default=str), encoding="utf-8")
    (out / "split_manifest.json").write_text(json.dumps({v: (split_of or {}).get(v, "unassigned") for v in vids}, indent=1), encoding="utf-8")
    with open(store.root / "exports.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"export_id": eid, "coordinator": coordinator, "at": prov["at"], "videos": vids, "dest": str(out)}) + "\n")
    return {"export_id": eid, "path": str(out), "videos": len(vids), "evidence": len(ev_man)}
