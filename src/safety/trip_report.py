"""Trip safety report: aggregates the analysed clips and stored events of one trip.

Descriptive only: counts, rates per hour, timeline, most frequent triggering cues, and grounded explanations.
No composite "safety score" is produced, because no such score has been validated.
"""
from __future__ import annotations

import io
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.safety.near_miss import MEMORY_DIR


class TripLog:
    """Per-trip list of analysed clips: data/memory/trips/<trip_id>.json"""

    def __init__(self, root: str | Path = MEMORY_DIR):
        self.dir = Path(root) / "trips"
        self.dir.mkdir(parents=True, exist_ok=True)

    def add_clip(self, trip_id: str, res) -> None:
        p = self.dir / f"{trip_id}.json"
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"trip_id": trip_id, "clips": []}
        s = res.summary()
        data["clips"] = [c for c in data["clips"] if c["video"] != s["video"]] + [{
            "video": s["video"], "duration_s": s["duration_s"], "model": Path(res.model_dir).name if res.model_dir else "",
            "profile": getattr(res, "profile", "balanced"), "n_alarms": s["n_alarms"], "max_level": s["max_level"],
            "analysed": datetime.now().isoformat(timespec="seconds")}]
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def get(self, trip_id: str) -> dict:
        p = self.dir / f"{trip_id}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"trip_id": trip_id, "clips": []}


def build_report(trip: dict, events: list[dict]) -> dict:
    hours = sum(c["duration_s"] for c in trip["clips"]) / 3600.0
    by_type = Counter(e.get("category") or e["type"] for e in events)
    cues = Counter(f["feature"] for e in events for f in e.get("factors", []))
    objs = Counter((e.get("lead_object") or {}).get("class", "none") for e in events)
    return {
        "trip_id": trip["trip_id"], "generated": datetime.now().isoformat(timespec="seconds"),
        "n_clips": len(trip["clips"]), "duration_min": hours * 60,
        "models": sorted({c["model"] for c in trip["clips"]}), "profiles": sorted({c["profile"] for c in trip["clips"]}),
        "alerts": by_type.get("alert", 0), "near_miss_candidates": by_type.get("near_miss_candidate", 0),
        "alerts_per_hour": by_type.get("alert", 0) / hours if hours > 0 else None,
        "near_miss_candidates_per_hour": by_type.get("near_miss_candidate", 0) / hours if hours > 0 else None,
        "events_by_type": dict(by_type),
        "top_cues": cues.most_common(5), "lead_object_classes": objs.most_common(),
        "events": sorted(events, key=lambda e: (e["video"], e["t_start"])),
        "clips": trip["clips"],
        "disclaimer": "Research prototype. Events are model outputs on image-plane cues, not verified incidents. "
                      "No automatic braking; no guarantee of collision avoidance.",
    }


def _latin(s: str) -> str:
    return str(s).encode("latin-1", "replace").decode("latin-1")


def report_pdf(rep: dict) -> bytes:
    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(True, 14)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 9, _latin(f"SafeDrive AI - Trip safety report: {rep['trip_id']}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    lines = [f"Generated {rep['generated']} | clips {rep['n_clips']} | analysed {rep['duration_min']:.1f} min",
             f"Models: {', '.join(rep['models'])} | warning profile(s): {', '.join(rep['profiles'])}",
             f"Alerts: {rep['alerts']} ({(rep['alerts_per_hour'] or 0):.1f}/h) | near-miss candidates: {rep['near_miss_candidates']} "
             f"({(rep['near_miss_candidates_per_hour'] or 0):.1f}/h)",
             "Most frequent triggering cues: " + ", ".join(f"{k} ({v})" for k, v in rep["top_cues"]) if rep["top_cues"] else "No triggering cues.",
             "Lead road user at event peak: " + ", ".join(f"{k} ({v})" for k, v in rep["lead_object_classes"])]
    for ln in lines:
        pdf.multi_cell(0, 5.5, _latin(ln), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "Events", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    for e in rep["events"][:200]:
        pdf.multi_cell(0, 4.2, _latin(f"[{e['type']}] {e['video']} t={e['t_start']:.1f}-{e['t_end']:.1f}s peak P={e['p_peak']:.2f} "
                                      f"({e['level_peak']}) - {e['explanation']}"), new_x="LMARGIN", new_y="NEXT")
    if not rep["events"]:
        pdf.cell(0, 6, "No events recorded for this trip.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _latin(rep["disclaimer"]), new_x="LMARGIN", new_y="NEXT")
    buf = io.BytesIO(pdf.output())
    return buf.getvalue()
