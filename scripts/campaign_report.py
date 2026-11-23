"""v3.4 campaign report: dataset manifest (+ partitions, hashes, leakage / duplicate checks), validation report, annotation
agreement report, legacy-session audit and campaign metrics on consensus labels. Never trains or changes a model.

Outputs in data/campaign/: manifest.json, validation_report.json, agreement_report.json, legacy_audit.json,
campaign_metrics.json (train + dev only unless --final-test) and collection_summary.md.

Usage:
  python scripts/campaign_report.py                 # train + dev metrics
  python scripts/campaign_report.py --final-test --by "<coordinator>"   # also final_test (logged); only at campaign end
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.realworld import campaign  # noqa: E402
from src.realworld.campaign import CAMPAIGN_DIR  # noqa: E402
from src.realworld.trip_meta import SESSIONS_DIR  # noqa: E402

TARGET_HOURS = 10.0


def summary_md(val: dict, agr: dict, met: dict, legacy: list) -> str:
    ov = met["overall"]
    lines = ["# SafeDrive v3.4 campaign: generated summary", "",
             f"- Real-driving sessions: **{val['real_driving_sessions']}**, hours: **{val['real_driving_hours']:.2f} / {TARGET_HOURS:.0f} h**, "
             f"trips: {val['trips']}, drivers: {val['drivers']}",
             f"- Eligible sessions (frozen model, consent, provenance, timeline): {val['eligible_sessions']} ({val['eligible_hours']:.2f} h)",
             f"- Double-annotated (two full independent reviews): {val['double_annotated_sessions']}",
             f"- Sessions with missing steps or video/timeline desync: {val['sessions_with_missing_steps']} / {val['sessions_desynchronised']}",
             f"- Privacy-approved shared copies: {val['privacy_reviewed_shared_copies']}",
             f"- Partitions: " + ", ".join(f"{k} {v['sessions']} sessions / {v['hours']:.2f} h" for k, v in val["partitions"].items()),
             f"- Leakage / duplicate check: {'OK' if val['leakage_ok'] else 'PROBLEMS: ' + json.dumps(val['leakage'])}",
             f"- legacy_unknown sessions (kept, excluded): {len(legacy)}", "",
             f"## Metrics ({', '.join(met['partitions'])})",
             f"- Hours: {ov['hours']}, alarms {ov['alarms']}, false alarms {ov['false_alarms']}, FA/h {ov['fa_per_hour']} "
             f"[95 % CI {ov['fa_per_hour_ci95']}], excluded (disputed / single / uncertain) {ov['excluded_alarms']}"]
    for cat, o in ov["by_event_type"].items():
        lines.append(f"- {cat}: confirmed events {o['confirmed_events']}, recall {o['recall']}"
                     + (f" {o.get('recall_ci95')}, warned ≥1 s {o.get('warned_ge_1s')}, ≥2 s {o.get('warned_ge_2s')}" if "recall_ci95" in o else ""))
    lines += ["", "## Agreement", f"- {agr['note'] or json.dumps(agr['pooled'])}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--by", default="")
    ap.add_argument("--no-phash", action="store_true")
    a = ap.parse_args()
    if a.final_test and not a.by.strip():
        raise SystemExit("--final-test requires --by <coordinator> (access is logged)")
    man = campaign.build_manifest(SESSIONS_DIR, CAMPAIGN_DIR, with_phash=not a.no_phash)
    val = campaign.validation_report(man)
    agr = campaign.agreement_report(man)
    met = campaign.campaign_metrics(man, include_final_test=a.final_test, requested_by=a.by)
    legacy = campaign.legacy_audit(SESSIONS_DIR)
    for name, obj in (("validation_report", val), ("agreement_report", agr), ("campaign_metrics", met), ("legacy_audit", legacy)):
        (CAMPAIGN_DIR / f"{name}.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (CAMPAIGN_DIR / "collection_summary.md").write_text(summary_md(val, agr, met, legacy), encoding="utf-8")
    print(summary_md(val, agr, met, legacy))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
