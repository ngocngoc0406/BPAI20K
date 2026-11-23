"""v3.4 Data Collection Campaign: progress, data quality, manifest / partitions, leakage, agreement, metrics on consensus labels,
legacy-session audit and deletion log. Real recorded data only; nothing here trains or changes a model."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.realworld import campaign
from src.realworld.campaign import CAMPAIGN_DIR
from ui.common import card, setup

setup("Data Collection Campaign", "📋")
st.title("📋 Data Collection Campaign (v3.4)")
st.caption("Supervised real-driving data collection with the FROZEN v3 system. Only real driving with consent, provenance matching "
           "the frozen model and a timeline is eligible; legacy_unknown sessions are kept but excluded. Recall is reported only with "
           f"≥ {campaign.MIN_EVENTS_FOR_RECALL} confirmed (two-annotator) events per type.")

if st.button("↻ Rebuild manifest and reports (train + dev only)"):
    with st.spinner("Hashing videos, checking leakage, computing consensus metrics…"):
        man = campaign.build_manifest()
        for nm, obj in (("validation_report", campaign.validation_report(man)), ("agreement_report", campaign.agreement_report(man)),
                        ("campaign_metrics", campaign.campaign_metrics(man)), ("legacy_audit", campaign.legacy_audit())):
            (CAMPAIGN_DIR / f"{nm}.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    st.rerun()


def load(nm: str):
    p = CAMPAIGN_DIR / f"{nm}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


man, val, agr, met, legacy = (load(n) for n in ("manifest", "validation_report", "agreement_report", "campaign_metrics", "legacy_audit"))

# ---------------- v3.4.3 campaign status (read-only summary) ----------------
st.subheader("Campaign status")
real_h = float(val["real_driving_hours"]) if val else 0.0
st.markdown(f"<div style='font-size:2.1rem;font-weight:800;letter-spacing:.02em'>REAL-WORLD DRIVING HOURS: {real_h:.1f}</div>",
            unsafe_allow_html=True)
st.caption("Counts only sessions declared as real driving with trip metadata. Demo replays, Nexar clips, camera tests, the annotator "
           "training pack and the 23 legacy_unknown sessions never count.")
status = campaign.campaign_status(man)
c = st.columns(4)
card(c[0], "Eligible sessions", str(status["eligible_sessions"]), f"excluded {status['excluded_sessions']} (incl. legacy / demo)")
card(c[1], "Missing provenance", str(status["missing_provenance"]), "sessions with metadata but no provenance.json")
card(c[2], "Backup", status["backup"]["label"], status["backup"]["detail"])
card(c[3], "Privacy review", status["privacy"]["label"], status["privacy"]["detail"])
c = st.columns(4)
card(c[0], "Annotation", status["annotation"]["label"], status["annotation"]["detail"])
card(c[1], "Annotator training", status["training"]["label"], status["training"]["detail"])
card(c[2], "Dataset split", status["split"]["label"], status["split"]["detail"])
card(c[3], "final_test", status["final_test"]["label"], status["final_test"]["detail"])
rd = status["readiness"]
if rd is None:
    st.warning("Pre-campaign readiness has not been run: status **NO-GO** until it is (page *Campaign Readiness*).")
else:
    (st.success if rd["decision"]["verdict"] == "GO" else st.error)(f"Pre-campaign readiness ({rd['at']}): **{rd['decision']['verdict']}**")
    if rd["decision"]["blocking"]:
        st.dataframe(pd.DataFrame(rd["decision"]["blocking"]), hide_index=True, use_container_width=True)
st.divider()
if man is None:
    st.info("No manifest yet. Press **Rebuild** (or run `python scripts/campaign_report.py`).")
    st.stop()

c = st.columns(4)
card(c[0], "Real-driving hours", f"{val['real_driving_hours']:.2f} h", "target ≥ 10 h supervised")
card(c[1], "Sessions / trips / drivers", f"{val['real_driving_sessions']} / {val['trips']} / {val['drivers']}", "real driving")
card(c[2], "Eligible", f"{val['eligible_sessions']} ({val['eligible_hours']:.2f} h)", "frozen model, consent, provenance, timeline")
card(c[3], "Double-annotated", str(val["double_annotated_sessions"]), "two full independent reviews")
st.progress(min(val["real_driving_hours"] / 10.0, 1.0), text=f"{val['real_driving_hours']:.2f} / 10 h")
st.caption(f"Manifest built {man['created']}. Frozen reference: " + ", ".join(
    f"{k} model {v['model_sha256'][:12]}… features {v['feature_config_sha256'][:12]}… thresholds {v['threshold_config_sha256'][:12]}…"
    for k, v in man.get("frozen_reference", {}).items()))

tabs = st.tabs(["Data quality", "Partitions & leakage", "Agreement", "Metrics", "legacy_unknown", "Deletion log"])
with tabs[0]:
    rows = [{"session": e["session_id"], "kind": e["data_kind"], "trip": e["trip_id"], "driver": e["driver_id"], "hours": round(e["hours"], 3),
             "eligible": e["eligible"], "partition": e["partition"], "steps": e["timeline_steps"], "video frames": e["video_frames"],
             "missing steps": e["missing_steps"], "sync diff": None if e["sync_rel_diff"] is None else f"{e['sync_rel_diff']:.2%}",
             "provenance": "ok" if e["provenance_check"].get("eligible") else ("—" if not e["model_sha256"] else "MISMATCH"),
             "software same as frozen": e["provenance_check"].get("software_same"), "annotators": ", ".join(e["annotators"]),
             "fully reviewed": len(e["fully_reviewed_by"]), "shared copy": e["privacy_shared_copy"],
             "privacy approved": bool(e["privacy_review"] and e["privacy_review"].get("approved")), "issues": "; ".join(e["issues"])}
            for e in man["sessions"] if e["data_kind"] != "legacy_unknown"]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("No sessions with v3.4 metadata yet.")
    st.metric("Share of real-driving sessions with missing steps or desync",
              "—" if val["share_missing_or_desync"] is None else f"{val['share_missing_or_desync']:.1%}")
with tabs[1]:
    st.dataframe(pd.DataFrame([{"partition": k, **v} for k, v in val["partitions"].items()]), hide_index=True)
    lk = man["leakage"]
    (st.success if lk["ok"] else st.error)(f"Leakage / duplicate check: {'OK' if lk['ok'] else 'problems found'}")
    st.json(lk)
    st.caption("Groups = driver pseudonym, else trip, else session. Assignment is frozen in data/campaign/partitions.json; final_test is "
               "evaluated only via `scripts/campaign_report.py --final-test --by <coordinator>` (logged).")
with tabs[2]:
    if agr is None or not agr["sessions"]:
        st.info("No double-annotated, fully reviewed session yet: agreement cannot be computed.")
    else:
        st.json(agr["pooled"])
        st.dataframe(pd.DataFrame(agr["sessions"]), hide_index=True, use_container_width=True)
with tabs[3]:
    if met is None or met["double_annotated_sessions"] == 0:
        st.info("No double-annotated eligible session in train / dev: false alarms per hour and recall are not computed.")
    else:
        ov = met["overall"]
        st.markdown(f"**{', '.join(met['partitions'])}** · {ov['hours']} h · alarms {ov['alarms']} · false alarms {ov['false_alarms']} · "
                    f"FA/h {ov['fa_per_hour']} {ov['fa_per_hour_ci95']} · excluded {ov['excluded_alarms']}")
        st.dataframe(pd.DataFrame([{"type": k, **{kk: vv for kk, vv in v.items() if kk != 'lead_s_all'}} for k, v in ov["by_event_type"].items()]),
                     hide_index=True, use_container_width=True)
        for key in ("by_lighting", "by_weather", "by_road_type"):
            if met[key]:
                st.markdown(f"**{key.replace('_', ' ')}** (declared)")
                st.dataframe(pd.DataFrame([{"group": g, "hours": v["hours"], "FA/h": v["fa_per_hour"], "FA/h CI": v["fa_per_hour_ci95"],
                                            "all-event recall": v["by_event_type"]["all_events"]["recall"]} for g, v in met[key].items()]),
                             hide_index=True, use_container_width=True)
with tabs[4]:
    if legacy:
        st.caption("Kept unchanged, never given metadata, excluded from every partition and from official metrics. 'Observed origin' is "
                   "read only from the recorded source field.")
        st.dataframe(pd.DataFrame(legacy).drop(columns=["files"]), hide_index=True, use_container_width=True)
with tabs[5]:
    p = CAMPAIGN_DIR / "deletion_log.jsonl"
    if p.exists():
        st.dataframe(pd.DataFrame([json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]), hide_index=True)
    else:
        st.caption("No deletions recorded.")
