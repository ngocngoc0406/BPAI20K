"""Ground-truth annotation (schema v3.4): two independent, BLIND annotators label situations from the full-session video;
a coordinator measures agreement, records disagreement reasons (no forced agreement), prepares privacy-reviewed shared copies
and handles deletion requests."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from src.realworld import annotation as ann
from src.realworld import privacy, training
from src.realworld.trip_meta import DATA_KINDS, SESSIONS_DIR, delete_session, list_sessions, load_meta, save_meta
from ui.common import setup

setup("Ground Truth Annotation", "🏷️")
st.title("🏷️ Ground Truth Annotation")
st.caption("Label what **you see in the video**, never what the model said. Annotators work **blind**: no model output, no event "
           "clips (they reveal where the model warned) and no other annotator's labels. Insufficient evidence → **uncertain**. "
           "Annotate only after the trip, never while driving.")

workspace = st.radio("Workspace", ["Campaign sessions", "Annotator training (practice pack, never evaluation data)"], horizontal=True)
TRAINING = workspace.startswith("Annotator")
ROOT = training.TRAINING_SESSIONS if TRAINING else SESSIONS_DIR
if TRAINING:
    st.warning(f"**{training.MARK}.** Practise blind labelling here. Results measure annotator readiness only and never enter the "
               "campaign manifest, agreement report or metrics.")
sessions = list_sessions(ROOT)
if not sessions:
    st.info("No sessions yet. Record one on the **Live Driving Assistant** page.")
    st.stop()


def label(sid: str) -> str:
    m = load_meta(sid, ROOT)
    return f"{sid} · {m.data_kind if m else 'legacy_unknown'}"


sid = st.selectbox("Session", sessions, index=len(sessions) - 1, format_func=label)
d = ROOT / sid
meta = load_meta(sid, ROOT)
name = st.text_input("Your name / annotator code (each person annotates separately)", key="annotator").strip()
role = st.radio("Role", ["Annotator", "Coordinator"], horizontal=True)
if meta is None:
    st.warning("legacy_unknown session (no trip metadata): it is not part of the campaign and cannot be annotated for evaluation.")
    st.stop()
st.caption(f"Data kind: **{DATA_KINDS.get(meta.data_kind, meta.data_kind)}** · trip {meta.trip_id or '—'} · driver {meta.driver_id or '—'} · "
           f"declared lighting **{meta.lighting}**, weather **{meta.weather}**, road **{', '.join(meta.road_types) or 'unknown'}**, "
           f"traffic **{meta.traffic}** · fully reviewed by {len(meta.fully_reviewed_by)} annotator(s)")

raw_ok = privacy.can_view_raw(name)
shared_ok = privacy.shareable(d)
src_dir = d / "video" if raw_ok else (d / "shared" if shared_ok else None)
segs = sorted(src_dir.glob("segment_*.mp4")) if src_dir else []
seg_info = json.loads((d / "video" / "segments.json").read_text(encoding="utf-8")) if (d / "video" / "segments.json").exists() else {}


def seg_start(k: int) -> float:
    s = seg_info.get("segments", [])
    return float(s[k]["t_start"]) if k < len(s) else 0.0


def frame_at(t: float):
    """Frame of the (raw or shared) video at session time t."""
    starts = [seg_start(k) for k in range(len(segs))]
    k = max([i for i, s0 in enumerate(starts) if s0 <= t] or [0])
    cap = cv2.VideoCapture(str(segs[k]))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t - starts[k]) * 1000)
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


# =============================== annotator ===============================
if role == "Annotator":
    if not name:
        st.info("Enter your name / code first.")
        st.stop()
    if not segs:
        st.error("No video you are allowed to view: " + ("this session has no full-session video." if not (d / "video").exists() else
                 "you are not on the raw-access list (data/campaign/access.json) and no privacy-approved shared copy exists."))
        st.stop()
    st.caption(f"Viewing the **{'raw' if raw_ok else 'blurred shared'}** video.")
    k = st.selectbox("Segment", range(len(segs)), format_func=lambda i: f"segment {i} — starts at session time {seg_start(i):.1f} s")
    st.video(str(segs[k]))
    st.caption(f"Session time = {seg_start(k):.1f} s + video position.")

    locked = TRAINING and training.is_locked(d, name)
    if locked:
        st.success("You SUBMITTED the training pack: your labels are locked (any change would invalidate the result). "
                   "The coordinator sees the result when both annotators have submitted.")
        st.stop()
    st.subheader("Add a situation")
    with st.form("ann", clear_on_submit=True):
        f1, f2, f3 = st.columns(3)
        cat = f1.selectbox("Category", ann.CATEGORIES, index=2,
                           help="collision: contact; near_miss: evasive action or very small gap; risk_event: a warning would have "
                                "helped; normal: a reviewed stretch with nothing notable; uncertain: insufficient evidence")
        conf = f1.selectbox("Confidence", ann.CONFIDENCE, index=1)
        on_t = f2.number_input("Start (s)", 0.0, step=0.1, format="%.1f")
        ev_t = f2.number_input("Event time (s): impact / closest approach / latest moment a warning was needed", 0.0, step=0.1, format="%.1f")
        end_t = f3.number_input("End (s)", 0.0, step=0.1, format="%.1f")
        tags = f3.multiselect("Tags", ann.TAGS)
        evidence = st.text_input("Visual evidence (what in the image shows it)")
        ev_frames = st.text_input("Evidence frame times (s, comma separated)", placeholder="e.g. 123.4, 124.0")
        note = st.text_input("Note")
        if st.form_submit_button("Save", type="primary"):
            try:
                times = [float(x) for x in ev_frames.replace(";", ",").split(",") if x.strip()]
                a = ann.add(d, ann.Annotation(sid, name, cat, float(on_t), float(end_t),
                                              float(ev_t) if (ev_t > 0 or cat in ann.EVENT_CATEGORIES) else None,
                                              tags, conf, note, evidence, times))
                out = d / "evidence"
                out.mkdir(exist_ok=True)
                for t in times:
                    fr = frame_at(t)
                    if fr is not None:
                        cv2.imwrite(str(out / f"{a.annot_id}_{t:.1f}{'' if raw_ok else '_shared'}.jpg"), fr)
                st.success("Saved.")
            except (ann.AnnotationError, ValueError) as ex:
                st.error(str(ex))

    mine = ann.load(d, name)
    if mine:
        st.dataframe(pd.DataFrame([asdict(a) for a in mine]).drop(columns=["session_id", "created", "schema"]),
                     hide_index=True, use_container_width=True)
        c1, c2 = st.columns(2)
        rm = c1.selectbox("Delete one of my annotations", [a.annot_id for a in mine])
        if c1.button("Delete annotation"):
            ann.delete(d, name, rm)
            st.rerun()
    if TRAINING:
        st.markdown("**Submit when finished.** Work alone; do not look at anyone else's labels. After submitting, your labels are "
                    "locked by their SHA-256 and cannot be changed.")
        if st.button("📤 Submit my training labels (final, once)"):
            try:
                training.submit(d, name)
                st.rerun()
            except training.TrainingError as ex:
                st.error(str(ex))
        st.stop()
    if name not in meta.fully_reviewed_by:
        if st.button("✔ I watched the WHOLE session and labelled every situation (needed for false alarms and recall)"):
            meta.fully_reviewed_by.append(name)
            save_meta(meta, ROOT)
            st.rerun()
    else:
        st.success("You marked this session as fully reviewed.")
    st.stop()

# =============================== coordinator ===============================
if not privacy.is_coordinator(name):
    st.error("Coordinator functions need a name on the coordinators list in data/campaign/access.json.")
    st.stop()
done = [n for n in meta.fully_reviewed_by if n in ann.annotators(d)]
if TRAINING:
    st.subheader("Training result (against the hidden key)")
    trainees = list(privacy.load_access().get("raw_access", []))
    subs_ = training.submissions(d.name)
    if not training.key_visible(d, trainees):
        st.info(f"Shown when every annotator on the raw-access list ({', '.join(trainees) or 'none'}) has SUBMITTED "
                f"(now: {', '.join(n for n in trainees if n in subs_) or 'none'}). The key stays hidden until then.")
    else:
        rep_ = training.report(d, trainees, submissions_path=training.SUBMISSIONS)
        if rep_.get("invalid"):
            st.error(f"Invalid submissions: {rep_['invalid']}")
        training.save_report(rep_)
        (st.success if rep_["passed"] else st.error)(f"Training {'PASSED' if rep_['passed'] else 'NOT PASSED'} (criteria in "
                                                     "docs/SAFEDRIVE_V343_ANNOTATOR_TRAINING.md)")
        st.dataframe(pd.DataFrame([{"annotator": n, **{k: v for k, v in r.items() if k != "criteria"}, **{f"✔ {k}": v for k, v in r["criteria"].items()}}
                                   for n, r in rep_["per_annotator"].items()]).astype(str), hide_index=True, use_container_width=True)
        st.json(rep_["pair"])
        st.dataframe(pd.DataFrame(training.load_key(d)["clips"]).astype(str), hide_index=True, use_container_width=True)
st.subheader("Agreement and disagreements")
if len(done) < 2:
    st.info(f"Needs two annotators who finished a full review (now: {', '.join(done) or 'none'}). Results stay hidden until then "
            "so annotators remain independent.")
else:
    a1, a2 = st.columns(2)
    n1 = a1.selectbox("Annotator A", done, index=0)
    n2 = a2.selectbox("Annotator B", done, index=1)
    A, B = ann.load(d, n1), ann.load(d, n2)
    r = ann.agreement(A, B)
    c = st.columns(4)
    c[0].metric("Matched situations", r["matched"])
    c[1].metric("Only A / only B", f"{r['only_first']} / {r['only_second']}")
    c[2].metric("κ (with unmatched)", "—" if pd.isna(r["kappa_with_unmatched"]) else f"{r['kappa_with_unmatched']:.2f}")
    c[3].metric("κ event vs not", "—" if pd.isna(r["kappa_event_vs_not"]) else f"{r['kappa_event_vs_not']:.2f}")
    st.caption(f"Event-time offset: {r['offset_event_time']} · start offset: {r['offset_start']} · end offset: {r['offset_end']}")
    cons = ann.consensus(A, B, ann.load_disagreements(d))
    st.dataframe(pd.DataFrame(cons), hide_index=True, use_container_width=True)
    open_items = [x for x in cons if x["status"] != "confirmed"]
    if open_items:
        st.markdown("**Record a disagreement reason** (each annotator gives their own reason; agreement is never forced).")
        with st.form("dis"):
            key = st.selectbox("Situation", [x["key"] for x in open_items],
                               format_func=lambda k: next(f"{x['category']} @ {x['risk_onset']:.1f}s ({x['status']})" for x in open_items if x["key"] == k))
            who = st.selectbox("Reason given by", [n1, n2])
            reason = st.text_area("Reason")
            status = st.selectbox("Status", ["open", "unresolved", "agreed_after_discussion"])
            agreed = st.selectbox("Agreed category (only if agreed after discussion)", [""] + ann.CATEGORIES)
            if st.form_submit_button("Save"):
                try:
                    ann.add_disagreement(d, ann.Disagreement(sid, key, who, reason, status, agreed))
                    st.rerun()
                except ann.AnnotationError as ex:
                    st.error(str(ex))

st.subheader("Privacy: shared (blurred) copy")
st.caption("Raw video stays on this computer. A blurred copy can be shared within the research team only after a human review. "
           "No face / plate detector exists: blurring is conservative and region-based (see docs/SAFEDRIVE_V34_PRIVACY_PROTOCOL.md).")
if (d / "video").exists():
    if st.button("Create / re-create the blurred shared copy (runs the detector on every frame; may take minutes)"):
        with st.spinner("Blurring…"):
            log = privacy.make_shared_copy(d)
        st.success(f"Shared copy written: {sum(s['frames'] for s in log['segments'])} frames.")
    shared = sorted((d / "shared").glob("segment_*.mp4")) if (d / "shared").exists() else []
    if shared:
        k2 = st.selectbox("Check shared segment", range(len(shared)), format_func=lambda i: shared[i].name)
        st.video(str(shared[k2]))
        st.caption(f"Status: **{privacy.shared_status(d)}** (pending_review → approved / rejected; stale = changed after the "
                   "review; revoked = withdrawn)")
        with st.form("rev"):
            checked = st.multiselect("Segments I watched completely, frame by frame where needed", [p.name for p in shared])
            finding = st.radio("Is any face or licence plate readable in ANY checked segment?",
                               ["not answered", "No - nothing readable", "Yes - something is readable (reject)"])
            notes = st.text_input("Notes (where / what was readable; no plate numbers)")
            if st.form_submit_button("Save privacy review"):
                try:
                    found = None if finding == "not answered" else finding.startswith("Yes")
                    r_ = privacy.review(d, name, found is False, checked, notes, identifiable_found=found)
                    (st.success if r_["approved"] else st.warning)(f"Review saved: {r_['decision']}.")
                except (ValueError, FileNotFoundError, PermissionError) as ex:
                    st.error(str(ex))
        st.caption(f"Shareable: **{privacy.shareable(d)}** · sharing policy enabled: {privacy.sharing_policy().get('sharing_enabled')}")
        with st.form("revoke"):
            st.markdown("**Revoke this shared copy** (blur found insufficient): blocks further export, lists earlier exports.")
            why_r = st.text_input("Reason")
            del_e = st.checkbox("Also delete the exported folders that are still reachable")
            if st.form_submit_button("Revoke"):
                try:
                    rr = privacy.revoke_shared(d, name, why_r, del_e)
                    st.warning(f"Revoked. Earlier exports: {rr['earlier_exports'] or 'none'}; deleted: {rr['deleted_exports'] or 'none'}; "
                               f"contact holders of: {rr['unreachable_exports'] or 'none'}")
                except (ValueError, FileNotFoundError, PermissionError) as ex:
                    st.error(str(ex))
        with st.form("export"):
            st.markdown("**Export the approved blurred copy** (the only sanctioned way to share; raw video is never exported)")
            dest = st.text_input("Destination folder (inside the research team's storage, never a public cloud)")
            if st.form_submit_button("Export", disabled=not privacy.shareable(d)):
                try:
                    rec = privacy.export_shared(d, Path(dest), name)
                    st.success(f"Exported {len(rec['files_sha256'])} verified files to {rec['destination']} (logged).")
                except (privacy.ExportRefused, FileExistsError, OSError) as ex:
                    st.error(f"Export refused: {ex}")
else:
    st.caption("No full-session video in this session.")

with st.expander("🗑️ Delete this trip's data (on request)"):
    st.write("Deletes the session folder (timeline, raw + shared video, annotations, evidence, report) and its event clips and "
             "metadata in Near-Miss Memory. A deletion record (ids and counts only) is kept in data/campaign/deletion_log.jsonl.")
    who = st.text_input("Requested by (pseudonym / role)")
    why = st.text_input("Reason")
    okd = st.text_input(f"Type the session id **{sid}** to confirm")
    if st.button("Delete permanently", disabled=okd != sid or not who.strip(), type="primary"):
        st.success(f"Deleted: {delete_session(sid, root=ROOT, requested_by=who, reason=why)}")
        st.rerun()
