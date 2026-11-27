"""Annotation Intelligence: evidence-grounded, uncertainty-aware, multi-annotator annotation (schema ann-v4.0).

Blindness: an annotator never sees model predictions, risk levels, SHAP or another annotator's labels on this page. The active
queue shows the annotator only the ORDER; its reasons (model uncertainty) and agreement / adjudication / export are coordinator
views. All data is written to the encrypted store only (V:)."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from safedrive_v4.annotation.schema import (ACTOR_CLASSES, BLUR, CATEGORIES, CONFIDENCE, LIGHTING, OBSERVATION,  # noqa: E402
                                            OCCLUSION, RELATIONS, VISIBILITY, Actor, Counterfactual, Interaction,
                                            TemporalEvent, ValidationError)
from safedrive_v4.app.ui import frame_at, page  # noqa: E402

page("Annotation Intelligence", "🏷️")
st.title("🏷️ Annotation Intelligence")
st.caption("Human annotation with evidence frames, uncertainty and provenance. The model never writes a label, and annotators "
           "do not see model output or each other's labels.")

try:
    from safedrive_v4.annotation.store import AccessError, AnnotationStore
    S = AnnotationStore()
except AccessError as e:  # noqa: F841
    st.error("The encrypted session volume (VeraCrypt V:) is not mounted: annotation data is never written unencrypted. "
             "Mount D:\\SafeDriveVault.hc as V: and reload.")
    st.stop()


@st.cache_data(show_spinner=False)
def nexar_videos() -> dict:
    from src.data.dataset_loader import load_dataset
    from src.utils.config import load_config, resolve
    c = load_config()
    return {r.video_id: {"path": str(r.path),
                         "meta": {k: r.meta.get(k) for k in ("light_conditions", "weather", "scene")}}
            for r in load_dataset(resolve(c["paths"]["annotations"]), resolve(c["paths"]["raw_dir"]))}


@st.cache_data(show_spinner=False)
def video_info(path: str) -> tuple[float, float]:
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or fps * 40
    cap.release()
    return float(n / fps), float(fps)


VIDS = nexar_videos()
c1, c2, c3 = st.columns([2, 2, 3])
role = c1.radio("Role", ["Annotator", "Coordinator"], horizontal=True)
who = c2.text_input("Pseudonym (no real name)", key="ann_who").strip()
if not who:
    st.info("Enter your pseudonym to start (e.g. ANN-A).")
    st.stop()
try:
    S.check_annotator(who)
except AccessError as e:
    st.error(str(e))
    st.stop()
is_coord = role == "Coordinator"
if is_coord and not S.is_coordinator(who):
    st.error("This pseudonym is not on the coordinator list (data/campaign/access.json).")
    st.stop()

TABS = ["Video browser", "Timeline & boundaries", "Interactions", "Evidence", "Counterfactual (research)", "Active queue"]
if is_coord:
    TABS += ["Agreement", "Dataset quality", "Adjudication", "Export"]
T = dict(zip(TABS, st.tabs(TABS)))

# ------------------------------------------------------------------ 1 video browser
with T["Video browser"]:
    done = {v for v in S.videos() if S.submitted(v, who)}
    started = set(S.videos())
    flt = st.radio("Show", ["all", "not started by me", "my drafts", "my submitted"], horizontal=True)
    mine = {v for v in started if S.ops(v, who)}
    ids = sorted(VIDS)
    ids = {"all": ids, "not started by me": [v for v in ids if v not in mine], "my drafts": sorted(mine - done),
           "my submitted": sorted(done)}[flt]
    if not ids:
        st.info("Nothing in this list.")
    else:
        cur = st.session_state.get("ann_video")
        vid = st.selectbox("Video", ids, index=ids.index(cur) if cur in ids else 0, key="ann_pick")
        st.session_state["ann_video"] = vid
        st.caption(f"Conditions: {VIDS[vid]['meta']} · my status: "
                   f"{'submitted' if vid in done else ('draft' if vid in mine else 'not started')}")
VID = st.session_state.get("ann_video")
if not VID:
    st.stop()
PATH = VIDS[VID]["path"]
DUR, FPS = video_info(PATH)
DUR = round(DUR - 0.05, 1)
SUB = S.submitted(VID, who)
st.session_state.setdefault("ann_ev", {})
PEND = st.session_state["ann_ev"].setdefault(VID, [])     # evidence captured but not yet attached


def _step(k: str, d: float) -> None:
    st.session_state[k] = float(min(DUR, max(0.0, round(st.session_state.get(k, 0.0) + d, 1))))


def _set(dst: str, src: str) -> None:
    st.session_state[dst] = round(float(st.session_state.get(src, 0.0)), 1)


def _frame_viewer(key: str) -> float:
    st.session_state.setdefault(f"t_{key}", 0.0)
    t = st.slider("Time (s)", 0.0, float(DUR), step=0.1, key=f"t_{key}")
    a, b = st.columns(2)
    a.button("◀ −0.1 s", key=f"m_{key}", on_click=_step, args=(f"t_{key}", -0.1))
    b.button("+0.1 s ▶", key=f"p_{key}", on_click=_step, args=(f"t_{key}", 0.1))
    fr = frame_at(PATH, t)
    if fr is None:
        st.warning("Video file not available.")
    else:
        st.image(fr[:, :, ::-1], caption=f"{VID} @ {t:.1f} s", use_container_width=True)
        note = st.text_input("Evidence note (what is visible)", key=f"n_{key}")
        if st.button("📸 Capture this frame as evidence", key=f"c_{key}"):
            full = frame_at(PATH, t, 1280)
            PEND.append(S.save_evidence(VID, full, round(t, 2), int(round(t * FPS)), note=note))
            st.success(f"Captured {t:.1f} s ({len(PEND)} pending)")
    return t


def _pending_pick(key: str) -> list:
    if not PEND:
        st.caption("No captured evidence yet (capture frames above).")
        return []
    lab = {i: f"{e['t']:.1f} s · {e['image_sha256'][:8]} · {e.get('note', '')}" for i, e in enumerate(PEND)}
    return [PEND[i] for i in st.multiselect("Attach evidence frames", list(lab), default=list(lab), format_func=lab.get, key=key)]


def _save(rtype: str, rec) -> None:
    reason = st.session_state.get("rev_reason", "")
    try:
        S.put(rtype, rec, reason)
        st.success(f"Saved {rtype} {rec.record_id}")
    except ValidationError as e:
        st.error(f"Not saved: {e}")


if SUB:
    st.text_input("This video is submitted: a revision needs a reason (kept in the history)", key="rev_reason")

# ------------------------------------------------------------------ 2 timeline / boundaries / uncertainty
with T["Timeline & boundaries"]:
    L, R = st.columns([3, 2])
    with L:
        t = _frame_viewer("tl")
    with R:
        for k in ("s", "p", "e"):
            st.session_state.setdefault(f"b_{k}", 0.0)
        b1, b2, b3 = st.columns(3)
        b1.button("Set start", key="bs", on_click=_set, args=("b_s", "t_tl"))
        b2.button("Set peak", key="bp", on_click=_set, args=("b_p", "t_tl"))
        b3.button("Set end", key="be", on_click=_set, args=("b_e", "t_tl"))
        ts_ = st.number_input("Start (s)", 0.0, DUR, step=0.1, key="b_s")
        tp_ = st.number_input("Peak (s)", 0.0, DUR, step=0.1, key="b_p")
        te_ = st.number_input("End (s)", 0.0, DUR, step=0.1, key="b_e")
        cat = st.selectbox("Category", CATEGORIES, key="cat")
        obs = st.selectbox("Observation", OBSERVATION, key="obs")
        acts = [a.label for a in S.records(VID, who, "actor")]
        sel_acts = st.multiselect("Actors involved", acts, key="ev_acts")
        rat = st.text_area("Rationale: what is SEEN (not what the model says)", key="rat")
        st.markdown("**Uncertainty**")
        u1, u2 = st.columns(2)
        unc = {"visibility_level": u1.selectbox("Visibility", VISIBILITY, key="u_v"),
               "occlusion_level": u2.selectbox("Occlusion", OCCLUSION, key="u_o"),
               "motion_blur": u1.selectbox("Motion blur", BLUR, key="u_b"),
               "lighting_quality": u2.selectbox("Lighting", LIGHTING, key="u_l"),
               "annotation_confidence": u1.selectbox("My confidence", CONFIDENCE, index=1, key="u_c"),
               "uncertainty_reason": st.text_input("Uncertainty reason (needed for uncertain / low confidence / poor visibility)", key="u_r")}
        evs = _pending_pick("ev_pick")
        if st.button("💾 Save temporal record", type="primary"):
            _save("event", TemporalEvent(VID, who, cat, ts_, te_, tp_ if cat not in ("normal",) else None, sel_acts, evs, rat, obs, unc))
        with st.expander("Actors"):
            a1, a2 = st.columns(2)
            lbl = a1.text_input("Label (e.g. A)", key="a_lbl")
            cls = a2.selectbox("Class", ACTOR_CLASSES, index=1, key="a_cls")
            tid = a1.number_input("Tracker id (−1 = none)", -1, 10_000, -1, key="a_tid")
            box = a2.text_input("Box x1,y1,x2,y2 (0-1) at the current time", key="a_box")
            if st.button("Add actor"):
                bx = [float(x) for x in box.split(",")] if box.count(",") == 3 else None
                _save("actor", Actor(VID, who, lbl, cls, None if tid < 0 else int(tid), round(t, 1) if bx else None, bx))
    mine_ev = S.records(VID, who, "event")
    st.markdown("**My temporal records**")
    st.dataframe(pd.DataFrame([{"id": e.record_id, "category": e.category, "start": e.t_start, "peak": e.t_peak, "end": e.t_end,
                                "evidence": len(e.evidence), "confidence": e.uncertainty["annotation_confidence"]} for e in mine_ev]),
                 hide_index=True, use_container_width=True)
    if mine_ev:
        rid = st.selectbox("Record", [e.record_id for e in mine_ev], key="hist_rid")
        h = S.history(VID, who, rid)
        st.dataframe(pd.DataFrame([{"seq": o["seq"], "op": o["op"], "at": o["at"], "reason": o.get("reason", "")} for o in h]), hide_index=True)
        r1, r2 = st.columns(2)
        why = r1.text_input("Reason (retract / restore)", key="hist_why")
        if r1.button("Retract"):
            try:
                S.retract(VID, who, rid, why)
                st.rerun()
            except ValidationError as e:
                st.error(str(e))
        seq = r2.selectbox("Restore version (seq)", [o["seq"] for o in h if "record" in o], key="hist_seq")
        if r2.button("Restore"):
            S.restore(VID, who, rid, seq, why)
            st.rerun()
    if not SUB and st.button("✅ Submit this video (locks it; revisions then need a reason)"):
        try:
            S.submit(VID, who)
            st.rerun()
        except ValidationError as e:
            st.error(str(e))

# ------------------------------------------------------------------ 3 interactions
with T["Interactions"]:
    L, R = st.columns([3, 2])
    with L:
        t = _frame_viewer("ia")
    with R:
        acts = [a.label for a in S.records(VID, who, "actor")]
        if not acts:
            st.info("Add actors first (Timeline tab → Actors).")
        else:
            subj = st.selectbox("Subject", acts, key="i_s")
            rel = st.selectbox("Relation", RELATIONS, key="i_r")
            objs = ["ego", "(none: unary)"] + [a for a in acts if a != subj]
            obj = st.selectbox("Object", objs, key="i_o")
            i1, i2 = st.columns(2)
            it0 = i1.number_input("From (s)", 0.0, DUR, step=0.1, key="i_t0")
            it1 = i2.number_input("To (s)", 0.0, DUR, step=0.1, key="i_t1")
            iconf = st.selectbox("My confidence", CONFIDENCE, index=1, key="i_c")
            iobs = st.selectbox("Observation", OBSERVATION, key="i_obs")
            evts = {e.record_id: f"{e.category} {e.t_start}-{e.t_end}" for e in S.records(VID, who, "event")}
            ieid = st.selectbox("Part of event", [None] + list(evts), format_func=lambda k: "—" if k is None else evts[k], key="i_e")
            irat = st.text_input("Rationale", key="i_rat")
            ievs = _pending_pick("i_pick")
            st.caption("A relation needs evidence frames at ≥ 2 different times and lasts ≥ 0.2 s.")
            if st.button("💾 Save interaction", type="primary"):
                _save("interaction", Interaction(VID, who, subj, rel, it0, it1, None if obj.startswith("(none") else obj, ievs,
                                                 iconf, iobs, ieid, irat))
    st.dataframe(pd.DataFrame([{"id": i.record_id, "subject": i.subject, "relation": i.relation, "object": i.object,
                                "from": i.t_start, "to": i.t_end, "evidence": len(i.evidence), "observation": i.observation}
                               for i in S.records(VID, who, "interaction")]), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ 4 evidence viewer
with T["Evidence"]:
    refs = [(r.record_id, e) for rt in ("event", "interaction") for r in S.records(VID, who, rt) for e in r.evidence] + \
           [("pending", e) for e in PEND]
    if not refs:
        st.info("No evidence frames yet.")
    cols = st.columns(4)
    for k, (rid, e) in enumerate(refs):
        p = S.evidence_path(VID, e["image_sha256"])
        ok = S.verify_evidence(e)
        with cols[k % 4]:
            if p.exists():
                st.image(str(p), use_container_width=True)
            st.caption(f"{rid} · {e['t']:.1f} s · {'✔ verified' if ok else '✖ hash mismatch'} · {e.get('note', '')}")

# ------------------------------------------------------------------ 5 counterfactual (research)
with T["Counterfactual (research)"]:
    st.warning("Research only: never ground truth, never exported, never used for training. Keep what was SEEN, your "
               "INTERPRETATION and the WHAT-IF separate.")
    obs_recs = {r.record_id: f"{type(r).__name__} {getattr(r, 'category', getattr(r, 'relation', ''))} {r.t_start}-{r.t_end}"
                for rt in ("event", "interaction") for r in S.records(VID, who, rt) if r.evidence}
    my_events = [r.record_id for r in S.records(VID, who, "event") if r.evidence]
    if not my_events:
        st.info("Needs one of your own events with evidence first.")
    else:
        cf_e = st.selectbox("Event", my_events, format_func=obs_recs.get, key="cf_e")
        cf_o = st.multiselect("Observed facts (your records with evidence)", list(obs_recs), default=[cf_e], format_func=obs_recs.get, key="cf_o")
        cf_i = st.text_area("Interpretation of the observed facts", key="cf_i")
        cf_h = st.text_area("Hypothetical (what if …)", key="cf_h")
        cf_a = st.selectbox("Answerable from the video?", ["partly", "yes", "not_answerable_from_video"], key="cf_a")
        if st.button("💾 Save counterfactual (research)"):
            _save("counterfactual", Counterfactual(VID, who, cf_e, cf_o, cf_i, cf_h, cf_a))
        for c in S.records(VID, who, "counterfactual"):
            st.markdown(f"- **seen:** {c.observed_facts} · **interpretation:** {c.interpretation} · **what-if:** {c.hypothetical} "
                        f"· *{c.answerable}*")

# ------------------------------------------------------------------ 6 active queue
with T["Active queue"]:
    from types import SimpleNamespace

    from safedrive_v4.annotation.active import queue
    cand = sorted(VIDS)[:300] if not is_coord else sorted(VIDS)
    q = queue(S, cand, {v: SimpleNamespace(meta=VIDS[v]["meta"]) for v in cand})
    st.caption("Ordering: rare conditions first, open disagreements up, agreed double-submitted videos last. "
               "No label and no model output is used here; model-uncertainty ordering is a coordinator research option (E5).")
    if is_coord:
        st.dataframe(pd.DataFrame([{"video": r["video_id"], "priority": round(r["priority"], 3), **r["reasons_coordinator_only"]}
                                   for r in q[:50]]), hide_index=True, use_container_width=True)
    else:
        st.dataframe(pd.DataFrame([{"#": i + 1, "video": r["video_id"]} for i, r in enumerate(q[:30]) if not S.submitted(r["video_id"], who)]),
                     hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ coordinator views
if is_coord:
    from safedrive_v4.annotation.agreement import dataset_agreement, video_agreement
    anns = sorted({a for v in S.videos() for a in S.annotators(v)})
    with T["Agreement"]:
        if len(anns) < 2:
            st.info("NOT EVALUATED: fewer than two annotators.")
        else:
            a1, a2 = st.columns(2)
            A = a1.selectbox("Annotator A", anns, key="ag_a")
            B = a2.selectbox("Annotator B", [x for x in anns if x != A], key="ag_b")
            d = dataset_agreement(S, A, B)
            st.json(d)
            if S.submitted(VID, A) and S.submitted(VID, B):
                va = video_agreement(S, VID, A, B)
                st.markdown(f"**This video** · disagreements: {len(va['disagreements'])}")
                st.dataframe(pd.DataFrame(va["disagreements"]), hide_index=True, use_container_width=True)
    with T["Dataset quality"]:
        rows = []
        for v in S.videos():
            for a in S.annotators(v):
                recs = [r for rt in ("event", "interaction") for r in S.records(v, a, rt)]
                ev_ok = all(S.verify_evidence(e) for r in recs for e in r.evidence)
                evs = S.records(v, a, "event")
                rows.append({"video": v, "annotator": a, "submitted": S.submitted(v, a), "events": len(evs),
                             "interactions": len(recs) - len(evs),
                             "uncertain/low-conf": sum(e.category == "uncertain" or e.uncertainty["annotation_confidence"] == "low" for e in evs),
                             "hash chain ok": S.verify_chain(v, a)["ok"], "evidence ok": ev_ok})
        if rows:
            df = pd.DataFrame(rows)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Videos", df.video.nunique())
            m2.metric("Double-submitted", int((df[df.submitted].groupby("video").size() >= 2).sum()))
            m3.metric("Chain / evidence failures", int((~df["hash chain ok"]).sum() + (~df["evidence ok"]).sum()))
            m4.metric("Adjudicated videos", sum(bool(S.adjudications(v)) for v in S.videos()))
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.info("No annotation yet.")
    with T["Adjudication"]:
        subs = [a for a in S.annotators(VID) if S.submitted(VID, a)]
        if len(subs) < 2:
            st.info("Adjudication opens after two annotators submitted this video.")
        else:
            A, B = subs[0], subs[1]
            ca, cb = st.columns(2)
            ra = S.view_other(VID, who, A)
            rb = S.view_other(VID, who, B)
            ca.markdown(f"**{A}**")
            ca.json(ra)
            cb.markdown(f"**{B}**")
            cb.json(rb)
            ea = {r.record_id: r for r in S.records(VID, A, "event") + S.records(VID, A, "interaction")}
            eb = {r.record_id: r for r in S.records(VID, B, "event") + S.records(VID, B, "interaction")}
            dec = st.selectbox("Decision", ["accept_A", "accept_B", "unresolved"], key="adj_d")
            pa = st.selectbox(f"{A} record", [None] + list(ea), key="adj_a")
            pb = st.selectbox(f"{B} record", [None] + list(eb), key="adj_b")
            note = st.text_input("Note", key="adj_n")
            if st.button("Record decision"):
                rec = ea.get(pa) if dec == "accept_A" else (eb.get(pb) if dec == "accept_B" else None)
                try:
                    S.adjudicate(VID, who, dec, {A: pa, B: pb}, rec, note)
                    st.success("Recorded (originals unchanged).")
                except (ValidationError, AccessError) as e:
                    st.error(str(e))
            st.dataframe(pd.DataFrame(S.adjudications(VID)), hide_index=True, use_container_width=True)
    with T["Export"]:
        st.caption("Exports adjudicated records (else submitted ones, marked single-annotator). Counterfactuals are never exported. "
                   "Evidence images are not copied (hashes only). Only public (Nexar) or privacy-approved videos.")
        dest = st.text_input("Destination folder (inside the encrypted volume recommended)", str(S.root / "exports"))
        if st.button("Export dataset"):
            from safedrive_v4.annotation.export import ExportRefused, export, nexar_split_of
            try:
                r = export(S, dest, who, public=set(VIDS), split_of=nexar_split_of())
                st.success(f"Exported {r['videos']} videos · {r['evidence']} evidence references → {r['path']}")
            except ExportRefused as e:
                st.error(f"Refused: {e}")
        p = S.root / "exports.jsonl"
        if p.exists():
            st.dataframe(pd.DataFrame([json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]), hide_index=True)
