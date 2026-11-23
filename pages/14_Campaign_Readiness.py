"""v3.4.3 Campaign readiness: the 15 pre-campaign checks with GO / NO-GO, coordinator attestations and the sharing decision,
the camera test (only when a person presses the button) and the operator checklist required before every real-driving session.
Nothing here starts a recording, changes Windows permissions or turns NO-GO into GO by itself."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.realworld import checklist, privacy, readiness
from ui.common import setup

setup("Campaign Readiness", "✅")
st.title("✅ Campaign Readiness & Operator Checklist (v3.4.4)")
st.caption("GO only when all 15 mandatory items are PASS with evidence. A configuration value is never evidence on its own; "
           "human confirmations are recorded by a named coordinator with an evidence reference. Do this page while PARKED.")

COLORS = {"PASS": "🟢", "FAIL": "🔴", "BLOCKED": "🟠", "NOT VERIFIED": "⚪"}
tab_r, tab_c, tab_s, tab_a = st.tabs(["Pre-campaign readiness", "Pre-session checklist", "Camera / safety & legal checklists",
                                      "Coordinator confirmations"])

with tab_r:
    if st.button("↻ Run the readiness checks now (reads drives, BitLocker, permissions, hashes; never opens the camera)"):
        with st.spinner("Checking…"):
            readiness.save_latest(readiness.check_all())
        st.rerun()
    rd = readiness.load_latest()
    if rd is None:
        st.info("No readiness run yet. Press the button above or run `python scripts/precampaign_audit.py`.")
    else:
        dec = rd["decision"]
        (st.success if dec["verdict"] == "GO" else st.error)(
            f"**{dec['verdict']}** — checked {rd['at']} · " + " · ".join(f"{k}: {v}" for k, v in dec["counts"].items()))
        st.dataframe(pd.DataFrame([{"": COLORS[i["status"]], "item": i["key"], "check": i["title"], "status": i["status"],
                                    "evidence from": i.get("evidence_at") or "—", "checked at": i.get("checked_at") or rd["at"],
                                    "who acts": i.get("who") or "", "next action": i.get("action") or ""} for i in rd["items"]]),
                     hide_index=True, use_container_width=True)
        st.caption("PASS = verified with evidence · FAIL = checked and not met · BLOCKED = cannot be checked now (rights / hardware "
                   "missing) · NOT VERIFIED = no evidence / no human confirmation yet. Only 15 × PASS gives GO.")
        for i in rd["items"]:
            with st.expander(f"{COLORS[i['status']]} {i['key']} — {i['status']}"):
                st.json(i["evidence"])

with tab_c:
    st.markdown("**Before every real-driving session, parked.** The session cannot start without a complete checklist from the "
                f"last {checklist.CHECKLIST_VALID_H:g} h (enforced by the software).")
    c1, c2 = st.columns(2)
    op = c1.text_input("Operator pseudonym", key="cl_op").strip()
    sup = c2.text_input("Supervisor pseudonym (another person)", key="cl_sup").strip()
    st.markdown("**Camera test** (opens the camera for 3 s, stores numbers only, no image). Stop the camera on the Live page first.")
    cc1, cc2 = st.columns([1, 3])
    cam = cc1.number_input("Camera index", 0, 9, 0, key="cl_cam")
    if cc2.button("📷 Test camera (3 s)", disabled=not op):
        r = readiness.camera_check(int(cam), 3.0, op)
        (st.success if r["ok"] else st.error)(f"{'OK' if r['ok'] else 'Problems: ' + '; '.join(r['problems'])} · {r.get('fps')} fps · "
                                              f"{r.get('resolution')} · brightness {r.get('brightness_median')} · sharpness {r.get('sharpness_median')}")
    st.markdown("**Confirmations** (operator + supervisor, vehicle parked)")
    manual = {k: st.checkbox(v, key=f"cl_{k}") for k, v in checklist.MANUAL_ITEMS.items()}
    if st.button("Check automatic items and save the checklist", type="primary", disabled=not (op and sup)):
        with st.spinner("Checking access, disk, backup, camera test, model hashes, readiness…"):
            try:
                rec = checklist.save(op, sup, manual, checklist.auto_checks())
                (st.success if rec["complete"] else st.error)("Checklist COMPLETE: a real-driving session may be started on the Live page."
                                                              if rec["complete"] else "Checklist INCOMPLETE — the session will be refused. "
                                                              f"Missing: {', '.join(rec['missing_manual'] + rec['failed_auto'])}")
                st.dataframe(pd.DataFrame([{"automatic item": k, "ok": v["ok"], "detail": str(v["detail"])[:160]}
                                           for k, v in rec["auto"].items()]), hide_index=True, use_container_width=True)
            except ValueError as e:
                st.error(str(e))
    last = checklist.latest()
    if last:
        st.caption(f"Latest checklist: {last['at']} by {last['operator']} / {last['supervisor']} · complete: **{last['complete']}**")

with tab_s:
    st.markdown("**One-off checklists answered by the coordinator** (point by point; the software never fills them in). "
                "*confirmed* and *not applicable* need a basis. *needs legal advice* keeps the item open (NOT VERIFIED).")
    cname = st.text_input("Coordinator pseudonym", key="ck_name").strip()
    c_ok = bool(cname) and privacy.is_coordinator(cname)
    if cname and not c_ok:
        st.error("Not on the coordinators list in data/campaign/access.json.")
    kind = st.radio("Checklist", list(readiness.CHECKLIST_ATTESTS), horizontal=True,
                    format_func=lambda k: {"camera_safety": "Camera & safety (item 12)", "legal_checklist": "Legal (item 13)"}[k])
    with st.form(f"ck_{kind}"):
        answers = {}
        for k, txt in readiness.CHECKLIST_ATTESTS[kind].items():
            a1, a2 = st.columns([2, 3])
            opts = [o for o in readiness.ANSWERS[kind] if not (o == "not_applicable" and k in readiness.NEVER_NOT_APPLICABLE)]
            st_ = a1.selectbox(txt, ["(not answered)"] + list(opts), key=f"ck_{kind}_{k}")
            basis = a2.text_input("Basis / what was checked / document", key=f"ckb_{kind}_{k}")
            if st_ != "(not answered)":
                answers[k] = {"status": st_, "basis": basis}
        ref = st.text_input("Evidence reference (document, photo id kept locally, sheet number)", key=f"ckr_{kind}")
        if st.form_submit_button("Record checklist", disabled=not c_ok):
            try:
                rec = readiness.attest_checklist(kind, cname, answers, ref)
                (st.success if rec["complete"] else st.warning)("Recorded. " + ("Complete." if rec["complete"] else
                                                                f"Open points: {', '.join(rec['open'])}"))
            except (ValueError, PermissionError) as e:
                st.error(str(e))
    last_ck = readiness.checklist_attestation(kind)
    if last_ck:
        st.caption(f"Latest: {last_ck['at']} by {last_ck['by']} · complete **{last_ck['complete']}** · open: "
                   f"{', '.join(last_ck['open']) or 'none'} · legal advice needed: {', '.join(last_ck.get('needs_legal_advice', [])) or 'none'}")
        st.dataframe(pd.DataFrame([{"point": k, **v} for k, v in last_ck["answers"].items()]), hide_index=True, use_container_width=True)

with tab_a:
    name = st.text_input("Coordinator pseudonym", key="att_name").strip()
    if name and not privacy.is_coordinator(name):
        st.error("Not on the coordinators list in data/campaign/access.json.")
    is_c = bool(name) and privacy.is_coordinator(name)
    st.markdown("**Human confirmations** — the software cannot verify these. Record them only after doing them yourself; give a "
                "reference to the evidence (document, checklist sheet, local photo id). Never record for someone else.")
    with st.form("att"):
        simple = [k for k in readiness.ATTEST_ITEMS if k not in ("legal_review", "camera_mount")]  # v3.4.4: these use the checklists
        key = st.selectbox("Item", simple, format_func=lambda k: f"{k}: {readiness.ATTEST_ITEMS[k]}")
        stmt = st.text_area("Statement (what you checked, when, how)")
        ref = st.text_input("Evidence reference")
        if st.form_submit_button("Record confirmation", disabled=not is_c):
            try:
                readiness.attest(key, name, stmt, ref)
                st.success("Recorded.")
            except (ValueError, PermissionError) as e:
                st.error(str(e))
    rows = readiness._read_jsonl(readiness.ATTEST_PATH)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.markdown("**Sharing decision** — disabling sharing is the written 'nothing is shared until the blur was reviewed on "
                "campaign-camera footage' decision; the export function then refuses every copy.")
    pol = privacy.sharing_policy()
    st.caption(f"Current: sharing_enabled = **{pol.get('sharing_enabled')}** (decided by {pol.get('decided_by') or '—'} "
               f"{pol.get('decided_at') or ''})")
    with st.form("pol"):
        en = st.radio("Sharing of blurred copies", ["disabled (no-sharing decision)", "enabled (only approved copies)"]) .startswith("enabled")
        why = st.text_input("Reason")
        if st.form_submit_button("Save decision", disabled=not is_c):
            try:
                privacy.set_sharing_policy(en, name, why)
                st.success("Saved.")
            except (ValueError, PermissionError) as e:
                st.error(str(e))
