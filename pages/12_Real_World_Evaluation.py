"""Real-world Evaluation (v3.3). Every number on this page is read from recorded results; the four kinds of evidence are kept in
separate tabs and are never pooled: benchmark video (Nexar held-out), live camera tests, controlled tests, real driving."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.realworld.metrics import aggregate, session_metrics
from src.realworld.trip_meta import SESSIONS_DIR, list_sessions, load_meta
from src.utils.config import resolve
from ui.common import card, setup

setup("Real-world Evaluation", "🧭")
st.title("🧭 Real-world Evaluation")
st.caption("Only measured data. Benchmark-video, live-camera, controlled-test and real-driving results come from different conditions "
           "and are shown separately; they are not combined into one number.")

LATEST, LIVE = resolve("experiments/latest"), resolve("experiments/live")


@st.cache_data(ttl=30)
def all_sessions() -> list[dict]:
    rows = []
    for sid in list_sessions():
        try:
            rows.append(session_metrics(SESSIONS_DIR / sid))
        except Exception as e:  # noqa: BLE001
            rows.append({"session_id": sid, "data_kind": "error", "note": repr(e), "hours": 0.0})
    return rows


sess = all_sessions()
# ---------------- filters (session-based tabs) ----------------
with st.sidebar:
    st.markdown("**Filters (session tabs)**")
    models = sorted({r.get("model") or "?" for r in sess})
    f_model = st.multiselect("Model", models, default=models)
    f_light = st.multiselect("Lighting (declared)", sorted({r.get("lighting", "unknown") for r in sess}))
    f_road = st.multiselect("Road type (declared)", sorted({x for r in sess for x in (r.get("road_types") or [])}))
    f_sess = st.multiselect("Session", [r["session_id"] for r in sess])
    f_cat = st.multiselect("Event category (gallery)", ["missed_event", "false_alarm", "warning_without_hazard", "valid_warning",
                                                        "risk_event", "warning_event", "potential_near_miss"])


def keep(r: dict) -> bool:
    return ((r.get("model") or "?") in f_model and (not f_light or r.get("lighting") in f_light)
            and (not f_road or set(r.get("road_types") or []) & set(f_road)) and (not f_sess or r["session_id"] in f_sess))


sess_f = [r for r in sess if keep(r)]


def session_table(rows: list[dict]) -> None:
    if not rows:
        st.info("No sessions of this kind match the filters.")
        return
    t = pd.DataFrame([{"session": r["session_id"], "model": r.get("model"), "profile": r.get("profile"), "lighting": r.get("lighting"),
                       "road": ", ".join(r.get("road_types") or []) or "unknown", "hours": round(r.get("hours", 0), 3),
                       "alarms": r.get("alarms"), "alarms/h": round(r["alarms_per_hour"], 1) if r.get("alarms_per_hour") is not None else None,
                       "FA/h (annotated)": round(r["fa_per_hour"], 1) if r.get("fa_per_hour") is not None else "not annotated",
                       "warning recall": r.get("warning_recall") if r.get("warning_recall") is not None else "n/a",
                       "repeated": r.get("repeated_warnings"), "rate Hz": round(r.get("runtime", {}).get("processed_rate_hz", np.nan), 2),
                       "e2e p95 ms": r.get("runtime", {}).get("e2e_ms", {}).get("p95"),
                       "completeness": r.get("integrity", {}).get("completeness"),
                       "clips ok": f"{r.get('recording', {}).get('clips_saved', 0)}/{r.get('recording', {}).get('events', 0)}",
                       "MB/h": round(r.get("recording", {}).get("storage_mb_per_hour", np.nan), 0),
                       "note": r.get("note", "")} for r in rows])
    st.dataframe(t, hide_index=True, use_container_width=True)


tabs = st.tabs(["① Benchmark video (Nexar)", "② Live camera tests", "③ Controlled tests", "④ Real driving", "🖼️ Failure-case gallery"])

# ---------------- ① benchmark ----------------
with tabs[0]:
    p = LATEST / "v33_benchmark_strata.json"
    if not p.exists():
        st.warning("Run `python scripts/realworld_benchmark_strata.py` first.")
    else:
        B = json.loads(p.read_text(encoding="utf-8"))
        st.caption(f"Held-out {B['n_videos']} Nexar dashcam videos (real recorded driving, not seen in training or model selection), deployed v3 "
                   "pipeline. Strata use the dataset's own verified metadata; the turn proxy is measured camera motion (tertiles fixed on dev).")
        prof = st.radio("Warning sensitivity", ["conservative", "balanced", "sensitive"], index=1, horizontal=True)
        o = B["overall"][prof]
        c = st.columns(4)
        card(c[0], "Recall (alarm ≤ 5 s before event)", f"{o['recall']:.1%}", f"95% CI {o['recall_ci95'][0]:.1%}–{o['recall_ci95'][1]:.1%} · n={B['overall']['n_pos']}")
        card(c[1], "False alarms / h", f"{o['fa_h']:.1f}", f"95% CI {o['fa_h_ci95'][0]}–{o['fa_h_ci95'][1]} · {B['overall']['neg_hours']:.2f} h")
        card(c[2], "Warned ≥ 1 / 2 / 3 s", f"{o['warned_ge_1s']:.1%} / {o['warned_ge_2s']:.1%} / {o['warned_ge_3s']:.1%}", "of all collision videos")
        card(c[3], "Median lead (warned)", f"{o['median_lead_s']:.2f} s" if o["median_lead_s"] else "—", f"{o['hits']} warned events")
        names = {"light": "Lighting (Nexar label)", "weather": "Weather (Nexar label)", "scene": "Scene (Nexar label)",
                 "turn_stratum": "Turn proxy (|horizontal camera shift|)", "off_corridor_positives": "Largest road user vs fixed corridor (H3)"}
        col = st.selectbox("Stratify by", list(names), format_func=names.get)
        rows = []
        for g, v in B["strata"][col].items():
            pv = v[prof]
            rows.append({"group": g, "collision videos": v["n_pos"], "recall": pv["recall"], "ci_lo": pv["recall_ci95"][0], "ci_hi": pv["recall_ci95"][1],
                         "normal videos": v.get("n_neg"), "normal hours": v.get("neg_hours"), "false alarms": pv.get("false_alarms"),
                         "FA/h": pv.get("fa_h"), "FA/h CI": pv.get("fa_h_ci95")})
        df = pd.DataFrame(rows)
        fig = go.Figure(go.Bar(x=df.group, y=df.recall, error_y=dict(type="data", symmetric=False, array=df.ci_hi - df.recall,
                                                                       arrayminus=df.recall - df.ci_lo), marker_color="#3b82f6"))
        fig.update_layout(height=300, yaxis_title="recall (95% Wilson CI)", yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.caption("Small groups have very wide intervals; overlapping intervals mean no difference is shown. Rain FA/h rests on 3 normal videos.")

# ---------------- ② live camera ----------------
with tabs[1]:
    st.caption("The laptop webcam / demo replay through the full live pipeline (not a vehicle). Measures runtime and robustness, not accuracy.")
    rows = []
    for f in sorted(LIVE.glob("*.json")):
        if f.stem == "fidelity":
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        rows.append({"test": d.get("name"), "source": "camera" if str(d.get("source", "")).startswith("camera") else "demo video",
                     "device": d.get("device"), "minutes": round(d.get("ran_s", 0) / 60, 1), "rate Hz": d.get("processed_steps_per_s"),
                     "e2e p50/p95 ms": f"{d.get('e2e_ms_p50') or 0:.0f} / {d.get('e2e_ms_p95') or 0:.0f}",
                     "RSS start→end MB": f"{d.get('rss_mb_start', 0):.0f} → {d.get('rss_mb_end', 0):.0f}",
                     "RSS slope MB/min (2nd half)": d.get("rss_slope_mb_per_min_second_half"),
                     "interruptions": d.get("camera_interruptions"), "reconnects": d.get("camera_reconnects"),
                     "clips saved/failed": f"{d.get('clips_saved')}/{d.get('clips_failed')}", "alive at end": d.get("alive_at_end"),
                     "error": d.get("error")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    fid = LIVE / "fidelity.json"
    if fid.exists():
        st.caption("Live (real-time) vs offline predictions on held-out videos: see docs/SAFEDRIVE_V32_TEST_REPORT.md.")
    st.markdown("**Recorded live-camera sessions (v3.3 metadata)**")
    session_table([r for r in sess_f if r.get("data_kind") == "live_camera_test"])
    legacy = [r for r in sess_f if r.get("data_kind") == "legacy_unknown"]
    if legacy:
        with st.expander(f"{len(legacy)} legacy_unknown sessions (no metadata; excluded from official evaluation)"):
            session_table(legacy)

# ---------------- ③ controlled ----------------
with tabs[2]:
    st.caption("Controlled conditions: synthetic corruptions of held-out videos (re-processed by the full perception pipeline) and "
               "staged system faults. No dangerous situation was created on a public road.")
    robs = {k: LATEST / f for k, f in [("v3", "robustness_v3.json"), ("v3 light-change (v3.3)", "robustness_v3_light_change.json"),
                                       ("v3-stabilized", "robustness_v3_stabilized.json")]}
    rr = []
    for name, f in robs.items():
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            for cnd, v in d["conditions"].items():
                rr.append({"model": name, "condition": cnd, "mAP": round(v["mAP"], 3), "ROC-AUC": round(v["roc_auc"], 3),
                           "F1": round(v["f1"], 3), "ECE": round(v["ece"], 3), "mean MI": round(v["mean_mutual_info"], 4)})
    if rr:
        st.markdown("**Video corruptions (60 original test videos; window-level metrics)**")
        st.dataframe(pd.DataFrame(rr), hide_index=True, use_container_width=True)
    ls = LATEST / "v33_light_change_stream.json"
    if ls.exists():
        L = json.loads(ls.read_text(encoding="utf-8"))
        st.markdown(f"**Rapid light change, full streams, deployed alarm logic** ({L['n_pos']} collision / {L['n_neg']} normal test videos, "
                    f"{L['neg_hours']:.2f} normal h; paired with the clean re-extraction of the same videos)")
        lr = []
        for cnd, o in L["conditions"].items():
            if cnd == "clean_original_cache":
                continue
            b = o["balanced"]
            lr.append({"condition": cnd, "detections/step": round(o["det_per_step"], 2), "steps w/o detection": f"{o['zero_detection_step_share']:.1%}",
                       "mean P shift vs clean [95% CI]": f"{o.get('p_shift_mean_per_video', 0):+.3f} {o.get('p_shift_ci95', '')}",
                       "mean P before event (collision)": round(o["p_pos_pre_event_mean"], 3), "mean P (normal)": round(o["p_neg_mean"], 3),
                       "recall (balanced) [CI]": f"{b['recall']:.1%} {b['recall_ci95']}", "FA (balanced)": b["false_alarms"],
                       "paired gained/lost": f"{b['paired']['gained']}/{b['paired']['lost']} (McNemar p={b['paired']['mcnemar_exact_p']:.2f})" if "paired" in b else "baseline"})
        st.dataframe(pd.DataFrame(lr), hide_index=True, use_container_width=True)
        st.caption("Only 2–3 collisions are warned even on clean streams, so recall / FA changes cannot be tested here; "
                   "the detection and probability shifts are the measurable effects.")
    st.markdown("**System fault tests**")
    faults = []
    for n, what in [("cam_disconnect", "camera disconnect / reconnect"), ("cam_occupied", "camera used by another app"),
                    ("demo_disk_full", "near-full disk (simulated)"), ("demo_30min_record", "30 min continuous run with event + full-video recording")]:
        f = LIVE / f"{n}.json"
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            faults.append({"test": what, "file": f.name, "ran min": round(d.get("ran_s", 0) / 60, 1) if "ran_s" in d else None,
                           "alive at end": d.get("alive_at_end"), "reconnects": d.get("camera_reconnects"),
                           "recovery s": d.get("disconnect_recovery_s"), "clips saved/failed": f"{d.get('clips_saved')}/{d.get('clips_failed')}",
                           "RSS slope MB/min": d.get("rss_slope_mb_per_min_second_half"), "error": d.get("error") or d.get("message", "")})
        else:
            faults.append({"test": what, "file": f.name, "error": "not run yet"})
    st.dataframe(pd.DataFrame(faults), hide_index=True, use_container_width=True)
    st.markdown("**Controlled-test sessions (v3.3 metadata)**")
    session_table([r for r in sess_f if r.get("data_kind") in ("controlled_test", "demo_video")])

# ---------------- ④ real driving ----------------
with tabs[3]:
    real = [r for r in sess_f if r.get("data_kind") == "real_driving"]
    hrs = sum(r.get("hours", 0) for r in real)
    c = st.columns(4)
    card(c[0], "Real-driving hours", f"{hrs:.2f} h", "target ≥ 10 h of normal driving")
    card(c[1], "Sessions", str(len(real)), f"{sum(r.get('fully_reviewed', False) for r in real)} fully reviewed")
    card(c[2], "Annotated warning-worthy events", str(sum(r.get("warning_worthy_events", 0) or 0 for r in real)), "from full reviews")
    from src.realworld.annotation import annotators
    card(c[3], "Sessions with 2+ annotators", str(sum(len(annotators(SESSIONS_DIR / r["session_id"])) >= 2 for r in real)), "agreement: annotation page")
    st.progress(min(hrs / 10.0, 1.0), text=f"{hrs:.2f} / 10 h")
    if not real:
        st.warning("**No real-driving data has been recorded yet (0 h).** Real-world false-alarm rate, recall and lead time on the road "
                   "are therefore unknown. Collect data with the protocol in docs/SAFEDRIVE_V33_EVALUATION_PROTOCOL.md.")
    else:
        agg = aggregate(real, "lighting")
        st.markdown("**By declared lighting** (pooled only within real driving)")
        st.dataframe(agg, hide_index=True, use_container_width=True)
        session_table(real)
        leads = [x for r in real for x in (r.get("lead_times_s") or [])]
        if leads:
            fig = go.Figure(go.Histogram(x=leads, xbins=dict(start=0, end=5, size=0.5)))
            fig.update_layout(height=250, xaxis_title="lead time (s)", yaxis_title="warned events", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

# ---------------- gallery ----------------
with tabs[4]:
    src = st.radio("Source", ["Benchmark (held-out Nexar)", "Recorded sessions"], horizontal=True)
    if src.startswith("Benchmark"):
        pv = LATEST / "v33_benchmark_per_video.csv"
        if not pv.exists():
            st.info("Run scripts/realworld_benchmark_strata.py first.")
        else:
            df = pd.read_csv(pv, dtype={"video_id": str})
            df["case"] = np.where(df.label == 1, np.where(df.balanced_hit == True, "valid_warning", "missed_event"),  # noqa: E712
                                  np.where(df.balanced_fa > 0, "false_alarm", "correct_silence"))
            cats = [c for c in (f_cat or ["missed_event", "false_alarm"]) if c in ("missed_event", "false_alarm", "valid_warning")]
            sub = df[df.case.isin(cats)]
            st.caption(f"{len(sub)} cases (balanced profile). Frame shown: 1 s before the event (collision videos) or at the first false alarm.")
            from src.data.dataset_loader import index_videos
            vids = index_videos(resolve("data/raw/nexar"))
            n = st.slider("Show", 4, 40, 12, 4)
            cols = st.columns(4)
            for i, (_, r) in enumerate(sub.head(n).iterrows()):
                al = json.loads(r.alarms_balanced) if isinstance(r.alarms_balanced, str) else []
                t = (r.time_of_event - 1.0) if r.label == 1 else (al[0] if al else 5.0)
                img = None
                if r.video_id in vids:
                    cap = cv2.VideoCapture(str(vids[r.video_id]))
                    cap.set(cv2.CAP_PROP_POS_MSEC, max(t, 0) * 1000)
                    ok, fr = cap.read()
                    cap.release()
                    img = cv2.cvtColor(cv2.resize(fr, (320, 180)), cv2.COLOR_BGR2RGB) if ok else None
                with cols[i % 4]:
                    if img is not None:
                        st.image(img, use_container_width=True)
                    st.caption(f"**{r.case}** · {r.video_id} · {r.light}/{r.weather}/{r.scene} · turn={r.turn_stratum}"
                               + (f" · off-corridor={r.off_corridor}" if r.label == 1 else "")
                               + (f" · max P before event {r.p_max_pre_event:.2f}" if r.label == 1 and pd.notna(r.p_max_pre_event) else "")
                               + (f" · alarms at {al}" if al else ""))
    else:
        items = []
        for r in sess_f:
            d = SESSIONS_DIR / r["session_id"]
            rp = d / "session_report.json"
            if rp.exists():
                for e in json.loads(rp.read_text(encoding="utf-8")).get("events", []):
                    items.append({**e, "session": r["session_id"], "data_kind": r.get("data_kind")})
        if f_cat:
            items = [e for e in items if e.get("category") in f_cat]
        st.caption(f"{len(items)} recorded events. Categories are model-based (not verified) unless annotated on the annotation page.")
        cols = st.columns(3)
        for i, e in enumerate(items[:30]):
            with cols[i % 3]:
                if e.get("clip") and Path(e["clip"]).exists():
                    st.video(e["clip"])
                st.caption(f"**{e['category']}** · {e['data_kind']} · {e['session']} @ {e['t_stream']:.1f} s · P peak {e['p_peak']:.2f} · clip {e['clip_status']}")
