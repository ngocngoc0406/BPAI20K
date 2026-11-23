"""v3.4.3 pre-campaign readiness audit.

Default run (v3.4.3): the 15 mandatory readiness checks of src/realworld/readiness.py (hashes, provenance, access lists,
BitLocker, folder permissions, external backup drive, backup / restore verification, privacy review, camera readiness, legal
acknowledgement, annotator training, disk space) plus an active test that the no-sharing decision really blocks export.
GO only if all 15 are PASS. Writes experiments/audit_v344/<run>/ (v3.4.3 runs: experiments/audit_v343/)readiness_results.json and data/campaign/readiness_latest.json.
--full additionally re-runs the v3.4 mechanism tests below (privacy / integrity / annotation / governance on copies).

v3.4 mechanism audit (--full). Read-only with respect to the project: every test runs on COPIES in
experiments/audit_v34/<run>/ ; data/sessions, data/campaign/access.json, the frozen hashes and the models are never modified;
no camera is opened and no recording session is started (only a demo-file replay into the audit folder for the integrity test).

Writes experiments/audit_v34/<run>/audit_results.json and audit.log. Statuses: PASS / FAIL / BLOCKED / NOT VERIFIED.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.realworld import annotation as ann  # noqa: E402
from src.realworld import campaign, privacy, provenance  # noqa: E402
from src.realworld import metrics as rwm  # noqa: E402
from src.realworld.trip_meta import TripMeta, delete_session, load_meta, save_meta  # noqa: E402
from src.utils.config import resolve  # noqa: E402

RUN = resolve("experiments/audit_v344") / datetime.now().strftime("%Y%m%d_%H%M%S")
RUN.mkdir(parents=True, exist_ok=True)
LOG = open(RUN / "audit.log", "w", encoding="utf-8")
RESULTS: dict = {}
V3 = resolve("models/safedrive_v3")
DEMO_SESSION = resolve("data/sessions/20261121_101150_b430")  # v3.3 30-min demo replay with full video (read only)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def record(key, status, evidence, human=None):
    RESULTS[key] = {"status": status, "evidence": evidence, "human_action": human}
    log(f"[{status}] {key}: {json.dumps(evidence, default=str)[:600]}")


def guarded(key):
    def deco(fn):
        def run():
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                record(key, "FAIL", {"exception": repr(e), "trace": traceback.format_exc()[-1500:]})
        return run
    return deco


def sha_dir(d: Path) -> dict:
    return {str(f.relative_to(d)): provenance.sha256_file(f) for f in sorted(d.rglob("*")) if f.is_file()}


SNAP_BEFORE = {"sessions": sorted(p.name for p in resolve("data/sessions").iterdir()),
               "access": provenance.sha256_file(resolve("data/campaign/access.json")),
               "frozen": provenance.sha256_file(provenance.FROZEN_PATH)}


# ---------------- 1. frozen hashes ----------------
@guarded("1_frozen_hashes")
def audit_hashes():
    ref = provenance.frozen()
    ev = {"frozen_file": str(provenance.FROZEN_PATH), "frozen_file_sha256": SNAP_BEFORE["frozen"]}
    ok = True
    for m in ("models/safedrive_v3", "models/safedrive_v3_stabilized"):
        c = provenance.compute(m)
        chk = provenance.check(c)
        ev[Path(m).name] = {"check": chk, "frozen_at": ref[Path(m).name]["frozen_at"]}
        changelog = resolve("docs/SAFEDRIVE_SOFTWARE_CHANGELOG.md").read_text(encoding="utf-8")
        sw_logged = chk["software_same"] or c["software_sha256"][:16] in changelog
        ev[Path(m).name]["software_sha256"] = c["software_sha256"][:16]
        ev[Path(m).name]["software_change_in_changelog"] = sw_logged
        ok &= all(chk[k] for k in ("model_ok", "features_ok", "thresholds_ok")) and sw_logged
    # not overwritten: freeze() refuses; mtime equals frozen_at
    try:
        provenance.freeze()
        refused = False
    except FileExistsError:
        refused = True
    mtime = datetime.fromtimestamp(provenance.FROZEN_PATH.stat().st_mtime).isoformat(timespec="seconds")
    ev.update({"freeze_refuses_overwrite": refused, "file_mtime": mtime,
               "mtime_matches_frozen_at": mtime[:16] == ref["safedrive_v3"]["frozen_at"][:16]})
    # models identical to verified backups
    for m, b in (("safedrive_v3", "v3_1_before_live_camera/safedrive_v3"), ("safedrive_v3_stabilized", "v3_1_before_live_camera/safedrive_v3_stabilized"),
                 ("safedrive_v1", "v1_lstm_ensemble/model_bundle")):
        ev[f"{m}_identical_to_backup"] = sha_dir(resolve("models") / m) == sha_dir(resolve("backups") / b)
    ev["v2_light_weights_identical_to_backup"] = provenance.sha256_file(resolve("models/safedrive_v2_light/model.pkl")) == \
        provenance.sha256_file(resolve("backups/v3_20261121/safedrive_v2_light_bundle/model.pkl"))
    # mismatch detection on a copy-session
    root = RUN / "hash_mismatch" / "sessions"
    d = root / "mm1"
    d.mkdir(parents=True)
    p = {**provenance.compute(V3), "profile": "balanced"}
    p["threshold_config_sha256"] = "f" * 64  # simulated tampered threshold configuration
    p["check"] = provenance.check(p)
    (d / "provenance.json").write_text(json.dumps(p), encoding="utf-8")
    save_meta(TripMeta("mm1", data_kind="real_driving", driver_consent=True, occupants_informed=True, supervisor_present=True,
                       configured_before_driving=True, regulations_checked=True, campaign_id="audit", trip_id="T", driver_id="DX"), root)
    e = campaign.session_entry(d, with_phash=False)
    ev["mismatch_session"] = {"eligible": e["eligible"], "issues": e["issues"], "check": e["provenance_check"]}
    detected = (not e["eligible"]) and any("differs from the frozen reference" in i for i in e["issues"])
    all_ok = ok and refused and ev["mtime_matches_frozen_at"] and detected and all(v for k, v in ev.items() if k.endswith("identical_to_backup"))
    record("1_frozen_hashes", "PASS" if all_ok else "FAIL", ev)


# ---------------- 2. access control ----------------
@guarded("2_access_control")
def audit_access():
    acc = privacy.load_access()
    ev = {"raw_access": acc.get("raw_access"), "coordinators": acc.get("coordinators"),
          "can_view_raw('auditor-no-access')": privacy.can_view_raw("auditor-no-access"),
          "is_coordinator('auditor-no-access')": privacy.is_coordinator("auditor-no-access")}
    # page-level check: an unlisted name must not get the raw video (AppTest on page 11, real sessions, read only)
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(resolve("pages/11_Ground_Truth_Annotation.py")), default_timeout=120)
    at.run()
    at.selectbox[0].select(DEMO_SESSION.name).run()
    at.text_input[0].input("auditor-no-access").run()
    errs = [e.value for e in at.error]
    ev["page11_unlisted_annotator_errors"] = errs
    ev["page11_unlisted_annotator_video_elements"] = len(at.get("video"))
    at.radio[0].set_value("Coordinator").run()
    ev["page11_unlisted_coordinator_errors"] = [e.value for e in at.error]
    denied = (not ev["can_view_raw('auditor-no-access')"]) and ev["page11_unlisted_annotator_video_elements"] == 0 and \
        any("raw-access list" in x for x in errs) and any("coordinators list" in x for x in ev["page11_unlisted_coordinator_errors"])
    ev["unauthorised_denied"] = denied
    ev["note"] = "gating is application-level (name typed into the page); OS permissions / BitLocker not checked by software"
    import subprocess
    try:
        bl = subprocess.run(["powershell", "-NoProfile", "-Command",
                             "Get-BitLockerVolume -MountPoint D: | Select-Object MountPoint,ProtectionStatus,VolumeStatus | ConvertTo-Json"],
                            capture_output=True, text=True, timeout=30)
        ev["bitlocker_D"] = (bl.stdout.strip() or bl.stderr.strip())[:300]
    except Exception as e:  # noqa: BLE001
        ev["bitlocker_D"] = repr(e)
    ev["os_protection_verified"] = '"ProtectionStatus":  1' in ev["bitlocker_D"] or '"ProtectionStatus": 1' in ev["bitlocker_D"]
    if not acc.get("raw_access") or not acc.get("coordinators"):
        record("2_access_control", "BLOCKED", ev, "Fill data/campaign/access.json with pseudonyms for raw_access and coordinators "
               "(the audit does not fill it). Protect data/ with Windows account permissions and enable BitLocker.")
    else:
        record("2_access_control", "PASS" if denied else "FAIL", ev)
    record("2b_os_protection", "PASS" if ev["os_protection_verified"] else "NOT VERIFIED", {"bitlocker_D": ev["bitlocker_D"]},
           None if ev["os_protection_verified"] else "Enable BitLocker on the data drive (admin) and restrict data/ to the campaign "
           "accounts; this audit cannot verify or change it (Get-BitLockerVolume needs administrator rights).")


# ---------------- 3. privacy ----------------
@guarded("3_privacy")
def audit_privacy():
    from src.data.dataset_loader import index_videos
    vids = index_videos(resolve("data/raw/nexar"))
    pv = pd.read_csv(resolve("experiments/latest/v33_benchmark_per_video.csv"), dtype={"video_id": str})
    picks = [v for v in pv[(pv.scene == "Urban")].video_id.tolist() if v in vids][:3]
    d = RUN / "privacy_sample"
    (d / "video").mkdir(parents=True)
    for k, v in enumerate(picks):
        shutil.copy(vids[v], d / "video" / f"segment_{k:03d}.mp4")
    t0 = time.time()
    blog = privacy.make_shared_copy(d)
    ev = {"source_videos (held-out Nexar, Urban)": picks, "seconds": round(time.time() - t0, 1), "blur_log": blog["segments"]}
    # before / after frames with the most road users, for human inspection
    from src.detection.detector import Detector
    from src.utils.config import load_config
    from src.utils.device import get_device
    cfg = load_config(None)
    det = Detector(str(resolve(cfg["detection"]["weights"])), conf=0.25, classes=[0, 1, 2, 3, 5, 7], device=get_device())
    out = d / "review_frames"
    out.mkdir()
    pairs = []
    for k in range(len(picks)):
        cr, cs = cv2.VideoCapture(str(d / "video" / f"segment_{k:03d}.mp4")), cv2.VideoCapture(str(d / "shared" / f"segment_{k:03d}.mp4"))
        n = int(cr.get(cv2.CAP_PROP_FRAME_COUNT))
        best = []
        for i in range(0, n, 30):
            cr.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cr.read()
            if ok:
                ds = det.detect(fr, i)
                best.append((sum(x.cls in (0, 1, 3) for x in ds) * 3 + len(ds), i))
        for _, i in sorted(best, reverse=True)[:2]:
            cr.set(cv2.CAP_PROP_POS_FRAMES, i)
            cs.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok1, a = cr.read()
            ok2, b = cs.read()
            if ok1 and ok2:
                b = cv2.resize(b, (a.shape[1], a.shape[0]))
                pth = out / f"seg{k}_frame{i:05d}_raw_vs_shared.jpg"
                cv2.imwrite(str(pth), np.vstack([a, b]))
                pairs.append(str(pth))
        cr.release()
        cs.release()
    ev["review_frames"] = pairs
    shareable_before = privacy.shareable(d)
    try:
        privacy.review(d, (privacy.load_access().get("coordinators") or ["COORD-01"])[0], True, ["segment_000.mp4"], identifiable_found=False,
                       log_path=RUN / "test_review_log.jsonl")
        partial_refused = False
    except ValueError:
        partial_refused = True
    exp = {}
    try:
        privacy.export_shared(d, RUN / "export_test", "COORD-01", RUN / "export_log.jsonl")
        exp["unreviewed_export"] = "ALLOWED (bad)"
    except privacy.ExportRefused as e:
        exp["unreviewed_export"] = f"refused: {e}"
    try:
        privacy.export_shared(d, RUN / "export_test", "ANN-A", RUN / "export_log.jsonl")
        exp["non_coordinator_export"] = "ALLOWED (bad)"
    except privacy.ExportRefused as e:
        exp["non_coordinator_export"] = f"refused: {e}"
    ev.update({"shareable_without_review": shareable_before, "approval_with_unchecked_segments_refused": partial_refused,
               "export_guard": exp, "plate_band": privacy.PLATE_BAND, "box_pad": privacy.BOX_PAD,
               "note": "export_shared is the only sanctioned export (approved blurred copy, coordinator, hash-verified, logged); "
                       "it cannot stop someone from copying files by hand"})
    # deletion log on a copy
    root = RUN / "deletion_test" / "sessions"
    (root / "del1").mkdir(parents=True)
    (root / "del1" / "timeline.csv").write_text("step\n0\n")
    mem = RUN / "deletion_test" / "memory"
    mem.mkdir(parents=True)
    (mem / "events.jsonl").write_text(json.dumps({"event_id": "e1", "session_id": "del1"}) + "\n", encoding="utf-8")
    lp = RUN / "deletion_test" / "deletion_log.jsonl"
    r = delete_session("del1", root, mem, requested_by="audit", reason="deletion test", log_path=lp)
    rec = json.loads(lp.read_text(encoding="utf-8").splitlines()[-1])
    ev["deletion"] = {"result": r, "log_record": rec, "folder_gone": not (root / "del1").exists(),
                      "log_has_no_content_fields": set(rec) <= {"session_id", "deleted_at", "requested_by", "reason", "session_dir", "events", "clips"}}
    ev["real_deletion_log_exists"] = resolve("data/campaign/deletion_log.jsonl").exists()
    mech_ok = (not shareable_before) and partial_refused and ev["deletion"]["folder_gone"] and ev["deletion"]["log_has_no_content_fields"]         and all(v.startswith("refused") for v in exp.values())
    record("3_privacy", "NOT VERIFIED" if mech_ok else "FAIL", ev,
           "A human must inspect the review frames (and ideally whole shared segments) for readable faces / plates; blur recall is "
           "unmeasured and the sample is US dashcam footage, not the campaign camera. Decide whether a share/export guard is needed.")


# ---------------- 4. data integrity ----------------
@guarded("4_data_integrity")
def audit_integrity():
    ev = {}
    # (a) existing 30-min demo session (read only)
    tl = rwm.load_timeline(DEMO_SESSION)
    seg = json.loads((DEMO_SESSION / "video" / "segments.json").read_text(encoding="utf-8"))
    frames_decoded = 0
    for s in seg["segments"]:
        c = cv2.VideoCapture(str(resolve(s["file"])) if Path(s["file"]).is_absolute() else str(DEMO_SESSION / "video" / Path(s["file"]).name))
        frames_decoded += int(c.get(cv2.CAP_PROP_FRAME_COUNT))
        c.release()
    ts = tl.t_stream.to_numpy()
    resets = np.flatnonzero(tl.step.diff().fillna(1).to_numpy() <= 0)
    mono = all(np.all(np.diff(x) > 0) for x in np.split(ts, resets))
    audio = [json.loads(x) for x in (DEMO_SESSION / "audio_log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    ev["demo_30min_session"] = {"timeline_steps": len(tl), "video_frames_written": seg["frames_written"], "video_frames_decoded": frames_decoded,
                                "frames_dropped": seg["frames_dropped"], "integrity": rwm.frame_integrity(tl), "timestamps_monotonic_per_segment": mono,
                                "audio_log_records": len(audio), "audio_t_within_timeline": all(ts.min() <= a["t_stream"] <= ts.max() for a in audio),
                                "provenance_present": (DEMO_SESSION / "provenance.json").exists(),
                                "note": "recorded with v3.3 software: no provenance / system log / camera columns"}
    # (b) end-to-end v3.4 recording path: demo FILE replay into the audit folder (no camera, not data/sessions)
    from src.data.dataset_loader import index_videos
    from src.live.audio import AudioConfig
    from src.live.controller import LiveController
    from src.live.recorder import RecorderConfig
    from src.live.sources import VideoFileSource
    demo = index_videos(resolve("data/raw/nexar"))["00288"]
    root = RUN / "e2e" / "sessions"
    c = LiveController()
    c.start(VideoFileSource(str(demo), realtime=True, loop=True), V3, "balanced", None, RecorderConfig(min_free_mb=0),
            AudioConfig(enabled=True, volume=0.0), audio_backend="none", store_root=RUN / "e2e" / "memory")
    m = TripMeta("x", data_kind="demo_video", record_full_video=True, driver_consent=True, occupants_informed=True,
                 notes="pre-campaign audit: demo FILE replay, not real driving")
    s = c.start_session(m, sessions_root=root)
    time.sleep(120)
    rep = c.stop()
    d = root / s.session_id
    tl2 = rwm.load_timeline(d)
    sysl = pd.read_csv(d / "system_log.csv")
    prov = json.loads((d / "provenance.json").read_text(encoding="utf-8"))
    ent = campaign.session_entry(d)
    ev["v34_e2e_2min_demo"] = {"timeline_steps": len(tl2), "video": rep["video"] and {k: v for k, v in rep["video"].items() if k != "segments"},
                               "integrity": rwm.frame_integrity(tl2), "camera_state_values": sorted(tl2.camera_state.astype(str).unique()),
                               "system_log_rows": len(sysl), "system_log_gpu_filled": bool(pd.to_numeric(sysl.gpu_util, errors="coerce").notna().any()),
                               "provenance_check": prov["check"], "audio_log_exists": (d / "audio_log.jsonl").exists(),
                               "manifest_issues": ent["issues"], "sync_rel_diff": ent["sync_rel_diff"]}
    # (c) fault injection on a COPY: drop timeline rows and truncate the video -> must be flagged
    f = RUN / "fault" / "sessions"
    shutil.copytree(d, f / "fault1")
    t3 = pd.read_csv(f / "fault1" / "timeline.csv")
    t3.drop(index=range(100, 140)).to_csv(f / "fault1" / "timeline.csv", index=False)
    sj = json.loads((f / "fault1" / "video" / "segments.json").read_text(encoding="utf-8"))
    sj["frames_written"] = int(sj["frames_written"] * 0.8)
    (f / "fault1" / "video" / "segments.json").write_text(json.dumps(sj), encoding="utf-8")
    fe = campaign.session_entry(f / "fault1", with_phash=False)
    ev["fault_injection"] = {"issues": fe["issues"], "missing_steps": fe["missing_steps"]}
    detected = any("missing 10 Hz steps" in i for i in fe["issues"]) and any("desynchronised" in i for i in fe["issues"])
    # (d) storage estimate for 10 h from measured rates
    vid_mb = sum(p.stat().st_size for p in (DEMO_SESSION / "video").glob("*.mp4")) / 1e6
    hrs = len(tl) * 0.1 / 3600
    rep30 = json.loads((DEMO_SESSION / "session_report.json").read_text(encoding="utf-8"))
    clip_mb = np.mean([Path(e["clip"]).stat().st_size / 1e6 for e in rep30["events"] if e.get("clip") and Path(e["clip"]).exists()])
    est = {"full_video_mb_per_h": round(vid_mb / hrs, 0), "timeline_mb_per_h": round((DEMO_SESSION / "timeline.csv").stat().st_size / 1e6 / hrs, 1),
           "mean_clip_mb": round(float(clip_mb), 2)}
    for name, ev_per_h in (("benchmark-like 9 alarms/h + 20 CAUTION/h", 29), ("demo-like 108 events/h (upper bound)", 108)):
        tot = 10 * (est["full_video_mb_per_h"] + est["timeline_mb_per_h"] + ev_per_h * est["mean_clip_mb"])
        est[f"10h_raw_gb ({name})"] = round(tot / 1000, 1)
        est[f"10h_with_shared_copy_gb ({name})"] = round((tot + 10 * est["full_video_mb_per_h"]) / 1000, 1)
    est["disk_free_gb_D"] = round(shutil.disk_usage(resolve(".")).free / 1e9, 1)
    ev["storage_estimate"] = est
    # (e) backup / restore round trip of a session (copy -> verify SHA-256 -> restore -> verify)
    bk = RUN / "backup_test"
    src_h = sha_dir(d)
    shutil.copytree(d, bk / "backup" / d.name)
    shutil.copytree(bk / "backup" / d.name, bk / "restored" / d.name)
    ev["backup_restore_roundtrip_identical"] = sha_dir(bk / "restored" / d.name) == src_h
    import importlib.util
    spec = importlib.util.spec_from_file_location("cb", resolve("scripts/campaign_backup.py"))
    cb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cb)
    camp_copy = RUN / "backup_tool" / "campaign"
    camp_copy.mkdir(parents=True)
    r1 = cb.backup(RUN / "backup_tool" / "dest", None, True, root, camp_copy)
    r2 = cb.backup(RUN / "backup_tool" / "dest", None, True, root, camp_copy)
    v = cb.verify(RUN / "backup_tool" / "dest")
    rs = cb.restore(RUN / "backup_tool" / "dest", s.session_id, RUN / "backup_tool" / "restored")
    ev["backup_tool"] = {"first_run": r1, "second_run_incremental": r2, "verify": {k: v[k] for k in ("files", "ok")}, "restore": rs}
    import psutil
    data_drive = resolve(".").drive.upper()
    ev["drives"] = [{"mount": p_.mountpoint, "opts": p_.opts} for p_ in psutil.disk_partitions()]
    ext = [x for x in ev["drives"] if x["mount"].upper()[:2] not in (data_drive, "C:")]
    ev["external_backup_drive_present"] = bool(ext)
    ev["campaign_backup_target_defined"] = bool(ext)
    e2e_ok = ev["v34_e2e_2min_demo"]["integrity"]["missing_steps"] == 0 and prov["check"]["eligible"] and \
        not ev["v34_e2e_2min_demo"]["sync_rel_diff"] or (ev["v34_e2e_2min_demo"]["sync_rel_diff"] or 0) <= 0.01
    ok = detected and e2e_ok and ev["backup_restore_roundtrip_identical"] and v["ok"] and r2["copied"] == 0 and mono and est["disk_free_gb_D"] > est["10h_with_shared_copy_gb (demo-like 108 events/h (upper bound))"]
    status = "FAIL" if not ok else ("PASS" if ev["external_backup_drive_present"] else "BLOCKED")
    record("4_data_integrity", status, ev,
           None if status == "PASS" else "Integrity checks and the backup tool pass, but no external (encrypted) backup drive is "
           "connected. Connect one and run: python scripts/campaign_backup.py backup --dest <drive>, then verify / restore test.")


# ---------------- 5. annotation ----------------
@guarded("5_annotation")
def audit_annotation():
    from streamlit.testing.v1 import AppTest
    import src.realworld.trip_meta as tmod
    root = RUN / "annotation_sample" / "sessions"
    d = root / "ann_sample"
    (d / "video").mkdir(parents=True)
    shutil.copy(DEMO_SESSION / "video" / "segment_000.mp4", d / "video" / "segment_000.mp4")
    (d / "video" / "segments.json").write_text(json.dumps({"segments": [{"t_start": 0.0, "t_end": 299.9, "frames": 3000}],
                                                           "frames_written": 3000, "frames_dropped": 0, "errors": []}), encoding="utf-8")
    shutil.copy(DEMO_SESSION / "timeline.csv", d / "timeline.csv")
    save_meta(TripMeta("ann_sample", data_kind="demo_video", record_full_video=True, driver_consent=True, occupants_informed=True,
                       notes="AUDIT SAMPLE (demo replay), not campaign data"), root)
    E = lambda who, cat, t0, t1, ct, note: ann.Annotation("ann_sample", who, cat, t0, t1, ct, evidence="visible gap closing to lead car",  # noqa: E731
                                                          evidence_frames=[ct] if ct is not None else [], note=note)
    # two independent annotators (simulated labels on the sample, only to exercise the workflow)
    for a in [E("ANN-A", "risk_event", 10, 14, 13, "A-NOTE-1"), E("ANN-A", "near_miss", 40, 45, 44, "A-NOTE-2"),
              ann.Annotation("ann_sample", "ANN-A", "uncertain", 70, 72, note="A-NOTE-3"), E("ANN-A", "risk_event", 100, 103, 102, "A-NOTE-4")]:
        ann.add(d, a)
    for b in [E("ANN-B", "risk_event", 10.4, 14, 13.5, "B-NOTE-1"), E("ANN-B", "risk_event", 40.2, 45, 44.5, "B-NOTE-2"),
              ann.Annotation("ann_sample", "ANN-B", "uncertain", 70, 72.5, note="B-NOTE-3")]:
        ann.add(d, b)
    patched = {"raw_access": ["ANN-A", "ANN-B"], "coordinators": ["COORD"]}
    orig = (privacy.load_access, tmod.SESSIONS_DIR, tmod.list_sessions, tmod.load_meta, tmod.save_meta)
    # audit-only, in-process patches: the access list and the sessions root point at the audit sample
    # (data/campaign/access.json and data/sessions are untouched)
    privacy.load_access = lambda path=None: patched
    tmod.SESSIONS_DIR = root
    tmod.list_sessions = lambda r=root: orig[2](root)
    tmod.load_meta = lambda sid, r=root: orig[3](sid, root)
    tmod.save_meta = lambda m, r=root: orig[4](m, root)
    ev = {}
    try:
        at = AppTest.from_file(str(resolve("pages/11_Ground_Truth_Annotation.py")), default_timeout=120)
        at.run()
        at.text_input[0].input("ANN-A").run()
        page_text = " ".join(str(getattr(e, "value", "")) for e in list(at.markdown) + list(at.caption) + list(at.info) + list(at.success))
        dfs = [x.value for x in at.dataframe]
        shown = pd.concat(dfs) if dfs else pd.DataFrame()
        ev["annotator_view"] = {"videos": len(at.get("video")), "plotly_charts": len(at.get("plotly_chart")),
                                "dataframes": len(dfs), "rows_shown": len(shown),
                                "annotators_in_table": sorted(shown["annotator"].unique()) if "annotator" in shown else [],
                                "other_annotator_notes_visible": any("B-NOTE" in str(v) for v in shown.get("note", [])) or "B-NOTE" in page_text,
                                "model_output_terms_visible": [t for t in ("P(collision)", "alarm onset", "Model output") if t in page_text],
                                "kappa_visible": "κ" in page_text}
        blind = ev["annotator_view"]["videos"] == 1 and ev["annotator_view"]["plotly_charts"] == 0 and \
            ev["annotator_view"]["annotators_in_table"] == ["ANN-A"] and not ev["annotator_view"]["other_annotator_notes_visible"] and \
            not ev["annotator_view"]["model_output_terms_visible"] and not ev["annotator_view"]["kappa_visible"]
        # coordinator before full reviews -> hidden
        at.radio[0].set_value("Coordinator").run()
        at.text_input[0].input("COORD").run()
        ev["coordinator_before_full_reviews"] = [i.value for i in at.info]
        hidden_before = any("Needs two annotators" in x for x in ev["coordinator_before_full_reviews"])
        m = orig[3]("ann_sample", root)
        m.fully_reviewed_by = ["ANN-A", "ANN-B"]
        orig[4](m, root)
        at2 = AppTest.from_file(str(resolve("pages/11_Ground_Truth_Annotation.py")), default_timeout=120)
        at2.run()
        at2.radio[0].set_value("Coordinator").run()
        at2.text_input[0].input("COORD").run()
        ev["coordinator_after_metrics"] = {x.label: x.value for x in at2.metric}
    finally:
        privacy.load_access, tmod.SESSIONS_DIR, tmod.list_sessions, tmod.load_meta, tmod.save_meta = orig
    A, B = ann.load(d, "ANN-A"), ann.load(d, "ANN-B")
    r = ann.agreement(A, B)
    cons = ann.consensus(A, B)
    disputed = next(x for x in cons if x["status"] == "disputed")
    ann.add_disagreement(d, ann.Disagreement("ann_sample", disputed["key"], "ANN-A", "evasive swerve visible at 44 s", "unresolved"))
    ann.add_disagreement(d, ann.Disagreement("ann_sample", disputed["key"], "ANN-B", "no evasive action, normal braking", "unresolved"))
    cons2 = ann.consensus(A, B, ann.load_disagreements(d))
    # expected values computed by hand for this sample
    exp_kappa_matched = ann.cohen_kappa(["risk_event", "near_miss", "uncertain"], ["risk_event", "risk_event", "uncertain"])
    ev["agreement"] = {k: r[k] for k in ("matched", "only_first", "only_second", "kappa_category", "kappa_with_unmatched",
                                         "kappa_event_vs_not", "offset_start", "offset_event_time", "offset_end")}
    ev["consensus_after_disagreement"] = [(x["category"], x["status"], x["reason"]) for x in cons2]
    checks = {"kappa_matches_hand_computation": abs(r["kappa_category"] - exp_kappa_matched) < 1e-12,
              "event_time_offsets_mean": r["offset_event_time"]["mean_abs_s"], "expected_event_offset_mean": 0.5,
              "disputed_stays_disputed": [x[1] for x in ev["consensus_after_disagreement"]] == ["confirmed", "disputed", "confirmed", "single"],
              "uncertain_confirmed_as_uncertain": any(x[0] == "uncertain" for x in ev["consensus_after_disagreement"])}
    # metrics: an alarm in the uncertain window is excluded, not false
    tl = rwm.load_timeline(d)
    on = rwm.alarm_onsets(tl)
    checks["timeline_alarm_onsets_in_sample"] = int(len(on))
    ev["checks"] = checks
    ok = blind and hidden_before and checks["kappa_matches_hand_computation"] and abs(checks["event_time_offsets_mean"] - 0.5) < 1e-9 and \
        checks["disputed_stays_disputed"] and len(ev["coordinator_after_metrics"]) >= 4
    record("5_annotation", "PASS" if ok else "FAIL", ev,
           "Workflow verified with simulated labels on a demo sample only; label quality with real annotators is unverified. "
           "Blindness relies on separate names: two people must not share a name, and the page cannot stop someone from typing another name.")


# ---------------- 6. dataset governance ----------------
@guarded("6_dataset_governance")
def audit_governance():
    man = campaign.build_manifest(resolve("data/sessions"), RUN / "governance_campaign_copy", with_phash=True, assign=False)
    ss = man["sessions"]
    legacy = [e for e in ss if e["data_kind"] == "legacy_unknown"]
    ev = {"sessions": len(ss), "legacy_unknown": len(legacy), "legacy_in_partition": [e["session_id"] for e in legacy if e["partition"] != "excluded"],
          "legacy_eligible": [e["session_id"] for e in legacy if e["eligible"]],
          "sessions_with_video": sum(bool(e["video_segments"]) for e in ss),
          "all_segments_have_sha256_and_phash": all(len(s["sha256"]) == 64 and len(s["phash"]) == 8 for e in ss for s in e["video_segments"]),
          "leakage": man["leakage"]}
    # group-wise split on a synthetic campaign copy (directory of empty sessions; no video)
    cdir = RUN / "governance_split"
    groups = {f"driver:D{i:02d}": campaign.assign_partition(f"driver:D{i:02d}", cdir) for i in range(30)}
    again = {g: campaign.assign_partition(g, cdir) for g in groups}
    ev["split"] = {"partitions_used": sorted(set(groups.values())), "stable_on_reassign": groups == again,
                   "counts": {p: sum(v == p for v in groups.values()) for p in campaign.PARTITIONS}}
    # final_test locked by default
    met = campaign.campaign_metrics(man, root=resolve("data/sessions"), cdir=RUN / "governance_campaign_copy")
    ev["default_metric_partitions"] = met["partitions"]
    ev["final_test_access_log_created_by_default"] = (RUN / "governance_campaign_copy" / "final_test_access_log.jsonl").exists()
    import subprocess
    pr = subprocess.run([sys.executable, str(resolve("scripts/campaign_report.py")), "--final-test"], capture_output=True, text=True, cwd=str(resolve(".")))
    ev["campaign_report_final_test_without_by"] = {"returncode": pr.returncode, "stderr_tail": pr.stderr.strip()[-200:]}
    ok = not ev["legacy_in_partition"] and not ev["legacy_eligible"] and ev["all_segments_have_sha256_and_phash"] and ev["split"]["stable_on_reassign"] \
        and "final_test" not in met["partitions"] and not ev["final_test_access_log_created_by_default"] and pr.returncode != 0 and man["leakage"]["ok"]
    record("6_dataset_governance", "PASS" if ok else "FAIL", ev,
           "Partitions become meaningful only with several drivers / trips; with one driver all data falls into one partition.")


def sharing_block_test() -> dict:
    """If the coordinator disabled sharing, prove on a throw-away copy that export is refused even for an approved copy."""
    pol = privacy.sharing_policy()
    if pol.get("sharing_enabled", True):
        return {"applicable": False}
    d = RUN / "sharing_block_test" / "s_test"
    (d / "shared").mkdir(parents=True)
    (d / "shared" / "segment_000.mp4").write_bytes(b"BLURRED-TEST")
    coord = privacy.load_access().get("coordinators", [""])[0]
    privacy.review(d, coord, True, ["segment_000.mp4"], "audit test copy", identifiable_found=False,
                   log_path=RUN / "sharing_block_test" / "review_log.jsonl")  # never the real review log
    try:
        privacy.export_shared(d, RUN / "sharing_block_test" / "out", coord, RUN / "sharing_block_test" / "export_log.jsonl")
        return {"applicable": True, "export_refused": False}
    except privacy.ExportRefused as e:
        return {"applicable": True, "export_refused": True, "reason": str(e)}


def main() -> int:
    import argparse
    from src.realworld import readiness
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also re-run the v3.4 mechanism tests on copies")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    log("SafeDrive v3.4.4 pre-campaign readiness audit", RUN, "elevated =", readiness.is_elevated())
    items = readiness.check_all()
    sb = sharing_block_test()
    for it in items:
        if it["key"] == "11_privacy_review" and it["evidence"].get("mode", "").startswith("sharing disabled"):
            it["evidence"]["active_export_block_test"] = sb
            if not sb.get("export_refused"):
                it["status"], it["action"] = "FAIL", "Export was NOT refused while sharing is disabled: fix before any session."
    for it in items:
        log(f"[{it['status']}] {it['key']}: {it['title']}" + (f"  -> {it['action']}" if it.get("action") else ""))
    rec = readiness.save_latest(items, run_dir=str(RUN))
    (RUN / "readiness_results.json").write_text(json.dumps(rec, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    log("VERDICT", rec["decision"]["verdict"], rec["decision"]["counts"])
    if a.full:
        for fn in (audit_hashes, audit_access, audit_privacy, audit_integrity, audit_annotation, audit_governance):
            fn()
        after = {"sessions": sorted(p.name for p in resolve("data/sessions").iterdir()),
                 "access": provenance.sha256_file(resolve("data/campaign/access.json")),
                 "frozen": provenance.sha256_file(provenance.FROZEN_PATH)}
        RESULTS["project_unchanged_by_audit"] = {"status": "PASS" if after == SNAP_BEFORE else "FAIL",
                                                 "evidence": {"before": SNAP_BEFORE, "after": after}}
        (RUN / "audit_results.json").write_text(json.dumps(RESULTS, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        log("MECHANISM STATUS", {k: v["status"] for k, v in RESULTS.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
