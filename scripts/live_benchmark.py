"""Headless evaluation of the Live Driving Assistant (v3.2). Writes experiments/live/<name>.json.

Measures, end to end through the real LiveController: camera capture FPS, processed steps per second (target 10 Hz),
per-step processing time, end-to-end latency (frame capture -> annotated frame ready), CPU %, GPU % and VRAM
(nvidia-smi), process RSS over time (MB/min slope), audio decisions and API latency, event-recording reliability (each clip
re-opened and its frame count checked), camera interruptions / reconnects.

Privacy: camera scenarios use --no-record by default, so no camera image is written to disk.

Examples:
  python scripts/live_benchmark.py --name cam5 --source camera:0 --minutes 5 --no-record
  python scripts/live_benchmark.py --name demo_rec --source file:<video> --minutes 5 --profile sensitive --audio-volume 0
  python scripts/live_benchmark.py --name disconnect --source camera:0 --minutes 1.5 --no-record --simulate-disconnect 30
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import psutil  # noqa: E402

from src.live.audio import AudioConfig  # noqa: E402
from src.live.controller import LiveController  # noqa: E402
from src.live.recorder import RecorderConfig  # noqa: E402
from src.live.sources import CameraSource, VideoFileSource  # noqa: E402
from src.utils.config import resolve  # noqa: E402
from src.utils.io import save_json  # noqa: E402


def gpu_sample() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().split(",")
        return {"gpu_util": float(out[0]), "gpu_mem_mb": float(out[1])}
    except Exception:  # noqa: BLE001
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--source", default="camera:0")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--model", default="models/safedrive_v3")
    ap.add_argument("--profile", default="balanced")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--min-free-mb", type=float, default=1024.0)
    ap.add_argument("--audio-backend", default="auto")
    ap.add_argument("--audio-volume", type=float, default=0.0, help="0 = silent playback (exercises the audio path)")
    ap.add_argument("--simulate-disconnect", type=float, default=None, help="seconds after start: force-release the device once")
    ap.add_argument("--store", default="data/live_bench_store")
    ap.add_argument("--data-kind", default=None, help="v3.3: write trip metadata with this data kind (demo_video / live_camera_test)")
    ap.add_argument("--full-video", action="store_true", help="v3.3: also record the full session video (demo / consented only)")
    ap.add_argument("--keep-awake", action="store_true", help="prevent system sleep while THIS process runs (process-scoped)")
    a = ap.parse_args()
    if a.keep_awake and sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    kind, val = a.source.split(":", 1)
    src = CameraSource(int(val)) if kind == "camera" else VideoFileSource(val, realtime=True, loop=True)
    store = resolve(a.store) / a.name
    ctl = LiveController()
    t_load = time.time()
    ctl.start(src, resolve(a.model), a.profile, a.device, RecorderConfig(min_free_mb=a.min_free_mb),
              AudioConfig(enabled=True, volume=a.audio_volume), record_events=not a.no_record, audio_backend=a.audio_backend,
              store_root=store)
    load_s = time.time() - t_load
    meta = None
    if a.data_kind:
        from src.realworld.trip_meta import TripMeta
        meta = TripMeta("pending", data_kind=a.data_kind, record_full_video=a.full_video, driver_consent=a.full_video,
                        occupants_informed=a.full_video, notes=f"live_benchmark {a.name}: {a.source}")
    sess = ctl.start_session(meta)
    proc = psutil.Process()
    proc.cpu_percent(None)
    samples, t0, disc_done, recover = [], time.time(), False, None
    dur = a.minutes * 60
    last_gpu = 0.0
    while time.time() - t0 < dur and ctl.running:
        time.sleep(2.0)
        s, _ = ctl.snapshot()
        row = {"t": time.time() - t0, "rss_mb": proc.memory_info().rss / 1e6, "cpu_percent": proc.cpu_percent(None) / psutil.cpu_count(),
               "capture_fps": s.get("capture_fps"), "step": s.get("step"), "step_ms": s.get("step_ms"), "e2e_ms": s.get("e2e_ms"),
               "camera": s.get("camera"), "interruptions": s.get("interruptions")}
        if time.time() - last_gpu > 10:
            row.update(gpu_sample())
            last_gpu = time.time()
        samples.append(row)
        if a.simulate_disconnect and not disc_done and row["t"] >= a.simulate_disconnect:
            disc_done, t_disc = True, time.time()
            ctl.capture.source.release()  # the device disappears under the capture thread
        if disc_done and recover is None and ctl.capture.stats.reconnects > 0:
            recover = time.time() - t_disc
    alive_at_end = ctl.running
    err = ctl.state.get("error", "")
    audio_log = list(ctl.audio.log) if ctl.audio else []
    rep = ctl.stop()
    sm = rep["summary"] if rep else {}
    clips = []
    for e in (rep or {}).get("events", []):
        info = {"category": e["category"], "clip_status": e["clip_status"], "frames_written": e["clip_frames"]}
        if e.get("clip"):
            cap = cv2.VideoCapture(e["clip"])
            info["frames_decoded"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        clips.append(info)
    rss = np.array([r["rss_mb"] for r in samples]) if samples else np.zeros(1)
    tt = np.array([r["t"] for r in samples]) if samples else np.zeros(1)
    half = len(rss) // 2
    gpu = [r for r in samples if "gpu_util" in r]
    out = {
        "name": a.name, "source": a.source, "model": a.model, "profile": a.profile, "device": ctl.state.get("device"),
        "minutes_requested": a.minutes, "ran_s": float(tt[-1]) if len(tt) else 0.0, "alive_at_end": alive_at_end, "error": err,
        "model_load_and_open_s": load_s,
        "capture_fps_mean": float(np.nanmean([r["capture_fps"] or np.nan for r in samples])) if samples else None,
        "processed_steps_per_s": sm.get("processed_steps_per_s"), "late_steps_fraction": sm.get("late_steps_fraction"),
        "mean_step_interval_s": sm.get("mean_step_interval_s"),
        "step_ms_p50": sm.get("step_ms_p50"), "step_ms_p95": sm.get("step_ms_p95"), "e2e_ms_p50": sm.get("e2e_ms_p50"), "e2e_ms_p95": sm.get("e2e_ms_p95"),
        "cpu_percent_of_machine_mean": float(np.mean([r["cpu_percent"] for r in samples[1:]])) if len(samples) > 1 else None,
        "gpu_util_mean": float(np.mean([r["gpu_util"] for r in gpu])) if gpu else None,
        "gpu_mem_mb_max": float(np.max([r["gpu_mem_mb"] for r in gpu])) if gpu else None,
        "rss_mb_start": float(rss[0]), "rss_mb_end": float(rss[-1]), "rss_mb_max": float(rss.max()),
        "rss_slope_mb_per_min_second_half": float(np.polyfit(tt[half:] / 60, rss[half:], 1)[0]) if len(rss) - half > 3 else None,
        "camera_interruptions": sm.get("camera_interruptions"), "camera_reconnects": sm.get("camera_reconnects"),
        "disconnect_recovery_s": recover,
        "alarms": sm.get("alarms"), "level_onsets": sm.get("level_onsets"), "events": sm.get("events"),
        "clips": clips, "clips_saved": sm.get("clips_saved"), "clips_failed": sm.get("clips_failed"),
        "audio_decisions": len(audio_log), "audio_played": sum(r["played"] for r in audio_log),
        "audio_reasons": {k: sum(r["reason"] == k for r in audio_log) for k in {r["reason"] for r in audio_log}},
        "audio_api_ms_max": max([r["api_ms"] for r in audio_log if r["api_ms"] is not None], default=None),
        "audio_min_gap_s": float(np.min(np.diff([r["t"] for r in audio_log if r["played"]]))) if sum(r["played"] for r in audio_log) > 1 else None,
        "samples": samples,
    }
    if a.data_kind:
        from src.realworld.metrics import session_metrics
        out["session_id"] = sess.session_id
        out["realworld_metrics"] = session_metrics(resolve("data/sessions") / sess.session_id)
        out["full_video"] = (rep or {}).get("video")
    p = resolve("experiments/live") / f"{a.name}.json"
    save_json(out, p)
    print("RESULT", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items() if k not in ("samples", "clips")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
