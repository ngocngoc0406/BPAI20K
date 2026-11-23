"""Print a compact summary of experiments/live/*.json (no new measurement)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import resolve  # noqa: E402

KEYS = ["ran_s", "alive_at_end", "error", "device", "capture_fps_mean", "processed_steps_per_s", "late_steps_fraction", "step_ms_p50",
        "step_ms_p95", "e2e_ms_p50", "e2e_ms_p95", "cpu_percent_of_machine_mean", "gpu_util_mean", "gpu_mem_mb_max", "rss_mb_start",
        "rss_mb_end", "rss_mb_max", "rss_slope_mb_per_min_second_half", "camera_interruptions", "camera_reconnects",
        "disconnect_recovery_s", "alarms", "events", "clips_saved", "clips_failed", "audio_decisions", "audio_played",
        "audio_reasons", "audio_api_ms_max", "audio_min_gap_s"]
for f in sorted(resolve("experiments/live").glob("*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    if "samples" not in d:
        print(f.name, {k: v for k, v in d.items() if k != "rows"})
        continue
    print(d["name"], {k: (round(d[k], 2) if isinstance(d.get(k), float) else d.get(k)) for k in KEYS})
    if d.get("clips"):
        print("   clips", d["clips"][:12])
