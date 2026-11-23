"""End-to-end benchmark of the deployed pipelines (decode -> detect/track -> features -> prediction -> risk) on real test
videos: wall-clock FPS, per-stage time, peak VRAM, process RSS. Output: experiments/latest/pipeline_benchmark.json
Usage: python scripts/benchmark_pipeline.py [--n 5]
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psutil  # noqa: E402
import torch  # noqa: E402

from src.data.dataset_loader import index_videos  # noqa: E402
from src.pipeline import SafeDrivePipeline  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.io import load_json, save_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    cfg = load_config()
    vids = index_videos(resolve(cfg["paths"]["raw_dir"]))
    test = load_json(resolve(cfg["paths"]["splits"]))["test"][: a.n]
    out = {"videos": test}
    ap_models = [("v3", "models/safedrive_v3"), ("v3_stabilized", "models/safedrive_v3_stabilized"),
                 ("v2_light", "models/safedrive_v2_light"), ("v1_lstm", "models/safedrive_v1")]
    for name, d in [m for m in ap_models if (resolve(m[1]) / "bundle.json").exists()]:
        for dev in (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]):
            if dev == "cpu" and name == "v1_lstm":
                continue  # v1 on CPU is very slow (ResNet + 100 MC passes); measured on GPU only
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            pipe = SafeDrivePipeline(resolve(d), cfg=cfg, device=dev)
            pipe.analyze(vids[test[0]], keep_frames=False, max_duration_s=3.0)  # warm-up
            frames = ext = pred = 0.0
            t0 = time.perf_counter()
            for v in test:
                r = pipe.analyze(vids[v], keep_frames=False)
                frames += len(r.features)
                ext += r.timings["extract_s"]
                pred += r.timings["predict_s"]
            wall = time.perf_counter() - t0
            key = f"{name}@{dev}"
            out[key] = {"frames": frames, "wall_s": wall, "end_to_end_fps": frames / wall, "realtime_factor_10hz": frames / wall / 10.0,
                        "extract_s": ext, "predict_s": pred, "predict_ms_per_frame": 1000 * pred / frames,
                        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0,
                        "process_rss_gb": psutil.Process().memory_info().rss / 1e9, "cnn_loaded": pipe.extractor.encoder is not None}
            print(key, {k: round(v, 3) if isinstance(v, float) else v for k, v in out[key].items()}, flush=True)
            del pipe
    save_json(out, resolve("experiments/latest/pipeline_benchmark_v3.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
