"""Profile every perception component on real Nexar clips (val split only, first 15 s of N videos).

Measures per-frame wall time (ms) of: decode+resize, detector+tracker variants (YOLOv8n / YOLO11n at several input sizes),
Farneback flow, ResNet-18 embedding, and a cheap frame-difference descriptor; plus peak VRAM / RSS.
Without box ground truth, detector variants are compared to YOLOv8n@640 by detection count and track-agreement only.
Output: experiments/latest/perception_profile.json
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402

from src.data.dataset_loader import index_videos  # noqa: E402
from src.detection.detector import COCO_ROAD  # noqa: E402
from src.features.frame_diff import FrameDiff  # noqa: E402
from src.features.optical_flow import FlowComputer  # noqa: E402
from src.features.visual_features import VisualEncoder  # noqa: E402
from src.preprocessing.frame_extractor import resize_width  # noqa: E402
from src.preprocessing.frame_sampler import sample_on_grid  # noqa: E402
from src.preprocessing.video_reader import iter_frames  # noqa: E402
from src.tracking.tracker import Tracker  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.io import load_json, save_json  # noqa: E402


def load_frames(path, seconds=15.0, width=640):
    t0 = time.perf_counter()
    fr = [resize_width(f, width) for _, _, _, f in sample_on_grid(iter_frames(path, 0, seconds), 10.0)]
    return fr, (time.perf_counter() - t0) * 1000 / max(len(fr), 1)


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def main() -> int:
    cfg = load_config()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vids = index_videos(resolve(cfg["paths"]["raw_dir"]))
    val = load_json(resolve(cfg["paths"]["splits"]))["val"][:8]
    clips = []
    dec = []
    for v in val:
        fr, ms = load_frames(vids[v])
        clips.append(fr)
        dec.append(ms)
    n_frames = sum(len(c) for c in clips)
    out = {"device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None, "n_videos": len(clips),
           "n_frames": n_frames, "decode_resize_ms_per_frame": float(np.mean(dec)), "detectors": {}}
    ref = None
    for weights, imgsz in [("yolov8n.pt", 640), ("yolov8n.pt", 480), ("yolov8n.pt", 320), ("yolo11n.pt", 640), ("yolo11n.pt", 480),
                           ("yolo11n.pt", 320)]:
        wpath = resolve("models") / weights
        tr = Tracker(str(wpath) if wpath.exists() else weights, "bytetrack.yaml", 0.25, 0.5, list(COCO_ROAD), dev, imgsz)
        if not wpath.exists() and Path(weights).exists():
            Path(weights).replace(wpath)
        for _ in range(5):
            tr.update(clips[0][0], 0)
        torch.cuda.reset_peak_memory_stats() if dev == "cuda" else None
        dets_all, t_all = [], 0.0
        for c in clips:
            tr.reset()
            for i, f in enumerate(c):
                t0 = time.perf_counter()
                d = tr.update(f, i)
                t_all += time.perf_counter() - t0
                dets_all.append([x.xyxy for x in d])
        key = f"{Path(weights).stem}@{imgsz}"
        row = {"ms_per_frame": 1000 * t_all / n_frames, "fps": n_frames / t_all,
               "mean_dets_per_frame": float(np.mean([len(d) for d in dets_all])),
               "peak_vram_mb": torch.cuda.max_memory_allocated() / 1e6 if dev == "cuda" else None}
        if ref is None:
            ref = dets_all
        else:  # agreement with YOLOv8n@640: fraction of reference boxes matched (IoU>=0.5)
            m = tot = 0
            for a, b in zip(ref, dets_all):
                tot += len(a)
                m += sum(any(iou(x, y) >= 0.5 for y in b) for x in a)
            row["recall_vs_v8n640"] = m / max(tot, 1)
        out["detectors"][key] = row
        print(key, {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
        del tr
        torch.cuda.empty_cache() if dev == "cuda" else None
    fl = FlowComputer(tuple(cfg["preprocessing"]["flow_size"]))
    t0 = time.perf_counter()
    for c in clips:
        fl.reset()
        for f in c:
            fl(f, 0.1)
    out["flow_ms_per_frame"] = 1000 * (time.perf_counter() - t0) / n_frames
    fd = FrameDiff()
    t0 = time.perf_counter()
    for c in clips:
        fd.reset()
        for f in c:
            fd(f)
    out["frame_diff_ms_per_frame"] = 1000 * (time.perf_counter() - t0) / n_frames
    enc = VisualEncoder(dev)
    enc.encode(clips[0][:32])
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for c in clips:
        enc.encode(c)
    if dev == "cuda":
        torch.cuda.synchronize()
    out["resnet18_ms_per_frame"] = 1000 * (time.perf_counter() - t0) / n_frames
    out["process_rss_gb"] = psutil.Process().memory_info().rss / 1e9
    print({k: v for k, v in out.items() if k != "detectors"})
    save_json(out, resolve("experiments/latest/perception_profile.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
