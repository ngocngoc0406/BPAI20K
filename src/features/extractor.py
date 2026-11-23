"""End-to-end per-video feature extraction: decode -> FPS grid -> detect+track -> motion/flow -> CNN."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np

from src.features.ego_motion import EgoMotionEstimator
from src.features.frame_diff import FrameDiff
from src.features.motion_features import MOTION_FEATURES, MotionFeatureComputer
from src.features.optical_flow import FlowComputer
from src.features.visual_features import VisualEncoder
from src.preprocessing.feature_cache import VideoFeatures
from src.preprocessing.frame_extractor import resize_width
from src.preprocessing.frame_sampler import sample_on_grid
from src.preprocessing.video_reader import get_meta, iter_frames
from src.tracking.tracker import Tracker
from src.utils.config import resolve
from src.utils.device import get_device, is_oom
from src.utils.logging_utils import get_logger

log = get_logger("extractor")


class ProcessingCancelled(RuntimeError):
    pass


class FeatureExtractor:
    """Holds the heavy models; reuse one instance for many videos."""

    def __init__(self, cfg: dict, device: str | None = None, use_cnn: bool = True, use_ego: bool = True):
        """use_cnn=False is the lightweight mode: no ResNet-18 (visual has 0 columns); frame-difference is always computed."""
        self.cfg = cfg
        self.use_cnn = use_cnn
        self.use_ego = use_ego  # camera ego-motion (~4 ms/frame); skipped when the model does not use it
        self.device = device or get_device()
        det, pre = cfg["detection"], cfg["preprocessing"]
        self.fps = float(pre["target_fps"])
        self.width = int(pre["detect_width"])
        weights = resolve(det["weights"])
        weights = str(weights) if weights.exists() else Path(det["weights"]).name  # official auto-download
        self.tracker = Tracker(weights, cfg["tracking"]["tracker"], det["conf"], det["iou"],
                               det["classes"], self.device, imgsz=self.width)
        self.encoder = VisualEncoder(self.device, pre["visual_size"]) if use_cnn else None
        self.flow_size = tuple(pre["flow_size"])
        self.k = int(cfg["tracking"]["history_steps"])

    def _fallback_cpu(self) -> None:
        log.warning("CUDA out of memory -> falling back to CPU")
        import torch
        torch.cuda.empty_cache()
        self.device = "cpu"
        self.tracker.device = "cpu"
        if self.use_cnn:
            self.encoder = VisualEncoder("cpu", self.cfg["preprocessing"]["visual_size"])

    def extract(self, video_path: str | Path, video_id: str | None = None, start_s: float = 0.0,
                end_s: float | None = None, transform: Callable[[np.ndarray], np.ndarray] | None = None,
                progress: Callable[[float, str], None] | None = None,
                should_cancel: Callable[[], bool] | None = None, keep_frames: bool = False,
                max_duration_s: float | None = None) -> tuple[VideoFeatures, list[np.ndarray]]:
        try:
            return self._extract(video_path, video_id, start_s, end_s, transform, progress, should_cancel,
                                 keep_frames, max_duration_s)
        except RuntimeError as e:
            if self.device == "cuda" and is_oom(e):
                self._fallback_cpu()
                return self._extract(video_path, video_id, start_s, end_s, transform, progress, should_cancel,
                                     keep_frames, max_duration_s)
            raise

    def _extract(self, video_path, video_id, start_s, end_s, transform, progress, should_cancel, keep_frames,
                 max_duration_s):
        meta = get_meta(video_path)
        if max_duration_s is not None and end_s is None and meta.duration_s > max_duration_s:
            end_s = start_s + max_duration_s
            log.warning("Video %.1fs longer than limit; analysing first %.1fs", meta.duration_s, max_duration_s)
        total = ((end_s if end_s is not None else meta.duration_s) - start_s) * self.fps
        self.tracker.reset()
        motion = MotionFeatureComputer(self.fps, self.k)
        flow = FlowComputer(self.flow_size)
        fdiff = FrameDiff(self.flow_size)
        egom = EgoMotionEstimator()
        diffs: list[np.ndarray] = []
        egos: list[np.ndarray] = []
        dt = 1.0 / self.fps
        times, mot, track_rows, vis_frames, kept = [], [], [], [], []
        t0 = time.perf_counter()
        visual_chunks: list[np.ndarray] = []
        for step, t, _src, frame in sample_on_grid(iter_frames(video_path, start_s, end_s), self.fps, start_s):
            if should_cancel is not None and should_cancel():
                raise ProcessingCancelled("Processing cancelled by user")
            frame = resize_width(frame, self.width)
            if transform is not None:
                frame = transform(frame)
            h, w = frame.shape[:2]
            dets = self.tracker.update(frame, step)
            boxes = np.array([d.xyxy for d in dets], np.float32).reshape(-1, 4) / np.array([w, h, w, h], np.float32)
            ids = np.array([d.track_id for d in dets], int)
            cls = np.array([d.cls for d in dets], int)
            conf = np.array([d.conf for d in dets], np.float32)
            fl = flow(frame, dt)
            diffs.append(fdiff(frame))
            if self.use_ego:
                egos.append(egom(frame, boxes, dt, flow.last_field, self.flow_size))
            mot.append(motion.update(step, boxes, ids, cls, conf, fl, float(frame.mean() / 255.0)))
            for i in range(len(dets)):
                track_rows.append([step, ids[i], cls[i], conf[i], *boxes[i]])
            times.append(t)
            if self.use_cnn:
                vis_frames.append(frame)
            if keep_frames:
                kept.append(frame)
            if len(vis_frames) >= 64:
                visual_chunks.append(self.encoder.encode(vis_frames))
                vis_frames = []
            if progress is not None and step % 5 == 0:
                progress(min(step / max(total, 1), 0.99), f"frame {step}/{int(total)}")
        if vis_frames:
            visual_chunks.append(self.encoder.encode(vis_frames))
        if not times:
            raise ValueError(f"No frames decoded from {video_path} in [{start_s}, {end_s}]")
        elapsed = time.perf_counter() - t0
        feats = VideoFeatures(
            video_id=video_id or Path(video_path).stem,
            times=np.array(times, np.float32),
            visual=(np.concatenate(visual_chunks) if visual_chunks else np.zeros((len(times), 0))).astype(np.float16),
            motion=np.stack(mot).astype(np.float32),
            tracks=np.array(track_rows, np.float32).reshape(-1, 8),
            meta={"target_fps": self.fps, "source_fps": meta.fps, "width": meta.width, "height": meta.height,
                  "duration_s": meta.duration_s, "start_s": start_s, "end_s": end_s, "device": self.device,
                  "extract_seconds": elapsed, "extract_fps": len(times) / max(elapsed, 1e-6),
                  "motion_features": MOTION_FEATURES, "use_cnn": self.use_cnn},
            diff=np.stack(diffs).astype(np.float32),
            ego=np.stack(egos).astype(np.float32) if egos else None,
        )
        if progress is not None:
            progress(1.0, "done")
        return feats, kept
