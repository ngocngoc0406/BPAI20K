"""Live inference engine: processes one frame per 10 Hz grid step with the SAME components and call order as the offline
extractor (src/features/extractor.py) and predicts with the SAME bundles (no retraining, no feature change).

Memory is bounded: per-step arrays live in ring buffers (BUFFER_STEPS); predictions use the last PRED_STEPS steps, which
cover the observation window plus the look-back of the interaction cues (k = 3 steps).
Equivalence with the offline pipeline is checked by tests/test_live.py.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.features.ego_motion import EgoMotionEstimator
from src.features.extractor import FeatureExtractor
from src.features.frame_diff import FrameDiff
from src.features.motion_features import MotionFeatureComputer
from src.features.optical_flow import FlowComputer
from src.pipeline import load_predictor
from src.preprocessing.feature_cache import VideoFeatures
from src.preprocessing.frame_extractor import resize_width
from src.risk.risk_engine import RiskConfig, RiskEngine
from src.utils.config import load_config
from src.utils.device import get_device
from src.visualization.overlay import draw_frame

BUFFER_STEPS = 100   # 10 s of per-step history
PRED_STEPS = 100     # steps handed to the predictor: = BUFFER_STEPS gives exact equivalence with offline (tests/test_live.py)
HIST_PRED = 20       # prediction history for the 1 s risk trend


class AlarmState:
    """Incremental version of src.evaluation.early_warning.alarm_onsets (same semantics)."""

    def __init__(self, k: int = 3):
        self.k, self.run, self.below, self.armed = k, 0, k, True

    def update(self, score: float, thr: float) -> bool:
        if score >= thr:
            self.run += 1
            self.below = 0
            if self.armed and self.run >= self.k:
                self.armed = False
                return True
        else:
            self.run = 0
            self.below += 1
            if self.below >= self.k:
                self.armed = True
        return False


@dataclass
class StepResult:
    step: int
    t: float                      # nominal stream time (s) on the 10 Hz grid
    wall_capture: float           # wall-clock time the frame was captured
    frame: np.ndarray             # annotated frame (BGR)
    n_objects: int
    predicted: bool
    level: str
    p: float
    risk_score: float
    mutual_info: float
    alarm: bool
    assessment: object | None
    timings_ms: dict = field(default_factory=dict)
    rows: np.ndarray | None = None  # tracks of this step [step, id, cls, conf, x1, y1, x2, y2]


class LiveEngine:
    def __init__(self, model_dir: str | Path, profile: str | None = None, device: str | None = None, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.device = device or get_device()
        self.model_dir = str(model_dir)
        self.predictor = load_predictor(model_dir, self.device)
        meta = self.predictor.meta
        self.risk = RiskEngine(RiskConfig.from_dict(meta.get("risk", {})))
        prof = meta.get("risk", {}).get("profiles", {})
        self.profile = profile if profile in prof else "balanced"
        self.alarm_thr = float(prof.get(self.profile, self.risk.cfg.warning))
        self.use_p = self.risk.cfg.level_signal == "probability"
        self.warmup = float(meta.get("warmup_s", self.predictor.window_s))
        self.ema_alpha = float(meta.get("alarm_ema_alpha", 1.0))
        self.uses_cnn = bool(getattr(self.predictor, "uses_cnn", True))
        self.needs_ego = bool(getattr(self.predictor, "needs_ego", False))
        self.fx = FeatureExtractor(self.cfg, self.device, use_cnn=self.uses_cnn, use_ego=self.needs_ego)
        self.fps = float(self.cfg["preprocessing"]["target_fps"])
        self.width = int(self.cfg["preprocessing"]["detect_width"])
        self.stride = int(self.cfg["windows"]["inference_stride_steps"])
        self.reset()

    def reset(self) -> None:
        self.fx.tracker.reset()
        self.motion = MotionFeatureComputer(self.fps, int(self.cfg["tracking"]["history_steps"]))
        self.flow = FlowComputer(self.fx.flow_size)
        self.fdiff = FrameDiff(self.fx.flow_size)
        self.egom = EgoMotionEstimator()
        self.step = 0
        self.buf = {k: deque(maxlen=BUFFER_STEPS) for k in ("t", "motion", "diff", "ego", "visual", "tracks")}
        self.hist = {k: deque(maxlen=HIST_PRED) for k in ("t", "p_mean", "p_std", "entropy", "aleatoric", "mutual_info", "motion")}
        self.alarm = AlarmState(self.risk.cfg.consecutive_steps)
        self.ema_val: float | None = None
        self.last = {"level": "NORMAL", "p": 0.0, "risk_score": 0.0, "mi": 0.0, "assessment": None}

    # ---------- one grid step ----------
    def process(self, frame: np.ndarray, wall_capture: float | None = None) -> StepResult:
        t0 = time.perf_counter()
        dt = 1.0 / self.fps
        t = self.step / self.fps
        frame = resize_width(frame, self.width)
        h, w = frame.shape[:2]
        dets = self.fx.tracker.update(frame, self.step)
        boxes = np.array([d.xyxy for d in dets], np.float32).reshape(-1, 4) / np.array([w, h, w, h], np.float32)
        ids = np.array([d.track_id for d in dets], int)
        cls = np.array([d.cls for d in dets], int)
        conf = np.array([d.conf for d in dets], np.float32)
        t_det = time.perf_counter()
        fl = self.flow(frame, dt)
        diff = self.fdiff(frame)
        ego = self.egom(frame, boxes, dt, self.flow.last_field, self.fx.flow_size) if self.needs_ego else None
        mot = self.motion.update(self.step, boxes, ids, cls, conf, fl, float(frame.mean() / 255.0))
        vis = self.fx.encoder.encode([frame])[0] if self.uses_cnn else np.zeros(0, np.float32)
        rows = np.array([[self.step, ids[i], cls[i], conf[i], *boxes[i]] for i in range(len(dets))], np.float32).reshape(-1, 8)
        for k, v in (("t", t), ("motion", mot), ("diff", diff), ("ego", ego), ("visual", vis), ("tracks", rows)):
            self.buf[k].append(v)
        t_feat = time.perf_counter()
        predicted = self.step % self.stride == 0
        alarm = False
        if predicted:
            alarm = self._predict(t, mot)
        t_pred = time.perf_counter()
        lv = self.last
        annotated = draw_frame(frame, rows, lv["level"], lv["risk_score"], lv["p"], t)
        res = StepResult(self.step, t, wall_capture or time.time(), annotated, len(dets), predicted, lv["level"], lv["p"],
                         lv["risk_score"], lv["mi"], alarm, lv["assessment"],
                         {"detect_track": 1000 * (t_det - t0), "features": 1000 * (t_feat - t_det), "predict": 1000 * (t_pred - t_feat),
                          "render": 1000 * (time.perf_counter() - t_pred), "total": 1000 * (time.perf_counter() - t0)}, rows)
        self.step += 1
        return res

    def _buffer_features(self) -> VideoFeatures:
        n = min(PRED_STEPS, len(self.buf["t"]))
        sl = lambda k: list(self.buf[k])[-n:]  # noqa: E731
        times = np.array(sl("t"), np.float32)
        first_step = self.step - n + 1
        tr = [r for r in sl("tracks") if len(r)]
        tracks = np.concatenate(tr).copy() if tr else np.zeros((0, 8), np.float32)
        if len(tracks):
            tracks[:, 0] -= first_step  # re-index steps to the buffer
        ego = np.stack(sl("ego")) if self.needs_ego else None
        visual = np.stack(sl("visual")).astype(np.float16) if self.uses_cnn else np.zeros((n, 0), np.float16)
        return VideoFeatures("live", times, visual, np.stack(sl("motion")), tracks, {"target_fps": self.fps}, np.stack(sl("diff")), ego)

    def _predict(self, t: float, mot: np.ndarray) -> bool:
        feat = self._buffer_features()
        out = self.predictor.predict_at(feat, np.array([t]))
        for k in ("p_mean", "p_std", "entropy", "aleatoric", "mutual_info"):
            self.hist[k].append(float(np.asarray(out[k])[0]))
        self.hist["t"].append(t)
        self.hist["motion"].append(mot)
        hp = {k: np.array(self.hist[k]) for k in ("p_mean", "p_std", "entropy", "aleatoric", "mutual_info")}
        ass = self.risk.assess(np.array(self.hist["t"]), hp, np.stack(self.hist["motion"]))[-1]
        sig = ass.prediction_score if self.use_p else ass.risk_score
        if ass.level == "UNCERTAIN" or t < self.warmup - 1e-6:
            sig = 0.0
        if self.ema_alpha < 1.0:
            self.ema_val = sig if self.ema_val is None else self.ema_alpha * sig + (1 - self.ema_alpha) * self.ema_val
            sig = self.ema_val
        alarm = self.alarm.update(sig, self.alarm_thr)
        self.last = {"level": ass.level, "p": ass.prediction_score, "risk_score": ass.risk_score, "mi": ass.uncertainty, "assessment": ass}
        return alarm

    def caution_threshold(self) -> float:
        return float(self.risk.cfg.caution)

    def high_threshold(self) -> float:
        return float(self.risk.cfg.high)
