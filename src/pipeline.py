"""SafeDrive AI end-to-end inference: video -> features -> calibrated prediction + uncertainty -> risk."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.features.extractor import FeatureExtractor
from src.prediction.predictor import EnsemblePredictor
from src.preprocessing.feature_cache import VideoFeatures
from src.risk.risk_engine import RiskAssessment, RiskConfig, RiskEngine
from src.tracking.tracker import track_statistics
from src.utils.config import load_config, resolve
from src.utils.device import get_device
from src.utils.logging_utils import get_logger
from src.visualization.overlay import draw_frame, write_video

log = get_logger("pipeline")
V1_MODEL_DIR = resolve("models/safedrive_v1")          # LSTM ensemble + ResNet features (archived baseline)
V2_MODEL_DIR = resolve("models/safedrive_v2_light")    # lightweight tabular model, no CNN
V3_MODEL_DIR = resolve("models/safedrive_v3")          # v2 + vehicle-interaction cues + dense-negative training (default)
DEFAULT_MODEL_DIR = next((d for d in (V3_MODEL_DIR, V2_MODEL_DIR, V1_MODEL_DIR) if (d / "bundle.json").exists()), V1_MODEL_DIR)


def load_predictor(model_dir: str | Path, device: str = "cpu"):
    """Dispatch on bundle.json 'type': 'tabular' -> TabularPredictor, otherwise the v1 neural EnsemblePredictor."""
    from src.prediction.predictor import ModelNotFoundError
    from src.prediction.tabular_predictor import TabularPredictor
    from src.utils.io import load_json
    bj = Path(model_dir) / "bundle.json"
    if not bj.exists():
        raise ModelNotFoundError(f"No trained model bundle at {model_dir}.")
    if load_json(bj).get("type") == "tabular":
        return TabularPredictor.load(model_dir)
    return EnsemblePredictor.load(model_dir, device)


@dataclass
class AnalysisResult:
    video_path: str
    features: VideoFeatures
    timeline: pd.DataFrame
    assessments: list[RiskAssessment]
    alarms: list[float]
    track_stats: dict
    timings: dict
    frames: list[np.ndarray] = field(default_factory=list, repr=False)
    model_dir: str = ""  # bundle that produced this result (explanations must use the same model)
    profile: str = "balanced"  # warning-sensitivity profile used for the alarms
    alarm_threshold: float = float("nan")

    @property
    def max_level(self) -> str:
        order = ["NORMAL", "UNCERTAIN", "CAUTION", "WARNING", "HIGH RISK"]
        return max((a.level for a in self.assessments), key=order.index, default="NORMAL")

    def summary(self) -> dict:
        tl = self.timeline
        return {
            "video": Path(self.video_path).name, "duration_s": float(self.features.times[-1]),
            "frames_analysed": len(self.features), "max_level": self.max_level,
            "max_probability": float(tl.p_mean.max()) if len(tl) else 0.0,
            "max_risk_score": float(tl.risk_score.max()) if len(tl) else 0.0,
            "first_alarm_s": self.alarms[0] if self.alarms else None, "n_alarms": len(self.alarms),
            "level_counts": tl.level.value_counts().to_dict() if len(tl) else {},
            "tracking": self.track_stats, "timings": self.timings,
        }


class SafeDrivePipeline:
    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, cfg: dict | None = None, device: str | None = None):
        self.cfg = cfg or load_config()
        self.device = device or get_device()
        self.model_dir = Path(model_dir)
        self.predictor = load_predictor(model_dir, self.device)
        self.risk = RiskEngine(RiskConfig.from_dict(self.predictor.meta.get("risk", {})))
        # ResNet-18 is only loaded when the selected model needs it (v1); v2-light uses motion + frame difference only
        self.extractor = FeatureExtractor(self.cfg, self.device, use_cnn=getattr(self.predictor, "uses_cnn", True),
                                          use_ego=getattr(self.predictor, "needs_ego", False))

    def analyze(self, video_path: str | Path, progress: Callable[[float, str], None] | None = None,
                should_cancel: Callable[[], bool] | None = None, keep_frames: bool = True,
                max_duration_s: float | None = 120.0, profile: str | None = None) -> AnalysisResult:
        t0 = time.perf_counter()
        feats, frames = self.extractor.extract(video_path, progress=progress, should_cancel=should_cancel,
                                               keep_frames=keep_frames, max_duration_s=max_duration_s)
        t1 = time.perf_counter()
        return self.analyze_features(feats, str(video_path), frames, {"extract_s": t1 - t0}, profile=profile)

    def analyze_features(self, feats: VideoFeatures, video_path: str = "", frames: list | None = None,
                         timings: dict | None = None, profile: str | None = None) -> AnalysisResult:
        t1 = time.perf_counter()
        stride = int(self.cfg["windows"]["inference_stride_steps"])
        pred = self.predictor.predict_video(feats, stride)
        t2 = time.perf_counter()
        motion_rows = feats.motion[pred["step"]]
        ass = self.risk.assess(pred["times"], pred, motion_rows)
        from src.evaluation.early_warning import alarm_onsets
        warm = float(self.predictor.meta.get("warmup_s", self.predictor.window_s))
        use_p = self.risk.cfg.level_signal == "probability"
        alarm_score = np.array([(a.prediction_score if use_p else a.risk_score) if a.level != "UNCERTAIN" and a.timestamp >= warm - 1e-6 else 0.0
                                for a in ass])  # no alarm before one full observation window
        alpha = float(self.predictor.meta.get("alarm_ema_alpha", 1.0))
        if alpha < 1.0:  # causal EMA smoothing of the alarm signal (selected on dev streams when used)
            from src.evaluation.streaming import ema
            alarm_score = ema(alarm_score, alpha)
        profile, thr = self.alarm_threshold(profile)
        alarms = alarm_onsets(pred["times"], alarm_score, thr, self.risk.cfg.consecutive_steps)
        tl = pd.DataFrame({
            "time_s": pred["times"], "step": pred["step"], "p_mean": pred["p_mean"], "p_std": pred["p_std"],
            "entropy": pred["entropy"], "mutual_info": pred["mutual_info"],
            "risk_score": [a.risk_score for a in ass], "trend": [a.trend for a in ass],
            "level": [a.level for a in ass],
            "factors": ["; ".join(f["feature"] for f in a.triggering_factors) for a in ass],
            "explanation": [a.explanation for a in ass],
        })
        n_win = len(tl)
        tim = dict(timings or {})
        tim.update({"predict_s": t2 - t1, "windows": n_win,
                    "latency_ms_per_window": 1000 * (t2 - t1) / max(n_win, 1),
                    "extract_fps": feats.meta.get("extract_fps")})
        return AnalysisResult(video_path, feats, tl, ass, alarms, track_statistics(feats.tracks), tim, frames or [],
                              str(self.model_dir), profile, thr)

    @property
    def profiles(self) -> dict[str, float]:
        """Warning-sensitivity profiles bound to false-alarm budgets measured on dev streams (empty for v1 / v2 bundles)."""
        return dict(self.predictor.meta.get("risk", {}).get("profiles", {}))

    def safety_events(self, res: AnalysisResult, trip_id: str = ""):
        """Near-Miss Memory events for an analysis (alerts + near-miss candidates at the CAUTION threshold)."""
        from src.safety.near_miss import extract_events
        return extract_events(res, float(self.risk.cfg.caution), trip_id, self.risk.cfg.consecutive_steps,
                              float(self.predictor.meta.get("warmup_s", self.predictor.window_s)),
                              "p_mean" if self.risk.cfg.level_signal == "probability" else "risk_score")

    def alarm_threshold(self, profile: str | None) -> tuple[str, float]:
        prof = self.profiles
        if profile and profile in prof:
            return profile, float(prof[profile])
        return "balanced", float(self.risk.cfg.warning)

    def render(self, res: AnalysisResult, out_path: str | Path) -> Path:
        if not res.frames:
            raise ValueError("Frames were not kept; call analyze(keep_frames=True)")
        tl = res.timeline
        steps = tl.step.to_numpy()
        out = []
        for s, frame in enumerate(res.frames):
            rows = res.features.tracks[res.features.tracks[:, 0] == s]
            j = int(np.searchsorted(steps, s, side="right") - 1)
            if j >= 0:
                r = tl.iloc[j]
                out.append(draw_frame(frame, rows, r.level, r.risk_score, r.p_mean, float(res.features.times[s])))
            else:
                out.append(draw_frame(frame, rows, "NORMAL", None, None, float(res.features.times[s])))
        return write_video(out, out_path, res.features.fps)
