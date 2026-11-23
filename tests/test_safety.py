"""Near-Miss Memory, trip report, warning-sensitivity profiles, feature-version guard."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src.risk.risk_engine import RiskAssessment
from src.safety.near_miss import NearMissMemory, extract_events
from src.safety.trip_report import TripLog, build_report, report_pdf


class _Feat:
    def __init__(self, T):
        self.times = np.arange(T) / 10.0
        self.tracks = np.array([[s, 1, 2, 0.9, 0.4, 0.5, 0.6, 0.7] for s in range(T)], np.float32)
        self.fps = 10.0
        self.meta = {}


class _Res:
    def __init__(self, p, alarms, video="00001.mp4"):
        T = len(p)
        self.features = _Feat(T * 2)
        steps = np.arange(T) * 2
        self.timeline = pd.DataFrame({"time_s": steps / 10.0, "step": steps, "p_mean": p, "risk_score": p, "mutual_info": 0.01,
                                      "level": ["WARNING" if x >= 0.9 else "CAUTION" if x >= 0.7 else "NORMAL" for x in p]})
        self.assessments = [RiskAssessment(float(t), "NORMAL", float(x), float(x), 0.0, 0.0, 0.0,
                                           [{"feature": "looming_center", "value": 1.0, "typical": 0.0, "strength": 2.0,
                                             "description": "growth"}], f"P={x:.2f}") for t, x in zip(steps / 10.0, p)]
        self.alarms = alarms
        self.video_path, self.model_dir, self.profile, self.frames = video, "models/safedrive_v3", "balanced", []

    def summary(self):
        return {"video": self.video_path, "duration_s": float(self.features.times[-1]), "n_alarms": len(self.alarms), "max_level": "WARNING"}


def test_extract_events_alert_and_candidate():
    p = np.zeros(60)
    p[10:14] = 0.95     # alarm episode
    p[30:34] = 0.75     # near-miss candidate (>= caution, no alarm)
    p[45:46] = 0.8      # too short (1 step) -> ignored
    res = _Res(p, alarms=[2.0 + 0.4])
    ev = extract_events(res, caution=0.7, trip_id="t", k=3, warmup=1.0)
    assert [e.type for e in ev] == ["alert", "near_miss_candidate"]
    assert ev[0].p_peak == pytest.approx(0.95) and ev[0].lead_object["class"] == "car"
    assert ev[1].t_start == pytest.approx(6.0)
    # warm-up: nothing before 1 s
    p2 = np.zeros(20)
    p2[0:4] = 0.9
    assert extract_events(_Res(p2, []), 0.7, warmup=1.0) == []


def test_memory_store_and_trip_report(tmp_path):
    mem = NearMissMemory(tmp_path)
    p = np.zeros(60)
    p[30:34] = 0.75
    res = _Res(p, [])
    mem.add(extract_events(res, 0.7, "trip-a"), res, save_clips=False)
    TripLog(tmp_path).add_clip("trip-a", res)
    assert mem.trips() == ["trip-a"] and len(mem.by_trip("trip-a")) == 1
    rep = build_report(TripLog(tmp_path).get("trip-a"), mem.by_trip("trip-a"))
    assert rep["near_miss_candidates"] == 1 and rep["alerts"] == 0 and "safety score" not in json.dumps(rep).lower()
    assert report_pdf(rep)[:4] == b"%PDF"
    assert mem.delete(mem.by_trip("trip-a")[0]["event_id"]) and mem.by_trip("trip-a") == []


def test_warning_profiles_order_and_pipeline(tmp_path):
    from src.pipeline import SafeDrivePipeline
    from src.utils.config import resolve
    pipe = SafeDrivePipeline(resolve("models/safedrive_v3"), device="cpu")
    prof = pipe.profiles
    assert prof["conservative"] >= prof["balanced"] >= prof["sensitive"]
    assert pipe.alarm_threshold("sensitive") == ("sensitive", prof["sensitive"])
    assert pipe.alarm_threshold(None)[0] == "balanced"


def test_feature_version_guard(tmp_path):
    from src.prediction.tabular_predictor import TabularPredictor
    from src.utils.config import resolve
    dst = tmp_path / "b"
    shutil.copytree(resolve("models/safedrive_v3"), dst)
    meta = json.loads((dst / "bundle.json").read_text(encoding="utf-8"))
    meta["feature_version"] = 999
    (dst / "bundle.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError):
        TabularPredictor.load(dst)
    assert TabularPredictor.load(resolve("models/safedrive_v3")).meta["feature_version"] == 1
