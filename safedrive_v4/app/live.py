"""v4 Live Driving Assistant: camera (opened ONLY when the user presses Start) or a demo video played in real time.

Per 10 Hz step: v3 LiveEngine (frozen v3 perception + prediction, read-only; ego motion switched on for the scene graph only,
which the v3 model ignores) -> v4 scene graph + uncertainty-aware reasoning -> input-quality monitor -> 5-state warning policy
-> evidence trace -> event recording (v3 EventRecorder: 5 s pre / 5 s post clip) into the encrypted v4 event store.
The warning path never waits for drawing or explanations; SHAP is skipped live (no per-step model matrix); traces store the
graph / KG evidence and the v3 probability.
"""
from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np

from safedrive_v4.event_memory.store import EventStore, V4Event, auto_category, new_event_id
from safedrive_v4.uncertainty.input_quality import InputQualityMonitor


def _engine_cls():
    from src.live.engine import LiveEngine

    class V4LiveEngine(LiveEngine):
        """Frozen v3 engine; the only difference is that ego motion is also estimated (input of the v4 scene graph)."""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.needs_ego = True
    return V4LiveEngine


class V4LiveController:
    def __init__(self, store: EventStore | None = None):
        self.store = store or EventStore()
        self.lock = threading.Lock()
        self.state: dict = {"status": "stopped"}
        self.frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._th: threading.Thread | None = None
        self.timeline: list[dict] = []
        self.events: list[V4Event] = []

    @property
    def running(self) -> bool:
        return self._th is not None and self._th.is_alive()

    def start(self, source, model_dir: str = "models/safedrive_v3", trip_name: str = "live", data_kind: str = "live_camera_test"):
        if self.running:
            return
        self._stop.clear()
        self.timeline, self.events = [], []
        self._th = threading.Thread(target=self._run, args=(source, model_dir, trip_name, data_kind), daemon=True, name="v4-live")
        self._th.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._th:
            self._th.join(timeout=30)
        return dict(self.state)

    def snapshot(self) -> tuple[dict, np.ndarray | None]:
        with self.lock:
            return dict(self.state), None if self.frame is None else self.frame.copy()

    def _run(self, source, model_dir, trip_name, data_kind):
        from safedrive_v4.app.core import TripStore
        from safedrive_v4.pipeline import V3ReadOnlyModel, V4Pipeline
        from src.features.ego_motion import EGO_LAYOUT
        from src.features.motion_features import MOTION_FEATURES
        from src.live.recorder import EventRecorder, RecorderConfig
        try:
            self.state = {"status": "starting"}
            eng = _engine_cls()(model_dir)
            pipe = V4Pipeline(V3ReadOnlyModel(model_dir))
            qm = InputQualityMonitor()
            rec = EventRecorder(RecorderConfig(), root=self.store.root / "live_tmp") if self.store.clips_allowed else None
            from src.live.capture import CaptureThread
            cap = CaptureThread(source)   # v3 capture thread: newest frame only, reconnect / end-of-video handling
            cap.start()
            ts = TripStore(self.store)
            trip_id = ts.new_id(trip_name)
            cols = {n: MOTION_FEATURES.index(n) for n in ("flow_mag_mean", "flow_dx_median", "brightness", "mean_det_conf")}
            fps_hist: deque = deque(maxlen=30)
            last_level, active, traces = None, None, []
            rep_tracks, rep_H, rep_scene = [], [], []   # exact replay of the scene graph later (bounded by the session length)
            t_prev = time.time()
            period = 1.0 / eng.fps
            next_t, last_id, last_wall = time.time(), None, None
            while not self._stop.is_set():
                st_ = cap.stats.state
                if st_ == "failed":
                    with self.lock:
                        self.state.update(status=f"camera failed: {cap.stats.error}", level="DEGRADED_INPUT")
                    break
                item = cap.latest()
                if st_ == "ended" and item is not None and item[0] == last_id:
                    break                                             # end of the demo video
                now = time.time()
                if now < next_t:                                      # 10 Hz grid: the models assume 0.1 s steps
                    time.sleep(min(next_t - now, 0.01))
                    continue
                if item is None or item[0] == last_id:
                    q = qm.update(None, now)
                    if last_wall is not None and now - last_wall > 1.0:
                        with self.lock:
                            self.state.update(status="no new frames", level="DEGRADED_INPUT", input_reasons=q.reasons)
                    time.sleep(0.003)
                    continue
                next_t = max(next_t + period, now)
                fid, t_cap, frame = item
                if last_wall is not None and t_cap - last_wall > 1.0:
                    eng.reset()                                       # stream gap: track history no longer valid
                    pipe.reset()
                last_id, last_wall = fid, t_cap
                q = qm.update(frame, t_cap)
                res = eng.process(frame, t_cap)
                rows = res.rows if res.rows is not None else np.zeros((0, 8), np.float32)
                ego = eng.buf["ego"][-1] if eng.buf["ego"] and eng.buf["ego"][-1] is not None else None
                M = None if ego is None else np.asarray(ego[EGO_LAYOUT["H"]], float).reshape(3, 3)
                mot = eng.buf["motion"][-1]
                scene = {n: float(mot[i]) for n, i in cols.items()}
                rep_tracks.append(rows), rep_H.append(np.eye(3).ravel() if M is None else M.ravel()), rep_scene.append([scene[k] for k in cols])
                row, tr, snap, ures = pipe.step(res.step, res.t, rows[:, 1:8], M, scene, None, f"live:{trip_id}", q, True, last_level,
                                                p_external=(float(res.p), float(res.mutual_info)))
                last_level = row["level"]
                if tr is not None:
                    traces.append(tr)
                if rec is not None:
                    rec.push(res.t, res.frame)
                # event episodes (MODERATE / HIGH)
                if row["level"] in ("MODERATE", "HIGH"):
                    if active is None:
                        active = V4Event(new_event_id(), trip_id, f"live:{getattr(source, 'key', 'source')}", "risk_event", res.t, res.t, res.t,
                                         row["level"], row["p"], row["kg_risk_expectation"], row["reliability"], row["cautious"], [], [],
                                         trace_file=tr.event_id if tr else None)
                        if rec is not None:
                            from src.live.recorder import new_event
                            active._rec = rec.trigger(new_event(trip_id, "warning_event", res.t, res, "v3", "balanced"))
                    active.t_end = res.t
                    if row["p"] > active.p_peak:
                        active.p_peak, active.t_peak = row["p"], res.t
                    active.level_peak = "HIGH" if "HIGH" in (active.level_peak, row["level"]) else "MODERATE"
                    active.kg_evidence_peak = max(active.kg_evidence_peak, row["kg_risk_expectation"])
                    active.reliability_min = min(active.reliability_min, row["reliability"])
                    active.cautious = active.cautious or row["cautious"]
                    if tr:
                        active.track_ids = sorted(set(active.track_ids) | set(tr.track_ids))
                        active.explanation += [" -> ".join(p) for p in tr.kg_paths][:3]
                elif active is not None:
                    active.category_auto = auto_category(active.level_peak, active.cautious, active.kg_evidence_peak)
                    self._close(active)
                    active = None
                self.timeline.append(row)
                fps_hist.append(now - t_prev)
                t_prev = now
                with self.lock:
                    self.frame = res.frame
                    self.state = {"status": "running", "t": res.t, "level": row["level"], "cautious": row["cautious"], "p": row["p"],
                                  "reliability": row["reliability"], "kg_evidence": row["kg_risk_expectation"],
                                  "input_reasons": q.reasons, "input_degraded": q.degraded, "fps": 1.0 / max(np.median(fps_hist), 1e-6),
                                  "latency_ms": {**{k: round(v, 1) for k, v in res.timings_ms.items()},
                                                 "v4_graph_reasoning": round(pipe.latency["graph"][-1] + pipe.latency["reasoning"][-1], 2)},
                                  "n_objects": int(res.n_objects), "events": len(self.events), "trip_id": trip_id,
                                  "explanation": [" -> ".join(p) for p in (tr.kg_paths if tr else [])][:2] or self.state.get("explanation", [])}
            if active is not None:
                active.category_auto = auto_category(active.level_peak, active.cautious, active.kg_evidence_peak)
                self._close(active)
            if rec is not None:
                rec.flush()
                self._attach_clips(rec)
            cap.stop()
            meta = {"name": trip_name, "source": f"live:{getattr(source, 'key', 'source')}", "data_kind": data_kind,
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "duration_s": self.timeline[-1]["t"] if self.timeline else 0.0,
                    "model": pipe.model.meta.get("version"), "steps": len(self.timeline), "n_events": len(self.events),
                    "latency": {k: {"mean_ms": float(np.mean(v))} for k, v in pipe.latency.items() if v},
                    "encrypted_store": self.store.encrypted, "counts_as_real_driving": data_kind == "real_driving"}
            d = ts.save(trip_id, meta, self.timeline, traces, np.concatenate(rep_tracks) if rep_tracks else np.zeros((0, 8), np.float32))
            np.savez_compressed(d / "replay.npz", H=np.array(rep_H, np.float32), scene=np.array(rep_scene, np.float32),
                                scene_cols=np.array(list(cols)), times=np.array([r["t"] for r in self.timeline], np.float32))
            with self.lock:
                self.state.update(status="stopped", trip_id=trip_id)
        except Exception as e:  # noqa: BLE001
            with self.lock:
                self.state = {"status": "error", "error": repr(e), "trace": traceback.format_exc()[-2000:]}
            try:
                cap.stop()
            except Exception:  # noqa: BLE001
                pass

    def _close(self, ev: V4Event) -> None:
        rec_ev = getattr(ev, "_rec", None)
        if hasattr(ev, "_rec"):
            delattr(ev, "_rec")
        ev._rec_id = rec_ev.event_id if rec_ev else None
        self.events.append(ev)
        d = {k: v for k, v in ev.__dict__.items() if not k.startswith("_")}
        self.store.save(V4Event(**d))

    def _attach_clips(self, rec) -> None:
        done = {e.event_id: e for e in rec.done}
        for ev in self.events:
            rid = getattr(ev, "_rec_id", None)
            le = done.get(rid)
            stored = self.store.get(ev.event_id)
            if le is None:
                self.store.attach_clip(stored, None, "clip not recorded")
            else:
                stored.clip_offsets = {"pre_start": 0.0, "onset": le.pre_s, "post_end": le.pre_s + le.post_s}
                self.store.attach_clip(stored, Path(le.clip) if le.clip else None, le.clip_status)
