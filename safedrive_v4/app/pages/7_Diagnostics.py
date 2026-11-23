from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd  # noqa: E402
import psutil  # noqa: E402
import streamlit as st  # noqa: E402

from safedrive_v4.app.ui import page, pipeline, stores  # noqa: E402
from safedrive_v4.config import resolve  # noqa: E402

page("Model & System Diagnostics", "🩺")
st.title("🩺 Model & System Diagnostics")
ev, ts = stores()
p = pipeline()
st.subheader("Model provenance")
from src.realworld import provenance  # noqa: E402

c = provenance.compute("models/safedrive_v3")
chk = provenance.check(c)
st.json({"predictor": p.model.meta.get("version"), "source": p.model.meta.get("source"),
         "model_sha256": c["model_sha256"][:16], "feature_sha256": c["feature_config_sha256"][:16],
         "threshold_sha256": c["threshold_config_sha256"][:16], "equals_frozen_reference": chk.get("eligible"),
         "warning thresholds (dev OOF)": p.model.meta.get("thresholds_from_dev_oof"),
         "v4 schemas": {"scene graph": "sg-1.0", "knowledge graph": "kg-1.0", "features": "v4feat-1.0"}})
st.subheader("Benchmark (held-out Nexar, 224 videos)")
ab = resolve("experiments/v4/latest_ablation.json")
if ab.exists():
    r = json.loads(ab.read_text(encoding="utf-8"))
    st.dataframe(pd.DataFrame([{"config": k, "mAP": round(h["window"]["mAP"], 3), "ROC-AUC": round(h["window"]["roc_auc"], 3),
                                "ECE": round(h["window"]["ece"], 3), **{f"recall@{b}": f"{h['streaming'][b]['recall']:.3f} ({h['streaming'][b]['fa_per_hour']:.1f} FA/h)"
                                                                       for b in h["streaming"]}} for k, h in r.get("heldout", {}).items()]),
                 hide_index=True, use_container_width=True)
hy = resolve("experiments/v4/latest_hypotheses.json")
if hy.exists():
    h = json.loads(hy.read_text(encoding="utf-8"))
    st.write({k: h[k]["verdict"] for k in ("H2", "H4") if k in h})
st.caption("Full tables: docs/SAFEDRIVE_V4_EXPERIMENT_PLAN.md")
st.subheader("Latency (last analysed trips, ms per 10 Hz step, excluding perception)")
rows = []
for t in ts.list()[:10]:
    for k, v in (t.get("latency") or {}).items():
        rows.append({"trip": t["trip_id"], "stage": k, "mean_ms": round(v.get("mean_ms", 0), 2)})
if rows:
    st.dataframe(pd.DataFrame(rows).pivot_table(index="trip", columns="stage", values="mean_ms"), use_container_width=True)
st.subheader("Resources")
proc = psutil.Process()
gpu = {}
try:
    import torch
    if torch.cuda.is_available():
        gpu = {"device": torch.cuda.get_device_name(0), "vram_allocated_mb": round(torch.cuda.memory_allocated() / 1e6, 1),
               "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1e6)}
except Exception:  # noqa: BLE001
    pass
st.json({"cpu_percent": psutil.cpu_percent(interval=0.2), "ram_process_mb": round(proc.memory_info().rss / 1e6),
         "ram_system_used_percent": psutil.virtual_memory().percent, **gpu,
         "event_store": str(ev.root), "event_store_encrypted": ev.encrypted})
st.subheader("Input quality (time share by state over stored trips)")
agg: dict = {}
for t in ts.list():
    for k, v in (t.get("levels") or {}).items():
        agg[k] = agg.get(k, 0) + v
if agg:
    tot = sum(agg.values())
    st.write({k: f"{100 * v / tot:.1f} %" for k, v in agg.items()})
