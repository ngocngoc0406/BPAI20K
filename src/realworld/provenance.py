"""Provenance for every recorded session (v3.4): SHA-256 of the model bundle, the feature configuration, the threshold /
alert / calibration configuration and the software, compared with the FROZEN reference recorded at the start of the campaign.

A session whose model / feature / threshold hash differs from the frozen reference is NOT eligible for the campaign's
evaluation sets (the campaign evaluates one frozen system). Software hash changes are allowed but must have an entry in
docs/SAFEDRIVE_SOFTWARE_CHANGELOG.md.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.utils.config import load_config, resolve

FROZEN_PATH = resolve("configs/frozen_v3_hashes.json")
SOFTWARE_DIRS = ("src", "pages", "ui")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def sha256_tree(root: Path, patterns=("*",)) -> str:
    root = Path(root)
    h = hashlib.sha256()
    files = sorted({f for p in patterns for f in root.rglob(p) if f.is_file() and "__pycache__" not in f.parts})
    for f in files:
        h.update(f.relative_to(root).as_posix().encode())
        h.update(sha256_file(f).encode())
    return h.hexdigest()


def compute(model_dir: str | Path) -> dict:
    from src.features.window_features import FEATURE_VERSION
    model_dir = resolve(str(model_dir))
    b = json.loads((model_dir / "bundle.json").read_text(encoding="utf-8"))
    cfg = load_config(None)
    weights = resolve(cfg["detection"]["weights"])
    feature_cfg = {"feature_version": FEATURE_VERSION, "groups": b.get("groups"), "window_s": b.get("window_s"), "fps": b.get("fps"),
                   "feature_names": b.get("feature_names"), "motion_features": b.get("motion_features"),
                   "preprocessing": cfg.get("preprocessing"), "detection": cfg.get("detection"), "tracking": cfg.get("tracking"),
                   "detector_weights_sha256": sha256_file(weights) if weights.exists() else "missing"}
    threshold_cfg = {"risk": b.get("risk"), "calibration": b.get("calibration"), "alarm_ema_alpha": b.get("alarm_ema_alpha"),
                     "warmup_s": b.get("warmup_s"), "threshold_f1_val": b.get("threshold_f1_val"), "config_risk": cfg.get("risk")}
    root = resolve(".")
    return {"model_dir": model_dir.name,
            "model_sha256": sha256_tree(model_dir),
            "feature_config_sha256": sha256_json(feature_cfg),
            "threshold_config_sha256": sha256_json(threshold_cfg),
            "software_sha256": sha256_json({d: sha256_tree(root / d, ("*.py",)) for d in SOFTWARE_DIRS}),
            "app_sha256": sha256_file(root / "app.py") if (root / "app.py").exists() else None}


def frozen(path: Path = FROZEN_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}


def check(prov: dict, ref: dict | None = None) -> dict:
    """Compare a session's provenance with the frozen reference for its model."""
    ref = (ref if ref is not None else frozen()).get(prov.get("model_dir", ""), {})
    if not ref:
        return {"frozen_reference": False, "model_ok": None, "features_ok": None, "thresholds_ok": None, "software_same": None,
                "eligible": False}
    out = {"frozen_reference": True, "model_ok": prov["model_sha256"] == ref["model_sha256"],
           "features_ok": prov["feature_config_sha256"] == ref["feature_config_sha256"],
           "thresholds_ok": prov["threshold_config_sha256"] == ref["threshold_config_sha256"],
           "software_same": prov["software_sha256"] == ref["software_sha256"]}
    out["eligible"] = bool(out["model_ok"] and out["features_ok"] and out["thresholds_ok"])
    return out


def save_session(session_dir: Path, model_dir: str | Path, profile: str) -> dict:
    p = {**compute(model_dir), "profile": profile}
    p["check"] = check(p)
    Path(session_dir).mkdir(parents=True, exist_ok=True)
    (Path(session_dir) / "provenance.json").write_text(json.dumps(p, indent=2), encoding="utf-8")
    return p


def load_session(session_dir: Path) -> dict | None:
    f = Path(session_dir) / "provenance.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def freeze(model_dirs=("models/safedrive_v3", "models/safedrive_v3_stabilized"), path: Path = FROZEN_PATH, force: bool = False) -> dict:
    """Write the frozen reference once (refuses to overwrite unless force=True)."""
    if Path(path).exists() and not force:
        raise FileExistsError(f"{path} exists; the frozen reference is not overwritten")
    from datetime import datetime
    ref = {Path(m).name: {**compute(m), "frozen_at": datetime.now().isoformat(timespec="seconds")} for m in model_dirs}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(ref, indent=2), encoding="utf-8")
    return ref
