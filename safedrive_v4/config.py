"""Versioned v4 configuration (JSON). Every artefact records config_version + sha256 of the config."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "v4_default.json"


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    cfg = json.loads(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    for sect, vals in (overrides or {}).items():
        if isinstance(vals, dict):
            cfg.setdefault(sect, {}).update(vals)
        else:
            cfg[sect] = vals
    return cfg


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def with_overrides(cfg: dict, **sections) -> dict:
    out = copy.deepcopy(cfg)
    for s, v in sections.items():
        out.setdefault(s, {}).update(v)
    return out


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
