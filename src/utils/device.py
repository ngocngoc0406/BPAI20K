"""Device selection with CPU fallback and environment reporting."""
from __future__ import annotations

import os
import platform
import shutil
from typing import Any

import psutil


def get_device(prefer: str | None = None) -> str:
    """Return 'cuda' if usable (and not disabled via SAFEDRIVE_DEVICE=cpu), else 'cpu'."""
    want = (prefer or os.environ.get("SAFEDRIVE_DEVICE", "auto")).lower()
    if want == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")  # verify the runtime really works with this driver
            return "cuda"
    except Exception:  # noqa: BLE001 — any CUDA init failure falls back to CPU
        pass
    return "cpu"


def environment_report() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "disk_free_gb": round(shutil.disk_usage(".").free / 1e9, 1),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["vram_gb"] = round(p.total_memory / 1e9, 2)
            info["cuda_runtime"] = torch.version.cuda
    except ImportError:
        info["torch"] = None
    return info


def is_oom(err: BaseException) -> bool:
    return "out of memory" in str(err).lower()
