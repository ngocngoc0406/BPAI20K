"""Download a balanced Nexar subset (gated dataset — requires HF login + accepted conditions).

Usage:  python scripts/download_nexar.py [--per-class 200]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from src.data.downloader import AccessError, download_subset  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.io import save_json  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

log = get_logger("download", resolve("experiments/logs/download.log"))


def main() -> int:
    load_dotenv(resolve(".env"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--per-class", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    per = a.per_class or cfg["dataset"]["subset_per_class"]
    out = resolve(cfg["paths"]["raw_dir"])
    try:
        got = download_subset(cfg["dataset"]["hf_repo"], out, per, cfg["seed"])
    except AccessError as e:
        log.error("%s", e)
        log.error("Steps: 1) open https://huggingface.co/datasets/%s and accept the conditions; "
                  "2) run `.venv\\Scripts\\huggingface-cli login` or put HF_TOKEN=... in .env", cfg["dataset"]["hf_repo"])
        return 2
    save_json(got, resolve("data/raw/download_manifest.json"))
    log.info("metadata files: %s", got["metadata"])
    log.info("videos: %d", len(got["videos"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
