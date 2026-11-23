"""Dashboard smoke tests: every page must render without exceptions (with or without a trained model)."""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["app.py"] + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "pages").glob("*.py"))


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.fixture(scope="module")
def real_result(zoom_video):
    if not (ROOT / "models" / "safedrive_v1" / "bundle.json").exists():
        pytest.skip("no trained bundle")
    from src.pipeline import SafeDrivePipeline
    pipe = SafeDrivePipeline()
    return pipe.analyze(zoom_video)


@pytest.mark.parametrize("page", PAGES)
def test_page_with_analysis(page, real_result):
    """End-to-end: real trained bundle -> analysis -> every page renders it (incl. Grad-CAM / IG)."""
    at = AppTest.from_file(str(ROOT / page), default_timeout=300)
    at.session_state["result"] = real_result
    at.run()
    assert not at.exception, [e.value for e in at.exception]
