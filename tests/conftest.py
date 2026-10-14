import os
import sys
from pathlib import Path

import pytest

from kdp.config import load_settings

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture(scope="session", autouse=True)
def _close_vlm_loop():
    yield
    genai = sys.modules.get("paddlex.inference.models.common.genai")
    if genai:
        genai.close_aio_loop()


@pytest.fixture(scope="session")
def korean_font(settings, tmp_path_factory):
    src = settings.env.test.korean_font
    if not src or not Path(src).exists():
        pytest.skip("한글 글꼴이 없다(configs/env.yaml 의 test.korean_font)")
    from tests.fixtures.fonts import make_test_font

    return str(make_test_font(src, tmp_path_factory.mktemp("font") / "ko.ttf"))
