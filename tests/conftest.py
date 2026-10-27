import os
import sys
from pathlib import Path

import pytest

from kdp.config import load_settings

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


@pytest.fixture(scope="session")
def settings():
    s = load_settings()
    s.parse.describe.enabled = False      # 그림 설명(D19)은 켜는 시험만 가짜 VLM 으로 — 기본 설정이 켜도 다른 시험이 설명 서버를 부르지 않게
    s.parse.figure_ocr.enabled = False    # 그림 다시 읽기(D23)도 같다
    return s


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
