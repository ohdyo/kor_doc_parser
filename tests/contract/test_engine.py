"""래퍼(KdpOCRVL)·predict_iter·_save 까지 실제 경로로 — 검출기만 가짜라 가중치가 필요 없다."""
import numpy as np
import pytest

pytest.importorskip("paddlex")

from kdp.engine.base import PageInput  # noqa: E402
from kdp.engine.paddle_vl import PaddleVLEngine, _wrapper_class, wrapper_kwargs  # noqa: E402
from tests.contract.test_hooks import FakeLayout, _box  # noqa: E402
from tests.fixtures.fake_vlm import FakeVLM  # noqa: E402

FIG = (150, 150, 400, 300)


def _boxes(k):
    return [_box("table", (100, 100, 1000, 500), cls_id=21), _box("image", FIG, cls_id=14),
            _box("chart", (100, 600, 600, 900), cls_id=3), _box("text", (100, 950, 900, 1000), order=1)]


def _respond(q, im):
    if q.startswith("Table"):
        return 200, "<fcel>가<fcel>나<nl>", "stop"          # 그림 토큰을 돌려주지 않는다
    if q.startswith("Chart"):
        return 200, "분기 | 매출\n1분기 | 120", "stop"
    return 200, "글", "stop"


@pytest.mark.parametrize("d14", [True, False])
def test_engine_save(tmp_path, settings, d14):
    page = np.full((1700, 1200, 3), 255, np.uint8)
    page[150:300, 150:400] = 0
    page[600:900, 100:600] = 40
    with FakeVLM(_respond) as vlm:
        s = settings.model_copy(deep=True)
        s.env.vlm.backend, s.env.vlm.server_url, s.env.vlm.api_model_name = "vllm-server", vlm.url, "served-name"
        s.env.vlm.api_key = "sk-SECRET"
        s.parse.paddle_vl.use_ocr_for_image_block = s.parse.paddle_vl.use_chart_recognition = d14
        kw = wrapper_kwargs(s)
        kw["use_layout_detection"] = False                     # 검출기 가중치 없이 만든다
        ocr = _wrapper_class()(**kw)
        ocr.paddlex_pipeline.layout_det_model = FakeLayout(_boxes)
        ocr.paddlex_pipeline.use_layout_detection = True
        eng = PaddleVLEngine(s, ocr=ocr)
        try:
            inp = PageInput(page_no=1, sheet_no=1, scale=2.0, sheet_size=(600.0, 850.0))
            raw = list(eng.run([(page, inp)], tmp_path))[0]
        finally:
            eng.close()

    assert [b.label for b in raw.blocks] == ["table", "chart", "text"]
    assert (tmp_path / "p0001.json").exists() and (tmp_path / "p0001.sidecar.json").exists()
    chart = raw.blocks[1]
    assert chart.img and (tmp_path / "imgs" / chart.img).exists()          # 차트 인식을 켜도 크롭은 우리가 남긴다
    assert bool(chart.content) == d14
    fig = next(iter(raw.table_imgs.values()))
    assert fig["in_html"] is False and (tmp_path / "imgs" / fig["name"]).exists()   # HTML 에 안 나온 표 안 그림도 남긴다

    snap = eng.config_snapshot()
    assert "sk-SECRET" not in snap and vlm.url.split("//")[1] not in snap and "max_concurrency" not in snap
    desc = eng.describe()
    assert desc["response_hook"] is True and desc["weights"]["served"] == {"root": None, "max_model_len": None}
    assert desc["pipeline"] == "PaddleOCR-VL-1.6"
