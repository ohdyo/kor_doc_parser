"""훅 C — 로고·PDF 글자가 덮는 글 블록은 VLM 에 보내지 않는다(쪽마다 검증 표본 3개는 보낸다). 실제 PaddleX 경로로."""
import json

import numpy as np
import pytest

pytest.importorskip("paddlex")

from kdp.engine.base import PageInput  # noqa: E402
from kdp.engine.paddle_vl import PaddleVLEngine, _wrapper_class, wrapper_kwargs  # noqa: E402
from kdp.io.write import write_json_gz  # noqa: E402
from tests.contract.test_hooks import FakeLayout, _box  # noqa: E402
from tests.fixtures.fake_vlm import FakeVLM  # noqa: E402

H_PT = 850.0
TEXTS = {(100, 200, 1100, 260): 30, (100, 300, 1100, 360): 20, (100, 400, 1100, 460): 10, (100, 500, 1100, 560): 25,
         (100, 40, 1100, 80): 15}                                    # 블록(px) → 레이어 글자 수. 마지막은 머리말


def _boxes(_):
    out = [_box("text", b, order=i + 1) for i, b in enumerate(list(TEXTS)[:4])]
    return out + [_box("header", list(TEXTS)[4], cls_id=12), _box("footer_image", (900, 1600, 1100, 1650), cls_id=9)]


def _layer():
    """블록마다 가로로 늘어선 글자(사용자 공간, pt = px / 2)."""
    chars = []
    for (x0, y0, x1, y1), n in TEXTS.items():
        l, r, top, bot = x0 / 2 + 2, x1 / 2 - 2, y0 / 2 + 2, y1 / 2 - 2
        w = (r - l) / n
        for k in range(n):
            chars.append({"c": "가", "box": [l + k * w, H_PT - bot, l + (k + 1) * w - 1, H_PT - top], "gen": False})
    return chars


def test_skip_text_layer_blocks_and_logos(tmp_path, settings):
    page = np.full((1700, 1200, 3), 255, np.uint8)
    page[1600:1650, 900:1100] = 30
    write_json_gz(tmp_path / "textlayer" / "p0001.json.gz", _layer())
    with FakeVLM(lambda q, im: (200, "가" * 10, "stop")) as vlm:
        s = settings.model_copy(deep=True)
        s.env.vlm.backend, s.env.vlm.server_url, s.env.vlm.api_model_name = "vllm-server", vlm.url, "served-name"
        s.parse.paddle_vl.use_ocr_for_image_block = True
        s.parse.paddle_vl.vlm_skip = ["logo", "text_layer"]
        s.parse.text_layer.mode = "prefer"
        kw = wrapper_kwargs(s)
        kw["use_layout_detection"] = False
        ocr = _wrapper_class()(**kw)
        ocr.paddlex_pipeline.layout_det_model = FakeLayout(_boxes)
        ocr.paddlex_pipeline.use_layout_detection = True
        eng = PaddleVLEngine(s, ocr=ocr)
        try:
            inp = PageInput(page_no=1, sheet_no=1, scale=2.0, sheet_size=(600.0, H_PT), user_bbox=(0, 0, 600, H_PT))
            raw = list(eng.run([(page, inp)], tmp_path))[0]
        finally:
            eng.close()
        assert eng.describe()["vlm_skip"] == ["logo", "text_layer"]

    assert len(vlm.requests) == 3                                    # 검증 표본 3개만(글자 30·25·20 블록)
    side = json.loads((tmp_path / "p0001.sidecar.json").read_text())
    why = {tuple(round(v) for v in raw.blocks[m["idx"]].bbox): (m["vlm"] or {}).get("skipped") for m in side["blocks"]}
    assert why[(100, 200, 1100, 260)] is None and why[(100, 500, 1100, 560)] is None and why[(100, 300, 1100, 360)] is None
    assert why[(100, 400, 1100, 460)] == why[(100, 40, 1100, 80)] == "text_layer"
    assert why[(900, 1600, 1100, 1650)] == "logo"
    assert ocr.paddlex_pipeline.kdp_plans == {}                       # 묶음이 끝나면 비운다


def test_no_skip_without_prefer(tmp_path, settings):
    page = np.full((1700, 1200, 3), 255, np.uint8)
    write_json_gz(tmp_path / "textlayer" / "p0001.json.gz", _layer())
    with FakeVLM(lambda q, im: (200, "가", "stop")) as vlm:
        s = settings.model_copy(deep=True)
        s.env.vlm.backend, s.env.vlm.server_url, s.env.vlm.api_model_name = "vllm-server", vlm.url, "served-name"
        s.parse.text_layer.mode = "off"
        kw = wrapper_kwargs(s)
        kw["use_layout_detection"] = False
        ocr = _wrapper_class()(**kw)
        ocr.paddlex_pipeline.layout_det_model = FakeLayout(lambda _: _boxes(_)[:5])
        ocr.paddlex_pipeline.use_layout_detection = True
        eng = PaddleVLEngine(s, ocr=ocr)
        try:
            inp = PageInput(page_no=1, sheet_no=1, scale=2.0, sheet_size=(600.0, H_PT), user_bbox=(0, 0, 600, H_PT))
            list(eng.run([(page, inp)], tmp_path))
        finally:
            eng.close()
        assert eng.describe()["vlm_skip"] == ["logo"]
    assert len(vlm.requests) == 5                                    # 레이어 우선이 꺼지면 글 블록은 모두 VLM
