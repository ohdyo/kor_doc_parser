"""설치본 PaddleX 계약 — 가짜 VLM 서버·가짜 검출기로 훅 A·B, 오류 격리, 배열 입력(큐 켬)을 확인한다.
paddlex 버전을 올리면 이 테스트부터 깨진다."""
from types import SimpleNamespace

import numpy as np
import pytest

paddlex = pytest.importorskip("paddlex")

from paddlex.inference.common.batch_sampler import ImageBatchSampler  # noqa: E402
from paddlex.inference.common.reader import ReadImage  # noqa: E402
from paddlex.inference.pipelines.components import CropByBoxes  # noqa: E402

from kdp.engine.hooks import KdpPipeline, install_response_hook  # noqa: E402
from tests.fixtures.fake_vlm import FakeVLM  # noqa: E402


class FakeLayout:
    """PP-DocLayoutV3 대신 정해 둔 상자를 낸다. 쪽 번호는 픽셀 (0,0) 에 칠해 둔다."""
    batch_sampler = SimpleNamespace(batch_size=8)

    def __init__(self, boxes_for):
        self.boxes_for = boxes_for

    def __call__(self, images, **kw):
        for img in images:
            yield {"input_path": None, "page_index": None, "boxes": self.boxes_for(int(img[0, 0, 0]))}


def _box(label, coord, cls_id=22, order=None):
    x1, y1, x2, y2 = coord
    return {"cls_id": cls_id, "label": label, "score": 0.9, "coordinate": list(coord), "order": order,
            "polygon_points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]}


def _pages(n, paint=()):
    out = []
    for k in range(n):
        a = np.full((1700, 1200, 3), 255, np.uint8)
        a[0, 0] = k
        for x1, y1, x2, y2 in paint:
            a[y1:y2, x1:x2] = 0
        out.append(a)
    return out


def _run(respond, boxes_for, pages):
    with FakeVLM(respond) as vlm:
        pred = paddlex.create_predictor(
            model_name="PaddleOCR-VL-1.6-0.9B", batch_size=-1,
            genai_config={"backend": "vllm-server", "server_url": vlm.url,
                          "client_kwargs": {"model_name": "served-name", "max_retries": 0}})
        assert install_response_hook(pred)
        p = KdpPipeline(config={}, initial_predictor=False)
        p.use_doc_preprocessor, p.use_layout_detection = False, True
        p.layout_det_model, p.vl_rec_model = FakeLayout(boxes_for), pred
        p.use_chart_recognition = p.use_seal_recognition = p.use_ocr_for_image_block = False
        p.format_block_content = False
        p.batch_sampler, p.img_reader, p.crop_by_boxes = ImageBatchSampler(batch_size=64), ReadImage("BGR"), CropByBoxes()
        p.use_queues, p.merge_layout_blocks, p.layout_prep_cpu_workers = True, True, 0
        p.markdown_ignore_labels = []
        try:
            return list(p.predict(pages))
        finally:
            pred.close()


def _one_text(k):
    return [_box("text", (50, 50, 50 + 500 + 111 * k, 150), order=1)]


def test_array_input_order_and_hooks():
    out = _run(lambda q, im: (200, f"글자 {im.size[0]}", "stop"), _one_text, _pages(4))
    assert len(out) == 4
    for k, r in enumerate(out):
        assert r["page_index"] is None
        b = r["parsing_res_list"][0]
        assert b.content == f"글자 {500 + 111 * k}"           # 결과는 입력 순서대로
        assert b.kdp["finish_reason"] == "stop" and b.kdp["usage"]["completion_tokens"] == 5
        assert b.kdp["raw"] == b.content and b.kdp["repetition_truncated"] is False


def test_length_and_repetition_flags():
    def respond(q, im):
        return (200, "가나다라", "length") if im.size[0] == 500 else (200, "반복" * 40, "stop")

    boxes = lambda k: [_box("text", (50, 50, 550, 150), order=1), _box("text", (50, 400, 650, 500), order=2)]
    a, b = _run(respond, boxes, _pages(1))[0]["parsing_res_list"]
    assert a.kdp["finish_reason"] == "length" and not a.kdp["repetition_truncated"]
    assert b.kdp["repetition_truncated"] and len(b.content) < len(b.kdp["raw"])


def test_block_error_is_isolated():
    def respond(q, im):
        return (400, "too long", "stop") if im.size[0] == 611 else (200, "정상", "stop")

    out = _run(respond, _one_text, _pages(4))
    assert [r["parsing_res_list"][0].content for r in out] == ["정상", "", "정상", "정상"]
    assert out[1]["parsing_res_list"][0].kdp["error"].startswith("400")


def test_server_error_stops_document():
    def respond(q, im):
        return (500, "boom", "stop") if im.size[0] == 611 else (200, "정상", "stop")

    with pytest.raises(Exception):
        _run(respond, _one_text, _pages(4))


def test_merge_group_and_table_figures():
    figs = [(150, 150, 400, 300), (500, 150, 900, 300)]

    def boxes(k):
        return [
            _box("table", (100, 100, 1000, 500), cls_id=21),
            _box("image", figs[0], cls_id=14), _box("image", figs[1], cls_id=14),       # 표 안 그림
            _box("text", (100, 600, 700, 700), order=1),                                  # A
            _box("text", (100, 705, 600, 800), order=2),                                  # B — A 와 병합
            _box("image", (650, 650, 1000, 780), cls_id=14),
            _box("text", (100, 900, 1000, 1000), order=3),
        ]

    def respond(q, im):
        return (200, "<fcel>[F2]<fcel>[F3]<nl>", "stop") if q.startswith("Table") else (200, "글", "stop")

    res = _run(respond, boxes, _pages(1, paint=figs + [(650, 650, 1000, 780)]))[0]
    blocks = res["parsing_res_list"]
    assert [b.label for b in blocks] == ["table", "text", "text", "image", "text"]   # 표 안 그림 2개는 빠진다
    assert '<img src="imgs/img_in_image_box_' in blocks[0].content
    assert blocks[2].content == "" and blocks[2].kdp_merged_into == 1 and blocks[2].kdp is None
    assert blocks[1].kdp["raw"] == "글" and blocks[4].kdp_merged_into is None
    assert blocks[3].kdp is None                                                    # 그림은 VLM 호출 없음
