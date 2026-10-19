"""그림 설명 VLM 부르기(D19) — 가짜 OpenAI 호환 서버로 요청 모양, 응답 기록, 기록된 그림은 다시 부르지 않음, 실패는 적지 않음."""
import json

import numpy as np
import pytest

from kdp import figure_desc
from kdp.assemble.figures import attach_descriptions, route_figures
from kdp.io.write import write_png
from kdp.schema import Block, Document, Page, ParserInfo, Source
from tests.fixtures.fake_vlm import FakeVLM


def _doc():
    blocks = [Block(id="p1.b01", type="chart", bbox=(70, 100, 520, 400), figure_text="기온(℃)\n23.1",
                    img_path="imgs/p1.b01.png"),
              Block(id="p1.b02", type="image", bbox=(70, 420, 520, 700), figure_text="머리\n촉수", img_path="imgs/p1.b02.png")]
    return route_figures(Document(doc_id="sha256:ab", source=Source(paths=["a.pdf"], pages=1),
                                  parser=ParserInfo(kdp_version="0", engine={}, engine_hash="e", assemble_hash="a", run_at="t"),
                                  pages=[Page(page_no=1, sheet_no=1, size=(595, 842), blocks=blocks)]))


def _settings(settings, url):
    s = settings.model_copy(deep=True)
    s.parse.describe.enabled, s.parse.describe.model = True, "served-name"     # FakeVLM 이 올린 이름
    s.env.describer.server_url = url
    return s


def test_run_records_responses_and_skips_recorded(tmp_path, settings):
    for n in (1, 2):
        write_png(tmp_path / "raw" / "imgs" / f"p1.b0{n}.png", np.full((20, 30, 3), 255, np.uint8))
    fail = [True]

    def respond(query, image):
        if "촉수" in query:
            return (500, "boom", "stop") if fail else (200, "머리 앞으로 촉수가 길게 뻗어 있다.", "stop")
        return 200, "남서쪽이 23.1℃로 가장 높다.", "stop"

    with FakeVLM(respond) as vlm:
        s = _settings(settings, vlm.url)
        new, failed = figure_desc.run(_doc(), tmp_path, s)
        assert new == 1 and len(failed) == 1 and failed[0].startswith("p1.b02: ")   # 실패한 그림은 적지 않는다
        body = next(r["body"] for r in vlm.requests if "23.1" in r["query"])
        assert (body["model"], body["temperature"], body["max_tokens"]) == ("served-name", 0, 512)
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
        rows = [json.loads(x) for x in (tmp_path / "raw" / "describe.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [(r["block_id"], r["model"], r["text"], r["finish_reason"]) for r in rows] == \
            [("p1.b01", "served-name", "남서쪽이 23.1℃로 가장 높다.", "stop")]

        fail.clear()
        n = len(vlm.requests)
        assert figure_desc.run(_doc(), tmp_path, s) == (1, [])                   # 실패했던 그림만 다시
        assert len(vlm.requests) == n + 1 and figure_desc.run(_doc(), tmp_path, s) == (0, [])

    out = attach_descriptions(_doc(), figure_desc.load(tmp_path / "raw"), s.parse.describe)
    assert [b.figure_desc for b in out.pages[0].blocks] == ["남서쪽이 23.1℃로 가장 높다.", "머리 앞으로 촉수가 길게 뻗어 있다."]


def test_run_needs_server_only_when_there_is_work(tmp_path, settings):
    s = _settings(settings, None)
    with pytest.raises(RuntimeError, match="describer.server_url"):
        figure_desc.run(_doc(), tmp_path, s)
    empty = route_figures(_doc().model_copy(update={"pages": []}))
    assert figure_desc.run(empty, tmp_path, s) == (0, [])
