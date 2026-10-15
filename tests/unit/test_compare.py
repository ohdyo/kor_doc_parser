"""비교군 시험 지표(eval/compare.py) — 표 격자, 조각 찾기, 지표 계산."""
import importlib.util
from pathlib import Path

import pytest

from kdp.io.write import write_json
from kdp.schema import Block, Document, Page, ParserInfo, Source

_spec = importlib.util.spec_from_file_location("kdp_compare", Path(__file__).parents[2] / "eval" / "compare.py")
cmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp)


def _box(x):
    return [x, 0, x + 10, 10]


def test_html_grid_spans():
    html = '<table><tr><td rowspan="2">a</td><td colspan="2">b</td></tr><tr><td>c</td><td>d</td></tr></table>'
    assert cmp.html_grid(html) == [2, 3]
    assert cmp.html_grid("<p>x</p>") is None
    assert cmp.th_rows("<table><tr><th>a</th><th>b</th></tr><tr><th>c</th><td>d</td></tr></table>") == 1


def test_present_fuzzy():
    text = cmp.norm("가나다라마바사아자차카타파하 끝")
    assert cmp.present(cmp.norm("다라 마바"), text)
    assert not cmp.present("다라X바", text)                    # 10자 미만은 정확히
    assert cmp.present("가나다라마바사아자X카타", text)         # 12자 중 1자 틀림
    assert not cmp.present("가나다라XX사아자차카타", text)      # 2자 틀림


def test_segments_split_at_rule_or_wide_gap():
    cs = [("가", _box(0)), ("나", _box(10)), ("다", _box(25)), ("라", _box(35))]   # 5pt 틈은 같은 조각
    assert [s for s, _ in cmp._segments(cs)] == ["가나다라"]
    assert [s for s, _ in cmp._segments(cs, [(22, 0, 10)])] == ["가나", "다라"]      # 세로 테두리선
    far = [("가", _box(0)), ("나", _box(10)), ("다", _box(50))]                       # 글자 높이 2배보다 넓은 틈
    assert [s for s, _ in cmp._segments(far)] == ["가나", "다"]


def test_norm_drops_markup_bullets_and_leaders():
    assert cmp.norm("- 가 나 ...... 3") == "가나3"
    assert cmp.norm("• 가 | **나**") == "가나"
    assert cmp.norm("‘가’ “나”") == cmp.norm("'가' \"나\"")


def test_evaluate_counts():
    golden = {"pages": {1: "normal", 2: "toc"}, "junk": [{"page": 1, "kind": "쪽번호", "bbox": [90, 190, 110, 200]}],
              "tables": [{"page": 1, "bbox": [0, 100, 100, 150], "rows": 2, "cols": 3, "header_rows": 1,
                          "caption": [0, 85, 100, 95]}], "figures": [{"page": 1, "bbox": [200, 0, 300, 100]}]}
    segs = {1: [("본문첫째줄", [0, 0, 100, 10]), ("12", [90, 190, 110, 200])],
            2: [("목차", [0, 0, 20, 10]), ("둘째항목", [0, 20, 40, 30])]}
    html = "<table><tr><th>a</th><th>b</th><th>c</th></tr><tr><td>d</td><td>가\\n나</td><td>f</td></tr></table>"
    run = {"seconds": 1.0, "pages": {"1": [
        cmp._blk("text", [0, 0, 100, 10], True, "본문 첫째 줄 $m^2$ 단가 $5$ C:\\nas", "text"),
        cmp._blk("table", [0, 100, 100, 150], True, None, "table", html=html),
        cmp._blk("text", [90, 190, 110, 200], False, "12", "number"),
        cmp._blk("text", [210, 10, 290, 50], True, "그림 속 글자", "text"),
        {**cmp._blk("text", [200, 0, 300, 100], True, "그림을 감싼 문단", "text"), "joined": True}],
        "2": [cmp._blk("text", [0, 0, 20, 10], True, "목차", "paragraph_title"),
              cmp._blk("text", [0, 20, 40, 30], False, "둘째 항목", "content")]}}
    run["pages"]["1"][1]["caption_bbox"] = [10, 86, 90, 94]
    r = cmp.evaluate(segs, golden, run)
    assert r["text"]["normal"] == [1, 1]
    assert r["leak"]["toc"] == [2, 6]                           # 목차 쪽 6자 중 제목 2자가 본문에 남음
    assert r["junk_removed"] == {"쪽번호": [1, 1]}
    assert r["tables"] == {"detected": 1, "grid_ok": 1, "header_ok": 1, "caption_ok": 1, "total": 1}
    assert r["junk_text"] == 2                                   # 표 칸에 글자 그대로 남은 \n, LaTeX 한 개(금액·경로는 아님)
    assert r["figure_chars"] == 5                                # 그림 감싼 문단을 이은 블록은 세지 않는다


def _heading(bbox, level, body=True):
    return cmp._blk("text", bbox, body, "제목", "paragraph_title", heading=True, heading_level=level)


def _item(bbox, level):
    return cmp._blk("text", bbox, True, "항목", "text", list_level=level)


def test_structure_counts():
    golden = {"pages": {1: "normal"},
              "headings": [{"page": 1, "bbox": [50, 100, 300, 115], "level": 1},
                           {"page": 1, "bbox": [50, 200, 300, 215], "level": 2},
                           {"page": 1, "bbox": [50, 300, 300, 315], "level": 2},
                           {"page": 1, "bbox": [50, 400, 300, 415], "level": 1}],
              "lists": [{"page": 1, "bbox": [60, 500, 500, 515], "level": 1},
                        {"page": 1, "bbox": [80, 520, 500, 535], "level": 2},
                        {"page": 1, "bbox": [80, 560, 500, 575], "level": 2}]}
    run = {"seconds": 1.0, "pages": {"1": [
        _heading([50, 101, 300, 116], 1),                                    # 찾음, 수준 맞음
        _heading([50, 200, 520, 260], 1),                                    # 제목과 다음 줄을 묶음(IoU 낮음, 덮음) — 찾음, 수준 틀림
        cmp._blk("text", [50, 300, 300, 330], True, "본문", "text"),          # 글 블록만 겹침 — 못 찾음
        _heading([50, 400, 300, 415], 1, body=False),                        # 버린 블록은 찾음이 아니다
        _heading([50, 700, 300, 715], 1),                                    # 정답에 없는 제목은 세지 않는다
        _item([60, 500, 500, 515], 1), _item([40, 495, 560, 518], 2),        # 더 많이 겹치는 블록의 수준을 본다
        _item([80, 520, 500, 535], 1),                                       # 수준 틀림
        _item([80, 560, 200, 575], 2)]}}                                     # 조금만 겹침 — 겹친 것으로 보지 않는다
    r = cmp.evaluate({}, golden, run)
    assert r["headings"] == {"found": [2, 4], "level_ok": [1, 2]}
    assert r["lists"] == {"level_ok": [1, 3]}


def test_structure_without_golden_or_headings():
    run = {"seconds": None, "pages": {"1": [cmp._blk("text", [0, 0, 595, 842], True, "쪽 전체 글", "page")]}}   # 텍스트 레이어 꼴
    none = cmp.evaluate({}, {"pages": {1: "normal"}}, run)
    some = cmp.evaluate({}, {"pages": {1: "normal"}, "headings": [{"page": 1, "bbox": [50, 100, 300, 115], "level": 1}],
                             "lists": [{"page": 1, "bbox": [60, 500, 500, 515], "level": 1}]}, run)
    assert none["headings"] == {"found": [0, 0], "level_ok": [0, 0]} and none["lists"] == {"level_ok": [0, 0]}
    assert some["headings"] == {"found": [0, 1], "level_ok": [0, 0]} and some["lists"] == {"level_ok": [0, 1]}
    rows = {c[0]: c[1:] for c in ([s.strip() for s in line.strip("|").split("|")]
                                  for line in cmp.render({"a": none, "b": some}).splitlines())}
    assert rows["제목 찾음"] == ["—", "0/1 (0.0%)"]                      # 정답에 제목이 없으면 '—'
    assert rows["제목 수준 일치(찾은 제목 중)"] == ["—", "—"]             # 찾은 제목이 없으면 '—'
    assert rows["항목 수준 일치"] == ["—", "0/1 (0.0%)"]


def test_load_kdp_heading_and_list_levels(tmp_path):
    blocks = [Block(id="p1.b01", type="paragraph_title", bbox=(50, 100, 300, 115), text="Ⅰ. 개요", heading_level=1),
              Block(id="p1.b02", type="paragraph_title", bbox=(50, 120, 300, 135), text="수준 모름"),
              Block(id="p1.b03", type="text", bbox=(50, 140, 300, 155), text="가. 규칙이 올린 제목", heading_level=3),
              Block(id="p1.b04", type="text", bbox=(60, 160, 500, 175), text="○ 항목", list_level=2),
              Block(id="p1.b05", type="text", bbox=(60, 180, 500, 195), text="본문")]
    doc = Document(doc_id="sha256:ab", source=Source(paths=["a.pdf"], pages=1),
                   parser=ParserInfo(kdp_version="0", engine={}, engine_hash="e", assemble_hash="a", run_at="t"),
                   pages=[Page(page_no=1, sheet_no=1, size=(595, 842), blocks=blocks)])
    write_json(tmp_path / "document.json", doc.model_dump(mode="json", exclude_none=True))   # 정본처럼 None 은 빠진다
    write_json(tmp_path / "raw" / "engine.json", {"seconds": 1.0})
    got = [(b["heading"], b["heading_level"], b["list_level"]) for b in cmp.load_kdp(tmp_path)["pages"][1]]
    assert got == [(True, 1, None), (True, None, None), (True, 3, None), (False, None, 2), (False, None, None)]


def test_paddle_default_title_level():
    lv = cmp.paddle_title_level                                             # 설치본 기본 마크다운: 제목 안 '.' 개수
    assert lv("doc_title", "결과 보고서") == 0
    assert [lv("paragraph_title", t) for t in ("사업 개요", "1. 사업 개요", "1.1 추진 경과", "Ⅰ. 개요", "2026.03.15 계획")] \
        == [1, 2, 2, 2, 3]
    assert lv("text", "1. 사업 개요") is None and lv("figure_title", "<표 1> 현황") is None


def test_docling_heading_and_list_depth():
    dc = pytest.importorskip("docling_core.types.doc")                     # docling 은 파서 env 밖에 둘 수 있다
    d = dc.DoclingDocument(name="t")
    title, head = d.add_title("결과 보고서"), d.add_heading("1.1 추진 경과", level=2)
    outer = d.add_list_item("○ 항목", parent=d.add_list_group())
    inner = d.add_list_item("- 하위", parent=d.add_list_group(parent=outer))
    text = d.add_text(label=dc.DocItemLabel.TEXT, text="본문")
    assert [cmp._docling_struct(i, d) for i in (title, head, outer, inner, text)] == [
        {"heading": True, "heading_level": 0}, {"heading": True, "heading_level": 2},
        {"list_level": 1}, {"list_level": 2}, {"list_level": None}]


def test_list_item_split_into_two_blocks():
    golden = {"pages": {1: "normal"}, "lists": [{"page": 1, "bbox": [100, 200, 500, 260], "level": 1}]}
    def run(lv2):
        return {"pages": {"1": [cmp._blk("text", [98, 199, 502, 229], True, "• 앞 두 줄", "text", list_level=1),
                                cmp._blk("text", [98, 231, 502, 261], True, "이어지는 두 줄", "text", list_level=lv2)]}}
    assert cmp.evaluate({}, golden, run(1))["lists"] == {"level_ok": [1, 1]}   # 검출기가 나눈 조각이 모두 맞으면 맞음
    assert cmp.evaluate({}, golden, run(None))["lists"] == {"level_ok": [0, 1]}
