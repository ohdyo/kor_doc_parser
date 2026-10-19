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


def test_figure_image_text_and_meta():
    golden = {"pages": {1: "normal"}, "meta": {"title": "보도 제목", "date": "2026-06-22"},
              "figures": [{"page": 1, "bbox": [0, 0, 100, 100], "kind": "chart", "caption": [0, -20, 100, -10]},
                          {"page": 1, "bbox": [200, 0, 300, 100], "kind": "photo", "caption": None},
                          {"page": 1, "bbox": [0, 200, 300, 400], "kind": "poster", "caption": None},
                          {"page": 1, "bbox": [400, 0, 500, 100], "kind": "map", "caption": [400, -20, 500, -10]}],
              "image_text": [{"page": 1, "lines": ["모기기피제 사용", "야외활동 후 샤워", "없는 줄입니다"]}]}
    run = {"meta": {"title": "보도  제목", "date": "2026-06-21"}, "pages": {"1": [
        {**cmp._blk("figure", [2, 2, 98, 98], True, "", "chart"), "caption_bbox": [10, -18, 90, -12]},   # 찾음, 제목 이음
        cmp._blk("text", [10, 80, 60, 95], True, "20주 21주", "text"),           # 그래프 속 글자가 글 블록으로 — 섞임
        cmp._blk("figure", [200, 0, 245, 100], True, "", "image"),               # 사진을 둘로 나눔 — 함께 덮으면 찾음
        cmp._blk("figure", [250, 0, 300, 100], True, "", "image"),
        cmp._blk("figure", [0, 0, 600, 800], True, "", "image"),                 # 쪽 전체 그림은 어느 그림도 아니다
        cmp._blk("text", [10, 210, 290, 390], True, "모기기피제 사용 야외활동 후 샤워", "text"),   # 포스터 글은 섞임이 아니다
        cmp._blk("figure", [400, 0, 500, 100], False, "", "image")]}}           # 버린 그림은 못 찾음
    r = cmp.evaluate({}, golden, run)
    assert r["figures"] == {"found": [2, 4], "caption_ok": [1, 2], "leak_chars": 6}
    assert r["image_text"] == [2, 3]
    assert r["meta"] == [1, 2]                                                    # 띄어쓰기 차이는 같게 본다
    assert r["figure_chars"] is None                                              # 그림으로 된 표(종류 없는 그림)가 없다
    rows = {c[0]: c[1:] for c in ([s.strip() for s in line.strip("|").split("|")]
                                  for line in cmp.render({"a": r, "b": cmp.evaluate({}, {"pages": {1: "normal"}}, run)})
                                  .splitlines())}
    assert rows["그림 찾음"] == ["2/4 (50.0%)", "—"]                              # 정답에 그림 종류가 없는 쪽은 '—'
    assert rows["그림 속 글자 읽음(포스터 줄)"] == ["2/3 (66.7%)", "—"]
    assert rows["문서 정보 맞음(제목·기관·날짜)"] == ["1/2 (50.0%)", "—"]
    assert rows["그림 영역에서 읽은 글자 수"] == ["—", "—"]
    no_meta = cmp.evaluate({}, golden, {"pages": run["pages"]})                   # 비교군은 메타를 내지 않는다
    assert no_meta["meta"] is None


def test_report_rows_unchanged_without_new_golden_fields():
    r = cmp.evaluate({}, {"pages": {1: "normal"}}, {"seconds": 1.0, "pages": {"1": []}})
    labels = [line.strip("|").split("|")[0].strip() for line in cmp.render({"a": r}).splitlines()[2:]]
    assert not {"그림 찾음", "그림 속 글자 읽음(포스터 줄)", "문서 정보 맞음(제목·기관·날짜)", "그래프·지도 속 글자 읽음"} & set(labels)


def test_figure_text_read_and_wrong_lines():
    # 그래프·지도 속 글자: 정답 줄(2자 이상)이 그 자리 그림 블록 글에 있나, 글자가 든 줄 가운데 정답에도 그 자리 글자층에도 없는
    # 낱말이 섞인 줄 — 숫자뿐인 줄·한 글자('N')는 세지 않고, 그림에 붙은 글자층 글('(a)')은 맞는 글이다
    golden = {"pages": {1: "normal"},
              "figure_text": [{"page": 1, "bbox": [0, 0, 100, 100], "lines": ["모기지수", "20주", "인천", "서구(2)", "N"]},
                              {"page": 1, "bbox": [200, 0, 300, 100], "lines": ["평년 평균 기온"]}]}
    run = {"pages": {"1": [
        cmp._blk("figure", [2, 2, 98, 98], True, "20주 21주\n인천 서구(2)\n0.5 1.0\n13월 모기지수\nN\n(a)", "chart"),   # 21주·13월은 없다
        cmp._blk("text", [10, 80, 60, 95], True, "평년 평균 기온", "text"),       # 글 블록은 그림 글이 아니다
        cmp._blk("figure", [200, 0, 300, 100], True, "", "chart")]}}            # 글을 읽지 않은 그래프
    r = cmp.evaluate({1: [("(a)", [4, 4, 14, 12])]}, golden, run)
    assert r["figure_text"] == {"read": [4, 5], "wrong_lines": [2, 4]}
    rows = {c[0]: c[1:] for c in ([s.strip() for s in line.strip("|").split("|")]
                                  for line in cmp.render({"a": r, "b": cmp.evaluate({}, {"pages": {1: "normal"}}, run)})
                                  .splitlines())}
    assert rows["그래프·지도 속 글자 읽음"] == ["4/5 (80.0%)", "—"]
    assert rows["그래프·지도 글 중 정답에 없는 줄(오독·지어냄)"] == ["2/4 (50.0%)", "—"]


def test_figure_route_and_description():
    """그림 갈래·설명(D19) — 갈래는 그 자리 그림 블록이 모두 맞아야, 사실은 설명의 한 문장 안에 무리마다 낱말 하나라도
    (YAML 이 수로 읽은 0.5 도), 버린 문장은 근거 검사 기록에서 센다. 비교군은 갈래·설명이 없다."""
    golden = {"pages": {1: "normal"}, "figures": [
        {"page": 1, "bbox": [0, 0, 100, 100], "kind": "chart", "caption": None, "route": "image",
         "facts": [[["남서", "서쪽"], ["높"]], [[0.5], ["넘", "초과"]], [["북동"], ["낮"]]]},
        {"page": 1, "bbox": [200, 0, 300, 100], "kind": "photo", "caption": None, "route": "image", "facts": [[["45"]]]},
        {"page": 1, "bbox": [400, 0, 500, 100], "kind": "poster", "caption": None}]}
    def fig(bbox, route, desc="", dropped=0):
        return {**cmp._blk("figure", bbox, True, "", "chart"), "route": route, "desc": desc, "desc_dropped": dropped}
    run = {"pages": {"1": [
        fig([2, 2, 98, 98], "image", "서쪽 지역이 가장 높다. 0.5 선을 넘은 곳이 많다. 북동쪽은 높다.", 2),   # 셋째는 다른 문장의 '낮'뿐
        fig([200, 0, 245, 100], "image"), fig([255, 0, 300, 100], "text"),                         # 둘로 나눈 사진 — 하나가 틀림
        fig([400, 0, 500, 100], "image")]}}
    r = cmp.evaluate({}, golden, run)
    assert r["figure_desc"] == {"route_ok": [1, 2], "described": [1, 2], "facts": [2, 4], "dropped": 2}
    base = cmp.evaluate({}, golden, {"pages": {"1": [cmp._blk("figure", [2, 2, 98, 98], True, "", "chart")]}})
    assert base["figure_desc"] == {"route_ok": [0, 2], "described": [0, 2], "facts": [0, 4], "dropped": 0}
    rows = {c[0]: c[1:] for c in ([x.strip() for x in line.strip("|").split("|")]
                                  for line in cmp.render({"a": r, "b": base}).splitlines())}
    assert rows["그림 설명에 든 정답 사실"] == ["2/4 (50.0%)", "0/4 (0.0%)"]
    assert rows["그림 설명에서 버린 문장(근거 없는 숫자·이름)"] == ["2", "0"]


def test_fact_words_match_whole_numbers_and_split_figure_pieces():
    """숫자 낱말은 앞뒤가 숫자가 아닐 때만('1일' 은 '21일' 에, '0일' 은 '30일' 에 없다). 엔진이 한 정답 그림을 큰 조각과 작은 조각으로
    잡으면(작은 조각은 IoU 가 낮아도 정답 상자 안) 둘의 설명을 함께 본다."""
    assert cmp._has("1일", "9월1일에가장많다.") and not cmp._has("1일", "21일에가장많다.")
    assert not cmp._has(0, "2026년이다.") and cmp._has(0.5, "0.5선을넘었다.") and not cmp._has(0.5, "10.5선이다.")
    golden = {"pages": {1: "normal"}, "figures": [
        {"page": 1, "bbox": [246, 353, 522, 436], "kind": "photo", "caption": None, "route": "image",
         "facts": [[["날개"], ["반점"]], [["촉수"], ["주둥이"]]]}]}
    run = {"pages": {"1": [
        {**cmp._blk("figure", [360, 352, 522, 436], True, "", "image"), "route": "image", "desc": "날개에 검은 반점이 있다."},
        {**cmp._blk("figure", [244, 352, 344, 436], True, "", "image"), "route": "image", "desc": "촉수가 주둥이 옆에 있다."}]}}
    assert cmp.evaluate({}, golden, run)["figure_desc"] == {"route_ok": [1, 1], "described": [1, 1], "facts": [2, 2], "dropped": 0}


def test_load_kdp_figure_route_and_description(tmp_path):
    blocks = [Block(id="p1.b01", type="chart", bbox=(50, 100, 500, 400), img_path="imgs/p1.b01.png", route="image",
                    route_reason="chart", figure_desc="남서쪽이 높다.",
                    provenance=[{"rule": "figure_desc", "model": "m", "dropped": ["가.", "나."]}]),
              Block(id="p1.b02", type="text", bbox=(50, 420, 500, 440), text="본문")]
    doc = Document(doc_id="sha256:ab", source=Source(paths=["a.pdf"], pages=1),
                   parser=ParserInfo(kdp_version="0", engine={}, engine_hash="e", assemble_hash="a", run_at="t"),
                   pages=[Page(page_no=1, sheet_no=1, size=(595, 842), blocks=blocks)])
    write_json(tmp_path / "document.json", doc.model_dump(mode="json", exclude_none=True))
    write_json(tmp_path / "raw" / "engine.json", {"seconds": 1.0})
    a, b = cmp.load_kdp(tmp_path)["pages"][1]
    assert (a["route"], a["desc"], a["desc_dropped"]) == ("image", "남서쪽이 높다.", 2) and "route" not in b


def test_list_item_split_into_two_blocks():
    golden = {"pages": {1: "normal"}, "lists": [{"page": 1, "bbox": [100, 200, 500, 260], "level": 1}]}
    def run(lv2):
        return {"pages": {"1": [cmp._blk("text", [98, 199, 502, 229], True, "• 앞 두 줄", "text", list_level=1),
                                cmp._blk("text", [98, 231, 502, 261], True, "이어지는 두 줄", "text", list_level=lv2)]}}
    assert cmp.evaluate({}, golden, run(1))["lists"] == {"level_ok": [1, 1]}   # 검출기가 나눈 조각이 모두 맞으면 맞음
    assert cmp.evaluate({}, golden, run(None))["lists"] == {"level_ok": [0, 1]}
