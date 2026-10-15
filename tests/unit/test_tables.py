"""표(기획서 4.3 규칙 6·13, D18) — 머리행, 제목·단위·출처 연결, 표 갈래."""
from kdp.assemble.tables import header_rows, parse_rows, tables
from kdp.config import QCCfg
from kdp.qc.checks import run_checks
from kdp.schema import Block, Document, Page, ParserInfo, Source

T = ("<table><tr><td>영역</td><td>주요 지적사항</td><td>개선 현황</td></tr>"
     "<tr><td>사업관리</td><td>요구사항 추적표 보완이 필요함</td><td>추적표를 보완하였음</td></tr>"
     "<tr><td>데이터</td><td>표준화 점검 결과 반영이 필요함</td><td>반영 완료</td></tr></table>")


def _doc(*blocks):
    page = Page(page_no=1, sheet_no=1, size=(595, 842), blocks=[
        Block(id=f"p1.b{i + 1:02d}", type=t, bbox=bb, text=x if t != "table" else None, html=x if t == "table" else None)
        for i, (t, bb, x) in enumerate(blocks)])
    return Document(doc_id="sha256:ab", source=Source(paths=["a.pdf"], pages=1),
                    parser=ParserInfo(kdp_version="0", engine={}, engine_hash="e", assemble_hash="a", run_at="t"),
                    pages=[page])


def test_header_rows():
    assert header_rows(parse_rows(T)) == 1
    numeric = "<table><tr><td>2025</td><td>2026</td></tr><tr><td>10</td><td>20</td></tr></table>"
    assert header_rows(parse_rows(numeric)) == 0
    two = ('<table><tr><td rowspan="2">구분</td><td colspan="2">예산</td></tr><tr><td>본예산</td><td>추경</td></tr>'
           "<tr><td>사업비</td><td>1,500</td><td>300</td></tr></table>")
    assert header_rows(parse_rows(two)) == 2
    assert header_rows(parse_rows("<table><tr><td>합계</td></tr></table>")) == 0


def test_caption_unit_source_and_th():
    doc = tables(_doc(("figure_title", (120, 425, 360, 436), "[도표 1] 설계단계 주요 지적사항 (단위: 건)"),
                      ("table", (84, 438, 550, 724), T),
                      ("text", (84, 726, 300, 738), "자료: 감리 결과")))
    t = doc.pages[0].blocks[1]
    assert (t.caption_id, t.unit, t.note_ids, t.header_rows) == ("p1.b01", "건", ["p1.b03"], 1)
    assert t.html.startswith("<table><tr><th>영역</th><th>주요 지적사항</th><th>개선 현황</th></tr><tr><td>")
    assert (t.route, t.route_reason) == ("html", None)
    plain = tables(_doc(("figure_title", (120, 425, 360, 436), "종료단계 감리 결과"), ("table", (84, 438, 550, 724), T)))
    assert plain.pages[0].blocks[1].caption_id is None                 # 패턴 없는 구역 제목은 잇지 않는다
    far = tables(_doc(("figure_title", (120, 300, 360, 311), "[표 2] 일정"), ("table", (84, 438, 550, 724), T)))
    assert far.pages[0].blocks[1].caption_id is None


def test_route():
    photo = T.replace("반영 완료", '<img src="imgs/p1.t01.png" alt="Image" />반영 완료')
    empty = "<table><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table>"
    doc = _doc(("table", (0, 0, 100, 100), photo), ("table", (0, 200, 100, 300), empty),
               ("table", (0, 400, 100, 500), T), ("table", (0, 600, 100, 700), T))
    doc.pages[0].blocks[2].truncated = "length"
    doc.pages[0].blocks[3].provenance = [{"rule": "text_layer", "agreement": 0.3, "cells": 0}]
    out = tables(doc).pages[0].blocks
    assert [(b.route, b.route_reason) for b in out] == [("image", "photo_in_cell"), ("image", "failed"),
                                                        ("image", "failed"), ("image", "failed")]


def test_layout_table_becomes_text():
    box = "<table><tr><td>※ 참고: 본 사업은 2단계로 나누어 추진하며 1단계 결과를 보고 2단계 범위를 정한다</td></tr></table>"
    b = tables(_doc(("table", (80, 100, 520, 160), box))).pages[0].blocks[0]
    assert (b.type, b.html, b.route) == ("text", None, None) and b.text.startswith("※ 참고")
    assert b.provenance == [{"rule": "layout_table"}]


def test_route_forms_and_unparsed():
    form = ("<table><tr><td>성명</td><td></td></tr><tr><td>소속</td><td></td></tr>"
            "<tr><td>연락처</td><td></td></tr><tr><td>신청 사유</td><td></td></tr></table>")
    plain = "구분 내용 비고 사업관리 요구사항"                              # OTSL 없이 평문이 온 표
    out = tables(_doc(("table", (0, 0, 100, 100), form), ("table", (0, 200, 100, 300), plain))).pages[0].blocks
    assert [(b.route, b.route_reason) for b in out] == [("html", None), ("image", "failed")]


def test_caption_and_unit_stay_with_their_table():
    t2 = T.replace("영역", "구분")
    doc = tables(_doc(("figure_title", (84, 100, 400, 112), "<표 1> 투입 인력 현황"),
                      ("table", (84, 115, 550, 300), T),
                      ("figure_title", (84, 310, 400, 322), "<표 2> 소요 예산 (단위: 백만원)"),
                      ("table", (84, 325, 550, 500), t2)))
    a, b = doc.pages[0].blocks[1], doc.pages[0].blocks[3]
    assert (a.caption_id, a.unit) == ("p1.b01", None) and (b.caption_id, b.unit) == ("p1.b03", "백만원")
    no_title = tables(_doc(("table", (84, 115, 550, 300), T),
                           ("figure_title", (84, 310, 400, 322), "<표 2> 소요 예산 (단위: 백만원)"),
                           ("table", (84, 325, 550, 500), t2))).pages[0].blocks
    assert (no_title[0].caption_id, no_title[0].unit) == (None, None) and no_title[2].caption_id == "p1.b02"
    fig = tables(_doc(("figure_title", (84, 100, 400, 112), "[그림 3] 조직 구성도"),
                      ("table", (84, 115, 550, 300), T))).pages[0].blocks[1]
    assert fig.caption_id is None                                    # 그림 제목은 표에 잇지 않는다


def test_layout_table_keeps_photo_and_crop():
    box = "<table><tr><td>※ 참고: 현장 점검 사진과 함께 보완 결과를 정리하여 다음 회의에서 보고한다</td></tr></table>"
    lost = _doc(("table", (80, 100, 520, 160), box))
    lost.pages[0].blocks[0].provenance = [{"rule": "table_img_missing", "names": ["p1.t01.png"]}]
    b = tables(lost).pages[0].blocks[0]
    assert (b.type, b.route, b.route_reason) == ("table", "image", "photo_in_cell")
    ok = _doc(("table", (80, 100, 520, 160), box))
    ok.pages[0].blocks[0].img_path = "imgs/p1.b01.png"
    b = tables(ok).pages[0].blocks[0]
    assert b.type == "text" and b.img_path == "imgs/p1.b01.png"


def test_caption_below_side_by_side_and_source():
    below = tables(_doc(("table", (84, 100, 550, 300), T), ("figure_title", (84, 304, 400, 316), "<표 1> 인력 현황"),
                        ("table", (84, 326, 550, 500), T), ("figure_title", (84, 504, 400, 516), "<표 2> 예산 현황")))
    b = below.pages[0].blocks
    assert (b[0].caption_id, b[2].caption_id) == ("p1.b02", "p1.b04")   # 제목이 표 아래에 오는 문서
    side = tables(_doc(("figure_title", (84, 90, 500, 102), "<표 3> 예산 비교 (단위: 백만원)"),
                       ("table", (84, 105, 300, 300), T), ("table", (310, 105, 550, 300), T))).pages[0].blocks
    assert [(x.caption_id, x.unit) for x in side[1:]] == [("p1.b01", "백만원")] * 2   # 나란한 두 표는 한 제목을 함께
    src = tables(_doc(("table", (84, 100, 550, 300), T), ("text", (84, 303, 400, 313), "자료: 행정안전부"),
                      ("chart", (84, 322, 550, 500), None))).pages[0].blocks
    assert src[0].note_ids == ["p1.b02"]                                 # 출처 주는 자기 표 아래
    fig = tables(_doc(("figure_title", (84, 100, 400, 112), "[그림 3] 연도별 예산 추이 (단위: 억원)"),
                      ("table", (84, 115, 550, 300), T))).pages[0].blocks[1]
    assert (fig.caption_id, fig.unit) == (None, None)


def test_blank_signature_sheet_is_not_failed():
    rows = "".join("<tr><td></td><td></td><td></td><td></td></tr>" for _ in range(10))
    sheet = "<table><tr><td>성명</td><td>소속</td><td>직위</td><td>서명</td></tr>" + rows + "</table>"
    t = tables(_doc(("table", (0, 0, 100, 100), sheet))).pages[0].blocks[0]
    assert (t.route, t.route_reason) == ("html", None)


def test_layer_agreement_routes_and_qc():
    doc = _doc(("table", (0, 0, 100, 100), T), ("table", (0, 200, 100, 300), T))
    doc.pages[0].blocks[0].provenance = [{"rule": "text_layer", "agreement": 0.7, "cells": 0}]    # 모델이 행을 빠뜨림
    doc.pages[0].blocks[1].provenance = [{"rule": "text_layer", "agreement": 0.85, "cells": 2}]
    out = tables(doc)
    assert [(b.route, b.route_reason) for b in out.pages[0].blocks] == [("image", "failed"), ("html", None)]
    assert "table_layer_agreement" in run_checks(out, [], QCCfg()).qc.reasons      # 0.9 미만은 검수
