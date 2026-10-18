"""표(기획서 4.3 규칙 1·6·13, D18) — 머리행, 제목·단위·출처 연결, 표 갈래, '이하 빈칸', 쪽 넘김 표."""
from kdp.assemble.tables import appendage, header_rows, merge_tables, parse_rows, tables
from kdp.classify.blocks import classify_blocks
from kdp.config import Classify, QCCfg
from kdp.qc.checks import run_checks
from kdp.schema import Block, Document, NUp, Page, ParserInfo, Source

T = ("<table><tr><td>영역</td><td>점검 내용</td><td>조치 결과</td></tr>"
     "<tr><td>사업관리</td><td>관리 대장 보완이 필요함</td><td>관리 대장을 보완하였음</td></tr>"
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
    doc = tables(_doc(("figure_title", (120, 425, 360, 436), "[도표 1] 분기별 점검 결과 (단위: 건)"),
                      ("table", (84, 438, 550, 724), T),
                      ("text", (84, 726, 300, 738), "자료: 분기 점검 결과")))
    t = doc.pages[0].blocks[1]
    assert (t.caption_id, t.unit, t.note_ids, t.header_rows) == ("p1.b01", "건", ["p1.b03"], 1)
    assert t.html.startswith("<table><tr><th>영역</th><th>점검 내용</th><th>조치 결과</th></tr><tr><td>")
    assert (t.route, t.route_reason) == ("html", None)
    plain = tables(_doc(("figure_title", (120, 425, 360, 436), "하반기 점검 결과"), ("table", (84, 438, 550, 724), T)))
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


def test_blank_below_rows_removed():
    blank = ('<table><tr><td>품명</td><td>수량</td><td>비고</td></tr><tr><td>노트북</td><td>2</td><td>교체</td></tr>'
             '<tr><td colspan="3">- 이하 빈칸 -</td></tr><tr><td></td><td></td><td></td></tr></table>')
    split = blank.replace('<td colspan="3">- 이하 빈칸 -</td>', "<td>이하</td><td>빈칸</td><td></td>")
    value = blank.replace('<td colspan="3">- 이하 빈칸 -</td>', "<td>3</td><td>이하 빈칸</td><td></td>")
    spanned = ('<table><tr><td rowspan="3">구분</td><td>품명</td></tr><tr><td>노트북</td></tr>'
               "<tr><td>이하 빈칸</td></tr></table>")
    only = "<table><tr><td>(이하 여백)</td></tr></table>"
    out = tables(_doc(*(("table", (0, 200 * i, 500, 200 * i + 150), h)
                        for i, h in enumerate((blank, split, value, spanned, only))))).pages[0].blocks
    assert "이하" not in out[0].html and "<td>노트북</td>" in out[0].html and out[0].html.count("<tr>") == 3
    assert out[0].provenance == [{"rule": "blank_below", "removed": ["- 이하 빈칸 -"]}]
    assert "이하" not in out[1].html                                      # 칸에 나뉘어 적혀도
    assert "이하 빈칸" in out[2].html and not out[2].provenance            # 다른 값이 있는 행은 둔다
    assert "이하 빈칸" in out[3].html                                      # 세로 병합이 걸친 행은 둔다(칸이 어긋날 수 있다)
    assert "이하 여백" in (out[4].html or out[4].text)                     # 지우면 빈 표가 되는 것은 둔다


HEAD = ("header", (70, 40, 520, 55), "○○ 사업 결과 보고서")
BUDGET = ("<table><tr><td>구분</td><td>본예산</td><td>집행액</td></tr><tr><td>인건비</td><td>1,500</td><td>1,200</td></tr>"
          "<tr><td>운영비</td><td>800</td><td>650</td></tr></table>")
BUDGET2 = ("<table><tr><td>구분</td><td>본예산</td><td>집행액</td></tr><tr><td>장비비</td><td>2,000</td><td>1,900</td></tr>"
           "<tr><td>합계</td><td>4,300</td><td>3,750</td></tr></table>")
REST = ("<table><tr><td>보안</td><td>접근 통제 점검 보완 필요</td><td>보완 완료</td></tr>"
        "<tr><td>품질</td><td>시험 결과서와 산출물 목록의 정합성 보완이 필요함</td><td>보완 예정</td></tr></table>")
STAFF = ("<table><tr><td>직급</td><td>정원</td><td>현원</td></tr><tr><td>책임</td><td>4</td><td>3</td></tr>"
         "<tr><td>선임</td><td>6</td><td>6</td></tr></table>")                # 열 수·폭이 BUDGET 과 같은 다른 표


def _page(no, *blocks, size=(595, 842), sheet=None, n_up=None):
    return Page(page_no=no, sheet_no=sheet or no, size=size, n_up=n_up, blocks=[
        Block(id=f"p{no}.b{i + 1:02d}", type=t, bbox=bb, text=x if t != "table" else None, html=x if t == "table" else None)
        for i, (t, bb, x) in enumerate(blocks)])


def _merge(*pages):
    doc = Document(doc_id="sha256:ab", source=Source(paths=["a.pdf"], pages=len(pages)),
                   parser=ParserInfo(kdp_version="0", engine={}, engine_hash="e", assemble_hash="a", run_at="t"),
                   pages=list(pages))
    return merge_tables(tables(classify_blocks(doc, Classify())))


def _num(no):
    return ("number", (290, 800, 305, 812), f"- {no} -")


def _end_page(html=BUDGET, *after):
    return _page(1, HEAD, ("figure_title", (84, 560, 400, 572), "<표 3> 사업비 집행 현황"),
                 ("text", (470, 560, 550, 572), "(단위: 천원)"), ("table", (84, 575, 550, 760), html), *after, _num(1))


def test_merge_across_continued_unit_and_title():
    doc = _merge(_end_page(BUDGET, ("text", (450, 762, 550, 772), "(다음 쪽에 계속)"),
                           ("footnote", (70, 780, 520, 792), "1) 2026년 6월 말 기준")),
                 _page(2, HEAD, ("text", (84, 60, 150, 72), "(계속)"),
                       ("figure_title", (84, 75, 400, 87), "<표 3> 사업비 집행 현황(계속)"),
                       ("text", (470, 75, 550, 87), "(단위: 천원)"), ("table", (84, 90, 550, 300), BUDGET2),
                       ("text", (84, 320, 550, 360), "집행률은 계획 대비 87% 이다."), _num(2)))
    p1, p2 = doc.pages[0].blocks, doc.pages[1].blocks
    t = p2[4]
    assert (t.continues_from, t.header_rows, t.unit, t.caption_id) == ("p1.b04", 1, "천원", "p2.b03")
    assert t.provenance == [{"rule": "merge_tables", "repeated_header_rows": 1}]
    assert t.html.count("<th>") == 3 and "<td>장비비</td>" in t.html          # HTML 은 합치지 않는다 — 조각마다 제 쪽에
    for m in (p1[4], p2[1]):                                                 # '(계속)' 표시만 버린다
        assert (m.keep, m.drop_reason, m.provenance) == (False, "continued", [{"rule": "merge_tables", "table": "p2.b05"}])
    assert all(b.keep for b in (p1[5], p2[2], p2[3]))                        # 각주·제목 반복·단위 줄은 남는다
    assert p1[3].continues_from is None and p1[3].provenance is None


def test_not_merged():
    four = ("<table><tr><td>구분</td><td>본예산</td><td>집행액</td><td>비고</td></tr>"
            "<tr><td>장비비</td><td>2,000</td><td>1,900</td><td></td></tr></table>")
    cases = {
        "열 수": _page(2, HEAD, ("table", (84, 90, 550, 300), four)),
        "폭": _page(2, HEAD, ("table", (84, 90, 400, 300), BUDGET2)),
        "다른 표 제목": _page(2, HEAD, ("figure_title", (84, 75, 400, 87), "<표 4> 인력 투입 현황"),
                          ("table", (84, 90, 550, 300), BUDGET2)),
        "제목": _page(2, HEAD, ("paragraph_title", (84, 60, 300, 75), "Ⅲ. 추진 결과"), ("table", (84, 90, 550, 300), BUDGET2)),
        "표를 가리키는 본문": _page(2, HEAD, ("text", (84, 60, 550, 85), "<표 3>의 인건비 세부 내역은 다음과 같다."),
                             ("table", (84, 90, 550, 300), BUDGET2)),
        "다른 단위": _page(2, HEAD, ("text", (84, 60, 150, 72), "(계속)"), ("text", (470, 75, 550, 87), "(단위: 백만원)"),
                       ("table", (84, 90, 550, 300), BUDGET2)),
    }
    for name, nxt in cases.items():
        doc = _merge(_end_page(), nxt)
        assert all(b.continues_from is None for p in doc.pages for b in p.blocks), name
        assert all(b.keep for b in doc.pages[1].blocks if b.type != "header"), name   # 잇지 못하면 '(계속)' 도 남긴다
    for label in ("text", "vision_footnote", "footnote"):                   # 출처 주가 붙은 표는 거기서 끝난다 — 라벨과 상관없이
        source = _merge(_end_page(BUDGET, (label, (84, 762, 300, 772), "자료: 기획재정부")),
                        _page(2, HEAD, ("table", (84, 90, 550, 300), STAFF)))
        t = source.pages[1].blocks[1]
        assert source.pages[0].blocks[3].note_ids == ["p1.b05"], label
        assert (t.continues_from, t.header_rows, t.provenance) == (None, 1, None) and "<th>직급</th>" in t.html, label
    dropped = _page(2, HEAD, ("table", (84, 90, 550, 300), BUDGET2))
    dropped.keep = False
    assert _merge(_end_page(), dropped).pages[1].blocks[1].continues_from is None


def test_unrepeated_header_cleared():
    doc = _merge(_page(1, HEAD, ("table", (84, 575, 550, 790), T), _num(1)),
                 _page(2, HEAD, ("table", (84, 60, 550, 200), REST), _num(2)))
    t = doc.pages[1].blocks[1]
    assert tables(_doc(("table", (84, 60, 550, 200), REST))).pages[0].blocks[0].header_rows == 1   # 따로 보면 첫 데이터 행이 머리행 같다
    assert (t.continues_from, t.header_rows) == ("p1.b02", None) and "<th>" not in t.html
    assert t.provenance == [{"rule": "merge_tables", "header_from": "p1.b02", "header_rows_was": 1}]


def test_three_page_chain():
    full = REST.replace("</table>", "<tr><td>운영</td><td>장애 대응 절차서 보완</td><td>보완 완료</td></tr></table>")
    again = T.replace("사업관리", "교육").replace("데이터", "인수인계")
    doc = _merge(_page(1, HEAD, ("text", (84, 100, 550, 560), "점검 결과는 다음과 같다."),
                       ("table", (84, 575, 550, 790), T), _num(1)),
                 _page(2, HEAD, ("table", (84, 60, 550, 790), full), _num(2)),
                 _page(3, HEAD, ("table", (84, 60, 550, 300), again), _num(3)))
    t2, t3 = doc.pages[1].blocks[1], doc.pages[2].blocks[1]
    assert (t2.continues_from, t3.continues_from) == ("p1.b03", "p2.b02")
    assert t2.provenance == [{"rule": "merge_tables", "header_from": "p1.b03", "header_rows_was": 1}] and t2.header_rows is None
    assert t3.provenance == [{"rule": "merge_tables", "repeated_header_rows": 1}] and t3.header_rows == 1   # 첫 조각과 견준다


def test_n_up_pages_use_logical_width():
    sheet = (842, 595)
    left = NUp(cols=2, rows=1, cell=(0, 0), region=(0, 0, 421, 595))
    right = NUp(cols=2, rows=1, cell=(0, 1), region=(421, 0, 842, 595))
    doc = _merge(_page(1, ("table", (40, 400, 380, 560), BUDGET), size=sheet, sheet=1, n_up=left),
                 _page(2, ("table", (461, 40, 801, 560), BUDGET2), size=sheet, sheet=1, n_up=right),
                 _page(3, ("table", (60, 60, 540, 400), BUDGET2), sheet=2))       # 보통 쪽 — 시트 pt 폭은 41% 크지만 쪽 대비 폭이 같다
    assert [p.blocks[0].continues_from for p in doc.pages] == [None, "p1.b01", "p2.b01"]
    same_pt = _merge(_page(1, ("table", (60, 500, 400, 800), BUDGET)),               # 시트 pt 폭이 같아도 쪽 대비 폭이 다르면 잇지 않는다
                     _page(2, ("table", (40, 40, 380, 300), BUDGET2), size=sheet, sheet=2, n_up=left))
    assert same_pt.pages[1].blocks[0].continues_from is None


def test_ended_table_not_merged():
    """'이하 빈칸' 으로 끝난 표와 쪽 가운데서 끝난 표는 다음 쪽의 같은 모양 표와 잇지 않는다 — 표는 자리가 모자랄 때만
    넘어간다. 큰 행이 통째로 밀려 앞 조각 아래가 조금 빈 것은 잇는다."""
    people = ("<table><tr><td>성명</td><td>소속</td><td>직위</td></tr><tr><td>홍길동</td><td>정보화팀</td><td>팀장</td></tr>"
              "<tr><td>김철수</td><td>정보화팀</td><td>주무관</td></tr></table>")
    blank = BUDGET.replace("</table>", '<tr><td colspan="3">- 이하 빈칸 -</td></tr>'
                           + "<tr><td></td><td></td><td></td></tr>" * 6 + "</table>")
    for name, last in (("이하 빈칸", ("table", (84, 150, 550, 790), blank)), ("쪽 가운데", ("table", (84, 150, 550, 420), BUDGET))):
        doc = _merge(_page(1, HEAD, ("text", (84, 100, 550, 140), "1. 구매 물품"), last, _num(1)),
                     _page(2, HEAD, ("table", (84, 60, 550, 200), people), _num(2)))
        t = doc.pages[1].blocks[1]
        assert (t.continues_from, t.header_rows, t.provenance) == (None, 1, None) and "<th>성명</th>" in t.html, name
    pushed = _merge(_page(1, HEAD, ("table", (84, 300, 550, 640), T), _num(1)),          # 아래 140pt 빔 — 밀린 큰 행
                    _page(2, HEAD, ("table", (84, 60, 550, 400), REST), _num(2)))
    assert pushed.pages[1].blocks[1].continues_from == "p1.b02"
    marked = _merge(_page(1, HEAD, ("table", (84, 100, 550, 420), T), _num(1)),         # 크게 비어도 '(계속)' 이라 적었으면 잇는다
                    _page(2, HEAD, ("text", (84, 45, 150, 57), "(계속)"), ("table", (84, 60, 550, 400), REST), _num(2)))
    assert marked.pages[1].blocks[2].continues_from == "p1.b02"


def test_spilled_row_stays_table_and_merges():
    """2열 표의 마지막 한 행만 다음 쪽으로 넘어간 조각은 배치용 표(글상자)로 풀지 않는다 — 앞 표와 잇는다."""
    two = ("<table><tr><td>구분</td><td>내용</td></tr><tr><td>목적</td><td>민원 접수 처리 시스템 고도화</td></tr>"
           "<tr><td>기간</td><td>2026. 3. ~ 2026. 12.</td></tr></table>")
    row = "<table><tr><td>기타</td><td>그 밖에 사업 추진에 필요한 세부 사항은 발주기관과 협의하여 따로 정한다</td></tr></table>"
    doc = _merge(_page(1, HEAD, ("text", (84, 500, 550, 540), "사업 개요는 다음과 같다."), ("table", (84, 560, 550, 790), two),
                       _num(1)),
                 _page(2, HEAD, ("table", (84, 60, 550, 100), row), ("text", (84, 120, 550, 160), "세부 일정은 붙임과 같다."),
                       _num(2)))
    t = doc.pages[1].blocks[1]
    assert (t.type, t.continues_from, t.text) == ("table", "p1.b03", None)
    other = _merge(_page(1, HEAD, ("table", (84, 560, 550, 790), T), _num(1)),          # 앞 쪽 끝 표와 열 수가 다르면 글상자
                   _page(2, HEAD, ("table", (84, 60, 550, 100), row), _num(2)))
    assert other.pages[1].blocks[1].type == "text"


def test_unnumbered_title_repeated_with_continued_mark():
    """번호 없는 표 제목('□ …')이 다음 쪽에서 '(계속)' 을 달고 되풀이되면 건너뛰고 잇는다. '(계속)' 이 없거나 글이 다르면
    잇지 않는다(번호가 없어 같은 표라는 증거가 약하다)."""
    def run(label, t1, t2):
        return _merge(_page(1, HEAD, (label, (84, 560, 400, 572), t1), ("table", (84, 575, 550, 790), BUDGET), _num(1)),
                      _page(2, HEAD, (label, (84, 60, 400, 72), t2), ("table", (84, 75, 550, 300), BUDGET2), _num(2)))
    for label, t1, t2 in (("paragraph_title", "□ 사업비 집행 현황", "□ 사업비 집행 현황(계속)"),
                          ("figure_title", "사업비 집행 현황", "사업비 집행 현황 (계속)")):
        doc = run(label, t1, t2)
        assert doc.pages[1].blocks[2].continues_from == "p1.b03" and doc.pages[1].blocks[1].keep, label   # 제목 반복은 남긴다
    for t2 in ("□ 사업비 집행 현황", "□ 인력 투입 현황(계속)"):
        assert run("paragraph_title", "□ 사업비 집행 현황", t2).pages[1].blocks[2].continues_from is None, t2


def test_repeated_header_with_ocr_noise():
    """스캔 OCR 이 되풀이한 머리행의 한 글자를 달리 읽어도 반복 머리행으로 센다 — tables 가 단 <th> 를 지키고, 데이터
    행은 완전 일치만 센다."""
    noisy = BUDGET2.replace("<td>집행액</td>", "<td>집행엑</td>", 1)
    doc = _merge(_page(1, HEAD, ("table", (84, 575, 550, 790), BUDGET), _num(1)),
                 _page(2, HEAD, ("table", (84, 60, 550, 300), noisy), _num(2)))
    t = doc.pages[1].blocks[1]
    assert (t.continues_from, t.header_rows, t.html.count("<th>")) == ("p1.b02", 1, 3)
    assert t.provenance == [{"rule": "merge_tables", "repeated_header_rows": 1}]


def test_failed_table_without_html_routed_to_image():
    """VLM 오류·빈 출력으로 HTML 이 없는 표도 이미지 갈래(D18) — 덜 망가진 표만 표시가 붙지 않게."""
    doc = _doc(("table", (0, 0, 100, 100), None), ("table", (0, 200, 100, 300), ""))
    doc.pages[0].blocks[0].error = "413: prompt too long"
    assert [(b.route, b.route_reason) for b in tables(doc).pages[0].blocks] == [("image", "failed")] * 2


def test_blank_text_blocks_do_not_block_merge():
    """공백뿐인 글 블록(VLM 이 빈 영역에서 줄바꿈만 낸 것)은 출력 블록이 아니라 쪽 끝·첫머리 판정에 끼지 않는다."""
    doc = _merge(_end_page(BUDGET, ("text", (84, 765, 520, 778), " \n ")),
                 _page(2, HEAD, ("text", (84, 50, 520, 58), " "), ("table", (84, 90, 550, 300), BUDGET2), _num(2)))
    assert doc.pages[1].blocks[2].continues_from == "p1.b04"


def test_layout_boxes_at_page_edge_not_spilled():
    """쪽 넘김 조각 예외(한 행만 넘어간 조각은 표로 둔다)는 한쪽만 글상자 꼴일 때만 — 쪽 경계의 두 글상자는 풀어 둔다(문단
    이음이 맡는다). 쪽 끝 글상자 다음 쪽의 제 머리행 있는 표는 잇지 않고 머리행을 지킨다(사진이 있어 풀지 못한 글상자도).
    '이하 빈칸' 으로 끝난 표 다음 쪽의 글상자도 푼다."""
    box1 = "<table><tr><td>※ 참고: 2025년 시범사업 결과를 반영하여 2026년부터 전국 단위로 확대 추진한다.</td></tr></table>"
    box2 = ("<table><tr><td>핵심 요약: 정부는 행정 업무의 디지털 전환을 위하여 공통 기반 시스템을 구축하고 단계적으로 "
            "확산한다.</td></tr></table>")
    doc = _merge(_page(1, HEAD, ("text", (84, 100, 550, 560), "사업 추진 현황은 다음과 같다."),
                       ("table", (84, 600, 550, 700), box1), _num(1)),
                 _page(2, HEAD, ("table", (84, 60, 550, 130), box2), ("text", (84, 150, 550, 200), "세부 내용은 붙임과 같다."),
                       _num(2)))
    a, b = doc.pages[0].blocks[2], doc.pages[1].blocks[1]
    assert (a.type, b.type, b.continues_from) == ("text", "text", None)
    assert a.provenance == b.provenance == [{"rule": "layout_table"}]
    sched = ("<table><tr><td>구분</td><td>일정</td></tr><tr><td>착수</td><td>2026. 3.</td></tr>"
             "<tr><td>완료</td><td>2026. 12.</td></tr></table>")
    box = "<table><tr><td>추진 배경</td><td>민원 접수 처리 방식이 기관마다 달라 문서 유통과 공동 활용에 한계가 있었다.</td></tr></table>"
    photo = box.replace("<td>민원", '<td><img src="imgs/p1.t01.png" alt="Image" />민원')
    for name, edge in (("글상자", box), ("사진 글상자", photo)):
        doc = _merge(_page(1, HEAD, ("text", (84, 100, 550, 560), "□ 사업 개요"), ("table", (84, 620, 550, 700), edge), _num(1)),
                     _page(2, HEAD, ("table", (84, 60, 550, 200), sched), _num(2)))
        t = doc.pages[1].blocks[1]
        assert (t.continues_from, t.header_rows, t.provenance) == (None, 1, None) and "<th>구분</th>" in t.html, name
    assert doc.pages[0].blocks[2].type == "table"                             # 사진은 잃지 않게 표로 둔다
    for edge in (box, photo):                                                 # 이어진다고 적었으면 잇는다
        doc = _merge(_page(1, HEAD, ("text", (84, 100, 550, 560), "□ 사업 개요"), ("table", (84, 620, 550, 700), edge), _num(1)),
                     _page(2, HEAD, ("text", (84, 45, 150, 57), "(계속)"), ("table", (84, 60, 550, 200), sched), _num(2)))
        assert doc.pages[1].blocks[2].continues_from == "p1.b03" and not doc.pages[1].blocks[1].keep
    head = "<tr><td>세부 추진 과제명과 주요 내용 요약</td><td>추진 일정 및 담당 기관</td></tr>"
    rest = ("<table>" + head + "<tr><td>공통 기반 시스템 구축과 기관별 단계적 확산</td><td>2026년 하반기 정보화팀</td></tr>"
            "<tr><td>민원 접수와 게시판 기능 통합</td><td>2027년 상반기 운영팀</td></tr></table>")
    doc = _merge(_page(1, HEAD, ("text", (84, 100, 550, 560), "□ 추진 과제"), ("table", (84, 740, 550, 790), f"<table>{head}</table>"),
                       _num(1)),
                 _page(2, HEAD, ("table", (84, 60, 550, 300), rest), _num(2)))    # 머리행만 쪽 끝에 남고 다음 쪽이 되풀이했다
    t = doc.pages[1].blocks[1]
    assert (doc.pages[0].blocks[2].type, t.continues_from, t.header_rows) == ("table", "p1.b03", 1)
    form = ("<table><tr><td>품명</td><td>비고</td></tr><tr><td>노트북</td><td>교체</td></tr>"
            '<tr><td colspan="2">- 이하 빈칸 -</td></tr>' + "<tr><td></td><td></td></tr>" * 6 + "</table>")
    notice = ("<table><tr><td>유의 사항</td><td>신청서는 기한 안에 소속 기관을 거쳐 제출하여야 하며 늦게 낸 신청서는 받지 "
              "않는다.</td></tr></table>")
    doc = _merge(_page(1, HEAD, ("text", (84, 100, 550, 140), "1. 구매 물품"), ("table", (84, 150, 550, 790), form), _num(1)),
                 _page(2, HEAD, ("table", (84, 60, 550, 120), notice), ("text", (84, 140, 550, 180), "문의는 담당 부서로 한다."),
                       _num(2)))
    b = doc.pages[1].blocks[1]
    assert (b.type, b.route, b.continues_from, b.provenance) == ("text", None, None, [{"rule": "layout_table"}])


def test_unnumbered_section_title_repeat_checks_room():
    """번호 없는 구역 제목 '□ …(계속)' 반복은 같은 표라는 증거가 약하다 — 쪽 가운데서 끝난 표는 다음 쪽의 같은 모양 다른
    표와 잇지 않고 그 표의 머리행을 지킨다. 번호 있는 표 제목 반복('<표 3> …(계속)')은 빈자리를 보지 않고 잇는다."""
    result = ("<table><tr><td>과제</td><td>목표</td><td>실적</td></tr><tr><td>민원 접수 고도화</td><td>100%</td><td>95%</td></tr>"
              "<tr><td>문서관리 통합</td><td>3개 기관</td><td>3개 기관</td></tr></table>")

    def run(label, t1, t2):
        return _merge(_page(1, HEAD, (label, (84, 90, 400, 102), t1), ("table", (84, 105, 550, 400), result), _num(1)),
                      _page(2, HEAD, (label, (84, 60, 400, 72), t2), ("table", (84, 75, 550, 300), BUDGET),
                            ("text", (84, 320, 550, 360), "인건비 집행률은 계획 대비 80% 수준이다."), _num(2)))
    t = run("paragraph_title", "□ 추진 실적", "□ 추진 실적(계속)").pages[1].blocks[2]
    assert (t.continues_from, t.header_rows, t.provenance) == (None, 1, None) and "<th>구분</th>" in t.html
    numbered = run("figure_title", "<표 3> 추진 실적", "<표 3> 추진 실적(계속)")
    assert numbered.pages[1].blocks[2].continues_from == "p1.b03"


def test_unnumbered_title_with_unit_line_between():
    """[번호 없는 제목 / '(단위: 천원)' 한 줄 / 표] 꼴 — 단위 줄을 건너 그 위 제목을 앞 표 제목 자리로 보고, 다음 쪽의 같은 꼴
    '(계속)' 반복을 건너뛰어 잇는다."""
    doc = _merge(_page(1, HEAD, ("paragraph_title", (84, 540, 400, 552), "□ 사업비 집행 현황"),
                       ("text", (470, 558, 550, 570), "(단위: 천원)"), ("table", (84, 575, 550, 790), BUDGET), _num(1)),
                 _page(2, HEAD, ("paragraph_title", (84, 40, 400, 52), "□ 사업비 집행 현황(계속)"),
                       ("text", (470, 58, 550, 70), "(단위: 천원)"), ("table", (84, 75, 550, 300), BUDGET2), _num(2)))
    t = doc.pages[1].blocks[3]
    assert (t.continues_from, t.header_rows) == ("p1.b04", 1)
    other = _merge(_page(1, HEAD, ("paragraph_title", (84, 540, 400, 552), "□ 사업비 집행 현황"),   # 제목 글이 다르면 잇지 않는다
                         ("text", (470, 558, 550, 570), "(단위: 천원)"), ("table", (84, 575, 550, 790), BUDGET), _num(1)),
                   _page(2, HEAD, ("paragraph_title", (84, 40, 400, 52), "□ 인력 투입 현황(계속)"),
                         ("text", (470, 58, 550, 70), "(단위: 천원)"), ("table", (84, 75, 550, 300), BUDGET2), _num(2)))
    assert other.pages[1].blocks[3].continues_from is None


def test_appendage_lines_and_references():
    """부속 줄(제목·주·'이하 빈칸')과 표·그림을 가리키는 본문 — 번호 뒤에 한글·다른 번호·접속이 오면 본문이다."""
    for s in ("주: 2026년 6월 말 기준", "주1) 집행액은 6월 말 기준", "이하 빈칸", "표 3. 사업비 집행 현황", "그림 3. 추진 체계도",
              "사진 2 현장 점검 모습", "<표 1> 단계별 추진 기관 (단위: 개)", "[그림 3] 조직 구성도", "표 3-1 세부 내역",
              "표 3 과제별 추진 현황", "자료: 기획재정부"):
        assert appendage(s), s
    for s in ("표 1, 2와 같이 사업비 집행률은", "표 1~3은 사업비 집행 현황을", "그림 2·3 처럼 시스템은", "표 3 및 그림 2에서 보듯이",
              "표 3과 같이 집행률은", "<표 3>의 세부 내역은", "[표 1], [표 2]와 같이", "표 3 과 같이 집행률은", "표 3ㆍ4 는"):
        assert not appendage(s), s


def test_bracket_title():
    from kdp.assemble.tables import bracket_title
    assert bracket_title("[ 위험지역의 최근 4주간 기상요인 변화 ]") and bracket_title("<말라리아 주의보 및 경보 발령 기준>")
    assert not bracket_title("[참고]") and not bracket_title("<붙임>") and not bracket_title("(단위: 천원)")
    assert not bracket_title("[단위: 천원]") and not bracket_title("[ 제목 ] 뒤에 글") and not bracket_title("(계속)")


def test_bracket_title_captions_table_above_only():
    doc = tables(_doc(("text", (180, 100, 420, 114), "<말라리아 경보 발령 기준>"), ("table", (60, 120, 540, 200), T),
                      ("text", (180, 205, 420, 219), "[ 아래 줄 괄호 제목 ]"), ("table", (60, 400, 540, 480), T),
                      ("text", (60, 380, 120, 394), "[참고]")))
    b = doc.pages[0].blocks
    assert b[1].caption_id == "p1.b01"
    assert b[3].caption_id is None                    # 위 '[참고]' 는 꼬리표, 앞 표 바로 아래 괄호 줄은 받지 않는다


def test_figure_caption_prefers_bracket_title_above():
    doc = tables(_doc(("figure_title", (180, 100, 420, 114), "[ 국내 환자 주별 발생 추이 ]"),
                      ("chart", (60, 118, 540, 300), None),
                      ("figure_title", (280, 304, 310, 314), "주차"),                     # 그래프 바로 아래 축 이름
                      ("figure_title", (100, 400, 230, 412), "얼룩날개모기 암컷"),          # 괄호 없는 그림 제목만
                      ("image", (100, 416, 230, 500), None),
                      ("text", (240, 400, 520, 412), "[ 머리와 날개 특징 ]"),
                      ("image", (240, 416, 370, 500), None), ("image", (380, 416, 520, 500), None),
                      ("image", (60, 600, 90, 630), None)))                               # 제목 없는 작은 그림
    b = {x.id: x for x in doc.pages[0].blocks}
    assert b["p1.b02"].caption_id == "p1.b01" and b["p1.b05"].caption_id == "p1.b04"
    assert b["p1.b07"].caption_id == b["p1.b08"].caption_id == "p1.b06"             # 나란한 두 그림이 한 제목을 함께
    assert b["p1.b09"].caption_id is None


def test_bracket_tags_and_heading_labels_are_not_captions():
    for typ, text in [("text", "[별지 제1호서식]"), ("text", "[붙임 1-1]"), ("text", "<별첨 2-1>"), ("text", "[참고자료]"),
                      ("paragraph_title", "【 추진 실적 】")]:
        doc = tables(_doc((typ, (60, 100, 300, 114), text), ("table", (60, 120, 540, 200), T)))
        assert doc.pages[0].blocks[1].caption_id is None, text
    doc = tables(_doc(("doc_title", (150, 100, 450, 114), "「 2026년 정보보안 기본계획 」"), ("image", (100, 120, 500, 300), None)))
    assert doc.pages[0].blocks[1].caption_id is None                          # 문서 제목은 그림 제목이 아니다


def test_figure_caption_order():
    doc = tables(_doc(("text", (180, 100, 420, 114), "【 추진 체계 】"), ("image", (100, 120, 500, 300), None),
                      ("figure_title", (180, 304, 420, 316), "<그림 1> 추진 체계도")))
    assert doc.pages[0].blocks[1].caption_id == "p1.b03"                     # 아래 '<그림 n>' 이 위 괄호 줄보다 먼저
    doc = tables(_doc(("image", (100, 100, 500, 200), None), ("figure_title", (180, 204, 420, 216), "<그림 1> 시스템 구성도"),
                      ("image", (100, 220, 500, 320), None), ("figure_title", (180, 324, 420, 336), "<그림 2> 데이터 흐름도")))
    b = doc.pages[0].blocks
    assert (b[0].caption_id, b[2].caption_id) == ("p1.b02", "p1.b04")          # 쌓인 그림은 제 아래 제목
    doc = tables(_doc(("text", (100, 100, 500, 114), "<그림 2>는 최근 5년간 환자 수를 나타낸 것이다."),
                      ("chart", (100, 120, 500, 300), None), ("figure_title", (180, 304, 420, 316), "<그림 2> 연도별 환자 수")))
    assert doc.pages[0].blocks[1].caption_id == "p1.b03"                     # 그림을 가리키는 본문 문장은 제목이 아니다


def test_numbered_section_title_beside_figure_is_not_caption():
    doc = tables(_doc(("figure_title", (60, 100, 540, 114), "2. 시스템 구성"), ("text", (60, 122, 290, 300), "왼쪽 단 글"),
                      ("image", (300, 122, 540, 300), None)))
    assert doc.pages[0].blocks[2].caption_id is None                          # 읽기 순서로 사이에 글이 있다
    doc = tables(_doc(("text", (60, 100, 540, 114), "【단위: 천원】"), ("table", (60, 120, 540, 200), T),
                      ("text", (60, 300, 540, 314), "<개정 2024. 1. 1.>"), ("table", (60, 320, 540, 400), T)))
    assert [b.caption_id for b in doc.pages[0].blocks if b.type == "table"] == [None, None]   # 괄호 안 단위·개정 표시


def test_header_rows_numeric_two_level_and_contact():
    # 숫자 표는 머리글이 본문보다 길어도 머리행, 가로 병합 아래 칸 이름 행도 머리행(실측 test3.pdf). 전화가 든 첫 행은 자료 행
    long = ("<table><tr><td>요소(단위)</td><td>2026년 9월(a)</td><td>9월 평년값 (1991-2020) (c)</td></tr>"
            "<tr><td>평균기온(℃)</td><td>20.6</td><td>19.6</td></tr><tr><td>강수량(mm)</td><td>11.2</td><td>140.2</td></tr></table>")
    assert header_rows(parse_rows(long)) == 1
    two = ('<table><tr><td colspan="3">지점</td><td colspan="2">1위</td></tr>'
           "<tr><td>번호</td><td>지점명</td><td>관측개시</td><td>연도</td><td>값</td></tr>"
           "<tr><td>131</td><td>청주</td><td>1967.01.01</td><td>2024</td><td>26.4</td></tr></table>")
    assert header_rows(parse_rows(two)) == 2
    cells = ('<table><tr><td rowspan="2">담당 부서</td><td rowspan="2">기후서비스과</td><td>책임자</td><td>과장</td>'
             "<td>홍길동</td><td>(043-123-4567)</td></tr>"
             "<tr><td>담당자</td><td>주무관</td><td>김철수</td><td>(043-123-4568)</td></tr></table>")
    joined = ('<table><tr><td rowspan="2">담당 부서</td><td rowspan="2">기후서비스과</td><td>책임자</td>'
              "<td>과장 홍길동 (043-123-4567)</td></tr><tr><td>담당자</td><td>주무관 김철수 (043-123-4568)</td></tr></table>")
    assert header_rows(parse_rows(cells)) == header_rows(parse_rows(joined)) == 0


def test_figure_parts_join_figure():
    # 그래프 그림 맨 위 줄(글자층 없음)을 검출기가 그림 밖 그림 제목으로, 패널 표지 '(a)' 를 그림 바로 위 줄로 잡았다(실측
    # test3.pdf). '【그림 n】' 제목과 그림 틀이 덮은 글자층 본문 줄은 그대로
    doc = _doc(("figure_title", (120, 456, 512, 471), "상순(8.5mm, 17.5%) 중순(0.0mm, 0.0%) 하순(2.7mm, 7.2%)"),
               ("chart", (78, 473, 518, 651), None),
               ("figure_title", (162, 659, 428, 675), "【그림 2】 2026년 9월 일별 충북 강수량 시계열"),
               ("figure_title", (72, 276, 86, 289), "(a)"), ("image", (64, 291, 296, 457), None),
               ("text", (90, 600, 300, 640), "그림 틀 아래로 이어지는 본문 문단"))
    doc.pages[0].blocks[0].provenance = [{"rule": "no_layer_chars"}]          # 글자층 있는 쪽인데 그 자리에 글자가 없다
    b = tables(doc).pages[0].blocks
    assert (b[0].merged_into, b[3].merged_into) == ("p1.b02", "p1.b05")
    assert b[1].figure_text.startswith("상순") and b[1].caption_id == "p1.b03" and b[4].figure_text == "(a)"
    assert b[5].keep and b[5].merged_into is None


def test_figure_title_with_layer_chars_stays_caption():
    # 모델이 글자층과 똑같이 읽은 그림 제목(표시 없음)·잘린 글 블록은 그림 글로 옮기지 않는다(검토 지적)
    doc = _doc(("chart", (78, 300, 518, 450), None), ("figure_title", (180, 453, 420, 466), "연도별 9월 강수량 추이"),
               ("text", (120, 320, 400, 340), "아 아 아 아 아 아 아 아"))
    doc.pages[0].blocks[2].provenance, doc.pages[0].blocks[2].truncated = [{"rule": "no_layer_chars"}], "length"
    b = tables(doc).pages[0].blocks
    assert b[0].caption_id == "p1.b02" and b[0].figure_text is None
    assert b[1].merged_into is None and b[2].merged_into is None and b[2].keep


def test_numbered_figure_caption_a_bit_far():
    # 지도 두 장과 '【그림 1】' 사이 16pt — 꼴이 분명한 그림 제목은 30pt 까지 잇는다(실측 test3.pdf). 라벨뿐인 줄은 15pt 그대로
    doc = tables(_doc(("chart", (62, 142, 288, 402), None), ("chart", (298, 142, 526, 402), None),
                      ("figure_title", (117, 418, 472, 434), "【그림 1】 2026년 9월 충북 평균기온 분포도"),
                      ("image", (100, 500, 500, 600), None), ("figure_title", (150, 620, 450, 632), "강수량 분포")))
    b = doc.pages[0].blocks
    assert b[0].caption_id == b[1].caption_id == "p1.b03" and b[3].caption_id is None


def test_header_rows_with_blank_corner_and_repeated_sublabels():
    # 대각선 머리칸 글을 모델이 둘째 행에 두어 첫 행 첫 칸이 비고, 둘째 행은 묶음마다 '21·22' 가 되풀이된다(실측 test2.pdf GPU)
    rep = ('<table><tr><td></td><td colspan="3">평균기온(℃)</td><td colspan="3">일강수량(mm)</td></tr>'
           "<tr><td>구분 주차</td><td>21</td><td>22</td><td>23</td><td>21</td><td>22</td><td>23</td></tr>"
           "<tr><td>평년</td><td>18.4</td><td>19.4</td><td>20.2</td><td>2.9</td><td>0.6</td><td>2.6</td></tr></table>")
    assert header_rows(parse_rows(rep)) == 2
    zeros = ('<table><tr><td>구분</td><td colspan="2">상반기</td><td colspan="2">하반기</td></tr>'
             "<tr><td>사업 수</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>"
             "<tr><td>예산</td><td>10</td><td>20</td><td>30</td><td>40</td></tr></table>")
    assert header_rows(parse_rows(zeros)) == 1                         # 값이 같은 자료 행은 칸 이름 행이 아니다


def test_invented_figure_text_dropped():
    # 글 없는 만화 그림에서 그림 속 글자 읽기가 지어낸 글(실측 test2.pdf 8쪽) — 길이 한도로 잘렸거나 한글·영문·숫자가 없으면 버린다
    doc = _doc(("image", (309, 548, 352, 594), None), ("image", (239, 597, 284, 643), None), ("image", (266, 591, 330, 626), None))
    a, b, c = doc.pages[0].blocks
    a.figure_text, a.truncated = "ふ～～～", "length"
    b.figure_text, c.figure_text = "うっ", "OPEN 공공누리 2026"
    a, b, c = tables(doc).pages[0].blocks
    assert (a.figure_text, b.figure_text, c.figure_text) == (None, None, "OPEN 공공누리 2026")
    assert (a.truncated, a.provenance) == (None, [{"rule": "figure_text_noise", "before": "ふ～～～", "truncated": "length"}])


def test_header_rows_time_quarter_and_data_rows():
    # 시각·분기 머리글은 머리행, 병합 머리글 아래 행 이름이 있는 글 자료행·우연히 같은 값 자료행은 둘째 머리행이 아니다(검토 지적)
    time = "<table><tr><td>구분</td><td>09:00</td><td>12:00</td></tr><tr><td>기온</td><td>20.1</td><td>24.3</td></tr></table>"
    quarter = "<table><tr><td>구분</td><td>1/4</td><td>2/4</td></tr><tr><td>매출</td><td>10</td><td>20</td></tr></table>"
    text = ('<table><tr><td>구분</td><td colspan="2">주요 내용</td></tr><tr><td>목표</td><td>디지털 전환</td><td>데이터 개방</td></tr>'
            "<tr><td>예산</td><td>10</td><td>20</td></tr></table>")
    same = ('<table><tr><td>구분</td><td colspan="2">상반기</td><td colspan="2">하반기</td></tr>'
            "<tr><td>사업 수</td><td>1</td><td>0</td><td>1</td><td>0</td></tr><tr><td>예산</td><td>10</td><td>20</td><td>30</td><td>40</td></tr></table>")
    assert [header_rows(parse_rows(h)) for h in (time, quarter, text, same)] == [1, 1, 1, 1]


def test_figure_text_kept_unless_only_kana_or_symbols():
    # 잘린 그림 글이라도 글자가 있으면 둔다(QC 가 잘림으로 센다), 한자·전각 숫자·그리스 문자도 글 — 가나·기호뿐인 글만 버린다(검토 지적)
    doc = _doc(*[("image", (60, 60 + 60 * i, 200, 110 + 60 * i), None) for i in range(5)])
    texts = ["2026 국민 안전 수칙", "國立氣象科學院", "１２３", "α β γ", "ぐっぴっっっっ"]
    for b, t in zip(doc.pages[0].blocks, texts):
        b.figure_text = t
    doc.pages[0].blocks[0].truncated = "length"
    out = tables(doc).pages[0].blocks
    assert [b.figure_text for b in out] == texts[:4] + [None]
    assert out[0].truncated == "length"


def test_header_rows_title_row_and_repeated_values():
    # 표 전체를 덮는 제목 칸 아래 항목-값 개요표, 묶음마다 우연히 같은 범주 값('적합·해당 없음')이 든 첫 자료 행은 머리행이
    # 아니다. '요소 | 평균기온(2칸) …' 아래 '순위 | 연도 | 값 …' 은 머리행(실측 test3.pdf 9쪽)(검토 지적)
    kv = ('<table><tr><td colspan="2">사업 개요</td></tr><tr><td>사업명</td><td>지능형 민원 상담 시스템 구축</td></tr>'
          "<tr><td>사업기간</td><td>2026. 3. ~ 2026. 12.</td></tr><tr><td>사업비</td><td>1,200</td></tr></table>")
    cat = ('<table><tr><td>기관명</td><td colspan="2">1차 점검</td><td colspan="2">2차 점검</td></tr>'
           "<tr><td>가기관</td><td>적합</td><td>해당 없음</td><td>적합</td><td>해당 없음</td></tr>"
           "<tr><td>나기관</td><td>부적합</td><td>적합</td><td>적합</td><td>적합</td></tr></table>")
    rank = ('<table><tr><td>요소</td><td colspan="2">평균기온(℃)</td><td colspan="2">강수량(mm)</td></tr>'
            "<tr><td>순위</td><td>연도</td><td>값</td><td>연도</td><td>값</td></tr>"
            "<tr><td>1</td><td>2024</td><td>24.2</td><td>2007</td><td>358.9</td></tr>"
            "<tr><td>2</td><td>2025</td><td>22.2</td><td>1990</td><td>300.4</td></tr>"
            "<tr><td>3</td><td>1998</td><td>21.9</td><td>1985</td><td>284.0</td></tr></table>")
    assert [header_rows(parse_rows(h)) for h in (kv, cat, rank)] == [1, 1, 2]


def test_caption_above_convention_keeps_near_title():
    # 제목을 그림 위에 다는 문서에서 그림 아래 쪽 끝에 다음 그림 제목만 남았다(그림은 다음 쪽) — 15pt 밖 '<그림 n>' 은 15pt 안
    # 후보가 있으면 쓰지 않는다. 괄호 제목 문서도 같다(검토 지적)
    for own in ("[그림 1] 연도별 출생아 수 추이", "[ 연도별 출생아 수 추이 ]"):
        doc = _doc(("figure_title", (100, 100, 495, 114), own), ("image", (100, 119, 495, 400), None),
                   ("text", (100, 403, 300, 413), "자료: 통계청"), ("figure_title", (100, 422, 495, 436), "<그림 2> 시도별 출생아 수"))
        assert tables(doc).pages[0].blocks[1].caption_id == "p1.b01"


def test_stacked_panel_labels_go_to_nearest_figure():
    # 표지를 그림 위에 단 그림 두 장을 위아래로 붙였다 — '(나)' 는 위 그림 5pt 아래·아래 그림 2pt 위라 아래 그림의 것이다.
    # 거리가 같으면 어느 그림 것인지 몰라 그대로 둔다(검토 지적)
    def doc(g):
        return _doc(("figure_title", (72, 100, 87, 113), "(가)"), ("image", (64, 115, 296, 280), None),
                    ("figure_title", (72, 280 + g, 87, 293 + g), "(나)"), ("image", (64, 295 + g, 296, 460 + g), None))
    b = tables(doc(5)).pages[0].blocks
    assert (b[0].merged_into, b[2].merged_into, b[1].figure_text, b[3].figure_text) == ("p1.b02", "p1.b04", "(가)", "(나)")
    b = tables(doc(2)).pages[0].blocks
    assert (b[2].keep, b[2].merged_into, b[1].figure_text) == (True, None, "(가)")


def test_drawn_title_alone_is_caption():
    # 글자층 글자가 없는 그림 바로 아래 제목 줄(다른 보고서 그래프를 제목째 캡처)은 그 그림에 번호·괄호 제목이 따로 없으면
    # 그림 글이 아니라 그림 제목이다(검토 지적)
    doc = _doc(("chart", (78, 100, 518, 300), None), ("figure_title", (150, 303, 450, 317), "연도별 출생아 수 추이"))
    doc.pages[0].blocks[1].provenance = [{"rule": "no_layer_chars"}]
    b = tables(doc).pages[0].blocks
    assert (b[1].merged_into, b[0].caption_id) == (None, "p1.b02")


def test_absorbed_lines_are_not_captions():
    # 그림 글로 묶인 줄을 다른 그림이 제목으로 잇지 않는다(출력에 없는 블록을 가리킴). 묶인 줄이 그림과 축 이름 사이에 있었으면
    # 축 이름도 제목으로 잇지 않는다(검토 지적)
    doc = _doc(("image", (64, 100, 200, 200), None), ("figure_title", (80, 203, 400, 215), "일강수량(mm) 누적강수량(mm)"),
               ("chart", (64, 218, 520, 400), None), ("figure_title", (150, 405, 450, 419), "【그림 1】 9월 일별 강수량"))
    doc.pages[0].blocks[1].provenance = [{"rule": "no_layer_chars"}]
    b = tables(doc).pages[0].blocks
    assert (b[1].merged_into, b[0].caption_id, b[2].caption_id) == ("p1.b03", None, "p1.b04")
    doc = _doc(("image", (64, 100, 520, 300), None), ("text", (300, 110, 500, 124), "2025년 2026년 평년"),
               ("figure_title", (280, 308, 320, 320), "주차"))
    doc.pages[0].blocks[1].provenance = [{"rule": "no_layer_chars"}]
    b = tables(doc).pages[0].blocks
    assert (b[1].merged_into, b[0].caption_id) == ("p1.b01", None)
