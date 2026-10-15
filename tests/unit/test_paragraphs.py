"""쪽 넘김 문단 이음(기획서 4.3 규칙 2) — 잘못 잇지 않는 쪽을 우선한다."""
from kdp.assemble.figures import join_wrapped
from kdp.assemble.paragraphs import join_paragraphs
from kdp.classify.blocks import classify_blocks
from kdp.config import Classify
from kdp.schema import Block, Document, NUp, Page, ParserInfo, Source

HEAD = ("header", (70, 40, 520, 55), "○○ 사업 결과 보고서")
OPEN = "시스템은 민원 접수 처리를 돕고 일정 관리와"
NEXT = "문서관리 기능을 하나로 묶어 제공한다."


def _page(no, *blocks, size=(595, 842), sheet=None, n_up=None):
    return Page(page_no=no, sheet_no=sheet or no, size=size, n_up=n_up, blocks=[
        Block(id=f"p{no}.b{i + 1:02d}", type=t, bbox=bb, text=x if t != "table" else None, html=x if t == "table" else None)
        for i, (t, bb, x) in enumerate(blocks)])


def _doc(*pages):
    return Document(doc_id="sha256:ab", source=Source(paths=["a.pdf"], pages=len(pages)),
                    parser=ParserInfo(kdp_version="0", engine={}, engine_hash="e", assemble_hash="a", run_at="t"),
                    pages=list(pages))


def _join(*pages):
    return join_paragraphs(classify_blocks(_doc(*pages), Classify()))


def _pair(last, first, *tail):
    """1쪽 끝 블록 last 와 2쪽 첫 블록 first 사이에 머리말·쪽번호를 둔 두 쪽."""
    return _join(_page(1, HEAD, ("text", (70, 100, 520, 400), "사업 개요는 다음과 같다."), last, *tail,
                       ("number", (290, 800, 305, 812), "- 1 -")),
                 _page(2, HEAD, first, ("number", (290, 800, 305, 812), "- 2 -")))


def test_joined_across_pages():
    doc = _pair(("text", (70, 700, 520, 790), OPEN), ("text", (70, 70, 520, 120), NEXT))
    a, b = doc.pages[0].blocks[2], doc.pages[1].blocks[1]
    assert b.continues_from == "p1.b03" and b.provenance == [{"rule": "join_paragraphs"}]
    assert (a.text, b.text, a.continues_from) == (OPEN, NEXT, None)            # 글자는 합치지 않는다(렌더러가 잇는다)
    assert doc.pages[0].blocks[1].continues_from is None                       # 같은 쪽 안 블록은 잇지 않는다


def test_footnote_at_page_end_skipped():
    doc = _pair(("text", (70, 650, 520, 740), OPEN), ("text", (70, 70, 520, 120), NEXT),
                ("footnote", (70, 760, 520, 790), "1) 행정안전부 고시 제2026-1호"))
    assert doc.pages[1].blocks[1].continues_from == "p1.b03"


def test_not_joined():
    para = ("text", (70, 700, 520, 790), OPEN)
    cases = {
        "종결 표현": (("text", (70, 700, 520, 790), "민원 접수 기능을 구축하였다."), ("text", (70, 70, 520, 120), NEXT)),
        "개조식 끝": (("text", (70, 700, 520, 790), "2026년 하반기 착수 예정"), ("text", (70, 70, 520, 120), NEXT)),
        "앞이 항목": (("text", (70, 700, 520, 790), "○ 정보화 사업 추진 배경"), ("text", (70, 70, 520, 120), NEXT)),
        "앞 끝 줄이 항목": (("text", (70, 700, 520, 790), "추진 내용은 아래와 같다.\n○ 시스템 통합 구축"),
                       ("text", (70, 70, 520, 120), NEXT)),
        "□ 항목": (para, ("text", (70, 70, 520, 120), "□ 추진 배경")),
        "- 항목": (para, ("text", (70, 70, 520, 120), "- 민원 접수 고도화")),
        "번호": (para, ("text", (70, 70, 520, 120), "1. 사업 개요")),
        "가.": (para, ("text", (70, 70, 520, 120), "가. 추진 일정")),
        "①": (para, ("text", (70, 70, 520, 120), "① 문서관리")),
        "제목 번호": (para, ("text", (70, 70, 520, 120), "Ⅲ. 추진 결과")),
        "제목 블록": (para, ("paragraph_title", (70, 70, 520, 90), "추진 결과")),
        "(계속)": (para, ("text", (70, 70, 150, 82), "(계속)")),
        "표": (para, ("table", (70, 70, 520, 300), "<table><tr><td>구분</td><td>내용</td></tr></table>")),
    }
    for name, (last, first) in cases.items():
        doc = _pair(last, first)
        assert all(b.continues_from is None for p in doc.pages for b in p.blocks), name


def test_levels_from_headings_and_lists_block_join():
    plain = _doc(_page(1, ("text", (70, 700, 520, 790), OPEN)), _page(2, ("text", (70, 70, 520, 120), NEXT)))
    assert join_paragraphs(plain).pages[1].blocks[0].continues_from == "p1.b01"
    for side, field in ((0, "list_level"), (1, "list_level"), (1, "heading_level"), (0, "heading_level")):
        doc = _doc(_page(1, ("text", (70, 700, 520, 790), OPEN)), _page(2, ("text", (70, 70, 520, 120), NEXT)))
        setattr(doc.pages[side].blocks[0], field, 2)
        assert join_paragraphs(doc).pages[1].blocks[0].continues_from is None, (side, field)


def test_dropped_page_between_and_chain():
    middle = _page(2, ("text", (70, 60, 520, 790), "보고 체계를 하나로 묶고 결재 단계를 줄여"))
    doc = _join(_page(1, ("text", (70, 700, 520, 790), OPEN)), middle, _page(3, ("text", (70, 70, 520, 120), NEXT)))
    assert [p.blocks[0].continues_from for p in doc.pages] == [None, "p1.b01", "p2.b01"]   # 세 쪽 사슬
    middle.keep = False                                                         # 버린 쪽(목차·간지 등)을 넘어 잇지 않는다
    doc = _join(_page(1, ("text", (70, 700, 520, 790), OPEN)), middle, _page(3, ("text", (70, 70, 520, 120), NEXT)))
    assert all(p.blocks[0].continues_from is None for p in doc.pages)


def test_n_up_logical_pages():
    sheet = (842, 595)
    left = NUp(cols=2, rows=1, cell=(0, 0), region=(0, 0, 421, 595))
    right = NUp(cols=2, rows=1, cell=(0, 1), region=(421, 0, 842, 595))
    doc = _join(_page(1, ("text", (30, 400, 390, 560), OPEN), ("number", (200, 570, 220, 580), "- 1 -"),
                      size=sheet, sheet=1, n_up=left),
                _page(2, ("header", (450, 15, 800, 25), "○○ 사업 결과 보고서"), ("text", (451, 40, 811, 100), NEXT),
                      size=sheet, sheet=1, n_up=right))
    assert doc.pages[1].blocks[1].continues_from == "p1.b01"                   # 같은 시트의 왼쪽 → 오른쪽 논리 페이지


def test_table_and_figure_appendages_not_joined():
    """쪽 끝 표·그림 아래 부속 줄(출처·주·제목·'이하 빈칸')은 명사로 끝나도 본문이 아니다 — 다음 쪽 문단과 잇지 않는다."""
    table = ("table", (84, 575, 550, 760), "<table><tr><td>구분</td><td>집행액</td></tr><tr><td>인건비</td><td>1,200</td></tr></table>")
    image = ("image", (84, 575, 550, 760), None)
    nxt = ("text", (84, 60, 550, 100), "인건비 집행률은 계획 대비 80% 수준이며 하반기에 나머지를 집행한다.")
    for above, note in ((table, "자료: 기획재정부"), (table, "출처: 행정안전부 내부 집계"), (table, "주: 2026년 6월 말 기준"),
                        (table, "주1) 집행액은 6월 말 기준"), (table, "이하 빈칸"), (table, "표 3. 사업비 집행 현황"),
                        (image, "그림 3. 추진 체계도"), (image, "사진 2 현장 점검 모습")):
        doc = _pair(above, nxt, ("text", (84, 762, 300, 772), note))
        assert doc.pages[1].blocks[1].continues_from is None, note
    linked = _doc(_page(1, table, ("text", (84, 762, 300, 772), "기획재정부 내부 집계")), _page(2, nxt))
    linked.pages[0].blocks[0].note_ids = ["p1.b02"]                            # tables 가 이은 출처 줄 — 꼴이 낯설어도
    assert join_paragraphs(linked).pages[1].blocks[0].continues_from is None
    ref = _pair(("text", (70, 700, 520, 790), "그림 3과 같이 시스템은 민원 접수 처리를 돕고 일정 관리와"),
                ("text", (70, 70, 520, 120), NEXT))
    assert ref.pages[1].blocks[1].continues_from == "p1.b03"                  # 그림을 가리키는 본문은 잇는다


def test_wrapped_paragraph_at_page_end_joined():
    """그림 감싼 문단이 쪽 끝에서 다음 쪽으로 이어지면 묶음 대표를 가리킨다 — 읽기 순서가 [옆 글, 그림, 아래 글] 이나
    [옆 글, 아래 글, 그림] 이어도(감싼 그림은 쪽 끝으로 보지 않는다). 감싸지 않은 그림이 쪽 끝이면 잇지 않는다."""
    side = ("text", (70, 600, 310, 740), "시스템은 민원 접수 처리를 돕고 기관 내부의 여러 부서가 함께 쓰는 공통 기반으로서")
    fig = ("image", (320, 600, 520, 740), None)
    below = ("text", (70, 745, 520, 790), "민원 접수와 일정 관리, 게시판 기능을 하나로 묶고 일정 관리와")
    for order in ((side, fig, below), (side, below, fig), (fig, side, below)):
        doc = _doc(_page(1, *order), _page(2, ("text", (70, 70, 520, 120), NEXT)))
        next(b for b in doc.pages[0].blocks if b.type == "image").img_path = "imgs/fig.png"
        doc = join_paragraphs(join_wrapped(doc))
        rep = next(b for b in doc.pages[0].blocks if b.provenance)
        assert rep.text == side[2] + " " + below[2] and doc.pages[1].blocks[0].continues_from == rep.id, order
    alone = _doc(_page(1, ("text", (70, 600, 520, 740), OPEN), fig), _page(2, ("text", (70, 70, 520, 120), NEXT)))
    alone.pages[0].blocks[1].img_path = "imgs/fig.png"
    assert join_paragraphs(join_wrapped(alone)).pages[1].blocks[0].continues_from is None


def test_blank_text_blocks_skipped():
    """공백뿐인 글 블록은 출력 블록이 아니라 쪽 끝·첫머리로 보지 않는다."""
    doc = _join(_page(1, ("text", (70, 700, 520, 780), OPEN), ("text", (70, 782, 520, 790), " \n ")),
                _page(2, ("text", (70, 50, 520, 58), " "), ("text", (70, 70, 520, 120), NEXT)))
    assert doc.pages[1].blocks[1].continues_from == "p1.b01"


def test_body_referring_to_several_tables_joined():
    """'표 1, 2와 같이'·'표 1~3은'·'그림 2·3 처럼'·'표 3 및 그림 2에서' 처럼 여러 표·그림을 가리키는 본문은 부속 줄이
    아니다 — 다음 쪽 문단과 잇는다."""
    for s in ("표 1, 2와 같이 사업비 집행률은 계획 대비 80% 수준이며 하반기에",
              "표 1~3은 사업비 집행 현황을 보여 주며 인건비 집행률은 계획 대비 80% 수준으로 하반기에",
              "그림 2·3 처럼 시스템은 민원 접수 처리를 돕고 하반기에",
              "표 3 및 그림 2에서 보듯이 집행률은 계획 대비 80% 수준이며 하반기에"):
        doc = _pair(("text", (70, 700, 520, 790), s), ("text", (70, 70, 520, 120), "나머지 예산을 집행할 계획이다."))
        assert doc.pages[1].blocks[1].continues_from == "p1.b03", s


def test_title_label_sentence_not_joined_as_next():
    """검출기가 제목으로 본 줄(문장 꼴이라 classify_blocks 가 text 로 바꾼 제목 라벨)은 새 항목·문단의 시작 — 앞 쪽 끝 글에
    잇지 않는다. 같은 글이 본문 라벨이면 잇는다."""
    tail = "2025년에는 착수 단계와 중간 단계의 산출물 관리 체계를 정비하고 점검 결과를 부서별로 공유"
    line = "하반기 산출물 보완 현황은 아래와 같음"
    for prov, want in (([{"rule": "title_sentence", "type": "paragraph_title"}], None), (None, "p1.b01")):
        doc = _doc(_page(1, ("text", (70, 600, 520, 640), tail)), _page(2, ("text", (70, 60, 520, 80), line)))
        doc.pages[1].blocks[0].provenance = prov
        assert join_paragraphs(doc).pages[1].blocks[0].continues_from == want, prov
    labeled = _pair(("text", (70, 700, 520, 790), tail), ("paragraph_title", (70, 60, 520, 80), line))   # 실제 분류를 거쳐도
    assert labeled.pages[1].blocks[1].continues_from is None
