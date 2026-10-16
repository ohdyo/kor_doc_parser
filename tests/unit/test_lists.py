"""항목 계층(기획서 4.3 규칙 5)."""
from kdp.assemble.headings import headings
from kdp.assemble.lists import lists
from kdp.schema import Block, Document, NUp, Page, ParserInfo, Source


def _page(no, *blocks, **kw):
    return Page(page_no=no, sheet_no=kw.pop("sheet_no", no), size=kw.pop("size", (595, 842)), **kw, blocks=[
        Block(id=f"p{no}.b{i + 1:02d}", type=t, bbox=bb, text=x) for i, (t, bb, x) in enumerate(blocks)])


def _doc(*pages):
    return Document(doc_id="sha256:ab", source=Source(paths=["a.pdf"], pages=len(pages)),
                    parser=ParserInfo(kdp_version="0", engine={}, engine_hash="e", assemble_hash="a", run_at="t"),
                    pages=list(pages))


def I(text, x, y, w=400.0):
    return "text", (x, y, x + w, y + 14), text


def T(text, y):
    return "paragraph_title", (70, y, 370, y + 16), text


def _run(*pages):
    return lists(headings(_doc(*pages)))


def _levels(doc):
    return [b.list_level for p in doc.pages for b in p.blocks]


def test_official_eight_levels_by_indent():
    marks = ["1. 관련", "가. 일시", "1) 세부", "가) 세부", "(1) 세부", "(가) 세부", "① 세부", "㉮ 세부"]
    doc = _run(_page(1, T("1. 개요", 40), *[I(t, 70 + 10 * i, 70 + 20 * i) for i, t in enumerate(marks)],
                     I("2. 협조 요청", 70, 240)))
    assert _levels(doc) == [None, 1, 2, 3, 4, 5, 6, 7, 8, 1]                     # 제목 블록에는 달지 않는다
    assert [b.text for b in doc.pages[0].blocks[1:9]] == marks                    # 기호는 원문대로
    assert doc.pages[0].blocks[1].provenance == [{"rule": "list_level", "by": "indent"}]


def test_report_bullets_and_paragraphs():
    doc = _run(_page(1, T("Ⅰ. 추진 배경", 40), I("정보화 사업 전반을 점검하였다.", 70, 70), I("□ 현황", 70, 90),
                     I("○ 예산 증가", 82, 110), I("- 전년 대비 10% 증가", 94, 130), I("· 세부 내역", 106, 150),
                     I("○ 인력 부족", 82, 170), I("※ 참고: 인력 현황표", 94, 190), I("맺음 문단입니다.", 70, 210)))
    assert _levels(doc) == [None, None, 1, 2, 3, 4, 2, 3, None]                  # 기호 없이 왼쪽 끝인 문단은 비운다
    assert doc.pages[0].blocks[1].provenance is None


def test_misread_or_lost_bullets_use_indent():
    doc = _run(_page(1, T("1. 현황", 40), I("□ 추진 현황", 70, 70), I("☐ 예산 증가", 82, 90), I("○ 인력 부족", 82, 110),
                     I("보안 위협 증가", 82, 130), I("- 세부 내용", 94, 150), I("이어진 설명 글입니다", 88, 170)))
    # ○ 를 ☐ 로 읽어도, 기호를 잃어도 같은 x 면 같은 수준. 묶음 사이 x 의 기호 없는 글(항목의 이어진 글)은 비운다
    assert _levels(doc) == [None, 1, 2, 2, 2, 3, None]


def test_no_list_section_keeps_indented_paragraphs_empty():
    doc = _run(_page(1, T("1. 개요", 40), I("본 사업은 정보 보안을 강화하려는 것이다.", 70, 70),
                     I("아래와 같이 보고합니다.", 85, 100)))
    assert _levels(doc) == [None, None, None]                                    # 첫 줄 들여쓰기 한 줄 문단일 수 있다


def test_levels_restart_per_section():
    doc = _run(_page(1, T("1. 현황", 40), I("□ 예산", 70, 70), I("○ 세부", 82, 90),
                     T("2. 계획", 120), I("○ 추진 일정", 82, 150), I("- 1단계", 94, 170)))
    assert _levels(doc) == [None, 1, 2, None, 1, 2]


def test_n_up_uses_logical_coordinates():
    """2×2 모아찍기 — 오른쪽 칸 쪽의 시트 x 는 크지만 논리 쪽 안 들여쓰기는 왼쪽 칸과 같다. 한 섹션이 두 칸에 걸친다."""
    left = NUp(cols=2, rows=2, cell=(0, 0), region=(0, 0, 297.5, 421))
    right = NUp(cols=2, rows=2, cell=(0, 1), region=(297.5, 0, 595, 421))
    doc = _run(_page(1, T("1. 현황", 20), I("□ 예산", 35, 50, 200), I("○ 세부", 41, 60, 200), n_up=left, sheet_no=1),
               _page(2, I("○ 세부 둘", 297.5 + 41, 20, 200), I("□ 인력", 297.5 + 35, 30, 200), I("본문 문단", 297.5 + 35, 40, 200),
                     n_up=right, sheet_no=1))
    assert _levels(doc) == [None, 1, 2, 2, 1, None]


def test_unindented_numbering_split_by_number_order():
    doc = _run(_page(1, T("Ⅰ. 개요", 40), I("1. 목적", 70, 70), I("가. 배경", 70, 90), I("나. 범위", 70, 110),
                     I("2. 내용", 70, 130), I("가. 세부", 70, 150), I("※ 참고 사항", 70, 170), I("○ 하위 항목", 84, 190)))
    assert _levels(doc) == [None, 1, 2, 2, 1, 2, 2, 3]                           # 한 x 에 1. 과 가. — 번호 차례로 나눈다
    assert doc.pages[0].blocks[2].provenance == [{"rule": "list_level", "by": "marker"}]


def test_decoration_far_right_and_non_text_blocks():
    doc = _run(_page(1, T("1. 안내", 40), I("□ 일시: 3월 15일", 70, 70), I("- 아 래 -", 270, 90, 60),
                     I("○ 비고", 330, 110, 100), I("2026. 3. 15.", 420, 130, 100),
                     ("table", (70, 150, 520, 250), None), ("figure_title", (70, 260, 300, 272), "1. 예산 현황")))
    assert _levels(doc) == [None, 1, None, None, None, None, None]               # 가운데 장식 줄·오른쪽 글·표·표 제목


def test_dropped_blocks_untouched():
    page = _page(1, T("1. 개요", 40), I("□ 항목", 70, 70), I("○ 하위", 82, 90))
    page.blocks[2].keep = False
    dropped = _page(2, I("□ 목차 항목", 70, 70), I("○ 목차 하위", 82, 90), page_class="toc", keep=False)
    doc = _run(page, dropped)
    assert _levels(doc) == [None, 1, None, None, None]


def test_caption_and_repeated_titles_do_not_cut_section():
    """표 제목으로 이은 제목·쪽마다 다시 찍은 제목은 경로를 바꾸지 않으므로 섹션도 끊지 않는다 — 표 아래·다음 쪽 하위 항목이 1 이 되지 않게."""
    page = _page(1, T("1. 예산 현황", 40), I("□ 사업비", 70, 70), I("○ 인건비", 82, 90),
                 ("paragraph_title", (70, 110, 300, 124), "<표 1> 인건비 내역"), ("table", (70, 130, 520, 230), None),
                 I("○ 운영비", 82, 240), I("○ 관리비", 82, 260))
    page.blocks[4].caption_id = "p1.b04"                                        # tables() 가 표 제목으로 이은 블록
    assert _levels(_run(page)) == [None, 1, 2, None, None, 2, 2]
    rep = _run(_page(1, T("Ⅱ. 사업 내용", 40), T("1. 개요", 70), I("□ 예산", 70, 100), I("○ 인건비", 82, 120)),
               _page(2, T("Ⅱ. 사업 내용", 40), I("○ 운영비", 82, 70), I("○ 관리비", 82, 90)))
    assert _levels(rep) == [None, None, 1, 2, None, 2, 2]


def test_page_shift_is_aligned_across_pages():
    """한 섹션이 여러 쪽에 걸칠 때 쪽마다 다른 가로 밀림(스캔 밀림·맞쪽 여백)을 맞춘 뒤 묶는다."""
    for shift in (0, 5, 15, -6):
        doc = _run(_page(1, T("1. 현황", 40), I("□ 예산", 70, 70), I("○ 세부 하나", 82, 90)),
                   _page(2, I("○ 세부 둘", 82 + shift, 40), I("□ 인력", 70 + shift, 60), I("○ 인력 세부", 82 + shift, 80)))
        assert _levels(doc) == [None, 1, 2, 2, 1, 2], shift
    mirror = _run(_page(1, T("1. 현황", 40), I("□ 예산", 85, 70), I("○ 세부 하나", 97, 90)),      # 홀짝 쪽 여백 10mm 차
                  _page(2, I("○ 세부 둘", 69, 40), I("□ 인력", 57, 60), I("○ 인력 세부", 69, 80)),
                  _page(3, I("□ 장비", 85, 40), I("○ 장비 세부", 97, 60)))
    assert _levels(mirror) == [None, 1, 2, 2, 1, 2, 1, 2]
    sub = _run(_page(1, T("1. 현황", 40), I("□ 예산", 70, 70), I("○ 세부", 82, 90)),
               _page(2, I("○ 세부 둘", 82, 40), I("- 더 세부", 94, 60)))
    assert _levels(sub) == [None, 1, 2, 2, 3]                                    # 하위 항목만 이어진 쪽은 끌어올리지 않는다


# ---- 2차 수정(재확인 회귀) — 재확인자가 적은 기대를 그대로 단언한다 ----

def _texts(doc):
    return [b.list_level for p in doc.pages for b in p.blocks if b.type == "text"]


def test_page_shift_needs_two_systems():
    """쪽 밀림은 서로 다른 체계 둘 이상이 같은 차이를 가리킬 때만 뺀다 — 같은 기호 모양 하나로는 하위 항목만 이어진 쪽,
    잘못 읽은 기호, 두 수준에 쓴 같은 기호와 가를 수 없고, '※' 는 자리가 일정하지 않다."""
    misread = _run(_page(1, T("1. 현황", 60), I("☐ 예산", 70, 100), I("○ 세부 하나", 82, 130)),     # □·○ 를 모두 ☐ 로
                   _page(2, I("☐ 세부 둘", 82, 60), I("- 상세", 94, 90), I("☐ 세부 셋", 82, 120)))
    assert _texts(misread) == [1, 2, 2, 3, 2]
    ppt = _run(_page(1, T("추진 계획", 30), I("• 목표", 40, 80), I("– 세부 목표", 60, 110), size=(960, 540)),
               _page(2, I("• 상세 하나", 80, 60), I("• 상세 둘", 80, 90), size=(960, 540)))
    assert _texts(ppt) == [1, 2, 3, 3]
    note = _run(_page(1, T("1. 현황", 60), I("□ 예산 현황", 70, 100), I("※ 단위: 백만원", 82, 120), I("○ 인건비 증가", 82, 140)),
                _page(2, I("※ 증감은 결산 기준", 70, 60), I("○ 운영비 감소", 82, 80), I("□ 인력 현황", 70, 110),
                      I("○ 정원 부족", 82, 130)))
    assert _texts(note) == [1, 2, 2, 1, 2, 1, 2]


def test_first_seen_doc_title_cuts_section():
    """경로 밖인 문서 제목(수준 0)도 처음 보는 것이면 섹션을 끊는다 — 공문 뒤 계획서, 쪽마다 다른 슬라이드 제목."""
    letter = _run(_page(1, ("doc_title", (70, 40, 370, 64), "정보보안 교육 실시 알림"),
                        I("1. 관련: 정보보안 업무 지침", 70, 90), I("2. 위와 관련하여 교육을 실시합니다.", 70, 110),
                        I("가. 일시: 3. 15.(금) 14:00", 82, 130), I("나. 장소: 본관 대회의실", 82, 150)),
                  _page(2, ("doc_title", (70, 40, 370, 64), "2026년 정보보안 교육 운영 계획"),
                        I("□ 추진 배경", 85, 90), I("○ 정보보안 사고가 늘고 있음", 97, 110),
                        I("□ 추진 계획", 85, 140), I("○ 교육 대상: 전 직원", 97, 160)))
    assert _texts(letter) == [1, 1, 2, 2, 1, 2, 1, 2]
    ppt = _run(_page(1, ("doc_title", (40, 30, 400, 58), "추진 배경"), I("• 정보보안 사고 증가", 40, 90),
                     I("– 해킹 시도 30% 늘어", 60, 115), size=(960, 540)),
               _page(2, ("doc_title", (40, 30, 400, 58), "추진 계획"), I("• 교육 대상 확대", 100, 90),
                     I("– 전 직원 연 2회", 120, 115), size=(960, 540)))
    assert _texts(ppt) == [1, 2, 1, 2]
    same = _run(_page(1, ("doc_title", (70, 40, 370, 64), "교육 계획"), I("□ 예산", 70, 90), I("○ 인건비", 82, 110)),
                _page(2, ("doc_title", (70, 40, 370, 64), "교육 계획"), I("○ 운영비", 82, 90)))
    assert _texts(same) == [1, 2, 2]                                             # 쪽마다 다시 찍은 같은 문서 제목은 끊지 않는다


def test_caption_above_chart_and_banded_repeat_do_not_cut_section():
    """차트 위 번호 캡션(제목 아님)과, 쪽 맨 위 띠 글 뒤에 다시 찍은 장 제목은 경로를 바꾸지 않아 섹션도 끊지 않는다."""
    chart = _page(1, T("Ⅰ. 시장 분석", 60), T("2. 시장 현황", 100), I("□ 매출 현황", 70, 130), I("○ 국내 매출 증가", 82, 150),
                  ("figure_title", (70, 180, 270, 194), "① 연도별 매출 추이"), ("chart", (70, 198, 520, 318), None),
                  I("○ 해외 매출 감소", 82, 330), I("○ 온라인 매출 정체", 82, 350))
    chart.blocks[5].img_path = "imgs/c.png"
    assert _texts(_run(chart)) == [1, 2, 2, 2]
    band = _run(_page(1, T("Ⅱ. 사업 내용", 60), T("1. 개요", 100), I("□ 예산", 70, 130), I("○ 인건비", 82, 160)),
                _page(2, ("text", (380, 30, 530, 42), "2026 정보보안 업무계획"), T("Ⅱ. 사업 내용", 60),
                      I("○ 운영비", 82, 100), I("○ 관리비", 82, 130)))
    assert _texts(band) == [1, 2, None, 2, 2]


def test_other_circled_numbers_nest_under_circle_items():
    doc = _run(_page(1, T("□ 조치사항", 40), I("○ 경보 발령 시", 63, 70), I("➀ 군집사례 기준: 홍보", 83, 90),
                     I("➁ 매개모기 기준: 방제", 83, 110)))
    assert _levels(doc) == [None, 1, 2, 2]


def test_attachment_list_is_one_list():
    doc = _run(_page(1, ("doc_title", (150, 40, 450, 60), "보도 제목"), I("- 요약 하나", 68, 80), I("본문 문단입니다.", 57, 110),
                     I("<붙임> 1. 첫째 붙임", 57, 140), I("2. 둘째 붙임", 113, 160), I("3. 셋째 붙임", 113, 180)))
    assert _levels(doc)[1:] == [1, None, 1, 1, 1]                             # 꼬리표 폭만큼 들여 썼어도 하위가 아니다
    assert doc.pages[0].blocks[4].provenance[-1] == {"rule": "list_level", "by": "attachment"}


def test_attachment_label_alone_opens_section():
    doc = _run(_page(1, T("1. 개요", 40), I("가. 첫 항목", 80, 70), I("붙임", 70, 100, w=40), I("1. 계획서 1부.", 120, 100),
                     I("2. 명단 1부.  끝.", 120, 120)))
    assert _levels(doc) == [None, 1, None, 1, 1]                              # 꼬리표만 있는 블록은 붙임 구역을 연다(headings)
