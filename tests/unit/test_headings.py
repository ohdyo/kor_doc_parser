"""제목 수준·섹션 경로(기획서 4.3 규칙 3·4)."""
from kdp.assemble.headings import headings, marker, numbering
from kdp.schema import Block, Document, NUp, Page, ParserInfo, Source

JUNK = {"header": "header", "number": "page_number"}


def _page(no, *blocks, **kw):
    return Page(page_no=no, sheet_no=kw.pop("sheet_no", no), size=kw.pop("size", (595, 842)), **kw, blocks=[
        Block(id=f"p{no}.b{i + 1:02d}", type=t, bbox=bb, text=x, keep=t not in JUNK, drop_reason=JUNK.get(t))
        for i, (t, bb, x) in enumerate(blocks)])


def _doc(*pages):
    return Document(doc_id="sha256:ab", source=Source(paths=["a.pdf"], pages=len(pages)),
                    parser=ParserInfo(kdp_version="0", engine={}, engine_hash="e", assemble_hash="a", run_at="t"),
                    pages=list(pages))


def T(text, y, h=16.0, typ="paragraph_title", w=300.0):
    return typ, (70, y, 70 + w, y + h), text


def X(text, y):
    return "text", (70, y, 520, y + 30), text


def H(text):
    return "header", (70, 30, 400, 42), text


def _by_text(doc):
    return {b.text: b for p in doc.pages for b in p.blocks}


def test_korean_numbering_levels_and_paths():
    doc = headings(_doc(_page(1, T("Ⅰ. 사업 개요", 60), X("개요 본문", 80), T("1. 추진 배경", 120), X("배경 본문", 140),
                              T("가. 현황", 180), X("현황 본문", 200), T("2. 추진 목표", 240), X("목표 본문", 260)),
                        _page(2, T("Ⅱ. 추진 계획", 60), T("1. 일정", 100), X("일정 본문", 120))))
    b = _by_text(doc)
    assert [b[t].heading_level for t in ("Ⅰ. 사업 개요", "1. 추진 배경", "가. 현황", "2. 추진 목표", "Ⅱ. 추진 계획", "1. 일정")] \
        == [1, 2, 3, 2, 1, 2]
    assert b["가. 현황"].section_path == ["Ⅰ. 사업 개요", "1. 추진 배경", "가. 현황"]       # 제목 자신도 경로에
    assert b["현황 본문"].section_path == ["Ⅰ. 사업 개요", "1. 추진 배경", "가. 현황"]
    assert b["목표 본문"].section_path == ["Ⅰ. 사업 개요", "2. 추진 목표"]
    assert b["일정 본문"].section_path == ["Ⅱ. 추진 계획", "1. 일정"]
    assert b["1. 일정"].provenance == [{"rule": "heading_level", "by": "marker"}]
    assert b["개요 본문"].heading_level is None and b["개요 본문"].provenance is None


def test_official_eight_levels_and_return():
    titles = ["1. 관련", "가. 일시", "1) 세부", "가) 세부", "(1) 세부", "(가) 세부", "① 세부", "㉮ 세부", "나. 장소", "2. 협조"]
    doc = headings(_doc(_page(1, *[T(t, 40 + 30 * i) for i, t in enumerate(titles)])))
    assert [b.heading_level for b in doc.pages[0].blocks] == [1, 2, 3, 4, 5, 6, 7, 8, 2, 1]
    assert doc.pages[0].blocks[8].section_path == ["1. 관련", "나. 장소"]


def test_first_seen_order_and_rank_correction():
    seen = headings(_doc(_page(1, T("1. 개요", 40), T("(1) 배경", 70), T("가. 세부", 100), T("(2) 범위", 130))))
    assert [b.heading_level for b in seen.pages[0].blocks] == [1, 2, 3, 2]    # 이 문서는 1. → (1) → 가. (처음 나온 차례)
    late = headings(_doc(_page(1, T("1. 요약", 40), T("가. 배경", 70), T("Ⅰ. 서론", 100), X("서론 본문", 120),
                               T("1. 목적", 160))))
    b = _by_text(late)
    assert (b["Ⅰ. 서론"].heading_level, b["1. 목적"].heading_level) == (1, 2)   # 'Ⅰ.' 을 '가.' 밑에 두지 않는다
    assert b["서론 본문"].section_path == ["Ⅰ. 서론"]


def test_report_bullets_and_chapters():
    doc = headings(_doc(_page(1, T("제1장 총칙", 40), T("제1절 목적", 70), T("□ 추진 배경", 100), X("본문", 120),
                              T("○ 세부 현황", 160), T("□ 추진 방향", 200), T("제2절 범위", 240), T("제2장 사업", 280))))
    b = _by_text(doc)
    assert [b[t].heading_level for t in ("제1장 총칙", "제1절 목적", "□ 추진 배경", "○ 세부 현황", "□ 추진 방향",
                                         "제2절 범위", "제2장 사업")] == [1, 2, 3, 4, 3, 2, 1]
    assert b["본문"].section_path == ["제1장 총칙", "제1절 목적", "□ 추진 배경"]


def test_numbers_that_are_not_numbering():
    for s in ("2026년 사업 계획", "2025. 3. 15. 회의 결과", "3. 15. 회의", "Ⅱ단계 사업", "(사)한국정보협회", "10.5% 증가",
              "제3장에서는 결과를 다룬다", "○○ 사업 추진", "- 아 래 -"):
        assert marker(s) is None, s
    assert numbering("1. 3월 실적") == ("1.", "1") and numbering("Ⅱ-1. 사업 범위") == ("Ⅰ-1", "Ⅱ-1")
    assert numbering("II. 사업 내용") == ("Ⅰ", "Ⅱ") and numbering("제1조(목적)") == ("조", "1")
    assert numbering("1.2 배경")[0] == "1.1" and numbering("□ 현황") is None and marker("ㅇ 현황") == ("○", "○")
    doc = headings(_doc(_page(1, T("Ⅰ. 개요", 40, h=22), T("2026년 사업 계획", 80, h=19), X("계획 본문", 100),
                              T("2025. 3. 15. 회의 결과", 140, h=19), X("회의 본문", 160))))
    b = _by_text(doc)
    assert b["2026년 사업 계획"].heading_level is None and b["2025. 3. 15. 회의 결과"].heading_level is None
    assert b["회의 본문"].section_path == ["Ⅰ. 개요", "2025. 3. 15. 회의 결과"]     # 앞 잎 제목은 떼고 새 잎


def test_unknown_level_heading_is_a_leaf():
    doc = headings(_doc(_page(1, T("Ⅰ. 개요", 40, h=22), T("추진 배경", 80, h=19), X("배경 본문", 100),
                              T("1. 현황", 140, h=16), X("현황 본문", 160))))
    b = _by_text(doc)
    assert b["추진 배경"].heading_level is None and b["추진 배경"].provenance is None
    assert b["추진 배경"].section_path == ["Ⅰ. 개요", "추진 배경"] == b["배경 본문"].section_path
    assert (b["1. 현황"].heading_level, b["현황 본문"].section_path) == (2, ["Ⅰ. 개요", "1. 현황"])


def test_height_cluster_assigns_level_only_when_sure():
    doc = headings(_doc(_page(1, T("Ⅰ. 개요", 40, h=22), T("1. 배경", 70, h=16), T("세부 현황", 100, h=16.5),
                              X("현황 본문", 120), T("2. 계획", 160, h=16), X("계획 본문", 180),
                              T("두 줄로 접힌 아주 긴 번호 없는 제목입니다", 220, h=33, w=120))))
    b = _by_text(doc)
    assert b["세부 현황"].heading_level == 2 and b["세부 현황"].provenance == [{"rule": "heading_level", "by": "height"}]
    assert b["현황 본문"].section_path == ["Ⅰ. 개요", "세부 현황"]
    assert (b["2. 계획"].heading_level, b["계획 본문"].section_path) == (2, ["Ⅰ. 개요", "2. 계획"])   # 형제로 돌아온다
    assert b["두 줄로 접힌 아주 긴 번호 없는 제목입니다"].heading_level is None               # 줄바꿈 없는 두 줄 — 높이를 믿지 않는다
    mixed = headings(_doc(_page(1, T("Ⅰ. 개요", 40, h=16), T("□ 배경", 70, h=16), T("현황", 100, h=16))))
    assert mixed.pages[0].blocks[2].heading_level is None                     # 한 높이에 수준이 둘 — 비운다
    flat = headings(_doc(_page(1, T("개요", 40, h=22), T("배경", 70, h=16), X("본문", 90))))
    assert [b.heading_level for b in flat.pages[0].blocks] == [None, None, None]   # 번호 제목이 없으면 기준이 없다
    assert flat.pages[0].blocks[2].section_path == ["배경"]


def test_height_on_n_up_page_uses_logical_page():
    """2×2 모아찍기 칸의 글자는 반으로 줄어 있다 — 높이는 논리 쪽 높이에 견주어야 보통 쪽 제목과 같은 크기로 본다."""
    cell = NUp(cols=2, rows=2, cell=(1, 0), region=(0, 421, 297.5, 842))
    doc = headings(_doc(_page(1, T("Ⅰ. 개요", 40, h=22), T("1. 배경", 70, h=16)),
                        _page(2, ("paragraph_title", (35, 450, 185, 458), "세부 현황"), sheet_no=1, n_up=cell)))
    assert doc.pages[1].blocks[0].heading_level == 2


def test_doc_title_level_zero_and_not_in_paths():
    doc = headings(_doc(_page(1, T("2026년 정보보안 교육 계획", 40, h=30, typ="doc_title"), X("머리 본문", 80),
                              T("1. 목적", 120), X("목적 본문", 140), T("Ⅱ. 사업 내용", 180, typ="doc_title"))))
    b = _by_text(doc)
    title = b["2026년 정보보안 교육 계획"]
    assert (title.heading_level, title.section_path) == (0, ["2026년 정보보안 교육 계획"])
    assert b["머리 본문"].section_path == [] and b["목적 본문"].section_path == ["1. 목적"]
    assert (b["1. 목적"].heading_level, b["Ⅱ. 사업 내용"].heading_level) == (1, 1)   # 번호 있는 doc_title 은 섹션


def test_divider_and_appendix_restart_path():
    doc = headings(_doc(_page(1, T("Ⅰ. 개요", 40), T("1. 배경", 70), X("배경 본문", 90)),
                        _page(2, T("붙임 1", 300), T("세부 추진 계획", 340, typ="doc_title"), ("number", (290, 800, 305, 810), "- 2 -"),
                              page_class="divider", keep=False),
                        _page(3, X("간지 뒤 본문", 40), T("1. 추진 배경", 80), X("배경 본문2", 100),
                              T("[붙임 2] 행사 운영 계획", 140), X("붙임2 본문", 160),
                              X("붙임 1. 교육 계획서 1부.  끝.", 200), X("별첨 3", 240), X("별첨 본문", 260))))
    b = _by_text(doc)
    assert b["간지 뒤 본문"].section_path == ["붙임 1 세부 추진 계획"]
    assert (b["1. 추진 배경"].heading_level, b["배경 본문2"].section_path) == (2, ["붙임 1 세부 추진 계획", "1. 추진 배경"])
    assert (b["[붙임 2] 행사 운영 계획"].heading_level, b["붙임2 본문"].section_path) == (1, ["[붙임 2] 행사 운영 계획"])
    assert b["붙임 1. 교육 계획서 1부.  끝."].section_path == ["[붙임 2] 행사 운영 계획"]   # 붙임 목록은 경로를 바꾸지 않는다
    assert (b["별첨 3"].heading_level, b["별첨 본문"].section_path) == (1, ["별첨 3"])      # '별첨 3' 만 있는 글 블록
    assert b["세부 추진 계획"].section_path is None                                      # 버린 쪽 블록에는 달지 않는다
    chapters = headings(_doc(_page(1, T("제1장 총칙", 40), T("제1절 목적", 70)),
                             _page(2, T("제2장 사업 계획", 300), page_class="divider", keep=False),
                             _page(3, X("본문", 40), T("제1절 범위", 80))))
    assert chapters.pages[2].blocks[0].section_path == ["제2장 사업 계획"]                # 번호 간지는 번호 차례대로
    assert chapters.pages[2].blocks[1].heading_level == 2


def test_attachment_list_title_is_not_restart():
    doc = headings(_doc(_page(1, T("1. 목적", 40), T("붙임  1. 교육 계획 1부.  끝.", 70, h=19), X("본문", 90))))
    b = doc.pages[0].blocks
    assert b[1].heading_level is None and b[2].section_path == ["1. 목적", "붙임 1. 교육 계획 1부. 끝."]


def test_linked_caption_is_not_a_heading():
    doc = _doc(_page(1, T("Ⅰ. 개요", 40), T("<표 1> 투입 인력", 70), ("table", (70, 90, 520, 200), None),
                     X("표 아래 본문", 220)))
    doc.pages[0].blocks[2].caption_id = "p1.b02"                 # tables() 가 표 제목으로 이은 블록
    b = headings(doc).pages[0].blocks
    assert b[1].heading_level is None and b[3].section_path == ["Ⅰ. 개요"]


def test_repeated_header_as_section_candidate():
    doc = headings(_doc(_page(1, H("Ⅲ. 수행 결과"), X("이어진 본문", 60), T("1. 개요", 100), X("개요 본문", 120)),
                        _page(2, H("Ⅲ. 수행 결과"), X("다음 쪽 본문", 60)),
                        _page(3, H("Ⅳ. 결론"), X("쪽 위 본문", 60), T("Ⅳ. 결론", 300), X("결론 본문", 320)),
                        _page(4, H("○○ 사업 결과 보고서"), H("□ 추진 현황"), X("넷째 쪽 본문", 60))))
    p1, p2, p3, p4 = (p.blocks for p in doc.pages)
    assert p1[1].section_path == ["Ⅲ. 수행 결과"] and p1[0].provenance == [{"rule": "header_section"}]
    assert (p1[2].heading_level, p1[3].section_path) == (2, ["Ⅲ. 수행 결과", "1. 개요"])
    assert p2[1].section_path == ["Ⅲ. 수행 결과", "1. 개요"] and p2[0].provenance is None   # 이미 경로에 있다
    assert p3[1].section_path == ["Ⅲ. 수행 결과", "1. 개요"]                              # 이 쪽 제목이 맡는다
    assert p3[3].section_path == ["Ⅳ. 결론"] and p3[0].provenance is None
    assert p4[2].section_path == ["Ⅳ. 결론"]                                            # 번호 없는·기호 머리말은 안 쓴다
    assert all(b.section_path is None and b.heading_level is None for b in (p1[0], p4[0], p4[1]))


def test_stale_header_does_not_pull_path_back():
    doc = headings(_doc(_page(1, H("Ⅲ. 수행 결과"), T("Ⅲ. 수행 결과", 60), X("셋째 장 본문", 80), T("Ⅳ. 결론", 300)),
                        _page(2, H("Ⅲ. 수행 결과"), X("넷째 장 본문", 60)),              # 구역이 바뀔 때만 고치는 머리말
                        _page(3, H("Ⅴ. 향후 계획"), X("다섯째 장 본문", 60))))         # 놓친 장 제목
    assert doc.pages[1].blocks[1].section_path == ["Ⅳ. 결론"] and doc.pages[1].blocks[0].provenance is None
    assert doc.pages[2].blocks[1].section_path == ["Ⅴ. 향후 계획"]


def test_word_style_numbers_in_titles_only():
    doc = headings(_doc(_page(1, T("1 서론", 40), T("1.1 연구 배경", 70), T("1.2 연구 범위", 100), T("2 본론", 130),
                              T("2.1 현황", 160), X("현황 본문", 180))))
    b = _by_text(doc)
    assert [b[t].heading_level for t in ("1 서론", "1.1 연구 배경", "1.2 연구 범위", "2 본론", "2.1 현황")] == [1, 2, 2, 1, 2]
    assert b["현황 본문"].section_path == ["2 본론", "2.1 현황"]
    assert marker("1 인당 비용은 같다") is None and marker("1 서론", title=True) == ("1.", "1")
    assert marker("2026 사업 계획", title=True) is None


def test_toc_hint():
    toc = [{"level": 0, "title": "○○ 결과 보고서", "page": 1}, {"level": 1, "title": "사업 개요", "page": 1},
           {"level": 2, "title": "추진 배경", "page": 1}, {"level": 2, "title": "추진 일정", "page": 2},
           {"level": 1, "title": "향후 계획", "page": 2}, {"level": 2, "title": "예산", "page": 9}]
    doc = headings(_doc(_page(1, T("사업 개요", 40), T("추진 배경", 70), X("배경 본문", 90)),
                        _page(2, T("추진 일정", 40), T("향후 계획", 70), T("예산", 100), X("예산 본문", 120))), toc)
    b = _by_text(doc)
    assert [b[t].heading_level for t in ("사업 개요", "추진 배경", "추진 일정", "향후 계획", "예산")] == [1, 2, 2, 1, None]
    assert b["배경 본문"].section_path == ["사업 개요", "추진 배경"]
    assert b["예산 본문"].section_path == ["향후 계획", "예산"]                            # 쪽이 먼 책갈피는 맞추지 않는다
    assert b["추진 일정"].provenance == [{"rule": "heading_level", "by": "toc"}]
    numbered = headings(_doc(_page(1, T("1. 개요", 40), T("□ 배경", 70), T("가. 세부", 100))),
                        [{"level": 0, "title": "1. 개요", "page": 1}, {"level": 1, "title": "가. 세부", "page": 1}])
    assert [x.heading_level for x in numbered.pages[0].blocks] == [1, 2, 2]               # 처음 나온 '가.' 의 자리를 책갈피가


def test_repeated_title_keeps_deeper_path():
    doc = headings(_doc(_page(1, T("Ⅱ. 사업 내용", 40), T("1. 개요", 70), X("개요 본문", 90)),
                        _page(2, T("Ⅱ. 사업 내용", 40), X("이어진 본문", 70))))
    p2 = doc.pages[1].blocks
    assert (p2[0].heading_level, p2[0].section_path) == (1, ["Ⅱ. 사업 내용"])
    assert p2[1].section_path == ["Ⅱ. 사업 내용", "1. 개요"]


def test_numbered_text_and_dropped_pages_are_not_headings():
    toc_page = _page(2, T("목 차", 40), ("content", (70, 80, 520, 200), "Ⅰ. 개요 ..... 3"), page_class="toc", keep=False)
    doc = headings(_doc(_page(1, T("Ⅰ. 개요", 40), X("1. 목적: 정보보안 강화", 70)), toc_page,
                        _page(3, X("다음 본문", 40))))
    p1 = doc.pages[0].blocks
    assert p1[1].heading_level is None and p1[1].section_path == ["Ⅰ. 개요"]           # 번호로 시작하는 글은 항목의 몫
    assert doc.pages[2].blocks[0].section_path == ["Ⅰ. 개요"]                         # 목차 쪽 제목은 경로가 아니다
    assert doc.pages[1].blocks[0].section_path is None


TBL = "<table><tr><td>구분</td><td>예산</td></tr><tr><td>인건비</td><td>1,500</td></tr></table>"


def test_unnumbered_first_title_takes_scheme_of_its_height():
    """번호 없는 첫 제목('요약')이 글자 높이로 수준을 받아도 같은 높이의 번호 제목을 자기 밑에 품지 않는다 — 그 군집의 번호 체계를 물려받는다."""
    doc = headings(_doc(_page(1, T("요약", 40, h=22), X("요약 본문", 70), T("Ⅰ. 서론", 120, h=22), T("1. 배경", 200),
                              X("배경 본문", 220)),
                        _page(2, T("Ⅱ. 본론", 40, h=22), X("본론 본문", 70))))
    b = _by_text(doc)
    assert [b[t].heading_level for t in ("요약", "Ⅰ. 서론", "1. 배경", "Ⅱ. 본론")] == [1, 1, 2, 1]
    assert b["요약 본문"].section_path == ["요약"] and b["본론 본문"].section_path == ["Ⅱ. 본론"]
    toc = [{"level": 0, "title": "Ⅰ. 서론", "page": 1}, {"level": 0, "title": "Ⅱ. 본론", "page": 2}]
    b = _by_text(headings(doc, toc))                                            # 책갈피 근거가 'marker' 로 바뀌어도 수준이 같으면 그대로
    assert (b["요약"].heading_level, b["본론 본문"].section_path) == (1, ["Ⅱ. 본론"])
    deep = headings(_doc(_page(1, T("Ⅰ. 개요", 40, h=22), T("추진 경과", 80), X("경과 본문", 100), T("1. 배경", 140),
                               X("배경 본문", 160))))
    b = _by_text(deep)
    assert (b["추진 경과"].heading_level, b["1. 배경"].heading_level) == (2, 2)     # 같은 자리 형제
    assert b["배경 본문"].section_path == ["Ⅰ. 개요", "1. 배경"]
    # 군집의 번호 체계가 둘(□·1.)이라 물려받을 수 없고, 넣으면 번호 제목 수준이 바뀐다 — 높이를 믿지 않고 잎으로
    mixed = headings(_doc(_page(1, T("Ⅰ. 개요", 40, h=22), T("□ 현황", 80), T("Ⅱ. 계획", 120, h=22), T("추진 일정", 160),
                                T("1. 세부", 200), X("세부 본문", 220))))
    b = _by_text(mixed)
    assert (b["추진 일정"].heading_level, b["1. 세부"].heading_level) == (None, 2)
    assert b["세부 본문"].section_path == ["Ⅱ. 계획", "1. 세부"]


def test_bigger_rank_title_goes_above_smaller_bullet_and_decorations():
    doc = headings(_doc(_page(1, T("□ 요약", 40, h=14), X("요약 본문", 60)),
                        _page(2, T("Ⅰ. 서론", 40, h=22), T("1. 배경", 110), X("배경 본문", 130)),
                        _page(3, T("Ⅱ. 본론", 40, h=22), X("본론 본문", 70))))
    b = _by_text(doc)
    assert [b[t].heading_level for t in ("□ 요약", "Ⅰ. 서론", "1. 배경", "Ⅱ. 본론")] == [1, 1, 2, 1]
    assert b["본론 본문"].section_path == ["Ⅱ. 본론"]
    same = headings(_doc(_page(1, T("□ 요약", 40), T("Ⅰ. 서론", 80))))
    assert same.pages[0].blocks[1].heading_level == 2                           # 글자가 같으면 처음 나온 차례 그대로
    sub = "- 공공기관 정보보안 역량 강화를 위한 교육 운영 -"
    assert marker(sub) is None and marker("◈ 2026년 업무보고 ◈") is None and marker("- 3 -") is None
    deco = headings(_doc(_page(1, T("◈ 2026년 업무보고 ◈", 20, h=30, typ="doc_title"), T(sub, 60, typ="doc_title"),
                               T("Ⅰ. 개요", 140, h=22), T("Ⅱ. 추진 계획", 220, h=22), X("계획 본문", 250))))
    b = _by_text(deco)
    assert [b[t].heading_level for t in ("◈ 2026년 업무보고 ◈", sub, "Ⅰ. 개요", "Ⅱ. 추진 계획")] == [0, 0, 1, 1]
    assert b["계획 본문"].section_path == ["Ⅱ. 추진 계획"]                        # 꾸민 문서 제목·부제는 경로 밖


def test_chapter_first_page_dropped_as_toc_keeps_chapter():
    """장 첫 쪽(장 제목 + 소목차)을 목차 쪽으로 버려도 장 제목은 경로에 들어간다. 쪽 머리가 아닌 같은 이름 절은 새 절이다."""
    mini = ("content", (100, 360, 500, 420), "1. 개요 ........ 15\n2. 성과 ........ 18")
    doc = headings(_doc(_page(1, T("Ⅱ. 사업 내용", 40, h=22), T("1. 개요", 80), T("가. 세부 범위", 120, h=14),
                              X("Ⅱ장 본문", 140)),
                        _page(2, T("Ⅲ. 수행 결과", 300, h=26), mini, page_class="toc", keep=False),
                        _page(3, X("Ⅲ장 첫 본문", 40), T("1. 개요", 100), X("Ⅲ장 개요 본문", 120))))
    b = _by_text(doc)
    assert b["Ⅲ장 첫 본문"].section_path == ["Ⅲ. 수행 결과"]
    assert b["Ⅲ장 개요 본문"].section_path == ["Ⅲ. 수행 결과", "1. 개요"]
    real = headings(_doc(_page(1, T("Ⅰ. 개요", 40)),                            # 목차 줄을 제목으로 잡은 진짜 목차 쪽
                         _page(2, T("Ⅱ. 사업 내용 ........ 5", 40), mini, page_class="toc", keep=False),
                         _page(3, X("다음 본문", 40))))
    assert real.pages[2].blocks[0].section_path == ["Ⅰ. 개요"]
    mid = headings(_doc(_page(1, T("1. 개요", 40), T("가. 세부", 70), X("세부 본문", 90)),
                        _page(2, X("이어진 본문", 40), T("1. 개요", 80), X("개요 본문", 100))))
    assert mid.pages[1].blocks[2].section_path == ["1. 개요"]                  # 반복 제목은 쪽 첫 블록일 때만


def test_landscape_page_title_height():
    """세로 문서 속 가로 쪽 제목 — 글자 높이를 쪽 긴 변으로 재어 같은 크기 제목과 한 군집이다."""
    doc = headings(_doc(_page(1, T("Ⅰ. 개요", 40, h=22), T("1. 배경", 80), X("배경 본문", 100)),
                        _page(2, T("세부 추진 일정", 40), X("일정 본문", 60), size=(842, 595)),
                        _page(3, T("2. 계획", 40), X("계획 본문", 60))))
    b = _by_text(doc)
    assert (b["세부 추진 일정"].heading_level, b["일정 본문"].section_path) == (2, ["Ⅰ. 개요", "세부 추진 일정"])
    assert b["계획 본문"].section_path == ["Ⅰ. 개요", "2. 계획"]


def test_toc_hint_keeps_divider_and_header_chapters():
    """장 책갈피가 간지·머리말에만 있어 맞지 않아도 절 책갈피가 장을 지우지 않는다(수준 1 은 책갈피 전체의 맨 위)."""
    toc = [{"level": 0, "title": "제1장 사업 개요", "page": 1}, {"level": 1, "title": "제1절 추진 배경", "page": 2},
           {"level": 0, "title": "제2장 사업 계획", "page": 3}, {"level": 1, "title": "제1절 범위", "page": 4}]
    doc = headings(_doc(_page(1, T("제1장 사업 개요", 300, h=30), page_class="divider", keep=False),
                        _page(2, T("제1절 추진 배경", 40), X("배경 본문", 60)),
                        _page(3, T("제2장 사업 계획", 300, h=30), page_class="divider", keep=False),
                        _page(4, T("제1절 범위", 40), X("범위 본문", 60))), toc)
    b = _by_text(doc)
    assert (b["제1절 범위"].heading_level, b["범위 본문"].section_path) == (2, ["제2장 사업 계획", "제1절 범위"])
    assert b["제1절 범위"].provenance == [{"rule": "heading_level", "by": "toc"}]
    header = headings(_doc(_page(1, H("Ⅲ. 수행 결과"), X("이어진 본문", 60), T("1. 개요", 100), X("개요 본문", 120))),
                      [{"level": 0, "title": "Ⅲ. 수행 결과", "page": 1}, {"level": 1, "title": "1. 개요", "page": 1}])
    assert header.pages[0].blocks[3].section_path == ["Ⅲ. 수행 결과", "1. 개요"]


def test_toc_exact_match_beats_bare_match():
    pages = (_page(1, T("제1장 개요", 40, h=22), T("제1절 개요", 80), X("절 본문", 100)),
             _page(2, T("제2절 범위", 40), X("범위 본문", 60)))
    numbered = [{"level": 0, "title": "제1장 개요", "page": 1}, {"level": 1, "title": "제1절 개요", "page": 1},
                {"level": 1, "title": "제2절 범위", "page": 2}]
    bare = [{"level": 0, "title": "개요", "page": 1}, {"level": 1, "title": "개요", "page": 1},
            {"level": 1, "title": "범위", "page": 2}]                          # 번호 없는 책갈피 — 수준이 갈리면 비운다
    for toc in (numbered, bare):
        b = _by_text(headings(_doc(*pages), toc))
        assert [b[t].heading_level for t in ("제1장 개요", "제1절 개요", "제2절 범위")] == [1, 2, 2]
        assert b["절 본문"].section_path == ["제1장 개요", "제1절 개요"]


def test_new_doc_title_after_sections_restarts_path():
    """이어 붙은 둘째 문서의 제목(처음 보는 doc_title)은 경로를 새로 시작한다. 쪽마다 다시 찍은 같은 문서 제목은 끊지 않고,
    '붙임 1' 라벨 바로 뒤 문서 제목은 붙임 뿌리를 둔다."""
    doc = headings(_doc(_page(1, T("Ⅲ. 향후 계획", 40, h=22), T("1. 추진 일정", 80), X("일정 본문", 100)),
                        _page(2, T("교육 운영 계획", 40, h=30, typ="doc_title"), T("□ 추진 배경", 100), X("배경 본문", 120)),
                        _page(3, T("교육 운영 계획", 40, h=30, typ="doc_title"), X("이어진 본문", 80))))
    p2, p3 = doc.pages[1].blocks, doc.pages[2].blocks
    assert (p2[0].heading_level, p2[0].section_path) == (0, ["교육 운영 계획"])
    assert (p2[1].heading_level, p2[2].section_path) == (1, ["□ 추진 배경"])
    assert p3[1].section_path == ["□ 추진 배경"]
    att = headings(_doc(_page(1, T("Ⅲ. 향후 계획", 40, h=22), X("일정 본문", 100)),
                        _page(2, X("붙임 1", 20), T("교육 운영 계획", 40, h=30, typ="doc_title"), T("□ 추진 배경", 100),
                              X("배경 본문", 120))))
    assert _by_text(att)["배경 본문"].section_path == ["붙임 1", "□ 추진 배경"]
    hdr = headings(_doc(_page(1, T("Ⅲ. 향후 계획", 40, h=22), T("1. 추진 일정", 80), X("일정 본문", 100)),
                        _page(2, H("Ⅳ. 결론"), T("결론 요약", 40, h=30, typ="doc_title"), X("결론 본문", 80))))
    assert hdr.pages[1].blocks[2].section_path == ["Ⅳ. 결론"]                  # 이 쪽 머리말이 고친 경로는 지우지 않는다


def test_long_appendix_title_restarts_path():
    long_t = "[붙임 2] 2026년 상반기 정보보안 실태 점검 결과 및 하반기 개선 조치 계획"
    doc = headings(_doc(_page(1, T("Ⅲ. 향후 계획", 40, h=22), T("1. 추진 일정", 80), X("일정 본문", 100)),
                        _page(2, T(long_t, 40, h=18, w=500), T("1. 점검 개요", 80), X("점검 본문", 100))))
    b = _by_text(doc)
    assert (b[long_t].heading_level, b["점검 본문"].section_path) == (1, [long_t, "1. 점검 개요"])
    loose = "붙임 자료를 참고하여 각 기관별 정보보안 교육 계획을 수립하고 그 결과를 제출할 것"   # 번호 없는 긴 글은 뿌리가 아니다
    b = _by_text(headings(_doc(_page(1, T("1. 목적", 40), T(loose, 70, h=19, w=500), X("본문", 90)))))
    assert b[loose].heading_level is None and b["본문"].section_path == ["1. 목적", loose]


def test_stale_header_not_reused_after_appendix():
    """붙임으로 경로를 새로 시작한 뒤에도 남은 본문 마지막 장 머리말은 다시 쓰지 않는다."""
    doc = headings(_doc(_page(1, H("Ⅴ. 결론"), T("Ⅴ. 결론", 60, h=22), X("결론 본문", 90)),
                        _page(2, T("붙임 1", 300), T("세부 추진 계획", 340, typ="doc_title"), page_class="divider", keep=False),
                        _page(3, H("Ⅴ. 결론"), T("1. 추진 배경", 60), X("붙임 배경 본문", 80))))
    p3 = doc.pages[2].blocks
    assert p3[2].section_path == ["붙임 1 세부 추진 계획", "1. 추진 배경"] and p3[0].provenance is None
    nodiv = headings(_doc(_page(1, H("Ⅴ. 결론"), T("Ⅴ. 결론", 60, h=22), X("결론 본문", 90)),
                          _page(2, H("Ⅴ. 결론"), T("[붙임 1] 세부 추진 계획", 60, h=20), T("1. 추진 배경", 100)),
                          _page(3, H("Ⅴ. 결론"), T("2. 추진 내용", 60), X("내용 본문", 80))))
    b = _by_text(nodiv)
    assert (b["2. 추진 내용"].heading_level, b["내용 본문"].section_path) == (2, ["[붙임 1] 세부 추진 계획", "2. 추진 내용"])


def test_numbered_figure_title_is_section_heading():
    """구역 제목에 붙은 figure_title(실측) — 번호로 시작하고 표·그림 바로 아래(캡션 자리)가 아니면 제목이다."""
    doc = _doc(_page(1, T("Ⅰ. 사업 개요", 40, h=22), T("1. 추진 목적", 80), X("목적 본문", 100),
                     ("figure_title", (70, 150, 370, 166), "2. 추진 배경"), X("배경 본문", 170),
                     ("table", (70, 220, 520, 300), None), ("figure_title", (70, 304, 370, 316), "3. 예산 현황"),
                     ("figure_title", (70, 330, 370, 342), "추진 체계도"), X("체계 본문", 350)))
    doc.pages[0].blocks[5].html = TBL
    b = _by_text(headings(doc))
    assert (b["2. 추진 배경"].heading_level, b["배경 본문"].section_path) == (2, ["Ⅰ. 사업 개요", "2. 추진 배경"])
    assert b["3. 예산 현황"].heading_level is None and b["추진 체계도"].heading_level is None
    assert b["체계 본문"].section_path == ["Ⅰ. 사업 개요", "2. 추진 배경"]


def test_misread_bullet_title_is_sibling():
    """VLM 이 ○·□ 를 ☐ 로 읽은 제목(실측) — 글자 높이·x 가 같은 기호 제목의 형제로 두고 그 자리 기호를 물려받는다."""
    doc = headings(_doc(_page(1, T("Ⅰ. 추진 배경", 40, h=22), T("□ 현황", 80), T("○ 예산 현황", 110, h=14),
                              X("예산 본문", 130), T("☐ 인력 현황", 170, h=14), X("인력 본문", 190),
                              T("☐ 추진 방향", 230), T("□ 기대 효과", 270), X("효과 본문", 290))))
    b = _by_text(doc)
    assert [b[t].heading_level for t in ("☐ 인력 현황", "☐ 추진 방향", "□ 기대 효과")] == [3, 2, 2]
    assert b["인력 본문"].section_path == ["Ⅰ. 추진 배경", "□ 현황", "☐ 인력 현황"]
    assert b["효과 본문"].section_path == ["Ⅰ. 추진 배경", "□ 기대 효과"]


def test_enclosed_digit_numbers_do_not_crash():
    """'⒈'·'⒉' — isdigit() 는 참이지만 int() 가 못 읽는 번호 글자도 규칙 4 의 번호 비교를 한다."""
    doc = headings(_doc(_page(1, T("⒈ 사업 개요", 40), X("개요 본문", 60)), _page(2, H("2. 추진 계획"), X("계획 본문", 60))))
    assert doc.pages[1].blocks[1].section_path == ["2. 추진 계획"]
    doc = headings(_doc(_page(1, T("1. 사업 개요", 40), X("개요 본문", 60)), _page(2, H("⒉ 추진 계획"), X("계획 본문", 60))))
    assert doc.pages[1].blocks[1].section_path == ["⒉ 추진 계획"]


def test_dotted_date_doc_title_is_not_numbered():
    """'6.25 …'·'12.31 …' 로 시작하는 문서 제목은 '1.1' 번호가 아니다 — 수준 0, 경로 밖."""
    doc = headings(_doc(_page(1, T("6.25 전쟁 제76주년 기념행사 추진 계획", 40, h=30, typ="doc_title", w=500),
                              T("□ 추진 배경", 100), X("배경 본문", 120))))
    b = _by_text(doc)
    assert (b["6.25 전쟁 제76주년 기념행사 추진 계획"].heading_level, b["□ 추진 배경"].heading_level) == (0, 1)
    assert b["배경 본문"].section_path == ["□ 추진 배경"]
    sec = headings(_doc(_page(1, T("1.1 연구 배경", 40), T("1.2 연구 범위", 70))))
    assert [x.heading_level for x in sec.pages[0].blocks] == [1, 1]              # 섹션 제목의 '1.1' 은 번호 그대로


def test_far_or_repeated_table_title_is_not_heading():
    """표 제목 꼴이고 '(단위: …)' 줄을 건너 바로 다음이 표인 블록 — 표와 멀어 잇지 못했거나 다음 쪽에 되풀이한 표 제목.
    잇지 못해 남은 '(계속)' 표시도 제목이 아니다."""
    doc = _doc(_page(1, T("1. 사업비 현황", 40), T("<표 1> 연도별 사업비", 560, h=15), ("table", (70, 600, 520, 780), None)),
               _page(2, T("(계속)", 30, w=60), T("<표 1> 연도별 사업비(계속)", 50, h=15),
                     ("text", (420, 70, 520, 80), "(단위: 천원)"), ("table", (70, 85, 520, 200), None),
                     X("집행률 본문", 230), T("<표 2> 참고 사항", 280, h=19), X("참고 본문", 300)))
    doc.pages[0].blocks[2].html = doc.pages[1].blocks[3].html = TBL
    out = headings(doc)
    b = _by_text(out)
    assert all(b[t].heading_level is None for t in ("<표 1> 연도별 사업비", "(계속)", "<표 1> 연도별 사업비(계속)"))
    assert out.pages[1].blocks[3].section_path == b["집행률 본문"].section_path == ["1. 사업비 현황"]   # 이어진 표·그 아래 글
    assert b["참고 본문"].section_path == ["1. 사업비 현황", "<표 2> 참고 사항"]   # 다음이 표가 아니면 잎 제목 그대로


# ---- 2차 수정(재확인 회귀·PARTIAL) — 재확인자가 적은 기대를 그대로 단언한다 ----

def _fig(doc, *ids):
    """차트·그림 블록에 크롭을 단다(출력이 있는 블록이어야 캡션 자리를 본다)."""
    for p in doc.pages:
        for b in p.blocks:
            if b.id in ids:
                b.img_path = f"imgs/{b.id}.png"
    return doc


def test_numbered_figure_title_above_figure_is_caption():
    """표·그림·차트 바로 위 figure_title 은 캡션 자리 — 번호가 있어도 제목이 아니다(아래 캡션과 대칭, '(단위: …)' 줄은 건넌다)."""
    chart = _fig(_doc(_page(1, T("Ⅰ. 시장 분석", 60, h=22), T("1. 시장 규모", 100), X("규모 본문", 120), T("2. 시장 현황", 170),
                            X("현황 본문", 190), T("1. 연도별 매출 추이", 240, h=14, typ="figure_title", w=200),
                            ("chart", (70, 258, 520, 408), None), X("매출 본문", 420), X("수출 본문", 460)),
                      _page(2, T("3. 전망", 60), X("전망 본문", 80))), "p1.b07")
    b = _by_text(headings(chart))
    assert b["1. 연도별 매출 추이"].heading_level is None
    assert b["매출 본문"].section_path == b["수출 본문"].section_path == ["Ⅰ. 시장 분석", "2. 시장 현황"]
    two = _fig(_doc(_page(1, T("Ⅱ. 조사 결과", 60, h=22), T("1. 응답자 특성", 100), X("응답자 본문", 120),
                          T("① 연령별 분포", 170, h=14, typ="figure_title", w=200), ("chart", (70, 188, 520, 308), None),
                          T("② 지역별 분포", 330, h=14, typ="figure_title", w=200), ("chart", (70, 348, 520, 468), None),
                          X("두 차트 설명", 490))), "p1.b05", "p1.b07")
    assert _by_text(headings(two))["두 차트 설명"].section_path == ["Ⅱ. 조사 결과", "1. 응답자 특성"]
    tbl = _doc(_page(1, T("Ⅲ. 추진 계획", 60, h=22), T("1. 추진 일정", 100), X("일정 본문", 120),
                     T("1) 세부 추진 일정", 170, h=14, typ="figure_title", w=200), ("text", (420, 186, 520, 196), "(단위: 일)"),
                     ("table", (70, 200, 520, 320), None), X("표 아래 글", 330)))
    tbl.pages[0].blocks[5].html = TBL
    b = _by_text(headings(tbl))
    assert b["1) 세부 추진 일정"].heading_level is None and b["표 아래 글"].section_path == ["Ⅲ. 추진 계획", "1. 추진 일정"]


def test_height_level_kept_unless_numbered_title_goes_deeper():
    """높이로 넣은 번호 없는 장('맺음말')이 앞 장의 깊은 스택을 풀어 번호 제목이 얕아지는 것은 맞다 — 높이를 버리지 않는다.
    번호 제목을 더 깊게 만든 높이 제목만 잎으로 두고, 다른 곳의 높이 수준은 지킨다."""
    end = headings(_doc(_page(1, T("Ⅰ. 개요", 60, h=22), T("1. 배경", 100), T("가. 국내 동향", 130, h=13), X("국내 본문", 150)),
                        _page(2, T("맺음말", 60, h=22), X("맺음 본문", 100), T("가. 요약", 150, h=13), X("요약 본문", 170))))
    b = _by_text(end)
    assert (b["맺음말"].heading_level, b["맺음 본문"].section_path) == (1, ["맺음말"])
    assert (b["가. 요약"].heading_level, b["요약 본문"].section_path) == (2, ["맺음말", "가. 요약"])
    both = headings(_doc(_page(1, T("Ⅰ. 개요", 60, h=22), T("추진 경과", 100), X("경과 본문", 120), T("1. 배경", 170),
                               T("가. 국내 동향", 200, h=13), X("국내 본문", 220)),
                         _page(2, T("맺음말", 60, h=22), X("맺음 본문", 100), T("가. 요약", 150, h=13), X("요약 본문", 170))))
    assert _by_text(both)["추진 경과"].heading_level == 2
    one = headings(_doc(_page(1, T("Ⅰ. 개요", 40, h=22), T("□ 현황", 80), X("현황 본문", 100)),
                        _page(2, T("Ⅱ. 계획", 40, h=22), T("추진 일정", 80), T("1. 세부", 120), X("세부 본문", 140)),
                        _page(3, T("맺음말", 40, h=22), X("맺음 본문", 80))))
    b = _by_text(one)                                                   # '추진 일정'(체계 둘인 군집)만 잎으로
    assert (b["추진 일정"].heading_level, b["1. 세부"].heading_level, b["맺음말"].heading_level) == (None, 2, 1)
    assert b["세부 본문"].section_path == ["Ⅱ. 계획", "1. 세부"] and b["맺음 본문"].section_path == ["맺음말"]


def test_toc_continuation_page_is_not_chapter_first_page():
    """여러 쪽 목차의 이어진 쪽(목차 제목 없음)에 장 줄 하나만 제목으로 잡혀도 장 첫 쪽으로 보지 않는다."""
    toc1 = ("content", (70, 100, 520, 700), "Ⅰ. 서론\n1. 연구 배경 ········ 1\nⅣ. 개선 방안\n1. 제도 개선 ········ 27")
    toc2 = ("content", (70, 100, 520, 300), "1. 결론 ········ 40\n2. 정책 제언 ········ 42")
    doc = headings(_doc(_page(1, T("목 차", 40, h=24), toc1, page_class="toc", keep=False),
                        _page(2, T("Ⅴ. 결론 및 제언", 70), toc2, page_class="toc", keep=False),
                        _page(3, T("연구 요약", 40, h=18), X("요약 본문", 80)),
                        _page(4, T("Ⅰ. 서론", 40, h=22), T("1. 연구 배경", 80), X("배경 본문", 100))))
    b = _by_text(doc)
    assert b["연구 요약"].section_path == b["요약 본문"].section_path == ["연구 요약"]
    assert b["배경 본문"].section_path == ["Ⅰ. 서론", "1. 연구 배경"]
    first = headings(_doc(_page(1, T("Ⅴ. 결론 및 제언", 70), toc2, page_class="toc", keep=False),   # 본문 제목이 아직 없다
                          _page(2, T("연구 요약", 40, h=18), X("요약 본문", 80))))
    assert _by_text(first)["요약 본문"].section_path == ["연구 요약"]


def test_a3_landscape_page_title_height():
    """A4 세로 문서 속 A3 가로 쪽(1191×842) 제목 — 글자 높이를 pt 그대로 재어 같은 크기 제목과 한 군집이다."""
    doc = headings(_doc(_page(1, T("Ⅰ. 개요", 60, h=22), T("1. 배경", 100), X("배경 본문", 120),
                              T("가. 세부 현황", 170, h=12), X("세부 본문", 190)),
                        _page(2, T("세부 추진 일정", 60), X("일정 본문", 90), size=(1191, 842)),
                        _page(3, T("2. 계획", 60), X("계획 본문", 80))))
    b = _by_text(doc)
    assert (b["세부 추진 일정"].heading_level, b["일정 본문"].section_path) == (2, ["Ⅰ. 개요", "세부 추진 일정"])
    assert b["계획 본문"].section_path == ["Ⅰ. 개요", "2. 계획"]


def test_toc_hint_puts_bullet_title_at_top():
    """책갈피가 맨 위라고 한 기호 제목은 책갈피와 맞은 장 밑으로 내리지 않는다(간지·머리말이 넣은 장만 바닥)."""
    toc = [{"title": "Ⅰ. 추진 배경", "level": 0, "page": 1}, {"title": "1. 현황", "level": 1, "page": 1},
           {"title": "Ⅱ. 추진 계획", "level": 0, "page": 2}, {"title": "1. 세부 일정", "level": 1, "page": 2},
           {"title": "◆ 참고 자료", "level": 0, "page": 3}]
    doc = headings(_doc(_page(1, T("Ⅰ. 추진 배경", 60, h=22), T("1. 현황", 100), X("현황 본문", 120)),
                        _page(2, T("Ⅱ. 추진 계획", 60, h=22), T("1. 세부 일정", 100), X("일정 본문", 120)),
                        _page(3, T("◆ 참고 자료", 60, h=22), X("참고 본문", 90))), toc)
    b = _by_text(doc)
    assert (b["◆ 참고 자료"].heading_level, b["참고 본문"].section_path) == (1, ["◆ 참고 자료"])
    div = [{"level": 0, "title": "제2장 사업 계획", "page": 1}, {"level": 1, "title": "□ 추진 배경", "page": 2}]
    doc = headings(_doc(_page(1, T("제2장 사업 계획", 300, h=30), page_class="divider", keep=False),
                        _page(2, T("□ 추진 배경", 40), X("배경 본문", 60))), div)
    assert _by_text(doc)["배경 본문"].section_path == ["제2장 사업 계획", "□ 추진 배경"]   # 간지 장은 지우지 않는다


def test_symmetric_decorated_paragraph_title_is_bullet_title():
    """양쪽에 같은 기호를 단 구역 제목('◆ 사업 개요 ◆')은 기호 제목 — 꾸밈 줄은 문서 제목과 '- … -' 류만."""
    doc = headings(_doc(_page(1, T("◆ 사업 개요 ◆", 60, h=18), X("개요 본문", 90), T("1. 추진 배경", 130), X("배경 본문", 150),
                              T("2. 추진 목표", 190), X("목표 본문", 210)),
                        _page(2, T("◆ 추진 계획 ◆", 60, h=18), T("1. 세부 일정", 100), X("일정 본문", 120))))
    b = _by_text(doc)
    assert b["배경 본문"].section_path == ["◆ 사업 개요 ◆", "1. 추진 배경"]
    assert b["일정 본문"].section_path == ["◆ 추진 계획 ◆", "1. 세부 일정"]
    sub = headings(_doc(_page(1, T("- 공공기관 정보보안 역량 강화를 위한 교육 운영 -", 40), T("Ⅰ. 개요", 80, h=22))))
    assert [x.heading_level for x in sub.pages[0].blocks] == [None, 1]           # '- … -' 부제는 문단 제목이어도 꾸밈


def test_mislabeled_doc_title_mid_body_keeps_chapter():
    """본문 중간 제목에 문서 제목 라벨이 잘못 붙어도(장 제목보다 확실히 작은 글자) 장 경로를 지우지 않는다 — 기호·'n.n' 은
    문단 제목처럼, 번호 없는 것은 경로를 끊지 않는 수준 0."""
    def run(mid, nxt):
        return _by_text(headings(_doc(_page(1, T("제2장 추진 현황", 60, h=22), X("장 본문", 90)),
                                      _page(2, T(mid, 60, h=18, typ="doc_title"), X("중간 본문", 90),
                                            T(nxt, 140, h=16), X("다음 본문", 160)))))
    b = run("□ 현황 및 문제점", "□ 개선 방향")
    assert b["중간 본문"].section_path == ["제2장 추진 현황", "□ 현황 및 문제점"]
    assert b["다음 본문"].section_path == ["제2장 추진 현황", "□ 개선 방향"]
    b = run("2.1 추진 경과", "2.2 개선 방향")
    assert (b["2.1 추진 경과"].heading_level, b["중간 본문"].section_path) == (2, ["제2장 추진 현황", "2.1 추진 경과"])
    assert b["다음 본문"].section_path == ["제2장 추진 현황", "2.2 개선 방향"]
    b = run("추진 경과", "개선 방향")
    assert (b["추진 경과"].heading_level, b["중간 본문"].section_path) == (0, ["제2장 추진 현황"])


def test_header_replaces_unnumbered_title_that_took_its_scheme():
    """높이로 'Ⅰ' 체계를 물려받은 번호 없는 '요약' 은 같은 체계 장 머리말(규칙 4)을 막지 않는다 — 번호를 읽을 수 없으니 새 장."""
    doc = headings(_doc(_page(1, T("요약", 60, h=22), X("요약 본문", 100)),
                        _page(2, H("Ⅰ. 서론"), X("Ⅰ. 서론", 60), X("서론 본문", 100), T("1. 배경", 150), X("배경 본문", 170)),
                        _page(3, H("Ⅱ. 본론"), T("Ⅱ. 본론", 60, h=22), X("본론 본문", 100))))
    b = _by_text(doc)
    assert b["서론 본문"].section_path == ["Ⅰ. 서론"] and b["배경 본문"].section_path == ["Ⅰ. 서론", "1. 배경"]
    assert doc.pages[1].blocks[0].provenance == [{"rule": "header_section"}]
    stale = headings(_doc(_page(1, T("Ⅰ. 서론", 40, h=22), T("Ⅱ. 현황", 120, h=22), X("현황 본문", 150)),
                          _page(2, T("참고", 60, h=22), X("참고 본문", 90)),
                          _page(3, H("Ⅱ. 현황 분석"), X("참고 이어진 본문", 60)),      # 지난 장 머리말 — 번호가 앞서지 않는다
                          _page(4, H("Ⅲ. 분석"), X("분석 본문", 60))))
    b = _by_text(stale)
    assert b["참고 이어진 본문"].section_path == ["참고"] and b["분석 본문"].section_path == ["Ⅲ. 분석"]


def test_repeated_title_after_side_band_keeps_deeper_path():
    """쪽 맨 위에 머리말로 안 잡힌 띠 글(제목과 가로로 겹치지 않음)이 먼저 와도 다시 찍은 장 제목은 되풀이다."""
    doc = headings(_doc(_page(1, T("Ⅱ. 사업 내용", 60, h=22), T("1. 개요", 100), X("개요 본문", 130)),
                        _page(2, ("text", (380, 30, 530, 42), "2026 정보보안 업무계획"), T("Ⅱ. 사업 내용", 60, h=22),
                              X("이어진 본문", 100))))
    p2 = doc.pages[1].blocks
    assert (p2[1].heading_level, p2[2].section_path) == (1, ["Ⅱ. 사업 내용", "1. 개요"])


def test_continued_title_repeats_earlier_title():
    """끝이 '(계속)' 이고 번호·기호를 뗀 글이 앞 제목과 같은 제목은 새 제목이 아니다 — 스택을 그대로 두고 그 블록 경로는 원래
    제목 경로(쪽 넘김 표를 이으면 마크다운도 원래 제목 아래에 둔다)."""
    doc = _doc(_page(1, T("1. 사업비 집행", 500), T("□ 사업비 집행 현황", 560), ("table", (84, 575, 550, 790), None)),
               _page(2, T("□ 사업비 집행 현황(계속)", 60), ("table", (84, 75, 550, 300), None), X("집행률 본문", 320)))
    doc.pages[0].blocks[2].html = doc.pages[1].blocks[1].html = TBL
    p2 = headings(doc).pages[1].blocks
    assert p2[0].heading_level is None and p2[0].provenance == [{"rule": "heading_repeat"}]
    assert p2[0].section_path == p2[1].section_path == p2[2].section_path == ["1. 사업비 집행", "□ 사업비 집행 현황"]
    leaf = headings(_doc(_page(1, T("Ⅰ. 개요", 40, h=22), T("사업비 현황", 80, h=19), X("본문 하나", 100)),
                         _page(2, T("사업비 현황 (계속)", 40, h=19), X("본문 둘", 70))))
    assert leaf.pages[1].blocks[1].section_path == ["Ⅰ. 개요", "사업비 현황"]          # 잎 제목의 되풀이


def test_report_bullet_after_misread_bullet_is_sibling():
    """잘못 읽힌 ☐ 가 먼저 나오고 바로 읽힌 보고서체 기호(○·□)가 뒤에 와도, 글자 높이·x 가 같으면 형제다."""
    doc = headings(_doc(_page(1, T("Ⅰ. 추진 배경", 40, h=22), T("□ 현황", 80), T("☐ 예산 현황", 110, h=14), X("앞 본문", 130),
                              T("○ 인력 현황", 170, h=14), X("뒤 본문", 190))))
    b = _by_text(doc)
    assert (b["○ 인력 현황"].heading_level, b["뒤 본문"].section_path) == (3, ["Ⅰ. 추진 배경", "□ 현황", "○ 인력 현황"])
    doc = headings(_doc(_page(1, T("Ⅰ. 추진 배경", 40, h=22), T("☐ 현황", 80), X("현황 본문", 100),
                              T("□ 추진 방향", 140), X("방향 본문", 160), T("○ 세부", 200, h=13), X("세부 본문", 220))))
    b = _by_text(doc)
    assert (b["□ 추진 방향"].heading_level, b["방향 본문"].section_path) == (2, ["Ⅰ. 추진 배경", "□ 추진 방향"])
    assert b["세부 본문"].section_path == ["Ⅰ. 추진 배경", "□ 추진 방향", "○ 세부"]   # 높이가 다르면 지금처럼 자식


def test_sentence_like_doc_title_is_level_zero():
    """표어·보도자료 꼴 문서 제목도 문서 제목(수준 0 — meta.title 승격 근거). 문장 꼴 문단 제목은 후보가 아니다."""
    doc = headings(_doc(_page(1, T("정부, 공공데이터 3만 개 추가 개방한다", 40, h=24, typ="doc_title", w=500), X("본문", 80),
                              T("하반기 산출물 보완 현황은 아래와 같음", 120, w=500), X("본문 둘", 140))))
    b = _by_text(doc)
    assert b["정부, 공공데이터 3만 개 추가 개방한다"].heading_level == 0
    assert b["하반기 산출물 보완 현황은 아래와 같음"].heading_level is None and b["본문 둘"].section_path == []
