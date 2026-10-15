"""쪽 판정(표지·목차)과 표지 메타 승격."""
from kdp.classify.blocks import classify_blocks
from kdp.classify.pages import classify_pages
from kdp.config import Classify, MetaCfg, QCCfg
from kdp.enrich.meta import promote_meta
from kdp.pipeline import jsonl_rows
from kdp.qc.checks import run_checks
from kdp.schema import Block, Document, Page, ParserInfo, Source

BODY = ("text", (70, 100, 520, 300), "가나다라마바사 " * 30)


def _page(no, *blocks):
    return Page(page_no=no, sheet_no=no, size=(595, 842), blocks=[
        Block(id=f"p{no}.b{i + 1:02d}", type=t, bbox=bb, text=x) for i, (t, bb, x) in enumerate(blocks)])


def _doc(*pages):
    return Document(doc_id="sha256:ab", source=Source(paths=["a.pdf"], pages=len(pages)),
                    parser=ParserInfo(kdp_version="0", engine={}, engine_hash="e", assemble_hash="a", run_at="t"),
                    pages=list(pages))


def _run(doc):
    return promote_meta(classify_pages(classify_blocks(doc, Classify()), Classify()), MetaCfg())


def _cover(no=1):
    return _page(no, ("text", (400, 115, 545, 126), "관리번호 제2026-수행-082호"),
                 ("doc_title", (80, 240, 520, 300), "○○ 시스템 고도화\n결과 보고서"),
                 ("text", (250, 440, 350, 460), "2026년 7월"), ("image", (240, 600, 380, 645), None))


def _toc(no=2, title="[목  차]"):
    return _page(no, ("header", (80, 55, 390, 70), "○○ 결과 보고서"),
                 ("paragraph_title", (260, 120, 360, 145), title),
                 ("content", (80, 150, 540, 190), "Ⅰ. 사업 개요 ........ 3\nⅡ. 수행 결과 ........ 10"),
                 ("number", (290, 780, 305, 790), "-i-"))


def test_cover_and_toc_dropped_meta_promoted():
    doc = _run(_doc(_cover(), _toc(), _page(3, BODY)))
    assert [(p.page_class, p.keep) for p in doc.pages] == [("cover", False), ("toc", False), ("normal", True)]
    assert doc.meta == {"doc_no": "제2026-수행-082호", "title": "○○ 시스템 고도화 결과 보고서", "date": "2026-07"}
    assert {r["page_no"] for r in jsonl_rows(doc)} == {3}            # 버린 쪽의 블록은 출력에 없다
    assert run_checks(doc, [], QCCfg(max_drop_char_ratio=1.0)).qc.status == "ok"


def test_not_cover():
    one_page = _run(_doc(_cover()))                                   # 한 쪽짜리 공문을 표지로 지우지 않는다
    long_first = _run(_doc(_page(1, ("doc_title", (80, 60, 520, 90), "결과 보고서"), BODY), _page(2, BODY)))
    third = _run(_doc(_page(1, BODY), _page(2, BODY), _cover(3)))     # 앞 두 쪽만 후보
    for doc in (one_page, long_first, third):
        assert all(p.keep and p.page_class == "normal" for p in doc.pages) and doc.meta == {}


def test_second_page_cover_only_after_blank_first():
    assert _run(_doc(_page(1, ("number", (290, 780, 305, 790), "1")), _cover(2), _page(3, BODY))).pages[1].page_class == "cover"
    assert _run(_doc(_page(1, BODY), _cover(2), _page(3, BODY))).pages[1].keep


def test_toc_title_variants():
    for title in ("[목  차]", "차 례", "CONTENTS", "< 표 목차 >", "그림 목차", "[도표 목차]"):
        assert _run(_doc(_page(1, BODY), _toc(2, title))).pages[1].page_class == "toc", title
    assert _run(_doc(_page(1, BODY), _toc(2, "목차 구성 방안"))).pages[1].page_class == "toc"   # 목차 블록·점선 줄 두 신호
    assert not _run(_doc(_page(1, BODY), _page(2, ("paragraph_title", (80, 100, 300, 120), "목차 구성 방안"), BODY))).pages[1].page_class_guess


def test_unsure_pages_kept_as_guess():
    title_only = _page(2, ("paragraph_title", (260, 120, 360, 145), "목 차"), ("text", (80, 150, 540, 170), "아래와 같다"))
    with_body = _page(3, ("paragraph_title", (260, 60, 360, 80), "목 차"),
                      ("content", (80, 90, 540, 130), "1. 개요 ..... 3\n2. 결과 ..... 5"), BODY)
    doc = _run(_doc(_page(1, BODY), title_only, with_body))
    assert [(p.keep, p.page_class_guess) for p in doc.pages] == [(True, None), (True, "toc"), (True, "toc")]
    assert not doc.pages[2].blocks[1].keep                            # 목차 블록은 라벨 정책으로 따로 버린다
    assert "page_class_guess" in run_checks(doc, [], QCCfg()).qc.reasons


def test_short_notice_first_page_is_not_cover():
    notice = _page(1, ("doc_title", (80, 80, 520, 110), "2026년 상반기 정보보안 교육 실시 안내"),
                   ("text", (80, 130, 520, 170), "정보보안 인식 제고를 위하여 아래와 같이 교육을 실시하오니 참석하여 주시기 바랍니다."),
                   ("text", (80, 190, 520, 205), "가. 일시: 2026. 3. 15.(금) 14:00~16:00"),
                   ("text", (80, 215, 520, 230), "나. 장소: 본관 3층 대회의실"))
    doc = _run(_doc(notice, _page(2, BODY), _page(3, BODY)))
    assert doc.pages[0].keep and doc.pages[0].page_class == "normal" and doc.meta == {}


def test_big_paragraph_title_cover_promotes_title():
    cover = _page(1, ("paragraph_title", (80, 240, 520, 300), "○○ 시스템 고도화\n결과 보고서"),
                  ("text", (250, 440, 350, 455), "2026년 7월"), ("text", (230, 600, 370, 615), "한국정보공사 정보화처"))
    doc = _run(_doc(cover, _page(2, BODY)))
    assert doc.pages[0].page_class == "cover"
    assert doc.meta == {"title": "○○ 시스템 고도화 결과 보고서", "date": "2026-07"}


def test_date_is_not_taken_from_numbers_or_ranges():
    cover = _page(1, ("text", (400, 80, 545, 92), "관리번호 제2025-12-001호"),
                  ("doc_title", (80, 240, 520, 300), "결과 보고서"),
                  ("text", (200, 400, 400, 412), "사업기간: 2025. 4. ~ 2026. 3."),
                  ("text", (250, 440, 350, 455), "2026년 3월"))
    doc = _run(_doc(cover, _page(2, BODY)))
    assert doc.meta["date"] == "2026-03" and doc.meta["doc_no"] == "제2025-12-001호"
    from kdp.classify.pages import find_date
    assert find_date("ISBN 979-11-2024-11-5") is None and find_date("2026.99.99") is None
    assert find_date("2026. 3. 15.") == (2026, 3, 15)


def test_cover_variants_after_review():
    research = _page(1, ("doc_title", (80, 200, 520, 240), "공공 데이터 플랫폼 고도화 방안 연구"),
                     ("text", (150, 300, 450, 312), "연구기간: 2025. 4. ~ 2026. 3."),
                     ("text", (150, 320, 450, 332), "주관연구기관: 한국정보공사"),
                     ("text", (150, 340, 450, 352), "공동연구기관: 국가정보원"),
                     ("text", (150, 360, 450, 372), "발주기관: 행정안전부"),
                     ("text", (150, 380, 450, 392), "사업 추진 개요"),
                     ("text", (100, 700, 500, 712), "※ 이 보고서의 무단 전재를 금합니다."),
                     ("text", (250, 600, 350, 612), "2026. 3."))
    doc = _run(_doc(research, _page(2, BODY)))
    assert doc.pages[0].page_class == "cover" and doc.meta["date"] == "2026-03"   # 항목 줄이 많은 연구보고서 표지
    dashed = _page(1, ("doc_title", (80, 80, 520, 110), "정보보안 교육 실시 안내"),
                   ("text", (80, 190, 520, 205), "- 일시: 2026. 3. 15.(금) 14:00"),
                   ("text", (80, 215, 520, 230), "- 장소: 본관 3층 대회의실"))
    assert _run(_doc(dashed, _page(2, BODY))).pages[0].page_class == "normal"   # '-' 항목 안내문은 표지가 아니다
    one_item = _page(1, ("doc_title", (80, 240, 520, 300), "결과 보고서"), ("text", (250, 440, 350, 455), "2026년 7월"),
                     ("text", (150, 500, 450, 512), "• 비공개 자료"))
    p = _run(_doc(one_item, _page(2, BODY))).pages[0]
    assert p.keep and p.page_class_guess == "cover"                  # 항목 줄 하나뿐이면 버리지 않고 검수로


def test_dates_with_brackets():
    from kdp.classify.pages import find_date
    assert find_date("2026.3.15(금)") == (2026, 3, 15) and find_date("(2026.03.15)") == (2026, 3, 15)
    assert find_date("작성일: 2026.3.15, 정보화처") == (2026, 3, 15) and find_date("제2025-12-001호") is None


def test_meeting_minutes_first_page_is_not_cover():
    minutes = _page(1, ("doc_title", (80, 80, 520, 110), "정보화 추진 협의회 회의록"),
                    ("text", (80, 190, 520, 205), "일시: 2026. 3. 15.(금) 14:00"),
                    ("text", (80, 215, 520, 230), "장소: 본관 3층 대회의실"),
                    ("text", (80, 240, 520, 255), "참석: 정보화처장 외 5명"))
    doc = _run(_doc(minutes, _page(2, BODY)))
    assert doc.pages[0].keep and doc.pages[0].page_class == "normal" and doc.meta == {}


def test_cover_date_line_shares_block_with_org():
    cover = _page(1, ("doc_title", (80, 240, 520, 300), "결과 보고서"),
                  ("text", (250, 600, 350, 640), "2026. 3.\n한국정보공사"))
    doc = _run(_doc(cover, _page(2, BODY)))
    assert not doc.pages[0].keep and doc.meta["date"] == "2026-03"   # 날짜 줄과 기관 줄이 한 블록이어도
    from kdp.classify.pages import doc_date
    assert doc_date("일시: 2026. 3. 15.(금) 14:00") is None and doc_date("작성일: 2026.3.15") == (2026, 3, 15)
