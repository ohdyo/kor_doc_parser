"""쪽 판정(표지·목차·빈 쪽·간지·결재·미기입 서식)과 표지 메타 승격."""
from kdp.classify.blocks import classify_blocks
from kdp.classify.pages import classify_pages
from kdp.classify.rules_meta import mark_meta_blocks
from kdp.config import Classify, MetaCfg, QCCfg
from kdp.enrich.meta import promote_meta
from kdp.pipeline import jsonl_rows
from kdp.qc.checks import run_checks
from kdp.schema import Block, Document, NUp, Page, ParserInfo, Source

BODY = ("text", (70, 100, 520, 300), "가나다라마바사 " * 30)


def _page(no, *blocks):
    return Page(page_no=no, sheet_no=no, size=(595, 842), blocks=[
        Block(id=f"p{no}.b{i + 1:02d}", type=t, bbox=bb, text=None if t == "table" else x, html=x if t == "table" else None)
        for i, (t, bb, x) in enumerate(blocks)])


def _doc(*pages):
    return Document(doc_id="sha256:ab", source=Source(paths=["a.pdf"], pages=len(pages)),
                    parser=ParserInfo(kdp_version="0", engine={}, engine_hash="e", assemble_hash="a", run_at="t"),
                    pages=list(pages))


def _run(doc, cfg=None):
    cfg = cfg or Classify()
    return promote_meta(classify_pages(mark_meta_blocks(classify_blocks(doc, cfg)), cfg), MetaCfg())


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
    assert doc.meta == {"title": "○○ 시스템 고도화 결과 보고서", "date": "2026-07",
                        "issuer": "한국정보공사 정보화처"}                     # 제목·날짜가 아닌 기관 줄 하나(기획서 5.1)


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


APPROVAL = ('<table><tr><td rowspan="2">결 재</td><td>담당</td><td>팀장</td><td>대표이사</td></tr>'
            "<tr><td>홍길동</td><td>3/5</td><td>전결</td></tr></table>")
FORM = ("<table><tr><td>성 명</td><td></td><td>소 속</td><td></td></tr><tr><td>직 위</td><td></td><td>연락처</td><td></td></tr>"
        '<tr><td>교육 과정</td><td colspan="3"></td></tr><tr><td>신청 사유</td><td colspan="3"></td></tr></table>')
GANTT = ("<table><tr><td>구분</td>" + "".join(f"<td>{m}월</td>" for m in range(1, 7)) + "</tr>"
         + "".join(f"<tr><td>{s}</td>" + "".join("<td>■</td>" if k in on else "<td></td>" for k in range(6)) + "</tr>"
                   for s, on in (("설계", (0, 1)), ("개발", (2, 3)), ("시험", (4, 5)))) + "</table>")


def test_blank_pages():
    doc = _run(_doc(_page(1, BODY),
                    _page(2, ("number", (290, 780, 305, 790), "- 2 -")),                       # 쪽번호뿐
                    _page(3, ("text", (150, 400, 450, 420), "이 페이지는 의도적으로 비워 두었습니다.")),
                    _page(4, ("aside_text", (450, 100, 560, 120), "대외비")),                   # 버린 띠지뿐
                    _page(5, ("text", (80, 100, 300, 115), "이상입니다.")),                     # 짧은 본문
                    _page(6, ("image", (80, 100, 500, 600), None)),                            # 그림만
                    _page(7, ("aside_text", (380, 100, 560, 200), "본 사업은 2단계로 추진한다. 1단계 결과를 보고 범위를 정한다.")),
                    _page(8, BODY)))
    assert [(p.page_class, p.keep) for p in doc.pages] == [("normal", True), ("blank", False), ("blank", False), ("blank", False),
                                                           ("normal", True), ("normal", True), ("normal", True), ("normal", True)]
    err = _doc(_page(1, BODY), _page(2, ("text", (80, 100, 500, 300), None)), _page(3, BODY))
    err.pages[1].blocks[0].error = "HTTP 500"
    assert _run(err).pages[1].keep                                        # 못 읽은 블록이 있는 쪽은 빈 쪽이 아니다


def test_divider_pages():
    doc = _run(_doc(_page(1, BODY),
                    _page(2, ("paragraph_title", (220, 380, 380, 410), "붙임 1"), ("text", (150, 420, 450, 440), "2026년 정보보안 교육 계획")),
                    _page(3, ("doc_title", (200, 380, 400, 440), "제 2 장\n사업 추진 현황")),
                    _page(4, ("paragraph_title", (250, 400, 350, 420), "[부록 A]")),
                    _page(5, ("text", (80, 100, 520, 120), "붙임  교육 계획 1부.  끝.")),              # 공문 끝이 넘어온 쪽
                    _page(6, ("paragraph_title", (80, 60, 300, 80), "붙임 2 추진 체계"), ("image", (80, 100, 520, 600), None)),
                    _page(7, ("paragraph_title", (80, 60, 300, 80), "제3장 결론"), ("text", (80, 100, 520, 120), "향후 과제를 정리하였다.")),
                    _page(8, ("text", (80, 100, 520, 120), "부록에서 다룬다")),
                    _page(9, BODY)))
    assert [(p.page_class, p.keep) for p in doc.pages[1:4]] == [("divider", False)] * 3
    assert all(p.keep and p.page_class == "normal" and not p.page_class_guess for p in doc.pages[4:])
    assert doc.pages[1].blocks[0].keep                                    # 간지 제목 블록은 그대로 — 섹션 경로가 읽는다


def test_approval_pages():
    meta = ("table", (60, 160, 550, 260), "<table><tr><td>문서번호</td><td>경영지원-2026-012</td></tr><tr><td>수 신</td><td>내부결재</td></tr></table>")
    doc = _run(_doc(_page(1, ("table", (300, 40, 550, 140), APPROVAL), meta),             # 결재선·공문 메타만
                    _page(2, ("table", (300, 40, 550, 140), APPROVAL), BODY),
                    _page(3, ("table", (60, 60, 550, 300), APPROVAL), ("text", (60, 320, 300, 335), "검토 의견 없음")),
                    _page(4, BODY)))
    assert [(p.page_class, p.keep, p.page_class_guess) for p in doc.pages] == [
        ("approval", False, None), ("normal", True, None), ("normal", True, "approval"), ("normal", True, None)]
    assert doc.meta == {"doc_no": "경영지원-2026-012"}


def test_form_blank_pages():
    form = _page(2, ("paragraph_title", (150, 60, 450, 80), "[별지 제1호서식] 교육 참가 신청서"),
                 ("table", (60, 100, 540, 500), FORM), ("text", (300, 700, 540, 715), "신청인          (서명 또는 인)"))
    filled = _page(3, ("table", (60, 100, 540, 500), FORM.replace("<td></td>", "<td>정보화팀</td>")))
    gantt = _page(4, ("table", (60, 100, 540, 300), GANTT))                # 빈 칸이 많은 일정표
    doc = _run(_doc(_page(1, BODY), form, filled, gantt, _page(5, BODY)))
    assert (doc.pages[1].page_class, doc.pages[1].keep, doc.pages[1].form) == ("form_blank", True, True)   # D2 — 살리고 표시만
    assert all(p.page_class == "normal" and p.form is None and not p.page_class_guess for p in doc.pages[2:])
    dropped = _run(_doc(_page(1, BODY), form), Classify(form_blank="drop")).pages[1]
    assert (dropped.page_class, dropped.keep) == ("form_blank", False)
    no_cue = _run(_doc(_page(1, BODY), _page(2, ("table", (60, 100, 540, 500), FORM)))).pages[1]
    assert (no_cue.keep, no_cue.page_class, no_cue.page_class_guess) == (True, "normal", "form_blank")   # 서식 표시가 없으면 검수로


def test_cover_with_doc_meta_block():
    meta = "<table><tr><td>문서번호</td><td>KDP-2026-07</td></tr><tr><td>시행일</td><td>2026. 7. 15.</td></tr></table>"
    cover = _page(1, ("table", (330, 40, 545, 90), meta), ("doc_title", (80, 240, 520, 300), "○○ 시스템 고도화\n결과 보고서"),
                  ("text", (250, 440, 350, 460), "2026년 7월"))
    doc = _run(_doc(cover, _page(2, BODY)))
    assert (doc.pages[0].page_class, doc.pages[0].keep, doc.pages[0].blocks[0].type) == ("cover", False, "doc_meta")
    assert doc.meta == {"doc_no": "KDP-2026-07", "date": "2026-07-15",                # 이름 붙은 시행일이 표지의 '2026년 7월' 보다 먼저
                        "title": "○○ 시스템 고도화 결과 보고서"}


def test_nup_page_uses_logical_page_size():
    cell = NUp(cols=2, rows=2, cell=(1, 1), region=(297.5, 421, 595, 842))   # 시트 오른쪽 아래 칸
    big = Page(page_no=2, sheet_no=1, size=(595, 842), n_up=cell,
               blocks=[Block(id="p2.b01", type="paragraph_title", bbox=(330, 560, 560, 700), text="붙임 1")])
    doc = _run(_doc(_page(1, BODY), big, _page(3, BODY)))
    assert (doc.pages[1].keep, doc.pages[1].page_class_guess) == (True, "divider")   # 논리 쪽의 26% — 시트로 재면 6%


def test_title_from_first_page_doc_title_without_cover():
    page = _page(1, ("doc_title", (80, 60, 520, 90), "정보화 사업 결과 보고서"), BODY)
    page.blocks[0].heading_level = 0                                  # 제목 수준 0 = 문서 제목(headings)
    doc = promote_meta(_doc(page, _page(2, BODY)), MetaCfg())
    assert doc.meta == {"title": "정보화 사업 결과 보고서"}


def test_divider_title_with_content_is_kept():
    """간지는 '붙임 n·제n장' 만 있는 쪽 — 제목 뒤에 항목·'※' 주·'이름: 값' 줄이 있으면 내용이 든 쪽이다."""
    pages = [_page(2, ("paragraph_title", (220, 380, 380, 410), "제5장 결론"), ("text", (150, 420, 450, 440), "□ 민원 처리 기간 30% 단축 달성")),
             _page(3, ("paragraph_title", (220, 380, 380, 410), "붙임 1"),
                   ("text", (100, 420, 500, 440), "교육 일시: 2026. 3. 15.(금) 14:00, 본관 대회의실")),
             _page(4, ("paragraph_title", (220, 380, 380, 410), "부록"), ("text", (150, 420, 450, 460), "1. 예산 총액 12억 원\n2. 사업 기간 2년")),
             _page(5, ("paragraph_title", (220, 380, 380, 410), "붙임 2"), ("text", (150, 420, 450, 440), "※ 참석자 명단은 당일 배부 예정"))]
    doc = _run(_doc(_page(1, BODY), *pages, _page(6, BODY)))
    assert all((p.page_class, p.keep, p.page_class_guess) == ("normal", True, None) for p in doc.pages)


def test_slogan_title_cover_is_still_cover():
    """표어 꼴(문장 꼴) 문서 제목도 표지 제목이다 — 표지로 버리고 제목·날짜를 올린다(M2 와 같음, 기관 줄은 issuer)."""
    cover = _page(1, ("doc_title", (80, 240, 520, 300), "데이터가 이끄는 행정, 국민이 체감한다"), ("text", (250, 440, 350, 455), "2026. 3."),
                  ("text", (230, 600, 370, 615), "한국정보공사"))
    doc = _run(_doc(cover, _page(2, BODY), _page(3, BODY)))
    assert (doc.pages[0].page_class, doc.pages[0].keep, doc.pages[0].blocks[0].type) == ("cover", False, "doc_title")
    assert doc.meta == {"title": "데이터가 이끄는 행정, 국민이 체감한다", "date": "2026-03", "issuer": "한국정보공사"}


def test_cover_subtitle_event_or_common_noun_is_not_issuer():
    """기관 이름 끝 글자(회·터)로 끝나도 행사 이름·일반 명사인 부제는 발신 기관이 아니다."""
    for sub in ("착수 보고회", "빅 데이터", "사업 설명회"):
        cover = _page(1, ("doc_title", (80, 240, 520, 300), "2026년 정보화 사업 추진 계획"), ("text", (230, 320, 370, 340), sub),
                      ("text", (250, 440, 350, 455), "2026. 3. 5."))
        doc = _run(_doc(cover, _page(2, BODY), _page(3, BODY)))
        assert doc.pages[0].page_class == "cover" and doc.meta == {"title": "2026년 정보화 사업 추진 계획", "date": "2026-03-05"}, sub
    org = _page(1, ("doc_title", (80, 240, 520, 300), "2026년 정보화 사업 추진 계획"), ("text", (250, 440, 350, 455), "2026. 3. 5."),
                ("text", (230, 600, 370, 615), "한국데이터센터"))
    assert _run(_doc(org, _page(2, BODY))).meta["issuer"] == "한국데이터센터"   # '…센터' 는 기관


def test_blank_phrase_must_fill_the_page():
    """빈 쪽 문구는 쪽 글 전체일 때만 — '이하 여백'·'돌봄 공백'·'백서' 가 든 짧은 쪽·장 제목 쪽은 남긴다."""
    kept = [_page(2, ("text", (80, 100, 500, 115), "5. 결론: 사업 목표 120% 달성"), ("text", (250, 130, 350, 145), "- 이하 여백 -")),
            _page(3, ("paragraph_title", (150, 380, 450, 410), "Ⅳ. 돌봄 공백 해소 방안")),
            _page(4, ("text", (80, 100, 500, 115), "○ 인력 공백 없이 상시 운영함")),
            _page(5, ("paragraph_title", (150, 380, 450, 410), "3. 국민참여 백서 발간"))]
    blank = [_page(6, ("text", (150, 400, 450, 420), "- 여 백 -")), _page(7, ("text", (150, 400, 450, 420), "(빈 페이지)")),
             _page(8, ("text", (150, 400, 450, 420), "This page intentionally left blank"))]
    doc = _run(_doc(_page(1, BODY), *kept, *blank, _page(9, BODY)))
    assert [(p.page_class, p.keep) for p in doc.pages[1:8]] == [("normal", True)] * 4 + [("blank", False)] * 3
