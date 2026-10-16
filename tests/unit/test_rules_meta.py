"""공문 메타(doc_meta)·결재란(approval_box) 규칙과 문서 메타 승격(D3·D15·has_stamp)."""
from kdp.classify.blocks import classify_blocks
from kdp.classify.pages import classify_pages
from kdp.classify.rules_meta import mark_meta_blocks
from kdp.config import Classify, MetaCfg
from kdp.enrich.meta import promote_meta
from kdp.pipeline import jsonl_rows
from kdp.schema import Block, Document, NUp, Page, ParserInfo, Source

BODY = ("text", (70, 300, 520, 500), "가나다라마바사 " * 30)
APPROVAL = ('<table><tr><td rowspan="2">결 재</td><td>담당</td><td>팀장</td><td>대표이사</td></tr>'
            "<tr><td>홍길동</td><td>3/5</td><td>전결</td></tr></table>")
META_TABLE = ("<table><tr><td>문서번호</td><td>경영지원-2026-012</td><td>기안일자</td><td>2026. 3. 5.</td></tr>"
              "<tr><td>발 신</td><td>정보화팀</td><td>수 신</td><td>내부결재</td></tr>"
              "<tr><td>참 조</td><td>재무팀장</td><td>제 목</td><td>업무용 노트북 구매 품의</td></tr></table>")


def _page(no, *blocks, n_up=None):
    return Page(page_no=no, sheet_no=no, size=(595, 842), n_up=n_up, blocks=[
        Block(id=f"p{no}.b{i + 1:02d}", type=t, bbox=bb, text=None if t == "table" else x, html=x if t == "table" else None)
        for i, (t, bb, x) in enumerate(blocks)])


def _doc(*pages):
    return Document(doc_id="sha256:ab", source=Source(paths=["a.pdf"], pages=len(pages)),
                    parser=ParserInfo(kdp_version="0", engine={}, engine_hash="e", assemble_hash="a", run_at="t"),
                    pages=list(pages))


def _run(doc, meta=None):
    doc = classify_pages(mark_meta_blocks(classify_blocks(doc, Classify())), Classify())
    return promote_meta(doc, meta or MetaCfg())


def _gongmun():
    """행정기관 공문 한 쪽 — 위 수신·경유·제목, 본문, '끝.' 아래 발신명의·도장·결재선·시행·연락처."""
    return _page(1, ("text", (70, 110, 520, 160), "수신  ○○기관장(정보화과장)\n(경유)\n제목  2026년 정보보안 교육 실시 알림"),
                 ("text", (70, 180, 520, 300), "1. 관련: 정보화과-1001(2026. 2. 20.)\n"
                  "2. 정보보안 인식을 높이기 위하여 아래와 같이 교육을 실시하오니 참석하여 주시기 바랍니다."),
                 ("text", (70, 320, 520, 340), "붙임  교육 계획 1부.  끝."),
                 ("text", (230, 380, 370, 400), "한국정보공사 사장"),
                 ("seal", (380, 370, 440, 420), None),
                 ("text", (70, 700, 520, 712), "기안자 주무관 홍길동  검토자 사무관 김철수  결재권자 정보화과장 이영희"),
                 ("text", (70, 716, 520, 740), "시행 정보화과-1234 (2026. 3. 5.)  접수\n우 30128 ○○시 ○○로 12 / www.example.kr"),
                 ("text", (70, 744, 520, 756), "전화번호 044-000-1234  전송 044-000-8888 / user@example.com / 대국민 공개"))


def test_gongmun_head_and_end_meta():
    doc = _run(_doc(_gongmun(), _page(2, ("text", (70, 400, 520, 430), "담당: 홍길동\n전화: 044-000-1234"))))
    b = doc.pages[0].blocks
    assert [(x.type, x.keep, x.drop_reason) for x in b] == [
        ("doc_meta", False, "meta"), ("text", True, None), ("text", True, None), ("text", True, None),
        ("seal", False, "stamp"), ("doc_meta", False, "meta"), ("doc_meta", False, "meta"), ("doc_meta", False, "meta")]
    assert b[0].meta == {"to": "○○기관장(정보화과장)", "title": "2026년 정보보안 교육 실시 알림"}
    assert b[0].provenance == [{"rule": "doc_meta", "type": "text"}]
    assert b[5].meta == {"drafter": "주무관 홍길동", "reviewer": "사무관 김철수", "approver": "정보화과장 이영희"}
    assert b[6].meta == {"date": "2026-03-05", "doc_no": "정보화과-1234", "address": "30128 ○○시 ○○로 12"}
    assert b[7].meta == {"phone": "044-000-1234", "fax": "044-000-8888"}
    assert doc.pages[1].blocks[0].keep                                  # 둘째 쪽 가운데 연락처 줄은 공문 메타 자리가 아니다
    assert [r["block_id"] for r in jsonl_rows(doc)] == ["p1.b02", "p1.b03", "p1.b04", "p2.b01"]
    assert all(p.keep and p.page_class == "normal" for p in doc.pages)


def test_promote_doc_meta_without_person_fields():
    doc = _doc(_gongmun(), _page(2, BODY))
    assert _run(doc).meta == {"title": "2026년 정보보안 교육 실시 알림", "doc_no": "정보화과-1234", "date": "2026-03-05",
                              "has_stamp": True}                            # 담당자·전화는 올리지 않는다(D15)
    person = _run(doc, MetaCfg(promote_person_fields=True)).meta
    assert person["drafter"] == "주무관 홍길동" and person["phone"] == "044-000-1234" and person["fax"] == "044-000-8888"
    assert "to" not in person and "address" not in person              # promote 목록 밖
    assert "has_stamp" not in _run(_doc(_page(1, BODY), _page(2, BODY))).meta


def test_draft_form_tables():
    """사내 기안서 — 위쪽 결재란 표와 메타 표, 그 아래 본문(결재란이 위에 있으면 아래 본문은 공문 끝 자리가 아니다)."""
    page = _page(1, ("table", (380, 40, 550, 110), APPROVAL), ("table", (60, 120, 550, 200), META_TABLE),
                 ("text", (60, 220, 550, 300), "1. 목적: 노후 장비 교체\n2. 내용: 업무용 노트북 5대 구매"),
                 ("text", (60, 400, 550, 430), "담당: 홍길동\n전화: 02-000-1234"))
    doc = _run(_doc(page, _page(2, BODY)))
    b = doc.pages[0].blocks
    assert [(x.type, x.keep, x.drop_reason) for x in b] == [("approval_box", False, "approval"), ("doc_meta", False, "meta"),
                                                            ("text", True, None), ("text", True, None)]
    assert b[0].provenance == [{"rule": "approval_box", "type": "table"}] and b[0].html == APPROVAL
    assert b[1].meta == {"doc_no": "경영지원-2026-012", "draft_date": "2026-03-05", "issuer": "정보화팀", "to": "내부결재",
                         "cc": "재무팀장", "title": "업무용 노트북 구매 품의"}           # 날짜는 이름별로(기안일)
    assert doc.meta == {"doc_no": "경영지원-2026-012", "date": "2026-03-05", "issuer": "정보화팀",
                        "title": "업무용 노트북 구매 품의"}
    assert doc.pages[0].keep and doc.pages[0].page_class == "normal"


def test_approval_box_variants():
    dated = "<table><tr><td>담당</td><td>과장</td><td>부장</td></tr><tr><td>3/5</td><td>3/6</td><td></td></tr></table>"
    unsigned = "<table><tr><td>담당</td><td>팀장</td><td>대표</td></tr><tr><td></td><td></td><td></td></tr></table>"
    doc = _run(_doc(_page(1, BODY), _page(2, ("table", (300, 40, 550, 100), dated), ("table", (300, 620, 550, 680), unsigned),
                                         BODY)))
    assert [b.type for b in doc.pages[1].blocks] == ["approval_box", "approval_box", "text"]   # 날짜 칸·빈 서명 줄


def test_tables_that_only_look_like_approval_or_meta_are_kept():
    duty = ("<table><tr><td>직위</td><td>성명</td><td>담당 업무</td></tr><tr><td>팀장</td><td>홍길동</td><td>사업 총괄</td></tr>"
            "<tr><td>과장</td><td>김철수</td><td>예산·계약</td></tr></table>")
    contact = ("<table><tr><td>구분</td><td>담당부서</td><td>담당자</td><td>전화번호</td></tr>"
               "<tr><td>시스템</td><td>정보화팀</td><td>홍길동 팀장</td><td>02-000-1234</td></tr>"
               "<tr><td>예산</td><td>재무팀</td><td>김철수 과장</td><td>02-000-5678</td></tr></table>")
    tall = ("<table><tr><td>결재</td><td>담당</td><td>팀장</td></tr><tr><td></td><td>홍길동</td><td></td></tr>"
            "<tr><td>협조</td><td>재무팀장</td><td></td></tr><tr><td></td><td></td><td></td></tr>"
            "<tr><td>비고</td><td></td><td></td></tr></table>")
    long_cell = ("<table><tr><td>제목</td><td>정보보안 교육 계획</td></tr><tr><td>수신</td><td>내부결재</td></tr><tr><td>내용</td><td>"
                 + "정보보안 인식을 높이기 위하여 전 직원을 대상으로 교육을 실시하고자 합니다. " * 3 + "</td></tr></table>")
    overview = ("<table><tr><td>사업명</td><td>차세대 정보시스템 구축 사업</td></tr><tr><td>사업 기간</td><td>2026. 3. ~ 2026. 12.</td></tr>"
                "<tr><td>사업 예산</td><td>1,500백만원</td></tr><tr><td>시행일</td><td>2026. 3. 5.</td></tr>"
                "<tr><td>전화</td><td>02-000-1234</td></tr></table>")              # 메타 줄 두 개가 든 사업 개요 표
    page = _page(1, ("table", (60, 40, 550, 100), duty), ("table", (60, 110, 550, 170), contact),
                 ("table", (60, 180, 300, 260), tall), ("table", (60, 270, 550, 320), long_cell),
                 ("table", (60, 325, 550, 335), overview), BODY)
    doc = _run(_doc(page, _page(2, BODY)))
    assert [(b.type, b.keep) for b in doc.pages[0].blocks] == [("table", True)] * 5 + [("text", True)]
    assert doc.meta == {}


def test_body_text_with_meta_words_is_kept():
    sentence = ("text", (70, 60, 520, 90), "민원 접수 및 처리 담당 부서의 업무 절차를 개선하고 전화 상담을 늘린다.")
    mixed = ("text", (70, 100, 520, 250), "수신: 내부결재\n참조: 정보화팀장\n"
             "정보보안 인식을 높이기 위하여 전 직원을 대상으로 교육을 실시하고자 하오니 검토하여 주시기 바랍니다.")
    one = ("text", (70, 260, 520, 280), "제목: 2026년 정보보안 교육 계획")
    contact = ("text", (70, 285, 520, 300), "담당: 정보화과 홍길동\n전화: 02-000-1234")      # 안내문 위 문의처 — 문서 항목이 없다
    doc = _run(_doc(_page(1, sentence, mixed, one, contact), _page(2, BODY)))
    assert all(b.type == "text" and b.keep for b in doc.pages[0].blocks)   # 줄 가운데 낱말·본문이 많은 블록·항목 하나·연락처뿐


def test_wrapped_title_value():
    head = ("text", (70, 60, 520, 120), "수신  ○○기관장\n(경유)\n제목  2026년도 공공기관 정보보안 관리실태 평가 결과에 따른\n"
            "후속 조치 계획 제출 요청")
    b = _run(_doc(_page(1, head, BODY), _page(2, BODY))).pages[0].blocks[0]
    assert b.type == "doc_meta" and b.meta["title"] == "2026년도 공공기관 정보보안 관리실태 평가 결과에 따른 후속 조치 계획 제출 요청"


def test_meta_below_low_approval_box():
    """'끝.' 이 없어도 쪽 아래쪽 결재란 밑은 공문 끝 메타 자리 — 결재란 위 본문은 아니다."""
    page = _page(2, ("text", (70, 80, 520, 120), "담당: 홍길동\n전화: 044-000-1234"),
                 ("table", (300, 600, 550, 660), APPROVAL),
                 ("text", (70, 680, 520, 710), "시행 정보화과-77 (2026. 4. 1.)\n전화 044-000-1234"))
    doc = _run(_doc(_page(1, BODY), page))
    assert [b.type for b in doc.pages[1].blocks] == ["text", "approval_box", "doc_meta"]
    assert doc.pages[1].blocks[2].meta == {"date": "2026-04-01", "doc_no": "정보화과-77", "phone": "044-000-1234"}


def test_nup_positions_use_logical_page():
    meta = ("text", (60, 440, 540, 470), "수신: 내부결재\n참조: 정보화팀장")
    lower = NUp(cols=1, rows=2, cell=(0, 1), region=(0, 421, 595, 842))        # 시트 아래 칸 논리 페이지
    assert _run(_doc(_page(1, meta, n_up=lower), _page(2, BODY))).pages[0].blocks[0].type == "doc_meta"   # 논리 쪽 위 19pt
    assert _run(_doc(_page(1, meta), _page(2, BODY))).pages[0].blocks[0].type == "text"      # 시트라면 쪽 가운데(440pt)

    upper = NUp(cols=1, rows=2, cell=(0, 0), region=(0, 0, 595, 421))          # 시트 위 칸 — 논리 쪽 높이 421pt
    blocks = (("text", (70, 40, 520, 200), "가나다라마바사 " * 20), ("table", (300, 250, 550, 300), APPROVAL),
              ("text", (70, 310, 520, 330), "시행 정보화과-77 (2026. 4. 1.)\n전화 044-000-1234"))
    nup = _run(_doc(_page(1, BODY), _page(2, *blocks, n_up=upper)))
    assert [b.type for b in nup.pages[1].blocks] == ["text", "approval_box", "doc_meta"]   # 결재란이 논리 쪽 아래 절반
    sheet = _run(_doc(_page(1, BODY), _page(2, *blocks)))
    assert [b.type for b in sheet.pages[1].blocks] == ["text", "approval_box", "text"]     # 시트로 재면 위 절반


def _t(rows):
    return "<table>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>"


SCHEDULE = _t([["일자", "내용", "담당"], ["3/5", "착수 보고", "팀장"], ["3/20", "중간 보고", "과장"]])


def test_body_tables_are_not_approval_boxes():
    """직위가 세로로 늘어선 본문 표·결재 낱말만 있는 처리 절차 표는 결재란이 아니다 — 같은 행에 직위 칸이 2개 이상이어야 한다."""
    tables = [SCHEDULE,
              _t([["안건", "담당", "결과"], ["검토 의견 반영", "기획팀", "완료"], ["결재 절차 간소화", "총무팀", "진행 중"]]),
              _t([["접수", "검토", "결재", "시행"]]),
              _t([["직위", "성명", "담당 업무"], ["팀장", "홍길동", "사업 총괄"], ["과장", "김철수", "예산·계약"], ["", "", ""]]),
              _t([["번호", "조치 사항", "담당", "기한"], ["1", "검토 결과 보고", "정보화팀", "3/15"]]),
              _t([["단계", "접수", "검토", "결재"], ["기간", "1일", "3일", "1일"]]),
              _t([["구분", "협조 요청 내용", "담당", "회신일"], ["예산", "협조 요청 완료", "재무팀", "3/10"]])]
    doc = _run(_doc(_page(1, BODY), _page(2, *[("table", (70, 60 + 50 * i, 520, 100 + 50 * i), x) for i, x in enumerate(tables)])))
    assert [(b.type, b.keep) for b in doc.pages[1].blocks] == [("table", True)] * len(tables)
    stamped = ('<table><tr><td>담당</td><td>팀장</td><td>단장</td></tr><tr><td>○○○</td>'
               '<td><img src="imgs/p1.t01.png" alt="○○○" /></td><td>전결 3/5</td></tr></table>')   # 합성 공문 실측 결재란
    doc = _run(_doc(_page(1, ("table", (300, 640, 525, 690), stamped)), _page(2, BODY)))
    assert doc.pages[0].blocks[0].type == "approval_box"


def test_body_table_cascades():
    """결재란 오판이 쪽 삭제·공문 끝 자리·표지 판정으로 번지지 않는다."""
    only = _run(_doc(_page(1, BODY), _page(2, ("table", (70, 100, 520, 300), SCHEDULE)), _page(3, BODY))).pages[1]
    assert (only.page_class, only.keep) == ("normal", True)
    notice = ("text", (70, 580, 520, 610), "○ 보고 장소: 본관 3층 대회의실\n○ 담당: 정보화팀 (전화 02-000-1234)")
    low = _run(_doc(_page(1, BODY), _page(2, BODY, ("table", (70, 450, 520, 560), SCHEDULE), notice), _page(3, BODY)))
    assert [(b.type, b.keep) for b in low.pages[1].blocks] == [("text", True), ("table", True), ("text", True)]
    first = _page(1, ("doc_title", (80, 60, 520, 100), "2026년 정보화 추진 계획"), ("text", (250, 110, 350, 125), "2026. 3."),
                  ("table", (80, 170, 520, 260), SCHEDULE))
    p = _run(_doc(first, _page(2, BODY))).pages[0]
    assert (p.page_class, p.keep) == ("normal", True)                   # M2 처럼 표가 있으면 표지가 아니다


def test_meta_with_body_lines_is_kept():
    """메타 줄 밖에 문장·항목 줄(표는 문장 칸)이 있으면 지우지 않고 값만 block.meta 에 — 승격은 그대로 한다."""
    draft = ("<table><tr><td>문서번호</td><td>경영지원-2026-012</td><td>기안일자</td><td>2026. 3. 5.</td></tr>"
             "<tr><td>발 신</td><td>정보화팀</td><td>수 신</td><td>내부결재</td></tr>"
             '<tr><td>제 목</td><td colspan="3">업무용 노트북 구매 품의</td></tr>'
             '<tr><td>내 용</td><td colspan="3">노후 노트북 5대를 교체하고자 하오니 재가하여 주시기 바랍니다.</td></tr></table>')
    doc = _run(_doc(_page(1, ("table", (380, 40, 550, 110), APPROVAL), ("table", (60, 120, 550, 260), draft)), _page(2, BODY)))
    t = doc.pages[0].blocks[1]
    assert (t.type, t.keep, t.provenance, doc.pages[0].page_class) == ("table", True, [{"rule": "meta_values"}], "normal")
    assert doc.meta == {"doc_no": "경영지원-2026-012", "issuer": "정보화팀", "title": "업무용 노트북 구매 품의", "date": "2026-03-05"}
    head = ("text", (70, 60, 520, 130), "수신: 내부결재\n참조: 정보화팀장\n제목: 업무용 노트북 구매\n노후 노트북 5대를 구매하고자 합니다.")
    b = _run(_doc(_page(1, head, BODY), _page(2, BODY))).pages[0].blocks[0]
    assert (b.type, b.keep, b.meta) == ("text", True, {"to": "내부결재", "cc": "정보화팀장", "title": "업무용 노트북 구매"})
    related = ("text", (70, 60, 520, 130), "수신 ○○기관장(정보화과장)\n(경유)\n제목 2026년 정보보안 교육 실시 알림\n"
               "1. 관련: 정보화과-1001(2026. 2. 20.)")
    doc = _run(_doc(_page(1, related, BODY), _page(2, BODY)))
    assert doc.pages[0].blocks[0].keep and doc.meta == {"title": "2026년 정보보안 교육 실시 알림"}   # 항목 줄은 제목에 붙이지 않는다


def test_split_head_lines_footer_meta_and_issuer_line():
    """합성 공문 실측 블록(PaddleOCR-VL) — 위 메타가 줄마다 다른 블록, 행정기관명 줄, 시행·접수 줄을 꼬리말로 잡았다."""
    real = [
        ("paragraph_title", (246.5, 63.0, 349.0, 83.5), "한국정보공사"),
        ("text", (69.5, 119.0, 142.5, 132.5), "수신 각 부서장"),
        ("text", (70.5, 139.0, 101.0, 153.5), "(경유)"),
        ("text", (69.5, 158.5, 197.5, 173.0), "제목 정보화 사업 협조 요청"),
        ("text", (69.5, 209.0, 258.0, 223.0), "1. 관련: 정보화추진단-1234(2026. 3. 2.)"),
        ("text", (70.5, 231.0, 444.5, 312.5), "2. 올해 정보화 사업을 다음과 같이 추진하고자 하오니 업무에 참고하시기 바랍니다."),
        ("text", (70.0, 339.0, 204.5, 353.0), "붙임 사업 추진 계획 1부. 끝."),
        ("footer", (69.0, 722.5, 343.0, 751.0), "시행 정보화추진단-1300 (2026. 3. 5.) 접수 우 04520 서울특별시 중구 세종대로 "
                                                 "/ 전화 02-000-0000 / 공개구분 공개")]
    page = Page(page_no=1, sheet_no=1, size=(595.28, 841.89),
                blocks=[Block(id=f"p1.b{i + 1:02d}", type=t, bbox=bb, text=x) for i, (t, bb, x) in enumerate(real)])
    doc = _run(_doc(page, _page(2, BODY)))
    b = doc.pages[0].blocks
    assert [(x.type, x.keep) for x in b] == [("doc_meta", False)] * 4 + [("text", True)] * 3 + [("doc_meta", False)]
    assert [x.meta for x in b[:4]] == [{"issuer": "한국정보공사"}, {"to": "각 부서장"}, None, {"title": "정보화 사업 협조 요청"}]
    assert b[7].provenance == [{"rule": "doc_meta", "type": "footer"}] and b[7].meta["doc_no"] == "정보화추진단-1300"
    assert doc.meta == {"issuer": "한국정보공사", "title": "정보화 사업 협조 요청", "doc_no": "정보화추진단-1300", "date": "2026-03-05"}
    form = _page(1, ("paragraph_title", (250, 60, 350, 80), "협조전"), ("text", (70, 110, 520, 125), "수신 재무팀장"),
                 ("text", (70, 130, 520, 145), "제목 노트북 구매 예산 협조 요청"), BODY)
    assert _run(_doc(form, _page(2, BODY))).pages[0].blocks[0].type == "paragraph_title"   # 서식 이름은 기관명이 아니다


def test_meta_rows_inside_approval_tables():
    combo = ('<table><tr><td rowspan="2">결 재</td><td>담당</td><td>팀장</td><td>대표</td></tr>'
             "<tr><td>홍길동</td><td>3/5</td><td>전결</td></tr>"
             "<tr><td>문서번호</td><td>경영-2026-12</td><td>기안일자</td><td>2026. 3. 5.</td></tr></table>")
    doc = _run(_doc(_page(1, ("table", (60, 40, 550, 140), combo), BODY), _page(2, BODY)))
    assert doc.pages[0].blocks[0].type == "approval_box" and doc.pages[0].blocks[0].meta == {"doc_no": "경영-2026-12",
                                                                                           "draft_date": "2026-03-05"}
    assert doc.meta == {"doc_no": "경영-2026-12", "date": "2026-03-05"}
    line = _t([["기안자", "주무관 홍길동", "검토자", "사무관 김철수", "결재권자", "과장 이영희"],
               ["시행", "정보화과-1234", "2026. 3. 5.", "접수", "", ""]])                # 공문 끝 표 모양 결재선 + 시행 행
    doc = _run(_doc(_page(1, ("text", (70, 100, 520, 300), "1. 교육을 실시하오니 참석하여 주시기 바랍니다.\n붙임 교육 계획 1부.  끝."),
                          ("table", (70, 700, 520, 740), line)), _page(2, BODY)))
    assert doc.pages[0].blocks[1].type == "doc_meta" and doc.meta == {"doc_no": "정보화과-1234", "date": "2026-03-05"}


def test_date_priority_and_first_document_unit():
    dates = _t([["문서번호", "경영-2026-12", "기안일자", "2026. 3. 2."], ["시행일자", "2026. 3. 5.", "수신", "내부결재"]])
    assert _run(_doc(_page(1, ("table", (60, 40, 550, 120), dates), BODY), _page(2, BODY))).meta["date"] == "2026-03-05"
    cover = _page(1, ("doc_title", (80, 240, 520, 300), "○○ 시스템 고도화\n결과 보고서"), ("text", (250, 440, 350, 460), "2026년 7월"))
    attach = _page(3, ("paragraph_title", (80, 60, 300, 80), "붙임 3  관련 공문 사본"),
                   ("text", (70, 100, 520, 300), "2. 고도화 사업 착수를 알리오니 협조하여 주시기 바랍니다.\n붙임 사업 계획서 1부.  끝."),
                   ("text", (70, 700, 520, 712), "시행 정보화과-1234 (2025. 12. 1.)  접수"))
    doc = _run(_doc(cover, _page(2, BODY), attach))
    assert doc.pages[2].blocks[2].type == "doc_meta"
    assert doc.meta == {"title": "○○ 시스템 고도화 결과 보고서", "date": "2026-07"}   # 붙임에 실린 다른 공문의 날짜·번호가 아니다


def test_issuer_without_person_info():
    """D15 — 발신 값의 담당자 이름·직급·연락처는 떼고 기관만 올린다."""
    fax = ("text", (70, 60, 520, 120), "수신: ○○기관 정보화과장\n발신: 한국정보공사 정보화팀 홍길동 (☎ 02-000-1234)\n제목: 협의회 회의 자료 송부")
    memo = ("text", (70, 60, 520, 120), "수신: 재무팀장\n발신: 정보화팀 홍길동 과장 / 010-0000-1234\n제목: 노트북 구매 예산 협조 요청")
    tel = ("text", (70, 60, 520, 120), "수신: 재무팀장\n발신: 정보화팀 홍길동 (전화 02-000-1234)\n제목: 노트북 구매 예산 협조 요청")
    alone = ("text", (70, 60, 520, 120), "수신: 재무팀장\n발신: 주무관 홍길동 (☎ 02-000-1234)\n제목: 노트북 구매 예산 협조 요청")
    won = ("text", (70, 60, 520, 120), "수신: 재무팀장\n발신: 정보화팀 박수원 연구원 (☎ 02-000-1234)\n제목: 노트북 구매 예산 협조 요청")
    assert [_run(_doc(_page(1, x, BODY), _page(2, BODY))).meta.get("issuer") for x in (fax, memo, tel, alone, won)] == [
        "한국정보공사 정보화팀", "정보화팀", "정보화팀", None, "정보화팀"]               # '…원' 으로 끝나는 이름·직급도 뗀다


def test_issuer_name_is_dropped_without_contact():
    """[5] 남은 것 — 연락처·직급 낱말이 없어도 기관 이름으로 끝나는 앞 낱말까지만 올린다(이름은 올리지 않는다, D15).
    기관장 직함('…장관'·'…팀장')은 떼고 기관으로, 기관 이름이 없으면 올리지 않는다."""
    values = ("정보화팀 홍길동", "정보화팀장 홍길동", "한국정보공사 정보화팀 홍길동", "홍길동", "행정안전부장관", "한국정보공사 정보화팀",
              "주식회사 가나정보 홍길동", "(주)가나정보")
    blocks = [("text", (70, 60, 520, 120), f"수신: 재무팀장\n발신: {v}\n제목: 노트북 구매 예산 협조 요청") for v in values]
    assert [_run(_doc(_page(1, x, BODY), _page(2, BODY))).meta.get("issuer") for x in blocks] == [
        "정보화팀", "정보화팀", "한국정보공사 정보화팀", None, "행정안전부", "한국정보공사 정보화팀",
        "주식회사 가나정보", "(주)가나정보"]                                    # 회사 꼴 뒤 이름은 회사 이름이다


def test_body_tables_with_post_words_in_header_are_kept():
    """[0] 남은 오판 — 머리행에 직위·결재 낱말 둘('담당부서·협조부서'·'검토의견·담당자'·'담당·검토')과 날짜·'일자' 칸이 있는
    본문 표. 결재란 행은 글 칸이 모두 직위·결재 칸이어야 한다(칸 낱말 뒤는 '자·권자·이사'만, 서명 낱말은 칸 전체일 때만)."""
    tables = [_t([["세부 과제", "담당부서", "협조부서", "완료 시기"], ["데이터 표준화", "정보화팀", "기획팀", "2026.6."],
                  ["검색 고도화", "정보화팀", "홍보팀", "2026.12."]]),
              _t([["번호", "검토의견", "담당자", "처리일자"], ["1", "용어 통일 필요", "정보화팀", "3/10"], ["2", "예산 근거 보완", "재무팀", "3/12"]]),
              _t([["과제", "담당", "검토", "기한"], ["시스템 개선", "정보화팀", "기획팀", "3/31"]]),
              _t([["일자", "담당", "팀장"], ["3/5(목)", "홍길동", "김철수"], ["3/6(금)", "이영희", "박민수"]]),     # 당직표
              _t([["구분", "담당", "검토"], ["2025년", "12.5", "3.2"]])]                                        # 소수 칸 통계표
    doc = _run(_doc(_page(1, BODY), _page(2, *[("table", (70, 60 + 60 * i, 520, 110 + 60 * i), x) for i, x in enumerate(tables)])))
    assert [(b.type, b.keep) for b in doc.pages[1].blocks] == [("table", True)] * len(tables)
    only = _run(_doc(_page(1, BODY), _page(2, ("table", (70, 100, 520, 300), tables[0])), _page(3, BODY)))
    assert (only.pages[1].page_class, only.pages[1].keep) == ("normal", True)      # 표만 있는 쪽(PPT 쪽 꼴)도 결재 쪽이 아니다
    assert any("데이터 표준화" in (r.get("html") or "") for r in jsonl_rows(only))


def test_approval_boxes_with_named_posts_or_without_ranks():
    """[0] 회귀 — 직위와 이름이 한 칸인 결재란('담당 홍길동'), 직위 없이 '기안|검토|결재' 에 서명·날짜 행이 붙은 결재란,
    회사·공공 직급 결재란('대리'·'주무관')은 결재란이다 — 결재자 이름이 본문에 나가지 않는다."""
    boxes = [_t([["담당 홍길동", "팀장 김철수", "부장 이영희"], ["3/5", "3/5", "전결 3/6"]]),
             _t([["기안", "검토", "결재"], ["홍길동", "김철수", "이영희"], ["3/5", "3/5", "3/6"]]),
             _t([["담당", "대리", "과장", "부장"], ["홍길동", "김철수", "이영희", "박민수"], ["3/4", "3/4", "3/5", "3/5"]]),
             _t([["담당", "주무관", "팀장"], ["홍길동", "", ""], ["", "", ""]])]
    for box in boxes:
        doc = _run(_doc(_page(1, ("table", (380, 40, 550, 110), box), ("text", (60, 130, 550, 300), "1. 목적: 노후 장비 교체")),
                        _page(2, BODY)))
        assert doc.pages[0].blocks[0].type == "approval_box", box
        assert not any("홍길동" in (r.get("html") or "") for r in jsonl_rows(doc))


def test_terse_body_line_and_body_cell_are_kept():
    """[3] 남은 것 — 개조식 끝맺음 본문 줄('… 예정임.')과 끝맺음 없는 '내 용' 칸 글(첫 쪽 위 기안 표)도 본문이다.
    지우지 않고 값만 block.meta 에 남기며, 본문 줄을 제목에 붙이지 않는다."""
    head = ("text", (70, 60, 520, 120), "수신: 각 부서장\n제목: 정보보안 교육\n교육은 3월 중 2회 실시할 예정임.")
    doc = _run(_doc(_page(1, head, BODY), _page(2, BODY)))
    b = doc.pages[0].blocks[0]
    assert (b.type, b.keep, b.meta) == ("text", True, {"to": "각 부서장", "title": "정보보안 교육"})
    assert doc.meta == {"title": "정보보안 교육"}
    draft = ("<table><tr><td>문서번호</td><td>경영지원-2026-012</td><td>기안일자</td><td>2026. 3. 5.</td></tr>"
             "<tr><td>발 신</td><td>정보화팀</td><td>수 신</td><td>내부결재</td></tr>"
             '<tr><td>제 목</td><td colspan="3">업무용 노트북 구매 품의</td></tr>'
             '<tr><td>내 용</td><td colspan="3">노후 노트북 5대 교체 구매(총 500만 원, 자산취득비)</td></tr></table>')
    doc = _run(_doc(_page(1, ("table", (380, 40, 550, 110), APPROVAL), ("table", (60, 120, 550, 260), draft)), _page(2, BODY)))
    t = doc.pages[0].blocks[1]
    assert (t.type, t.keep, t.provenance, doc.pages[0].keep) == ("table", True, [{"rule": "meta_values"}], True)
    assert doc.meta == {"doc_no": "경영지원-2026-012", "issuer": "정보화팀", "title": "업무용 노트북 구매 품의", "date": "2026-03-05"}
    assert any("자산취득비" in (r.get("html") or "") for r in jsonl_rows(doc))


def test_head_tables_with_other_name_cells_are_kept():
    """[9](나) 회귀 — 이름만 든 칸의 값은 바로 옆 한 칸('시행' 만 날짜 칸까지)이다. 출장 보고·회의록 머리 표의 다른 이름 칸
    ('출장지'·'장소')을 값으로 빨아들여 표째 지우거나 제목을 더럽히지 않는다 — 메타 칸이 절반이 못 되는 본문 표다."""
    trip = _t([["제목", "해외 출장 결과 보고", "출장지", "미국 워싱턴 D.C. 세계은행 본부"],
               ["시행일자", "2026. 3. 5.", "출장 목적", "디지털 정부 협력 방안 논의 및 우수 사례 조사"]])
    minutes = _t([["제목", "정보화 추진 협의회 결과", "일시", "2026. 3. 5.(목) 14:00~16:00"],
                  ["작성일", "2026. 3. 6.", "장소", "본관 3층 대회의실(화상 병행)"], ["참석", "정보화팀장 외 6명", "", ""]])
    for table, body in ((trip, "세계은행"), (minutes, "대회의실")):
        doc = _run(_doc(_page(1, ("table", (60, 60, 550, 160), table), BODY), _page(2, BODY)))
        assert (doc.pages[0].blocks[0].type, doc.pages[0].blocks[0].keep, doc.meta) == ("table", True, {})
        assert any(body in (r.get("html") or "") for r in jsonl_rows(doc))


def test_item_lines_are_not_head_meta():
    """통합(1) 회귀 — 항목 기호·번호로 시작하는 '이름: 값' 줄('○ 접수: …'·'가. 접수: …')은 본문 항목이다. 첫 쪽 위 묶음은
    공문 머리 항목(수신·참조·경유·제목·발신·문서번호) 블록만 잇는다 — 접수·시행 줄 블록은 지우지 않고 문서 날짜로도 올리지 않는다."""
    notice = _page(1, ("doc_title", (70, 50, 520, 80), "2026년 상반기 정보보안 교육 안내"),
                   ("text", (70, 100, 520, 115), "○ 교육 대상: 전 직원(필수)"),
                   ("text", (70, 120, 520, 135), "○ 접수: 3. 2.(월) ~ 3. 13.(금), 전자우편 접수"),
                   ("text", (70, 140, 520, 155), "○ 시행: 2026. 4. 1.부터 온라인 과정 운영"), BODY)
    letter = _page(1, ("text", (70, 80, 520, 95), "수신  각 부서장"), ("text", (70, 100, 520, 115), "제목  정보보안 교육 수강 신청 안내"),
                   ("text", (70, 140, 520, 155), "가. 접수: 2026. 3. 2.(월) ~ 3. 13.(금) 18:00까지"),
                   ("text", (70, 160, 520, 175), "나. 시행일: 2026. 4. 1.(수)부터 과정별 운영"), BODY)
    bare = _page(1, ("text", (70, 100, 520, 115), "접수: 3. 2.(월) ~ 3. 13.(금), 전자우편 접수"),
                 ("text", (70, 120, 520, 135), "시행: 2026. 4. 1.부터 온라인 과정 운영"), BODY)
    one = _page(1, ("text", (70, 120, 520, 155), "○ 접수: 3. 2.(월) ~ 3. 13.(금), 전자우편 접수\n○ 시행: 2026. 4. 1.부터 온라인 과정 운영"), BODY)
    for page in (notice, letter, bare, one):
        doc = _run(_doc(page, _page(2, BODY)))
        items = [b for b in doc.pages[0].blocks if "접수:" in (b.text or "") or "시행" in (b.text or "")]
        assert items and all(b.type == "text" and b.keep for b in items) and "date" not in doc.meta, page.blocks[0].text
    doc = _run(_doc(letter, _page(2, BODY)))
    assert [b.type for b in doc.pages[0].blocks[:2]] == ["doc_meta", "doc_meta"] and doc.meta == {"title": "정보보안 교육 수강 신청 안내"}


def test_event_title_above_head_meta_is_not_issuer():
    """기관명 판정 회귀 — 첫 쪽 맨 위 짧은 제목이 행사 이름('주간 업무보고회')이면 행정기관명 줄이 아니다(지우지 않고 issuer 로도 올리지 않는다)."""
    memo = _page(1, ("paragraph_title", (220, 40, 380, 60), "주간 업무보고회"), ("text", (70, 80, 520, 95), "수신: 각 팀장"),
                 ("text", (70, 100, 520, 115), "제목: 3월 2주차 업무 보고 자료 제출"), BODY)
    doc = _run(_doc(memo, _page(2, BODY)))
    assert [(b.type, b.keep) for b in doc.pages[0].blocks[:3]] == [("paragraph_title", True), ("doc_meta", False), ("doc_meta", False)]
    assert doc.meta == {"title": "3월 2주차 업무 보고 자료 제출"}


def test_press_release_date():
    doc = _run(_doc(_page(1, ("header", (244, 82, 329, 98), "보도참고자료"),
                          ("header", (200, 112, 243, 126), "보도시점"), ("text", (272, 112, 354, 126), "2026. 6. 22.(월)"),
                          ("header", (390, 112, 535, 126), "배포 2026. 6. 23.(화) 18:30"),
                          ("doc_title", (154, 138, 448, 164), "전국 말라리아 주의보 발령"), BODY)))
    b = doc.pages[0].blocks
    assert (b[2].type, b[2].keep, b[2].meta) == ("doc_meta", False, {"date": "2026-06-22"})   # 버린 꼬리표 옆 날짜 칸
    assert doc.meta["date"] == "2026-06-22"


def test_press_release_date_prefers_release_over_distribution():
    doc = _run(_doc(_page(1, ("text", (60, 112, 240, 126), "배포 2026. 6. 21.(일) 10:00"),
                          ("text", (300, 112, 540, 126), "보도일시 2026. 6. 22.(월) 12:00"), BODY)))
    b = doc.pages[0].blocks
    assert [(x.type, x.meta) for x in b[:2]] == [("doc_meta", None), ("doc_meta", {"date": "2026-06-22"})]
    assert doc.meta["date"] == "2026-06-22"


def test_press_words_in_body_are_not_dates():
    doc = _run(_doc(_page(1, ("text", (70, 110, 520, 124), "보도 자료에 따르면 2026. 3. 1. 시행"), BODY)))
    assert doc.pages[0].blocks[0].keep and "date" not in doc.meta
