"""개인정보 탐지(기획서 4.4·D15) — 종류만 표시하고 값은 남기지 않는다.
번호는 모두 합성이다: 주민번호는 검증번호가 틀린 것, 전화는 국번이 0 으로 시작해 쓰이지 않는 것."""
import pytest

from kdp.enrich.pii import detect_pii
from kdp.schema import Block, Document, Page, ParserInfo, Source


def _page(no, *blocks, keep=True):
    return Page(page_no=no, sheet_no=no, size=(595, 842), keep=keep, blocks=[
        Block(id=f"p{no}.b{i + 1:02d}", bbox=(70, 100 + 40 * i, 520, 130 + 40 * i), **kw) for i, kw in enumerate(blocks)])


def _doc(*pages):
    return Document(doc_id="sha256:ab", source=Source(paths=["a.pdf"], pages=len(pages)),
                    parser=ParserInfo(kdp_version="0", engine={}, engine_hash="e", assemble_hash="a", run_at="t"),
                    pages=list(pages))


def _b(t="text", **kw):
    return {"type": t, **kw}


def _pii(text=None, t="text", **kw):
    return detect_pii(_doc(_page(1, _b(t, text=text, **kw)))).pages[0].blocks[0].pii


@pytest.mark.parametrize("text", [
    "주민등록번호: 900101-1234567",
    "신청인 900101 – 1234567",                                # 다른 붙임표·띄어 씀
    "외국인등록번호 900101-5234567",
    "000229-3234567",                                         # 2000년 2월 29일(윤년)
    "주민등록번호 9001011234567",                               # 붙여 쓰면 이름표가 있을 때만
    "법인등록번호 110111-1234567, 대표자 주민등록번호 851231-2234567",
])
def test_rrn(text):
    assert _pii(text) == ["rrn"]


@pytest.mark.parametrize("text", [
    "9001011234567",                                          # 이름표 없는 13자리
    "901301-1234567",                                         # 13월
    "900230-1234567",                                         # 2월 30일
    "010229-3234567",                                         # 2001년은 윤년이 아니다
    "900101-9234567",                                         # 뒤 첫 자리 9·0
    "900101-0234567",
    "법인등록번호: 110111-1234567",                              # 꼴이 같은 법인등록번호
    "1900101-1234567",                                        # 더 긴 숫자열의 일부
    "900101-1******",                                         # 가린 번호
])
def test_not_rrn(text):
    assert _pii(text) is None


@pytest.mark.parametrize("text", [
    "010-0000-1234", "010.0000.1234", "010 0000 1234", "01000001234",
    "02-000-1234", "(02) 000-1234", "☎02)000-1234", "031-000-1234", "070-0000-1234", "0505-000-1234",
    "Tel.02-000-1234", "+82-10-0000-1234", "+82 2 000 1234",
    "고객센터 1588-0000", "1588-0000번으로 전화", "TEL 1544 0000",
])
def test_phone(text):
    assert _pii(text) == ["phone"]


@pytest.mark.parametrize("text", [
    "2026-03-15", "2026.03.15.", "26.03.15", "시행 2026. 3. 15.",            # 날짜
    "제2026-123호", "1,234,567원", "사업자등록번호 123-45-67890",          # 문서번호·금액·사업자등록번호
    "- 12 -", "12 / 30", "Ⅲ-2", "09:00-18:00",                          # 쪽번호·시각
    "1588-0000",                                                       # 대표번호는 전화 말이 가까이 있을 때만
    "1800-1900년대 자료", "Intel 1600-1800",                             # 연도·수 범위
    "콜센터 일평균 상담 1500-2000건",                                      # 바로 뒤에 단위
])
def test_not_phone(text):
    assert _pii(text) is None


@pytest.mark.parametrize("text", [
    "입금 계좌: 국민은행 123456-01-123456",
    "123-456-789012 (신한은행)",                                 # 은행 말이 뒤에
    "농협 302-0000-1234-56",
    "계좌번호 1234567890123",
])
def test_account(text):
    assert _pii(text) == ["account"]


@pytest.mark.parametrize("text", [
    "123-456-789012",                                         # 은행 말이 없으면 꼴만으로는 잡지 않는다
    "사업자등록번호 123-45-67890 (거래은행 국민)",
    "이체 금액 1234567890원", "○○은행 제2026-123456호", "은행 대출 1,234,567,890원",
    "ISBN 978-89-12345-67-8, 은행", "은행권 대출 2023-2024-2025 추이",
])
def test_not_account(text):
    assert _pii(text) is None


def test_phone_near_bank_is_not_account():
    assert _pii("○○은행 본점 전화 02-0000-1234") == ["phone"]


def test_table_html():
    people = ("<table><tr><td>성명</td><td>연락처</td><td>주민등록번호</td></tr>"
              "<tr><td>홍길동</td><td>010&nbsp;0000&nbsp;1234</td><td>900101-1234567</td></tr></table>")
    assert _pii(t="table", html=people) == ["phone", "rrn"]
    accounts = ("<table><tr><td>성명</td><td>은행</td><td>계좌번호</td><td>비고</td></tr>"
                "<tr><td>홍길동</td><td>국민</td><td>123456-01-123456</td><td>정기 지급 대상</td></tr></table>")
    assert _pii(t="table", html=accounts) == ["account"]                 # 머리칸이 단서 — 칸이 멀어도 잡는다
    cells = ("<table><tr><td>연락처</td><td>010</td><td>0000</td><td>1234</td></tr>"
             "<tr><td>1588</td><td>0000</td><td>1,234,567</td><td>2026-03-15</td></tr></table>")
    assert _pii(t="table", html=cells) is None                          # 이웃 칸 숫자를 한 번호로 잇지 않는다
    corp = ("<table><tr><td>업체명</td><td>법인등록번호</td></tr><tr><td>가나㈜</td><td>110111-1234567</td></tr>"
            "<tr><td>다라㈜</td><td>131111-2345678</td></tr></table>")
    assert _pii(t="table", html=corp) is None


def test_several_kinds_in_one_block():
    text = "담당 홍길동(010-0000-1234), 입금 계좌 국민은행 123456-01-123456, 주민등록번호 900101-1234567"
    assert _pii(text) == ["account", "phone", "rrn"]
    assert _pii("문의 02-000-1234", figure_text="주민등록번호 900101-1234567") == ["phone", "rrn"]   # 글과 그림 속 글자


def test_scope_values_and_input_untouched():
    phone = "담당 홍길동 전화 02-000-1234"
    doc = _doc(_page(1, _b(text=phone),
                     _b("doc_meta", text="문서번호 제2026-123호 " + phone, keep=False, drop_reason="meta"),
                     _b("footer", text=phone, keep=False, drop_reason="footer"),
                     _b("image", figure_text="문의 ☎ 02-000-1234"),
                     _b(text="2026년 3월 15일 회의 결과")),
               _page(2, _b(text=phone), _b("doc_meta", text=phone, keep=False, drop_reason="meta"), keep=False))
    out = detect_pii(doc)
    # 출력 블록과 doc_meta(버린 쪽에 있어도)만 — 꼬리말·버린 쪽의 블록은 보지 않는다
    assert [[b.pii for b in p.blocks] for p in out.pages] == [[["phone"], ["phone"], None, ["phone"], None],
                                                              [None, ["phone"]]]
    assert all(b.pii is None for p in doc.pages for b in p.blocks)       # 입력은 그대로
    for b in (b for p in out.pages for b in p.blocks):
        rest = str(b.model_dump(exclude={"text", "html", "figure_text"}))
        assert "000-1234" not in rest and b.provenance is None             # 값은 어디에도 적지 않는다
    assert out.pages[0].blocks[0].text == phone                         # 지우거나 가리지 않는다


def test_corp_number_exception_is_per_number():
    """'법인등록번호' 말이 한 번 있어도 대표자 주민번호는 잡는다 — 법인 예외는 그 번호 옆에 법인 말만 있을 때."""
    assert _pii("법인등록번호 110111-1234567, 대표자 홍길동(900101-1234567)") == ["rrn"]
    corp = ("<table><tr><td>상호</td><td>가나㈜</td><td>법인등록번호</td><td>110111-1234567</td></tr>"
            "<tr><td>대표자</td><td>홍길동</td><td>생년월일</td><td>900101-1234567</td></tr></table>")
    assert _pii(t="table", html=corp) == ["rrn"]


@pytest.mark.parametrize("text", [
    "회비 납부: 국민 123456-01-123456 (총무 홍길동)",
    "경조금 송금처 신한 110-123-456789 홍길동",
    "이체: 우리 1002-123-456789",
    "마음 전하실 곳: 신한 110-123-456789",
])
def test_account_transfer_words_and_bank_names(text):
    assert _pii(text) == ["account"]


def test_bank_short_names_only_next_to_number():
    assert _pii("우리 기관 계약번호 2026-123456-01") is None              # 은행 약칭 낱말이 번호에 붙어 있지 않다


def test_table_cues_from_same_row_or_column_header():
    """표의 단서 말은 같은 행과 그 열 머리칸에서만 — 다른 칸의 '센터' 때문에 건수 범위가 전화가 되지 않는다."""
    calls = ("<table><tr><td>센터</td><td>일평균 상담(건)</td></tr><tr><td>서울</td><td>1500-2000</td></tr>"
             "<tr><td>부산</td><td>1600-1800</td></tr></table>")
    servers = "<table><tr><td>데이터센터</td><td>서버 대수(대)</td></tr><tr><td>대전</td><td>1500-1700</td></tr></table>"
    assert _pii(t="table", html=calls) is None and _pii(t="table", html=servers) is None
    rep = ("<table><tr><td>구분</td><td>대표번호</td></tr><tr><td>본사</td><td>1588-0000</td></tr></table>")
    row = "<table><tr><td>고객센터</td><td>1588-0000</td></tr></table>"
    title = ('<table><tr><td colspan="2">직원 연락처</td></tr><tr><td>홍길동</td><td>1588-1234</td></tr></table>')
    assert _pii(t="table", html=rep) == _pii(t="table", html=row) == _pii(t="table", html=title) == ["phone"]


@pytest.mark.parametrize("text", [
    "법인등록번호 110111-1234567, 대표자 홍길동",
    "상호: 가나정보㈜  법인등록번호: 110111-1234567  대표자: 홍길동",
    "가나정보㈜(대표자 홍길동, 법인등록번호 110111-2345678)",
])
def test_corp_number_next_to_representative_is_not_rrn(text):
    """[7] 회귀 — 번호 바로 앞의 마지막 이름표가 '법인등록번호' 면 법인번호다. 옆에 '대표자' 말이 있어도 주민번호로 보지 않는다."""
    assert _pii(text) is None


def test_corp_number_table_row_with_representative():
    row = "<table><tr><td>법인등록번호</td><td>110111-1234567</td><td>대표자</td><td>홍길동</td></tr></table>"
    assert _pii(t="table", html=row) is None                            # 이름 칸(왼쪽 칸·열 머리칸)이 법인등록번호


def test_section_path_numbers_are_stripped_not_flagged():
    """[8] 회귀 — 제목에 든 연락처는 경로에서 떼고(제목 블록 자신은 글 그대로 pii 표시), 그 장의 블록들에는 pii 를 달지 않는다
    (장 제목의 대표번호 하나로 장 전체가 받는 쪽 색인에서 빠지지 않게). 경로에는 번호가 남지 않는다."""
    title = "Ⅲ. 고객센터(1588-1234) 운영 개선 방안"
    contact = "2. 문의처: 정보화팀 홍길동 주무관(010-1234-5678)"
    doc = detect_pii(_doc(_page(1, _b("paragraph_title", text=title, section_path=[title]),
                                _b(text="상담 인력을 늘린다.", section_path=[title, "1. 상담 인력 확충"]),
                                _b(text="담당 업무 안내", section_path=[contact]),
                                _b(text="본문", section_path=["1. 사업 개요", "문의: 정보화팀(☎ 02-000-1234)"]))))
    b = doc.pages[0].blocks
    assert [x.pii for x in b] == [["phone"], None, None, None]
    assert [x.section_path for x in b] == [["Ⅲ. 고객센터 운영 개선 방안"], ["Ⅲ. 고객센터 운영 개선 방안", "1. 상담 인력 확충"],
                                           ["2. 문의처: 정보화팀 홍길동 주무관"], ["1. 사업 개요", "문의: 정보화팀"]]
    assert b[0].text == title                                           # 제목 블록 글은 지우지 않는다


def test_figure_description_checked():
    """그림 설명(D19)은 모델이 쓴 글이라 그림 제목의 번호를 옮겨 적을 수 있다 — 설명도 검사한다."""
    b = detect_pii(_doc(_page(1, _b("image", figure_desc="그림 아래에 문의 전화 02-000-1234 가 적혀 있다.")))).pages[0].blocks[0]
    assert b.pii == ["phone"]
