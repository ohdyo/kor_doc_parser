"""개인정보 탐지(기획서 4.4·D15) — 블록에 든 개인정보의 종류만 pii 에 적는다: rrn(주민·외국인등록번호), phone(전화), account(계좌).
블록 글은 지우거나 가리지 않고 값도 남기지 않는다 — 색인 제외는 받는 쪽 몫이다.

오탐은 받는 쪽에서 멀쩡한 블록을 빼는 과잉 삭제가 되므로, 꼴만으로 애매한 번호(대표번호·계좌·붙임표 없는 주민번호)는
가까이에 그 말(전화·계좌·주민등록번호…)이 있을 때만 잡는다. 표는 칸마다 보고 그 말은 같은 행과 그 열의 머리칸에서 찾는다
(표 전체에서 찾으면 '센터' 말이 든 표의 건수 범위 '1500-2000' 이 전화가 된다).
대상은 출력 블록과 doc_meta 블록이다 — doc_meta 는 keep=false 라도 담당자 이름·전화와 함께 정본 JSON 에 남고,
D15 가 그 개인정보를 메타로 올리지 않고 pii 표시로만 넘기기로 했다. 머리말·꼬리말 같은 다른 버린 블록은 보지 않는다.
섹션 경로는 제목 글의 사본이라 그 안의 번호만 뗀다 — 번호는 pii 를 단 제목 블록에 남는다(경로로 pii 를 달면 장 제목의
대표번호 하나로 그 장 전체가 받는 쪽 색인에서 빠진다).
"""
from __future__ import annotations

import html as htmllib
import re
from datetime import date

from ..assemble.tables import parse_rows
from ..schema import Block, Document

NEAR = 20                     # 단서 말까지 거리(글자) — '계좌번호:'·'(국민은행)' 같은 이름표가 닿는 정도

_D = "-‐‑‒–−－"               # 붙임표 갈래(모델·텍스트 레이어가 여러 가지로 낸다)
_L = rf"(?<!\d)(?<!\d[{_D}.])"                        # 더 긴 숫자열의 일부가 아니다
_R = rf"(?!\d)(?![{_D}.]\d)"
_S = rf"(?: ?[{_D}.] ?| )"                            # 하이픈·점·공백 구분
RRN = re.compile(rf"{_L}(\d\d)(\d\d)(\d\d) ?([{_D}]?) ?([1-8])\d{{6}}{_R}")   # 뒤 첫 자리 1~4 내국인, 5~8 외국인
_AREA = r"(?:1[016789]|2|3[1-3]|4[1-4]|5[1-5]|6[1-4]|70|80|50[2-8])"        # 휴대·지역·인터넷·수신자부담·안심번호(앞 0 뺀 꼴)
PHONE = re.compile(rf"{_L}(?:\+82 ?[{_D}.]? ?(?:\(0\) ?)?|\(?0){_AREA}(?: ?\) ?|{_S})\d{{3,4}}{_S}\d{{4}}{_R}"
                   rf"|{_L}01[016789]\d{{7,8}}{_R}")                        # 휴대전화는 붙여 써도 잡는다
REP = re.compile(rf"{_L}1[568]\d\d{_S}\d{{4}}{_R}")    # 대표번호(15xx·16xx·18xx) — 연도·수 범위(1800-1900)와 꼴이 같다
ACCOUNT = re.compile(rf"{_L}\d{{2,6}}(?:[{_D}]\d{{2,7}}){{1,4}}{_R}|{_L}\d{{10,14}}{_R}")
_NOT_ACCOUNT = re.compile(rf"\d{{3}}[{_D}]\d{{2}}[{_D}]\d{{5}}|\d{{6}}[{_D}]\d{{7}}"   # 사업자등록번호·주민(법인)등록번호 꼴
                          rf"|(?:19|20)\d\d(?:[{_D}](?:19|20)\d\d)+")              # 연도 나열(2023-2024-2025)
_ID = re.compile(r"(?:주민|외국인)\s*(?:등록)?\s*번호")
_CORP = re.compile(r"법인\s*(?:등록)?\s*번호")          # 법인등록번호는 주민번호와 꼴(6-7자리)이 같다
_PERSON_ID = re.compile(r"주민|외국인|대표자")              # 법인번호 옆이어도 이 말이 있으면 사람 번호로 본다
_ID_CUE = re.compile(rf"{_CORP.pattern}|{_PERSON_ID.pattern}")   # 번호 앞 이름표 — 가장 가까운 것이 그 번호의 이름이다
_TEL = re.compile(r"전화|대표\s?번호|연락|문의|센터|국번|☎|☏|\b(?:tel|ars)\b", re.I)
_BANK = re.compile(r"계좌|은행|예금주|입금|송금|이체|납부|뱅크|농협|수협|신협|새마을금고|우체국")
# 번호 바로 앞의 은행 약칭('신한 110-…') — 낱말로는 흔해('우리 기관'·'하나의') 번호에 붙어 있을 때만
_BANK_NAME = re.compile(r"(?:국민|신한|우리|하나|기업|산업|씨티|SC제일|부산|대구|광주|전북|경남|제주|카카오|케이|토스)\s*[:：]?\s*$")
_QTY = re.compile(r"[천만억]?원|년|호|건|명|개|회|%")   # 숫자 바로 뒤 단위 — 금액·연도·문서번호(제n호)·수량
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[^\S\n]+")
# 번호를 뗀 뒤 비거나 연락처 말만 남은 괄호 — '고객센터(1588-1234)'·'(☎ 02-…)'
_EMPTY = re.compile(r"\s*[(（]\s*(?:[☎☏]|tel\.?|전\s*화|연락처|문\s*의|대표\s*번호|휴\s*대\s*전\s*화|fax|팩\s*스)?\s*[:：.]?\s*[)）]", re.I)


def _born(m: re.Match) -> bool:
    """앞 6자리가 실제 날짜인가 — 세기는 뒤 첫 자리로(1·2·5·6 → 1900년대, 3·4·7·8 → 2000년대)."""
    try:
        date((1900 if m[5] in "1256" else 2000) + int(m[1]), int(m[2]), int(m[3]))
    except ValueError:
        return False
    return True


def _account(v: str) -> bool:
    """10~14자리이고 사업자·주민(법인)등록번호나 전화번호 꼴이 아닌 것."""
    return 10 <= sum(c.isdigit() for c in v) <= 14 and not _NOT_ACCOUNT.fullmatch(v) and not PHONE.fullmatch(v)


def _hits(pat: re.Pattern, s: str, ok=None) -> list[re.Match]:
    """ok 를 통과하고 바로 뒤에 단위가 붙지 않은 번호들."""
    return [m for m in pat.finditer(s) if (ok is None or ok(m)) and not _QTY.match(s, m.end())]


def _found(s: str, ctx: str | None = None, label: str = "") -> list[tuple[str, re.Match]]:
    """(종류, 번호) — 단서 말은 ctx(표: 같은 행·열 머리칸)에서, ctx 가 없으면 번호 앞뒤 NEAR 글자 안에서 찾는다.
    label 은 표 칸의 이름 칸(열 머리칸·왼쪽 칸) 글 — 번호 바로 앞 이름표로 함께 본다."""

    def near(m: re.Match, words: re.Pattern) -> bool:
        return bool(words.search(ctx if ctx is not None else s[max(0, m.start() - NEAR):m.end() + NEAR]))

    def rrn(m: re.Match) -> bool:
        """법인등록번호는 그 번호만 뺀다 — 번호 바로 앞의 마지막 이름표가 '법인(등록)번호' 면 법인번호('법인등록번호 …, 대표자 홍길동'),
        '대표자·주민' 이면 사람 번호. 앞 이름표가 없으면 가까이에 법인 말만 있을 때 뺀다."""
        if not (_born(m) and (m[4] or near(m, _ID))):
            return False
        cues = _ID_CUE.findall(label + "\n" + s[max(0, m.start() - NEAR):m.start()])
        return not _CORP.match(cues[-1]) if cues else not (near(m, _CORP) and not near(m, _PERSON_ID))

    def account(m: re.Match) -> bool:
        return _account(m[0]) and (near(m, _BANK) or bool(_BANK_NAME.search(s[max(0, m.start() - NEAR):m.start()])))

    return ([("rrn", m) for m in _hits(RRN, s, rrn)] + [("phone", m) for m in _hits(PHONE, s)]
            + [("phone", m) for m in _hits(REP, s, lambda m: near(m, _TEL))]
            + [("account", m) for m in _hits(ACCOUNT, s, account)])


def _kinds(s: str, ctx: str | None = None, label: str = "") -> set[str]:
    """s 의 개인정보 종류."""
    return {k for k, _ in _found(_WS.sub(" ", s), ctx, label)}


def kinds(text: str | None) -> set[str]:
    """글 하나의 개인정보 종류 — 메타 승격 값 검사(D15)에도 쓴다."""
    return _kinds(text or "")


def _strip(s: str) -> str:
    """경로 글에서 개인정보 번호를 뗀다 — 비게 된 괄호·끝의 '☎'·쌍점도. 번호가 없으면 그대로."""
    t = _WS.sub(" ", s)
    keep, end = [], 0
    for a, b in sorted(m.span() for _, m in _found(t)):
        if t[a:b].count("(") > t[a:b].count(")") and (k := re.match(r"\s*\)", t[b:])):
            b += k.end()                                   # 번호가 연 괄호('(010-…')는 닫는 괄호까지
        if a >= end:
            keep.append(t[end:a])
        end = max(end, b)
    if not end:
        return s
    return re.sub(r"[\s/,:：☎☏]+$", "", " ".join(_EMPTY.sub("", "".join(keep) + t[end:]).split()))


def _cells(html: str):
    """표 칸마다 (글, 단서 글 — 같은 행의 칸들과 그 열의 머리칸, 이름 칸 글 — 열 머리칸과 왼쪽 칸). 병합 칸은 덮는 자리마다 놓고,
    이웃 칸 숫자는 잇지 않는다."""
    rows = parse_rows(html)
    grid: dict[tuple[int, int], str] = {}
    anchors = []
    for i, r in enumerate(rows):
        j = 0
        for c in r:
            while (i, j) in grid:
                j += 1
            anchors.append((i, j, c))
            for di in range(c.rowspan):
                for dj in range(c.colspan):
                    grid.setdefault((i + di, j + dj), c.text)
            j += c.colspan
    if not anchors:                                    # 칸을 못 읽은 표 — 태그만 떼고 글로 본다
        yield htmllib.unescape(_TAG.sub("\n", html)), None, ""
        return
    for i, j, c in anchors:
        row = [t for (a, _), t in sorted(grid.items()) if a == i]
        head = [grid.get((0, j + dj), "") for dj in range(c.colspan)]
        yield c.text, "\n".join(row + head), "\n".join(head + [grid.get((i, j - 1), "")])


def _texts(b: Block):
    """(글, 단서 글, 이름 칸 글) — 글·그림 속 글자는 번호 앞뒤에서, 표는 칸마다 같은 행·열 머리칸에서 단서를 찾는다."""
    for s in (b.text, b.figure_text):
        if s:
            yield s, None, ""
    if b.html:
        yield from _cells(b.html)


def detect_pii(doc: Document) -> Document:
    """출력 블록과 doc_meta 블록의 개인정보 종류를 pii 에 적는다(지우지 않는다). 섹션 경로에 실린 번호는 뗀다."""
    doc = doc.model_copy(deep=True)
    for page in doc.pages:
        for b in page.blocks:
            if b.section_path:
                b.section_path = [_strip(s) for s in b.section_path]
            if (page.keep and b.keep) or b.type == "doc_meta":
                found = {k for s, ctx, label in _texts(b) for k in _kinds(s, ctx, label)}
                if found:
                    b.pii = sorted(found)
    return doc
