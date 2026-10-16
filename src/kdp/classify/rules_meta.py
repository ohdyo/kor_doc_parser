"""공문 메타(doc_meta)·결재란(approval_box) 규칙 — 학습 전 대용(설계 5.2, 기획서 5.2·5.3).

공문 메타는 첫 논리 페이지 위쪽과 공문 본문 끝('끝.' 아래, 쪽 아래쪽 결재란 아래)에서만 찾는다 — '접수·담당·전화' 같은
낱말은 본문에도 흔하다. 항목 이름은 줄·칸 머리에 있거나 뒤에 쌍점이 올 때만 세고, 줄 가운데 이름은 공문 끝 자리에서만 센다.
항목 기호·번호로 시작하는 줄('가. 접수: …')은 메타 줄이 아니다. 메타 줄 밖에 문장·항목·개조식 줄(표는 문장 칸, 첫 쪽 위에서는
이름·값 칸이 아닌 긴 글 칸)이 섞인 블록은 지우지 않고 값만 block.meta 에 남긴다(provenance meta_values).
"""
from __future__ import annotations

import re

from ..assemble.tables import Cell, parse_rows
from ..geometry import local, page_size
from ..schema import Block, Document, Page
from .blocks import ITEM
from .pages import find_date

FIELDS = {          # 항목 이름 → block.meta 키. 글자 사이를 띄워 써도('수 신') 맞춘다
    "문서번호": "doc_no", "관리번호": "doc_no", "생산등록번호": "doc_no", "등록번호": "doc_no",
    "시행일자": "date", "시행일": "date", "작성일자": "write_date", "작성일": "write_date",
    "기안일자": "draft_date", "기안일": "draft_date",
    "시행": "enforce",                      # '처리과-번호 (날짜)' — doc_no·date 로 나눈다
    "접수": "receipt", "수신자": "to", "수신": "to", "경유": "via", "발신명의": "issuer", "발신": "issuer",
    "참조": "cc", "제목": "title", "담당부서": "dept", "담당자": "person", "담당": "person",
    "기안자": "drafter", "검토자": "reviewer", "협조자": "cooperator", "결재권자": "approver",
    "전화번호": "phone", "전화": "phone", "팩스번호": "fax", "팩스": "fax", "전송": "fax",
    "전자우편": "email", "이메일": "email", "주소": "address", "공개구분": "disclosure", "공개여부": "disclosure",
}
DATES = ("date", "write_date", "draft_date")   # 이름별 날짜 — 문서 날짜로는 앞의 것부터 올린다(시행일 > 작성일 > 기안일, D15)
PERSON_FIELDS = {"person", "drafter", "reviewer", "cooperator", "approver", "phone", "fax", "email"}   # 사람 정보(D15)
CONTACT_FIELDS = PERSON_FIELDS | {"address", "dept"}     # 연락처 항목 — 첫 쪽 위에서는 이것만으로 공문 메타로 보지 않는다
HEAD_FIELDS = {"to", "cc", "via", "title", "issuer", "doc_no"}   # 공문 머리 항목 — 줄마다 갈린 블록은 이것만 한 묶음으로 잇는다
MIN_FIELDS = 2              # 서로 다른 메타 항목이 이만큼 있어야 doc_meta
TOP_RATIO = 0.4             # 첫 논리 페이지 위쪽 — 블록 윗변이 쪽 높이의 이 비율 안
LOW_RATIO = 0.5             # 윗변이 이보다 아래인 결재란의 밑은 공문 끝 메타 자리
GAP = 2.0                   # 기준선보다 이만큼(pt) 위에서 시작한 블록까지 '아래'로 본다
META_MAX_CELL = 80          # 칸 글자가 이보다 길면 본문이 든 표다(전자결재 출력은 본문을 표 칸에 넣기도 한다)
BODY_CELL_MIN = 10          # 이름 칸도 값 칸도 아닌 칸의 글(띄어쓰기 뺀)이 이만큼이면 본문 — 첫 쪽 위 기안 표의 '내 용' 칸
TERSE_MIN = 8               # 개조식 끝맺음(…함·음·임·됨)은 이만큼 긴 줄만 본문으로 본다
APPROVAL_MAX_ROWS = 4
APPROVAL_MAX_CELL = 15      # 결재란 칸 글자(띄어쓰기 뺀) 상한 — 직위·이름·날짜·'전결'
# 직위 — 담당·대표·사원~전무·'…장'(팀장·단장·본부장)·'…관'(주무관·사무관)
RANK = r"담당|대표|사원|주임|대리|이사|상무|전무|감사|고문|책임|선임|수석|교감|원감|[가-힣]{1,3}장|[가-힣]{1,2}관"
POSTS = ("결재", "전결", "대결", "협조", "검토", "기안", "승인")   # 결재 낱말 — 이것만 늘어선 행('접수|검토|결재' 처리 절차)은 결재란이 아니다
TRIO = ({"기안", "검토"}, {"결재", "승인"})                     # 직위 없이 '기안|검토|결재(승인)' 가 늘어선 결재선
_WORDS = "|".join(POSTS)
_POST = re.compile(rf"(?P<w>{RANK}|{_WORDS})(?:자|권자|이사)?")   # 직위 칸 — 뒤는 '자·권자·이사'만('담당부서·검토의견' 은 아니다)
_NAMED = re.compile(rf"(?P<w>{RANK}|{_WORDS})\s+(?:[가-힣]{{3}}|○{{2,4}})")   # 직위 + 이름 한 칸('담당 홍길동' — '담당 부서' 는 아니다)
ORG_MAX_CHARS = 20          # 공문 첫 줄 기관명(행정기관명) 글자 수(띄어쓰기 뺀) 상한
# 기관 이름 끝 — 부·처·청·단·회·팀·과·센터·공사…('결과·요청' 같은 명사 끝은 아니다). '…원' 은 네 글자 이상만 — '박수원'·'사원' 은 사람이다
ORG = re.compile(r"(?:[부처단회팀]|센터|(?<![결효성])과|(?<!요)청|(?<=[가-힣]{3})(?<![지인직])원|병원|법원|공사|회사|은행|학교|기관"
                 r"|연구소|사무소|특별시|광역시|특별자치[시도]|㈜|\(주\))$")
# 기관 이름 끝이어도 기관이 아닌 낱말 — 행사('착수 보고회'·'업무보고회')·일반 명사('실태 진단'·'내부'·'연락처')
_NOT_ORG = re.compile(r"(?:보고|설명|간담|토론|공청|박람|전시|발표|강연|연수|시연|평가|대|총)회$"
                      r"|(?<!\S)(?:진단|판단|수단|차단|중단|내부|외부|일부|전부|세부|여부|첨부|출처|연락처|문의처|거래처|사회|기회)$")

_NAMES = "|".join(r"\s*".join(w) for w in sorted(FIELDS, key=len, reverse=True))
_LABEL = re.compile(rf"(?<![^\s/|(])(?:(?P<name>{_NAMES})|(?P<post>우)(?=\s*\d{{3}}-?\d{{2,3}}))(?=\s*[:：]|[\s()]|$)")
END = re.compile(r"(?:^|\s)끝\s*\.\s*$", re.M)                  # 공문 본문 끝(규정 제4조⑤ — '… 1부.  끝.')
# 보도자료 머리의 보도·배포 일시 꼬리표 — 그룹 press 가 있으면 보도 날짜(배포보다 앞세운다)
_PRESS = re.compile(r"\s*(?:(?P<press>보\s*도(?:\s*(?:시\s*점|일\s*시|일\s*자|일))?|엠\s*바\s*고)"
                    r"|배\s*포(?:\s*(?:일\s*시|일\s*자|일))?)(?![가-힣])\s*[:：]?\s*")
_DATED = re.compile(r"\s*[(\[]?\s*(?:19|20)\d{2}\s*[.년/\-]")      # 날짜로 시작하는 글 — 꼬리표 바로 뒤·옆 칸
_PRESS_HEAD = re.compile(r"(?:보\s*도\s*(?:참\s*고\s*|설\s*명\s*|해\s*명\s*)?|설\s*명\s*|해\s*명\s*)자\s*료")   # 보도자료 머리 — '배포' 날짜를 받는 단서
# 문장 줄 — 항목 이름이 있어도 메타 줄이 아니다. '개요·필요' 로 끝나는 제목은 문장이 아니다
_SENT = re.compile(r"(?:다|[어아여해워와봐줘져세네게데래대나까에]요)\s*[.。!?]?\s*$")
_TERSE = re.compile(r"[함음임됨]\s*\.?\s*$")                    # 개조식 끝맺음('… 예정임.')
# 서명·날짜 칸(띄어쓰기 뺀 글자) — 낱말은 칸 전체일 때만('처리일자' 머리칸은 아니다)
_SIGN = re.compile(r"^(?:서명|\(서명\)|\(인\)|날짜|일자)$|^\d{1,4}[./-]\d{1,2}(?:[./-]\d{1,2})?\.?$|^/$")


def is_org(s: str) -> bool:
    """기관 이름으로 끝나는가 — 행사·일반 명사로 끝나면 아니다."""
    return bool(ORG.search(s)) and not _NOT_ORG.search(s)


def _body_line(ln: str) -> bool:
    """본문 줄 — 문장 끝맺음, 항목 기호·번호로 시작하는 줄('가. 접수: …'), 긴 개조식 줄('… 실시할 예정임.')."""
    s = ln.strip()
    return bool(_SENT.search(s) or ITEM.match(s) or (len(s) >= TERSE_MIN and _TERSE.search(s)))


def _labels(ln: str, loose: bool) -> list[tuple[re.Match, str]]:
    """줄 안의 항목 이름과 키 — 줄 머리·쌍점·괄호 이름만, loose 면 줄 가운데 이름도. 제목 뒤 낱말은 제목 글이다."""
    out = []
    for m in _LABEL.finditer(ln):
        key = "address" if m.group("post") else FIELDS[re.sub(r"\s+", "", m.group("name"))]
        head = not ln[:m.start()].strip(" \t/|(")
        colon = re.match(r"\s*[:：]", ln[m.end():]) is not None
        paren = ln[m.start() - 1:m.start()] == "(" and ln[m.end():m.end() + 1] == ")"
        if loose or head or colon or paren:
            out.append((m, key))
            if key == "title":
                break
    return out


def _clean(v: str) -> str:
    v = " ".join(v.split()).strip(" :：/|,")
    while v.count("(") < v.count(")") and v.endswith(")"):      # '(담당: 홍길동)' 의 닫는 괄호
        v = v[:-1].rstrip()
    while v.count("(") > v.count(")") and v.endswith("("):
        v = v[:-1].rstrip()
    return v.strip(" :：/|,")


def _values(ln: str) -> list[tuple[str, str]]:
    """(키, 값) — 값은 다음 항목 이름이나 ' / '(공문 끝 줄의 칸 나눔)까지, 제목은 줄 끝까지."""
    ms = _labels(ln, loose=True)
    out = []
    for i, (m, key) in enumerate(ms):
        v = ln[m.end():ms[i + 1][0].start() if i + 1 < len(ms) else len(ln)]
        out.append((key, _clean(v if key == "title" else v.split(" / ")[0])))
    return out


def _iso(d: tuple[int, int, int | None]) -> str:
    return f"{d[0]}-{d[1]:02d}" + (f"-{d[2]:02d}" if d[2] else "")


def _put(vals: dict[str, str], key: str, v: str) -> None:
    if not v:
        return
    if key in DATES or key == "enforce":
        if d := find_date(v):
            vals.setdefault("date" if key == "enforce" else key, _iso(d))
        if key != "enforce":
            return
        v = _clean(re.split(r"[\s(（]+(?=(?:19|20)\d\d\s*[.년/\-])", v)[0])   # 시행 '처리과-번호 (날짜)'·'처리과-번호 날짜' 의 번호
        if not re.search(r"\d", v) or find_date(v):
            return
        key = "doc_no"
    if key == "doc_no":
        vals.setdefault(key, v)
    else:
        vals[key] = f"{vals[key]}, {v}" if key in vals else v


def _chars(lines: list[str]) -> int:
    return sum(not c.isspace() for ln in lines for c in ln)


def _enough(keys: set[str], loose: bool) -> bool:
    """서로 다른 항목 2개 이상 — 첫 쪽 위(엄격)에서는 문서 항목(문서번호·수신·제목 등)도 하나 있어야 한다
    (안내문 위 문의처 줄, 사업 개요 표의 담당·전화는 본문이다)."""
    return len(keys) >= MIN_FIELDS and (loose or bool(keys - CONTACT_FIELDS))


Found = tuple[set[str], dict[str, str], bool]       # (항목 키, 값, 본문 섞임)


def _text_meta(text: str | None, loose: bool) -> Found | None:
    """메타 줄이 글의 절반 이상일 때만. 메타 줄 밖의 문장·항목·개조식 줄은 본문이다 — 그런 블록은 지우지 않는다.
    항목 기호·번호로 시작하는 줄('가. 접수: …'·'○ 시행: …')은 쌍점 이름이 있어도 본문 항목이다(공문 머리 메타 줄은 기호가 없다)."""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    meta = [ln for ln in lines if not _SENT.search(ln) and not ITEM.match(ln) and _labels(ln, loose)]
    if not meta or 2 * _chars(meta) < _chars(lines):
        return None
    vals: dict[str, str] = {}
    last = None
    for ln in lines:
        if ln in meta:
            pairs = _values(ln)
            for k, v in pairs:
                _put(vals, k, v)
            last = pairs[-1][0] if pairs else None
        elif last == "title" and "title" in vals and not _body_line(ln):
            vals["title"] += " " + _clean(ln)                   # 두 줄로 넘어간 제목 — 항목·문장·개조식 줄은 아니다
        else:
            last = None
    body = any(ln not in meta and _body_line(ln) for ln in lines)
    return {k for ln in meta for _, k in _labels(ln, loose)}, vals, body


def _next(rows: list[list[Cell]], i: int, j: int, key: str) -> list[tuple[int, int]]:
    """이름만 든 칸의 값 칸 자리 — 바로 오른쪽 칸, 없으면 아래 칸. '시행' 만 '처리과-번호|날짜' 로 갈리므로 바로 뒤 날짜 칸까지 —
    다른 이름 칸('출장지|…'·'장소|…')을 값으로 빨아들이지 않는다."""
    r = rows[i]
    if j + 1 < len(r) and r[j + 1].text and not _LABEL.match(r[j + 1].text):
        at = [(i, j + 1)]
        if key == "enforce" and j + 2 < len(r) and find_date(r[j + 2].text) and not find_date(r[j + 1].text):
            at.append((i, j + 2))
        return at
    if i + 1 < len(rows) and j < len(rows[i + 1]) and rows[i + 1][j].text and not _LABEL.match(rows[i + 1][j].text):
        return [(i + 1, j)]
    return []


def _table_meta(rows: list[list[Cell]], loose: bool) -> Found | None:
    """항목 이름 칸과 그 값 칸(바로 옆 한 칸)이 표 글자의 절반 이상일 때만 — 메타 줄 몇 개가 든 사업 개요 표·출장 보고 머리 표는
    본문이다. 이름 칸이 아닌 칸에 문장이 있거나, 첫 쪽 위에서 이름 칸도 값 칸도 아닌 칸에 긴 글이 있으면 본문이 든 표다
    (전자결재 출력의 '내 용' 칸 — 끝맺음이 없어도)."""
    cells = [c for r in rows for c in r]
    if not cells or any(len(c.text) > META_MAX_CELL for c in cells):
        return None
    vals: dict[str, str] = {}
    keys, rows_with, covered, body, used = set(), set(), 0, False, set()
    for i, r in enumerate(rows):
        for j, c in enumerate(r):
            found = _labels(c.text, loose)
            if not found:
                body |= bool(_SENT.search(c.text))
                continue
            keys |= {k for _, k in found}
            rows_with.add(i)
            used.add((i, j))
            pairs = _values(c.text)
            if len(pairs) == 1 and not pairs[0][1] and _LABEL.match(c.text):   # 이름만 든 칸 — 값은 옆(없으면 아래) 칸
                at = _next(rows, i, j, pairs[0][0])
                used |= set(at)
                pairs = [(pairs[0][0], _clean(" ".join(rows[a][b].text for a, b in at)))]
                covered += _chars([rows[a][b].text for a, b in at[:1]])
            covered += _chars([c.text])
            for k, v in pairs:
                _put(vals, k, v)
    if not keys or (len(rows) >= 3 and rows_with == {0}) or 2 * covered < _chars([c.text for c in cells]):
        return None                                             # 머리행에만 항목 이름 — 연락처·명단 표
    body |= not loose and any((i, j) not in used and _chars([c.text]) >= BODY_CELL_MIN
                              for i, r in enumerate(rows) for j, c in enumerate(r))
    return keys, vals, body


def _post(text: str) -> tuple[str, bool] | None:
    """직위·결재 칸이면 (그 낱말, 이름이 붙었나) — '팀장'·'결 재'·'대표이사'·'결재권자', 직위 + 이름 한 칸('담당 홍길동')."""
    if m := _POST.fullmatch(re.sub(r"\s+", "", text)):
        return m["w"], False
    return (m["w"], True) if (m := _NAMED.fullmatch(text.strip())) else None


def _signed(r: list[Cell]) -> bool:
    """서명·날짜 행 — 서명·날짜 칸이나 '전결·대결' 칸이 있거나 칸이 모두 빈 서명 줄."""
    ts = [re.sub(r"\s+", "", c.text) for c in r]
    return bool(r) and (not any(ts) or any(_SIGN.search(t) or t.startswith(("전결", "대결")) for t in ts))


def _post_row(rows: list[list[Cell]]) -> int | None:
    """결재란이면 직위 칸이 늘어선 행 — 작은 표(2~4행)의 칸 글자가 짧고, 한 행의 글 칸이 모두 직위·결재 칸이며(2종 이상, 직위 하나는
    꼭 — '내부결재' 같은 값 칸은 아니다) '결재·전결·대결' 이나 서명·날짜 칸(빈 서명 줄 포함)이 있을 때만. 직위 없이 '기안|검토|결재'
    가 늘어선 행은 다른 행이 서명·날짜 행일 때만. 머리칸에 다른 이름이 섞인 본문 표('과제|담당|검토|기한'·당직표 '일자|담당|팀장')와
    직위가 세로로 늘어선 표(업무분장표·일정표의 담당 열)는 표로 남긴다 — 세로 결재란도 표로 남지만 그쪽이 안전하다.
    직위 칸과 '직위 이름' 칸이 섞인 행('기안자|주무관 홍길동|검토자|…')은 이름표|값 짝인 결재선 줄이라 결재란 행이 아니다."""
    cells = [re.sub(r"\s+", "", c.text) for r in rows for c in r]
    if not cells or not 2 <= len(rows) <= APPROVAL_MAX_ROWS or any(len(c) > APPROVAL_MAX_CELL for c in cells):
        return None
    marked = (any(_SIGN.search(c) or c.startswith(("결재", "전결", "대결")) for c in cells)
              or any(r and not any(c.text for c in r) for r in rows[1:]))
    for i, r in enumerate(rows):
        found = [_post(c.text) for c in r if c.text.strip()]
        if None in found or len({n for _, n in found}) != 1 or len(posts := {w for w, _ in found}) < 2:
            continue
        if marked and any(re.fullmatch(RANK, w) for w in posts):
            return i
        if TRIO[0] <= posts and TRIO[1] & posts and any(_signed(x) for k, x in enumerate(rows) if k != i):
            return i
    return None


def _approval_meta(rows: list[list[Cell]], post: int) -> dict[str, str] | None:
    """결재란 표에 함께 든 메타 행(문서번호·시행·기안일자 …)의 값 — 직위 행과 서명 행(사람 항목뿐인 행)은 뺀다."""
    meta = [r for i, r in enumerate(rows)
            if i != post and any(k not in PERSON_FIELDS for c in r for _, k in _labels(c.text, True))]
    found = _table_meta(meta, True) if meta else None
    return (found[1] or None) if found and _enough(found[0], True) else None


def _candidates(doc: Document):
    """doc_meta 후보 묶음과 줄 가운데 이름도 셀지 — 공문 끝('끝.'·아래쪽 결재란 아래, 느슨: 버린 머리말·꼬리말도 — 검출기가
    시행·접수 줄을 꼬리말로 잡는다), 첫 논리 페이지 위쪽(엄격: 공문 머리 항목(수신·참조·경유·제목·발신·문서번호) 줄로만 된 글
    블록이 이어지면 한 묶음 — 검출기가 '수신'·'(경유)'·'제목' 을 줄마다 다른 블록으로 낸다. 접수·시행 같은 줄은 잇지 않는다)."""
    for n, page in enumerate(doc.pages):
        h = page_size(page)[1]
        cuts = [local(page, b.bbox)[3] for b in page.blocks if b.merged_into is None
                and ((b.type == "text" and END.search(b.text or ""))
                     or (b.type == "approval_box" and local(page, b.bbox)[1] >= LOW_RATIO * h))]
        run: list[Block] = []
        for b in sorted((b for b in page.blocks if b.merged_into is None), key=lambda b: local(page, b.bbox)[1]):
            top = local(page, b.bbox)[1]
            if cuts and top >= min(cuts) - GAP:
                if (b.keep and b.type in ("text", "table") or b.type in ("header", "footer")) and (b.text or b.html):
                    yield [b], True
                continue
            if n or not b.keep or top >= TOP_RATIO * h:
                continue
            found = _text_meta(b.text, False) if b.type == "text" else None
            if found and not found[2] and found[0] <= HEAD_FIELDS:
                run.append(b)                                   # 머리 항목 줄로만 된 글 블록 — 이어지면 함께 판정
                continue
            if run:
                yield run, False
                run = []
            if b.type in ("text", "table") and (b.text or b.html):
                yield [b], False
        if run:
            yield run, False


def _judge(group: list[Block], loose: bool) -> list[Found] | None:
    """블록 하나는 _enough, 이어진 여러 블록은 연락처 말고 문서 항목이 2개 이상일 때만 — 블록이 갈린 만큼 근거를 더 본다."""
    found = [_table_meta(parse_rows(b.html), loose) if b.type == "table" else _text_meta(b.text, loose) for b in group]
    if None in found:
        return None
    keys = set().union(*(f[0] for f in found))
    ok = _enough(keys, loose) if len(group) == 1 else len(keys - CONTACT_FIELDS) >= MIN_FIELDS
    return found if ok else None


def _retype(b: Block, typ: str, reason: str) -> None:
    b.provenance = (b.provenance or []) + [{"rule": typ, "type": b.type}]
    b.type, b.keep, b.drop_reason = typ, False, reason


def _issuer_line(page: Page, heads: list[Block]) -> None:
    """공문 첫 쪽 위 메타 묶음 바로 위의 짧은 기관명 줄('한국정보공사' — 행정기관명) → doc_meta(issuer). 숫자·항목 이름이 없는
    한 줄이고 기관 이름으로 끝날 때만 — 서식 이름('협조전')·행사 이름('주간 업무보고회')·문서 제목은 그대로 둔다."""
    top = min(local(page, b.bbox)[1] for b in heads)
    above = [b for b in page.blocks if b.keep and b.merged_into is None and local(page, b.bbox)[3] <= top + GAP]
    if not above:
        return
    b = max(above, key=lambda b: local(page, b.bbox)[3])
    s = " ".join((b.text or "").split())
    if (b.type in ("text", "paragraph_title", "doc_title") and len((b.text or "").strip().splitlines()) == 1
            and 0 < len(s.replace(" ", "")) <= ORG_MAX_CHARS and len(s.split()) <= 2 and not re.search(r"\d", s)
            and not _labels(s, True) and is_org(s)):
        _retype(b, "doc_meta", "meta")
        b.meta = {"issuer": s}


def _press_dates(page: Page) -> None:
    """보도자료 머리의 보도·배포 일시 → doc_meta(date)(실측 test2.pdf — '보도시점' 꼬리표는 머리말로 버려지고 옆 날짜 칸만
    글로 남았다). 첫 논리 페이지의 보도자료 머리 띠(문서 제목 위 — 제목 아래 '배포 일시: …' 안내 줄·기사 날짜는 본문)에서
    꼬리표 블록(버린 것도) 같은 줄 바로 오른쪽 칸이 날짜로 시작하는 짧은 글 블록이거나,
    꼬리표 뒤에 날짜가 바로 오는 글 블록만. '배포' 날짜는 보도자료 단서(보도 꼬리표·'보도자료' 머리)가 있을 때만 — 안내문의
    '배포 일시' 는 본문이다. 날짜 값은 보도 날짜를 앞세워 한 블록에만 단다."""
    h = page_size(page)[1]
    titles = [local(page, b.bbox)[1] for b in page.blocks if b.merged_into is None and b.type == "doc_title"]
    top = [b for b in page.blocks if b.merged_into is None and (b.text or "").strip()   # 보도 머리 띠 — 문서 제목 위(없으면 위쪽)
           and (local(page, b.bbox)[3] <= min(titles) + GAP if titles else local(page, b.bbox)[1] < TOP_RATIO * h)]
    labels = [(b, m) for b in top if (m := _PRESS.match(" ".join(b.text.split())))]
    cue = any(m.group("press") for _, m in labels) or any(_PRESS_HEAD.search(b.text) for b in top)
    found: dict[str, tuple[bool, Block, tuple]] = {}    # 블록 id → (보도 날짜가 아님, 블록, 날짜)
    for lab, m in labels:
        rank = m.group("press") is None
        if rank and not cue:
            continue
        b, text = lab, " ".join(lab.text.split())[m.end():]   # 꼬리표와 날짜가 한 블록
        if not text:
            x0, y0, x1, y1 = local(page, lab.bbox)
            right = []
            for o in top:
                ox0, oy0, _, oy1 = local(page, o.bbox)
                if o is not lab and ox0 >= x1 - GAP and min(y1, oy1) - max(y0, oy0) >= 0.5 * min(y1 - y0, oy1 - oy0):
                    right.append((ox0, o))
            if not right:
                continue
            b = min(right, key=lambda r: r[0])[1]           # 같은 줄 바로 오른쪽 칸만 — 다른 꼬리표의 날짜를 건너 잡지 않는다
            text = " ".join(b.text.split())
        if b.keep and b.type == "text" and len(text) <= 30 and _DATED.match(text) and (d := find_date(text)) \
                and (b.id not in found or rank < found[b.id][0]):
            found[b.id] = (rank, b, d)
    for i, (_, b, d) in enumerate(sorted(found.values(), key=lambda f: f[0])):
        _retype(b, "doc_meta", "meta")
        b.meta = {"date": _iso(d)} if i == 0 else None


def mark_meta_blocks(doc: Document) -> Document:
    """결재란·공문 메타 블록의 type 을 바꾸고 keep=false 로 — 원래 type 은 provenance, 메타 값은 block.meta 에.
    본문이 섞인 블록은 그대로 두고 값만 남긴다. 결재란 표에 함께 든 메타 행의 값도 block.meta 에."""
    doc = doc.model_copy(deep=True)
    for page in doc.pages:
        for b in page.blocks:
            if b.type == "table" and b.html and b.merged_into is None and b.keep:
                rows = parse_rows(b.html)
                if (post := _post_row(rows)) is not None:
                    _retype(b, "approval_box", "approval")
                    b.meta = _approval_meta(rows, post)
    heads: list[Block] = []
    for group, loose in list(_candidates(doc)):
        found = _judge(group, loose)
        parts = [(group, found)] if found or len(group) == 1 else [([b], _judge([b], loose)) for b in group]   # 모자라면 블록마다
        for part, found in parts:
            if found is None:
                continue
            for b, (_, vals, body) in zip(part, found):
                if body and b.keep:                             # 본문이 섞였다 — 지우지 않고 값만
                    if vals:
                        b.provenance = (b.provenance or []) + [{"rule": "meta_values"}]
                        b.meta = vals
                else:
                    _retype(b, "doc_meta", "meta")
                    b.meta = vals or None
            if not loose:
                heads += part
    if heads:
        _issuer_line(doc.pages[0], heads)
    if doc.pages:
        _press_dates(doc.pages[0])
    return doc
