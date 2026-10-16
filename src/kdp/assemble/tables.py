"""표(기획서 4.3 규칙 1·6·13, D18) — 머리행, 표·그림 제목·단위·출처 연결, 표 갈래, '이하 빈칸' 행, 쪽 넘김 표.

표 제목은 라벨(figure_title)만 믿지 않는다 — 실측에서 일반 구역 제목에도 붙었다. '<표 n>'·'[도표 n]' 같은 패턴이 있고
표 바로 위나 아래에 붙은 블록, 또는 표 바로 위의 괄호로 감싼 한 줄 제목('[ 발령 기준 ]')만 잇는다. 그림은 바로 위·아래의
'<그림 n>'·괄호 제목, 그다음 검출기가 그림 제목(figure_title)으로 본 줄을 잇는다.
"""
from __future__ import annotations

import html as htmllib
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from ..geometry import local, page_size
from ..schema import Block, Document, Page
from .order import has_output

CAPTION = re.compile(r"^\s*[\[<〈【(]?\s*(?:표|도표|table)\s*[\dⅠ-Ⅹ]+(?:[-.]\d+)*\s*[\]>〉】)]?", re.I)   # 그림 제목은 받지 않는다
UNIT = re.compile(r"[(（]\s*단위\s*[:：]\s*([^)）]+?)\s*[)）]")
SOURCE = re.compile(r"^\s*(?:※\s*)?(?:자료|출처)\s*[:：]")
FIG_CAPTION = re.compile(r"^\s*[\[<〈【(]?\s*(?:그림|사진|figure|fig\.?)\s*\d", re.I)   # 그림 제목의 단위는 표에 붙이지 않는다
_BRACKET = re.compile(r"\s*(?:\[[^\[\]\n]+\]|<[^<>\n]+>|〈[^〈〉\n]+〉|【[^【】\n]+】|「[^「」\n]+」)\s*")   # 괄호로 감싼 한 줄
BRACKET_MIN, BRACKET_MAX = 4, 60   # 괄호 안 글자 수(띄어쓰기 뺀) — '[참고]'·'<붙임>' 꼬리표는 제목이 아니다
_ROW = re.compile(r"<tr[^>]*>.*?(?=<tr[\s>]|</table>|$)", re.S | re.I)
_CELL = re.compile(r"<t([dh])([^>]*)>(.*?)</t[dh]>", re.S | re.I)
MAX_GAP = 15.0                # 표와 제목·단위·출처 블록 사이 세로 틈(pt)
HEADER_MAX_CHARS = 20
EMPTY_FAILED = 0.9            # 거의 모든 칸이 비면 인식 실패로 본다(빈 서식 표는 절반쯤 비어도 정상)
LAYER_FAILED = 0.8            # born-digital 표에서 레이어 글자 중 모델 표에 있는 비율이 이보다 낮으면 인식 실패(행을 빠뜨림 등)
LAYOUT_MIN_CHARS = 30         # 1~2칸 표의 글자가 이보다 많으면 배치용 표(글상자)로 본다
WIDTH_TOL = 0.1               # 쪽 넘김 표 두 조각의 폭 차이(논리 페이지 폭 비율 기준) — PaddleX 병합과 같은 10%
MAX_REPEAT_ROWS = 5           # 뒤 조각 첫머리에서 반복 머리행으로 세는 최대 행 수(PaddleX 와 같다)
HEADER_SIMILAR = 0.8          # 반복 머리행으로 볼 글자 일치 비율 — 스캔 OCR 은 되풀이한 머리행의 한두 글자를 달리 읽는다
ROOM_MAX = 0.3                # 앞 표 아래 빈자리가 논리 페이지 높이의 이 비율을 넘으면 표가 거기서 끝났다(밀린 큰 행 하나보다 크다)

_BLANK_BELOW = re.compile(r"이하(?:빈칸|여백|공란)")                       # '이하 빈칸'(시행규칙 제4조⑤)과 같은 뜻의 관용 표기
_DECOR = re.compile(r"[\s\-–—=~.·*※()\[\]<>〈〉【】「」『』（）]")
CONT = re.compile(r"[-–(\[<〈【（]*(?:(?:다음|뒷|앞|전|이전)(?:쪽|면|장|페이지)?(?:에서|에|으로|로)?)?표?"
                  r"(?:계속|이어짐|이어서|이어집니다)(?:됨|됩니다|합니다)?[-–)\]>〉】）.]*")   # '(계속)' 표시 — 공백을 뺀 블록 글 전체
_CONT_TAIL = re.compile(r"[(\[<〈【（]?\s*계\s*속\s*[)\]>〉】）]?\s*$")                  # 제목 끝 '(계속)' — '계속사업' 은 아니다
_UNIT_LINE = re.compile(r"(?:※\s*)?[\[(（<〈]?\s*단위\s*[:：]\s*([^)\]）>〉]+?)\s*[)\]）>〉]?")   # '(단위: 천원)' 만 적힌 줄
_HALF = {c: c - 0xFEE0 for c in range(0xFF01, 0xFF5F)}                           # 전각 → 반각(PaddleX full_to_half)
FOOTNOTES = ("footnote", "vision_footnote")
# 표·그림 부속 줄 — '주: …'·'주1) …', '이하 빈칸', 번호 붙은 표·그림 제목. 번호 바로 뒤(괄호 뒤)에 한글이 붙거나,
# 다른 번호·접속('표 1, 2와'·'표 1~3은'·'그림 2·3 처럼'·'표 3 및 그림 2')이 오면 표를 가리키는 본문이다
_REF_TAIL = r"\s*(?:[,，、~∼〜～·ㆍ‧・∙]|(?:및|와|과|또는|내지)(?![가-힣]))"
_APPENDAGE = re.compile(r"\s*(?:주\s*\d*\s*[):：]|이하\s*(?:빈칸|여백|공란)"
                        r"|[\[<〈【(]?\s*(?:표|도표|그림|사진)\s*[\dⅠ-Ⅹ]+(?:[-.]\d+)*"
                        r"(?![\d가-힣]|[-.]\d|[\]>〉】)][가-힣]|[\]>〉】)]?" + _REF_TAIL + "))")


def appendage(text: str) -> bool:
    """표·그림 부속 줄(제목·출처·주·'이하 빈칸') — 본문 문단과 잇지 않는다."""
    return bool(SOURCE.match(text) or _APPENDAGE.match(text))


def attached(blocks) -> set[str]:
    """표에 이은 제목·출처 블록 id(caption_id·note_ids)."""
    return {i for b in blocks for i in (b.caption_id, *(b.note_ids or ())) if i}


@dataclass
class Cell:
    text: str
    rowspan: int
    colspan: int


def _span(attrs: str, name: str) -> int:
    m = re.search(name + r'\s*=\s*"?(\d+)', attrs, re.I)
    return int(m.group(1)) if m else 1


def parse_rows(html: str) -> list[list[Cell]]:
    return [[Cell(" ".join(htmllib.unescape(re.sub(r"<[^>]+>", " ", c.group(3))).split()),
                  _span(c.group(2), "rowspan"), _span(c.group(2), "colspan"))
             for c in _CELL.finditer(row)] for row in _ROW.findall(html)]


def header_rows(rows: list[list[Cell]]) -> int:
    """첫 행이 짧은 글자들이고 본문 칸보다 길지 않으면 머리행. 첫 행에 가로 병합이 있으면 세로 병합 깊이까지."""
    if len(rows) < 2 or not rows[0]:
        return 0
    first = rows[0]
    if any(not c.text or len(c.text) > HEADER_MAX_CHARS or re.fullmatch(r"[\d,.%\s\-~]+", c.text) for c in first):
        return 0
    body = [len(c.text) for r in rows[1:] for c in r if c.text]
    if body and max(len(c.text) for c in first) > max(body):
        return 0
    n = max(c.rowspan for c in first) if any(c.colspan > 1 for c in first) else 1
    return min(n, len(rows) - 1)


def _mark_th(html: str, n: int) -> str:
    rows = list(_ROW.finditer(html))
    for m in reversed(rows[:n]):
        row = re.sub(r"<td([\s>])", r"<th\1", m.group(0), flags=re.I).replace("</td>", "</th>")
        html = html[:m.start()] + row + html[m.end():]
    return html


def drop_blank_rows(html: str) -> tuple[str, list[str]]:
    """표 안 '이하 빈칸' 행은 표 데이터가 아니므로 지운다(기획서 5.3). 그 행에 다른 값이 있거나, 세로 병합이 걸쳐
    지우면 칸이 어긋날 수 있거나, 지우고 나면 글자 있는 칸이 남지 않으면 둔다. 지운 행의 글을 함께 돌려준다."""
    spans = [m.span() for m in _ROW.finditer(html)]
    rows = parse_rows(html)                                        # 같은 정규식이라 spans 와 행이 1:1
    drop, reach = [], -1
    for r, row in enumerate(rows):
        texts = [c.text for c in row if c.text]
        if texts and reach < r and all(c.rowspan == 1 for c in row) \
                and _BLANK_BELOW.fullmatch(_DECOR.sub("", "".join(texts))):
            drop.append(r)
        reach = max([reach] + [r + c.rowspan - 1 for c in row])
    if not drop or not any(c.text for r, row in enumerate(rows) if r not in drop for c in row):
        return html, []
    for r in reversed(drop):
        html = html[:spans[r][0]] + html[spans[r][1]:]
    return html, [" ".join(c.text for c in rows[r] if c.text) for r in drop]


def _gap(page: Page, b: Block, t: Block) -> tuple[float, float]:
    """(위로 떨어진 거리, 아래로 떨어진 거리) — 가로로 겹치지 않으면 무한대. 세운 쪽 기준."""
    bb, tb = local(page, b.bbox), local(page, t.bbox)
    if min(bb[2], tb[2]) - max(bb[0], tb[0]) <= 0:
        return float("inf"), float("inf")
    return tb[1] - bb[3], bb[1] - tb[3]


def _dist(page: Page, b: Block, t: Block) -> float:
    """블록과 표·그림 사이 세로 거리(위든 아래든). 가로로 안 겹치거나 멀면 무한대."""
    up, down = _gap(page, b, t)
    return min(d for d in (up, down) if d >= -3) if (-3 <= up <= MAX_GAP or -3 <= down <= MAX_GAP) else float("inf")


def _link(page: Page, t: Block) -> None:
    """표 바로 위·아래 블록에서 제목·단위·출처를 찾는다. 두 표(그림) 사이 블록은 더 가까운 쪽 것이다 — 제목이 표 아래에 오는
    문서와 '다음 표'의 제목을 함께 가른다. 나란한 표 위의 한 제목은 두 표가 함께 받는다. 출처 주는 늘 자기 표 아래다."""
    others = [o for o in page.blocks if o is not t and o.merged_into is None and o.type in ("table", "image", "chart")]
    near = []
    for b in page.blocks:
        if b is t or not b.keep or b.merged_into or not b.text or b.type in ("table", "image", "chart"):
            continue
        up, down = _gap(page, b, t)
        side = "up" if -3 <= up <= MAX_GAP else "down" if -3 <= down <= MAX_GAP else None
        if side is None:
            continue
        mine = up if side == "up" else down
        if side == "down" and SOURCE.match(b.text):
            t.note_ids = (t.note_ids or []) + [b.id]
        elif not any(_dist(page, b, o) < mine for o in others):
            near.append((mine, side, b))
    near.sort(key=lambda x: (x[1] != "up", x[0]))                  # 위쪽 먼저, 가까운 것 먼저
    cap = next((b for _, _, b in near if CAPTION.match(b.text)), None) \
        or next((b for _, side, b in near if side == "up" and bracket_title(b.text)), None)
    if cap:
        t.caption_id = cap.id
    for _, side, b in near:
        if not t.unit and (b is cap or (side == "up" and not FIG_CAPTION.match(b.text))) and (m := UNIT.search(b.text)):
            t.unit = m.group(1)


def bracket_title(text: str | None) -> bool:
    """괄호로 감싼 한 줄 제목('[ 국내 환자 발생 추이 ]'·'<발령 기준>') — 번호 없는 표·그림 제목(실측 test2.pdf). '(단위: …)'·
    '(계속)' 줄과 짧은 꼬리표는 아니다."""
    s = (text or "").strip()
    if not _BRACKET.fullmatch(s) or _UNIT_LINE.fullmatch(s) or CONT.fullmatch("".join(s.split())):
        return False
    return BRACKET_MIN <= len("".join(s.split())) - 2 <= BRACKET_MAX


def _link_figure(page: Page, f: Block) -> None:
    """그림·차트 바로 위·아래(MAX_GAP 안)의 제목 — '<그림 n>'·괄호 제목을 검출기 그림 제목(figure_title) 줄보다, 위쪽을
    아래쪽보다, 가까운 것을 먼저(그래프 바로 아래 축 이름도 figure_title 로 잡힌다 — 실측). 다른 표·그림에 더 가까운 블록은
    그쪽 것이고, 나란한 그림 위의 한 제목은 함께 받는다."""
    others = [o for o in page.blocks if o is not f and o.merged_into is None and o.type in ("table", "image", "chart")]
    near = []
    for b in page.blocks:
        if b is f or not b.keep or b.merged_into or not b.text or b.type in ("table", "image", "chart"):
            continue
        up, down = _gap(page, b, f)
        side = "up" if -3 <= up <= MAX_GAP else "down" if -3 <= down <= MAX_GAP else None
        if side is None:
            continue
        mine = up if side == "up" else down
        strong = bool(FIG_CAPTION.match(b.text) or bracket_title(b.text))
        if (strong or b.type == "figure_title") and not any(_dist(page, b, o) < mine for o in others):
            near.append((not strong, side != "up", mine, b))
    if near:
        f.caption_id = min(near, key=lambda x: x[:3])[3].id


def _route(t: Block, rows: list[list[Cell]]) -> tuple[str, str | None]:
    cells = [c for r in rows for c in r]
    if not cells:
        return "image", "failed"                                   # 칸을 하나도 못 읽었다(OTSL 없이 평문이 온 경우 등)
    if "<img" in (t.html or "") or any(p.get("rule") == "table_img_missing" for p in t.provenance or []):
        return "image", "photo_in_cell"
    if t.truncated or t.error:
        return "image", "failed"
    head = rows[0] if rows else []
    if len(cells) >= 4 and sum(not c.text for c in cells) >= EMPTY_FAILED * len(cells) \
            and sum(bool(c.text) for c in head) < max(1, len(head) // 2):
        return "image", "failed"                                   # 머리행까지 비면 인식 실패 — 머리행만 있는 빈 서식 표는 정상
    agree = next((p.get("agreement") for p in t.provenance or [] if p.get("rule") == "text_layer"), None)
    if agree is not None and agree < LAYER_FAILED:
        return "image", "failed"
    return "html", None


def _plain(t: Block) -> bool:
    """사진·잘림·오류가 없는 표 — 배치용 표로 풀어도 잃는 것이 없다."""
    lost = any(p.get("rule") == "table_img_missing" for p in t.provenance or [])
    return "<img" not in (t.html or "") and not lost and not t.truncated and not t.error


def _boxy(rows: list[list[Cell]]) -> bool:
    """배치용 표(글상자) 꼴 — 칸 1~2개·행 2개 이하에 글자가 LAYOUT_MIN_CHARS 이상."""
    cells = [c for r in rows for c in r]
    return bool(cells) and len(cells) <= 2 and len(rows) <= 2 and sum(len(c.text) for c in cells) >= LAYOUT_MIN_CHARS


def tables(doc: Document) -> Document:
    doc = doc.model_copy(deep=True)
    for t in (t for p in doc.pages for t in p.blocks if t.type == "table" and not t.merged_into and t.html):
        t.html, removed = drop_blank_rows(t.html)                     # 머리행·갈래·쪽 넘김 조각 판정(blank_below)보다 먼저
        if removed:
            t.provenance = (t.provenance or []) + [{"rule": "blank_below", "removed": removed}]
    spill = _spill_ids(doc)
    for page in doc.pages:
        for t in page.blocks:
            if t.type != "table" or t.merged_into:
                continue
            if not t.html:                                             # VLM 오류·빈 출력 — 칸을 하나도 못 읽었다
                t.route, t.route_reason = "image", "failed"
                continue
            rows = parse_rows(t.html)
            cells = [c for r in rows for c in r]
            if _boxy(rows) and _plain(t) and t.id not in spill:        # 다음 쪽으로 넘긴 한 행 조각은 표로 둔다
                t.type, t.text, t.html = "text", "\n".join(c.text for c in cells if c.text), None   # 크롭(img_path)은 남긴다
                t.provenance = (t.provenance or []) + [{"rule": "layout_table"}]   # 글상자처럼 쓴 표 → 본문
                continue
            n = header_rows(rows)
            if n:
                t.header_rows, t.html = n, _mark_th(t.html, n)
            t.route, t.route_reason = _route(t, rows)
            _link(page, t)
        for f in page.blocks:
            if f.type in ("image", "chart") and f.keep and f.merged_into is None:
                _link_figure(page, f)
    return doc


# 쪽 넘김 표의 열 수·머리행 비교는 설치본 PaddleX(merge_table.py)와 같은 기준을 여기서 다시 쓴다 — 그 함수들은 입력을
# 바꾸지 않지만, import 하면 paddlex.inference 전체(약 1.7초·모듈 4천 개)를 불러 모델 없는 재조립이 무거워지고, 입력이
# BeautifulSoup 이라 이 모듈의 칸 해석(parse_rows)과 갈린다.
def n_cols(rows: list[list[Cell]]) -> int:
    """열 수 — 세로·가로 병합 칸이 차지한 자리까지 센다(PaddleX calculate_table_total_columns 와 같은 기준)."""
    taken: dict[int, set[int]] = {}
    width = 0
    for r, row in enumerate(rows):
        c = 0
        for cell in row:
            while c in taken.get(r, ()):
                c += 1
            for rr in range(r, r + cell.rowspan):
                taken.setdefault(rr, set()).update(range(c, c + cell.colspan))
            c += cell.colspan
            width = max(width, c)
    return width


def repeated_rows(a: list[list[Cell]], b: list[list[Cell]], head: int = 0) -> int:
    """두 표 첫머리에서 같은 행 수 — 칸 수·colspan 이 같고 글자(공백·전각 무시)가 같아야 한다(PaddleX detect_table_headers
    와 같은 기준). 다만 a 의 머리행(head 행)은 스캔 OCR 이 한두 글자를 달리 읽을 수 있어 글자 일치 비율이 HEADER_SIMILAR
    이상이면 센다 — 데이터 행은 긴 칸이 같으면 비율이 높아 완전 일치만. 빈 행은 세지 않는다."""
    def key(row):
        return ["".join(c.text.translate(_HALF).split()) for c in row], [c.colspan for c in row]

    n = 0
    for x, y in zip(a[:MAX_REPEAT_ROWS], b[:MAX_REPEAT_ROWS]):
        (tx, sx), (ty, sy) = key(x), key(y)
        if not any(tx) or sx != sy or (tx != ty and (n >= head or SequenceMatcher(
                None, "|".join(tx), "|".join(ty)).ratio() < HEADER_SIMILAR)):
            break
        n += 1
    return n


def _is_mark(b: Block) -> bool:
    return bool(b.text) and bool(CONT.fullmatch("".join(b.text.split())))


def _unit_of(b: Block) -> str | None:
    """'(단위: 천원)' 만 적힌 줄의 단위."""
    m = _UNIT_LINE.fullmatch((b.text or "").strip())
    return m.group(1) if m else None


def _caption_no(text: str) -> str | None:
    m = CAPTION.match(text)
    return re.sub(r"[^\w\-.]", "", m.group(0)).lower() if m else None


def _title_key(text: str) -> str:
    """표 제목 글 비교용 — 번호·'(단위: …)'·'(계속)'·공백을 뺀다."""
    return "".join(_CONT_TAIL.sub("", CAPTION.sub("", UNIT.sub("", text), count=1)).split())


def _same_title(b: Block, cap: Block | None, above: Block | None = None) -> bool:
    """다음 쪽 첫머리의 표 제목 반복 — 번호를 뺀 제목 글이 같거나 번호만 남았다('<표 3> 계속'). 번호가 다르면 다른 표다.
    '<표 3>의 세부 내역은…' 처럼 표를 가리키는 본문은 아니다. 앞 표에 이은 제목이 없으면 끝에 '(계속)' 이 붙은 줄만 —
    표 제목 꼴이거나, 앞 표 바로 위 블록(above — 번호 없는 제목 '□ 사업비 집행 현황')과 글이 같아야 한다."""
    if not b.text:
        return False
    if cap is None or not cap.text:
        if not _CONT_TAIL.search(b.text):
            return False
        key = _title_key(b.text)
        return bool(CAPTION.match(b.text) or (key and above is not None and key == _title_key(above.text or "")))
    n1, n2 = _caption_no(b.text), _caption_no(cap.text)
    if n1 and n2 and n1 != n2:
        return False
    key = _title_key(b.text)
    return key == _title_key(cap.text) if key else bool(n1)


def _edge(blocks, skip) -> tuple[Block | None, list[Block]]:
    """쪽 끝(또는 첫머리)에서 출력에 나가는 첫 블록과 그 앞에서 건너뛴 블록들 — keep=false·빈 블록은 보지 않는다."""
    skipped = []
    for b in blocks:
        if not (b.keep and has_output(b)):
            continue
        if not (_is_mark(b) or skip(b)):
            return b, skipped
        skipped.append(b)
    return None, skipped


def _rel_width(page: Page, b: Block) -> float:
    """논리 페이지 폭에 대한 표 폭 — 모아찍기 쪽과 보통 쪽을 같은 잣대로 견준다(설계 4장)."""
    x0, _, x1, _ = local(page, b.bbox)
    return (x1 - x0) / page_size(page)[0]


def _set_header(t: Block, n: int) -> None:
    html = re.sub(r"<th([\s>])", r"<td\1", t.html, flags=re.I).replace("</th>", "</td>")   # th 는 모두 _mark_th 가 단 것
    t.html, t.header_rows = (_mark_th(html, n) if n else html), (n or None)


def _above(page: Page, t: Block) -> Block | None:
    """표 바로 위(MAX_GAP 안)의 글 블록 — 번호 없는 표 제목 자리. 사이의 '(단위: …)' 줄은 건너 그 위를 잰다
    ([제목 / '(단위: 천원)' / 표] 꼴)."""
    ups = sorted(((d, b) for b in page.blocks if b is not t and b.keep and b.text and (d := _gap(page, b, t)[0]) >= -3),
                 key=lambda x: x[0])
    edge = 0.0                                                         # 표 윗변에서 건너뛴 단위 줄 윗변까지
    for d, b in ups:
        if d - edge > MAX_GAP:
            return None
        if not _unit_of(b):
            return b
        y0, y1 = local(page, b.bbox)[1::2]
        edge = max(edge, d + y1 - y0)
    return None


def _edges(pa: Page, pb: Page) -> tuple[Block, list[Block], Block, list[Block]] | None:
    """앞 논리 페이지 끝 표와 다음 논리 페이지 첫 표, 그 사이에서 건너뛴 블록들(merge_tables 참고) — 둘 중 하나라도
    표가 아니면 None."""
    if not (pa.keep and pb.keep):
        return None
    a, skip_a = _edge(reversed(pa.blocks), lambda x: x.type in FOOTNOTES or _unit_of(x))
    if a is None or a.type != "table" or not a.html:
        return None
    cap, above = next((x for x in pa.blocks if a.caption_id and x.id == a.caption_id), None), _above(pa, a)
    b, skip_b = _edge(pb.blocks, lambda x: _unit_of(x) or _same_title(x, cap, above))
    if b is None or b.type != "table" or not b.html:
        return None
    return a, skip_a, b, skip_b


def _fits(pa: Page, a: Block, pb: Page, b: Block) -> bool:
    """열 수가 같고 폭(논리 페이지 폭 비율) 차이가 WIDTH_TOL 안인 두 표 조각."""
    na, wa, wb = n_cols(parse_rows(a.html)), _rel_width(pa, a), _rel_width(pb, b)
    return bool(na) and na == n_cols(parse_rows(b.html)) and min(wa, wb) > 0 and abs(wa - wb) / min(wa, wb) < WIDTH_TOL


def _told(skip_a: list[Block], skip_b: list[Block]) -> bool:
    """이어진다고 적었다 — 건너뛴 블록에 '(계속)' 표시나 번호 있는 표 제목 반복이 있다. 번호 없는 제목 반복('□ …(계속)')은
    구역이 이어진다는 뜻일 수 있어 같은 표라는 증거로는 약하다."""
    return any(map(_is_mark, skip_a + skip_b)) or any(CAPTION.match(x.text or "") for x in skip_b)


def _ended(pa: Page, a: Block, skip_a: list[Block], pb: Page, b: Block, skip_b: list[Block]) -> bool:
    """앞 표가 그 쪽에서 끝났다 — 표 아래 출처 주(라벨과 상관없이 — 출처는 늘 표 끝에 온다), '이하 빈칸' 행, 또는 표 아래
    빈자리. 표는 자리가 모자랄 때만 다음 쪽으로 넘어간다. 다만 HWP 는 들어가지 않는 큰 행을 통째로 다음 쪽으로 밀어 앞
    조각 아래가 비기도 하므로 느슨하게 — 쪽 높이의 ROOM_MAX 넘게 비었을 때만, '(계속)' 표시나 번호 있는 표 제목 반복이
    없을 때만. 쪽 아래 끝은 표 뒤 각주 등의 윗변, 없으면 다음 쪽 위 여백(첫 블록 윗변)만큼 뺀 자리로 본다."""
    if a.note_ids or any(SOURCE.match(x.text or "") for x in skip_a) \
            or any(p.get("rule") == "blank_below" for p in a.provenance or []):
        return True
    if _told(skip_a, skip_b):
        return False
    h = page_size(pa)[1]
    bottom = min([h - min(local(pb, x.bbox)[1] for x in [*skip_b, b])] + [local(pa, x.bbox)[1] for x in skip_a])
    return bottom - local(pa, a.bbox)[3] > ROOM_MAX * h


def _joins(pa: Page, a: Block, skip_a: list[Block], pb: Page, b: Block, skip_b: list[Block]) -> bool:
    """두 조각이 한 표다(merge_tables·_spill_ids 가 함께 쓰는 근거) — 열 수·폭·단위가 같고 앞 표가 그 쪽에서 끝나지
    않았다. 앞 조각이 글상자 꼴인데 뒤 표가 제 머리행을 가지면 쪽 끝 글상자와 다음 쪽의 다른 표다 — 그 행을 되풀이했거나
    이어진다고 적었으면(_told) 아니다."""
    units = {"".join(u.split()) for u in [a.unit, b.unit, *map(_unit_of, skip_a + skip_b)] if u}   # 단위가 다르면 다른 표다
    ra, rb = parse_rows(a.html), parse_rows(b.html)
    return _fits(pa, a, pb, b) and len(units) <= 1 and not _ended(pa, a, skip_a, pb, b, skip_b) \
        and not (_boxy(ra) and header_rows(rb) and not repeated_rows(ra, rb) and not _told(skip_a, skip_b))


def _spill_ids(doc: Document) -> set[str]:
    """쪽 넘김 표 조각 후보(_joins) — 한 행만 넘어간 1~2칸 조각을 배치용 표로 풀면 merge_tables 가 잇지 못한다. 두 조각이
    모두 글상자 꼴이면 쪽 경계의 두 글상자로 보고 풀어 둔다(문단 이음이 맡는다). '이하 빈칸'(blank_below)은 보지만, 제목·단위·
    출처는 잇기 전이라 아직 보지 않는다."""
    ids = set()
    for pa, pb in zip(doc.pages, doc.pages[1:]):
        if (pair := _edges(pa, pb)) is None:
            continue
        a, skip_a, b, skip_b = pair
        if _joins(pa, a, skip_a, pb, b, skip_b) and not (_boxy(parse_rows(a.html)) and _boxy(parse_rows(b.html))):
            ids |= {a.id, b.id}
    return ids


def merge_tables(doc: Document) -> Document:
    """쪽 넘김 표(규칙 1) — 앞 논리 페이지의 마지막 표와 다음 논리 페이지의 첫 표가 열 수·폭이 같으면 뒤 조각에
    continues_from(세 쪽 이상이면 사슬). HTML 은 합치지 않는다 — 조각 bbox 가 제 쪽에 남아 하이라이트가 된다.

    사이의 keep=false 블록(머리말·쪽번호·결재란·공문 메타 등), '(계속)' 표시, '(단위: …)' 줄, 앞 쪽 끝 각주, 다음 쪽의
    같은 표 제목 반복은 건너뛴다. 앞 표가 그 쪽에서 끝났으면(_ended — 출처 주·'이하 빈칸'·큰 빈자리), 앞 조각이 글상자
    꼴인데 뒤 표가 제 머리행을 가지면 잇지 않는다(_joins).
    이은 경우에만 '(계속)' 표시를 keep=false(drop_reason "continued")로 — 이어진다는
    인쇄 표시일 뿐 내용이 아니고, 그 뜻은 continues_from 이 대신 전한다. 잇지 못하면 표시를 단서로 남긴다.
    뒤 조각의 header_rows 는 앞 조각 머리행을 되풀이한 행 수로 맞춘다 — 반복이 없으면 비우고(tables 가 첫 데이터 행을
    머리행으로 잡았을 수 있다) 앞 조각 머리행을 쓰라고 provenance 에 header_from 을 남긴다."""
    doc = doc.model_copy(deep=True)
    by_id = {b.id: b for p in doc.pages for b in p.blocks if b.type == "table"}
    for pa, pb in zip(doc.pages, doc.pages[1:]):
        pair = _edges(pa, pb)
        if pair is None:
            continue
        a, skip_a, b, skip_b = pair
        if not _joins(pa, a, skip_a, pb, b, skip_b):
            continue
        head = a
        while head.continues_from in by_id:
            head = by_id[head.continues_from]
        rep = repeated_rows(parse_rows(head.html), parse_rows(b.html), head.header_rows or 0)   # 사슬 가운데 조각은 머리행이 없을 수 있어 첫 조각과 견준다
        b.continues_from = a.id
        prov = {"rule": "merge_tables", "repeated_header_rows": rep} if rep else \
            {"rule": "merge_tables", "header_from": head.id}
        if (b.header_rows or 0) != rep:
            prov["header_rows_was"] = b.header_rows
            _set_header(b, rep)
        b.provenance = (b.provenance or []) + [prov]
        for m in (x for x in skip_a + skip_b if _is_mark(x)):
            m.keep, m.drop_reason = False, "continued"
            m.provenance = (m.provenance or []) + [{"rule": "merge_tables", "table": b.id}]
    return doc
