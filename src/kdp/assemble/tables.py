"""표(기획서 4.3 규칙 1·6·13, D18) — 머리행, 표·그림 제목·단위·출처 연결, 표 갈래, '이하 빈칸' 행, 쪽 넘김 표.

표 제목은 라벨(figure_title)만 믿지 않는다 — 실측에서 일반 구역 제목에도 붙었다. '<표 n>'·'[도표 n]' 같은 패턴이 있고
표 바로 위나 아래에 붙은 블록, 또는 표 바로 위의 괄호로 감싼 한 줄 제목('[ 발령 기준 ]')만 잇는다. 그림은 바로 위·아래의
'<그림 n>'·괄호 제목, 그다음 검출기가 그림 제목(figure_title)으로 본 줄을 잇는다.
"""
from __future__ import annotations

import html as htmllib
import re
import unicodedata
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
_NOTE_IN = re.compile(r"\s*(?:단\s*위\s*[:：]|(?:개\s*정|신\s*설|삭\s*제|전\s*문\s*개\s*정|본\s*조\s*신\s*설|종\s*전|시\s*행)(?![가-힣]))")   # 괄호 안 단위·개정 표시('【단위: 천원】'·'<개정 2024. 1. 1.>')
_TAG_ONLY = re.compile(r"(?:붙\s*임|별\s*첨|첨\s*부|부\s*록|별\s*지|별\s*표|참\s*고)(?:\s*자\s*료)?\s*(?:제\s*)?"
                       r"(?:[\dⅠ-Ⅻ]+(?:\s*[-–]\s*\d+)*)?\s*(?:호)?\s*(?:서\s*식)?")   # 괄호 안이 꼬리표뿐('[별지 제1호서식]'·'[붙임 1-1]')
_ROW = re.compile(r"<tr[^>]*>.*?(?=<tr[\s>]|</table>|$)", re.S | re.I)
_CELL = re.compile(r"<t([dh])([^>]*)>(.*?)</t[dh]>", re.S | re.I)
MAX_GAP = 15.0                # 표와 제목·단위·출처 블록 사이 세로 틈(pt)
FIG_GAP = 30.0                # '【그림 n】' 처럼 꼴이 분명한 그림 제목은 이만큼 떨어져도 잇는다(실측 test3.pdf 16pt)
HEADER_MAX_CHARS = 20         # 머리글 칸 글자 수(띄어쓰기 뺀) 상한
_NUMERIC = re.compile(r"[\d\s,.%\-–~+()℃°/:]+")   # 숫자 칸(값·연도·날짜·전화) — 본문 칸·칸 이름 행 판정
_DIGITS = re.compile(r"[\d,.%\s\-~]+")              # 첫 행의 숫자 칸('2025'·'1,000') — '09:00'·'1/4' 은 시각·분기 머리글이다
_PHONE = re.compile(r"\d{2,4}\s*[-)]\s*\d{3,4}\s*-\s*\d{4}")   # 전화번호가 든 칸('과장 ○○○ (043-…)') — 담당 부서 표의 자료 행
NUMERIC_BODY = 0.6            # 본문 칸 중 이만큼 이상이 숫자면 숫자 표 — 머리글이 숫자보다 긴 것이 보통이다
PANEL = re.compile(r"\s*[(（]?\s*[a-zA-Z가-하]\s*[)）]\s*")   # 나란한 그림의 패널 표지 '(a)'·'(나)'
PANEL_GAP = 6.0               # 패널 표지와 그림 사이 세로 틈(pt)
INSIDE = 0.9                  # 그림 상자 안에 이만큼 이상 든 글 블록은 그림 속 글이다
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


def _labels(row: list[Cell]) -> bool:
    """칸 이름 행 — 모든 칸이 짧은 글이고 숫자 칸이 없다."""
    return len(row) >= 2 and all(c.text and len(_flat(c.text)) <= HEADER_MAX_CHARS and not _NUMERIC.fullmatch(c.text)
                                 for c in row)


def _numbers(row: list[Cell]) -> bool:
    texts = [c.text for c in row if c.text]
    return bool(texts) and 2 * sum(bool(_NUMERIC.fullmatch(t)) for t in texts) >= len(texts)


def _flat(text: str) -> str:
    return "".join(text.split())


def _grid(first: list[Cell], second: list[Cell]) -> tuple[list[tuple[int, int, bool]], dict[int, str]]:
    """첫 행 칸마다 (시작 열, 끝 열, 세로 병합)과 둘째 행 칸의 시작 열 → 글 — 첫 행 세로 병합 칸 아래 열에는 둘째 행 칸이 없다."""
    spans, col = [], 0
    for c in first:
        spans.append((col, col + c.colspan, c.rowspan > 1))
        col += c.colspan
    free = [i for i in range(col) if not any(tall and a <= i < b for a, b, tall in spans)]
    at, k = {}, 0
    for c in second:
        if k < len(free):
            at[free[k]] = c.text
        k += c.colspan
    return spans, at


def _num(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _sub_labels(first: list[Cell], second: list[Cell]) -> bool:
    """가로 병합 아래 칸 이름 행 — 모든 칸이 짧은 글이고 모두 가로 병합 칸 아래 있다(첫 행 보통 칸 아래 행 이름이 있으면 자료 행)."""
    spans, at = _grid(first, second)
    return _labels(second) and all(any(b - a > 1 and a <= i < b for a, b, _ in spans) for i in at)


def _repeated(first: list[Cell], second: list[Cell], below: list[list[Cell]]) -> bool:
    """가로 병합 칸마다 그 아래 둘째 행 칸 이름이 똑같이 되풀이된다('연도·값' × 5, '21·22·23·24' × 기온·강수 — 실측 test3.pdf,
    test2.pdf GPU 출력). 숫자 칸 이름은 셋 이상 늘어나는 차례일 때만 — 우연히 같은 값('1·0')은 자료 행이다. 글 칸 이름이
    아래 행(below)의 같은 열 값으로 다시 나오면 자료 행이다('적합·해당 없음' × 1차·2차 점검)."""
    spans, at = _grid(first, second)
    groups = [tuple(at.get(i) for i in range(a, b)) for a, b, tall in spans if b - a > 1 and not tall]
    if len(groups) < 2 or len(set(groups)) != 1 or not all(groups[0]) or len(set(groups[0])) != len(groups[0]):
        return False
    nums = [_num(t) for t in groups[0]]
    if all(v is None for v in nums):
        cols = [i for a, b, tall in spans if b - a > 1 and not tall for i in range(a, b)]
        return not {_grid(first, r)[1].get(i) for r in below for i in cols} & set(groups[0])
    return len(nums) >= 3 and None not in nums and all(a < b for a, b in zip(nums, nums[1:]))


def header_rows(rows: list[list[Cell]]) -> int:
    """첫 행이 짧은 글자들이고(숫자·전화 칸 없음) 본문 칸보다 길지 않으면 머리행 — 본문이 숫자 위주인 표는 길이를 견주지
    않는다(머리글이 숫자보다 길다, 실측 test3.pdf). 첫 행에 가로 병합이 있으면 세로 병합 깊이까지, 병합 아래 칸 이름 행
    ('번호·지점명·연도·값' — 그 아래 행이 숫자, 또는 병합 칸마다 되풀이되는 이름)까지. 첫 행이 표 전체를 덮는 제목 칸 하나면
    그 아래는 칸 이름 행으로 보지 않는다(개요표 '사업명 | …' 는 자료 행). 가로 병합 머리글의 대각선 머리칸은
    비어 있어도 된다(모델이 머리칸 글을 둘째 행에 둔다)."""
    if len(rows) < 2 or not rows[0]:
        return 0
    first = rows[0]
    wide = any(c.colspan > 1 for c in first)
    if any(not c.text or len(_flat(c.text)) > HEADER_MAX_CHARS or _DIGITS.fullmatch(c.text) or _PHONE.search(c.text)
           for c in (first[1:] if wide and not first[0].text else first)):
        return 0
    body = [c.text for r in rows[1:] for c in r if c.text]
    numeric = sum(bool(_NUMERIC.fullmatch(t)) for t in body) >= NUMERIC_BODY * len(body)
    if body and not numeric and max(len(c.text) for c in first) > max(len(t) for t in body):
        return 0
    n = max(c.rowspan for c in first) if wide else 1
    if wide and n == 1 and len(rows) > 2 and (len(first) > 1 and _sub_labels(first, rows[1]) and _numbers(rows[2])
                                              or _repeated(first, rows[1], rows[2:])):
        n = 2
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


def _dist(page: Page, b: Block, t: Block, lim: float = MAX_GAP) -> float:
    """블록과 표·그림 사이 세로 거리(위든 아래든). 가로로 안 겹치거나 lim 보다 멀면 무한대."""
    up, down = _gap(page, b, t)
    return min(d for d in (up, down) if d >= -3) if (-3 <= up <= lim or -3 <= down <= lim) else float("inf")


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
        or next((b for _, side, b in near if side == "up" and b.type in ("text", "figure_title") and bracket_title(b.text)), None)
    if cap:
        t.caption_id = cap.id
    for _, side, b in near:
        if not t.unit and (b is cap or (side == "up" and not FIG_CAPTION.match(b.text))) and (m := UNIT.search(b.text)):
            t.unit = m.group(1)


def bracket_title(text: str | None) -> bool:
    """괄호로 감싼 한 줄 제목('[ 국내 환자 발생 추이 ]'·'<발령 기준>') — 번호 없는 표·그림 제목(실측 test2.pdf). '(단위: …)'·
    '(계속)' 줄과 꼬리표('[참고]'·'[별지 제1호서식]'·'[붙임 1-1]' — 붙임 구역 뿌리다)는 아니다."""
    s = (text or "").strip()
    if not _BRACKET.fullmatch(s) or _UNIT_LINE.fullmatch(s) or CONT.fullmatch("".join(s.split())) \
            or _TAG_ONLY.fullmatch(s[1:-1].strip()) or _NOTE_IN.match(s[1:-1]):
        return False
    return BRACKET_MIN <= len("".join(s.split())) - 2 <= BRACKET_MAX


def _link_figure(page: Page, f: Block) -> None:
    """그림·차트 바로 위·아래(MAX_GAP 안)의 제목 — '<그림 n>'(아래 먼저 — 관례) → 괄호 제목(위 먼저) → 검출기 그림 제목
    (figure_title, 위 먼저 — 그래프 바로 아래 축 이름도 figure_title 로 잡힌다, 실측 — 읽기 순서로도 바로 앞뒤일 때만) 차례로,
    같으면 가까운 것. '<그림 n>' 은 FIG_GAP 까지 받되 MAX_GAP 밖이면 MAX_GAP 안 후보에 진다(제목을 위에 다는
    문서에서 쪽 끝에 남은 다음 그림 제목 — MAX_GAP 안 후보가 하나라도 있으면 그쪽에서 고른다). '<그림 2>는 …'
    같은 본문 참조와 괄호로 감싼 구역·문서 제목 라벨(paragraph_title·doc_title)은 받지 않는다. 다른 표·그림에 더 가까운
    블록은 그쪽 것이고, 나란한 그림 위의 한 제목은 함께 받는다."""
    others = [o for o in page.blocks if o is not f and o.merged_into is None and o.type in ("table", "image", "chart")]
    parts = {p["with"] for o in others + [f] for p in o.provenance or [] if p.get("rule") == "figure_part"}
    order = [o for o in page.blocks if o.keep and o.merged_into is None or o.id in parts]   # 그림 글로 묶인 줄도 사이 블록이다
    near = []
    for b in page.blocks:
        if b is f or not b.keep or b.merged_into or not b.text or b.type in ("table", "image", "chart"):
            continue
        fig = bool(FIG_CAPTION.match(b.text) and appendage(b.text))
        lim = FIG_GAP if fig else MAX_GAP
        up, down = _gap(page, b, f)
        side = "up" if -3 <= up <= lim else "down" if -3 <= down <= lim else None
        if side is None:
            continue
        mine = up if side == "up" else down
        bracket = b.type in ("text", "figure_title") and bracket_title(b.text)
        weak = not (fig or bracket) and b.type == "figure_title" and _beside(order, b, f)   # 라벨뿐인 줄은 읽기 순서로도 바로 앞뒤
        if (fig or bracket or weak) and not any(_dist(page, b, o, lim) < mine for o in others):
            near.append((mine > MAX_GAP, 0 if fig else 1 if bracket else 2, side != ("down" if fig else "up"), mine, b))
    if near:
        f.caption_id = min(near, key=lambda x: x[:4])[4].id


def _beside(order: list[Block], a: Block, b: Block) -> bool:
    """읽기 순서로 바로 앞뒤 — 사이의 '(단위: …)' 줄은 건너뛴다(headings._candidates 와 같은 기준 — 그림 옆 단의 글을 건너
    뛰어 구역 제목을 잇지 않는다)."""
    i, j = sorted((order.index(a), order.index(b)))
    return all(_unit_of(x) for x in order[i + 1:j])


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
        figs = [f for f in page.blocks if f.type in ("image", "chart") and f.keep and f.merged_into is None]
        for f in figs:
            _noise(f)
            _absorb(page, f)
        for f in figs:                                             # 그림 글을 모두 묶은 뒤 — 묶인 줄을 다른 그림 제목으로 잇지 않게
            _link_figure(page, f)
    return doc


def _letters(text: str) -> bool:
    """가나가 아닌 글자·숫자가 있나 — 한글·영문·한자·전각 숫자는 글, 가나와 기호뿐이면 지어낸 글."""
    return any(ch.isalnum() and not "\u3040" <= ch <= "\u30ff" for ch in unicodedata.normalize("NFKC", text))


def _noise(f: Block) -> None:
    """글 없는 그림(만화·삽화)에서 그림 속 글자 읽기가 지어낸 글 — 가나·기호뿐인 글('ふ～～～'·'うっ')은 버린다(실측 test2.pdf
    8쪽 — CPU·GPU 모두, GPU 는 길이 한도로 잘림). 그래프 글은 잘렸으면 모두 버린다 — 그래프 속 글자는 짧아 한도까지 간 글은
    지어낸 되풀이고 그 앞줄의 값도 그림에 없었다(실측 test3.pdf — 등고선 지도 '203.0, 203.1 …', 그래프 '100.0 / 경수량' 되풀이).
    버린 글의 잘림 표시는 기록으로 옮긴다(출력에 남은 잘린 글이 없다). 그 밖의 잘린 글은 글자가 있으면 둔다(QC 가 잘림으로 센다)."""
    if f.figure_text and (not _letters(f.figure_text) or f.type == "chart" and f.truncated):
        f.provenance = (f.provenance or []) + [{"rule": "figure_text_noise", "before": f.figure_text[:40],
                                                **({"truncated": f.truncated} if f.truncated else {})}]
        f.figure_text, f.truncated = None, None


def _side(page: Page, b: Block, f: Block) -> float:
    """블록이 표·그림 바로 위·아래에 붙었으면(가로로 그 폭 안, 세로 틈 -3~PANEL_GAP) 그 틈, 아니면 무한대."""
    x0, y0, x1, y1 = local(page, b.bbox)
    fx0, fy0, fx1, fy1 = local(page, f.bbox)
    gaps = [d for d in (fy0 - y1, y0 - fy1) if -3 <= d <= PANEL_GAP] if x0 >= fx0 - 3 and x1 <= fx1 + 3 else []
    return min(gaps, default=float("inf"))


def _named_near(page: Page, f: Block) -> bool:
    """그림 위·아래에 번호·괄호 제목 후보('【그림 2】 …'·'< … >')가 따로 있나 — _link_figure 와 같은 꼴·거리."""
    for o in page.blocks:
        if o is f or not o.keep or o.merged_into or not o.text or o.type not in ("text", "figure_title"):
            continue
        fig = bool(FIG_CAPTION.match(o.text) and appendage(o.text))
        if (fig or bracket_title(o.text)) and _dist(page, o, f, FIG_GAP if fig else MAX_GAP) < float("inf"):
            return True
    return False


def _absorb(page: Page, f: Block) -> None:
    """그림에 딸린 글 블록을 그림 글(figure_text)로 옮긴다 — 글자층이 있는 쪽에서 블록 자리에 글자층 글자가 없는(no_layer_chars,
    글이 그림에 그려졌다) 블록 가운데 그림 상자 안에 든 것과 그림 바로 위·아래에 붙은 그림 제목 라벨 줄(검출기가 그래프 그림 맨
    위 줄을 그림 밖 그림 제목으로 잡는다, 실측 test3.pdf — 그 그림에 번호·괄호 제목이 따로 있을 때만, 없으면 그 줄이 그림 제목이다),
    그리고 그림 바로 위·아래의 패널 표지('(a)') — 위·아래 줄은 다른 표·그림이 그만큼 가깝지 않을 때만. 글자층 글자가 있는
    글은 그림 상자가 덮어도 본문이다(그림 틀이 글 위로 겹친 쪽). 잘린·오류 블록은 옮기지 않는다. 옮긴 블록은 병합 묶음 규약대로."""
    fx0, fy0, fx1, fy1 = local(page, f.bbox)
    others = [o for o in page.blocks if o is not f and o.merged_into is None and o.type in ("table", "image", "chart")]
    titled = _named_near(page, f)
    for b in page.blocks:
        if b is f or not b.keep or b.merged_into or not b.text or b.type not in ("text", "figure_title"):
            continue
        x0, y0, x1, y1 = local(page, b.bbox)
        area = max(x1 - x0, 0) * max(y1 - y0, 0)
        inner = max(min(x1, fx1) - max(x0, fx0), 0) * max(min(y1, fy1) - max(y0, fy0), 0)
        if b.truncated or b.error:                       # 잘린·오류 블록은 그대로 둬 QC 가 센다
            continue
        bare = any(p.get("rule") == "no_layer_chars" for p in b.provenance or [])   # 글자층 있는 쪽인데 블록 자리에 글자가 없다
        named = FIG_CAPTION.match(b.text) or CAPTION.match(b.text) or bracket_title(b.text)   # 그림 상자에 걸친 제목은 제목
        inside = area and inner >= INSIDE * area and bare and not named
        gap = _side(page, b, f)
        beside = gap < float("inf") and not any(_side(page, b, o) <= gap for o in others)  # 위아래로 붙인 그림은 가까운 쪽(같으면 두기)
        panel = beside and PANEL.fullmatch(b.text)
        drawn = beside and b.type == "figure_title" and bare and not named and titled
        if inside or panel or drawn:
            f.figure_text = "\n".join(t for t in (f.figure_text, b.text.strip()) if t)
            f.provenance = (f.provenance or []) + [{"rule": "figure_part", "with": b.id}]
            b.keep, b.drop_reason, b.merged_into, b.text = False, "merged", f.id, None


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


@dataclass
class _Slot:
    """행 묶음으로 다시 쓸 칸 — 원래 태그·속성·안쪽 HTML 그대로, 차지하는 행 [top, bottom)과 시작 열."""
    tag: str
    attrs: str
    inner: str
    top: int
    bottom: int
    col: int


_ROWSPAN = re.compile(r"""\s*rowspan\s*=\s*["']?\d+["']?""", re.I)
_TABLE_TAG = re.compile(r"\s*(<table[^>]*>)", re.I)


def _slots(html: str, skip: int = 0, offset: int = 0) -> tuple[int, list[_Slot]]:
    """칸 자리 — 세로 병합이 차지한 자리를 건너 열을 센다(n_cols 와 같은 기준). 앞 skip 행은 빼고 남은 행을 offset 부터 센다
    (뺀 행에서 내려온 세로 병합 칸은 남은 행만 차지한다). (남은 행 수, 칸)."""
    rows = _ROW.findall(html)
    taken: dict[int, set[int]] = {}
    out = []
    for r, row in enumerate(rows):
        c = 0
        for m in _CELL.finditer(row):
            while c in taken.get(r, ()):
                c += 1
            cs = max(_span(m.group(2), "colspan"), 1)
            end = min(r + max(_span(m.group(2), "rowspan"), 1), len(rows))
            for rr in range(r, end):
                taken.setdefault(rr, set()).update(range(c, c + cs))
            if end > skip:
                out.append(_Slot("t" + m.group(1).lower(), _ROWSPAN.sub("", m.group(2)), m.group(3),
                                 offset + max(r, skip) - skip, offset + end - skip, c))
            c += cs
    return max(len(rows) - skip, 0), out


def _rows_html(slots: list[_Slot], lo: int, hi: int) -> str:
    """행 [lo, hi) 의 <tr> — 첫 행에는 위에서 내려온 세로 병합 칸을 다시 쓰고, 세로 병합은 hi 에서 자른다."""
    out = []
    for r in range(lo, hi):
        cells = []
        for s in sorted((s for s in slots if s.top == r or (r == lo and s.top < lo < s.bottom)), key=lambda s: s.col):
            n = min(s.bottom, hi) - r
            span = f' rowspan="{n}"' if n > 1 else ""
            cells.append(f"<{s.tag}{span}{s.attrs}>{s.inner}</{s.tag}>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    return "".join(out)


def regroup(header: tuple[str, int], pieces: list[tuple[str, int]], max_chars: int) -> list[str]:
    """표를 행 묶음 표들로 다시 쓴다(D21 — RAG 적재용 마크다운). header = (머리행이 든 표 HTML, 머리행 수), pieces = [(표 HTML,
    앞에서 뺄 행 수)] — 쪽 넘김 표면 첫 조각은 머리행만큼, 뒤 조각은 되풀이한 머리행만큼 뺀다. 남은 행을 이어 묶음마다 머리행을
    앞에 두고, 묶음 HTML 이 max_chars 를 넘기 전에 끊는다(행 하나로 넘으면 그 행만). 세로 병합 칸이 경계에 걸리면 두 묶음에
    나눠 쓴다(글은 되풀이). 남은 행이 없으면 머리행만 든 표 하나."""
    html, n_head = header
    tag = m.group(1) if (m := _TABLE_TAG.match(html)) else "<table>"
    rows, cells = _slots(html)
    head = _rows_html(cells, 0, min(n_head, rows))
    slots, n = [], 0
    for h, skip in pieces:
        k, s = _slots(h, skip, n)
        slots += s
        n += k

    def table(lo: int, hi: int) -> str:
        return tag + head + _rows_html(slots, lo, hi) + "</table>"

    if not n:
        return [tag + head + "</table>"]
    out, lo = [], 0
    while lo < n:
        hi = lo + 1
        while hi < n and len(table(lo, hi + 1)) <= max_chars:
            hi += 1
        out.append(table(lo, hi))
        lo = hi
    return out


def _expand(html: str) -> list[list[str]]:
    """표 HTML → 칸 글 격자 — 세로·가로 병합 칸의 글을 차지한 자리마다 되풀이한다(행 하나만 떼어도 읽히게)."""
    rows = _ROW.findall(html)
    grid: dict[tuple[int, int], str] = {}
    width = 0
    for r, row in enumerate(rows):
        c = 0
        for m in _CELL.finditer(row):
            while (r, c) in grid:
                c += 1
            cs, rs = max(_span(m.group(2), "colspan"), 1), max(_span(m.group(2), "rowspan"), 1)
            text = " ".join(htmllib.unescape(re.sub(r"<[^>]+>", " ", m.group(3))).split())
            for rr in range(r, min(r + rs, len(rows))):
                for cc in range(c, c + cs):
                    grid[(rr, cc)] = text
            c += cs
            width = max(width, c)
    return [[grid.get((r, c), "") for c in range(width)] for r in range(len(rows))]


def _pipe(cells: list[str], width: int) -> str:
    cells = cells + [""] * (width - len(cells))
    return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"


def pipe_regroup(header: tuple[str, int], pieces: list[tuple[str, int]], max_chars: int) -> list[str]:
    """regroup 과 같은 묶음을 HTML 대신 파이프 표로(D21 — RAG 적재용 표의 가벼운 꼴). 병합 칸은 풀어 행마다 칸 글을 되풀이하고,
    머리행 여러 줄은 열마다 위에서 아래로 다른 글만 ' · ' 로 이어 한 줄로 합친다. 머리행이 없으면 칸 이름 줄·구분 줄 없이 행만."""
    html, n_head = header
    top = _expand(html)[:n_head]
    body = [row for h, skip in pieces for row in _expand(h)[skip:]]
    width = max(map(len, top + body), default=0)
    names = []
    for c in range(width):
        parts = []
        for row in top:
            t = row[c] if c < len(row) else ""
            if t and (not parts or parts[-1] != t):
                parts.append(t)
        names.append(" · ".join(parts))
    head = [_pipe(names, width), "|" + "---|" * width] if top else []
    rows = [_pipe(r, width) for r in body]
    if not rows:
        return ["\n".join(head)] if head else []
    out, lo = [], 0
    while lo < len(rows):
        hi = lo + 1
        while hi < len(rows) and len("\n".join(head + rows[lo:hi + 1])) <= max_chars:
            hi += 1
        out.append("\n".join(head + rows[lo:hi]))
        lo = hi
    return out


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
