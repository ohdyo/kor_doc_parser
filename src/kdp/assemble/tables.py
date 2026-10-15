"""표(기획서 4.3 규칙 6·13, D18) — 머리행, 표 제목·단위·출처 연결, 표 갈래.

표 제목은 라벨(figure_title)만 믿지 않는다 — 실측에서 일반 구역 제목에도 붙었다. '<표 n>'·'[도표 n]' 같은 패턴이 있고
표 바로 위나 아래에 붙은 블록만 잇는다. 쪽 넘김 표 병합(규칙 1)은 M3.
"""
from __future__ import annotations

import html as htmllib
import re
from dataclasses import dataclass

from ..schema import Block, Document, Page

CAPTION = re.compile(r"^\s*[\[<〈【(]?\s*(?:표|도표|table)\s*[\dⅠ-Ⅹ]+(?:[-.]\d+)*\s*[\]>〉】)]?", re.I)   # 그림 제목은 받지 않는다
UNIT = re.compile(r"[(（]\s*단위\s*[:：]\s*([^)）]+?)\s*[)）]")
SOURCE = re.compile(r"^\s*(?:※\s*)?(?:자료|출처)\s*[:：]")
FIG_CAPTION = re.compile(r"^\s*[\[<〈【(]?\s*(?:그림|사진|figure|fig\.?)\s*\d", re.I)   # 그림 제목의 단위는 표에 붙이지 않는다
_ROW = re.compile(r"<tr[^>]*>.*?(?=<tr[\s>]|</table>|$)", re.S | re.I)
_CELL = re.compile(r"<t([dh])([^>]*)>(.*?)</t[dh]>", re.S | re.I)
MAX_GAP = 15.0                # 표와 제목·단위·출처 블록 사이 세로 틈(pt)
HEADER_MAX_CHARS = 20
EMPTY_FAILED = 0.9            # 거의 모든 칸이 비면 인식 실패로 본다(빈 서식 표는 절반쯤 비어도 정상)
LAYER_FAILED = 0.8            # born-digital 표에서 레이어 글자 중 모델 표에 있는 비율이 이보다 낮으면 인식 실패(행을 빠뜨림 등)
LAYOUT_MIN_CHARS = 30         # 1~2칸 표의 글자가 이보다 많으면 배치용 표(글상자)로 본다


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


def _gap(b: Block, t: Block) -> tuple[float, float]:
    """(위로 떨어진 거리, 아래로 떨어진 거리) — 가로로 겹치지 않으면 무한대."""
    if min(b.bbox[2], t.bbox[2]) - max(b.bbox[0], t.bbox[0]) <= 0:
        return float("inf"), float("inf")
    return t.bbox[1] - b.bbox[3], b.bbox[1] - t.bbox[3]


def _dist(b: Block, t: Block) -> float:
    """블록과 표·그림 사이 세로 거리(위든 아래든). 가로로 안 겹치거나 멀면 무한대."""
    up, down = _gap(b, t)
    return min(d for d in (up, down) if d >= -3) if (-3 <= up <= MAX_GAP or -3 <= down <= MAX_GAP) else float("inf")


def _link(page: Page, t: Block) -> None:
    """표 바로 위·아래 블록에서 제목·단위·출처를 찾는다. 두 표(그림) 사이 블록은 더 가까운 쪽 것이다 — 제목이 표 아래에 오는
    문서와 '다음 표'의 제목을 함께 가른다. 나란한 표 위의 한 제목은 두 표가 함께 받는다. 출처 주는 늘 자기 표 아래다."""
    others = [o for o in page.blocks if o is not t and o.merged_into is None and o.type in ("table", "image", "chart")]
    near = []
    for b in page.blocks:
        if b is t or not b.keep or b.merged_into or not b.text or b.type in ("table", "image", "chart"):
            continue
        up, down = _gap(b, t)
        side = "up" if -3 <= up <= MAX_GAP else "down" if -3 <= down <= MAX_GAP else None
        if side is None:
            continue
        mine = up if side == "up" else down
        if side == "down" and SOURCE.match(b.text):
            t.note_ids = (t.note_ids or []) + [b.id]
        elif not any(_dist(b, o) < mine for o in others):
            near.append((mine, side, b))
    near.sort(key=lambda x: (x[1] != "up", x[0]))                  # 위쪽 먼저, 가까운 것 먼저
    cap = next((b for _, _, b in near if CAPTION.match(b.text)), None)
    if cap:
        t.caption_id = cap.id
    for _, side, b in near:
        if not t.unit and (b is cap or (side == "up" and not FIG_CAPTION.match(b.text))) and (m := UNIT.search(b.text)):
            t.unit = m.group(1)


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


def tables(doc: Document) -> Document:
    doc = doc.model_copy(deep=True)
    for page in doc.pages:
        for t in page.blocks:
            if t.type != "table" or t.merged_into or not t.html:
                continue
            rows = parse_rows(t.html)
            cells = [c for r in rows for c in r]
            if cells and len(cells) <= 2 and len(rows) <= 2 and sum(len(c.text) for c in cells) >= LAYOUT_MIN_CHARS \
                    and _plain(t):
                t.type, t.text, t.html = "text", "\n".join(c.text for c in cells if c.text), None   # 크롭(img_path)은 남긴다
                t.provenance = (t.provenance or []) + [{"rule": "layout_table"}]   # 글상자처럼 쓴 표 → 본문
                continue
            n = header_rows(rows)
            if n:
                t.header_rows, t.html = n, _mark_th(t.html, n)
            t.route, t.route_reason = _route(t, rows)
            _link(page, t)
    return doc
