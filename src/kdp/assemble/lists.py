"""항목 계층(기획서 4.3 규칙 5) — 공문 8단계(1. → 가. → 1) → 가) → (1) → (가) → ① → ㉮)와 보고서체(□ → ○ → - → ·).

VLM 이 읽은 글머리 기호는 믿을 수 없다(○ 가 ☐ 로, PPT 기호는 사라짐 — 실측). 그래서 섹션(가장 가까운 제목) 안에서 번호·기호로
시작하는 항목의 들여쓰기 x(논리 쪽 너비 대비, 쪽마다 가로 밀림을 맞춘 뒤)를 묶어 왼쪽부터 1, 2, … 로 매기고, VLM 이 읽은 기호
모양은 수준에 쓰지 않는다. 번호는 보조로만 —
들여쓰지 않아 한 x 에 번호 체계가 섞이면(1. 과 가.) 번호 차례로 나눈다. 글자층에서 온 글의 보고서체 기호(□ ○ - ·)는 믿을 수
있어 번호처럼 차례로 나눈다(실측 test3.pdf — 검출기 상자 왼쪽 끝이 들쭉날쭉해 □ 와 들여쓴 ○ 가 한 x 묶음이 됐다). 모델이 읽은
기호(검증 표본)는 같은 묶음에서 글자층으로 확인된 기호일 때만 쓴다. 번호가 있는 묶음의 기호는 나누지 않고(번호와 한 스택에
섞지 않는다), 같은 기호가 상자 흔들림만큼 밀려 갈린 옆 묶음은 합치며, 얕은 항목 아래 더 깊은 묶음은 갈래마다 새로 쌓는다.
기호 없는 블록은 섹션 왼쪽 끝이면 문단, 번호·기호 있는
항목과 같은 x 면 기호를 잃은 항목으로 그 수준을 받고, 그 밖(첫 줄 들여쓰기 문단·항목의 이어진 글)은 비운다. 기호는 원문대로 둔다.
"""
from __future__ import annotations

import re
from itertools import groupby

from ..geometry import local, page_size
from ..schema import Block, Document, Page
from .headings import _ORDER, TITLE_TYPES, _ordinal, doc_title_key, marker, numbering, place

INDENT_TOL = 0.008        # 같은 수준으로 보는 x 차이(논리 쪽 너비 대비, A4 에서 약 5pt) — 블록 상자 흔들림
MAX_INDENT = 0.3          # 섹션 왼쪽 끝에서 이보다 오른쪽(가운데 맞춤 글·오른쪽 날짜·다른 단)은 항목으로 보지 않는다
MAX_SHIFT = 0.06          # 쪽 사이 가로 밀림으로 보는 최대 x 차이(논리 쪽 너비 대비, A4 약 36pt) — 스캔 밀림·맞쪽(제본) 여백
NOTE = "※"                # 주석 기호 — 자리가 일정하지 않아(□ 자리·○ 자리) 쪽 밀림 근거로 쓰지 않는다
_START = object()
_ATTACH = re.compile(r"\s*[\[<〈【(「『]?\s*(?:붙\s*임|첨\s*부|별\s*첨)\s*[\]>〉】)」』]?\s*[:：]?\s*")   # 번호 없는 붙임 꼬리표
_END = re.compile(r"끝\s*\.\s*$")      # 붙임 목록 끝(공문 '… 1부.  끝.')
_BARE = re.compile(r"\s*(\d{1,2})\s+\S")   # 마침표를 놓친 번호('3 개인정보 …' — OCR)


def _sections(doc: Document):
    """섹션마다 (쪽, 글 블록) 목록 — 경로(section_path)가 바뀌는 곳과 처음 보는 문서 제목(수준 0 — 경로 밖이라 경로가 그대로일
    수 있다: 공문 뒤 계획서, 쪽마다 다른 슬라이드 제목)에서 끊는다. 그 밖의 제목 블록은 빼기만 한다 — 표 제목으로 이은 제목과
    쪽마다 다시 찍은 제목(같은 문서 제목 포함)은 경로를 바꾸지 않으므로 섹션도 끊지 않는다(끊으면 표 아래·다음 쪽 하위 항목이
    1 이 된다)."""
    sec, last, docs = [], _START, set()
    for page in doc.pages:
        if not page.keep:
            continue
        for b in page.blocks:
            if not b.keep or b.merged_into is not None:
                continue
            if b.heading_level == 0 and (k := doc_title_key(b.text)) not in docs:
                docs.add(k)
                if sec:
                    yield sec
                sec, last = [], _START
            if b.type in TITLE_TYPES or b.heading_level is not None:
                continue
            if b.section_path != last:
                if sec:
                    yield sec
                sec, last = [], b.section_path
            if b.type == "text" and (b.text or "").strip():
                sec.append((page, b))
    if sec:
        yield sec


def _shifts(rows: list[tuple[int, Block, float, tuple | None]]) -> dict[int, float]:
    """논리 페이지마다 가로 밀림(스캔 밀림·맞쪽 여백) — 그 쪽의 체계(번호·기호, '※' 빼고)마다 맨 왼쪽 항목 x 를 앞 쪽들에 나온
    같은 체계의 가장 가까운 x 에 맞춘 차이를 재고, 서로 다른 체계 둘 이상이 같은 차이(상자 흔들림 안)를 가리킬 때만 뺀다 —
    한 체계만으로는 하위 항목만 이어진 쪽('○' 만), 잘못 읽은 기호(□·○ 가 모두 ☐), 두 수준에 쓴 같은 기호(PPT • – •)와 가를 수
    없다. 차이가 상자 흔들림 안이거나 너무 크면 0. x 가 같은 기호끼리 글자 순으로 고르지 않는다."""
    seen: dict[str, list[float]] = {}
    out = {}
    for no, group in groupby(rows, key=lambda r: r[0]):
        items = [(x, m[0]) for _, _, x, m in group if m]
        left: dict[str, float] = {}
        for x, k in items:
            if k != NOTE:
                left[k] = min(x, left.get(k, x))
        diffs = sorted(min((x - r for r in seen[k]), key=abs) for k, x in left.items() if k in seen)
        votes = [(sum(abs(d - c) <= INDENT_TOL for d in diffs), -abs(c), c) for c in diffs]
        n, _, s = max(votes, default=(0, 0.0, 0.0))
        out[no] = s if n >= 2 and INDENT_TOL < abs(s) <= MAX_SHIFT else 0.0
        for x, k in items:
            seen.setdefault(k, []).append(x - out[no])
    return out


def _kind(n: tuple[str, str] | None) -> str | None:
    """항목 수준에서는 ❶ 과 ① 을 한 체계로 — 한 목록을 VLM·글자층이 다른 원 숫자로 읽기도 한다(제목 스택에서는 따로)."""
    return None if n is None else "①" if n[0] == "❶" else n[0]


def _symbol(b: Block) -> str | None:
    """글자층에서 온 글의 보고서체 기호(□ ○ - ·) — 모델이 읽은 기호는 믿을 수 없지만 글자층 기호는 원문 그대로다."""
    if not any(p.get("rule") == "text_layer" for p in b.provenance or []):
        return None
    m = marker(b.text)
    return m[0] if m and m[0] in _ORDER else None


def _place(kinds: list, kind: str) -> int:
    """place() 에 보고서체 기호 차례를 더한다 — 처음 나온 기호가 스택에 쌓인 아래 차례 기호보다 위면 그 자리로 올린다(제목
    상자 요약 '-' 줄 뒤에 처음 나온 □ 문장이 '-' 밑으로 들어가지 않게)."""
    if kind not in kinds and kind in _ORDER:
        for i, k in enumerate(kinds):
            if k in _ORDER and _ORDER.index(k) > _ORDER.index(kind):
                return i
    return place(kinds, kind)


def _rows(items: list[tuple[Page, Block]]) -> list[tuple[Block, float, tuple | None]]:
    """(블록, 쪽 가로 밀림을 뺀 x, 줄 머리 번호·기호)."""
    rows = [(page.page_no, b, local(page, b.bbox)[0] / page_size(page)[0], marker(b.text)) for page, b in items]
    shift = _shifts(rows)
    return [(b, x - shift[no], m) for no, b, x, m in rows]


def _levels(rows: list[tuple[Block, float, tuple | None]], skip=frozenset()) -> list[tuple[Block, int, str]]:
    """skip(붙임 목록) 블록은 수준을 매기지 않지만 섹션 왼쪽 끝은 그것까지 넣어 잰다."""
    base = min(x for _, x, _ in rows)                       # 섹션 왼쪽 끝
    xs = sorted(x for b, x, m in rows if m and b.id not in skip and x - base <= MAX_INDENT)
    if not xs:
        return []                                         # 항목이 없는 섹션 — 들여쓴 글은 첫 줄 들여쓰기 문단일 수 있다
    spans = [[xs[0], xs[0]]]                              # 번호·기호 있는 항목의 x 묶음
    for x in xs[1:]:
        if x - spans[-1][1] > INDENT_TOL:
            spans.append([x, x])
        else:
            spans[-1][1] = x
    def span(x):
        return next((i for i, (lo, hi) in enumerate(spans) if lo - INDENT_TOL <= x <= hi + INDENT_TOL), None)

    sure = [set() for _ in spans]                         # 묶음마다 글자층으로 확인된 보고서체 기호 — 모델이 읽은 기호는 이것만 쓴다
    for b, x, _ in rows:
        if b.id not in skip and (g := span(x)) is not None and (s := _symbol(b)):
            sure[g].add(s)
    for g in range(len(spans) - 1, 0, -1):                # 같은 기호가 상자 흔들림만큼 밀려 옆 묶음이 됐으면 한 묶음으로
        if sure[g] & sure[g - 1] and spans[g][0] - spans[g - 1][1] <= 2 * INDENT_TOL:
            spans[g - 1][1] = spans.pop(g)[1]
            sure[g - 1] |= sure.pop(g)
    kinds = [[] for _ in spans]                           # 묶음마다 번호 체계 스택(한 x 에 번호 체계가 섞인 문서)
    cur, depth, found = [0] * len(spans), [1] * len(spans), []
    for b, x, m in rows:
        if b.id in skip:
            continue
        g = span(x)
        if g is None or x - base > MAX_INDENT or (not m and x - base <= INDENT_TOL):
            continue                                      # 묶음 밖 글, 기호 없이 왼쪽 끝인 문단
        if k := m and _kind(numbering(b.text)):
            kinds[g] = [c for c in kinds[g] if c not in _ORDER]   # 번호와 기호는 한 스택에 섞지 않는다
        elif m and all(c in _ORDER for c in kinds[g]):    # 번호가 있는 묶음의 기호는 앞 번호의 수준을 받는다
            k = _symbol(b) or (m[0] if m[0] in sure[g] else None)
        if k:
            t = _place(kinds[g], k)
            del kinds[g][t:]
            kinds[g].append(k)
            cur[g], depth[g] = t, max(depth[g], t + 1)
            for h in range(g + 1, len(spans)):            # 얕은 항목 아래 더 깊은 묶음은 새로 쌓는다(갈래마다 다른 기호)
                kinds[h], cur[h] = [], 0
        found.append((b, g, cur[g]))                       # 기호·기호 없는 항목은 앞 번호 항목의 하위 수준을 받는다
    offset = [sum(depth[:i]) for i in range(len(spans))]
    return [(b, offset[g] + sub + 1, "marker" if depth[g] > 1 else "indent") for b, g, sub in found]


def _attachments(rows: list[tuple[Block, float, tuple | None]]) -> dict[str, int | None]:
    """붙임 목록('붙임 1. 계획 1부. 2. …'·'<붙임> 1. …') → {블록 id: 수준}. 섹션 왼쪽 끝의 '꼬리표 + 번호' 줄과, 그 오른쪽(꼬리표
    폭만큼 들여 쓴) 한 x 에서 다음 번호로 이어지는 같은 체계 항목(마침표를 놓친 '3 …' 도)은 수준 1 — 앞 항목의 하위가 아니다
    (실측 test2.pdf). 그보다 깊은 줄은 건너며 기호 있는 줄은 들여쓰기 깊이대로 2, 3, …, 기호 없는 줄(주·넘친 줄)은 비운다.
    '끝.' 줄 다음이나 그 밖의 블록에서 끝낸다. 꼬리표만 있는 블록은 headings 가 붙임 구역을 새로 연다."""
    base = min(x for _, x, _ in rows)
    out: dict[str, int | None] = {}
    i = 0
    while i < len(rows):
        b, x, _ = rows[i]
        i += 1
        m = _ATTACH.match(b.text or "")
        n = numbering((b.text or "")[m.end():]) if m else None
        prev = _ordinal(n[1]) if n else ()
        if not prev or x - base > INDENT_TOL:
            continue
        out[b.id], kind, ex, deep = 1, _kind(n), None, []
        while i < len(rows) and not _END.search(rows[i - 1][0].text or ""):
            c, cx, cm = rows[i]
            nxt = prev[:-1] + (prev[-1] + 1,)
            cn = numbering(c.text)
            bare = _BARE.match(c.text or "") if kind == "1." and len(prev) == 1 else None
            if (cn and _kind(cn) == kind and _ordinal(cn[1]) == nxt or bare and int(bare.group(1)) == nxt[0]) \
                    and cx > x + INDENT_TOL and (ex is None or abs(cx - ex) <= INDENT_TOL):
                out[c.id], prev, ex = 1, nxt, cx if ex is None else ex
            elif ex is not None and cx > ex + INDENT_TOL:
                deep.append((c, cx, cm))                   # 항목 아래 주·하위 항목·넘친 줄
            else:
                break
            i += 1
        xs = sorted(cx for _, cx, cm in deep if cm)        # 하위 항목 x 묶음 — 왼쪽부터 2, 3, …
        spans = []
        for v in xs:
            if not spans or v - spans[-1] > INDENT_TOL:
                spans.append(v)
            else:
                spans[-1] = v
        for c, cx, cm in deep:
            out[c.id] = 2 + next(k for k, v in enumerate(spans) if cx <= v + INDENT_TOL) if cm else None
    return out


def lists(doc: Document) -> Document:
    """항목 블록에 list_level — 섹션 안에서 1 부터. 붙임 목록은 따로(_attachments)."""
    doc = doc.model_copy(deep=True)
    for items in _sections(doc):
        rows = _rows(items)
        attach = _attachments(rows)
        found = [(b, lv, "attachment") for b, _, _ in rows if (lv := attach.get(b.id))] + _levels(rows, set(attach))
        for b, level, by in found:
            b.list_level = level
            b.provenance = (b.provenance or []) + [{"rule": "list_level", "by": by}]
    return doc
