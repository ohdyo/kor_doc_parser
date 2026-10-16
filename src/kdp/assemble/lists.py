"""항목 계층(기획서 4.3 규칙 5) — 공문 8단계(1. → 가. → 1) → 가) → (1) → (가) → ① → ㉮)와 보고서체(□ → ○ → - → ·).

VLM 이 읽은 글머리 기호는 믿을 수 없다(○ 가 ☐ 로, PPT 기호는 사라짐 — 실측). 그래서 섹션(가장 가까운 제목) 안에서 번호·기호로
시작하는 항목의 들여쓰기 x(논리 쪽 너비 대비, 쪽마다 가로 밀림을 맞춘 뒤)를 묶어 왼쪽부터 1, 2, … 로 매기고 기호 모양은 수준에
쓰지 않는다. 번호는 보조로만 —
들여쓰지 않아 한 x 에 번호 체계가 섞이면(1. 과 가.) 번호 차례로 나눈다. 기호 없는 블록은 섹션 왼쪽 끝이면 문단, 번호·기호 있는
항목과 같은 x 면 기호를 잃은 항목으로 그 수준을 받고, 그 밖(첫 줄 들여쓰기 문단·항목의 이어진 글)은 비운다. 기호는 원문대로 둔다.
"""
from __future__ import annotations

import re
from itertools import groupby

from ..geometry import local, page_size
from ..schema import Block, Document, Page
from .headings import TITLE_TYPES, _ordinal, doc_title_key, marker, numbering, place

INDENT_TOL = 0.008        # 같은 수준으로 보는 x 차이(논리 쪽 너비 대비, A4 에서 약 5pt) — 블록 상자 흔들림
MAX_INDENT = 0.3          # 섹션 왼쪽 끝에서 이보다 오른쪽(가운데 맞춤 글·오른쪽 날짜·다른 단)은 항목으로 보지 않는다
MAX_SHIFT = 0.06          # 쪽 사이 가로 밀림으로 보는 최대 x 차이(논리 쪽 너비 대비, A4 약 36pt) — 스캔 밀림·맞쪽(제본) 여백
NOTE = "※"                # 주석 기호 — 자리가 일정하지 않아(□ 자리·○ 자리) 쪽 밀림 근거로 쓰지 않는다
_START = object()
_ATTACH = re.compile(r"\s*[\[<〈【(「『]?\s*(?:붙\s*임|첨\s*부|별\s*첨)\s*[\]>〉】)」』]?\s*[:：]?\s*")   # 번호 없는 붙임 꼬리표
_END = re.compile(r"끝\s*\.\s*$")      # 붙임 목록 끝(공문 '… 1부.  끝.')


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
    kinds = [[] for _ in spans]                           # 묶음마다 번호 체계 스택(한 x 에 번호 체계가 섞인 문서)
    cur, depth, found = [0] * len(spans), [1] * len(spans), []
    for b, x, m in rows:
        if b.id in skip:
            continue
        g = next((i for i, (lo, hi) in enumerate(spans) if lo - INDENT_TOL <= x <= hi + INDENT_TOL), None)
        if g is None or x - base > MAX_INDENT or (not m and x - base <= INDENT_TOL):
            continue                                      # 묶음 밖 글, 기호 없이 왼쪽 끝인 문단
        if m and (k := _kind(numbering(b.text))):
            t = place(kinds[g], k)
            del kinds[g][t:]
            kinds[g].append(k)
            cur[g], depth[g] = t, max(depth[g], t + 1)
        found.append((b, g, cur[g]))                       # 기호·기호 없는 항목은 앞 번호 항목의 하위 수준을 받는다
    offset = [sum(depth[:i]) for i in range(len(spans))]
    return [(b, offset[g] + sub + 1, "marker" if depth[g] > 1 else "indent") for b, g, sub in found]


def _attachments(rows: list[tuple[Block, float, tuple | None]]) -> dict[str, int]:
    """붙임 목록('붙임 1. 계획 1부. 2. …'·'<붙임> 1. …') → {블록 id: 수준}. 섹션 왼쪽 끝의 '꼬리표 + 번호' 줄과, 그 오른쪽(꼬리표
    폭만큼 들여 쓴) 한 x 에서 다음 번호로 이어지는 같은 체계 항목은 수준 1 — 앞 항목의 하위가 아니다(실측 test2.pdf). 그보다
    깊은 주·하위 항목은 2 로 두고 건너며, '끝.' 줄 다음이나 다른 블록에서 끝낸다. 꼬리표만 있는 블록은 headings 가 붙임
    구역을 새로 연다."""
    base = min(x for _, x, _ in rows)
    out, i = {}, 0
    while i < len(rows):
        b, x, _ = rows[i]
        i += 1
        m = _ATTACH.match(b.text or "")
        n = numbering((b.text or "")[m.end():]) if m else None
        prev = _ordinal(n[1]) if n else ()
        if not prev or x - base > INDENT_TOL:
            continue
        out[b.id], kind, ex = 1, _kind(n), None
        while i < len(rows) and not _END.search(rows[i - 1][0].text or ""):
            c, cx, cm = rows[i]
            cn = numbering(c.text)
            if cn and _kind(cn) == kind and _ordinal(cn[1]) == prev[:-1] + (prev[-1] + 1,) and cx > x + INDENT_TOL \
                    and (ex is None or abs(cx - ex) <= INDENT_TOL):
                out[c.id], prev, ex = 1, _ordinal(cn[1]), cx if ex is None else ex
            elif ex is not None and cm and cx > ex + INDENT_TOL:
                out[c.id] = 2
            else:
                break
            i += 1
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
