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
from .headings import TITLE_TYPES, doc_title_key, marker, numbering, place

INDENT_TOL = 0.008        # 같은 수준으로 보는 x 차이(논리 쪽 너비 대비, A4 에서 약 5pt) — 블록 상자 흔들림
MAX_INDENT = 0.3          # 섹션 왼쪽 끝에서 이보다 오른쪽(가운데 맞춤 글·오른쪽 날짜·다른 단)은 항목으로 보지 않는다
MAX_SHIFT = 0.06          # 쪽 사이 가로 밀림으로 보는 최대 x 차이(논리 쪽 너비 대비, A4 약 36pt) — 스캔 밀림·맞쪽(제본) 여백
NOTE = "※"                # 주석 기호 — 자리가 일정하지 않아(□ 자리·○ 자리) 쪽 밀림 근거로 쓰지 않는다
_START = object()
_ATTACH = re.compile(r"\s*[\[<〈【(「『]?\s*(?:붙\s*임|첨\s*부|별\s*첨)\s*[\]>〉】)」』]?\s*[:：]?\s*")   # 번호 없는 붙임 꼬리표


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


def _levels(items: list[tuple[Page, Block]]) -> list[tuple[Block, int, str]]:
    rows = [(page.page_no, b, local(page, b.bbox)[0] / page_size(page)[0], marker(b.text)) for page, b in items]
    shift = _shifts(rows)
    rows = [(b, x - shift[no], m) for no, b, x, m in rows]
    base = min(x for _, x, _ in rows)                       # 섹션 왼쪽 끝
    xs = sorted(x for _, x, m in rows if m and x - base <= MAX_INDENT)
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
        g = next((i for i, (lo, hi) in enumerate(spans) if lo - INDENT_TOL <= x <= hi + INDENT_TOL), None)
        if g is None or x - base > MAX_INDENT or (not m and x - base <= INDENT_TOL):
            continue                                      # 묶음 밖 글, 기호 없이 왼쪽 끝인 문단
        if m and (n := numbering(b.text)):
            t = place(kinds[g], n[0])
            del kinds[g][t:]
            kinds[g].append(n[0])
            cur[g], depth[g] = t, max(depth[g], t + 1)
        found.append((b, g, cur[g]))                       # 기호·기호 없는 항목은 앞 번호 항목의 하위 수준을 받는다
    offset = [sum(depth[:i]) for i in range(len(spans))]
    return [(b, offset[g] + sub + 1, "marker" if depth[g] > 1 else "indent") for b, g, sub in found]


def _attachments(items: list[tuple[Page, Block]]) -> list[Block]:
    """붙임 목록('붙임 1. 계획 1부. 2. …'·'<붙임> 1. …') — 꼬리표 뒤 번호 항목과 이어진 같은 체계의 번호 항목들. 들여쓰기는
    꼬리표 폭 때문이라 앞 항목의 하위가 아니다(실측 test2.pdf — 앞쪽 항목보다 오른쪽이라 수준 2 가 됐다). 꼬리표만 있는
    블록은 headings 가 붙임 구역을 새로 열어 그 아래 항목이 구역 안에서 1 이 된다."""
    out, kind = [], None
    for _, b in items:
        text = b.text or ""
        n = numbering(text)
        if kind is not None and n and n[0] == kind:
            out.append(b)
            continue
        kind = None
        if (m := _ATTACH.match(text)) and (n := numbering(text[m.end():])):
            out.append(b)
            kind = n[0]
    return out


def lists(doc: Document) -> Document:
    """항목 블록에 list_level — 섹션 안에서 1 부터. 붙임 목록은 따로 수준 1."""
    doc = doc.model_copy(deep=True)
    for items in _sections(doc):
        attach = _attachments(items)
        for b in attach:
            b.list_level = 1
            b.provenance = (b.provenance or []) + [{"rule": "list_level", "by": "attachment"}]
        skip = {b.id for b in attach}
        rest = [(p, b) for p, b in items if b.id not in skip]
        for b, level, by in (_levels(rest) if rest else []):
            b.list_level = level
            b.provenance = (b.provenance or []) + [{"rule": "list_level", "by": by}]
    return doc
