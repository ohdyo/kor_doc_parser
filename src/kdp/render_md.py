"""정본 → 마크다운(기획서 4.2 원칙 1·2, 4.4, 설계 6장) — keep 블록을 읽기 순서로 이은 파생물.

설치본 기본 포매터는 '2-4.' 를 '2 -4.' 로 바꾸고 제목 안 '.' 개수로 수준을 정하므로 정본에서 직접 만든다(4.3 규칙 10).
- 앵커: 블록마다 제 줄에 <a id="p1.b04"></a>, 블록 글은 빈 줄 없이 바로 붙인다. HTML 을 받는 도구(GitHub·GitLab·VS Code 등)
  에서 보이지 않는 링크 대상이 된다 — {#id} 속성은 pandoc·kramdown 전용이라 다른 도구에서 글자로 보이고, 주석은 링크 대상이
  못 되며, <a id=…/> 는 HTML 에서 닫히지 않아 뒤 글을 링크로 감싼다. <a> 는 HTML 블록 태그가 아니라서 앵커와 글이 한 문단이
  되고 글의 첫 줄도 문단 이어짐 줄로 읽힌다 — 이스케이프가 문단을 끊는 줄 머리로 줄고, 들여쓴 항목이 코드 블록이 되지 않는다.
  앵커만 있는 줄이라 청커가 줄 단위로 뗄 수 있다.
- 쪽 경계: 블록 밖 <!-- page N -->(N = 논리 페이지, 그 쪽이 시작하는 자리) — 출력 블록이 있는 쪽마다 하나, 쪽 순서대로.
  continues_from 으로 이어진 문단·표 사이에는 넣지 않는다 — 뒤 조각을 앞 조각 바로 뒤로 끌어 붙이고(사이에 낀 블록은 이어짐
  뒤로 밀린다) 쪽 주석은 이어짐이 끝난 뒤에 둔다. 다만 섹션 경로가 다른 조각은 끌어오지 않는다(마크다운 자리 = 정본 섹션).
- RAG 적재용 마크다운(document.rag.md, D21): 같은 블록·차례·글이되 마크다운만 받는 도구에 맞춘다 — render_rag_markdown.
"""
from __future__ import annotations

import re

from .assemble.order import kept
from .assemble.tables import pipe_regroup, regroup
from .convert import FIGURE_LABELS
from .schema import Block, Document

TITLES = ("doc_title", "paragraph_title")      # 수준이 없으면 굵은 글 — 틀린 수준을 지어내지 않는다(4.3 규칙 3)
INDENT = "  "                                  # 항목 한 수준(list_level 2 부터)
# 문단 이어짐 줄을 마크다운이 다르게 읽는 줄 머리(CommonMark·GFM): 제목 #, 글머리 - + *, 인용 >, 코드 울타리, HTML 블록,
# 기호만 있는 줄(구분선·setext 밑줄·표 구분 줄) — 첫 기호 앞에 \ 를 넣는다
_ESC = re.compile(r"^( {0,3})(?=#{1,6}(?:[ \t]|$)|[-+*](?:[ \t]|$)|>|```|~~~|<[A-Za-z/!?]|[-=_*+:|~`][-=_*+:|~` \t]*$)")
_ONE = re.compile(r"^( {0,3}0*1)(?=[.)](?:[ \t]|$))")    # 번호 목록은 1 로 시작할 때만 문단을 끊는다 — '2.'·'2026.' 은 둔다
_DOLLAR = re.compile(r"(?<!\\)\$")                         # 수식 구분자 — \$ 는 글자
_TABLE = re.compile(r"\s*<table[\s>]", re.I)              # 표 HTML — 칸을 못 읽으면 PaddleX 가 VLM 원문(평문)을 그대로 둔다
_NUM_START = re.compile(r"^(\d{1,9})(?=[.)](?:[ \t]|$))")  # 문단 첫 줄의 번호 — '2.'·'10)' 도 번호 목록을 연다
_LINK_DEF = re.compile(r"^( {0,3})(?=\[[^\]]*\]:)")         # 새 문단 첫 줄의 '[표 1]: 현황' 은 링크 참조 정의로 읽혀 사라진다
ALT = {"chart": "도표", "table": "표", "seal": "도장"}    # RAG 마크다운의 크롭 링크 이름 — 그 밖의 그림은 '그림'
FIGURE_TEXT = "그림 속 글자: "                             # RAG 마크다운에서 울타리 대신 그림 속 글자 앞에 붙이는 말
RAG_TABLE_CHARS = 400                                      # 이보다 긴 표는 행 묶음으로(설정 output.rag_table_chars)


def _lines(text: str) -> list[str]:
    """빈 줄은 뺀다 — 마크다운의 빈 줄은 블록 경계다(청커가 블록 가운데서 자르지 않게)."""
    return [ln.rstrip() for ln in text.splitlines() if ln.strip()]


def _escape(line: str) -> str:
    return _ONE.sub(r"\1\\", _ESC.sub(r"\1\\", line, count=1), count=1)


def _lead(line: str) -> str:
    """앵커 없는 블록의 첫 줄(D21) — 새 문단을 여는 줄이라 이어짐 줄보다 막을 것이 많다: 들여쓰기(4칸이면 코드), 어느 번호든
    번호 목록, '[…]:' 링크 정의."""
    line = _escape(line.lstrip())
    return _LINK_DEF.sub(r"\1\\", _NUM_START.sub(r"\1\\", line, count=1), count=1)


def _anchor(b: Block) -> str:
    return f'<a id="{b.id}"></a>'


def _text(b: Block, text: str, rag: bool = False) -> str:
    """제목은 '#' × 수준(최대 6), 수준 없는 제목은 굵은 글, 나머지는 문단 — 항목은 기호 그대로 list_level 만큼 들여쓴다.
    rag 면 들여쓰지 않고 첫 줄을 새 문단 첫 줄로 막는다(앵커가 없어 블록 첫 줄이 새 문단을 연다 — D21)."""
    if b.heading_level is not None:                       # 문서 제목(0)이 '#', 섹션 1 이 '##'
        return "#" * min(b.heading_level + 1, 6) + " " + " ".join(text.split())
    if rag and b.type not in TITLES:
        lines = _lines(text)
        return "\n".join([_lead(lines[0]), *map(_escape, lines[1:])]) if lines else ""
    lines = ["**" + " ".join(text.split()) + "**"] if b.type in TITLES else [_escape(ln) for ln in _lines(text)]
    pad = "" if rag else INDENT * max((b.list_level or 1) - 1, 0)
    return "\n".join(pad + ln for ln in lines)


def _parts(b: Block, rag: bool = False) -> list[str]:
    """본문(표 HTML·수식·글), 크롭 이미지 링크, 그림 설명, 그림 속 글자 순. 글 갈래 그림(D19)의 그림 속 글은 내용이라 문단으로,
    그 밖의 그림 속 글은 그림에 딸린 이름표라 줄을 지키는 울타리로 낸다. rag 면 링크 이름을 한국어로, 울타리 대신
    '그림 속 글자:' 줄로, 표 HTML 은 맨 뒤로 낸다(D21 — HTML 블록은 빈 줄까지 뒤 줄을 삼켜 링크·글자 줄이 날 글이 된다)."""
    text = (b.text or "").strip()
    parts = []
    if b.html:                                     # 표는 HTML 그대로(병합 셀) — 표 제목 블록은 제자리에서 따로 나간다
        parts.append(b.html.strip() if _TABLE.match(b.html) else _text(b, b.html, rag))   # 평문은 글처럼(목록·제목으로 읽히지 않게)
    elif b.type == "display_formula" and text:
        body = "\n".join(_lines(_DOLLAR.sub(" ", text).strip()))   # 붙어 온 $$(PaddleX 가 \[ \] 를 바꾼 것)는 떼고 다시 감싼다
        parts += [f"$$\n{body}\n$$"] if body else []
    elif text:
        parts.append(_text(b, text, rag))
    if b.img_path and (b.type in FIGURE_LABELS or b.route == "image" or not parts):
        alt = ALT.get(b.type, "그림") if rag else b.type
        parts.append(f"![{alt}]({b.img_path})")    # 그림·차트, 이미지 갈래 표(D18), 글이 없는 블록
    if b.figure_desc:
        parts.append(_text(b, "그림 설명: " + b.figure_desc, rag))
    if b.figure_text and (lines := _lines(b.figure_text)):
        if b.route == "text" and b.type in FIGURE_LABELS:           # 글 갈래 그림(D19) — 그림 속 글이 곧 내용
            body = [_escape(ln) for ln in lines]
            body[0] = _lead(lines[0]) if rag else _NUM_START.sub(r"\1\\", _escape(lines[0].lstrip()), count=1)   # 새 문단 첫 줄 — 어느 번호든 목록·들여쓰면 코드가 된다
            parts.append("\n".join(body))
        elif rag:
            parts.append("\n".join([FIGURE_TEXT + lines[0].strip(), *map(_escape, lines[1:])]))
        else:
            fence = "`" * max(3, max(map(len, re.findall(r"`+", b.figure_text)), default=0) + 1)
            parts.append(f"{fence}text\n" + "\n".join(lines) + f"\n{fence}")   # 그림 속 글자 — 줄을 지키고 이스케이프 없이
    if rag and b.html and _TABLE.match(b.html):
        parts = parts[1:] + parts[:1]
    return parts


def _block(b: Block) -> str:
    body = "\n\n".join(_parts(b))
    return _anchor(b) + ("\n" + body if body else "")


def _is_para(b: Block) -> bool:
    """이어 붙일 글 블록 — 글만 있고 표·수식·그림이 아니다."""
    return bool((b.text or "").strip()) and not (b.html or b.figure_text) \
        and b.type not in FIGURE_LABELS and b.type != "display_formula"


def _unit(blocks: list[Block]) -> str:
    """continues_from 묶음 — 문단은 글을 이어 한 문단으로(앵커는 모두 앞에), 표 등은 쪽 주석 없이 잇따라 낸다."""
    if len(blocks) > 1 and all(map(_is_para, blocks)):
        return "\n".join([*map(_anchor, blocks), _text(blocks[0], " ".join(b.text.strip() for b in blocks))])
    return "\n\n".join(map(_block, blocks))


def _walk(doc: Document):
    """마크다운에 나가는 차례 — (쪽 번호, None) 쪽 경계 또는 (None, continues_from 으로 이은 블록 묶음)."""
    items = list(kept(doc))
    pos = {b.id: i for i, (_, b) in enumerate(items)}
    nxt: dict[int, int] = {}                       # 앞 조각 → 그것을 잇는 뒤 조각
    for i, (_, b) in enumerate(items):
        j = pos.get(b.continues_from)
        if j is not None and j < i and j not in nxt and b.section_path == items[j][1].section_path:
            nxt[j] = i                             # 섹션이 바뀐 조각(사이에 '□ …(계속)' 같은 새 제목)은 제자리에 — 정본 경로와 같게
    pending, done, last = [], set(), None
    for i, (page, _) in enumerate(items):
        if page is not last:                       # 쪽 주석은 그 쪽에서 새로 시작하는 첫 묶음 앞에 — 이어진 조각만 있던 쪽은 다음 묶음 앞에
            pending.append(page.page_no)
            last = page
        if i in done:
            continue
        chain = [i]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        done.update(chain)
        yield from ((n, None) for n in pending)
        pending = []
        yield None, [items[k][1] for k in chain]
    yield from ((n, None) for n in pending)


def render_markdown(doc: Document) -> str:
    """keep 블록을 읽기 순서로. 쪽 경계는 문단 밖 주석, 블록 앵커."""
    out = [f"<!-- page {n} -->" if blocks is None else _unit(blocks) for n, blocks in _walk(doc)]
    return "\n\n".join(out) + "\n" if out else ""


def render_rag_markdown(doc: Document, max_table_chars: int = RAG_TABLE_CHARS, table_format: str = "pipe") -> str:
    """RAG 적재용 마크다운(D21) — document.md 와 같은 블록·차례·글이되 마크다운만 받는 도구(길이로 자르는 청커)에 맞춘다.
    블록 앵커는 빼고 쪽 주석만 둔다. 표·그림 제목(caption_id)은 제자리에서 빼 표·그림 바로 앞에 붙이고, 블록 하나를 빈 줄 없는
    한 덩이로 낸다(제목·그림 링크·설명·그림 속 글자가 갈라지지 않게). 쪽 넘김 표는 한 표로 잇고, max_table_chars 보다 긴 표는
    행 묶음으로 나눠 묶음마다 머리행과 '제목 (계속)' 줄을 되풀이한다(4.1 — 표 뒷부분이 머리행 없는 청크가 되지 않게).
    table_format 'pipe'(기본 — 사용자 결정 2026-10-26)면 칸을 읽은 표를 HTML 대신 파이프 표로(병합 칸을 풀고 머리행을 한 줄로 —
    pipe_regroup), 'html' 이면 표 HTML 그대로."""
    by_id = {b.id: b for _, b in kept(doc)}
    caps = _captions(by_id)
    moved = {c.id for c in caps.values()}
    out = []
    for n, blocks in _walk(doc):
        if blocks is None:
            out.append(f"<!-- page {n} -->")
        elif body := _rag_unit([b for b in blocks if b.id not in moved], by_id, caps, max_table_chars, table_format):
            out.append(body)
    return "\n\n".join(out) + "\n" if out else ""


def _captions(by_id: dict[str, Block]) -> dict[str, Block]:
    """표·그림 → 앞에 붙일 제목 블록. 제목 수준이 있는 블록(섹션 제목 노릇)은 제자리에 두고, 쪽 넘김 표의 뒤 조각에 따로 이은
    제목도 제자리에 둔다(앞 조각 제목이 이은 표 전체의 제목이다). 나란한 표 둘이 받은 한 제목은 둘 다 앞에 붙인다."""
    out = {}
    for b in by_id.values():
        c = by_id.get(b.caption_id) if b.caption_id else None
        if c is not None and c is not b and c.heading_level is None and (c.text or "").strip() \
                and (b.type == "table" or b.type in FIGURE_LABELS) and b.continues_from not in by_id:
            out[b.id] = c
    return out


def _splittable(b: Block) -> bool:
    """행 묶음으로 다시 쓸 표 — 칸을 읽은 HTML 표이고 이미지 갈래가 아니다(칸 안 사진·인식 실패 표는 크롭과 함께 그대로)."""
    return b.type == "table" and bool(b.html) and bool(_TABLE.match(b.html)) and b.route != "image"


def _rag_block(b: Block, cap: Block | None = None, line: str = "") -> str:
    """블록 하나를 빈 줄 없는 한 덩이로 — 붙일 제목(쪽 넘김 표 뒤 조각이면 '제목 (계속)' 줄), 본문, 크롭 링크, 그림 설명, 그림 속
    글자 순(이미지 갈래 표는 표 HTML 을 맨 뒤에)."""
    head = [_text(cap, cap.text, rag=True)] if cap else [_lead(line)] if line else []
    return "\n".join(head + _parts(b, rag=True))


def _first_piece(b: Block, by_id: dict[str, Block]) -> Block:
    """쪽 넘김 표 사슬의 첫 조각 — continues_from 을 끝까지 따라간다(이미지 갈래 조각도 건넌다)."""
    seen = {b.id}
    while b.continues_from in by_id and by_id[b.continues_from].type == "table" and b.continues_from not in seen:
        b = by_id[b.continues_from]
        seen.add(b.id)
    return b


def _continued(first: Block, cap: Block | None) -> str:
    """쪽 넘김 표 뒤 조각·긴 표 뒤 묶음 앞에 둘 줄 — '제목 (계속)'과 단위."""
    return " ".join(x for x in (f"{' '.join(cap.text.split())} (계속)" if cap else "",
                                f"(단위: {first.unit})" if first.unit else "") if x)


def _rag_unit(blocks: list[Block], by_id: dict[str, Block], caps: dict[str, Block], max_chars: int, fmt: str) -> str:
    if not blocks:
        return ""
    if len(blocks) > 1 and all(map(_is_para, blocks)):
        return _text(blocks[0], " ".join(b.text.strip() for b in blocks), rag=True)
    if all(map(_splittable, blocks)):
        return _rag_table(blocks, by_id, caps, max_chars, fmt)
    out = []                                        # 섞인 사슬(이미지 갈래 표 조각이 낀 쪽 넘김 표 등) — 블록마다
    for b in blocks:
        if _splittable(b):
            out.append(_rag_table([b], by_id, caps, max_chars, fmt))
        elif b.type == "table" and (first := _first_piece(b, by_id)) is not b:
            out.append(_rag_block(b, line=_continued(first, caps.get(first.id))))
        else:
            out.append(_rag_block(b, caps.get(b.id)))
    return "\n\n".join(out)


def _rag_table(tables: list[Block], by_id: dict[str, Block], caps: dict[str, Block], max_chars: int, fmt: str) -> str:
    """표(쪽 넘김 사슬이면 이은 한 표)를 행 묶음으로 — 첫 묶음 앞에 제목, 뒤 묶음 앞에 '제목 (계속)'(단위가 있으면 함께).
    사슬 첫 조각이 이 묶음 밖에 있으면(섹션이 달라 끌어오지 않았거나 이미지 갈래 조각이 사이에 있다) 첫 묶음부터 '제목 (계속)'.
    머리행은 사슬 첫 조각에서 빌리고(그 조각이 이미지 갈래면 이 묶음 첫 조각에서), 수는 조각들 header_rows 의 가장 큰 값이다 —
    뒤 조각의 header_rows 는 merge_tables 가 첫 조각 첫 행들을 되풀이한 행 수로 단 것이라, 첫 조각 머리행을 못 찾았어도(연도
    머리글 등) 칸 이름 행이다."""
    first = _first_piece(tables[0], by_id)
    lone = first is not tables[0]
    cap = caps.get(first.id)
    src = first if _splittable(first) else tables[0]
    n_head = max([src.header_rows or 0] + [t.header_rows or 0 for t in tables if t is not src])
    pieces = [(t.html, n_head if t is src else t.header_rows or 0) for t in tables]
    if fmt == "pipe":
        groups = pipe_regroup((src.html, n_head), pieces, max_chars)
    elif not lone and len(tables) == 1 and len(src.html.strip()) <= max_chars:
        groups = [src.html.strip()]
    else:
        groups = regroup((src.html, n_head), pieces, max_chars)
    more = _continued(first, cap)
    out = []
    for k, g in enumerate(groups):
        if k == 0 and not lone:
            line = _text(cap, cap.text, rag=True) if cap else ""
        else:
            line = _lead(more) if more else ""
        out.append(f"{line}\n{g}" if line else g)
    return "\n\n".join(out)
