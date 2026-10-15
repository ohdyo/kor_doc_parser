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
"""
from __future__ import annotations

import re

from .assemble.order import kept
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


def _lines(text: str) -> list[str]:
    """빈 줄은 뺀다 — 마크다운의 빈 줄은 블록 경계다(청커가 블록 가운데서 자르지 않게)."""
    return [ln.rstrip() for ln in text.splitlines() if ln.strip()]


def _escape(line: str) -> str:
    return _ONE.sub(r"\1\\", _ESC.sub(r"\1\\", line, count=1), count=1)


def _anchor(b: Block) -> str:
    return f'<a id="{b.id}"></a>'


def _text(b: Block, text: str) -> str:
    """제목은 '#' × 수준(최대 6), 수준 없는 제목은 굵은 글, 나머지는 문단 — 항목은 기호 그대로 list_level 만큼 들여쓴다."""
    if b.heading_level is not None:                       # 문서 제목(0)이 '#', 섹션 1 이 '##'
        return "#" * min(b.heading_level + 1, 6) + " " + " ".join(text.split())
    lines = ["**" + " ".join(text.split()) + "**"] if b.type in TITLES else [_escape(ln) for ln in _lines(text)]
    pad = INDENT * max((b.list_level or 1) - 1, 0)
    return "\n".join(pad + ln for ln in lines)


def _parts(b: Block) -> list[str]:
    """본문(표 HTML·수식·글), 크롭 이미지 링크, 그림 속 글자 순."""
    text = (b.text or "").strip()
    parts = []
    if b.html:                                     # 표는 HTML 그대로(병합 셀) — 표 제목 블록은 제자리에서 따로 나간다
        parts.append(b.html.strip() if _TABLE.match(b.html) else _text(b, b.html))   # 평문은 글처럼(목록·제목으로 읽히지 않게)
    elif b.type == "display_formula" and text:
        body = "\n".join(_lines(_DOLLAR.sub(" ", text).strip()))   # 붙어 온 $$(PaddleX 가 \[ \] 를 바꾼 것)는 떼고 다시 감싼다
        parts += [f"$$\n{body}\n$$"] if body else []
    elif text:
        parts.append(_text(b, text))
    if b.img_path and (b.type in FIGURE_LABELS or b.route == "image" or not parts):
        parts.append(f"![{b.type}]({b.img_path})")  # 그림·차트, 이미지 갈래 표(D18), 글이 없는 블록
    if b.figure_text and (lines := _lines(b.figure_text)):
        fence = "`" * max(3, max(map(len, re.findall(r"`+", b.figure_text)), default=0) + 1)
        parts.append(f"{fence}text\n" + "\n".join(lines) + f"\n{fence}")   # 그림 속 글자 — 줄을 지키고 이스케이프 없이
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


def render_markdown(doc: Document) -> str:
    """keep 블록을 읽기 순서로. 쪽 경계는 문단 밖 주석, 블록 앵커."""
    items = list(kept(doc))
    pos = {b.id: i for i, (_, b) in enumerate(items)}
    nxt: dict[int, int] = {}                       # 앞 조각 → 그것을 잇는 뒤 조각
    for i, (_, b) in enumerate(items):
        j = pos.get(b.continues_from)
        if j is not None and j < i and j not in nxt and b.section_path == items[j][1].section_path:
            nxt[j] = i                             # 섹션이 바뀐 조각(사이에 '□ …(계속)' 같은 새 제목)은 제자리에 — 정본 경로와 같게
    out, pending, done, last = [], [], set(), None
    for i, (page, _) in enumerate(items):
        if page is not last:                       # 쪽 주석은 그 쪽에서 새로 시작하는 첫 묶음 앞에 — 이어진 조각만 있던 쪽은 다음 묶음 앞에
            pending.append(f"<!-- page {page.page_no} -->")
            last = page
        if i in done:
            continue
        chain = [i]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        done.update(chain)
        out += pending + [_unit([items[k][1] for k in chain])]
        pending = []
    return "\n\n".join(out + pending) + "\n" if out else ""
