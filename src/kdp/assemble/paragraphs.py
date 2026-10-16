"""쪽 넘김 문단 이음(기획서 4.3 규칙 2) — 뒤 블록에 continues_from 만 단다. 글자는 합치지 않는다(렌더러가 잇는다).

잘못 잇지 않는 쪽을 우선한다 — 개조식 항목은 명사형으로 끝나 종결 표현이 없는 경우가 많다(부록 C). 그래서 항목 기호로
시작하는 블록은 앞뒤 어느 쪽이든 잇지 않는다. 표·그림 부속 줄(제목·출처·주·'이하 빈칸')도 명사로 끝나지만 본문이 아니다.
같은 쪽 안 이음(그림 감싼 문단)은 figures.join_wrapped 의 몫이다.
"""
from __future__ import annotations

import re

from ..schema import Block, Document, Page
from .order import has_output
from .tables import FOOTNOTES, appendage, attached

_END = re.compile(r"(?:[.!?。…:;」』”’)\]]|[다요까오함음임됨것등중]|예정)\s*$")         # 종결 표현 — 문장 끝과 개조식 명사형 끝
_START = re.compile(r"\s*(?:[□■◇◆○●◦•▪▫ㅇ❍\-–·∙※▶►▷➢‣*✓√]"                        # 보고서체 항목 기호
                    r"|\(?\d{1,3}[.)]|\(?[가나다라마바사아자차카타파하][.)]|[①-⑳➀-➓❶-❿㉠-㉭㉮-㉻]"   # 1. 1) (1) 가. (가) ① ㉠ ㉮
                    r"|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]|제\s*\d+\s*[장절관조항편]|붙임|별첨|별표|별지|부록"     # 제목 번호
                    r"|[\[(<〈《【「『（])")                                                # 괄호 — 표 제목·단위·'(계속)'·참고


def _first(blocks) -> Block | None:
    """출력에 나가는 첫 블록 — keep=false(머리말·꼬리말·쪽번호 등)·빈 블록은 건너뛴다."""
    return next((b for b in blocks if b.keep and has_output(b)), None)


def _last(page: Page) -> Block | None:
    """쪽 끝 출력 블록 — 각주는 건너뛰고 그 앞 본문을 본다. 병합 구성원(그림 감싼 문단의 아래 글 등)을 만나면 그 대표다 —
    대표 글이 구성원 글로 끝난다. 그 문단이 감싼 그림은 읽기 순서상 뒤에 와도 쪽 끝으로 보지 않는다."""
    by_id = {b.id: b for b in page.blocks}
    wrapped = {p.get("figure") for b in page.blocks for p in b.provenance or [] if p.get("rule") == "join_wrapped"}
    for b in reversed(page.blocks):
        b = by_id.get(b.merged_into, b)
        if b.keep and has_output(b) and b.type not in FOOTNOTES and b.id not in wrapped:
            return b
    return None


def titled(b: Block) -> bool:
    """검출기가 제목으로 본 줄 — 문장 꼴이라 classify_blocks 가 text 로 바꾼 제목 라벨. 새 항목·문단의 시작이라 앞 글에
    잇지 않는다(이 꼴은 늘 끝맺음으로 끝나 앞 블록으로도 잇지 않는다)."""
    return any(p.get("rule") == "title_sentence" for p in b.provenance or [])


def _body(b: Block | None, refs: set[str]) -> bool:
    """이을 수 있는 본문 글 — 제목·항목·표 그림 부속 줄이 아닌 text 블록(headings·lists 가 단 수준이 있거나 표에 이은
    제목·출처(refs)거나 검출기가 제목으로 본 줄이면 아니다)."""
    return (b is not None and b.type == "text" and bool((b.text or "").strip()) and b.heading_level is None
            and b.list_level is None and not _START.match(b.text) and b.id not in refs and not appendage(b.text)
            and not titled(b))


def join_paragraphs(doc: Document) -> Document:
    """앞 논리 페이지 끝 본문이 종결 표현 없이 끝나고 다음 논리 페이지 첫 블록이 본문이면 뒤 블록에 continues_from."""
    doc = doc.model_copy(deep=True)
    refs = attached(b for p in doc.pages for b in p.blocks)
    for pa, pb in zip(doc.pages, doc.pages[1:]):
        if not (pa.keep and pb.keep):
            continue
        a, b = _last(pa), _first(pb.blocks)
        if not (_body(a, refs) and _body(b, refs)) or _END.search(a.text) or _START.match(a.text.strip().splitlines()[-1]):
            continue                                                  # 마지막 줄이 항목이면 개조식 끝이다
        b.continues_from = a.id
        b.provenance = (b.provenance or []) + [{"rule": "join_paragraphs"}]
    return doc
