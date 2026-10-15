"""쪽 판정(기획서 5.1, 설계 5.2) — 학습 전 규칙. 지금은 표지·목차, 나머지 쪽 종류는 M3.

점수가 classify.page_drop_min_score 이상이면 쪽을 버리고(keep=false), 0.5 이상이면 page_class_guess 만 남긴다(검수 대상).
학습한 쪽 분류기(기획서 7.2 (B))가 생기면 이 함수 자리를 바꾼다.
"""
from __future__ import annotations

import re

from ..config import Classify
from ..convert import FIGURE_LABELS
from ..qc.checks import visible_chars
from ..schema import Block, Document, Page

JUNK_LABELS = {"header", "header_image", "footer", "footer_image", "number", "seal", "aside_text"}
COVER_MAX_BLOCK_CHARS = 100
COVER_MAX_CHARS = 250
TOC_MAX_OTHER_CHARS = 100          # 목차 줄이 아닌 본문이 이보다 많으면 쪽은 남기고 목차 블록만 버린다

_STRIP = re.compile(r"[\s\[\]【】()<>〈〉《》『』「」:：]")
_TOC_TITLE = re.compile(r"(목차|차례|contents|tableofcontents|(표|그림|도표|사진|부록)(목차|차례))", re.I)
_TOC_LINE = re.compile(r"((\.{3,}|…{2,}|·{3,}|‥{2,}|_{3,})\s*\S{0,6}|\S\s+\d{1,4})\s*$")    # 점선 + 쪽, 또는 끝의 쪽번호
_DATE = re.compile(r"(?<![\d\-./])((?:19|20)\d{2})\s*(?:년|[.\-/])\s*(\d{1,2})\s*(?:월|[.\-/]|(?=[\s(),\]}~]|$))"
                   r"(?:\s*(\d{1,2})\s*(?:일|\.|(?=[\s(),\]}~]|$)))?(?!\d)")
DOC_NO = re.compile(r"(?:(?:문서|관리)\s*번\s*호\s*[:：]?\s*(\S+)|(\S*\d{2,4}\s*-\s*\S*?\d+\s*호))\s*$")
_SENTENCE = re.compile(r"(?:니다|십시오|[이하한된인였었했있없겠]다)\s*[.。!?]?\s*$|니다")   # 문장 끝맺음 — '개요'·'주요' 같은 명사 끝은 아니다
_TERSE = re.compile(r"[함음임됨]\s*\.?\s*$")                                            # 개조식 끝맺음(긴 줄만 본다)
_ITEM = re.compile(r"^(?:[가나다라마바사아자차카타파하][.)]|\d{1,2}[.)]|\(\d+\)|[①-⑳□■○●◦▪ㅇ]|[-•▶►◆◇➢*·]\s"
                   r"|(?:일\s*시|장\s*소|참석자?|안\s*건|대\s*상|목\s*적|방\s*법|내\s*용|주\s*제|회의명)\s*[:：])")   # 행사·회의 항목 줄


def content_blocks(page: Page) -> list[Block]:
    """판정에 쓰는 블록 — 잡요소 라벨과 병합 구성원은 뺀다."""
    return [b for b in page.blocks if b.merged_into is None and b.type not in JUNK_LABELS]


def is_toc_title(b: Block) -> bool:
    return b.type != "content" and bool(_TOC_TITLE.fullmatch(_STRIP.sub("", b.text or "")))


def _line_h(b: Block) -> float:
    return (b.bbox[3] - b.bbox[1]) / max(1, (b.text or "").count("\n") + 1)


def _date_match(text: str):
    """숫자 경계와 월·일 범위를 본 첫 날짜(문서번호·ISBN 의 숫자를 날짜로 잡지 않게)."""
    for m in _DATE.finditer(text or ""):
        mo, d = int(m.group(2)), m.group(3)
        if 1 <= mo <= 12 and (d is None or 1 <= int(d) <= 31):
            return m
    return None


def _ymd(m) -> tuple[int, int, int | None]:
    return int(m.group(1)), int(m.group(2)), int(m.group(3)) if m.group(3) else None


def find_date(text: str) -> tuple[int, int, int | None] | None:
    """(년, 월, 일)."""
    m = _date_match(text)
    return _ymd(m) if m else None


def doc_date(text: str) -> tuple[int, int, int | None] | None:
    """날짜만 적힌 줄의 날짜 — '일시: 2026. 3. 15.(금) 14:00' 처럼 날짜가 다른 글에 든 줄은 보지 않는다."""
    for ln in (text or "").splitlines():
        m = _date_match(ln)
        if m and len(_STRIP.sub("", ln)) <= len(_STRIP.sub("", m.group(0))) + 4:
            return _ymd(m)
    return None


def date_blocks(blocks: list[Block]) -> list[Block]:
    """날짜 줄이 있는 블록 — 문서번호 줄과 기간(~) 줄은 빼고 짧은 블록부터."""
    return sorted((b for b in blocks if b.text and "~" not in b.text and not DOC_NO.search(b.text.strip())
                   and doc_date(b.text)), key=lambda b: len(b.text))


def title_blocks(page: Page) -> list[Block]:
    """표지 제목 — doc_title, 없으면 다른 글보다 1.4배 이상 크게 쓴 paragraph_title."""
    texts = [b for b in content_blocks(page) if b.type not in FIGURE_LABELS and b.text]
    docs = [b for b in texts if b.type == "doc_title"]
    if docs or len(texts) < 2:
        return docs
    med = sorted(_line_h(b) for b in texts)[len(texts) // 2]
    return [b for b in texts if b.type == "paragraph_title" and _line_h(b) >= 1.4 * med]


def _body_lines(text: str) -> tuple[int, bool]:
    """본문처럼 보이는 줄 수와 문장 끝맺음 여부 — '※' 주와 '- 최종보고서 -' 같은 장식 줄은 뺀다."""
    n, sentence = 0, False
    for ln in (text or "").splitlines():
        s = ln.strip()
        if not s or s.startswith("※") or (len(s) > 2 and s[0] in "-–·" and s[-1] in "-–·"):
            continue
        hit = bool(_SENTENCE.search(s))
        sentence |= hit
        n += hit or bool(_ITEM.match(s)) or (len(s) >= 8 and bool(_TERSE.search(s)))
    return n, sentence


def _cover_score(page: Page, idx: int, n_pages: int, first_blank: bool) -> float:
    if n_pages < 2 or idx > 1 or (idx == 1 and not first_blank):
        return 0.0
    blocks = content_blocks(page)
    texts = [b for b in blocks if b.type not in FIGURE_LABELS]
    chars = [visible_chars(b) for b in texts]
    if (not texts or max(chars) > COVER_MAX_BLOCK_CHARS or sum(chars) > COVER_MAX_CHARS
            or any(b.type in ("table", "content") for b in texts)):
        return 0.0
    titles = title_blocks(page)
    dates = date_blocks(texts)
    rest = [b for b in texts if b not in titles and b not in dates[:1] and not DOC_NO.search((b.text or "").strip())]
    found = [_body_lines(b.text) for b in rest]
    lines, sentence = sum(n for n, _ in found), any(s for _, s in found)
    if sentence or lines >= 2:
        return 0.0                                        # 본문 문장·항목이 있으면 표지가 아니다(설계 5.2)
    title, date = bool(titles), bool(dates)
    w, h = page.size
    sparse = sum((b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]) for b in blocks) < 0.35 * w * h
    score = min(1.0, 0.5 + 0.2 * title + 0.15 * date + 0.1 * sparse + 0.05 * (len(texts) <= 6))
    return min(score, 0.6) if lines else score            # 항목 같은 줄이 하나뿐이면 버리지 않고 검수로


def _chars(s: str) -> int:
    return sum(not c.isspace() for c in s)


def _toc_score(page: Page) -> float:
    blocks = [b for b in content_blocks(page) if b.type not in FIGURE_LABELS]
    title = any(is_toc_title(b) for b in blocks)
    toc_chars = sum(visible_chars(b) for b in blocks if b.type == "content")
    lines = [ln for b in blocks if b.type != "table" for ln in (b.text or "").splitlines() if ln.strip()]
    toc_lines = sum(bool(_TOC_LINE.search(ln)) for ln in lines)
    other = sum(visible_chars(b) for b in blocks if b.type == "table") + sum(      # 목차 줄 모양이 아닌 본문
        _chars(ln) for b in blocks if b.type not in ("content", "table") and not is_toc_title(b)
        for ln in (b.text or "").splitlines() if not _TOC_LINE.search(ln))
    signals = (title + (toc_chars > 0 and toc_chars >= other)
               + (toc_lines >= 2 and toc_lines >= 0.5 * len(lines)))
    if other > TOC_MAX_OTHER_CHARS:
        return 0.5 if signals else 0.0
    return (0.0, 0.7, 0.95, 1.0)[signals]


def classify_pages(doc: Document, cfg: Classify) -> Document:
    doc = doc.model_copy(deep=True)
    pages = doc.pages
    first_blank = bool(pages) and not any(visible_chars(b) for b in content_blocks(pages[0]))
    for i, page in enumerate(pages):
        scores = {"cover": _cover_score(page, i, len(pages), first_blank), "toc": _toc_score(page)}
        cls, score = max(scores.items(), key=lambda kv: kv[1])
        if score >= cfg.page_drop_min_score:
            page.page_class, page.keep = cls, False
        elif score >= 0.5:
            page.page_class_guess = cls
    return doc
