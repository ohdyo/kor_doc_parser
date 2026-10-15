"""운영 QC(기획서 8.5). 지금은 블록 오류·잘림·빈 출력·drop 이상만 — 텍스트 레이어 일치·반복·환각은 M4."""
from __future__ import annotations

import html
import re

from ..config import QCCfg
from ..engine.base import RawPage
from ..schema import QC, Document

INK_MIN = 0.01
_TAG = re.compile(r"<[^>]+>")


def visible_chars(b) -> int:
    """보이는 글자 수 — 표 HTML 의 태그는 세지 않는다('<표 1>' 같은 본문 글자는 그대로 센다)."""
    s = html.unescape(_TAG.sub(" ", b.html)) if b.html else (b.text or b.figure_text or "")
    return sum(1 for c in s if not c.isspace())


def run_checks(doc: Document, raws: list[RawPage], cfg: QCCfg) -> Document:
    doc = doc.model_copy(deep=True)
    blocks = [b for p in doc.pages for b in p.blocks]
    errors = sum(1 for b in blocks if b.error)
    truncated = sum(1 for b in blocks if b.truncated)
    ink = {r.input.page_no: r.ink_ratio for r in raws}
    empty = sum(1 for p in doc.pages
                if p.keep and ink.get(p.page_no, 0) > INK_MIN
                and not any(b.keep and visible_chars(b) for b in p.blocks))
    total = sum(visible_chars(b) for b in blocks)
    dropped = sum(visible_chars(b) for p in doc.pages for b in p.blocks if not (p.keep and b.keep))
    drop_ratio = round(dropped / total, 4) if total else 0.0

    reasons = []
    if errors > cfg.max_block_errors:
        reasons.append("block_errors")
    if truncated:
        reasons.append("truncated")
    if empty:
        reasons.append("empty_pages")
    if drop_ratio > cfg.max_drop_char_ratio or (doc.pages and not any(p.keep for p in doc.pages)):
        reasons.append("drop_ratio")
    status = "error" if "block_errors" in reasons else ("review" if reasons else "ok")
    doc.qc = QC(status=status, reasons=reasons, block_errors=errors, truncated_blocks=truncated,
                empty_pages=empty, drop_char_ratio=drop_ratio)
    return doc
