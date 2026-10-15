"""keep 블록 순서와 output_hash."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator

from ..schema import Block, Document, Page


def has_output(b: Block) -> bool:
    return bool(b.text or b.html or b.figure_text or b.img_path)


def kept(doc: Document) -> Iterator[tuple[Page, Block]]:
    """출력에 들어가는 블록 — 쪽 keep 과 블록 keep 이 모두 참이고 내용이 있는 것."""
    for page in doc.pages:
        if not page.keep:
            continue
        for b in page.blocks:
            if b.keep and has_output(b):
                yield page, b


def assign_order(doc: Document) -> Document:
    doc = doc.model_copy(deep=True)
    for page in doc.pages:
        for b in page.blocks:
            b.order = None
    for n, (_, b) in enumerate(kept(doc), start=1):
        b.order = n
    return doc


OUTPUT_FIELDS = ("type", "bbox", "section_path", "list_level", "text", "html", "continues_from", "header_rows",
                 "caption_id", "unit", "note_ids", "route", "route_reason", "img_path", "figure_text", "truncated", "pii")


def output_hash(doc: Document) -> str:
    """청킹용 블록 목록에 나가는 값 전부와 문서 메타 — 이것이 바뀐 문서만 다시 적재한다(기획서 4.4)."""
    h = hashlib.sha256(json.dumps(doc.meta, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\x1d")
    for _, b in kept(doc):
        row = {f: getattr(b, f) for f in OUTPUT_FIELDS}
        h.update(json.dumps(row, sort_keys=True, ensure_ascii=False, default=list).encode("utf-8") + b"\x1e")
    return "sha256:" + h.hexdigest()
