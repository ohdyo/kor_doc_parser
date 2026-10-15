"""keep 블록 순서와 output_hash."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator

from ..schema import Block, Document, Page


def has_output(b: Block) -> bool:
    """내용이 있는 블록 — 공백뿐인 글은 내용이 아니다(VLM 이 빈 영역에서 줄바꿈만 내기도 한다)."""
    return bool((b.text or "").strip() or (b.html or "").strip() or (b.figure_text or "").strip() or b.img_path)


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


OUTPUT_FIELDS = ("type", "bbox", "section_path", "heading_level", "list_level", "text", "html", "continues_from", "header_rows",
                 "caption_id", "unit", "note_ids", "route", "route_reason", "img_path", "figure_text", "truncated", "pii")


def output_rows(doc: Document):
    """청킹용 블록 목록(blocks.jsonl) 줄 — output_hash 도 이 줄을 그대로 해시한다(둘이 어긋나지 않게)."""
    for page, b in kept(doc):
        row = {"doc_id": doc.doc_id, "block_id": b.id, "page_no": page.page_no, "sheet_no": page.sheet_no}
        for f in OUTPUT_FIELDS:
            v = getattr(b, f)
            if v is not None:
                row[f] = list(v) if isinstance(v, tuple) else v
        yield row


def output_hash(doc: Document) -> str:
    """블록 목록 줄 전부와 문서 메타 — 이것이 바뀐 문서만 다시 적재한다(기획서 4.4). 줄에 블록 번호·쪽 번호가 있어서
    버리는 블록이 하나 더 잡혀 번호만 밀려도 해시가 바뀐다(옛 block_id 가 새 정본에서 다른 블록을 가리키지 않게)."""
    h = hashlib.sha256(json.dumps(doc.meta, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\x1d")
    for row in output_rows(doc):
        h.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\x1e")
    return "sha256:" + h.hexdigest()
