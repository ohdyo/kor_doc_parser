"""keep 블록 순서와 output_hash."""
from __future__ import annotations

import hashlib
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


def output_hash(doc: Document) -> str:
    h = hashlib.sha256()
    for _, b in kept(doc):
        for part in (b.type, b.text, b.html, b.figure_text):
            h.update((part or "").encode("utf-8") + b"\x1f")
        h.update(b"\x1e")
    return "sha256:" + h.hexdigest()
