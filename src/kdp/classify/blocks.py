"""블록 라벨 정책(기획서 5.2). 글상자 예외·결재란·공문 메타 규칙은 M2 에서 더한다."""
from __future__ import annotations

from ..config import Classify
from ..schema import Document

DROP_LABELS = {
    "header": "header",
    "header_image": "logo",
    "footer": "footer",
    "footer_image": "logo",
    "number": "page_number",
    "seal": "stamp",
    "content": "toc",
    "aside_text": "aside",
}


def classify_blocks(doc: Document, cfg: Classify) -> Document:
    drop = dict(DROP_LABELS)
    if cfg.footnote == "drop":
        drop["footnote"] = "footnote"
        drop["vision_footnote"] = "footnote"
    doc = doc.model_copy(deep=True)
    for page in doc.pages:
        for b in page.blocks:
            if b.merged_into is None and b.type in drop:
                b.keep, b.drop_reason = False, drop[b.type]
    return doc
