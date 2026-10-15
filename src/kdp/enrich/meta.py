"""문서 메타 승격(기획서 4.2·D3) — 지금은 표지에서 제목·날짜·문서번호. 담당자 이름·전화는 올리지 않는다(D15)."""
from __future__ import annotations

from ..classify.pages import DOC_NO, content_blocks, date_blocks, doc_date, title_blocks
from ..config import MetaCfg
from ..schema import Document


def promote_meta(doc: Document, cfg: MetaCfg) -> Document:
    doc = doc.model_copy(deep=True)
    covers = [p for p in doc.pages if p.page_class == "cover"]
    blocks = [b for p in covers for b in content_blocks(p) if b.text]
    meta: dict[str, str] = {}
    titles = [" ".join(b.text.split()) for p in covers for b in title_blocks(p)]   # 표지 판정이 제목으로 본 블록
    if "title" in cfg.promote and titles:
        meta["title"] = " ".join(titles)
    if "date" in cfg.promote and (dates := date_blocks(blocks)):
        y, mo, d = doc_date(dates[0].text)
        meta["date"] = f"{y}-{mo:02d}" + (f"-{d:02d}" if d else "")
    for b in blocks:
        if "doc_no" in cfg.promote and "doc_no" not in meta and (m := DOC_NO.search(b.text.strip())):
            meta["doc_no"] = " ".join((m.group(1) or m.group(2)).split())
    doc.meta = {**meta, **doc.meta}
    return doc
