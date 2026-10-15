"""운영 QC(기획서 8.5). 지금은 블록 오류·잘림·빈 출력·drop 이상만 — 텍스트 레이어 일치·반복·환각은 M4."""
from __future__ import annotations

import html
import re

from ..config import QCCfg
from ..engine.base import RawPage
from ..schema import QC, Document

INK_MIN = 0.01                  # 쪽번호·간지 글자만 있는 쪽은 0.0~0.002, 본문 6줄 쪽은 0.015(합성 PDF 실측)
_TAG = re.compile(r"<[^>]+>")
_LEADER = re.compile(r"\.{2,}|…+|‥+|·{2,}")      # 목차 점선


def visible_chars(b) -> int:
    """보이는 글자 수 — 표 HTML 의 태그와 목차 점선은 세지 않는다('<표 1>' 같은 본문 글자는 그대로 센다)."""
    s = html.unescape(_TAG.sub(" ", b.html)) if b.html else (b.text or b.figure_text or "")
    return sum(1 for c in _LEADER.sub("", s) if not c.isspace())


def run_checks(doc: Document, raws: list[RawPage], cfg: QCCfg) -> Document:
    doc = doc.model_copy(deep=True)
    blocks = [b for p in doc.pages for b in p.blocks]
    errors = sum(1 for b in blocks if b.error)
    truncated = sum(1 for b in blocks if b.truncated)
    ink = {r.input.page_no: r.ink_ratio for r in raws}
    # 내용 없는 쪽 — 공문 메타·결재란만 남은 쪽은 내용이 있는 것으로, 빈 쪽으로 버렸는데 잉크가 있는 쪽(검출이 아무것도 못 찾았거나
    # 쪽번호·머리말만 찾은 쪽 — '비워 둠' 문구 블록이 있는 쪽은 아니다)은 빈 것으로 센다(기획서 8.5 — 잉크가 있는 쪽인데 keep 글자 0)
    empty = sum(1 for p in doc.pages if ink.get(p.page_no, 0) > INK_MIN and (
        (p.keep and not any(visible_chars(b) for b in p.blocks if b.keep or b.drop_reason in ("meta", "approval")))
        or (p.page_class == "blank" and not any(visible_chars(b) for b in p.blocks if b.keep))))
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
    agree = [p.text_layer_agreement for p in doc.pages if p.text_layer_agreement is not None]
    layer = round(sum(agree) / len(agree), 4) if agree else None
    if layer is not None and layer < cfg.min_text_layer_agreement:
        reasons.append("text_layer_agreement")
    if any(pv.get("rule") == "region_mix_suspect" for p in doc.pages if p.keep for b in p.blocks if b.keep
           for pv in b.provenance or []):
        reasons.append("region_mix")                 # 서로 다른 영역이 섞인 의심 블록 — 검수 대상
    tables = [pv["agreement"] for p in doc.pages if p.keep for b in p.blocks if b.keep and b.type == "table"
              for pv in b.provenance or [] if pv.get("rule") == "text_layer" and pv.get("agreement") is not None]
    if any(a < cfg.min_text_layer_agreement for a in tables):
        reasons.append("table_layer_agreement")       # 모델 표가 레이어 글자를 다 담지 못했다(행 누락 등) — 검수
    if any(p.page_class_guess for p in doc.pages):
        reasons.append("page_class_guess")         # 쪽 판정을 확신하지 못해 남겨 둔 쪽 — 검수 대상
    status = "error" if "block_errors" in reasons else ("review" if reasons else "ok")
    doc.qc = QC(status=status, reasons=reasons, block_errors=errors, truncated_blocks=truncated,
                empty_pages=empty, drop_char_ratio=drop_ratio, text_layer_agreement=layer)
    return doc
