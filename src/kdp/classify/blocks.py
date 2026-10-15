"""블록 라벨 정책(기획서 5.2). 글상자(aside_text)는 버리되 내용이 있는 것은 살리고 aside 를 단다.
문장 꼴인 문단 제목 라벨은 본문(text)으로 바꾼다 — 문서 제목(doc_title)은 그대로(표어·보도자료 제목은 문장 꼴이어도 문서 제목).
결재란·공문 메타 규칙은 rules_meta."""
from __future__ import annotations

import re

from ..config import Classify
from ..schema import Block, Document

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
TITLE_SENTENCE_MIN = 12     # 이보다 짧은(띄어쓰기 뺀) 제목은 끝맺음 꼴이어도 그대로 둔다 — 명사 제목('추진 배경')은 짧다
_NOTE = re.compile(r"^\s*(?:※|[□■○●◦▪*]?\s*[\[<〈【(]?\s*참\s*고(?:\s*(?:자\s*료|사\s*항))?(?![가-힣]))")   # '※'·'□ 참고'·'참고자료' 박스
_END = re.compile(r"(?<=[가-힣]{2})[.!?。](?=\s|$)|(?:다|요|함|음|임|됨)\s*$")   # 문장 끝 — '가.'·'2026.' 은 아니다
ITEM = re.compile(r"^\s*(?:[□■○●◦▪ㅇ◆◇▶►➢•\-–·*]|\d{1,2}[.)]|[가-하][.)]|\(\d{1,2}\)|[①-⑳])\s*\S")   # 개조식 항목 줄
# 제목 라벨의 문장 끝 — 개조식(…같음·있음·함·임·됨)과 …니다·한다·같다·요. '책임·포함'·'개요·필요'·'만들다'(표어 제목)는 아니다
_CLAUSE = re.compile(r"(?:[같있없였었했겠않]음|(?<!포)함|(?<![책모위])임|됨|니다|[이하한된인였었했있없겠같]다"
                     r"|[어아여해워와봐줘져세네게데래대나까에]요)\s*[.。!?]?\s*$")


def sentences(text: str) -> int:
    """문장 수 — 끝맺음(다·요·함…·마침표)과 개조식 항목 줄(옆 요약)을 하나씩 센다."""
    return sum(max(len(_END.findall(ln)), int(bool(ITEM.match(ln)))) for ln in (text or "").splitlines() if ln.strip())


def sentence_like(text: str | None) -> bool:
    """제목 라벨인데 문장 꼴 — 개조식·문장 끝맺음이고 충분히 길다(실측: 사내 시험 문서의 '■ … 아래와 같음' 꼴 항목 줄을 검출기가
    paragraph_title 로 잡았다)."""
    s = " ".join((text or "").split())
    return len(s.replace(" ", "")) >= TITLE_SENTENCE_MIN and bool(_CLAUSE.search(s))


def _keep_aside(b: Block, cfg: Classify) -> bool:
    """문장이 기준 이상인 글상자·'※ 참고' 박스·옆 요약 — 여백 메모·띠지('대외비')는 버린다."""
    return bool(_NOTE.match(b.text or "")) or sentences(b.text) >= cfg.aside_keep_min_sentences


def classify_blocks(doc: Document, cfg: Classify) -> Document:
    drop = dict(DROP_LABELS)
    if cfg.footnote == "drop":
        drop["footnote"] = "footnote"
        drop["vision_footnote"] = "footnote"
    doc = doc.model_copy(deep=True)
    for page in doc.pages:
        for b in page.blocks:
            if b.merged_into is None and b.type == "paragraph_title" and sentence_like(b.text):
                b.provenance = (b.provenance or []) + [{"rule": "title_sentence", "type": b.type}]
                b.type = "text"                      # 문장이 문단 제목 라벨을 받았다 — 섹션 경로·표지 제목이 되지 않게
                continue
            if b.merged_into is not None or b.type not in drop:
                continue
            if b.type == "aside_text" and _keep_aside(b, cfg):
                b.aside = True
                b.provenance = (b.provenance or []) + [{"rule": "aside_keep"}]
                continue
            b.keep, b.drop_reason = False, drop[b.type]
    return doc
