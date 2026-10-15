"""문서 메타 승격(기획서 4.2·D3) — 공문 메타 블록 값, 표지의 제목·기관·날짜·문서번호, 도장 유무(has_stamp).
담당자 이름·전화 같은 사람 정보는 promote_person_fields 가 참일 때만 올리고, 다른 항목 값에 든 연락처도 올리지 않는다(D15).

메타 블록은 첫 문서 단위(첫 쪽 — 공문이면 첫 '끝.' 쪽까지)의 것만 표지보다 앞세운다. 그 뒤 쪽의 것(붙임에 실린 다른 공문
사본)은 표지가 없는 문서에서만, 비어 있는 키에만 쓴다.
"""
from __future__ import annotations

import re

from ..classify.pages import DOC_NO, content_blocks, date_blocks, doc_date, title_blocks
from ..classify.rules_meta import DATES, END, ORG_MAX_CHARS, PERSON_FIELDS, is_org
from ..config import MetaCfg
from ..schema import Block, Document, Page
from .pii import PHONE, kinds

_CUE = re.compile(r"[☎☏@]|전\s*화|연락처|휴\s*대|내\s*선|팩\s*스|전\s*송|이메일|전자\s*우편|e-?mail|\btel\b|\bfax\b", re.I)
_RANK = re.compile(r"주무관|사무관|서기관|주사보?|서기보?|연구관|연구사|과장|팀장|차장|부장|실장|대리|주임|사원|선임|책임|수석|연구원"
                   r"|담당자?")                                   # 이름 앞뒤의 직급(낱말 하나 전체)
_HEAD = re.compile(r"(?<=[가-힣]{2})(?:장관|장)$")              # 기관장 직함 꼬리 — '행정안전부장관'·'정보화팀장' 은 그 기관으로
_LEGAL = re.compile(r"(?:주식|유한|합자|합명)회사|(?:재단|사단)법인|\(주\)|㈜")   # 회사·법인 꼴 — 이것만으로는 기관 이름이 아니다


def _pii(v: str) -> bool:
    return bool(kinds(v)) or "@" in v


def _org_word(w: str) -> str:
    """기관장 직함을 뗀 기관 낱말('정보화팀장' → '정보화팀') — 떼어도 기관 이름이 아니면 그대로."""
    h = _HEAD.sub("", w)
    return h if not is_org(w) and is_org(h) else w


def _org_end(words: list[str]) -> bool:
    """끝 낱말이 기관 이름인가 — 기관 이름 끝('…팀'·'…공사'), 회사·법인 꼴이 붙거나 앞에 온 이름('(주)가나정보'·'주식회사 가나정보').
    회사·법인 꼴만('주식회사')은 아니다."""
    w = words[-1]
    if _LEGAL.fullmatch(w):
        return False
    return is_org(w) or bool(_LEGAL.match(w)) or (len(words) > 1 and bool(_LEGAL.fullmatch(words[-2])))


def _issuer(v: str) -> str | None:
    """발신 값에서 기관만 — 괄호 연락처·'/ 연락처' 꼬리를 떼고, 기관 이름으로 끝나는 앞 낱말까지만 남긴다(뒤의 이름·직급은 연락처가
    없어도 뗀다 — '정보화팀 홍길동' → '정보화팀'). 그런 앞부분이 없거나 개인정보가 남으면 올리지 않는다(D15 — 확신이 없으면 승격 안 함)."""
    s = re.sub(r"\s*[(（][^()（）]*[)）]", lambda m: "" if _CUE.search(m[0]) or _pii(m[0]) else m[0], v)   # '(☎ 02-…)'
    s = re.split(r"\s*[/|,;]\s*", s)[0]                                                                  # '/ 010-…' 꼬리
    if m := _CUE.search(s) or PHONE.search(s):
        s = s[:m.start()]
    words = [_org_word(w) for w in s.split()]
    while words and (_RANK.fullmatch(words[-1]) or not _org_end(words)):    # '정보화팀 홍길동 과장' → '정보화팀'
        words.pop()
    s = " ".join(words)
    return s if s and not _pii(s) else None


def _promote(meta: dict, blocks: list[Block], cfg: MetaCfg) -> None:
    """메타 블록 값 — 같은 키는 읽는 순서로 앞의 것, 날짜는 이름별로 시행일 > 작성일 > 기안일(D15)."""
    for b in blocks:
        for k, v in b.meta.items():
            if k in DATES:
                continue
            if k in PERSON_FIELDS:
                if cfg.promote_person_fields:
                    meta.setdefault(k, v)
                continue
            if k not in cfg.promote or k in meta:
                continue
            v = _issuer(v) if k == "issuer" else None if _pii(v) else v
            if v:
                meta[k] = v
    if "date" in cfg.promote:
        for k in DATES:
            if v := next((b.meta[k] for b in blocks if k in b.meta), None):
                meta.setdefault("date", v)
                break


def _cover_issuer(covers: list[Page]) -> str | None:
    """표지의 기관 줄 — 제목·날짜·문서번호가 아닌 줄이 하나뿐이고 숫자 없이 기관 이름으로 끝날 때만(여럿이면 비운다)."""
    lines = []
    for p in covers:
        titles = title_blocks(p)
        lines += [" ".join(ln.split()) for b in content_blocks(p) if b.text and b not in titles
                  for ln in b.text.splitlines() if ln.strip()]
    rest = [ln for ln in lines if not doc_date(ln) and not DOC_NO.search(ln)]
    if len(rest) != 1 or len(rest[0].replace(" ", "")) > ORG_MAX_CHARS or re.search(r"\d", rest[0]) or not is_org(rest[0]):
        return None
    return _issuer(rest[0])


def promote_meta(doc: Document, cfg: MetaCfg) -> Document:
    doc = doc.model_copy(deep=True)
    meta: dict = {}
    pages = doc.pages
    # 첫 문서 단위 — 첫 쪽, 공문(첫 쪽이 표지가 아니고 메타가 있다)이면 첫 '끝.' 쪽까지. 이름 붙은 값이라 표지보다 확실하다
    letter = bool(pages) and pages[0].page_class != "cover" and any(b.meta for b in pages[0].blocks)
    last = next((i for i, p in enumerate(pages) if any(b.type == "text" and END.search(b.text or "") for b in p.blocks)),
                0) if letter else 0
    with_meta = [(i, b) for i, p in enumerate(pages) for b in p.blocks if b.meta]
    _promote(meta, [b for i, b in with_meta if i <= last], cfg)
    covers = [p for p in pages if p.page_class == "cover"]
    blocks = [b for p in covers for b in content_blocks(p) if b.text]
    titles = [" ".join(b.text.split()) for p in covers for b in title_blocks(p)]   # 표지 판정이 제목으로 본 블록
    if "title" in cfg.promote and titles:
        meta.setdefault("title", " ".join(titles))
    if "date" in cfg.promote and (dates := date_blocks(blocks)):
        y, mo, d = doc_date(dates[0].text)
        meta.setdefault("date", f"{y}-{mo:02d}" + (f"-{d:02d}" if d else ""))
    for b in blocks:
        if "doc_no" in cfg.promote and "doc_no" not in meta and (m := DOC_NO.search(b.text.strip())):
            meta["doc_no"] = " ".join((m.group(1) or m.group(2)).split())
    if "issuer" in cfg.promote and covers and (org := _cover_issuer(covers)):
        meta.setdefault("issuer", org)
    first = next((p for p in pages if p.keep), None)        # 표지가 없으면 첫 쪽의 문서 제목(제목 수준 0)
    if "title" in cfg.promote and "title" not in meta and first:
        top = next((b for b in first.blocks if b.keep and b.heading_level == 0 and (b.text or "").strip()), None)
        if top:
            meta["title"] = " ".join(top.text.split())
    if not covers:                                          # 뒤쪽 메타 블록은 표지가 없을 때 빈 키에만
        _promote(meta, [b for i, b in with_meta if i > last], cfg)
    if any(b.type == "seal" for p in pages for b in p.blocks):
        meta["has_stamp"] = True                           # 도장은 버리고 있다는 것만 남긴다(기획서 5.2)
    doc.meta = {**meta, **doc.meta}
    return doc
