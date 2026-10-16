"""쪽 판정(기획서 5.1, 설계 5.2) — 학습 전 규칙. 표지·목차·빈 쪽·간지·결재 쪽·미기입 서식.

점수가 classify.page_drop_min_score 이상이면 쪽을 버리고(keep=false), 0.5 이상이면 page_class_guess 만 남긴다(검수 대상).
미기입 서식은 classify.form_blank 가 keep_flag 면 버리지 않고 page.form 만 단다(D2). 면적은 논리 페이지로 잰다(모아찍기).
학습한 쪽 분류기(기획서 7.2 (B))가 생기면 이 함수 자리를 바꾼다.
"""
from __future__ import annotations

import re

from ..assemble.tables import parse_rows
from ..config import Classify
from ..convert import FIGURE_LABELS
from ..geometry import local, page_size
from ..qc.checks import visible_chars
from ..schema import Block, Document, Page

JUNK_LABELS = {"header", "header_image", "footer", "footer_image", "number", "seal", "aside_text"}
META_LABELS = {"doc_meta", "approval_box"}   # 공문 메타·결재란(rules_meta) — 표지는 이것이 있어도 표지다(설계 5.2)
TEXT_LABELS = {"text", "paragraph_title", "doc_title"}
COVER_MAX_BLOCK_CHARS = 100
COVER_MAX_CHARS = 250
TOC_MAX_OTHER_CHARS = 100          # 목차 줄이 아닌 본문이 이보다 많으면 쪽은 남기고 목차 블록만 버린다
DIVIDER_MAX_CHARS = 40             # 간지 글자 수 상한('붙임 1' + 짧은 부제)
DIVIDER_MAX_AREA = 0.2             # 간지 글 블록 면적 / 논리 페이지 면적 — 넘으면 버리지 않고 검수로
APPROVAL_MAX_OTHER = 50            # 결재란·공문 메타 말고 다른 글이 이보다 많으면 결재 쪽이 아니다
FORM_MIN_CELLS = 8
FORM_EMPTY = 0.4                   # 빈 칸 비율 — 서식 표시('…신청서'·'(서명 또는 인)')도 있어야 쪽 판정, 없으면 검수로
FORM_GUESS = 0.25
FORM_MIN_LABELS = 0.15             # 글 칸 비율 하한 — 칸을 거의 못 읽은 표(인식 실패)는 서식이 아니다
FORM_MAX_DATA = 0.15               # 숫자·표시(■●) 칸이 글 칸의 이 비율을 넘으면 자료 표(일정표 등)
FORM_TABLE_SHARE = 0.5             # 표가 쪽 내용 면적의 이 비율 이상

_STRIP = re.compile(r"[\s\[\]【】()<>〈〉《》『』「」:：]")
_TOC_TITLE = re.compile(r"(목차|차례|contents|tableofcontents|(표|그림|도표|사진|부록)(목차|차례))", re.I)
_TOC_LINE = re.compile(r"((\.{3,}|…{2,}|·{3,}|‥{2,}|_{3,})\s*\S{0,6}|\S\s+\d{1,4})\s*$")    # 점선 + 쪽, 또는 끝의 쪽번호
_DATE = re.compile(r"(?<![\d\-./])((?:19|20)\d{2})\s*(?:년|[.\-/])\s*(\d{1,2})\s*(?:월|[.\-/]|(?=[\s(),\]}~]|$))"
                   r"(?:\s*(\d{1,2})\s*(?:일|\.|(?=[\s(),\]}~]|$)))?(?!\d)")
DOC_NO = re.compile(r"(?:(?:문서|관리)\s*번\s*호\s*[:：]?\s*(\S+)|(\S*\d{2,4}\s*-\s*\S*?\d+\s*호))\s*$")
_SENTENCE = re.compile(r"(?:니다|십시오|[이하한된인였었했있없겠]다)\s*[.。!?]?\s*$|니다")   # 문장 끝맺음 — '개요'·'주요' 같은 명사 끝은 아니다
_TERSE = re.compile(r"[함음임됨]\s*\.?\s*$")                                            # 개조식 끝맺음(긴 줄만 본다)
_ITEM = re.compile(r"^(?:[가나다라마바사아자차카타파하][.)]|\d{1,2}[.)]|\(\d+\)|[①-⑳➀-➓❶-❿□■○●◦▪ㅇ]|[-•▶►◆◇➢*·]\s"
                   r"|(?:일\s*시|장\s*소|참석자?|안\s*건|대\s*상|목\s*적|방\s*법|내\s*용|주\s*제|회의명)\s*[:：])")   # 행사·회의 항목 줄
# 빈 쪽 문구 — 쪽 글 전체(장식·띄어쓰기 뺀)가 이것일 때만. '이하 여백'·'돌봄 공백'·'국민참여 백서' 는 내용이 있는 쪽이다
_BLANK = re.compile(r"(?:(?:이|본)?(?:페이지|쪽|면)[은는]?)?(?:의도적으로)?(?:비워(?:두었습니다|둡니다|두었음|둠|두었다|놓았습니다|놓았음|놓음"
                    r"|둔(?:쪽|면|페이지)(?:입니다|임)?)|비어있(?:습니다|음)|빈(?:쪽|면|페이지)(?:입니다|임)?|백지|공백|여백|공란)"
                    r"|(?:(?:this)?page(?:is|hasbeen)?)?(?:intentionally)?(?:left)?blank(?:page)?", re.I)
_DECO = re.compile(r"[\s.\-–—―=~·*()\[\]<>〈〉【】「」『』]")                 # 빈 쪽 문구를 감싼 장식
_NAME_VALUE = re.compile(r"\S\s*[:：]\s*\S")                               # '교육 일시: 2026. 3. 15.' 같은 '이름: 값' 줄
_DIVIDER = re.compile(r"\s*[\[<〈【(]?\s*(?:(?:붙\s*임|별\s*첨|첨\s*부|부\s*록)(?:\s*(?:\d{1,2}|[A-Z]|[Ⅰ-Ⅻ]))?"
                      r"|제\s*\d{1,2}\s*[장편부])\s*[\]>〉】)]?(?![가-힣])")
_COPIES = re.compile(r"\d\s*부\s*\.?\s*$", re.M)                        # '○○계획 1부.' — 붙임 목록 줄(간지가 아니다)
_PLACEHOLDER = re.compile(r"[\s()（）/.:~\-]|년|월|일|서명|또는|인")       # 서식 빈 칸에 미리 찍힌 글자
_NUM = re.compile(r"[\d,.%~\-+/()]*\d[\d,.%~\-+/()]*")
_ROWNO = re.compile(r"\d{1,2}\.?")                                       # 번호 칸 — 빈 명단 서식에도 있다
_MARK = re.compile(r"[■●▲◆★◎○◉✓✔☑√]+")                                 # 일정표 표시 칸
_FORM_CUE = re.compile(r"서\s*식|양\s*식|신\s*청\s*서|확\s*인\s*서|동\s*의\s*서|신\s*고\s*서|서\s*약\s*서|서\s*명"
                       r"|\(\s*인\s*\)|년\s+월\s+일")


def content_blocks(page: Page) -> list[Block]:
    """표지·목차 판정에 쓰는 블록 — 잡요소 라벨·공문 메타·결재란과 병합 구성원은 뺀다."""
    return [b for b in page.blocks if b.merged_into is None and b.type not in JUNK_LABELS and b.type not in META_LABELS]


def _content(page: Page) -> list[Block]:
    """내용이 있는 블록 — 잡요소(버린 글상자 포함)·병합 구성원은 빼고 살린 글상자·공문 메타·결재란·그림·오류 블록은 넣는다."""
    return [b for b in page.blocks if b.merged_into is None and (b.type not in JUNK_LABELS or b.aside)
            and (b.error or b.html or b.img_path or b.type in FIGURE_LABELS or visible_chars(b))]


def _area(b: Block) -> float:
    return (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1])


def is_toc_title(b: Block) -> bool:
    return b.type != "content" and bool(_TOC_TITLE.fullmatch(_STRIP.sub("", b.text or "")))


def _line_h(page: Page, b: Block) -> float:
    _, y0, _, y1 = local(page, b.bbox)                    # 세운 쪽 기준(돌린 스캔)
    return (y1 - y0) / max(1, (b.text or "").count("\n") + 1)


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
    med = sorted(_line_h(page, b) for b in texts)[len(texts) // 2]
    return [b for b in texts if b.type == "paragraph_title" and _line_h(page, b) >= 1.4 * med]


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
    w, h = page_size(page)
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


def _blank_score(page: Page) -> float:
    """빈 쪽 — 내용 블록이 없거나 글 전체가 '의도적으로 비워 둠' 류 문구. 짧은 본문은 빈 쪽이 아니다.
    잉크는 여기서 보지 않는다 — 버린 빈 쪽에 잉크가 있으면 QC 가 빈 출력으로 센다(run_checks)."""
    blocks = _content(page)
    if not blocks:
        return 1.0
    if any(b.type not in TEXT_LABELS or b.error for b in blocks):
        return 0.0
    text = _DECO.sub("", "".join(b.text or "" for b in blocks))
    return 0.95 if text and _BLANK.fullmatch(text) else 0.0


def _divider_score(page: Page) -> float:
    """간지 — 짧은 '붙임 n·별첨·부록·제n장' 제목 하나(+ 짧은 부제)와 여백. 붙임 목록·'끝.'·문장, 제목 뒤의 항목·'※' 주·
    '이름: 값' 줄이 있으면 내용이 든 쪽이다(기획서 5.1 — 간지는 '붙임 n·제n장' 만 있는 쪽)."""
    blocks = _content(page)
    if not blocks or len(blocks) > 2 or any(b.type not in TEXT_LABELS or b.error for b in blocks):
        return 0.0
    text = "\n".join(b.text or "" for b in blocks).strip()
    m = _DIVIDER.match(text)
    if (not m or _chars(text) > DIVIDER_MAX_CHARS or "끝." in re.sub(r"\s+", "", text) or _COPIES.search(text[m.end():])
            or any(_SENTENCE.search(ln) for ln in text.splitlines())):
        return 0.0
    rest = [s for s in (ln.strip() for ln in text[m.end():].splitlines()) if s]
    if any(_ITEM.match(s) or s.startswith("※") or _NAME_VALUE.search(s) or (len(s) >= 8 and _TERSE.search(s)) for s in rest):
        return 0.0
    w, h = page_size(page)
    return 0.95 if sum(_area(b) for b in blocks) <= DIVIDER_MAX_AREA * w * h else 0.6


def _approval_score(page: Page) -> float:
    """결재 쪽 — 결재란·공문 메타만 있으면 버리고, 결재란·메타가 면적 대부분이고 다른 글이 짧으면 검수로."""
    blocks = _content(page)
    if not any(b.type == "approval_box" for b in blocks):
        return 0.0
    other = [b for b in blocks if b.type not in META_LABELS]
    if not other:
        return 1.0
    meta = sum(_area(b) for b in blocks if b.type in META_LABELS)
    if sum(visible_chars(b) for b in other) <= APPROVAL_MAX_OTHER and meta >= 0.5 * sum(_area(b) for b in blocks):
        return 0.6
    return 0.0


def _data_cell(s: str) -> bool:
    s = re.sub(r"\s+", "", s)
    return bool(_MARK.fullmatch(s) or (_NUM.fullmatch(s) and not _ROWNO.fullmatch(s)))


def _form_score(page: Page) -> float:
    """미기입 서식 — 쪽 대부분이 표이고 빈 칸이 많으며 글 칸은 항목 이름. 서식 표시가 없으면 검수로만."""
    blocks = _content(page)
    tables = [b for b in blocks if b.type == "table" and b.html]
    if not tables or sum(_area(b) for b in tables) < FORM_TABLE_SHARE * sum(_area(b) for b in blocks):
        return 0.0
    cells = [c.text for t in tables for r in parse_rows(t.html) for c in r]
    filled = [c for c in cells if _PLACEHOLDER.sub("", c)]
    if (len(cells) < FORM_MIN_CELLS or len(filled) < max(3, FORM_MIN_LABELS * len(cells))
            or sum(_data_cell(c) for c in filled) > FORM_MAX_DATA * len(filled)):
        return 0.0
    empty = 1 - len(filled) / len(cells)
    cue = bool(_FORM_CUE.search("\n".join([b.text or "" for b in blocks] + cells)))
    if empty >= FORM_EMPTY:
        return 0.95 if cue else 0.6
    return 0.6 if empty >= FORM_GUESS and cue else 0.0


def classify_pages(doc: Document, cfg: Classify) -> Document:
    doc = doc.model_copy(deep=True)
    pages = doc.pages
    first_blank = bool(pages) and not any(visible_chars(b) for b in content_blocks(pages[0]))
    for i, page in enumerate(pages):
        scores = {"cover": _cover_score(page, i, len(pages), first_blank), "toc": _toc_score(page),
                  "blank": _blank_score(page), "divider": _divider_score(page),
                  "approval": _approval_score(page), "form_blank": _form_score(page)}
        cls, score = max(scores.items(), key=lambda kv: kv[1])
        if score >= cfg.page_drop_min_score:
            page.page_class = cls
            page.keep = cls == "form_blank" and cfg.form_blank == "keep_flag"   # 미기입 서식은 살리고 표시만(D2)
            if cls == "form_blank":
                page.form = True
        elif score >= 0.5:
            page.page_class_guess = cls
    return doc
