"""블록 라벨 정책(기획서 5.2). 글상자(aside_text)는 버리되 내용이 있는 것은 살리고 aside 를 단다.
문장 꼴인 문단 제목 라벨은 본문(text)으로 바꾼다 — 문서 제목(doc_title)은 그대로(표어·보도자료 제목은 문장 꼴이어도 문서 제목).
'붙임 1' 꼬리표 옆 같은 줄 제목은 꼬리표와 한 제목 블록으로 합친다(라벨을 버리기 전에).
결재란·공문 메타 규칙은 rules_meta."""
from __future__ import annotations

import re
from collections import Counter

from ..assemble.tables import CAPTION, FIG_CAPTION, bracket_title
from ..config import Classify
from ..geometry import local, page_size
from ..schema import Block, Document, Page

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
ITEM = re.compile(r"^\s*(?:[□■○●◦▪ㅇ◆◇▶►➢•\-–·*]|\d{1,2}[.)]|[가-하][.)]|\(\d{1,2}\)|[①-⑳➀-➓❶-❿])\s*\S")   # 개조식 항목 줄
# 번호 붙은 붙임 꼬리표만 있는 블록('붙임 1'·'[붙임 2]'·'별첨 3') — 옆 상자의 글이 그 붙임의 제목이다
APPENDIX_TAG = re.compile(r"[\[<〈【(「『]?\s*(?:붙\s*임|별\s*첨|첨\s*부|부\s*록|별\s*지)\s*(?:제\s*)?(?:\d{1,2}|[Ⅰ-Ⅻ]|[A-Z])"
                          r"\s*(?:호\s*)?(?:서\s*식\s*)?[\]>〉】)」』]?\s*\.?")
_TAG_HEAD = re.compile(APPENDIX_TAG.pattern + r"(?:\s+|$)")                 # 꼬리표로 시작하는 줄(뒤는 띄어쓰기·끝)
_CHAPTER = re.compile(r"\s*(?:[Ⅰ-Ⅻ](?:\s*[.)]|\s)|[IVX]{1,4}\s*[.)]|제\s*\d{1,3}\s*[편장절관조])")   # 장·절 번호 제목 — 꼬리표에 합치지 않는다
_TAG_NO = re.compile(r"[\[<〈【(「『]?\s*(붙\s*임|별\s*첨|첨\s*부|부\s*록|별\s*지)\s*(?:제\s*)?(\d{1,2}|[Ⅰ-Ⅻ]|[A-Z])(?!\d)")   # 꼬리표 이름·번호
_ATTACH_LIST = re.compile(r"\d\s*부\s*\.|\d\s*부\s*$|끝\s*\.?\s*$")         # '붙임 1. 계획 1부. 끝.' — 붙임 목록 줄
TAG_GAP = 0.1               # 꼬리표와 옆 제목 사이 가로 틈 상한(논리 쪽 너비 대비, A4 약 60pt)
TAG_TITLE_MAX = 60          # 꼬리표 옆 제목 글자 수(띄어쓰기 뺀) 상한 — 긴 것은 본문 문장
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


def appendix_title(b: Block) -> bool:
    """_tag_titles 가 꼬리표와 합쳤거나 머리말에서 돌린 '붙임 N …' 제목 — '붙임 1 서식에 따라 … 바랍니다.' 같은 안내 문장 라벨은 아니다."""
    return any(p.get("rule") == "appendix_title" for p in b.provenance or [])


def _key(text: str | None) -> str:
    return "".join((text or "").split())


def _tag_key(text: str | None) -> str | None:
    """번호 붙은 붙임 꼬리표의 이름·번호('붙임2') — 쪽 표시·판독 차이가 있어도 같은 붙임을 가린다."""
    m = _TAG_NO.match(text or "")
    return re.sub(r"\s", "", m.group(1)) + m.group(2) if m else None


def _masked(text: str | None) -> str:
    return re.sub(r"\d+", "#", _key(text))


def _running(doc: Document) -> tuple[set[str], set[str]]:
    """되풀이 머리말 — 숫자를 뺀 글이 두 쪽 이상에 나온 머리말('붙임 1 사업계획서'·'… (2/3)' 를 쪽마다 찍은 문서)과 그 꼬리표."""
    pages = [{_masked(b.text): b.text for b in p.blocks if b.type == "header" and b.text} for p in doc.pages]
    seen = Counter(k for p in pages for k in p)
    run = {k for k, n in seen.items() if n >= 2}
    return run, {t for p in pages for k, text in p.items() if k in run and (t := _tag_key(text))}


def _tag_titles(page: Page, running: tuple[set[str], set[str]], seen: set[str], titles: set[str]) -> None:
    """'붙임 1' 꼬리표와 그 붙임의 제목 — 쪽 맨 위 제목 상자를 검출기가 머리말(header)로 잡아 버리던 것(실측 test2.pdf, 쪽마다
    글이 다르다). 머리말이 꼬리표로 시작하고 그 붙임 번호가 처음 나온 것(앞쪽·이 쪽 제목에도, 되풀이 머리말에도 없음)이면
    제목으로 돌린다 — 꼬리표 뒤 제목 글이 앞쪽 제목·머리말과 같아도 되풀이다. 꼬리표 블록 오른쪽 같은 줄 블록(한 줄·짧고
    항목·번호·장 제목·표·그림 제목 꼴이 아님 — 상자가 꼬리표와 조금
    겹쳐도 오른쪽으로 더 뻗으면)의 글은 꼬리표 블록에 합친다. 문서 제목(doc_title)은 문서 제목으로 둔다. seen·titles 에 이 쪽
    꼬리표·제목 글(꼬리표를 뗀 _key)을 더한다."""
    run, run_tags = running
    live = [b for b in page.blocks if b.merged_into is None and (b.text or "").strip()]
    here = {k for b in live if b.keep and b.type in ("paragraph_title", "doc_title", "text") and (k := _tag_key(b.text))}
    for b in live:
        s = " ".join(b.text.split())
        if b.type == "header" and (k := _tag_key(s)) and k not in seen | here | run_tags and _masked(s) not in run \
                and "\n" not in b.text.strip() and (m := _TAG_HEAD.match(s)) and not _ATTACH_LIST.search(s) \
                and len(_key(s[m.end():])) <= TAG_TITLE_MAX and _key(s[m.end():]) not in titles:
            b.provenance = (b.provenance or []) + [{"rule": "appendix_title", "type": b.type}]
            b.type = "paragraph_title"
    width = page_size(page)[0]
    for tag in live:
        if not tag.keep or tag.merged_into is not None or tag.type not in ("paragraph_title", "doc_title", "text") \
                or not APPENDIX_TAG.fullmatch(" ".join(tag.text.split())):
            continue
        tx0, ty0, tx1, ty1 = local(page, tag.bbox)
        right = []
        for b in live:
            if b is tag or b.merged_into is not None or b.type not in ("header", "paragraph_title", "text") \
                    or (b.type == "header" and _masked(b.text) in run):
                continue
            x0, y0, x1, y1 = local(page, b.bbox)
            s = " ".join(b.text.split())
            if min(ty1, y1) - max(ty0, y0) >= 0.5 * min(ty1 - ty0, y1 - y0) and x0 >= tx0 - 2 and x1 > tx1 \
                    and x0 - tx1 <= TAG_GAP * width and "\n" not in b.text.strip() and len(s.replace(" ", "")) <= TAG_TITLE_MAX \
                    and not (ITEM.match(s) or _CHAPTER.match(s) or APPENDIX_TAG.match(s) or CAPTION.match(s) or FIG_CAPTION.match(s)
                             or bracket_title(s)
                             or _ATTACH_LIST.search(s)):                  # 붙임 목록 줄('… 1부.  끝.')은 제목이 아니다
                right.append((x0, b))
        if not right:
            continue
        b = min(right, key=lambda r: r[0])[1]
        tag.text = " ".join(tag.text.split()) + " " + " ".join(b.text.split())
        tag.bbox = (min(tag.bbox[0], b.bbox[0]), min(tag.bbox[1], b.bbox[1]),
                    max(tag.bbox[2], b.bbox[2]), max(tag.bbox[3], b.bbox[3]))
        tag.type = "paragraph_title" if tag.type == "text" else tag.type
        tag.provenance = (tag.provenance or []) + [{"rule": "appendix_title", "with": b.id, "type": b.type}]
        b.keep, b.drop_reason, b.merged_into, b.text = False, "merged", tag.id, None   # 글자는 대표에만(병합 묶음 규약)
    seen.update(k for b in page.blocks if b.keep and b.merged_into is None
                and b.type in ("paragraph_title", "doc_title", "text") and (k := _tag_key(b.text)))
    for b in page.blocks:                              # 앞쪽 제목·머리말 글 — 첫 쪽에 꼬리표 없이 제목만 둔 붙임의 뒤쪽 머리말
        if b.merged_into is None and b.text and (b.type == "header" or b.keep and b.type in ("paragraph_title", "doc_title")):
            s = " ".join(b.text.split())
            titles.add(_key(s[m.end():] if (m := _TAG_HEAD.match(s)) else s))


def classify_blocks(doc: Document, cfg: Classify) -> Document:
    drop = dict(DROP_LABELS)
    if cfg.footnote == "drop":
        drop["footnote"] = "footnote"
        drop["vision_footnote"] = "footnote"
    doc = doc.model_copy(deep=True)
    running, seen, titles = _running(doc), set(), set()
    for page in doc.pages:
        _tag_titles(page, running, seen, titles)
        for b in page.blocks:
            if b.merged_into is None and b.type == "paragraph_title" and sentence_like(b.text) \
                    and not appendix_title(b):        # 꼬리표와 합친·머리말에서 돌린 '붙임 2 …' 제목은 문장 꼴이어도 붙임 뿌리
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
