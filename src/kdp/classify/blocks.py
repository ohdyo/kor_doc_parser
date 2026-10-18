"""블록 라벨 정책(기획서 5.2). 글상자(aside_text)는 버리되 내용이 있는 것은 살리고 aside 를 단다.
문장 꼴인 문단 제목 라벨은 본문(text)으로 바꾼다 — 문서 제목(doc_title)은 그대로(표어·보도자료 제목은 문장 꼴이어도 문서 제목).
'붙임 1' 꼬리표 옆 같은 줄 제목은 꼬리표와 한 제목 블록으로 합친다(라벨을 버리기 전에).
결재란·공문 메타 규칙은 rules_meta."""
from __future__ import annotations

import re
from collections import Counter

from ..assemble.tables import CAPTION, FIG_CAPTION, _unit_of, bracket_title
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
_REF = re.compile(r"참\s*[고조](?:\s*자\s*료)?\s*[\]>〉】)」』]\s*$|참\s*[고조]\s*$")   # '<붙임 2 참고>'·'붙임 2 참조' — 본문 참조 줄
TAG_GAP = 0.1               # 꼬리표와 옆 제목 사이 가로 틈 상한(논리 쪽 너비 대비, A4 약 60pt)
TAG_TITLE_MAX = 60          # 꼬리표 옆 제목 글자 수(띄어쓰기 뺀) 상한 — 긴 것은 본문 문장
_NAME = re.compile(r"\s*[□■]\s*(?=\S)")      # 보고서체 구역 이름 줄의 기호
_MARKS = re.compile(r"[□■●○◆◇▲△▼▽━─—]")   # 범례·체크 상자 기호('■ 수출액 ● 증감률'·'■ 일강수량 ━ 누적') — 둘 이상이면 이름 줄이 아니다
NAME_MAX = 40               # □ 이름 줄 글자 수(띄어쓰기 뺀) 상한 — 긴 것은 문장 항목
NAME_INSET = 0.03           # 이름 줄 왼쪽 끝이 표·그림 왼쪽 끝보다 이만큼(쪽 너비 대비) 넘게 안쪽이면 그 안의 범례 칸이다
LOGO_W, LOGO_H = 0.25, 0.08  # 로고 줄의 작은 그림 — 너비·높이 상한(논리 쪽 대비)
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


def _running(doc: Document) -> tuple[set[str], set[str], set[str]]:
    """되풀이 머리말 — 숫자를 뺀 글이 두 쪽 이상에 나온 머리말('붙임 1 사업계획서'·'… (2/3)' 를 쪽마다 찍은 문서)과 그 꼬리표.
    꼬리표만 든 머리말('붙임 1')은 숫자를 가려 세지 않는다 — 번호가 그 붙임의 이름이라 숫자를 가리면 붙임마다 같은 글이 된다
    (실측 test3.pdf — 쪽마다 꼬리표와 제목을 따로 머리말로 잡아 '붙임 #' 이 다섯 쪽에 되풀이된 것으로 보였다). 번호까지 같은
    꼬리표만 든 머리말이 두 쪽 이상이면 쪽마다 찍은 꼬리표다(셋째 값)."""
    pages = [{_masked(b.text): b.text for b in p.blocks if b.type == "header" and b.text
              and not APPENDIX_TAG.fullmatch(" ".join(b.text.split()))} for p in doc.pages]
    seen = Counter(k for p in pages for k in p)
    run = {k for k, n in seen.items() if n >= 2}
    tags = Counter(k for p in doc.pages for k in {_key(b.text) for b in p.blocks if b.type == "header" and b.text
                                                  and APPENDIX_TAG.fullmatch(" ".join(b.text.split()))})
    return run, {t for p in pages for k, text in p.items() if k in run and (t := _tag_key(text))}, \
        {k for k, n in tags.items() if n >= 2}


def _beside_title(page: Page, tag: Block, live: list[Block], run: set[str]) -> Block | None:
    """꼬리표 블록 오른쪽 같은 줄의 제목 블록 — 한 줄·짧고 항목·번호·장 제목·표·그림 제목·붙임 목록 꼴이 아니며, 상자가
    꼬리표와 조금 겹쳐도 오른쪽으로 더 뻗으면. 가장 가까운 것."""
    width = page_size(page)[0]
    tx0, ty0, tx1, ty1 = local(page, tag.bbox)
    right = []
    for b in live:
        if b is tag or b.merged_into is not None or b.type not in ("header", "paragraph_title", "text", "figure_title") \
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
    return min(right, key=lambda r: r[0])[1] if right else None


def _titled(b: Block) -> bool:
    """붙임 번호가 '이미 나온' 것으로 셀 블록 — 제목 라벨, 또는 꼬리표로 시작하는 글. 본문의 붙임 목록 줄('붙임 1. … 1부.')과
    참조 줄('<붙임 2 참고>')은 아니다."""
    s = " ".join((b.text or "").split())
    return b.type in ("paragraph_title", "doc_title") \
        or b.type == "text" and bool(_TAG_HEAD.match(s)) and not _ATTACH_LIST.search(s) and not _REF.search(s)


def _tag_titles(page: Page, running: tuple[set[str], set[str], set[str]], seen: set[str], titles: set[str]) -> None:
    """'붙임 1' 꼬리표와 그 붙임의 제목 — 쪽 맨 위 제목 상자를 검출기가 머리말(header)로 잡아 버리던 것(실측 test2.pdf, 쪽마다
    글이 다르다). 머리말이 꼬리표로 시작하고 그 붙임 번호가 처음 나온 것(앞쪽·이 쪽 제목에도, 되풀이 머리말에도 없음)이면
    제목으로 돌린다 — 꼬리표 뒤 제목 글이 앞쪽 제목·머리말과 같아도 되풀이다. 쪽마다 찍은 꼬리표만 든 머리말은 옆에 합칠
    제목이 있을 때만 돌린다('붙임 3' 머리말이 문서 전체의 뿌리가 되지 않게). 꼬리표 블록 오른쪽 같은 줄 제목(_beside_title)의
    글은 꼬리표 블록에 합친다. 문서 제목(doc_title)은 문서 제목으로 둔다. seen·titles 에 이 쪽
    꼬리표·제목 글(꼬리표를 뗀 _key)을 더한다."""
    run, run_tags, stamped = running
    live = [b for b in page.blocks if b.merged_into is None and (b.text or "").strip()]
    here = {k for b in live if b.keep and _titled(b) and (k := _tag_key(b.text))}
    for b in live:
        s = " ".join(b.text.split())
        if b.type == "header" and (k := _tag_key(s)) and k not in seen | here | run_tags and _masked(s) not in run \
                and "\n" not in b.text.strip() and (m := _TAG_HEAD.match(s)) and not _ATTACH_LIST.search(s) \
                and len(rest := _key(s[m.end():])) <= TAG_TITLE_MAX \
                and (rest not in titles if rest else _key(s) not in stamped or _beside_title(page, b, live, run)):
            b.provenance = (b.provenance or []) + [{"rule": "appendix_title", "type": b.type}]
            b.type = "paragraph_title"
    for tag in live:
        if not tag.keep or tag.merged_into is not None or tag.type not in ("paragraph_title", "doc_title", "text") \
                or not APPENDIX_TAG.fullmatch(" ".join(tag.text.split())):
            continue
        if (b := _beside_title(page, tag, live, run)) is None:
            continue
        tag.text = " ".join(tag.text.split()) + " " + " ".join(b.text.split())
        tag.bbox = (min(tag.bbox[0], b.bbox[0]), min(tag.bbox[1], b.bbox[1]),
                    max(tag.bbox[2], b.bbox[2]), max(tag.bbox[3], b.bbox[3]))
        tag.type = "paragraph_title" if tag.type == "text" else tag.type
        tag.provenance = (tag.provenance or []) + [{"rule": "appendix_title", "with": b.id, "type": b.type}]
        b.keep, b.drop_reason, b.merged_into, b.text = False, "merged", tag.id, None   # 글자는 대표에만(병합 묶음 규약)
    seen.update(k for b in page.blocks if b.keep and b.merged_into is None and _titled(b) and (k := _tag_key(b.text)))
    for b in page.blocks:                              # 앞쪽 제목·머리말 글 — 첫 쪽에 꼬리표 없이 제목만 둔 붙임의 뒤쪽 머리말
        if b.merged_into is None and b.text and (b.type == "header" or b.keep and b.type in ("paragraph_title", "doc_title")):
            s = " ".join(b.text.split())
            if t := _key(s[m.end():] if (m := _TAG_HEAD.match(s)) else s):
                titles.add(t)


def _table_name(b: Block, rest: list[Block], where: dict[str, Page]) -> bool:
    """표·그래프 바로 위 짧은 □ 이름 줄인가 — 한 줄·짧고 문장·'항목: 값'·마침표 끝·범례·체크 상자('■ 2026년 ■ 평년'·'□ 신규
    □ 변경') 꼴이 아니며, 뒤 블록(rest) 가운데 '(단위: …)' 줄을 건너뛴 첫 블록이 표·그림이고 줄이 그 왼쪽 끝 바깥이나 같은
    들여쓰기에서 시작한다(그래프 안쪽에서 시작하는 범례 칸 '■ 일강수량(mm)' 은 아니다)."""
    s = " ".join((b.text or "").split())
    m = _NAME.match(s)
    if not m or "\n" in (b.text or "").strip() or len(_MARKS.findall(s)) > 1 or len(_key(s[m.end():])) > NAME_MAX \
            or sentence_like(s) or re.search(r"[:：]|[.。]$", s):
        return False
    nxt = next((o for o in rest if not _unit_of(o)), None)
    if nxt is None or nxt.type not in ("table", "image", "chart"):
        return False
    page, other = where[b.id], where[nxt.id]
    return local(page, b.bbox)[0] - local(other, nxt.bbox)[0] <= NAME_INSET * page_size(page)[0]


def _name_titles(doc: Document) -> None:
    """표·그래프 바로 위 짧은 □ 이름 줄('□ 평년 대비 기상요소 값') — 검출기가 그 표·그림의 제목(figure_title)이나 글로
    잡는다(실측 test3.pdf). 같은 구역(제목 블록 사이)의 □ 줄이 모두 그런 이름 줄일 때만 문단 제목으로 돌린다 — '□ 문제점'·
    '□ 개선 방향' 같은 □ 항목·□ 문단과 함께 있으면 표 위 □ 줄도 그 형제 항목이고, 그림 제목 라벨이면 그 표·그림의 제목이다.
    □ 줄은 범례 기호가 하나이고 글자층 글자가 없다는 표시가 없는 줄 — 그래프에서 떼어 낸 줄은 tables 가 그림 글로 묶는다."""
    order = [b for p in doc.pages for b in p.blocks if b.merged_into is None and b.type not in DROP_LABELS]
    where = {b.id: p for p in doc.pages for b in p.blocks}
    sections, cur = [], []
    for b in order:
        if b.type in ("paragraph_title", "doc_title"):
            sections.append(cur)
            cur = []
        else:
            cur.append(b)
    sections.append(cur)
    for sec in sections:
        boxes = [i for i, b in enumerate(sec) if b.type in ("figure_title", "text") and _NAME.match(b.text or "")
                 and len(_MARKS.findall(b.text or "")) == 1
                 and not any(p.get("rule") == "no_layer_chars" for p in b.provenance or [])]   # 그림에 그려진 줄은 그림 글이다
        names = [i for i in boxes if _table_name(sec[i], sec[i + 1:], where)]
        if len(names) < len(boxes):
            continue
        for i in names:
            b = sec[i]
            b.provenance = (b.provenance or []) + [{"rule": "name_title", "type": b.type}]
            b.type = "paragraph_title"


def _logo_row(page: Page) -> None:
    """로고 줄의 작은 그림 — 검출기가 로고(header_image·footer_image)로 잡은 그림과 같은 줄에 놓인 비슷한 높이의 작은 그림도
    로고다(실측 test3.pdf — 보도자료 끝 정책브리핑·표어·공공누리 표시 셋 중 둘을 그림으로 잡았다)."""
    w, h = page_size(page)
    logos = [local(page, b.bbox) for b in page.blocks if b.drop_reason == "logo" and b.merged_into is None]
    for b in page.blocks:
        if b.type != "image" or not b.keep or b.merged_into is not None:
            continue
        x0, y0, x1, y1 = local(page, b.bbox)
        if x1 - x0 > LOGO_W * w or y1 - y0 > LOGO_H * h:
            continue
        if any(min(y1, ly1) - max(y0, ly0) >= 0.5 * min(y1 - y0, ly1 - ly0) and 0.5 <= (y1 - y0) / max(ly1 - ly0, 1e-6) <= 2
               for _, ly0, _, ly1 in logos):
            b.keep, b.drop_reason = False, "logo"
            b.provenance = (b.provenance or []) + [{"rule": "logo_row"}]


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
    _name_titles(doc)
    for page in doc.pages:
        for b in page.blocks:
            if b.merged_into is not None or b.type not in drop:
                continue
            if b.type == "aside_text" and _keep_aside(b, cfg):
                b.aside = True
                b.provenance = (b.provenance or []) + [{"rule": "aside_keep"}]
                continue
            b.keep, b.drop_reason = False, drop[b.type]
        _logo_row(page)
    return doc
