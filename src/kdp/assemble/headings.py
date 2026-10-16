"""제목 수준·섹션 경로(기획서 4.3 규칙 3·4).

수준은 한국식 번호(제n장·Ⅰ.·1.·가.·1)·(1)·①·❶·㉮, 보고서체 □·○ 등)를 먼저 본다 — 문서에서 처음 나온 차례로 체계를 쌓고,
이미 쌓인 체계가 다시 나오면 그 수준으로 돌아간다(스택). 번호 없는 제목은 책갈피, 그다음 글자 높이 군집으로 정하고 확신이
없으면 비운다(docling 도 틀린 수준보다 없는 수준을 택한다).

- 문서 제목(번호 없는 doc_title)은 수준 0 이고 다른 블록의 경로에 넣지 않는다 — 문서마다 하나라 청크를 가르지 못하고,
  표지가 있는 문서는 제목이 버린 쪽에 있어 경로에 없으므로 넣으면 표지 유무에 따라 경로 모양이 달라진다. 문서 제목은 meta.title 몫.
  경로가 쌓인 뒤 처음 보는 문서 제목은 이어 붙은 둘째 문서(보고서 + 참고자료 등)의 시작이라 경로를 새로 시작한다.
- 수준을 모르는 제목은 지금 경로 끝에 잎으로 붙였다가 다음 제목이 나오면 뗀다 — 아는 구조를 흔들지 않으면서 가장 가까운
  제목을 청크에 남긴다.
"""
from __future__ import annotations

import math
import re
import unicodedata

from ..classify.blocks import sentence_like
from ..classify.pages import is_toc_title
from ..geometry import local, page_size
from ..schema import Block, Document, Page
from .order import has_output
from .tables import CAPTION, CONT

TITLE_TYPES = {"doc_title", "paragraph_title"}
FIGS = ("table", "image", "chart")   # 캡션이 붙는 블록 — 그 바로 위·아래 figure_title 은 캡션 자리
APPENDIX = "붙임"                 # 경로를 새로 시작한 뿌리(간지·붙임·부록)의 체계 이름
HEADER_MAX_CHARS = 40            # 경로로 쓰는 머리말 글자 수 상한 — 긴 머리말은 장 제목이 아니라 문서명·기관명 띠
APPENDIX_MAX_CHARS = 40          # 번호 없는 '붙임 …' 제목 글자 수 상한 — 긴 것은 본문 문장('붙임 n' 라벨 뒤 제목은 길이를 보지 않는다)
HEIGHT_GAP = 0.1                 # 글자 높이 군집을 가르는 차이(10%) — 제목 글자 크기는 보통 2pt(약 14%) 이상 다르다
LINE_EM_MAX = 2.0                # 블록 높이가 글자 폭으로 잰 글자 크기의 이 배를 넘으면 줄바꿈 없는 여러 줄 — 높이를 쓰지 않는다
X_TOL = 0.008                    # 같은 들여쓰기로 보는 x 차이(논리 쪽 너비 대비, A4 약 5pt) — lists.INDENT_TOL 과 같다

_SEQ = "가나다라마바사아자차카타파하"
_ROMAN_CHARS = "ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ"
_ROMAN = {"I": "Ⅰ", "II": "Ⅱ", "III": "Ⅲ", "IV": "Ⅳ", "V": "Ⅴ", "VI": "Ⅵ", "VII": "Ⅶ", "VIII": "Ⅷ", "IX": "Ⅸ",
          "X": "Ⅹ", "XI": "Ⅺ", "XII": "Ⅻ"}
_AR = "|".join(sorted(_ROMAN, key=len, reverse=True))
# (체계, 줄 머리 정규식) — 앞에서부터 맞춘다. 그룹 1 = 번호. '2026년'·'2025. 3. 15.'·'Ⅱ단계'·'(사)'·'1.5배' 는 번호가 아니다
_NUMBERING = tuple((k, re.compile(p)) for k, p in (
    ("장", r"제\s*(\d{1,3}|[Ⅰ-Ⅻ])\s*([편장절관조])(?=[\s(（.:：]|$)"),          # 체계 = 편·장·절·관·조
    ("Ⅰ-1", rf"([Ⅰ-Ⅻ]|{_AR})\s*-\s*(\d{{1,2}}(?:\s*-\s*\d{{1,2}})?)\s*\.?(?=\s|$)"),
    ("Ⅰ", r"([Ⅰ-Ⅻ])\s*(?:[.)]|(?=\s|$))"),
    ("Ⅰ", rf"({_AR})\.(?=\s|$)"),
    ("1.1", r"(\d{1,2}(?:[.\-]\d{1,2}){1,3})\.?(?=\s|$)"),                     # 1.1·1-1·1.1.1
    ("1.", r"(\d{1,2})\.(?![\d.])(?!\s*\d{1,2}\s*\.)"),                       # '3. 15.' 같은 날짜는 뺀다
    ("1.", r"([⒈-⒛])"),
    ("가.", rf"([{_SEQ}])\.(?![\d.])"),
    ("1)", r"(\d{1,2})\)"),
    ("가)", rf"([{_SEQ}])\)"),
    ("(1)", r"[(（](\d{1,2})[)）]"),
    ("(1)", r"([⑴-⒇])"),
    ("(가)", rf"[(（]([{_SEQ}])[)）](?=\s|$)"),
    ("(가)", r"([㈀-㈛])"),
    ("①", r"([①-⑳㉑-㉟㊱-㊿➀-➉])"),                                          # ➀ 은 글자층의 다른 원 숫자(실측 test2.pdf)
    ("❶", r"([❶-❿➊-➓])"),                                                  # 검은 원 숫자 — 흔히 ① 과 다른 단계
    ("㉮", r"([㉠-㉻])"),
))
_TITLE_NUM = re.compile(r"(\d{1,2})\s+(?=[가-힣])")   # 제목만 — Word 개요 번호 '1 서론'('1.1 배경' 의 위). 글 블록은 '1 인당 …' 이 걸린다
# 보고서체 기호 — 같은 글자가 이어지면('○○ 사업' 자리표시, '··' 점선) 기호가 아니다
_BULLET = re.compile(r"([□■◻◼▣❑❒☐○●◦❍◎•·∙・‧⋅▪▫\-–—―‐−▶►▷▸➢➤→⇒◆◇❖◈※*★☆✔✓☑☒√])(?!\1)\s*(?=\S)"
                     r"|(ㅇ|[oO])\s+(?=[가-힣])")
_SAME = {"ㅇ": "○", "o": "○", "O": "○", "–": "-", "—": "-", "―": "-", "‐": "-", "−": "-",
         "∙": "·", "・": "·", "‧": "·", "⋅": "·"}                     # 같은 기호의 변형
_ORDER = "□○-·"                  # 보고서체 기호 차례(□ → ○ → - → ·) — 글자 모양이 같아도 처음 나온 차례대로 쌓는다(_place)
_MISREAD = "☐"                   # VLM 이 □·○ 를 잘못 읽어 낸 기호(실측) — 뒤에 바로 읽힌 보고서체 기호가 그 자리 형제가 된다(_place)
# 순서가 문서마다 거의 같은 위쪽 체계 — 처음 나온 이 체계가 자기보다 아래 서열 밑으로 들어가지 않게 한다(place)
RANK = {k: i for i, k in enumerate(("편", "장", "절", "관", "조", "Ⅰ", "Ⅰ-1", "Ⅰ-1-1",
                                     "1.", "1.1", "1.1.1", "1.1.1.1", "가."))}
_APPENDIX = re.compile(r"[\[<〈【(「『]?\s*(?:붙\s*임|별\s*첨|첨\s*부|부\s*록|별\s*지)(?![가-힣])")
_ATTACH_LIST = re.compile(r"\d\s*부\s*\.|\d\s*부\s*$|끝\s*\.?\s*$")   # '붙임 1. 계획 1부. 끝.' — 붙임 목록(시행규칙 제4조④)
_LABEL = re.compile(r"[\[<〈【(「『]?\s*(?:붙\s*임|별\s*첨|첨\s*부|부\s*록|별\s*지)\s*(?:제\s*)?(?:\d{1,2}|[Ⅰ-Ⅻ]|[A-Z])?"
                    r"\s*(?:호\s*)?(?:서\s*식\s*)?[\]>〉】)」』]?\s*\.?")
_LABEL_TITLE = re.compile(r"[\[<〈【(「『]?\s*(?:붙\s*임|별\s*첨|첨\s*부|부\s*록|별\s*지)\s*(?:제\s*)?(?:\d{1,2}|[Ⅰ-Ⅻ]|[A-Z])"
                          r"\s*(?:호\s*)?(?:서\s*식\s*)?[\]>〉】)」』]?\s*\.?\s")   # '[붙임 2] 제목' — 번호 붙은 라벨 다음 공백
_LEADER = re.compile(r"(?:[.…·‥_]\s*){3,}\S{0,6}\s*$|\S\s+\d{1,4}\s*$")       # 목차 줄 끝(점선·쪽번호)
_UNIT_ROW = re.compile(r"[\s※\[(（<〈]*단\s*위\s*[:：]")                      # 표 제목과 표 사이 '(단위: 천원)' 줄
_CONT_END = re.compile(r"[(\[<〈【（]?\s*계\s*속\s*[)\]>〉】）]?\s*$")             # 제목 끝 '(계속)'(tables 와 같은 꼴) — '계속사업' 은 아니다


def _decor(s: str, wide: bool = True) -> bool:
    """같은 꾸밈 기호로 시작하고 끝나는 줄('- 아 래 -'·'- 3 -'·'- 긴 부제 -'·'◈ 2026년 업무보고 ◈') — 기호 항목이 아니다.
    wide 가 아니면(문단 제목) '-' 류만 — '◆ 사업 개요 ◆' 는 양쪽을 꾸민 기호 제목이다."""
    m = _BULLET.match(s)
    c = _SAME.get(m.group(1), m.group(1)) if m and m.group(1) else None
    return bool(c) and len(s) > m.end() and _SAME.get(s[-1], s[-1]) == c and (wide or c == "-")


def _parse(text: str | None, title: bool = False, wide: bool = True):
    """줄 머리 → (체계, 번호, 끝 위치, 번호인지). title 이면 제목에만 쓰는 번호 꼴도 본다. wide 는 _decor 참고."""
    s = (text or "").lstrip().split("\n", 1)[0]
    if _decor(s.rstrip(), wide):
        return None
    for kind, pat in _NUMBERING + ((("1.", _TITLE_NUM),) if title else ()):
        if m := pat.match(s):
            tok = re.sub(r"\s", "", m.group(1))
            if kind == "장":
                kind = m.group(2)
            elif kind == "Ⅰ-1":
                sub = re.sub(r"\s", "", m.group(2))
                kind, tok = ("Ⅰ-1-1" if "-" in sub else "Ⅰ-1"), _ROMAN.get(tok, tok) + "-" + sub
            elif kind == "1.1":
                kind = "1" + ".1" * len(re.findall(r"[.\-]", tok))
            return kind, _ROMAN.get(tok, tok), m.end(), True
    if m := _BULLET.match(s):
        c = m.group(1) or m.group(2)
        return _SAME.get(c, c), _SAME.get(c, c), m.end(), False
    return None


def marker(text: str | None, title: bool = False, wide: bool = True) -> tuple[str, str] | None:
    """줄 머리 번호·기호 → (체계, 번호 글자). 기호면 번호 글자도 기호."""
    p = _parse(text, title, wide)
    return p[:2] if p else None


def numbering(text: str | None, title: bool = False) -> tuple[str, str] | None:
    """줄 머리 번호(기호는 뺀다) → (체계, 번호 글자)."""
    p = _parse(text, title)
    return p[:2] if p and p[3] else None


def _ordinal(tok: str) -> tuple[int, ...]:
    """번호 글자 → 차례 값('Ⅱ-1' → (2, 1)). 모르는 꼴이면 빈 튜플."""
    out = []
    for part in re.split(r"[.\-]", tok):
        if part.isdecimal():
            out.append(int(part))
        elif len(part) == 1 and part in _ROMAN_CHARS:
            out.append(_ROMAN_CHARS.index(part) + 1)
        elif len(part) == 1 and part in _SEQ:
            out.append(_SEQ.index(part) + 1)
        elif len(part) == 1 and (v := unicodedata.numeric(part, None)) is not None and v == int(v):
            out.append(int(v))                     # '⒈'·'⑴'·'①' — isdigit() 는 참이어도 int() 가 못 읽는다
        else:
            return ()
    return tuple(out)


def place(kinds: list, kind: str) -> int:
    """번호 체계 kind 를 놓을 스택 자리(0부터) — 그 자리부터 아래는 걷어 낸다. 이미 쌓인 체계면 그 자리, 처음 나왔으면 맨 아래
    (처음 나온 차례). 단 처음 나온 체계가 RANK 체계이고 스택에 그보다 아래 서열의 RANK 체계가 있으면 그 자리로 올린다 —
    앞에 '1.'·'가.' 만 있던 문서에서 처음 나온 'Ⅰ.' 을 '가.' 밑에 두지 않는다."""
    if kind in kinds:
        return kinds.index(kind)
    if kind in RANK:
        for i, k in enumerate(kinds):
            if RANK.get(k, -1) >= RANK[kind]:
                return i
    return len(kinds)


def _bullet(text: str) -> bool:
    """줄 머리가 기호(번호가 아닌)인 제목."""
    p = _parse(text, title=True, wide=False)
    return p is not None and not p[3]


def _place(stack: list, kind: str, text: str, h: float | None, x: float | None) -> tuple[int, str]:
    """place() 에 글자 모양을 더한다(스택 항목 = (체계, 글자, 줄 높이, x)) → (자리, 쌓을 체계). 처음 나온 RANK 체계는 그보다
    확실히 작은 글자의 RANK 밖 제목('□ 요약' 상자·번호 없는 제목) 위로 올린다. 처음 나온 낯선 기호(☐ 등)는 글자 높이·x 가 같은
    기호 제목이 스택에 하나뿐이면 그 형제로 두고 그 자리 기호를 물려받는다 — VLM 이 ○·□ 를 ☐ 로 읽는다(실측). 보고서체
    기호(□ ○ - ·)는 처음 나온 차례 그대로 — 다만 글자 높이·x 가 같은 스택 제목이 잘못 읽힌 ☐ 하나뿐이면 그 형제(제 기호로)."""
    kinds = [s[0] for s in stack]
    t = place(kinds, kind)
    if kind in kinds or h is None:
        return t, kind
    if kind in RANK:
        return next((i for i, (k, _, hh, _) in enumerate(stack[:t])
                     if k not in RANK and k != APPENDIX and hh is not None and h > hh * (1 + HEIGHT_GAP)), t), kind
    if _bullet(text):
        report = kind in _ORDER
        same = [i for i, (k, s, hh, xx) in enumerate(stack)
                if hh is not None and (k in _MISREAD if report else _bullet(s))
                and max(h, hh) <= min(h, hh) * (1 + HEIGHT_GAP) and abs(x - xx) <= X_TOL]
        if len(same) == 1:
            return same[0], (kind if report else stack[same[0]][0])
    return t, kind


def _floor(kinds: list, kind: str) -> int:
    """스택에서 kind 보다 윗 서열인 RANK 체계 바로 아래 자리 — 책갈피 힌트도 이보다 얕게 놓지 않는다(간지·머리말이 넣은 장을
    절 책갈피가 지우지 않게)."""
    r = RANK.get(kind, len(RANK))
    return max((i + 1 for i, k in enumerate(kinds) if k in RANK and RANK[k] < r), default=0)


def _title_marker(b: Block, text: str) -> tuple[str, str] | None:
    """제목 블록의 줄 머리 번호·기호. 문서 제목은 번호만 본다 — 꾸밈 기호('◈ …')와 점 날짜('6.25 …'·'12.31 …' 의 '1.1' 꼴)로
    시작하는 문서 제목도 섹션이 아니라 문서 제목(수준 0)이다. 문단 제목의 양쪽 꾸밈('◆ 사업 개요 ◆')은 기호다."""
    if b.type != "doc_title":
        return marker(text, title=True, wide=False)
    m = numbering(text, title=True)
    return None if m is None or m[0].startswith("1.1") else m


def _clean(s: str | None) -> str:
    return " ".join((s or "").split())


def _key(s: str) -> str:
    """제목 맞춤용 — 공백·점·쌍점을 뺀다."""
    return re.sub(r"[\s.·:：]", "", unicodedata.normalize("NFC", s))


def doc_title_key(text: str | None) -> str:
    """문서 제목 맞춤 글자 — 쪽마다 다시 찍은 같은 문서 제목('… (계속)' 포함)을 한 제목으로 본다(lists 도 같은 기준)."""
    return _key(_CONT_END.sub("", _clean(text)))


def _bare(s: str) -> str:
    """번호를 뗀 제목 — 책갈피가 번호 없이 적힌 경우."""
    p = _parse(s, title=True)
    return s.lstrip()[p[2]:] if p else s


def _cont_key(s: str) -> str:
    """'(계속)' 제목 맞춤 글자 — 끝 '(계속)'·줄 머리 번호·기호를 뗀다."""
    return _key(_bare(_CONT_END.sub("", s)))


def _candidates(doc: Document) -> set[str]:
    """제목 후보 — keep 쪽의 keep doc_title·paragraph_title 과 번호로 시작하는 figure_title(구역 제목에도 붙는다, 실측 —
    표·그림·차트 바로 위·아래 것은 그 캡션이다). 병합 구성원·'(계속)' 표시·문장 꼴 문단 제목(classify_blocks 와 같은 기준 —
    문서 제목은 표어·보도자료 꼴이 있어 거르지 않는다)은 빼고, 표 제목도 뺀다 — 표 제목 라벨이 구역 제목에도 붙듯 그 반대도
    있어 경로에 '<표 1> …' 이 들어간다. 표에 이은 블록(caption_id)과, 표와 멀어 잇지 못했거나 다음 쪽에 되풀이한 표 제목
    ('<표 n>'·'[도표 n]'·'표 n.' 꼴이고 '(단위: …)' 줄을 건너 바로 다음이 표)."""
    caps = {b.caption_id for p in doc.pages for b in p.blocks if b.caption_id}
    out = set()
    for page in doc.pages:
        if not page.keep:
            continue
        live = [b for b in page.blocks if b.keep and b.merged_into is None and has_output(b)]
        for i, b in enumerate(live):
            text = (b.text or "").strip()
            fig = b.type == "figure_title"
            if (b.type not in TITLE_TYPES and not fig) or not text or b.id in caps \
                    or CONT.fullmatch("".join(text.split())) \
                    or (b.type != "doc_title" and sentence_like(text) and not _LABEL_TITLE.match(text)):   # '붙임 2 …' 제목은 문장 꼴이어도 뿌리
                continue
            nxt = next((x for x in live[i + 1:] if not _UNIT_ROW.match(x.text or "")
                        and not CONT.fullmatch("".join((x.text or "").split()))), None)
            if fig and (not numbering(text, title=True) or (i and live[i - 1].type in FIGS)
                        or (nxt is not None and nxt.type in FIGS)):
                continue
            if CAPTION.match(text) and nxt is not None and nxt.type == "table":
                continue
            out.add(b.id)
    return out


def _appendix(text: str) -> bool:
    """'붙임 n …' 제목 — 번호 붙은 라벨 다음 제목은 길이와 상관없이, 그 밖에는 짧은 것만(긴 것은 본문 문장). 붙임 목록은 아니다."""
    return bool(_APPENDIX.match(text)) and not _ATTACH_LIST.search(text) and (
        bool(_LABEL_TITLE.match(text)) or len(text) <= APPENDIX_MAX_CHARS)


def _units(s: str) -> float:
    """글자 폭 합(전각 1) — 한 줄 글자 크기를 블록 너비로 어림한다."""
    return sum(0.3 if c.isspace() else 1.0 if ord(c) >= 0x1100 else 0.55 for c in s)


def _line_h(page: Page, b: Block) -> float | None:
    """한 줄 글자 높이(pt — 쪽 크기로 나누지 않아 세로 문서 속 가로 쪽·A3 쪽도 같은 잣대). 모아찍기 칸은 칸 축소 비율
    √(열·행)로 되돌린다. 줄바꿈 없이 여러 줄이 든 블록은 None."""
    x0, y0, x1, y1 = local(page, b.bbox)
    lines = [ln.strip() for ln in b.text.strip().split("\n")]
    h = (y1 - y0) / len(lines)
    em = (x1 - x0) / max(1.0, max(_units(ln) for ln in lines))
    return None if h > LINE_EM_MAX * em else h * (math.sqrt(page.n_up.cols * page.n_up.rows) if page.n_up else 1.0)


def _left(page: Page, b: Block) -> float:
    """블록 왼쪽 x(논리 쪽 너비 대비)."""
    return local(page, b.bbox)[0] / page_size(page)[0]


def _divider_title(page: Page) -> str:
    """간지 제목 — 제목 블록, 없으면 글 블록(머리말·쪽번호 같은 버린 블록은 뺀다)."""
    blocks = [b for b in page.blocks if b.keep and b.merged_into is None and (b.text or "").strip()]
    titles = [b for b in blocks if b.type in TITLE_TYPES] or [b for b in blocks if b.type == "text"]
    return " ".join(_clean(b.text) for b in titles)


def _chapter_title(page: Page) -> str:
    """소목차가 있어 목차로 버린 장 첫 쪽의 장 제목 — 목차 제목('목 차')이 없는 쪽에서 쪽번호·점선 없는 RANK 번호 제목이
    하나뿐일 때만(목차 줄을 제목으로 잡은 진짜 목차 쪽은 넘긴다)."""
    blocks = [b for b in page.blocks
              if b.keep and b.merged_into is None and b.type in TITLE_TYPES and (b.text or "").strip()]
    if any(is_toc_title(b) for b in blocks):
        return ""
    found = [t for t in (_clean(b.text) for b in blocks)
             if (m := numbering(t, title=True)) and m[0] in RANK and not _LEADER.search(t)]
    return found[0] if len(found) == 1 else ""


def _toc_levels(doc: Document, cands: set[str], toc: list[dict]) -> dict[str, int]:
    """책갈피와 글자가 맞는 제목 → 책갈피 수준(쪽이 ±1 안, 가까운 쪽 먼저). 번호까지 같은 맞춤이 있으면 번호를 뗀 맞춤은
    보지 않고('제1절 개요' 가 '제1장 개요' 에 붙지 않게), 가장 가까운 맞춤들의 수준이 갈리면 비운다. 수준 1 은 책갈피 전체의
    맨 위로 정한다 — 맞은 것만으로 정하면 장이 간지·머리말에만 있을 때 절이 1 로 당겨진다. pypdfium2 수준은 0 부터이고,
    맨 위가 맞는 제목 없는 뿌리 하나뿐(문서 제목)이면 그 아래가 1 이다."""
    marks = [(t["level"], t.get("page"), _key(t["title"]), _key(_bare(t["title"])))
             for t in toc if t.get("title") and t.get("level") is not None]
    found, hit = {}, set()
    for page in doc.pages:
        for b in page.blocks:
            text = _clean(b.text)
            if b.id not in cands or (b.type == "doc_title" and _title_marker(b, text) is None):
                continue
            k, bare = _key(text), _key(_bare(text))
            near = [(abs(p - page.sheet_no) if p else 0, lv, i) for i, (lv, p, _, _) in enumerate(marks)
                    if p is None or abs(p - page.sheet_no) <= 1]
            same = [n for n in near if marks[n[2]][2] == k] or [n for n in near if bare and marks[n[2]][3] == bare]
            if same:
                d = min(n[0] for n in same)
                best = [(lv, i) for dist, lv, i in same if dist == d]
                if len({lv for lv, _ in best}) == 1:
                    found[b.id] = best[0][0]
                    hit.update(i for _, i in best)
    base = min((m[0] for m in marks), default=0)
    roots = [i for i, m in enumerate(marks) if m[0] == base]
    top = base + 1 if len(roots) == 1 and roots[0] not in hit else base
    return {i: lv - top + 1 for i, lv in found.items() if lv >= top}


def _section_header(h: Block, stack: list, titles: list[str], seen: set[str], last: dict[str, str]) -> bool:
    """규칙 4 — 장·절 번호 모양의 머리말이 지금 경로에도, 이 쪽 제목에도 없으면 앞 쪽에서 이어진 장이다.
    이 쪽에 같은 장 제목이 있으면 그 제목이 맡는다(머리말이 쪽 안 첫 제목을 보이는 문서 — 쪽 위 글은 아직 앞 장이다).
    경로에 같은 체계가 있으면 그보다 뒤 번호일 때만 — 구역이 바뀔 때만 고치는 머리말은 지난 장을 보인다. 그 자리가 체계를
    물려받은 번호 없는 제목('요약')이면 그 체계의 마지막 번호 제목(last)과 견주고, 없으면 새 장이다. 이미 경로에 넣었거나
    머리말로 본 글(seen)은 다시 쓰지 않는다 — 붙임·둘째 문서로 경로를 새로 시작한 뒤에도 남은 본문 마지막 장 머리말."""
    if h.type != "header" or h.merged_into is not None or not h.text:
        return False
    text = _clean(h.text)
    m = numbering(text, title=True)
    if m is None or m[0] not in RANK or len(text) > HEADER_MAX_CHARS or _key(text) in seen:
        return False
    if any(_key(t) == _key(text) or numbering(t, title=True) == m for t in [s[1] for s in stack] + titles):
        return False
    same = next((s[1] for s in stack if s[0] == m[0]), None)
    if same is not None and numbering(same, title=True) is None:
        same = last.get(m[0])
    if same is None:
        return True
    o = numbering(same, title=True)
    return bool(o and o[0] == m[0] and _ordinal(o[1]) and _ordinal(m[1]) > _ordinal(o[1]))


def _small(h: float | None, stack: list) -> bool:
    """제목 글자가 지금 경로의 가장 큰 제목보다 확실히(HEIGHT_GAP 넘게) 작다 — 높이를 모르면 아니다."""
    hs = [s[2] for s in stack if s[2] is not None]
    return h is not None and bool(hs) and h * (1 + HEIGHT_GAP) < max(hs)


def _page_top(page: Page, b: Block, cands: set[str]) -> bool:
    """쪽 머리 제목 — 이 쪽 첫 제목 후보이고, 앞선 출력 블록은 그 위에 있으면서 가로로 겹치지 않는 띠 글(문서명·보안 등급처럼
    머리말로 안 잡힌 것)뿐이다. 본문 글 뒤에 오는 같은 이름 제목은 새 절이다."""
    x0, y0, x1, _ = local(page, b.bbox)
    for o in page.blocks:
        if o is b:
            return True
        if not (o.keep and o.merged_into is None and has_output(o)):
            continue
        a0, _, a1, a3 = local(page, o.bbox)
        if o.id in cands or a3 > y0 + 3 or (a1 > x0 and a0 < x1):
            return False
    return False


def _continued(text: str, stack: list, leaf: list[str]) -> list[str] | None:
    """끝이 '(계속)' 이고 번호·기호를 뗀 글이 앞 제목(잎, 그다음 깊은 스택부터)과 같은 제목 → 그 제목의 경로. 아니면 None —
    쪽을 넘긴 구역 제목의 되풀이('□ 사업비 집행 현황(계속)')는 새 제목이 아니다."""
    if not _CONT_END.search(text) or not (k := _cont_key(text)):
        return None
    if leaf and _cont_key(leaf[0]) == k:
        return [s[1] for s in stack] + leaf
    i = next((i for i in range(len(stack) - 1, -1, -1) if _cont_key(stack[i][1]) == k), None)
    return None if i is None else [s[1] for s in stack[:i + 1]]


def _walk(doc: Document, cands: set[str], hints: dict[str, int], heights: dict[str, tuple[int, str | None]]):
    """읽기 순서로 제목을 따라가며 블록마다 (수준, 근거)·경로를 모은다. 경로로 쓴 머리말도."""
    stack: list[tuple[str | None, str, float | None, float | None]] = []   # (번호 체계, 제목 글자, 줄 높이, x) — 뒤로 갈수록 깊다
    leaf: list[str] = []                           # 수준을 모르는 제목 — 다음 제목이 나오면 뗀다
    levels: dict[str, tuple[int | None, str | None]] = {}
    paths: dict[str, list[str]] = {}
    used: set[str] = set()
    seen: set[str] = set()                         # 경로에 넣었거나 머리말로 본 글(_key)
    docs: set[str] = set()                         # 본 문서 제목(doc_title_key) — 쪽마다 다시 찍은 문서 제목은 경로를 끊지 않는다
    last: dict[str, str] = {}                      # 체계마다 마지막으로 경로에 넣은 번호 제목(_section_header)
    matched = {_key(_clean(b.text)) for p in doc.pages for b in p.blocks if b.id in hints}   # 책갈피와 맞은 제목(_floor)

    def put(t: int, kind: str | None, text: str, h: float | None = None, x: float | None = None) -> int:
        del stack[t:]
        stack.append((kind, text, h, x))
        leaf.clear()
        seen.add(_key(text))
        if (n := numbering(text, title=True)) and n[0] == kind:
            last[kind] = text
        return t + 1

    prev_toc = False                               # 앞 쪽이 목차로 버린 쪽
    for page in doc.pages:
        if not page.keep:
            # '제2장' 간지와 소목차 때문에 목차로 버린 장 첫 쪽은 번호대로, '붙임 1'·'부록' 간지는 경로를 새로 시작. 여러 쪽
            # 목차의 이어진 쪽(앞 쪽도 목차)과 본문 제목이 아직 없는 앞머리의 목차 쪽은 장 첫 쪽으로 보지 않는다
            title = _divider_title(page) if page.page_class == "divider" else \
                _chapter_title(page) if page.page_class == "toc" and levels and not prev_toc else ""
            if title and (m := numbering(title, title=True)):
                put(place([s[0] for s in stack], m[0]), m[0], title)
            elif title:
                put(0, APPENDIX, title)
            prev_toc = page.page_class == "toc"
            continue
        prev_toc = False
        titles = [_clean(b.text) for b in page.blocks if b.id in cands]
        head = False                                   # 이 쪽 머리말이 경로를 고쳤다
        for h in page.blocks:
            if _section_header(h, stack, titles, seen, last):
                text = _clean(h.text)
                kind = numbering(text, title=True)[0]
                put(place([s[0] for s in stack], kind), kind, text)
                used.add(h.id)
                head = True
        seen.update(_key(_clean(h.text)) for h in page.blocks if h.type == "header" and h.text)
        for b in page.blocks:
            if not b.keep or b.merged_into is not None:
                continue
            text = _clean(b.text)
            if b.id in cands:
                h, x = _line_h(page, b), _left(page, b)
                m = _title_marker(b, text)
                if m is None and b.type == "doc_title" and _small(h, stack):
                    m = marker(text, title=True)       # 본문 중간 기호·'n.n' 제목에 문서 제목 라벨을 잘못 단 것 — 문단 제목처럼
                if (again := _continued(text, stack, leaf)) is not None:
                    levels[b.id] = (None, "repeat")    # '(계속)' 되풀이 — 스택을 그대로 두고 블록 경로는 원래 제목 경로
                    paths[b.id] = again
                    continue
                if _appendix(text):
                    levels[b.id] = (put(0, APPENDIX, text), "appendix")
                elif m is None and b.type == "doc_title":
                    # 경로가 쌓인 뒤 처음 보는 문서 제목 — 이어 붙은 둘째 문서. 붙임 뿌리와 이 쪽 머리말이 고친 경로는 두고,
                    # 지금 경로의 가장 큰 제목보다 확실히 작은 글자(본문 중간 제목에 문서 제목 라벨을 잘못 단 것)도 끊지 않는다
                    if (stack or leaf) and doc_title_key(text) not in docs and not head and not _small(h, stack):
                        del stack[1 if stack and stack[0][0] == APPENDIX else 0:]
                        leaf.clear()
                    docs.add(doc_title_key(text))
                    levels[b.id] = (0, "doc_title")    # 문서 제목 — 섹션이 아니다
                    paths[b.id] = [s[1] for s in stack] + [text]
                    continue
                elif m:
                    kinds = [s[0] for s in stack]
                    if m[0] in kinds and _key(stack[kinds.index(m[0])][1]) == _key(text) and _page_top(page, b, cands):
                        i = kinds.index(m[0])          # 쪽 머리에 다시 찍은 같은 제목 — 그 아래 경로를 지킨다
                        levels[b.id] = (i + 1, "marker")
                        paths[b.id] = [s[1] for s in stack[:i + 1]]
                        continue
                    if m[0] not in kinds and b.id in hints:    # 처음 나온 체계의 자리는 책갈피가 정한다 — 윗 서열 장은 두고
                        # RANK 밖 기호 제목은 책갈피와 맞지 않은 장(간지·머리말이 넣은 것)만 바닥 — 맞은 장과는 책갈피 수준끼리 견준다
                        base = kinds if m[0] in RANK else [None if _key(s[1]) in matched else s[0] for s in stack]
                        t = max(min(hints[b.id] - 1, len(stack)), _floor(base, m[0]))
                        levels[b.id] = (put(t, m[0], text, h, x), "toc")
                    else:
                        t, kind = _place(stack, m[0], text, h, x)
                        levels[b.id] = (put(t, kind, text, h, x), "marker")
                elif b.id in hints or b.id in heights:
                    lv, kind = (hints[b.id], None) if b.id in hints else heights[b.id]
                    t = min(lv - 1, len(stack))        # 그 자리의 번호 체계를 물려받는다 — 같은 체계 다음 번호가 오면 그 자리로.
                    kind = stack[t][0] if t < len(stack) else kind     # 스택이 얕으면 같은 높이 군집의 번호 체계를
                    levels[b.id] = (put(t, kind, text, h, x), "toc" if b.id in hints else "height")
                else:
                    leaf[:] = [text]
                    levels[b.id] = (None, None)
            elif b.type == "text" and _LABEL.fullmatch(text):
                levels[b.id] = (put(0, APPENDIX, text), "appendix")    # '붙임 1' 만 있는 글 블록
            paths[b.id] = [s[1] for s in stack] + leaf
    return levels, paths, used


def _height_levels(doc: Document, cands: set[str], first: dict) -> dict[str, tuple[int, str | None]]:
    """번호 없는 제목의 수준을 글자 높이 군집으로 — 군집 안 번호·책갈피 제목이 모두 한 수준일 때만(아니면 비운다).
    그 수준 번호 제목들의 체계가 하나면 함께 준다 — 문서 앞 '요약' 이 같은 높이의 'Ⅰ.' 장들을 자기 밑에 품지 않게."""
    rows = []
    for page in doc.pages:
        for b in page.blocks:
            lv, by = first.get(b.id, (None, None))
            if b.id in cands and by not in ("doc_title", "appendix") and (h := _line_h(page, b)) is not None:
                m = _title_marker(b, _clean(b.text)) if by in ("marker", "toc") else None
                rows.append((h, b.id, lv if by in ("marker", "toc") else None, by is None, m[0] if m else None))
    rows.sort(key=lambda r: (r[0], r[1]))
    out, group = {}, []
    for r in rows + [None]:
        if group and (r is None or r[0] > group[-1][0] * (1 + HEIGHT_GAP)):
            known = {lv for _, _, lv, _, _ in group if lv is not None}
            if len(known) == 1:
                kinds = {k for _, _, _, _, k in group if k}
                kind = next(iter(kinds)) if len(kinds) == 1 else None
                out.update((i, (next(iter(known)), kind)) for _, i, _, unknown, _ in group if unknown)
            group = []
        if r is not None:
            group.append(r)
    return out


def _deeper(first: dict, levels: dict) -> list[str]:
    """첫 걷기보다 깊어진 번호·책갈피 제목 — 높이로 넣은 제목이 그 위에 끼어든 모순(앞 장 스택에서 풀려 얕아진 것은 받아들인다)."""
    return [i for i, (lv, by) in first.items() if by in ("marker", "toc") and (levels[i][0] or 0) > lv]


def headings(doc: Document, toc: list[dict] | None = None) -> Document:
    """keep 블록마다 section_path, 제목 블록에는 heading_level. toc = PDF 책갈피(힌트)."""
    doc = doc.model_copy(deep=True)
    cands = _candidates(doc)
    hints = _toc_levels(doc, cands, toc or [])
    first = _walk(doc, cands, hints, {})                   # 번호·책갈피 제목의 수준 → 글자 높이 군집의 기준
    heights = _height_levels(doc, cands, first[0])
    texts = {b.id: _clean(b.text) for p in doc.pages for b in p.blocks if b.id in heights}
    levels, paths, used = first
    while heights:
        levels, paths, used = _walk(doc, cands, hints, heights)
        deep = _deeper(first[0], levels)
        if not deep:
            break
        # 깊어진 제목의 경로 위에 높이로 넣은 제목만 믿지 않는다(잎으로) — 다른 곳의 높이 수준은 둔다
        above = {t for i in deep for t in paths[i][:-1]}
        bad = {i for i in heights if levels[i][1] == "height" and texts[i] in above}
        if not bad:
            levels, paths, used = first
            break
        heights = {i: v for i, v in heights.items() if i not in bad}
    else:
        levels, paths, used = first
    for page in doc.pages:
        for b in page.blocks:
            if b.id in paths:
                b.section_path = paths[b.id]
            lv, by = levels.get(b.id, (None, None))
            if lv is not None:
                b.heading_level = lv
                b.provenance = (b.provenance or []) + [{"rule": "heading_level", "by": by}]
            elif by == "repeat":
                b.provenance = (b.provenance or []) + [{"rule": "heading_repeat"}]
            if b.id in used:
                b.provenance = (b.provenance or []) + [{"rule": "header_section"}]
    return doc
