"""그림·차트(기획서 4.3 규칙 11) — 표 안 그림, 그림 감싼 문단, 그림 갈래·그림 설명(D19)."""
from __future__ import annotations

import hashlib
import html as htmllib
import json
import re

from ..classify.blocks import sentences
from ..geometry import local, page_size
from ..schema import Block, Document, Page
from .order import kept
from .paragraphs import titled
from .tables import appendage, attached

_TABLE_IMG = re.compile(r'<img src="(imgs/img_in_[^"]+)" alt="Image"" />')
_OUR_IMG = re.compile(r'<img src="imgs/([^"]+)"')


def fix_table_imgs(html: str, mapping: dict[str, str]) -> str:
    """표 안 그림 경로를 raw/imgs 이름으로 바꾸고 PaddleX 의 따옴표 버그(alt="Image"")를 고친다."""
    def sub(m):
        name = mapping.get(m.group(1))
        src = f"imgs/{name}" if name else m.group(1)
        return f'<img src="{src}" alt="Image" />'

    return _TABLE_IMG.sub(sub, html or "")


def table_img_names(html: str | None) -> list[str]:
    return _OUR_IMG.findall(html or "")


def split_photo_text(html: str, photos: dict[str, str]) -> tuple[str, list[str]]:
    """표 칸 안 사진 뒤에 붙은 사진 속 글자(그림 OCR)를 칸 값에서 떼어 alt 로 옮긴다(실측 — 칸 값과 섞인다).
    photos = {우리 이미지 이름: 그 사진의 OCR 원문}. 떼어 낸 글자 목록을 함께 돌려준다."""
    texts = []

    def sub(m):
        name, after = m.group(1), m.group(2)
        raw = " ".join((photos.get(name) or "").split())
        flat = " ".join(htmllib.unescape(after).split())
        if not raw or not flat.startswith(raw):
            return m.group(0)
        texts.append(raw)
        rest = flat[len(raw):].strip()
        return f'<img src="imgs/{name}" alt="{htmllib.escape(raw)}" />' + (" " + htmllib.escape(rest, quote=False) if rest else "")

    return re.sub(r'<img src="imgs/([^"]+)" alt="Image" />([^<]*)', sub, html), texts


_END = re.compile(r"([.!?:」』”)]|다|요|함|음|임|됨|것|등)\s*$")
_ITEM = re.compile(r"^\s*(?:[□■○●◦•▪\-·※ㅇ▶►◆◇➢‣*]|\d{1,2}[.)]|[가나다라마바사아자차카타파하][.)]|\(\d+\)|[①-⑳➀-➓❶-❿]"
                   r"|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.)")
WRAP_GAP = 20.0               # 그림 아래 블록까지의 세로 틈(pt)
EDGE = 10.0                   # 두 블록 끝 맞춤 허용(pt)


def join_wrapped(doc: Document) -> Document:
    """그림을 감싸 흐르는 한 문단이 그림 옆과 아래 두 블록으로 갈린 것을 잇는다(실측 — 그림 속 글자 읽기를 켠 부작용).
    앞 블록이 문장 끝으로 끝나지 않고 뒤 블록이 항목·번호로 시작하지 않으며 검출기가 제목으로 본 줄(문장 꼴이라 text 로
    바꾼 제목 라벨)이 아닐 때만. 표에 이은 제목·출처와 표·그림 부속 줄은 본문이 아니라 합치지 않는다(표의 caption_id 가
    버린 블록을 가리키게 된다). 위치는 세운 논리 페이지 좌표로 잰다."""
    doc = doc.model_copy(deep=True)
    for page in doc.pages:
        refs = attached(page.blocks)
        live = [b for b in page.blocks if b.keep and b.merged_into is None]
        texts = [b for b in live if b.type == "text" and b.text and b.id not in refs and not appendage(b.text)]
        box = {b.id: local(page, b.bbox) for b in live}
        for f in (b for b in live if b.type in ("image", "chart")):
            fx0, fy0, fx1, fy1 = box[f.id]
            beside = [a for a in texts if box[a.id][1] >= fy0 - 5 and box[a.id][3] <= fy1 + 5
                      and (box[a.id][2] <= fx0 + 5 or box[a.id][0] >= fx1 - 5)]
            if not beside:
                continue
            a = max(beside, key=lambda x: box[x.id][3])
            ab = box[a.id]
            right = ab[2] <= fx0 + 5                            # 그림이 글의 오른쪽
            below = [b for b in texts if b is not a and 0 <= box[b.id][1] - max(ab[3], fy1) <= WRAP_GAP
                     and box[b.id][0] <= min(ab[0], fx0) + EDGE and box[b.id][2] >= max(ab[2], fx1) - EDGE
                     and (abs(box[b.id][0] - ab[0]) <= EDGE if right else abs(box[b.id][2] - ab[2]) <= EDGE)]
            if not below:
                continue
            b = min(below, key=lambda x: box[x.id][1])
            if _END.search(a.text) or _ITEM.match(b.text) or titled(b):   # 검출기가 제목으로 본 줄은 새 문단의 시작
                continue
            a.text = a.text.rstrip() + " " + b.text.lstrip()
            a.bbox = (min(a.bbox[0], b.bbox[0]), min(a.bbox[1], b.bbox[1]), max(a.bbox[2], b.bbox[2]), max(a.bbox[3], b.bbox[3]))
            box[a.id] = local(page, a.bbox)
            b.keep, b.drop_reason, b.merged_into, b.text = False, "merged", a.id, None   # 글자는 대표에만(병합 묶음 규약)
            a.provenance = (a.provenance or []) + [{"rule": "join_wrapped", "with": b.id, "figure": f.id}]   # 감싼 그림 — 쪽 넘김 이음이 본다
            texts.remove(b)
    return doc


# ---- 그림 갈래·그림 설명(D19) ----

SENTENCE_HAN = 8              # 글 갈래 — 그림 속 글 한 줄의 평균 한글 수가 이만큼 이상(문장·긴 구절)
SENTENCE_MIN_HAN = 20         # 그림 속 한글이 이보다 적으면 글 그림으로 보지 않는다(짧은 이름표)
NUMERIC_MAX = 0.2             # 숫자 낱말 비율이 이보다 크면 그래프·표 같은 그림이다
_HAN = re.compile(r"[가-힣]")
_NUMWORD = re.compile(r"[\d.,%+\-–~℃°()/:]+")
CONTEXT_CHARS = 300           # 물음에 넣는 그림 앞·뒤 본문 글자 수 — 앞뒤 글과 이어지는 설명을 쓰게(사용자 요청 2026-10-19)
DESC_PROMPT = """아래는 한국어 문서의 한 부분이고, [그림] 자리에 그림이 있습니다. 그림을 보고, 그림이 보여 주는 내용을 이 문서의 본문 문장처럼 쓰세요. 앞뒤 글과 자연스럽게 이어져야 합니다.

문서 구역: {section}
앞 글: {before}
[그림] 제목: {caption}
그림에서 읽은 글자(틀린 글자가 있을 수 있습니다): {text}
뒤 글: {after}

쓰는 법
- 2~4문장, 이 문서처럼 '~다.'로 끝나는 문장으로 씁니다.
- '이 그림은', '이 지도는', '위 그래프는'처럼 그림을 가리키며 시작하지 말고, 그림이 보여 주는 내용을 바로 씁니다. 범례·축·제목이 어디에 있는지 같은 그림의 생김새는 쓰지 않습니다.
  나쁜 예: "이 그래프는 월별 방문객 수를 나타낸 그래프입니다."
  좋은 예: "월별 방문객은 3월부터 늘어 8월에 가장 많았고 겨울에는 줄었다."
- 앞뒤 글을 되풀이하지 말고, 그림을 봐야 알 수 있는 내용(어디가 높고 낮은지, 어떻게 늘고 주는지, 무엇이 무엇으로 이어지는지, 색이 뜻하는 것)을 보탭니다.
- 해·월·일·주 같은 때는 축을 보고 써도 됩니다. 값(기온·강수량·개수·비율 등)은 그림이나 앞뒤 글에 적힌 숫자만 그대로 쓰고, 눈금으로 어림한 값은 쓰지 않습니다.
- 그림이나 앞뒤 글에 적히지 않은 지명·이름은 쓰지 말고 위치(북동쪽, 왼쪽 위 등)로 씁니다.
- 그림에 없는 원인·평가·전망은 쓰지 않습니다."""
_FIG_WORD = r"(?:그림|지도|그래프|사진|도식|모식도|차트|이미지|도표|포스터|일러스트)"
_LEAD = re.compile(rf"^(?:(?:이|위|아래|본|해당|다음|두|왼쪽|오른쪽)\s*)?(?:막대\s*|선\s*)?{_FIG_WORD}"
                   r"(?:은|는|에서는|에서|에는|을\s*보면|를\s*보면)\s+")          # 그림을 가리키는 머리말('이 그림은')
_DEPICT = re.compile(r"나타[내냅낸냈]|보여\s*[주줍준줬]|표시[하합한했]|표현[하합한했]|구분[하합한했]|담[고은습겨]")   # 그리는 말('나타냅니다'·'보여 준다')
_INTRO_END = re.compile(rf"(?:나타낸|보여\s*주는|표시한|표현한|담은)\s*{_FIG_WORD}(?:이다|입니다)\s*[.]?$")   # '…를 나타낸 지도이다.'
_FINDING = re.compile(r"가장|높|낮|많|적|늘|줄|증가|감소|상승|하락|오르|올라|내리|내려|이어|몰려|집중|밀집|최고|최저|최대|최소"
                      r"|넘|초과|미만|이상|이하|비슷|차이|앞서|빠르|느리|진할수록|옅을수록|클수록|작을수록")   # 내용 낱말
_LAYOUT = re.compile(r"(?:범례|나침반|축척|척도|눈금|색\s*막대|색상\s*막대|가로\s*축|세로\s*축|[xyXY]\s*축|그림\s*제목)"
                     r"[가-힣]{0,2}\s*(?:가|이|는|은|도|에는)?\s*(?:있|제공|표시되어|그려져|위치|붙어)")   # 그림 생김새 문장
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_PLACE = re.compile(r"(?<![가-힣])[가-힣]{2,4}(?:시|군|구)(?=[이가은는을를의와과에도로만]|[^가-힣]|$)")   # 시·군·구 이름 꼴('강화군(7)이'·'파주시와')
_NOT_PLACE = {"광역시", "대도시", "북반구", "남반구", "출입구", "비상구", "또다시"}   # 이름 꼴이지만 지명이 아니다
_TIME = re.compile(r"(?:(\d{1,4})\s*[~∼〜\-–]\s*)?(\d{1,4})\s*(년도|년|분기|월|주차|주|일|시)(?!간)")   # 때·자리('6월 13일'·'2023~2025년'·'27주')
_TIME_RANGE = {"월": (1, 12), "일": (1, 31), "주": (1, 53), "주차": (1, 53), "분기": (1, 4), "시": (0, 24)}
_YEAR = re.compile(r"(?<!\d)(?:19|20|21)\d\d(?!\d)")
_YY = re.compile(r"['‘’](\d{2})(?!\d)")                                 # "'23" — 두 자리 해
_THINK = re.compile(r"<think>.*?</think>", re.S)
DESC_MAX_SENTENCES = 4        # 물음이 청한 문장 수 상한 — 되풀이로 길어진 응답을 자른다
_FOREIGN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")   # 가나·한자
_MARK = re.compile(r"^\s*(?:[-*•]|\d{1,2}[.)])\s+")                    # 모델이 붙인 글머리표·번호


def route_figures(doc: Document) -> Document:
    """그림 갈래(D19) — 글자만 꺼내도 뜻이 서는 그림(text)과 그림을 함께 봐야 뜻이 서는 그림(image)을 가른다. 검출기가 그래프·
    지도(chart)로 본 것은 늘 image, 글자가 없으면 image(no_text — 사진·삽화), 그림 속 글이 문장·긴 구절 위주(한 줄 평균 한글
    SENTENCE_HAN 이상, 숫자 낱말 NUMERIC_MAX 이하)면 text(sentences), 그 밖의 짧은 이름표·숫자는 image(labels — 도식, 사진 위
    이름표). 모르면 image 다 — 글 그림을 그림으로 보면 설명이 하나 늘 뿐이지만 도식을 글로 보면 뜻이 깨진다."""
    doc = doc.model_copy(deep=True)
    for b in (b for p in doc.pages for b in p.blocks):
        if b.type in ("image", "chart") and b.keep and b.merged_into is None:
            b.route, b.route_reason = _figure_route(b)
    return doc


def _figure_route(b: Block) -> tuple[str, str]:
    if b.type == "chart":
        return "image", "chart"
    text = b.figure_text or ""
    if not any(ch.isalnum() for ch in text):
        return "image", "no_text"
    lines = [ln for ln in text.splitlines() if ln.strip()]
    words = text.split()
    han = len(_HAN.findall(text))
    numeric = sum(bool(_NUMWORD.fullmatch(w)) for w in words) / len(words)
    if han >= SENTENCE_MIN_HAN and han / len(lines) >= SENTENCE_HAN and numeric <= NUMERIC_MAX \
            and 2 * sentences(text) >= len(lines):     # 끝맺음(다·요·함·마침표)·항목 줄이 절반 이상 — 긴 칸 이름뿐인 도식은 그림
        return "text", "sentences"
    return "image", "labels"


def describe_targets(doc: Document, cfg) -> list[tuple[Page, Block]]:
    """설명할 그림(D19) — 살린 쪽의 살린 그림 갈래(image) 블록 가운데 크롭이 있고 논리 페이지 넓이의 min_area 이상이며, 그래프·
    지도이거나 그림 제목·그림 속 글자가 있는 것. 제목도 글자도 없는 그림(포스터의 만화 조각 등)은 설명하지 않는다."""
    out = []
    for page in (p for p in doc.pages if p.keep):
        w, h = page_size(page)
        for b in page.blocks:
            if b.route != "image" or b.type not in ("image", "chart") or not b.keep or b.merged_into or not b.img_path:
                continue
            x0, y0, x1, y1 = b.bbox
            if (x1 - x0) * (y1 - y0) < cfg.min_area * w * h:
                continue
            if b.type == "chart" or b.caption_id or any(ch.isalnum() for ch in b.figure_text or ""):
                out.append((page, b))
    return out


def _caption(page: Page, b: Block) -> str:
    c = next((o for o in page.blocks if o.id == b.caption_id), None) if b.caption_id else None
    return " ".join((c.text or "").split()) if c else ""


def _context(doc: Document, b: Block) -> tuple[str, str]:
    """읽기 순서로 그림 바로 앞·뒤의 본문 글(제목·항목 포함) — 표·그림·표와 그림의 제목·주는 건너뛰고, 쪽을 넘어 앞은 끝에서,
    뒤는 처음부터 CONTEXT_CHARS 자까지."""
    seq = [x for _, x in kept(doc)]
    i = next((k for k, x in enumerate(seq) if x is b), None)
    if i is None:
        return "", ""
    side = {cid for p in doc.pages for cid in attached(p.blocks)} | {x.caption_id for x in seq if x.caption_id}

    def text_of(xs):
        return [" ".join(x.text.split()) for x in xs if x.text and x.text.strip() and x.id not in side
                and x.type not in ("table", "image", "chart", "display_formula")]

    before, after = " ".join(text_of(seq[max(0, i - 8):i])), " ".join(text_of(seq[i + 1:i + 9]))
    before = before if len(before) <= CONTEXT_CHARS else "…" + before[-CONTEXT_CHARS:]
    after = after if len(after) <= CONTEXT_CHARS else after[:CONTEXT_CHARS] + "…"
    return before, after


def desc_prompt(doc: Document, page: Page, b: Block) -> str:
    before, after = _context(doc, b)
    return DESC_PROMPT.format(section=" > ".join(b.section_path or []) or "(없음)", before=before or "(없음)",
                              caption=_caption(page, b) or "(없음)", after=after or "(없음)",
                              text=" / ".join(ln.strip() for ln in (b.figure_text or "").splitlines() if ln.strip()) or "(없음)")


def desc_key(model: str, b: Block, prompt: str, max_tokens: int) -> str:
    """응답 기록의 열쇠 — 모델·블록·크롭·물음·생성 한도가 같아야 같은 설명이다(조립 규칙이 그림 제목·그림 글을 바꾸거나 한도를
    올리면 다시 부른다)."""
    s = json.dumps([model, b.id, b.img_path, prompt, max_tokens], ensure_ascii=False)
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _num_key(m: str) -> str:
    """같은 수는 같은 열쇠 — 쉼표('1,250'), 앞 0('09-01' 의 9), 소수 끝 0('+1.0℃' 를 '1℃' 로 쓴 설명)을 맞춘다."""
    whole, _, frac = m.replace(",", "").partition(".")
    whole, frac = whole.lstrip("0") or "0", frac.rstrip("0")
    return whole + ("." + frac if frac else "")


def _reference(doc: Document, page: Page, b: Block) -> str:
    """설명의 근거 — 그림 속 글자, 그림 제목, 그림 상자 안에 가운데가 든 살린 글 블록(글자층 글자가 있어 본문으로 둔 그림 속 글),
    물음에 넣은 구역 이름과 앞뒤 글(문서에 적힌 값이라 지어낸 것이 아니다)."""
    x0, y0, x1, y1 = b.bbox
    inner = [o.text for o in page.blocks if o is not b and o.keep and o.text and o.merged_into is None
             and x0 <= (o.bbox[0] + o.bbox[2]) / 2 <= x1 and y0 <= (o.bbox[1] + o.bbox[3]) / 2 <= y1]
    return "\n".join(t for t in (b.figure_text, _caption(page, b), *inner, *(b.section_path or []), *_context(doc, b)) if t)


def _position_ok(n: str, unit: str, years: set[int]) -> bool | None:
    """때·자리 수(해·월·일·주·분기·시)가 말이 되나 — 축의 자리는 그림 글자와 맞추지 않는다(OCR 이 축 글자를 자주 놓치고, 축에 다
    적혀 있어 맞춰 봐야 막는 것이 없다). 해는 근거에 든 해의 범위(앞뒤 1년) 안, 근거에 해가 없으면 1900~2100. 기간('지난 3년')은
    값이라 None."""
    if unit in ("년", "년도"):
        if len(n) != 4:
            return None
        lo, hi = (min(years) - 1, max(years) + 1) if years else (1900, 2100)
    else:
        lo, hi = _TIME_RANGE[unit]
    return lo <= int(n) <= hi


def _grounded(s: str, ref: str, flat: str, nums: set[str], years: set[int]) -> bool:
    pos = {m.start(g): (m.group(g), m.group(3)) for m in _TIME.finditer(s) for g in (1, 2) if m.group(g)}
    for m in _NUM.finditer(s):
        p = pos.get(m.start())
        ok = _position_ok(*p, years) if p and p[0] == m.group() else None
        if not (ok if ok is not None else _num_key(m.group()) in nums):
            return False
    return all(p in flat for p in _PLACE.findall(s) if p not in _NOT_PLACE) and all(ch in ref for ch in _FOREIGN.findall(s))


def _tidy(s: str) -> str | None:
    """그림을 소개만 하는 문장('이 그래프는 … 변화를 나타냅니다' — 내용 낱말 없이 그리는 말로 맺음)과 생김새 문장('범례가 오른쪽에
    있다')은 None, 그림을 가리키는 머리말은 뗀다('지도에서는 서쪽이 높다' → '서쪽이 높다') — 앞뒤 글과 이어지는 본문 문장으로."""
    lead = _LEAD.match(s)
    body = s[lead.end():] if lead else s
    intro = lead and _DEPICT.search(body) or _INTRO_END.search(body)
    if intro and not _FINDING.search(body) or _LAYOUT.search(s):
        return None
    return body


def ground(text: str, ref: str, truncated: bool = False) -> tuple[list[str], list[str], list[str]]:
    """설명 문장을 (남길 문장, 근거 없어 버린 문장, 그림 소개·생김새라 뺀 문장)으로 — 값(기온·강수량·개수 등 때·자리가 아닌 수)·
    시군구 이름이 근거에 없거나, 때·자리 수가 말이 안 되거나('13월'), 근거에 없는 가나·한자가 섞인 문장은 버린다(그래프 표 읽기가
    값을 지어낸 것과 같은 위험 — 테스트_결과 6.5). 생각 칸은 떼고, 같은 문장은 한 번만, 길이 한도로 잘렸으면 끝 문장도 버리고,
    그림을 소개만 하는 문장은 빼고(_tidy), 남길 문장은 DESC_MAX_SENTENCES 개까지."""
    lines = [_MARK.sub("", ln).replace("**", "").strip() for ln in _THINK.sub("", text or "").splitlines()]
    sents = list(dict.fromkeys(s.strip() for s in _SENT.split("\n".join(ln for ln in lines if ln)) if s.strip()))
    cut = [sents.pop()] if truncated and sents else []
    flat = "".join(ref.split())
    nums = {_num_key(m) for m in _NUM.findall(ref)}
    years = {int(y) for y in _YEAR.findall(ref)} | {2000 + int(y) for y in _YY.findall(ref)}
    kept, dropped, meta = [], [], []
    for s in sents:
        t = _tidy(s)
        if t is None:
            meta.append(s)
        elif _grounded(t, ref, flat, nums, years):
            kept.append(t)
        else:
            dropped.append(s)
    return kept[:DESC_MAX_SENTENCES], dropped + cut, meta


def attach_descriptions(doc: Document, cache: dict[str, dict], cfg) -> Document:
    """그림 설명(D19)을 붙인다 — 응답 기록(raw/describe.jsonl)에서 같은 열쇠를 찾아 근거 검사(ground)를 통과한 문장만
    figure_desc 로, 버린 문장은 기록으로 남긴다. 응답이 없는 그림은 표시만 해 QC 가 센다(kdp describe 로 채운다)."""
    if not cfg.enabled:
        return doc
    doc = doc.model_copy(deep=True)
    for page, b in describe_targets(doc, cfg):
        hit = cache.get(desc_key(cfg.model, b, desc_prompt(doc, page, b), cfg.max_tokens))
        if hit is None:
            b.provenance = (b.provenance or []) + [{"rule": "figure_desc_missing"}]
            continue
        kept, dropped, meta = ground(hit["text"], _reference(doc, page, b), hit.get("finish_reason") == "length")
        b.figure_desc = " ".join(kept) or None
        b.provenance = (b.provenance or []) + [{"rule": "figure_desc", "model": cfg.model, **({"dropped": dropped} if dropped else {}),
                                                **({"meta": meta} if meta else {})}]
    return doc
