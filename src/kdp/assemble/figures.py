"""그림·차트(기획서 4.3 규칙 11) — 표 안 그림, 그림 감싼 문단, 그림 갈래·그림 설명(D19)."""
from __future__ import annotations

import hashlib
import html as htmllib
import json
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Callable, NamedTuple

from ..classify.blocks import sentences
from ..geometry import local, page_size
from ..schema import Block, Document, Page
from .normalize import norm_text
from .order import kept
from .paragraphs import titled
from .tables import _letters, appendage, attached, body_text, salvage

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
DESC_PROMPT = """아래는 한국어 문서의 한 부분이고, [그림] 자리에 그림이 있습니다. 그림을 보고, 그림에서 읽히는 내용을 이 문서의 본문 문장처럼 쓰세요. 앞뒤 글과 자연스럽게 이어져야 합니다.

문서 구역: {section}
앞 글: {before}
[그림] 제목: {caption}
그림에서 읽은 글자(틀린 글자가 있을 수 있습니다): {text}
뒤 글: {after}

쓰는 법
- 1~3문장, 이 문서처럼 '~다.'로 끝나는 문장으로 씁니다. 그림에서 분명히 보이는 것만 쓰고, 분명하지 않은 것은 쓰지 않습니다.
- '이 그림은', '이 지도는', '위 그래프는'처럼 그림을 가리키며 시작하지 말고, 그림이 보여 주는 내용을 바로 씁니다. 범례·축·색이 어디에 있고 무엇을 뜻하는지 같은 그림 읽는 법은 쓰지 않습니다.
  나쁜 예: "이 그래프는 월별 방문객 수를 나타낸 그래프입니다."
  좋은 예: "월별 방문객은 3월부터 늘어 8월에 가장 많았고 겨울에는 줄었다."
- 앞뒤 글의 문장을 옮겨 적지 말고, 그림을 봐야 알 수 있는 내용(어디가 높고 낮은지, 언제 늘고 줄었는지)을 씁니다. 장식·홍보 그림이라 보탤 내용이 없으면 '없음'이라고만 씁니다.
- 선·막대마다 자료가 끝나는 때가 다를 수 있습니다. 자료가 끝난 뒤의 때는 쓰지 않습니다.
- 범례의 눈금 숫자를 그림 속 값의 범위로 쓰지 않습니다.
- 색은 범례가 재는 것에 맞게 읽습니다. 평년과의 차이(편차)를 나타낸 지도에서 덜 높은 곳은 '평년보다 덜 높다'이지 '낮다'가 아니고, 퍼센타일·순위 지도의 색은 많고 적음의 순위입니다.
- 그림에 적힌 날짜·숫자·이름은 바꾸지 않고 그대로 씁니다. 값은 그림이나 앞뒤 글에 적힌 숫자만 쓰고, 눈금으로 어림한 값은 쓰지 않습니다.
- 그림이나 앞뒤 글에 적히지 않은 지명·이름은 쓰지 말고 위치(북동쪽, 왼쪽 위 등)로 씁니다.
- 원인과 결과는 그림이나 앞뒤 글에 적힌 방향 그대로만 씁니다. 적히지 않은 원인·평가·전망은 쓰지 않습니다."""
_FIG_WORD = r"(?:그림|지도|그래프|사진|도식|모식도|차트|이미지|도표|포스터|일러스트)"
_LEAD = re.compile(rf"^(?:(?:이|위|아래|본|해당|다음|두|왼쪽|오른쪽)\s*)?(?:막대\s*|선\s*)?{_FIG_WORD}"
                   r"(?:은|는|에서는|에서|에는|을\s*보면|를\s*보면)\s+")          # 그림을 가리키는 머리말('이 그림은')
_DEPICT = re.compile(r"나타[내냅낸냈]|보여\s*[주줍준줬]|표시[하합한했]|표현[하합한했]|구분[하합한했]|담[고은습겨]")   # 그리는 말('나타냅니다'·'보여 준다')
_INTRO_END = re.compile(rf"(?:나타낸|보여\s*주는|표시한|표현한|담은)\s*{_FIG_WORD}(?:이다|입니다)\s*[.]?$")   # '…를 나타낸 지도이다.'
_FINDING = re.compile(r"가장|높|낮|많|적|늘|줄|증가|감소|상승|하락|오르|올라|내리|내려|이어|몰려|집중|밀집|최고|최저|최대|최소"
                      r"|넘|초과|미만|이상|이하|비슷|차이|앞서|빠르|느리|클수록|작을수록")   # 내용 낱말
_READING = re.compile(r"(?:색|색상|색깔|점|선|막대|기호|화살표)\s*(?:으로|로)\s*(?:구분|표시|나타|표현)|(?:쉽게|한눈에)\s*(?:파악|확인|알|볼)\s*수"
                      r"|(?:진할|옅을|짙을|밝을|어두울|연할)수록|(?:을|를)\s*(?:의미|뜻)한다")   # 그림 읽는 법('색이 진할수록 많음을 뜻한다' — 실측에서 거꾸로 읽음)
_DEPICT_END = re.compile(r"(?:변화|분포|추이|모습|현황|흐름|양상|추세|구성|위치|관계|특징|모양|형태|구조)(?:를|을)\s*"
                         rf"(?:{_DEPICT.pattern})[가-힣\s]{{0,6}}[.]?$")                    # '…의 변화를 보여준다.' — 주어 없는 소개 문장
_LAYOUT = re.compile(r"(?:범례|나침반|축척|척도|눈금|색\s*막대|색상\s*막대|가로\s*축|세로\s*축|[xyXY]\s*축|그림\s*제목)"
                     r"[가-힣]{0,2}\s*(?:가|이|는|은|도|에는)?\s*(?:있|제공|표시되어|그려져|위치|붙어)")   # 그림 생김새 문장
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_PLACE = re.compile(r"(?<![가-힣])[가-힣]{2,4}(?:시|군|구)(?=[이가은는을를의와과에도로만]|[^가-힣]|$)")   # 시·군·구 이름 꼴('강화군(7)이'·'파주시와')
_NOT_PLACE = {"광역시", "대도시", "북반구", "남반구", "출입구", "비상구", "또다시"}   # 이름 꼴이지만 지명이 아니다
_TIME = re.compile(r"(?:(\d{1,4})\s*[~∼〜\-–]\s*)?(\d{1,4})\s*(년도|년|분기|월|주차|주|일|시)"
                   r"(?!간|\s*(?:동안|이상|이하|미만|초과|정도|가량|남짓|넘게|째|연속))")   # 때·자리('6월 13일'·'2023~2025년'·'27주') — 뒤에 '이상·동안' 이 오면 길이(값)
_DATE = re.compile(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일((?:\s*(?:과|와|,|및|·|~|∼|부터|에서|-|–)\s*\d{1,2}\s*일)*)")   # '9월 26일과 27일'
_MD = re.compile(r"(?<![\d.])(\d{1,2})\.\s?(\d{1,2})\.?(?![\d])")                  # '9.26' · '6.13.' — 월.일
_DECADE = re.compile(r"((?:19|20)\d)0\s*년대\s*(초반|중반|후반|말|초|중)?")
_DECADE_PART = {"초반": 0, "초": 0, "중반": 3, "중": 3, "후반": 7, "말": 7, None: 0}  # '2020년대 후반' 은 2027년부터
_PERIOD = re.compile(r"((?:19|20)\d\d)\s*년?\s*\(\s*[^()]*?[~∼〜\-–]\s*(\d{1,2})\s*[./월]\s*\d{1,2}\s*[.일]?\s*\)")   # '2026년(1.1.~6.13.)' — 그 해 자료가 끝나는 달
_MONTH = re.compile(r"(?<!\d)(\d{1,2})\s*월")
_LINK = re.compile(r"이로\s*인해|이\s*때문에|이에\s*따라|그\s*결과|이로써|따라서")   # 앞 문장과 잇는 인과 — 근거에 없으면 모델이 지은 인과
_SPECULATE = re.compile(r"(?:예상|전망|예측|우려)(?:된다|되며|되고|됩니다|한다|합니다|하고|되어)|[될할칠줄늘클갈올] 것(?:이다|입니다|으로)")   # 전망
COPY_SHARE = 0.7              # 문장 글자 4개 묶음 가운데 이만큼 이상이 앞뒤 본문에 있으면(그림 글자보다 많이) 본문을 옮긴 문장
SCALE_RUN = 5                 # 그림 글자에서 수가 이만큼 이상 한 방향으로 이어지면 범례·축 눈금
_TIME_RANGE = {"월": (1, 12), "일": (1, 31), "주": (1, 53), "주차": (1, 53), "분기": (1, 4), "시": (0, 24)}
_YEAR = re.compile(r"(?<!\d)(?:19|20|21)\d\d(?!\d)")
_YY = re.compile(r"['‘’](\d{2})(?!\d)")                                 # "'23" — 두 자리 해
_ITEM_NAME = re.compile(r"최[고저]\s*기온|최대\s*풍속|최소\s*습도|이상\s*고온|증가율|감소율")   # 내용 낱말을 품은 항목 이름 — 소개 문장 판단에서 뺀다
_SENT_END = re.compile(r"다[.!?…'\"」』)\]]*$")                       # '~다.' 로 맺은 문장 — 그림 글자를 옮긴 조각('(1일) 정체전선 영향 비')은 아니다
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


def _reference(doc: Document, page: Page, b: Block) -> tuple[str, str]:
    """설명의 근거 — (그림 쪽: 그림 속 글자, 그림 제목, 그림 상자 안에 가운데가 든 살린 글 블록(글자층 글자가 있어 본문으로 둔
    그림 속 글), 구역 이름 / 앞뒤 글: 물음에 넣은 앞뒤 본문). 둘 다 문서에 적힌 값이라 지어낸 것이 아니다. 앞뒤 글은 본문을 옮겨
    적은 문장을 가리는 데도 쓴다."""
    x0, y0, x1, y1 = b.bbox
    inner = [o.text for o in page.blocks if o is not b and o.keep and o.text and o.merged_into is None
             and x0 <= (o.bbox[0] + o.bbox[2]) / 2 <= x1 and y0 <= (o.bbox[1] + o.bbox[3]) / 2 <= y1]
    fig = "\n".join(t for t in (b.figure_text, _caption(page, b), *inner, *(b.section_path or [])) if t)
    return fig, "\n".join(t for t in _context(doc, b) if t)


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


def _days(ref: str) -> set[int]:
    """근거에 적힌 날 — 'N일'·'25~26일' 의 두 끝, '9.26'·'6.13.' 의 일."""
    return _written(ref, ("일",)) | {int(d) for mo, d in _MD.findall(ref) if 1 <= int(mo) <= 12 and 1 <= int(d) <= 31}


def _written(ref: str, units: tuple[str, ...]) -> set[int]:
    """근거에 그 단위로 적힌 때의 수('20주'·'25~26일' 의 두 끝)."""
    return {int(m.group(g)) for m in _TIME.finditer(ref) if m.group(3) in units for g in (1, 2) if m.group(g)}


def _values(s: str) -> set[str]:
    """문장의 값(때·자리가 아닌 수)."""
    pos = {m.start(g) for m in _TIME.finditer(s) for g in (1, 2) if m.group(g)}
    return {_num_key(m.group()) for m in _NUM.finditer(s) if m.start() not in pos}


def _scales(ref: str) -> list[tuple[str, str]]:
    """그림 글자에서 수가 SCALE_RUN 개 이상 한 방향으로 이어진 곳(범례·축 눈금 '31 / 29.3 / … / 5') — 양 끝 수. 해 꼴 수(19xx·20xx)는
    때라 끊는다."""
    out, run = [], []
    for tok in ref.split() + [""]:
        m = re.fullmatch(r"([+\-−–]?)(\d+(?:\.\d+)?)(?:°C|℃|%|mm|일)?", tok)
        if m and not _YEAR.fullmatch(m.group(2)):
            run.append((float(m.group(2)) * (-1 if m.group(1) else 1), m.group(2)))
            continue
        i = 0
        while i < len(run) - 1:                           # 한 방향으로 이어지는 가장 긴 조각들
            j, up = i + 1, run[i + 1][0] > run[i][0]
            while j + 1 < len(run) and run[j + 1][0] != run[j][0] and (run[j + 1][0] > run[j][0]) == up:
                j += 1
            if j - i + 1 >= SCALE_RUN and run[i][0] != run[i + 1][0]:
                seg = run[i:j + 1]
                out.append((_num_key(min(seg)[1]), _num_key(max(seg)[1])))
            i = j
        run = []
    return out


def _sensible(s: str, flat: str, days: set[int], weeks: set[int], years: set[int], periods: dict[int, int],
              scales: list[tuple[str, str]]) -> bool:
    """때·범위가 근거와 어긋나지 않나(실측 오류 — 테스트_결과 9.4·10장) — 날짜('9월 26일과 27일')는 근거에 적힌 날이나 1일·말일만,
    주('2주차')는 근거에 주가 적혀 있으면 그 주만(축에 수만 적힌 그래프는 보지 않는다), '2020년대 후반' 은 근거의 해가 거기까지 있을
    때만, 그림이 밝힌 자료 기간('2026년(1.1.~6.13.)') 뒤의 달은 안 되고, 범례·축 눈금의 양 끝을 값의 범위로 쓰지 않고('5°C에서
    31°C 사이'), 전망('…것으로 예상된다')과 근거에 없는 인과 접속('이로 인해 찬 공기가 유입되었다' — 실측에서 지은 인과)은 쓰지
    않는다."""
    for m in _DATE.finditer(s):
        for d in (int(x) for x in (m.group(2), *re.findall(r"(\d{1,2})\s*일", m.group(3)))):
            if d not in days and d != 1 and not 28 <= d <= 31:
                return False
    if weeks and any(int(m.group(g)) not in weeks for m in _TIME.finditer(s) if m.group(3) in ("주", "주차")
                     for g in (1, 2) if m.group(g)):
        return False
    for m in _DECADE.finditer(s):
        if years and max(years) < int(m.group(1)) * 10 + _DECADE_PART[m.group(2)]:
            return False
    for y, end in periods.items():
        if f"{y}년" in s and any(int(mo) > end for mo in _MONTH.findall(s)):
            return False
    vals = _values(s)
    return not any(lo in vals and hi in vals for lo, hi in scales) and not _SPECULATE.search(s) \
        and all("".join(m.group().split()) in flat for m in _LINK.finditer(s))


def _grams(s: str) -> set[str]:
    s = re.sub(r"[\W_]+", "", s)
    return {s[i:i + 4] for i in range(len(s) - 3)}


def _tidy(s: str) -> str | None:
    """그림을 소개만 하는 문장('이 그래프는 … 변화를 나타냅니다'·'…의 변화를 보여준다' — 내용 낱말 없이 그리는 말로 맺음),
    생김새 문장('범례가 오른쪽에 있다'), 그림 읽는 법('진한 색일수록 많음을 뜻한다' — 실측에서 거꾸로 읽었다), '~다' 로 맺지 않은
    조각(그림 글자를 옮긴 줄·영어로 샌 생각), 보탤 것이 없다는 답('없음')은 None, 그림을 가리키는 머리말은 뗀다('지도에서는 서쪽이
    높다' → '서쪽이 높다') — 앞뒤 글과 이어지는 본문 문장으로."""
    lead = _LEAD.match(s)
    body = s[lead.end():] if lead else s
    intro = lead and _DEPICT.search(body) or _INTRO_END.search(body) or _DEPICT_END.search(body)
    if intro and not _FINDING.search(_ITEM_NAME.sub("", body)) or _LAYOUT.search(s) or _READING.search(s) or not _SENT_END.search(body) \
            or body.strip(" .'\"") == "없다":
        return None
    return body


class Grounded(NamedTuple):
    kept: list[str]           # 남길 문장
    dropped: list[str]        # 근거 없음·말이 안 됨(값·지명·때·범례 눈금·전망)
    meta: list[str]           # 그림 소개·생김새·그림 읽는 법·'없음'
    copied: list[str]         # 앞뒤 본문을 옮겨 적은 문장
    unsure: list[str]         # 모델이 확신하지 못한 문장(토큰 평균 로그확률이 min_logprob 아래)


def sentence_logprob(text: str, logprobs: list | None) -> Callable[[str], float | None]:
    """응답 기록의 토큰 확률([[토큰, 로그확률], …]) → 문장마다 그 문장에 걸친 토큰의 평균 로그확률. 기록에 확률이 없거나 문장을
    응답에서 찾지 못하면 None(거르지 않는다)."""
    if not logprobs:
        return lambda s: None
    joined, spans, pos = "", [], 0
    for tok, lp in logprobs:
        spans.append((pos, pos + len(tok), lp))
        joined, pos = joined + tok, pos + len(tok)

    def mean(s: str) -> float | None:
        i = joined.rfind(s)                                   # 끝에서 — 생각 칸에 같은 문장을 먼저 써 볼 수 있다
        if i < 0:
            return None
        lps = [lp for a, b, lp in spans if b > i and a < i + len(s)]
        return sum(lps) / len(lps) if lps else None
    return mean


def _answer(text: str) -> str:
    """생각 칸을 뗀 답 — 마지막 '</think>' 뒤(물음 틀이 여는 꼬리표를 넣어 닫는 꼬리표만 올 수 있다). 생각이 끝나지 않았으면 빈 답."""
    text = text or ""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return "" if "<think>" in text else text


def ground(text: str, ref: str, truncated: bool = False, context: str = "",
           conf: Callable[[str], float | None] | None = None, min_logprob: float | None = None) -> Grounded:
    """설명 문장을 가른다(Grounded) — 값(기온·강수량·개수 등 때·자리가 아닌 수)·시군구 이름이 근거(그림 쪽 ref + 앞뒤 글 context)에
    없거나, 때·자리 수가 말이 안 되거나('13월'), 근거에 없는 가나·한자가 섞이거나, 때·범위가 근거와 어긋나는(_sensible) 문장은
    버린다(그래프 표 읽기가 값을 지어낸 것과 같은 위험 — 테스트_결과 6.5). 앞뒤 본문을 옮겨 적은 문장(4글자 묶음의 COPY_SHARE 이상이
    앞뒤 글에 있고 그림 쪽보다 많음 — 본문에 이미 있다)과 모델이 확신하지 못한 문장(conf 가 min_logprob 아래 — 그림 해석을 지어낸
    문장이 많다, 테스트_결과 10장)도 뺀다. 생각 칸은 떼고, 같은 문장은 한 번만, 길이 한도로 잘렸으면 끝 문장도 버리고, 그림을
    소개만 하는 문장은 빼고(_tidy), 남길 문장은 DESC_MAX_SENTENCES 개까지."""
    lines = [_MARK.sub("", ln).replace("**", "").strip() for ln in _answer(text).splitlines()]
    sents = list(dict.fromkeys(s.strip() for s in _SENT.split("\n".join(ln for ln in lines if ln)) if s.strip()))
    cut = [sents.pop()] if truncated and sents else []
    full = ref + "\n" + context if context else ref
    flat = "".join(full.split())
    nums = {_num_key(m) for m in _NUM.findall(full)}
    years = {int(y) for y in _YEAR.findall(full)} | {2000 + int(y) for y in _YY.findall(full)}
    days, weeks, scales = _days(full), _written(full, ("주", "주차")), _scales(ref)
    periods = {int(y): int(mo) for y, mo in _PERIOD.findall(full)}
    ctx_grams, ref_grams = _grams(context), _grams(ref)
    out = Grounded([], [], [], [], [])
    for s in sents:
        t = _tidy(s)
        g = _grams(t) if t else set()
        if t is None:
            out.meta.append(s)
        elif not (_grounded(t, full, flat, nums, years) and _sensible(t, flat, days, weeks, years, periods, scales)):
            out.dropped.append(s)
        elif g and len(g & ctx_grams) >= COPY_SHARE * len(g) and len(g & ctx_grams) > len(g & ref_grams):
            out.copied.append(s)
        elif min_logprob is not None and conf and (c := conf(s)) is not None and c < min_logprob:
            out.unsure.append(s)
        else:
            out.kept.append(t)
    return out._replace(kept=out.kept[:DESC_MAX_SENTENCES], dropped=out.dropped + cut)


# ---- 그림 다시 읽기(그림 속 글자 — 테스트_결과 14.10) ----

OCR_MIN_PIXELS = 112_896      # PaddleOCR-VL 에 보내는 화소 하한(PaddleX 기본)
OCR_NEAR = 0.8                # 엔진 글 줄과 다시 읽은 줄의 글자 일치가 이만큼 이상이면 같은 줄 — 다시 읽은 쪽을 둔다('주세' → '추세')
_SPOT_TEXT = re.compile(r"<\|TEXT_START\|>(.*?)<\|TEXT_END\|>", re.S)
_SPOT_LOC_BLOCK = re.compile(r"<\|LOC_BEGIN\|>(.*?)<\|LOC_END\|>", re.S)
_SPOT_LOC = re.compile(r"<\|LOC_(\d+)\|>")
_DIGITS = re.compile(r"\d+")


def ocr_targets(doc: Document, cfg) -> list[tuple[Page, Block]]:
    """다시 읽을 그림 — 살린 쪽의 살린 그림 블록 가운데 types 갈래(기본 설정은 그래프 chart 만)이고 논리 페이지 넓이의 min_area 이상.
    엔진 글이 끝까지 읽힌 그림도 읽는다(값 표지를 건너뛰거나 한두 글자를 잘못 읽고도 정상으로 끝난다 — 실측 test4.pdf 5쪽·test3.pdf 7쪽)."""
    out = []
    for page in (p for p in doc.pages if p.keep):
        w, h = page_size(page)
        for b in page.blocks:
            if b.type not in cfg.types or not b.keep or b.merged_into:
                continue
            x0, y0, x1, y1 = b.bbox
            if (x1 - x0) * (y1 - y0) >= cfg.min_area * w * h:
                out.append((page, b))
    return out


def ocr_region(page: Page, b: Block, pad: float) -> tuple[float, float, float, float]:
    """다시 그릴 시트 영역(표시 pt) — 그림 상자를 pad 만큼 넓혀(검출 상자가 그림 맨 위 제목 줄을 몇 화소 자른다 — 실측 test3.pdf
    12쪽) 시트 안으로 자른다."""
    x0, y0, x1, y1 = b.bbox
    w, h = page.size
    return round(max(0.0, x0 - pad), 2), round(max(0.0, y0 - pad), 2), round(min(w, x1 + pad), 2), round(min(h, y1 + pad), 2)


def ocr_key(cfg, b: Block, region) -> str:
    """읽기 기록의 열쇠 — 모델·블록·영역·dpi·물음·생성 한도·화소 상한·되풀이 벌점이 같아야 같은 읽기다. 그림 글은 넣지 않는다(조립
    규칙이 그림 글을 바꿔도 기록을 찾는다)."""
    s = json.dumps([cfg.model, b.id, list(region), cfg.dpi, cfg.prompt, cfg.max_tokens, cfg.max_pixels, cfg.repetition_penalty],
                   ensure_ascii=False)
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def spotting_lines(raw: str) -> list[str]:
    """'Spotting:' 응답 → 글 줄 — '<|TEXT_START|>글<|TEXT_END|>' 뒤에 위치 토큰 8개 이상이 든 짝만, 그런 짝이 없으면 위치 토큰 8개마다
    그 앞 글(PaddleX post_process_for_spotting 과 같다)."""
    out = [t.strip() for t, loc in zip(_SPOT_TEXT.findall(raw), _SPOT_LOC_BLOCK.findall(raw)) if len(_SPOT_LOC.findall(loc)) >= 8]
    if not out:
        locs, last = list(_SPOT_LOC.finditer(raw)), 0
        for i in range(0, len(locs) - 7, 8):
            out.append(raw[last:locs[i].start()].strip())
            last = locs[i + 7].end()
    return [t for t in out if t]


def _flat(s: str) -> str:
    return "".join(unicodedata.normalize("NFKC", s).split())


def _inside(e: str, r: str) -> bool:
    """e 가 r 안에 든다 — 수의 한가운데에서 잘리지 않게('5명미만' 은 '10~15명미만' 안에, '1,365' 는 '11,365명' 안에 들지 않는다)."""
    return re.search(("(?<!\\d)" if e[:1].isdigit() else "") + re.escape(e) + ("(?!\\d)" if e[-1:].isdigit() else ""), r) is not None


def _near(e: str, r: str) -> bool:
    """엔진 줄 e 를 다시 읽은 줄 r 이 다르게 읽은 것 — 띄어쓰기를 빼고 같거나, e(넷 글자 이상)가 r 안에 들었거나, 숫자 덩어리가 같고
    r 이 e 안에 들지 않으며 글자 일치가 OCR_NEAR 이상('주세' → '추세'). 숫자만 다른 줄('2025년'·'2026년')과 r 에 없는 글이 더 붙은
    e('온열질환자수(명) 폭염일수')는 다른 줄이다."""
    e, r = _flat(e), _flat(r)
    return e == r or (len(e) >= 4 and _inside(e, r)) or (r not in e and _DIGITS.findall(e) == _DIGITS.findall(r)
                                                   and SequenceMatcher(None, e, r).ratio() >= OCR_NEAR)


def _beside(page: Page, b: Block, region) -> list[str]:
    """다시 읽은 영역에 걸친 살린 글 블록과 그림 제목의 글(띄어쓰기 뺀) — 넓힌 영역에 든 제목·단위 줄을 그림 글에 또 넣지 않게."""
    x0, y0, x1, y1 = region
    return [_flat(o.text) for o in page.blocks if o is not b and o.keep and not o.merged_into and o.text
            and (o.id == b.caption_id or (o.bbox[0] < x1 and o.bbox[2] > x0 and o.bbox[1] < y1 and o.bbox[3] > y0))]


def attach_figure_ocr(doc: Document, cache: dict[str, dict], cfg) -> Document:
    """그림 다시 읽기 기록(raw/figure_ocr.jsonl)을 그림 글에 합친다 — 같은 열쇠의 기록에서 다시 읽은 줄(길이 한도로 잘렸으면 되풀이
    앞 줄만 — salvage, 못 살리면 엔진 글 그대로)을 앞에, 엔진 글 줄 가운데 다시 읽은 줄이 대신하지 않은 것(다시 읽기가 놓친 줄·그림에
    묶인 라벨 블록)을 뒤에 둔다. 다시 읽은 줄 하나는 엔진 줄 하나만 대신한다 — 똑같은 줄끼리 먼저, 그다음 _near('평년보다 높음'을
    다시 읽었다고 '평년보다 낮음'이 지워지지 않게). 다시 읽은 줄 가운데 가나·기호뿐인 줄(_noise 와 같다), 한자·가나가 섞인 줄(한국
    문서 그림에서 다시 읽기가 한글을 바꿔 쓴 글 — 실측 test2~5 Spotting: 그래프 77줄 모두 틀림, '4月'·깨진 범례), 그림 제목·영역에
    걸친 글 블록에 이미 있는 줄은 뺀다(엔진 글 줄은 거르지 않는다). 끝까지 읽었으면 잘림 표시를 지운다(남은 글이 다 읽은 글이다). 기록이 없는 그림은 표시만 해 QC 가
    센다(kdp figure-ocr 로 채운다)."""
    if not cfg.enabled:
        return doc
    doc = doc.model_copy(deep=True)
    body = body_text(doc)
    for page, b in ocr_targets(doc, cfg):
        region = ocr_region(page, b, cfg.pad_pt)
        hit = cache.get(ocr_key(cfg, b, region))
        if hit is None:
            b.provenance = (b.provenance or []) + [{"rule": "figure_ocr_missing"}]
            continue
        cut = hit.get("finish_reason") == "length"
        text = norm_text(hit.get("text") or "", strict=True)
        lines = salvage(text, b.type, body) if cut else [s.strip() for s in text.split("\n") if s.strip()]
        near = _beside(page, b, region)
        lines = [ln for ln in lines or [] if _letters(ln) and not _FOREIGN.search(ln)
                 and not (len(_flat(ln)) >= 4 and any(_inside(_flat(ln), t) for t in near))]
        rec = {"rule": "figure_ocr", "model": cfg.model, **({"truncated": "length"} if cut else {})}
        if not lines:
            b.provenance = (b.provenance or []) + [{**rec, "added": 0}]
            continue
        engine = [s.strip() for s in (b.figure_text or "").split("\n") if s.strip()]
        free, left, rest = [_flat(r) for r in lines], [], []
        for e in engine:                               # 똑같은 줄끼리 먼저
            if _flat(e) in free:
                free.remove(_flat(e))
            else:
                left.append(e)
        for e in left:                                 # 다시 읽은 줄 하나는 엔진 줄 하나만 대신한다
            r = next((r for r in free if _near(e, r)), None)
            if r is None:
                rest.append(e)
            else:
                free.remove(r)
        b.provenance = (b.provenance or []) + [{**rec, "added": len(free), "replaced": len(engine) - len(rest)}]
        b.figure_text, b.truncated = "\n".join(lines + rest), "length" if cut else None
    return doc


def attach_descriptions(doc: Document, cache: dict[str, dict], cfg) -> Document:
    """그림 설명(D19)을 붙인다 — 응답 기록(raw/describe.jsonl)에서 같은 열쇠를 찾아 근거 검사(ground)를 통과한 문장만
    figure_desc 로, 뺀 문장은 까닭별로(dropped·meta·copied·unsure) 기록에 남긴다. 응답이 없는 그림은 표시만 해 QC 가 센다(kdp
    describe 로 채운다)."""
    if not cfg.enabled:
        return doc
    doc = doc.model_copy(deep=True)
    for page, b in describe_targets(doc, cfg):
        hit = cache.get(desc_key(cfg.model, b, desc_prompt(doc, page, b), cfg.max_tokens))
        if hit is None:
            b.provenance = (b.provenance or []) + [{"rule": "figure_desc_missing"}]
            continue
        ref, ctx = _reference(doc, page, b)
        g = ground(hit["text"], ref, hit.get("finish_reason") == "length", context=ctx,
                   conf=sentence_logprob(hit["text"], hit.get("logprobs")), min_logprob=cfg.min_logprob)
        b.figure_desc = " ".join(g.kept) or None
        b.provenance = (b.provenance or []) + [{"rule": "figure_desc", "model": cfg.model,
                                                **{k: v for k, v in g._asdict().items() if k != "kept" and v}}]
    return doc
