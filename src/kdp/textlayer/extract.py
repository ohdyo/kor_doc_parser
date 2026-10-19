"""텍스트 레이어(D13, 설계 1장) — born-digital 쪽은 블록 안 PDF 글자를 쓴다. 쪽별 레이어 일치율은 QC 가 쓴다.

정답으로 쓰는 것은 '보이는' 글자뿐이다 — 안 그리는 글자(스캔 PDF 의 투명 OCR 레이어)와 쪽 그림에서 가려진 글자(가림 상자·흰 글자)는 뺀다.
글 블록은 레이어 글자와 모델 글이 양쪽으로 맞을 때만 바꾸고(그림 속 글자·채운 입력란·깨진 레이어는 모델 글을 둔다),
표는 칸마다 레이어 글자를 앞에서부터 차례로 맞춰 가까운 것만 바꾼다(표 구조는 모델 것을 그대로 쓴다).
"""
from __future__ import annotations

import html as htmllib
import math
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

import numpy as np

from ..io.pdf import PdfInfo
from ..io.render import user_box_to_display
from ..schema import Block, Document

TEXT_LABELS = {"text", "paragraph_title", "doc_title", "figure_title", "abstract", "aside_text", "content",
               "footnote", "vision_footnote", "reference", "reference_content", "algorithm",
               "header", "footer", "number"}
MIN_PAGE_CHARS = 20            # 보이는 글자가 이보다 적으면 레이어가 없는 쪽(스캔)으로 본다
MIN_AGREEMENT = 0.8            # 쪽: 레이어 글자 중 모델 글에도 있는 비율이 이보다 낮으면 레이어를 쓰지 않는다
VLM_IN_LAYER = 0.9             # 블록: 모델 글자 중 레이어에도 있는 비율 — 레이어에 없는 줄(채운 입력란 등)이 있으면 바꾸지 않는다
VLM_ONLY_RUN = 3               # 블록: 모델 글에만 있는 글자가 이만큼 이어지면 레이어에 없는 내용(이름·입력란 값)으로 본다
LAYER_IN_VLM = 0.8             # 블록: 레이어 글자 중 모델 글에도 있는 비율
MIX_GAP = 6.0                  # 한 줄 안 빈틈이 글자 높이의 이 배수보다 넓으면 서로 다른 영역이 섞인 블록으로 의심
CELL_MIN_CHARS = 4
CELL_MAX_ERR = 0.2             # 표 칸: 모델 글자 수의 20% 까지 다르면 같은 칸으로 본다
QUOTES = set("'\"‘’‚‛“”„‟`´′″°ʼ")   # 표 칸을 맞춰 볼 때 빼는 따옴표 — 모델이 칸의 따옴표('23)를 빠뜨리거나(실측 test2.pdf GPU) ° 로 읽는다(CPU)
LAYER_SKIP_COST = 0.03        # 표 맞춤: 칸 사이에서 건너뛴 레이어 글자 하나의 비용(모델이 행을 빠뜨린 경우)
UNMATCHED_COST = 0.5          # 표 맞춤: 못 맞춘 칸의 글자 하나당 비용
LENGTH_TIE = 0.01             # 표 맞춤: 거리가 같으면 칸 길이에 가까운 맞춤
MAX_STATES = 200
_PUA = re.compile("[\ue000-\uf8ff]")
# Symbol·Wingdings 기호 글꼴의 흔한 글머리표·체크 상자 — 이 밖의 사용자 정의 문자는 뜻을 모른다
PUA_MAP = {"\uf0b7": "•", "\uf0a7": "▪", "\uf06c": "●", "\uf06e": "■", "\uf06f": "□", "\uf071": "❑",
           "\uf076": "❖", "\uf0d8": "➢", "\uf0fc": "✔", "\uf0fd": "☒", "\uf0fe": "☑", "\uf0a8": "☐"}
_LATIN1_OK = set("°±²³·×÷µ¹¼½¾«»©®§¶")      # 한국어 문서에도 흔한 라틴-1 기호 — 나머지가 많으면 글자 대응이 깨진 것
_SKIP = re.compile(r"\s+|\.{2,}|[\-–•·▪◦○●■□◆◇▶►]")
_CELL = re.compile(r"(<t[dh][^>]*>)(.*?)(</t[dh]>)", re.S | re.I)
_KSX1001 = {bytes([hi, lo]).decode("euc-kr") for hi in range(0xB0, 0xC9) for lo in range(0xA1, 0xFF)}   # 상용 한글 2,350자


def plausible(text: str, strict: bool = False) -> bool:
    """레이어 글자가 그럴듯한가 — 모델 없이 보는 검사. 깨진 글자 대응표는 U+FFFD·뜻 모를 사용자 정의 문자,
    드문 한글, 라틴-1 글자(EUC-KR 바이트를 잘못 읽은 모양)로 나타난다. 통과해도 조립 때 검증 표본과 다시 맞춰 본다.
    strict 는 블록 하나(훅 C 로 VLM 을 건너뛸지) — 짧은 글도 드문 한글을 보고, 한자로 깨진 모양도 막는다."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    bad = sum(c == "\ufffd" or (_PUA.match(c) is not None and c not in PUA_MAP)
              or unicodedata.category(c) in ("Cc", "Cn") for c in chars)
    hangul = [c for c in chars if "\uac00" <= c <= "\ud7a3"]
    rare = sum(c not in _KSX1001 for c in hangul)
    latin = sum("\u00a0" <= c <= "\u00ff" and c not in _LATIN1_OK for c in chars)
    ok = (bad <= 0.05 * len(chars) and (len(hangul) < 20 or rare <= 0.1 * len(hangul))
          and not (latin >= 2 and latin > 0.2 * len(chars)))
    if ok and strict:
        hanja = sum("\u4e00" <= c <= "\u9fff" or "\uf900" <= c <= "\ufaff" for c in chars)
        ok = (len(hangul) < 4 or rare <= 0.25 * len(hangul)) and hanja <= 0.3 * len(chars)
    return ok


def match_key(s: str) -> str:
    """비교용 — 공백·목차 점선·글머리표를 빼고 NFKC."""
    return _SKIP.sub("", unicodedata.normalize("NFKC", s or ""))


def _recall(a: str, b: str) -> float:
    """a 의 글자 중 b 에도 있는 비율."""
    ca, cb = Counter(match_key(a)), Counter(match_key(b))
    n = sum(ca.values())
    return sum((ca & cb).values()) / n if n else 1.0


def agreement(a: str, b: str) -> float:
    """두 글의 글자 구성이 겹치는 비율(긴 쪽 기준)."""
    return min(_recall(a, b), _recall(b, a))


def layer_recall(layer: str, output: str) -> float:
    """레이어 글자 중 모델 출력에도 있는 비율 — 쪽 단위. 그림 속 글자처럼 모델만 읽은 글자는 깎지 않는다."""
    return _recall(layer, output)


def replaceable(layer: str, vlm: str) -> bool:
    """모델 글을 레이어 글로 바꿔도 되나 — 양쪽으로 맞아야 하고, 모델 글에만 있는 글자가 3자 이상 이어지면 안 된다.
    흩어진 한두 글자 차이는 모델 오독이지만, 이어진 덩어리는 레이어에 없는 내용(채운 입력란·빠진 줄·이름)이다."""
    if _recall(vlm, layer) < VLM_IN_LAYER or _recall(layer, vlm) < LAYER_IN_VLM:
        return False
    a, b = match_key(vlm), match_key(layer)
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag in ("delete", "replace") and (i2 - i1) - (j2 - j1 if tag == "replace" else 0) >= VLM_ONLY_RUN:
            return False
    return True


def visible(ch: dict) -> bool:
    """정답으로 쓸 글자 — 안 그리는 글자(inv)·쪽 그림에서 가려진 글자(hid)는 뺀다."""
    return not ch.get("inv") and not ch.get("hid")


def mark_hidden(chars: list[dict], img: np.ndarray, page_bbox, rotation: int, scale: float) -> list[dict]:
    """그린 쪽 그림에서 글자 자리가 비었거나(흰 글자·흰 가림 상자) 한 색으로 꽉 칠해졌으면(검은 가림 상자) hid 를 단다.
    가린 개인정보가 레이어 글자로 되살아나지 않게 한다. 'l·1·ㅡ' 처럼 원래 획이 꽉 찬 글자는 가로세로 비율로 뺀다."""
    gray = img.mean(axis=2) if img.ndim == 3 else img.astype(float)
    hgt, wid = gray.shape[:2]
    for ch in chars:
        if ch["gen"] or ch.get("inv") or ch["c"].isspace():
            continue
        l, t, r, b = user_box_to_display(ch["box"], page_bbox, rotation)
        x0, y0 = max(0, int(l * scale)), max(0, int(t * scale))
        x1, y1 = min(wid, math.ceil(r * scale)), min(hgt, math.ceil(b * scale))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        box = gray[y0:y1, x0:x1]
        ratio = (x1 - x0) / (y1 - y0)
        if ch["c"].isalnum() and box.size >= 16 and (       # 문장부호는 보지 않는다 — 내장 안 한 글꼴은 상자가 어긋난다
                255 - box.min() < 30 or (0.4 <= ratio <= 2.5 and box.std() < 8 and box.mean() < 200)):
            ch["hid"] = True
    return chars


def display_chars(chars: list[dict], page_bbox, rotation: int) -> list[tuple[str, tuple, bool]]:
    return [(c["c"], user_box_to_display(c["box"], page_bbox, rotation), c["gen"]) for c in chars if visible(c)]


def _join(chars, newline: bool, keep_unknown: bool = False) -> tuple[str, bool]:
    """PDFium 이 끼워 넣은 공백·줄바꿈을 살려 잇는다(이 문서들은 원래 공백 문자가 거의 없다).
    뜻 모를 사용자 정의 문자가 빠지면 lost=True — 줄 머리라도 체크 상자일 수 있어 그 글은 모델 글을 둔다(keep_unknown 이면 남긴다)."""
    out, lost = [], False
    for c, _, gen in chars:
        if gen:
            out.append("\n" if newline and c in "\r\n" else " ")
            continue
        if c == "\ufffd":
            continue
        if _PUA.match(c):
            if c in PUA_MAP:
                out.append(PUA_MAP[c])
            elif keep_unknown:
                out.append(c)
            else:
                lost = True
            continue
        out.append(c)
    lines = [" ".join(ln.split()) for ln in "".join(out).split("\n")]
    return "\n".join(ln for ln in lines if ln), lost


def _assign(blocks: list[Block], chars) -> dict[str, list[int]]:
    """글자 가운데가 든 블록 중 가장 작은 블록에 준다. 생성 문자는 바로 앞 글자를 따른다."""
    area = {b.id: (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]) for b in blocks}
    out: dict[str, list[int]] = {b.id: [] for b in blocks}
    owner = None
    for i, (c, box, gen) in enumerate(chars):
        if gen:
            if owner:
                out[owner].append(i)
            continue
        x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        inside = [b for b in blocks if b.bbox[0] - 1 <= x <= b.bbox[2] + 1 and b.bbox[1] - 1 <= y <= b.bbox[3] + 1]
        owner = min(inside, key=lambda b: area[b.id]).id if inside else None
        if owner:
            out[owner].append(i)
    return out


def mixed_regions(chars) -> bool:
    """블록 글자를 줄로 묶어 한 줄 안에 큰 빈틈이 있는지 — 서로 다른 영역이 한 블록에 합쳐진 의심(실측: PPT 쪽 가로 병합)."""
    real = sorted(((c, box) for c, box, gen in chars if not gen and not c.isspace()), key=lambda x: (x[1][1] + x[1][3]) / 2)
    if len(real) < 2:
        return False
    h = sorted(box[3] - box[1] for _, box in real)[len(real) // 2]
    lines, cur = [], [real[0]]
    for item in real[1:]:
        if abs((item[1][1] + item[1][3]) / 2 - (cur[-1][1][1] + cur[-1][1][3]) / 2) <= 0.5 * h:
            cur.append(item)
        else:
            lines.append(cur)
            cur = [item]
    lines.append(cur)
    for ln in lines:
        xs = sorted(ln, key=lambda x: x[1][0])
        if any(b[1][0] - a[1][2] > MIX_GAP * h for a, b in zip(xs, xs[1:])):
            return True
    return False


def _dist_row(p: str, t: str, free_start: bool = True) -> np.ndarray:
    """j 마다 p 와 t[..j] 의 최소 편집 거리 — free_start 면 t 의 어디서 시작해도 되고, 아니면 t 맨 앞에서 시작."""
    tt = np.frombuffer(t.encode("utf-32-le"), dtype=np.uint32)
    ar = np.arange(len(t) + 1)
    prev = np.zeros(len(t) + 1, dtype=np.int64) if free_start else ar.copy()
    for i, ch in enumerate(p, 1):
        cur = np.empty_like(prev)
        cur[0] = i
        cur[1:] = np.minimum(prev[:-1] + (tt != ord(ch)), prev[1:] + 1)
        prev = np.minimum.accumulate(cur - ar) + ar
    return prev


def _cands(p: str, flat: str, size: int | None = None) -> list[tuple[int, int, float]]:
    """p 가 맞을 수 있는 레이어 자리 (시작, 끝, 비용) — 거리 한도 안의 모든 끝·길이. 한도는 size(따옴표를 뺀 칸이면 뺀
    따옴표까지 센 원래 칸 길이) 기준. 비용에 길이 차이를 조금 더해, 거리가 같으면 칸 길이에 가까운 맞춤이 이긴다."""
    limit = int((size or len(p)) * CELL_MAX_ERR)
    row = _dist_row(p, flat)
    out = []
    for e in (int(x) for x in np.nonzero(row <= limit)[0]):
        span = min(e, len(p) + limit)
        back = _dist_row(p[::-1], flat[e - span:e][::-1], free_start=False)   # back[n] = 거리(p, flat[e-n:e])
        for n in range(max(1, len(p) - limit), span + 1):
            if back[n] <= limit:
                out.append((e - n, e, float(back[n]) + LENGTH_TIE * abs(n - len(p))))
    return out


def _align(keys: list[str | None], flat: str, sizes: list[int] | None = None) -> list[tuple[int, int] | None]:
    """칸들을 순서대로 레이어 자리에 한 번에 맞춘다 — 거리 합 + 칸 사이에서 건너뛴 레이어 글자 + 못 맞춘 칸이 가장 작은 배치.
    모델이 행을 빠뜨린 표, 비슷한 값이 이어지는 표, 칸 끝에 글자를 하나 더 읽은 경우를 함께 가른다."""
    states: dict[int, float] = {0: 0.0}                       # 마지막으로 맞춘 끝 → 비용
    steps = []
    for i, k in enumerate(keys):
        if k is None:
            steps.append(None)
            continue
        cands = _cands(k, flat, sizes[i] if sizes else None)
        new: dict[int, tuple[float, int, tuple[int, int] | None]] = {}
        for last, cost in states.items():
            c = cost + UNMATCHED_COST * len(k)
            if c < new.get(last, (float("inf"),))[0]:
                new[last] = (c, last, None)
            for st, e, d in cands:
                if st >= last and (c := cost + d + LAYER_SKIP_COST * (st - last)) < new.get(e, (float("inf"),))[0]:
                    new[e] = (c, last, (st, e))
        best = sorted(new.items(), key=lambda kv: kv[1][0])[:MAX_STATES]
        steps.append(dict(best))
        states = {e: v[0] for e, v in best}
    end = min(states, key=states.get)
    out: list[tuple[int, int] | None] = []
    for step in reversed(steps):
        if step is None:
            out.append(None)
            continue
        _, end_prev, pick = step[end]
        out.append(pick)
        end = end_prev
    return out[::-1]


def _flat(text: str) -> tuple[str, list[int]]:
    """공백·따옴표를 뺀 비교용 글자열과, 그 글자마다 text 의 위치 — 글자마다 NFKC 하되 늘어난 글자(㎡→m2)도 같은 위치를 가리킨다."""
    chars, where = [], []
    for i, c in enumerate(text):
        if not c.isspace():
            n = "".join(ch for ch in unicodedata.normalize("NFKC", c) if ch not in QUOTES)   # '℃' → '°C' 의 ° 도 뺀다
            chars.append(n)
            where.extend([i] * len(n))
    return "".join(chars), where


def _fix_table(html: str, chars) -> tuple[str, int]:
    """칸 글자를 레이어 글자로 — 칸 순서대로 한 번에 맞춘다. 바꾼 칸 수를 함께 돌려준다.
    레이어 순서가 표 순서와 맞지 않으면(맞춘 칸이 절반 미만) 고치지 않는다."""
    text = unicodedata.normalize("NFC", _join(chars, newline=False, keep_unknown=True)[0])   # 풀린 자모도 합친다
    flat, where = _flat(text)
    cells = list(_CELL.finditer(html))
    keys, sizes = [], []
    for m in cells:
        full = unicodedata.normalize("NFKC", "".join(htmllib.unescape(m.group(2).replace("\\n", " ")).split()))
        key = "".join(ch for ch in full if ch not in QUOTES)       # 글자 수 하한·허용 오차는 따옴표까지 센 칸 길이로
        keys.append(key if "<" not in m.group(2) and key and len(full) >= CELL_MIN_CHARS else None)
        sizes.append(len(full))
    picks = _align(keys, flat, sizes) if flat else [None] * len(keys)
    usable = sum(k is not None for k in keys)
    if not usable or sum(p is not None for p in picks) < 0.5 * usable:
        return html, 0
    fixed, parts, last = 0, [], 0
    for m, pick in zip(cells, picks):
        if pick is None:
            continue
        s, e = pick
        cell = htmllib.unescape(m.group(2)).strip()
        a, z = where[s], where[e - 1] + 1                           # 따옴표를 빼고 맞췄다 — 칸 앞뒤에 붙은 레이어 따옴표는 되살린다
        while a > 0 and text[a - 1] in QUOTES and (cell[:1] in QUOTES or a < 2 or text[a - 2].isspace()):
            a -= 1
        while z < len(text) and text[z] in QUOTES and (cell[-1:] in QUOTES or z + 1 >= len(text) or text[z + 1].isspace()):
            z += 1
        piece = " ".join(text[a:z].split())
        if any(_PUA.match(c) and c not in PUA_MAP for c in piece):
            continue                                                # 뜻 모를 기호가 든 칸은 모델 글을 둔다
        new = htmllib.escape(piece, quote=False)
        parts.append(html[last:m.start(2)] + new)
        last = m.end(2)
        fixed += new != m.group(2)
    return "".join(parts) + html[last:], fixed


def _skipped(b: Block) -> bool:
    return any(p.get("rule") == "vlm_skipped" and p.get("reason") == "text_layer" for p in b.provenance or [])


def _vlm_text(b: Block) -> str:
    return b.text or b.figure_text or htmllib.unescape(re.sub(r"<[^>]+>", " ", b.html or ""))


def apply_text_layer(doc: Document, layers: dict[int, list[dict]], info: PdfInfo, prefer: bool, trust: bool = False) -> Document:
    """쪽마다 born_digital·레이어 일치율을 적고, prefer 면 맞는 블록의 글자를 레이어 글자로 바꾼다. trust 는 그림만 설정(vlm_skip 의
    non_figure — 그림 해석 학습 자료 준비, 기획서 7.10)용: 글 블록을 모두 건너뛰어 일치율을 잴 검증 표본이 없는 born_digital 쪽도
    레이어로 채운다(블록마다 plausible 검사는 그대로)."""
    doc = doc.model_copy(deep=True)
    geo = {p.sheet_no: p for p in info.pages}
    for page in doc.pages:
        g = geo[page.sheet_no]
        raw = layers.get(page.sheet_no, [])
        if page.n_up:                                               # 모아찍기: 이 논리 페이지 영역의 글자만
            x0, y0, x1, y1 = page.n_up.region

            def _in(c):
                box = user_box_to_display(c["box"], g.bbox, g.rotation)
                return x0 <= (box[0] + box[2]) / 2 <= x1 and y0 <= (box[1] + box[3]) / 2 <= y1

            raw = [c for c in raw if _in(c)]
        real = [c for c in raw if not c["gen"] and not c["c"].isspace()]
        seen = [c for c in real if visible(c)]
        page.born_digital = len(seen) >= MIN_PAGE_CHARS and len(seen) >= 0.5 * len(real)   # 투명 OCR 레이어 쪽은 아니다
        chars = display_chars(raw, g.bbox, g.rotation)
        blocks = [b for b in page.blocks if b.merged_into is None]
        skipped = [b for b in blocks if _skipped(b)]
        targets = [b for b in blocks if b.type in TEXT_LABELS or (b.type == "table" and b.html)]
        owned = _assign(targets, chars)
        if page.born_digital:                                       # 글자층 있는 쪽인데 자리에 글자가 없는 글 블록 — 그림에 그려진 글
            bare = [b for b in targets if b.type in TEXT_LABELS and not any(not chars[i][2] for i in owned[b.id])]
            if 2 * len(bare) <= sum(b.type in TEXT_LABELS for b in targets):  # 글 블록 절반 이상이 글자층인 쪽만 — 스캔 쪽에 머리말만 글자인 쪽은 아니다
                for b in bare:
                    b.provenance = (b.provenance or []) + [{"rule": "no_layer_chars"}]
        read = None
        if skipped:                                                 # 훅 C: 일치율은 VLM 이 읽은 글 블록(검증 표본)으로만 — 표는 뺀다
            read = [b for b in targets if b.type in TEXT_LABELS and not _skipped(b)]
            layer = "".join(chars[i][0] for b in read for i in owned[b.id] if not chars[i][2])
            vlm = " ".join(_vlm_text(b) for b in read)
        else:
            layer, vlm = "".join(c[0] for c in chars if not c[2]), " ".join(_vlm_text(b) for b in blocks)
        page.text_layer_agreement = round(layer_recall(layer, vlm), 4) if page.born_digital and layer else None
        unverified = trust and prefer and page.born_digital and bool(skipped) and not read   # 그림만 설정 — 표본 없이 레이어를 믿는다
        if not unverified and (not prefer or not page.text_layer_agreement or page.text_layer_agreement < MIN_AGREEMENT):
            for b in skipped:                                       # 글자가 없는 채로 남는다 — 레이어를 끄고 다시 처리해야 한다
                b.error = "VLM 을 건너뛴 블록인데 텍스트 레이어를 믿을 수 없다 — text_layer.mode off 로 다시 처리"
            continue
        for b in targets:
            mine = [chars[i] for i in owned[b.id]]
            if b.type == "table":
                if any(not c[2] for c in mine):
                    agree = round(layer_recall(_join(mine, newline=False)[0], re.sub(r"<[^>]+>", " ", b.html)), 4)
                    new_html, n = _fix_table(b.html, mine)
                    b.provenance = (b.provenance or []) + [{"rule": "text_layer", "agreement": agree, "cells": n}]
                    b.html = new_html                               # agreement 는 표 갈래(인식 실패) 판정이 쓴다
                continue
            if b.type not in ("content", "header", "footer", "number") and mixed_regions(mine):
                b.provenance = (b.provenance or []) + [{"rule": "region_mix_suspect"}]
            new, lost = _join(mine, newline=b.type == "content")
            if _skipped(b):                                         # 모델 글이 없다 — 블록 레이어 글을 다시 검사한다
                if new and not lost and plausible(new, strict=True):
                    b.text = new
                    b.provenance = (b.provenance or []) + [{"rule": "text_layer", "before": None}]
                else:
                    b.error = "VLM 을 건너뛴 블록인데 레이어 글이 비었거나 깨졌다 — text_layer.mode off 로 다시 처리"
                continue
            if new and not lost and new != b.text and (replaceable(new, b.text) if b.text else plausible(new)):
                b.provenance = (b.provenance or []) + [{"rule": "text_layer", "before": b.text}]
                b.text = new
    return doc
