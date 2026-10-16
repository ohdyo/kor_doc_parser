"""그림·차트(기획서 4.3 규칙 11)."""
from __future__ import annotations

import html as htmllib
import re

from ..geometry import local
from ..schema import Document
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
