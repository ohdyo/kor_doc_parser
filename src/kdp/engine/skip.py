"""훅 C — VLM 에 보낼 블록 고르기(설계 3.2). 로고와, PDF 글자가 덮는 글 블록은 보내지 않는다.
쪽마다 검증 표본 3개(글자가 가장 많은 본문, 글자 크기가 다른 블록, 그다음 많은 본문)는 보내 레이어가 맞는지 조립 때 확인한다(textlayer)."""
from __future__ import annotations

from ..textlayer.extract import _PUA, PUA_MAP, TEXT_LABELS, plausible

LOGO_LABELS = {"header_image", "footer_image"}
NOT_SAMPLE = {"header", "footer", "number"}
SAMPLES = 3                   # 검증 표본 수 — 글자가 가장 많은 본문, 글자 크기가 다른 블록, 그다음 많은 본문
LINE_GAP = 2.6                # 줄 사이가 글자 높이의 이 배수보다 벌면 레이어에 없는 줄이 낀 것(채운 입력란·윤곽선 글자 등)
WORD_GAP = 3.0                # 한 줄 안 빈틈이 이 배수보다 넓으면 레이어에 없는 글자가 낀 것


def _median(xs: list[float]) -> float:
    return sorted(xs)[len(xs) // 2]


def _lines(inside):
    rows, cur = [], []
    for item in sorted(inside, key=lambda c: (c[0][1] + c[0][3]) / 2):
        if cur and abs((item[0][1] + item[0][3]) / 2 - (cur[-1][0][1] + cur[-1][0][3]) / 2) > 0.5 * item[2]:
            rows.append(cur)
            cur = []
        cur.append(item)
    return rows + [cur] if cur else rows


def _covered(box, inside) -> bool:
    """레이어 글자가 블록을 빈 데 없이 덮나 — 바깥 범위와 함께 줄 사이·줄 안 빈틈도 본다."""
    if len(inside) < 2:
        return False
    x0, y0, x1, y1 = box
    h = _median([c[2] for c in inside])
    if (x1 - x0) - (max(c[0][2] for c in inside) - min(c[0][0] for c in inside)) > max(0.2 * (x1 - x0), 2 * h):
        return False
    if (y1 - y0) - (max(c[0][3] for c in inside) - min(c[0][1] for c in inside)) > max(0.2 * (y1 - y0), 1.5 * h):
        return False
    rows = _lines(inside)
    centers = [sum((c[0][1] + c[0][3]) / 2 for c in r) / len(r) for r in rows]
    if any(b - a > LINE_GAP * h for a, b in zip(centers, centers[1:])):
        return False
    for r in rows:
        xs = sorted(r, key=lambda c: c[0][0])
        if any(b[0][0] - a[0][2] > WORD_GAP * h for a, b in zip(xs, xs[1:])):
            return False
    return True


def _unexplained(box, inside, gray) -> bool:
    """블록 안 잉크 중 레이어 글자 상자로 설명되지 않는 것이 글자 하나 분량을 넘나 — 채운 입력란 값·윤곽선 글자·그림 속 글자."""
    hgt, wid = gray.shape[:2]
    x0, y0, x1, y1 = (int(max(0, box[0])), int(max(0, box[1])), int(min(wid, box[2])), int(min(hgt, box[3])))
    if x1 <= x0 or y1 <= y0:
        return False
    ink = gray[y0:y1, x0:x1] < 160
    boxes = [(max(0, int(b[0]) - 2 - x0), max(0, int(b[1]) - 2 - y0), max(0, int(b[2]) + 3 - x0), max(0, int(b[3]) + 3 - y0))
             for b, _, _ in inside]
    one = _median([int(ink[t:bt, l:r].sum()) for l, t, r, bt in boxes])      # 글자 하나의 잉크
    for l, t, r, bt in boxes:
        ink[t:bt, l:r] = False
    return int(ink.sum()) > max(one, 1)


class SkipPlan:
    def __init__(self, logos: bool, chars: list[tuple[tuple[float, float, float, float], str]] | None):
        self.logos = logos
        self.chars = chars                    # 보이는 레이어 글자 (상자(그 쪽 이미지 px), 글자). None 이면 레이어로 건너뛰지 않는다

    def choose(self, blocks: list[dict], image=None) -> dict[int, str]:
        """건너뛸 블록 {블록 순번: 이유}. image(그 쪽 그림)가 있으면 레이어로 설명되지 않는 잉크가 있는 블록은 남긴다."""
        gray = image.mean(axis=2) if image is not None and image.ndim == 3 else image
        skip, cands = {}, []
        for j, blk in enumerate(blocks):
            if blk.get("img") is None:
                continue
            if self.logos and blk["label"] in LOGO_LABELS:
                skip[j] = "logo"
                continue
            if not self.chars or blk["label"] not in TEXT_LABELS or blk.get("merge_aligns"):
                continue                      # 여러 블록을 묶은 대표(merge_aligns 가 빈 목록이 아님)는 구성원 영역까지 VLM 이 읽어야 한다
            x0, y0, x1, y1 = blk["box"]
            inside = [(b, c, b[3] - b[1]) for b, c in self.chars
                      if x0 <= (b[0] + b[2]) / 2 <= x1 and y0 <= (b[1] + b[3]) / 2 <= y1]
            text = "".join(c for _, c, _ in inside)
            if (_covered(blk["box"], inside) and plausible(text, strict=True)
                    and not any(_PUA.match(c) and c not in PUA_MAP for c in text)
                    and (gray is None or not _unexplained(blk["box"], inside, gray))):
                cands.append((len(inside), _median([h for _, _, h in inside]), j))
        body = sorted((c for c in cands if blocks[c[2]]["label"] not in NOT_SAMPLE), reverse=True)
        if not body:                          # 검증 표본이 없으면 레이어를 확인할 수 없으니 건너뛰지 않는다
            return skip
        sample = {body[0][2]}
        if len(body) > 1:                     # 둘째 표본은 글자 크기가 다른 블록(제목 글꼴 등)부터, 셋째는 그다음 많은 본문
            h0 = body[0][1]
            sample.add(max(body[1:], key=lambda c: (abs(c[1] - h0) >= 0.2 * h0, c[0]))[2])
            sample.update(c[2] for c in body[1:] if c[2] not in sample and len(sample) < SAMPLES)
        skip.update({j: "text_layer" for _, _, j in cands if j not in sample})
        return skip
