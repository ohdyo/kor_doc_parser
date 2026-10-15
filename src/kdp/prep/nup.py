"""모아찍기(n-up) 감지·분할(기획서 5.4, 설계 1장 prep·4장 좌표).
시트를 같은 크기 칸(2×2·2×1·1×2)으로 나눴을 때 ① 칸이 쪽 모양(긴 변/짧은 변)이고 ② 칸 경계 양옆 띠가 시트 끝까지 비었고
③ 칸 둘 이상의 같은 자리(칸 맨 위·아래 줄)에 쪽번호가 있을 때만 나눈다 — 두 단 편집 쪽은 가운데 빈 띠가 있어도 쪽번호가 하나다.
④ 칸 가운데에 그 쪽번호와 같은 x·높이의 떨어진 짧은 줄이 또 있으면(쪽이 위아래로 쌓인 6·8쪽 모아찍기) 나누지 않는다 —
지원하지 않는 배치는 시트 그대로 둔다. 그 줄은 born-digital 이면 쪽번호 꼴 글, 스캔이면 아래가 쪽 경계 빈 띠여야 한다(표 제목과 가른다).
⑤ born-digital 칸 쪽번호 값은 순서를 정하는 데만 쓴다 — 읽기 차례로 이웃한 칸(앞 시트 마지막 칸 포함)이 하나도 n, n+1 로 잇지
않거나(나란히 놓인 표의 합계 183·195, 소책자 배열 8·1) 칸마다 같은 값이면(1 은 빼고 — 한 쪽짜리 문서들) 나누지 않고,
세로 우선(왼쪽 열 먼저)으로 더 잘 이어지면 그 순서로 읽는다.
born-digital 은 텍스트 레이어 글자 상자와 쪽번호 꼴 글로, 스캔은 세운 시트 그림의 잉크 투영(짧고 떨어진 맨 아래·맨 위 줄)으로 본다.
스캔의 위·아래 표식은 그림만으로 가를 수 없어(뒤집힌 시트의 아래 쪽번호는 위 표식이다) 시트 방향은 방향 판정(orient)이 정한다.
칸 순서는 세운 시트의 왼쪽 위부터 행 우선(세로 우선 배치는 열 우선)이고 cell = (열, 행)."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from ..io.render import render_scale

LAYOUTS = ((2, 2), (2, 1), (1, 2))   # (열, 행) — 쪽번호 칸 비율·빈 칸 수가 같으면 앞의 것
PAGE_RATIO = (1.2, 1.7)              # 칸 긴 변/짧은 변이 쪽 모양(A판 1.41·레터 1.29·반 레터 1.55) — 세로 A4 를 좌우로 가른 칸은 2.83
GUTTER = 0.01                        # 칸 경계 양옆으로 시트 길이의 이 비율 띠가 시트 끝까지 비어야 한다
EDGE = 0.15                          # 쪽번호는 칸 위·아래 이 비율 안의 맨 끝 줄
POS_TOL = 0.04                       # 칸끼리 쪽번호 자리 차이(칸 크기 비율)
MIN_NUMBERED = 2                     # 같은 자리에 쪽번호가 있는 칸 수
SEG_GAP = 1.5                        # 한 줄 안 빈틈이 글자 높이의 이 배수보다 넓으면 다른 덩어리(꼬리말 글과 쪽번호)
PAGE_NO = re.compile(r"(?:[-‐–—−－]?\d{1,4}[-‐–—−－]?|[-‐–—−－][ivxlc]{1,6}[-‐–—−－]|[ⅰ-ⅹ]|[Ⅰ-Ⅻ]{1,3}[-‐–—−－]\d{1,3})"
                     r"(?:/\d{1,4})?", re.I)   # 공백을 뺀 글 — '- 3 -'·'3'·'3 / 12'·'- iv -'·'Ⅲ-2'
INK = 160                            # 스캔: 이보다 어두우면 잉크
TRIM = 0.01                          # 스캔: 시트 가장자리(스캐너 그림자)는 보지 않는다
NOISE = 0.002                        # 한 줄의 잉크(글자)가 그 길이의 이 비율 이하면 빈 줄(먼지)
BAND_JOIN = 0.004                    # 스캔: 이 비율(칸 높이) 이하의 빈 행은 한 줄 안(작은 글자의 가는 획)으로 보고 잇는다
DUST = 0.01                          # 스캔: 잉크 화소가 칸 폭(px)의 이 비율보다 적은 줄은 먼지
NUMBER_MAX_W = 0.25                  # 스캔: 쪽번호 줄 폭 상한(칸 폭 비율)
NUMBER_GAP = 2.0                     # 스캔: 쪽번호 줄과 이웃 줄 사이가 그 줄 높이의 이 배수 이상
SAME_H = 0.3                         # 쌓인 쪽: 칸 가운데 줄 높이가 쪽번호 줄 높이와 이 비율 안
STACK_GAP = 5.0                      # 쌓인 쪽(스캔): 쪽 경계 쪽 빈 띠가 그 줄 높이의 이 배수 이상(위 쪽 아래 여백 + 아래 쪽 위 여백) —
                                     # 합성 6·8쪽 모아찍기 5.7~9.6, 가운데 정렬 표·그림 제목 아래는 2.4~4.4
STACK_AT = (1 / 3, 1 / 2, 2 / 3)     # 쌓인 쪽(스캔): 그 빈 띠가 지나는 칸 높이 자리 — 2×4 를 2×2 로 보면 1/2, 2×3 이면 1/3·2/3
PLAIN = re.compile(r"[-‐–—−－]?(\d{1,4})[-‐–—−－]?(?:/\d{1,4})?")   # 아라비아 숫자 쪽번호 — 값을 읽는다


@dataclass(frozen=True)
class Layout:
    cols: int
    rows: int
    mark: tuple[str, float, float]   # 칸들이 함께 쓴 쪽번호 자리(위·아래, 칸 안 x·y 비율) — 같은 문서 다음 시트가 이어 받는다
    by_col: bool = False             # 세로 우선(왼쪽 열 위→아래 먼저) — born-digital 쪽번호 값으로 정한다
    last: int | None = None          # 읽기 차례 마지막 칸의 쪽번호 값(born-digital) — 다음 시트 첫 칸이 이어 받는다(12 → [13, 1])


def cells(size, cols: int, rows: int, by_col: bool = False) -> list[tuple[tuple[int, int], tuple[float, float, float, float]]]:
    """[(칸(열, 행), 영역 pt)] — 왼쪽 위부터 행 우선(by_col 이면 열 우선)."""
    w, h = size
    order = [(c, r) for c in range(cols) for r in range(rows)] if by_col else [(c, r) for r in range(rows) for c in range(cols)]
    return [((c, r), (c * w / cols, r * h / rows, (c + 1) * w / cols, (r + 1) * h / rows)) for c, r in order]


def cell_scale(size, n: int, dpi: int, max_pixels: int) -> float:
    """칸 하나(size pt)의 배율 — 시트 크기 쪽을 n 칸에 줄여 놓은 것으로 보고 그 쪽 기준 dpi 가 되게(설계 4장). 화소 상한은 칸 그림 기준."""
    f = math.sqrt(n)
    return render_scale((size[0] * f, size[1] * f), dpi, max_pixels) * f


def _clear(mask: np.ndarray, cols: int, rows: int) -> bool:
    """칸 경계 양옆 띠가 시트 끝까지 비었다 — 띠 안 세로·가로줄마다 잉크(글자)가 먼지 수준 이하."""
    hgt, wid = mask.shape
    for c in range(1, cols):
        x, g = round(c * wid / cols), max(1, round(GUTTER * wid))
        if (mask[:, x - g:x + g].sum(axis=0) > NOISE * hgt).any():
            return False
    for r in range(1, rows):
        y, g = round(r * hgt / rows), max(1, round(GUTTER * hgt))
        if (mask[y - g:y + g].sum(axis=1) > NOISE * wid).any():
            return False
    return True


def _text_mark(chars, region):
    """칸 맨 아래(또는 맨 위) 줄에 쪽번호 꼴 덩어리가 하나 있으면 그 자리 (위·아래, x 비율, y 비율, 값) — born-digital.
    값은 아라비아 숫자 쪽번호의 수(로마 숫자·'Ⅲ-2' 꼴이면 None). 한 줄에 쪽번호가 둘이면 칸 안에 쪽이 둘 있는 것(배치가 너무 크다)이라 보지 않는다."""
    if not chars:
        return None
    x0, y0, x1, y1 = region
    h = sorted(b[3] - b[1] for b, _ in chars)[len(chars) // 2]
    lines = []
    for item in sorted(chars, key=lambda c: c[0][1] + c[0][3]):
        b, last = item[0], lines[-1][-1][0] if lines else None
        if last and abs((b[1] + b[3]) - (last[1] + last[3])) / 2 <= 0.5 * h:
            lines[-1].append(item)
        else:
            lines.append([item])
    for edge, line in (("bottom", lines[-1]), ("top", lines[0])):
        xs = sorted(line, key=lambda c: c[0][0])
        segs = [[xs[0]]]
        for a, b in zip(xs, xs[1:]):
            if b[0][0] - a[0][2] > SEG_GAP * h:
                segs.append([])
            segs[-1].append(b)
        nums = [s for s in segs if PAGE_NO.fullmatch("".join(c for _, c in s))]
        if len(nums) != 1:
            continue
        boxes = [b for b, _ in nums[0]]
        cx = (min(b[0] for b in boxes) + max(b[2] for b in boxes)) / 2
        cy = (min(b[1] for b in boxes) + max(b[3] for b in boxes)) / 2
        ry = (cy - y0) / (y1 - y0)
        if ry >= 1 - EDGE if edge == "bottom" else ry <= EDGE:
            plain = PLAIN.fullmatch("".join(c for _, c in nums[0]))
            return edge, (cx - x0) / (x1 - x0), ry, int(plain.group(1)) if plain else None
    return None


def _bands(cell: np.ndarray) -> list[tuple[int, int]]:
    """잉크 줄 띠 [(시작, 끝)] — 잉크 행이 (좁은 빈 행을 건너) 이어진 곳. 잉크가 먼지만큼인 띠는 뺀다."""
    ink = cell.sum(axis=1)
    rows = np.flatnonzero(ink)
    if not len(rows):
        return []
    cut = np.flatnonzero(np.diff(rows) > 1 + round(BAND_JOIN * cell.shape[0])) + 1
    bands = [(int(r[0]), int(r[-1]) + 1) for r in np.split(rows, cut)]
    return [(a, b) for a, b in bands if ink[a:b].sum() > DUST * cell.shape[1]]


def _ink_mark(cell: np.ndarray, bands):
    """칸 맨 아래(또는 맨 위) 줄이 짧은 한 덩어리이고 이웃 줄과 떨어져 있으면 쪽번호 자리로 본다(스캔, 값은 None). 아래를 먼저 본다.
    한 덩어리 — 안쪽 빈틈이 SEG_GAP 보다 넓으면 쪽번호 둘(바깥쪽 쪽번호 두 쪽이 가운데에서 마주 본 것)이다."""
    if not bands:
        return None
    hgt, wid = cell.shape
    for edge, (a, b), gap in (("bottom", bands[-1], bands[-1][0] - (bands[-2][1] if len(bands) > 1 else 0)),
                              ("top", bands[0], (bands[1][0] if len(bands) > 1 else hgt) - bands[0][1])):
        xs = np.flatnonzero(cell[a:b].any(axis=0))
        y = (a + b) / 2 / hgt
        if ((y >= 1 - EDGE if edge == "bottom" else y <= EDGE) and xs[-1] + 1 - xs[0] <= NUMBER_MAX_W * wid
                and gap >= NUMBER_GAP * (b - a) and np.diff(xs).max(initial=1) - 1 <= SEG_GAP * (b - a)):
            return edge, float(xs[0] + xs[-1] + 1) / 2 / wid, y, None
    return None


def _stacked(cell: np.ndarray, bands, mark, nh: int) -> list[tuple[int, int, int, int, int, int]]:
    """칸 가운데(맨 위·아래 줄이 아닌 곳)에서 쪽번호 줄(높이 nh)과 같은 x·높이의 짧고 위아래로 떨어진 줄
    [(위 이웃 끝, 행 c, 행 d, 아래 이웃 시작, 열 x0, 열 x1)] — 쪽이 위아래로 쌓인 배치(6·8쪽 모아찍기를 2×2 로 본 것)의 후보.
    가운데 정렬 짧은 표·그림 제목도 이 꼴이라 _seam 으로 더 거른다."""
    wid, out = cell.shape[1], []
    for (_, above), (c, d), (below, _) in zip(bands, bands[1:], bands[2:]):
        xs = np.flatnonzero(cell[c:d].any(axis=0))
        if (abs((d - c) - nh) <= max(1, SAME_H * nh) and xs[-1] + 1 - xs[0] <= NUMBER_MAX_W * wid   # born-digital 은 1pt = 1칸이라 ±1
                and abs((xs[0] + xs[-1] + 1) / 2 / wid - mark[1]) <= POS_TOL
                and min(c - above, below - d) >= NUMBER_GAP * (d - c)):
            out.append((above, c, d, below, int(xs[0]), int(xs[-1]) + 1))
    return out


def _seam(line, hgt: int, mark, chars) -> bool:
    """쌓인 쪽 후보 줄이 정말 쪽 경계 위의 쪽번호다 — born-digital(chars = 칸 마스크 좌표의 글자)은 그 줄 글이 쪽번호 꼴이고,
    스캔(chars None)은 쪽 경계 쪽 빈 띠가 넓고(STACK_GAP) 쌓인 쪽 경계 자리(STACK_AT)를 지난다."""
    above, c, d, below, x0, x1 = line
    if chars is not None:
        return bool(PAGE_NO.fullmatch("".join(t for b, t in sorted(chars) if c <= (b[1] + b[3]) / 2 < d and x0 <= (b[0] + b[2]) / 2 < x1)))
    a, b = (d, below) if mark[0] == "bottom" else (above, c)              # 쪽 경계 쪽 빈 띠
    return b - a >= STACK_GAP * (d - c) and any(a <= f * hgt <= b for f in STACK_AT)


def _same(a, b) -> bool:
    return a[0] == b[0] and abs(a[1] - b[1]) <= POS_TOL and abs(a[2] - b[2]) <= POS_TOL


def _agree(marks, empty, ref):
    """(내용 있는 칸 중 같은 자리 쪽번호 칸 비율, 그 자리). ref(앞 시트 자리)면 내용 있는 칸이 모두 그 자리에 쪽번호가 있어야 한다."""
    found = [m for m in marks if m]
    if ref:
        ok = found and all(e or (m and _same(m, ref)) for m, e in zip(marks, empty))
        return (1.0, ref) if ok else None
    if not found:
        return None
    best = max(found, key=lambda m: sum(_same(m, o) for o in found))
    n = sum(_same(best, o) for o in found)
    return (n / sum(not e for e in empty), best) if n >= MIN_NUMBERED else None


def ink_mask(img: np.ndarray) -> np.ndarray:
    """스캔 시트 그림(BGR)의 잉크."""
    return img.min(axis=2) < INK


def key(size) -> tuple[int, int]:
    """앞 시트 배치를 찾는 열쇠 — 세운 시트 크기(pt)."""
    return round(size[0]), round(size[1])


def detect(size, chars=None, ink=None, prior: Layout | None = None) -> Layout | None:
    """모아찍기 배치 또는 None. size = 세운 시트 pt. chars = 보이는 레이어 글자 [(상자 pt, 글자)](born-digital) —
    없으면 ink(세운 시트 그림의 잉크, ink_mask)로 본다(스캔). prior = 같은 문서·같은 크기 앞 시트의 배치: 쪽이 덜 찬 마지막 시트처럼
    내용 있는 칸이 모두 그 자리에 쪽번호를 가지면 하나뿐이어도 나눈다."""
    w, h = size
    if chars is not None:                                     # 1 pt = 1 칸
        mask = np.zeros((math.ceil(h), math.ceil(w)), bool)
        for (l, t, r, b), _ in chars:
            mask[max(0, int(t)):max(0, math.ceil(b)), max(0, int(l)):max(0, math.ceil(r))] = True
    else:
        mask = ink.copy()
        ty, tx = max(1, round(TRIM * mask.shape[0])), max(1, round(TRIM * mask.shape[1]))
        mask[:ty] = mask[-ty:] = False
        mask[:, :tx] = mask[:, -tx:] = False
    sx, sy = mask.shape[1] / w, mask.shape[0] / h
    best = None
    tries = ([(prior.cols, prior.rows, prior.mark)] if prior else []) + [(c, r, None) for c, r in LAYOUTS]
    for cols, rows, ref in tries:
        cw, ch = w / cols, h / rows
        if not PAGE_RATIO[0] <= max(cw, ch) / min(cw, ch) <= PAGE_RATIO[1] or not _clear(mask, cols, rows):
            continue
        marks, empty, grid = [], [], []
        for _, (x0, y0, x1, y1) in cells(size, cols, rows):
            cell = mask[round(y0 * sy):round(y1 * sy), round(x0 * sx):round(x1 * sx)]
            bands, local = _bands(cell), None
            if chars is not None:
                inside = [(b, c) for b, c in chars if x0 <= (b[0] + b[2]) / 2 < x1 and y0 <= (b[1] + b[3]) / 2 < y1]
                marks.append(_text_mark(inside, (x0, y0, x1, y1)))
                empty.append(not inside)
                ox, oy = round(x0 * sx), round(y0 * sy)                           # 칸 마스크 좌표의 글자 — 쌓인 쪽 줄 글 읽기
                local = [((l * sx - ox, t * sy - oy, r * sx - ox, b * sy - oy), c) for (l, t, r, b), c in inside]
            else:
                marks.append(_ink_mark(cell, bands))
                empty.append(not bands)
            grid.append((cell, bands, local))
        got = _agree(marks, empty, ref)
        if not got:
            continue
        mark = got[1][:3]
        ends = [bands[-1] if mark[0] == "bottom" else bands[0] for m, (_, bands, _) in zip(marks, grid) if m and _same(m, mark)]
        nh = ends[0][1] - ends[0][0] if ends else 0          # 쪽번호 줄 높이 — 표식 칸의 끝 띠
        if nh and any(not e and any(_seam(s, cell.shape[0], mark, local) for s in _stacked(cell, bands, mark, nh))
                      for e, (cell, bands, local) in zip(empty, grid)):
            continue                                          # 쪽이 위아래로 쌓였다 — 더 잘게 나눈 배치(지원하지 않음)
        vals = [m and m[3] for m in marks]                    # 행 우선 칸 순서의 쪽번호 값(born-digital) — 바깥쪽 쪽번호처럼 자리가 달라도
        if any(m and m[3] is None for m in marks):
            vals = [None] * len(vals)                         # 로마 숫자 등이 섞였다 — 값을 쓰지 않는다
        down = [vals[r * cols + c] for c in range(cols) for r in range(rows)]                   # 열 우선 칸 순서
        by_col = prior.by_col if ref else False
        seen = {v for v in vals if v is not None}
        if len(seen) == 1 and sum(v is not None for v in vals) > 1 and seen != {1}:
            continue                                          # 칸마다 같은 값 — 쪽마다 되풀이되는 글(옆으로 보인 머리말의 연도 등)이지 쪽번호가 아니다
        if len(seen) > 1:                                     # 모두 1 이면(한 쪽짜리 문서 여럿) 순서를 못 정할 뿐 — 거르지 않는다
            prev = [prior.last] if ref and prior.last is not None else []
            across, along = _steps(prev + vals), _steps(prev + down)
            if not (across or along):
                continue                                      # 이웃 칸 값이 하나도 n, n+1 로 잇지 않는다 — 나란히 놓인 표의 합계, 소책자 배열
            by_col = along > across if along != across else by_col
        last = next((v for v in reversed(down if by_col else vals) if v is not None), None)
        # 내용 있는 칸 중 쪽번호 칸이 많은 배치(가로 시트의 2×1 을 2×2 로 보지 않게), 같으면 앞 시트 배치, 그다음 빈 칸이 적은 배치
        # (아래 절반이 빈 쪽 둘을 2×2 로 보면 칸 둘이 빈다)
        rank = (got[0], ref is not None, -sum(empty))
        if best is None or rank > best[0]:
            best = (rank, Layout(cols, rows, mark, by_col, last))
    return best[1] if best else None


def _steps(vals) -> int:
    """읽기 차례로 이웃한 값(빈 칸은 건너뛴다) 가운데 n, n+1 로 잇는 쌍의 수 — 값은 순서를 정하는 데만 쓰고, 이 수가 0 일 때만 거른다
    (쪽번호가 새로 시작하는 붙임 [23, 24, 1, 2] 는 받는다)."""
    seen = [v for v in vals if v is not None]
    return sum(b == a + 1 for a, b in zip(seen, seen[1:]))


def search(size, chars=None, ink=None, priors=None, turns=(0,)) -> dict[int, Layout]:
    """turns(np.rot90 횟수 k)마다 세운 시트에서 배치를 찾는다 → {k: 배치}. size·ink 는 돌리기 전 시트.
    스캔은 칸 위·아래 표식을 함께 보므로 180° 짝 방향 둘 다에서 찾을 수 있다 — 시트 방향은 이것으로 정하지 않는다."""
    out = {}
    for k in turns:
        sz = (size[1], size[0]) if k % 2 else tuple(size)
        lay = detect(sz, chars, None if ink is None else np.rot90(ink, k), (priors or {}).get(key(sz)))
        if lay:
            out[k] = lay
    return out
