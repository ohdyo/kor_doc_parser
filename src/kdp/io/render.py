"""쪽 렌더와 좌표 변환(설계 4장)."""
from __future__ import annotations

import math

import numpy as np

from .pdfium_lock import PDFIUM_LOCK


def render_scale(size_pt: tuple[float, float], dpi: int, max_pixels: int) -> float:
    """dpi 배율. 화소 상한을 넘는 쪽은 배율을 낮춘다(곁기록에 남는다)."""
    s = dpi / 72
    w, h = size_pt
    if w * h * s * s > max_pixels:
        s = math.sqrt(max_pixels / (w * h))
    return s


def render_page(page, scale: float) -> np.ndarray:
    """BGR uint8 (H, W, 3). 쪽의 /Rotate 는 PDFium 이 반영한다."""
    with PDFIUM_LOCK:
        bm = page.render(scale=scale)
        img = bm.to_numpy().copy()
        bm.close()
    return img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img


def user_to_display(x: float, y: float, bbox, rotation: int) -> tuple[float, float]:
    """PDF 사용자 공간(좌하단 원점) → 표시 pt(좌상단 원점, /Rotate 반영)."""
    l, b, r, t = bbox
    w, h = r - l, t - b
    u, v = x - l, y - b
    if rotation == 0:
        return u, h - v
    if rotation == 90:
        return v, u
    if rotation == 180:
        return w - u, v
    if rotation == 270:
        return h - v, w - u
    raise ValueError(f"rotation {rotation}")


def user_box_to_display(box, bbox, rotation: int) -> tuple[float, float, float, float]:
    l, b, r, t = box
    pts = [user_to_display(x, y, bbox, rotation) for x in (l, r) for y in (b, t)]
    xs, ys = zip(*pts)
    return min(xs), min(ys), max(xs), max(ys)


def px_box_to_pt(box, scale: float, origin=(0.0, 0.0)) -> tuple[float, float, float, float]:
    """논리 페이지 px → 시트 pt."""
    ox, oy = origin
    x1, y1, x2, y2 = box
    return ox + x1 / scale, oy + y1 / scale, ox + x2 / scale, oy + y2 / scale


def unrotate_box(box, k: int, size_px: tuple[int, int]):
    """np.rot90(img, k) 로 세운 이미지의 상자를 원래 이미지 좌표로 되돌린다. size_px = 세운 이미지 (W, H)."""
    x1, y1, x2, y2 = box
    w, h = size_px
    for _ in range(k % 4):
        # rot90 한 번: 원래 (x, y) = (W' - y', x'), 여기서 W' 는 원래 이미지 폭 = 세운 이미지 높이
        x1, y1, x2, y2 = h - y2, x1, h - y1, x2
        w, h = h, w
    return x1, y1, x2, y2
