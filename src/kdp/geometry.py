"""논리 페이지 좌표(설계 4장) — 위치 규칙(위·아래·들여쓰기·여백·같은 위치)은 이것으로 잰다.
정본 bbox 는 시트 pt 라 모아찍기 쪽에서 그대로 재면 아래 칸 쪽의 '위쪽'·면적 비율이 틀린다.
돌려 세운 스캔 쪽(page.rot90_k)은 세운 쪽 기준으로 돌려 잰다 — 시트 좌표의 '높이'는 세운 쪽의 너비다."""
from __future__ import annotations

from .schema import Page

Box = tuple[float, float, float, float]


def region(page: Page) -> Box:
    """시트 안 이 논리 페이지 영역(pt) — 모아찍기가 아니면 쪽 전체."""
    return tuple(page.n_up.region) if page.n_up else (0.0, 0.0, page.size[0], page.size[1])


def page_size(page: Page) -> tuple[float, float]:
    """세운 논리 페이지의 (너비, 높이)."""
    x0, y0, x1, y1 = region(page)
    w, h = x1 - x0, y1 - y0
    return (h, w) if (page.rot90_k or 0) % 2 else (w, h)


def local(page: Page, box) -> Box:
    """시트 pt → 세운 논리 페이지 pt(왼쪽 위가 원점). 돌리는 방향은 np.rot90(그림, k) 와 같다."""
    x0, y0, x1, y1 = region(page)
    w, h = x1 - x0, y1 - y0
    a, b, c, d = box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0
    for _ in range((page.rot90_k or 0) % 4):             # 반시계 90°: (x, y) → (y, w - x)
        a, b, c, d = b, w - c, d, w - a
        w, h = h, w
    return a, b, c, d
