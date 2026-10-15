"""논리 페이지 좌표 — 모아찍기 영역과 돌려 세운 스캔 쪽(np.rot90 과 같은 방향)."""
import numpy as np

from kdp.geometry import local, page_size
from kdp.io.render import unrotate_box
from kdp.schema import NUp, Page


def test_local_follows_np_rot90_and_region():
    w, h = 40, 60
    img = np.zeros((h, w), np.uint8)
    img[10:20, 5:15] = 1                                             # 시트 좌표 상자 (5, 10, 15, 20)
    for k in range(4):
        up = np.rot90(img, k)
        ys, xs = np.nonzero(up)
        want = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        page = Page(page_no=1, sheet_no=1, size=(w, h), rot90_k=k or None)
        cell = Page(page_no=2, sheet_no=1, size=(w + 100, h), rot90_k=k or None,
                    n_up=NUp(cols=2, rows=1, cell=(1, 0), region=(100, 0, 100 + w, h)))
        assert local(page, (5, 10, 15, 20)) == want and local(cell, (105, 10, 115, 20)) == want
        assert page_size(page) == page_size(cell) == (up.shape[1], up.shape[0])
        assert unrotate_box(want, k, page_size(page)) == (5, 10, 15, 20)   # 엔진 좌표 되돌리기와 맞물린다
