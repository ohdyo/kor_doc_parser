"""좌표 변환(설계 4장) — 실제 렌더 픽셀과 1~2px 안으로 맞아야 한다."""
import numpy as np
import pytest

from kdp.io.pdf import open_pdf, text_chars
from kdp.io.render import px_box_to_pt, render_page, render_scale, unrotate_box, user_box_to_display
from tests.fixtures.synth import RED, marker_pdf, with_page_box

SCALE = 2.0


def _pixel_box(mask):
    ys, xs = np.where(mask)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


@pytest.mark.parametrize("crop", [None, (50, 50, 545, 792)])
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_user_space_to_render(korean_font, rotation, crop):
    doc = open_pdf(with_page_box(marker_pdf(korean_font), rotation, crop))
    page = doc[0]
    img = render_page(page, SCALE)
    bbox, rot = page.get_bbox(), page.get_rotation()
    assert rot == rotation

    red = (img[:, :, 2] > 200) & (img[:, :, 1] < 60) & (img[:, :, 0] < 60)
    want = [v * SCALE for v in user_box_to_display(RED, bbox, rot)]
    assert all(abs(a - b) <= 2 for a, b in zip(_pixel_box(red), want))

    # 그린 글자가 텍스트 레이어에 그대로 있고, 글자 상자도 같은 변환으로 글자 위에 떨어진다
    chars = text_chars(page)
    assert "".join(c["c"] for c in chars if not c["gen"]).replace(" ", "") == "가나다ABC2026"
    for ch in chars:
        if ch["c"].strip():
            x1, y1, x2, y2 = (int(v * SCALE) for v in user_box_to_display(ch["box"], bbox, rot))
            assert (img[y1:y2, x1:x2].min(axis=2) < 128).mean() > 0.05, ch["c"]
    page.close()
    doc.close()


def test_render_scale_cap():
    assert render_scale((595, 842), 144, 10**9) == 2.0
    s = render_scale((595, 842), 144, 100_000)
    assert 595 * 842 * s * s <= 100_000 * 1.0001


def test_px_to_pt_with_origin():
    assert px_box_to_pt((20, 40, 220, 440), 2.0, (100, 50)) == (110, 70, 210, 270)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_unrotate_box(k):
    img = np.zeros((300, 200), np.uint8)
    img[30:60, 120:180] = 1                     # 원래 상자 (x1,y1,x2,y2) = (120,30,180,60)
    rot = np.rot90(img, k)
    ys, xs = np.where(rot)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    assert unrotate_box(box, k, (rot.shape[1], rot.shape[0])) == (120, 30, 180, 60)
