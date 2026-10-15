"""훅 C 블록 고르기 — 레이어가 블록을 빈 데 없이 덮을 때만, 묶음 대표·깨진 글자는 건너뛰지 않는다."""
import numpy as np

from kdp.engine.skip import SkipPlan


def _line(x0, y, n, h=20.0, w=18.0, gap=2.0, ch="가"):
    return [((x0 + k * (w + gap), y, x0 + k * (w + gap) + w, y + h), ch) for k in range(n)]


def _blk(box, label="text", **extra):
    return {"label": label, "box": box, "img": object(), **extra}


def _plan(blocks, chars):
    return SkipPlan(False, chars).choose(blocks)


def test_skips_covered_blocks_but_keeps_three_samples():
    chars = (_line(100, 100, 30) + _line(100, 200, 25) + _line(100, 300, 20) + _line(100, 400, 10, h=40, w=36)
             + _line(100, 500, 15))
    blocks = [_blk((95, 95, 705, 125)), _blk((95, 195, 605, 225)), _blk((95, 295, 505, 325)),
              _blk((95, 395, 495, 445), "paragraph_title"), _blk((95, 495, 405, 525))]
    skip = _plan(blocks, chars)
    assert skip == {2: "text_layer", 4: "text_layer"}               # 표본: 글자 최다(0)·글자 크기가 다른 제목(3)·그다음(1)


def test_not_skipped_when_layer_leaves_gaps_or_is_broken():
    gappy = _line(100, 100, 20) + _line(100, 200, 20)                # 가운데 줄(채운 입력란 등)이 레이어에 없다
    split = _line(100, 300, 5) + _line(400, 300, 5)                  # 한 줄 가운데가 비었다
    broken = _line(100, 400, 20, ch="ö")                             # 깨진 글자 대응표
    rare = _line(100, 700, 6, ch="똠")                               # 짧아도 드문 한글뿐이면 깨진 것으로 본다
    body = _line(100, 500, 30) + _line(100, 600, 30)
    blocks = [_blk((95, 95, 505, 225)), _blk((95, 295, 505, 325)), _blk((95, 395, 505, 425)),
              _blk((95, 495, 705, 525)), _blk((95, 595, 705, 625)), _blk((95, 495, 705, 625), merge_aligns=[1]),
              _blk((95, 695, 225, 725))]
    skip = _plan(blocks, gappy + split + broken + body + rare)
    assert not skip                                                  # 남은 후보는 표본 둘(3·4)뿐, 묶음 대표(5)도 건너뛰지 않는다


def test_logos_and_no_sample():
    blocks = [_blk((0, 0, 50, 50), "footer_image"), _blk((95, 95, 705, 125), "header")]
    assert SkipPlan(True, _line(100, 100, 30)).choose(blocks) == {0: "logo"}   # 본문 표본이 없으면 글은 건너뛰지 않는다


def test_ink_without_layer_chars_is_not_skipped():
    full = _line(100, 300, 20)
    chars = (_line(100, 100, 30) + _line(100, 200, 25) + [c for k, c in enumerate(full) if k not in (9, 10)]
             + _line(100, 400, 20))
    blocks = [_blk((95, 95, 705, 125)), _blk((95, 195, 605, 225)), _blk((95, 295, 505, 325)), _blk((95, 395, 505, 425))]
    img = np.full((600, 800, 3), 255, np.uint8)
    for (x0, y0, x1, y1), _ in chars + full:
        img[int(y0) + 5:int(y1) - 5, int(x0) + 4:int(x1) - 4] = 0  # 글자 잉크
    plan = SkipPlan(False, chars)
    assert plan.choose(blocks) == {2: "text_layer"}
    assert plan.choose(blocks, img) == {}                            # 레이어에 없는 두 글자(채운 입력란 값)의 잉크가 있다
