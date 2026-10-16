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


def _mixed(x0, y, n, dots=(2, 7)):
    """한글 PDF 한 줄 — 글자 상자 높이가 글자마다 다르다(한글 20·숫자 14·마침표 2, 마침표는 아래에 붙는다)."""
    out = []
    for k in range(n):
        l = x0 + k * 20
        if k in dots:
            out.append(((l, y + 17, l + 2, y + 19), "."))
        elif k % 4 == 1:
            out.append(((l, y + 4, l + 12, y + 18), "3"))
        else:
            out.append(((l, y, l + 18, y + 20), "가"))
    return out


def test_mixed_box_heights_stay_one_line():
    # 마침표들만 따로 한 줄로 묶이면 그 사이가 빈틈으로 보여 건너뛰지 못했다(test2.pdf 1~7쪽 글 블록 79개 중 22개만)
    chars = _line(100, 100, 30) + _line(100, 200, 25) + _line(100, 300, 24) + _mixed(100, 400, 10) + _mixed(100, 445, 10)
    blocks = [_blk((95, 95, 705, 125)), _blk((95, 195, 605, 225)), _blk((95, 295, 585, 325)), _blk((95, 395, 305, 470))]
    assert _plan(blocks, chars) == {3: "text_layer"}


def test_tight_lines_stay_apart():
    # 줄 사이가 좁아도 두 줄은 따로 묶는다 — 둘째 줄 가운데가 비면(레이어에 없는 글자) 건너뛰지 않는다
    second = [c for k, c in enumerate(_mixed(100, 422, 20, dots=(3,))) if not 6 <= k <= 13]
    chars = _line(100, 100, 30) + _line(100, 200, 25) + _line(100, 300, 24) + _mixed(100, 400, 20, dots=(3,)) + second
    blocks = [_blk((95, 95, 705, 125)), _blk((95, 195, 605, 225)), _blk((95, 295, 585, 325)), _blk((95, 395, 505, 447))]
    assert _plan(blocks, chars) == {}


def _samples():
    """검증 표본 세 블록(글자 40·36·34) — 넷째 블록이 덮이면 그것만 건너뛴다."""
    chars = _line(100, 100, 40) + _line(100, 200, 36) + _line(100, 300, 34)
    return chars, [_blk((95, 95, 905, 125)), _blk((95, 195, 825, 225)), _blk((95, 295, 785, 325))]


def test_big_mark_does_not_join_two_lines():
    # 두 줄 높이의 큰 'A'(질의응답 표지)가 줄 높이를 넓히면 둘째 줄까지 한 줄로 묶여 둘째 줄 빈틈(레이어에 없는 글자)을 못 본다
    chars, blocks = _samples()
    big = [((100, 400, 130, 452), "A")]
    second = [c for k, c in enumerate(_mixed(140, 432, 10, dots=())) if not 3 <= k <= 6]
    assert _plan(blocks + [_blk((95, 395, 345, 457))], chars + big + _mixed(140, 400, 10, dots=()) + second) == {}


def test_centered_label_does_not_join_two_lines():
    # 두 줄 가운데 높이에 놓인 1.3배 이름 칸(테두리 없는 표)이 줄 높이를 넓히면 줄 간격 110% 의 두 줄이 한 줄로 묶인다
    chars, blocks = _samples()
    label = [((100 + k * 25, 408, 124 + k * 25, 434), "사") for k in range(4)]
    second = [c for k, c in enumerate(_line(220, 422, 12)) if not 3 <= k <= 7]
    assert _plan(blocks + [_blk((95, 395, 465, 447))], chars + label + _line(220, 400, 12) + second) == {}


def test_filler_blank_is_not_covered():
    # 서식 빈칸의 마침표(레이어에 있다)가 빈 데를 메워도, 그 위 값(레이어에 없다)이 빠졌을 수 있어 건너뛰지 않는다
    chars, blocks = _samples()
    line = _line(100, 400, 14) + [((382 + k * 8, 417, 384 + k * 8, 419), ".") for k in range(12)] + _line(480, 400, 6)
    assert _plan(blocks + [_blk((95, 395, 605, 425))], chars + line) == {}


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
