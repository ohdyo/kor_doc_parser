"""모아찍기 감지·분할·영역 다시 렌더(기획서 5.4, 설계 1장 prep·4장 좌표) — 합성 PDF 로. 두 단 편집 보통 쪽은 나누지 않는다."""
import json

import numpy as np
import pytest
from reportlab.lib.pagesizes import A4, A5, landscape

from kdp.convert import to_page
from kdp.engine.base import PageInput, RawBlock, RawPage
from kdp.engine.paddle_vl import PaddleVLEngine
from kdp.io.ids import doc_dirname
from kdp.io.pdf import open_pdf, read_info, text_chars
from kdp.engine.base import page_stem
from kdp.engine.replay import ReplayEngine
from kdp.io.render import render_page
from kdp.io.write import write_json, write_json_gz
from kdp.pipeline import _pages, parse_pdf, reassemble
from kdp.prep import nup
from kdp.textlayer.extract import display_chars, mark_hidden
from tests.fixtures.stub_engine import StubEngine
from tests.fixtures.synth import (MARK_RGB, body_nup_pdf, caption_nup_pdf, nup_pdf, placed_pdf, report_pdf, scanned,
                                  two_column_pdf, two_tables_pdf)

LAYOUTS = [(2, 2), (2, 1), (1, 2)]


def _layouts(data, scan=False):
    """시트마다 배치 — 앞 시트 배치를 이어 받는다(_pages 와 같게)."""
    doc = open_pdf(data)
    priors, out = {}, []
    for i in range(len(doc)):
        page = doc[i]
        img = render_page(page, 2.0)
        bbox, rot, size = page.get_bbox(), page.get_rotation(), tuple(page.get_size())
        chars = mark_hidden(text_chars(page), img, bbox, rot, 2.0)
        page.close()
        layer = None if scan else [(b, c) for c, b, gen in display_chars(chars, bbox, rot) if not gen and not c.isspace()]
        lay = nup.detect(size, layer, nup.ink_mask(img) if scan else None, priors.get(nup.key(size)))
        if lay:
            priors[nup.key(size)] = lay
        out.append(lay and (lay.cols, lay.rows))
    doc.close()
    return out


def _first_sheet(data):
    doc = open_pdf(data)
    page = doc[0]
    img, size = render_page(page, 2.0), tuple(page.get_size())
    page.close()
    doc.close()
    return img, size


def _color_box(img, rgb):
    """색 표식의 px 상자 — 없으면 None."""
    want = np.array([round(v * 255) for v in rgb[::-1]])                 # BGR
    ys, xs = np.where((np.abs(img.astype(int) - want) < 40).all(axis=2))
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) > 20 else None


def _to_sheet_pt(img, inp, box):
    """엔진이 낸 논리 페이지 px 상자 → 정본 시트 pt(convert.to_page 와 같은 길)."""
    return to_page(RawPage(inp, img.shape[1], img.shape[0], 0.0, [RawBlock(0, "image", box, "")])).blocks[0].bbox


def _sheet_marks(data, scale=2.0):
    """시트 그림에서 찾은 색 표식 자리(시트 pt) — {(시트, 색 순번): 상자}."""
    doc = open_pdf(data)
    out = {}
    for i in range(len(doc)):
        page = doc[i]
        img = render_page(page, scale)
        page.close()
        for j, rgb in enumerate(MARK_RGB):
            box = _color_box(img, rgb)
            if box:
                out[i + 1, j] = tuple(v / scale for v in box)
    doc.close()
    return out


def _run_pages(tmp_path, data, settings):
    pdf = open_pdf(data)
    try:
        info = read_info(pdf, "sha256:ab", "a.pdf")
        return list(_pages(pdf, info, settings, tmp_path))
    finally:
        pdf.close()


@pytest.mark.parametrize("scan", [False, True])
@pytest.mark.parametrize("cols,rows", LAYOUTS)
def test_detects_layout_and_last_sheet(korean_font, cols, rows, scan):
    data = nup_pdf(korean_font, cols, rows, cols * rows + 1)          # 마지막 시트에는 쪽이 하나 — 앞 시트 배치를 이어 받는다
    if scan:
        data = scanned(data)
    assert _layouts(data, scan) == [(cols, rows), (cols, rows)]


@pytest.mark.parametrize("scan", [False, True])
@pytest.mark.parametrize("make", [
    lambda f: two_column_pdf(f),                                      # 세로 두 단 — 칸이 쪽 모양이 아니고 쪽번호가 빈 띠에 걸친다
    lambda f: two_column_pdf(f, wide=True, number="center"),
    lambda f: two_column_pdf(f, wide=True, number="right"),           # 가로 두 단 — 빈 띠는 끝까지 비었지만 쪽번호가 하나
    lambda f: report_pdf(f, 2),
], ids=["two_col", "two_col_wide", "two_col_wide_right", "report"])
def test_ordinary_pages_not_split(korean_font, make, scan):
    data = scanned(make(korean_font)) if scan else make(korean_font)
    assert set(_layouts(data, scan)) == {None}
    if scan:                                                          # 어느 방향으로 돌려 봐도 모아찍기로 보지 않는다
        img, size = _first_sheet(data)
        assert nup.search(size, None, nup.ink_mask(img), {}, range(4)) == {}


def test_prior_layout_needs_numbered_cells(korean_font):
    """앞 시트 배치를 이어 받아도, 내용 있는 칸이 그 자리에 쪽번호가 없으면 나누지 않는다(보통 쪽이 뒤에 붙은 문서)."""
    doc = open_pdf(report_pdf(korean_font, 1))
    page = doc[0]
    img, size = render_page(page, 2.0), tuple(page.get_size())
    layer = [(b, c) for c, b, gen in display_chars(text_chars(page), page.get_bbox(), 0) if not gen and not c.isspace()]
    page.close()
    doc.close()
    prior = nup.Layout(1, 2, ("bottom", 0.5, 0.927))                 # 세로 A4 에 가로 쪽 둘 — 아래 칸 쪽번호 자리와 거의 같다
    assert nup.detect(size, layer, None, prior) is None
    assert nup.detect(size, None, nup.ink_mask(img), prior) is None


@pytest.mark.parametrize("cols,rows", LAYOUTS)
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_scan_search_finds_both_ends(korean_font, cols, rows, k):
    """스캔은 칸 위·아래 표식을 함께 본다 — 바로 선 방향(아래 쪽번호)과 그 180° 짝(위 표식) 둘에서 찾고 옆 방향에서는 못 찾는다.
    그래서 시트 방향은 쪽번호 자리로 정하지 않는다(위 쪽번호 문서를 거꾸로 세우던 결함)."""
    img, size = _first_sheet(scanned(nup_pdf(korean_font, cols, rows, cols * rows), k=k))
    hits = nup.search(size, None, nup.ink_mask(img), {}, range(4))
    up = (4 - k) % 4
    assert {j: (lay.cols, lay.rows, lay.mark[0]) for j, lay in hits.items()} == {up: (cols, rows, "bottom"),
                                                                                  (up + 2) % 4: (cols, rows, "top")}


def test_pages_split_and_rerender(tmp_path, korean_font, settings, monkeypatch):
    monkeypatch.setattr("kdp.pipeline.judge", lambda *a: pytest.fail("born-digital 시트는 방향을 보지 않는다"))
    data = nup_pdf(korean_font, 2, 2, 5)
    out = _run_pages(tmp_path, data, settings)
    inps = [inp for _, inp in out]
    assert [i.page_no for i in inps] == list(range(1, 9))             # 문서 전체 논리 페이지 일련번호(빈 칸도 쪽)
    assert [i.sheet_no for i in inps] == [1] * 4 + [2] * 4
    assert [tuple(i.n_up["cell"]) for i in inps] == [(0, 0), (1, 0), (0, 1), (1, 1)] * 2   # 왼쪽 위부터 행 우선, (열, 행)
    w, h = inps[0].sheet_size
    assert inps[3].n_up["region"] == (round(w / 2, 2), round(h / 2, 2), round(w, 2), round(h, 2))
    assert sorted(p.name for p in (tmp_path / "textlayer").iterdir()) == ["p0001.json.gz", "p0002.json.gz"]   # 시트마다 한 번

    marks = _sheet_marks(data)
    for (img, inp) in out:
        assert inp.scale == pytest.approx(4.0)                        # 원래 쪽(A4) 기준 144dpi — 시트 렌더를 자르면 72dpi
        x0, y0, x1, y1 = inp.n_up["region"]
        assert abs(img.shape[1] - (x1 - x0) * inp.scale) <= 2 and abs(img.shape[0] - (y1 - y0) * inp.scale) <= 2
        j = (inp.page_no - 1) % 4
        box = _color_box(img, MARK_RGB[j])
        if inp.page_no > 5:                                           # 마지막 시트의 빈 칸
            assert box is None and img.min() > 200
            continue
        got = _to_sheet_pt(img, inp, box)                             # 논리 페이지 px → 시트 pt 가 시트 그림의 표식 자리와 맞는다
        assert all(abs(a - b) <= 1.0 for a, b in zip(got, marks[inp.sheet_no, j]))


def test_pages_rotated_scan(tmp_path, korean_font, settings, monkeypatch):
    """옆으로 스캔한 2×2: 방향 판정(여기서는 흉내 — 3)대로 세우고, 세운 방향 순서로 칸을 나눠 좌표를 돌리기 전 시트로 되돌린다."""
    monkeypatch.setattr("kdp.pipeline.judge", lambda *a: (3, (3,)))
    data = scanned(nup_pdf(korean_font, 2, 2, 4), k=1)
    out = _run_pages(tmp_path, data, settings)
    assert [inp.rot90_k for _, inp in out] == [3] * 4
    assert [tuple(inp.n_up["cell"]) for _, inp in out] == [(0, 0), (1, 0), (0, 1), (1, 1)]
    marks = _sheet_marks(data)
    for j, (img, inp) in enumerate(out):
        assert img.shape[0] > img.shape[1]                            # 세운 논리 페이지는 세로
        box = _color_box(img, MARK_RGB[j])                            # 칸 순서 = 논리 쪽 순서
        got = _to_sheet_pt(img, inp, box)
        assert all(abs(a - b) <= 1.0 for a, b in zip(got, marks[1, j]))


def test_nup_disabled_keeps_sheet(tmp_path, korean_font, settings):
    s = settings.model_copy(deep=True)
    s.parse.nup.enabled = False
    out = _run_pages(tmp_path, nup_pdf(korean_font, 2, 2, 4), s)
    assert len(out) == 1 and out[0][1].n_up is None and out[0][1].scale == pytest.approx(2.0)


def test_parse_with_stub_engine(tmp_path, korean_font, settings):
    """엔진 흐름: 논리 페이지마다 raw 파일, 정본 쪽의 n_up·시트 번호, 블록 bbox 는 시트 pt 로 영역 안, 재조립이 같다."""
    pdf = tmp_path / "nup.pdf"
    pdf.write_bytes(nup_pdf(korean_font, 2, 1, 3))
    doc = parse_pdf(pdf, tmp_path / "out", settings, engine=StubEngine())
    d = tmp_path / "out" / doc_dirname(doc.doc_id)
    assert [(p.page_no, p.sheet_no) for p in doc.pages] == [(1, 1), (2, 1), (3, 2), (4, 2)]
    assert [p.born_digital for p in doc.pages] == [True, True, True, False]          # 영역 안 레이어 글자로 판정
    for p in doc.pages:
        x0, y0, x1, y1 = p.n_up.region
        assert p.size == pytest.approx((842, 595.5), abs=1)                          # 쪽 크기는 시트
        assert all(x0 - 1 <= b.bbox[0] and b.bbox[2] <= x1 + 1 and y0 - 1 <= b.bbox[1] and b.bbox[3] <= y1 + 1
                   for b in p.blocks)
    side = json.loads((d / "raw" / "p0004.sidecar.json").read_text(encoding="utf-8"))
    assert side["input"]["sheet_no"] == 2 and side["input"]["n_up"]["cell"] == [1, 0]
    before = (d / "document.json").read_bytes()
    reassemble(d, settings)
    assert (d / "document.json").read_bytes() == before


def _engine_shell(settings):
    eng = object.__new__(PaddleVLEngine)                              # 모델 없이 _plan 만
    s = settings.model_copy(deep=True)
    s.parse.text_layer.mode = "prefer"
    s.parse.paddle_vl.vlm_skip = ["logo", "text_layer"]
    eng._settings = s
    return eng


def test_plan_uses_only_region_chars(tmp_path, settings):
    """훅 C: 모아찍기 논리 페이지에는 그 영역의 레이어 글자만, 논리 페이지 px 로 넘긴다(textlayer 와 같은 기준)."""
    h = 842.0

    def line(x, n, y=100.0):                                          # 사용자 공간 글자 상자, 한 줄
        return [{"c": "가", "box": [x + 10 * k, h - y - 10, x + 10 * k + 10, h - y], "gen": False} for k in range(n)]

    write_json_gz(tmp_path / "textlayer" / "p0001.json.gz", line(40, 22) + line(340, 5))
    eng = _engine_shell(settings)
    base = dict(sheet_no=1, sheet_size=(595.0, h), rotation=0, user_bbox=(0.0, 0.0, 595.0, h))
    left = PageInput(page_no=1, scale=4.0, origin=(0.0, 0.0),
                     n_up={"cols": 2, "rows": 2, "cell": (0, 0), "region": (0.0, 0.0, 297.5, 421.0)}, **base)
    right = PageInput(page_no=2, scale=4.0, origin=(297.5, 0.0),
                      n_up={"cols": 2, "rows": 2, "cell": (1, 0), "region": (297.5, 0.0, 595.0, 421.0)}, **base)
    chars = eng._plan(left, tmp_path).chars
    assert len(chars) == 22 and chars[0][0] == (160.0, 400.0, 200.0, 440.0)       # 표시 (40, 100, 50, 110) pt × 4
    assert eng._plan(right, tmp_path).chars is None                    # 이 영역 글자는 5자 — 레이어로 건너뛰지 않는다
    whole = PageInput(page_no=1, scale=2.0, **base)
    assert len(eng._plan(whole, tmp_path).chars) == 27


def _cell_marks(out):
    """논리 페이지(page_no 순)마다 든 색 표식 순번과, 그 표식이 칸 그림 위쪽 절반에 있는지(바로 섬)."""
    got = []
    for img, _ in out:
        found = [(j, _color_box(img, rgb)) for j, rgb in enumerate(MARK_RGB)]
        got.append([(j, (b[1] + b[3]) / 2 < img.shape[0] / 2) for j, b in found if b])
    return got


@pytest.mark.parametrize("cols,rows", [(2, 2), (2, 1)])
def test_pages_top_numbered_scan(tmp_path, korean_font, settings, monkeypatch, cols, rows):
    """쪽번호가 쪽 위에 있는 문서를 바로 세워 스캔한 모아찍기 — 방향 판정이 0 이면 뒤집지 않고 행 우선으로 나눈다(전에는 쪽번호 자리만 보고
    180° 로 세워 쪽이 모두 거꾸로·역순이 됐다). 방향을 모르면 위 표식으로는 나누지 않는다(볼 방향이 0 이어도)."""
    sheet = landscape(A4) if (cols, rows) == (2, 1) else A4
    data = scanned(placed_pdf(korean_font, [list(range(1, cols * rows + 1))], cols, rows, sheet=sheet, number="top"))
    monkeypatch.setattr("kdp.pipeline.judge", lambda *a: (0, (0,)))
    out = _run_pages(tmp_path, data, settings)
    assert [inp.rot90_k for _, inp in out] == [0] * cols * rows
    assert _cell_marks(out) == [[(j, True)] for j in range(cols * rows)]       # 칸 j 에 쪽 j+1, 바로 선 채
    for tries in ((), (0,)):
        monkeypatch.setattr("kdp.pipeline.judge", lambda *a: (None, tries))
        [(img, inp)] = _run_pages(tmp_path, data, settings)
        assert inp.rot90_k == 0 and inp.n_up is None


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_pages_unsure_scan_follows_lean(tmp_path, korean_font, settings, monkeypatch, k):
    """[회귀 3] 바로·옆·거꾸로 스캔한 아래 쪽번호 2×2 — 방향을 몰라도(표 80% 미만) 표가 기운 방향에서 칸 아래 쪽번호가 맞으면 그리로 세워
    차례대로 나눈다(전에는 누운 채 통째로). 볼 방향이 없거나 옆이거나 위 표식 방향뿐이면 돌리지도 나누지도 않는다."""
    up = (4 - k) % 4
    data = scanned(nup_pdf(korean_font, 2, 2, 4), k=k)
    monkeypatch.setattr("kdp.pipeline.judge", lambda *a: (None, (up,)))
    out = _run_pages(tmp_path, data, settings)
    assert [inp.rot90_k for _, inp in out] == [up] * 4 and _cell_marks(out) == [[(j, True)] for j in range(4)]
    for tries in ((), ((up + 1) % 4,), ((up + 2) % 4,)):
        monkeypatch.setattr("kdp.pipeline.judge", lambda *a: (None, tries))
        [(img, inp)] = _run_pages(tmp_path, data, settings)
        assert inp.rot90_k == 0 and inp.n_up is None


def test_pages_unsure_upright_keeps_bottom_numbers(tmp_path, korean_font, settings, monkeypatch):
    """바로 선 아래 쪽번호 2×2 스캔인데 모델이 모르고 180° 쪽으로 기울면(그림·장 제목 쪽 13:7~15:5) 그 방향은 위 표식이라 넘기고,
    그대로(0)의 칸 아래 쪽번호로 돌리지 않고 나눈다 — 고치기 전처럼."""
    monkeypatch.setattr("kdp.pipeline.judge", lambda *a: (None, (2, 0)))
    out = _run_pages(tmp_path, scanned(nup_pdf(korean_font, 2, 2, 4)), settings)
    assert [inp.rot90_k for _, inp in out] == [0] * 4 and _cell_marks(out) == [[(j, True)] for j in range(4)]


@pytest.mark.xfail(strict=True, reason="[지적 1] 남은 한계 — 위 쪽번호 문서를 거꾸로 스캔한 시트는 판정 입력(모름, 볼 방향 2→0, 0 에 아래 표식)이 "
                                       "바로 선 그림·장 제목 쪽 모아찍기(모델이 180° 로 기움)와 같아 가를 수 없다. 흔한 쪽(바로 선 아래 쪽번호)을 지킨다")
def test_pages_unsure_upside_down_top_numbers(tmp_path, korean_font, settings, monkeypatch):
    """기대: 위 쪽번호 2×2 를 거꾸로(k=2) 스캔했는데 방향을 모르면 돌리지도 나누지도 않는다(지금은 거꾸로·역순으로 나뉜다)."""
    monkeypatch.setattr("kdp.pipeline.judge", lambda *a: (None, (2, 0)))
    out = _run_pages(tmp_path, scanned(placed_pdf(korean_font, [[1, 2, 3, 4]], 2, 2, number="top"), k=2), settings)
    assert len(out) == 1 and out[0][1].rot90_k == 0 and out[0][1].n_up is None


@pytest.mark.xfail(strict=True, reason="[지적 4] 남은 한계 — 스캔은 쪽번호 값을 읽지 못해 세로 우선 배치를 행 우선으로 읽는다(QC 표시는 requests)")
def test_scan_column_major_order(tmp_path, korean_font, settings, monkeypatch):
    """기대: 세로 우선 2×2 스캔도 page_no 순서가 인쇄 쪽번호 순서다."""
    monkeypatch.setattr("kdp.pipeline.judge", lambda *a: (0, (0,)))
    out = _run_pages(tmp_path, scanned(placed_pdf(korean_font, [[1, 3, 2, 4]], 2, 2)), settings)
    assert [[j for j, _ in ms] for ms in _cell_marks(out)] == [[0], [1], [2], [3]]


def test_pages_unsure_sideways_marks_not_used(tmp_path, korean_font, settings, monkeypatch):
    """[회귀 3] 오른쪽 아래 쪽번호·머리말 없는 25행 2×2 를 옆(k=3)으로 스캔 — 누운 시트에서도 칸 아래 표식이 맞아 보이지만(검색 0 방향)
    방향을 모르고 볼 방향이 없으면(누운 시트는 글줄이 세로라 0 을 보지 않는다) 나누지 않는다(전에는 누운 칸으로 [3,1,4,2] 나눔).
    표가 바른 쪽(1)으로 기울면 그리로 세워 나눈다."""
    data = scanned(body_nup_pdf(korean_font, 25, "right"), k=3)
    img, size = _first_sheet(data)
    assert 0 in nup.search(size, None, nup.ink_mask(img), {}, (0,))                  # 누운 시트의 헛 표식
    monkeypatch.setattr("kdp.pipeline.judge", lambda *a: (None, ()))
    [(img, inp)] = _run_pages(tmp_path, data, settings)
    assert inp.rot90_k == 0 and inp.n_up is None
    monkeypatch.setattr("kdp.pipeline.judge", lambda *a: (None, (1,)))
    out = _run_pages(tmp_path, data, settings)
    assert [inp.rot90_k for _, inp in out] == [1] * 4 and _cell_marks(out) == [[(j, True)] for j in range(4)]


@pytest.mark.parametrize("scan", [False, True])
def test_outer_numbers_not_merged(korean_font, scan):
    """바깥쪽 쪽번호(홀수 오른쪽·짝수 왼쪽) 2×2 — 가운데에서 마주 본 두 쪽번호를 한 줄로 보고 위·아래 절반(1×2)으로 나누지 않는다."""
    data = placed_pdf(korean_font, [[1, 2, 3, 4]], 2, 2, number="outer")
    assert _layouts(scanned(data) if scan else data, scan) == [(2, 2)]


@pytest.mark.parametrize("scan", [False, True])
@pytest.mark.parametrize("cols,rows", [(2, 3), (2, 4)])
def test_stacked_pages_not_split(korean_font, cols, rows, scan):
    """세로 A4 에 6쪽(2열×3행)·8쪽(2열×4행) — 2×2 로 나누면 칸에 쪽이 위아래로 쌓인다(칸 가운데에 같은 자리 쪽번호). 나누지 않는다."""
    data = placed_pdf(korean_font, [list(range(1, cols * rows + 1))], cols, rows)
    assert _layouts(scanned(data) if scan else data, scan) == [None]


def test_column_major_order(tmp_path, korean_font, settings):
    """세로 우선 2×2(왼쪽 열 1·2, 오른쪽 열 3·4) born-digital — 칸 쪽번호 값으로 열 우선으로 읽는다. 다음 시트도 같다."""
    out = _run_pages(tmp_path, placed_pdf(korean_font, [[1, 3, 2, 4], [5, 7, 6, 8]], 2, 2), settings)
    assert [tuple(inp.n_up["cell"]) for _, inp in out] == [(0, 0), (0, 1), (1, 0), (1, 1)] * 2
    assert [m for m, in [[j for j, _ in ms] for ms in _cell_marks(out)]] == [0, 1, 2, 3] * 2   # page_no 순 = 인쇄 쪽번호 순


def test_number_values_must_fit_sheet(korean_font):
    """born-digital 칸 쪽번호 값이 이웃 칸끼리 하나도 n, n+1 로 잇지 않으면 나누지 않는다 — 나란히 놓인 표의 합계(183·195),
    소책자 배열(8·1 — 순서를 바로잡을 수 없다). 붙은 값(4·5)은 나눈다."""
    assert _layouts(two_tables_pdf(korean_font)) == [None]
    booklet = placed_pdf(korean_font, [[8, 1], [4, 5]], 2, 1, sheet=landscape(A4), page=A5)
    assert _layouts(booklet) == [None, (2, 1)]


def test_number_values_restart(korean_font):
    """[회귀 4] 값은 순서를 정하는 데만 쓴다 — 쪽번호가 새로 시작하는 붙임(23·24 다음 1·2), 앞 시트 끝에 이어지는 시트(12 → [13, 1]),
    한 쪽짜리 문서 둘('- 1 -' 둘)을 모아찍기한 시트도 나눈다(전에는 시트 통째)."""
    restart = placed_pdf(korean_font, [[19, 20, 21, 22], [23, 24, 1, 2], [3, 4, 5, 6]], 2, 2)
    assert _layouts(restart) == [(2, 2)] * 3
    assert _layouts(placed_pdf(korean_font, [[1, 1]], 2, 1, sheet=landscape(A4))) == [(2, 1)]
    assert _layouts(placed_pdf(korean_font, [[10, 11], [12, 1], [2, 3]], 2, 1, sheet=landscape(A4))) == [(2, 1)] * 3


@pytest.mark.parametrize("scan", [False, True])
@pytest.mark.parametrize("cols,rows", [(2, 2), (2, 1)])
def test_caption_pages_split(korean_font, cols, rows, scan):
    """[회귀 1] 가운데 정렬 짧은 표 제목('<표 n>' 10pt — 쪽번호 줄과 자리·높이가 같다)이 있는 쪽의 2×2·2×1 모아찍기는 쌓인 쪽이 아니다 —
    born-digital 은 그 줄 글이 쪽번호 꼴이 아니고, 스캔은 그 아래 빈 띠가 쪽 경계만큼 넓지 않다. 나눈다(전에는 시트 통째)."""
    data = caption_nup_pdf(korean_font, cols, rows)
    assert _layouts(scanned(data) if scan else data, scan) == [(cols, rows)]


def test_replay_reads_page_numbers_in_order(tmp_path):
    """논리 페이지가 1만 쪽을 넘어도(2×2 모아찍기 2,500 시트) 재조립은 page_no 숫자 순 — 이름 글자 순이면 p10000 이 p1000 뒤에 온다."""
    nos = [2, 999, 1000, 1001, 9999, 10000, 10001]
    for no in nos:
        inp = {"page_no": no, "sheet_no": no, "scale": 2.0, "sheet_size": [595.0, 842.0], "origin": [0.0, 0.0],
               "n_up": None, "rotation": 0, "rot90_k": 0, "user_bbox": [0, 0, 595, 842]}
        write_json(tmp_path / f"{page_stem(no)}.json", {"parsing_res_list": [], "layout_det_res": {"boxes": []}})
        write_json(tmp_path / f"{page_stem(no)}.sidecar.json", {"input": inp, "width": 10, "height": 10, "ink_ratio": 0.0,
                                                               "blocks": [], "table_imgs": {}})
    assert [r.input.page_no for r in ReplayEngine(tmp_path).run()] == nos
