"""스캔 방향(기획서 4.3 규칙 12, 설계 3.7) — 레이어 글자가 없는 시트만, 여백 틀 20표가 한 각도로 모이고 잉크 투영의 글줄 방향·글 양이
맞을 때만 세우고 좌표는 돌리기 전 시트로 되돌린다. 모델 시험은 가중치(PP-LCNet_x1_0_doc_ori)가 없으면 건너뛴다."""
import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest

from kdp.config import Env, engine_hash
from kdp.convert import to_page
from kdp.engine.base import PageInput, RawBlock, RawPage
from kdp.engine.paddle_vl import PaddleVLEngine, _weights_fingerprint
from kdp.io.ids import doc_dirname
from kdp.io.pdf import open_pdf, read_info
from kdp.io.render import render_page
from kdp.pipeline import _pages, parse_pdf, reassemble
from kdp.prep import orient
from tests.fixtures.stub_engine import StubEngine
from tests.fixtures.synth import (MARK_RGB, _letter_page, _one, _scan_source_page, body_nup_pdf, body_pdf, cover_pdf,
                                  diagram_slide_pdf, form_pdf, handout_pdf, m3_sample_pdf, marker_pdf, nup_pdf, placed_pdf,
                                  report_pdf, scanned, slide_pdf, table_page_pdf, title_page_pdf, toc_pdf, two_column_pdf)


@pytest.fixture()
def ori_env(settings):
    if not orient.model_dir(settings.env).is_dir():
        pytest.skip("방향 분류 가중치가 없다(configs/env.yaml 의 models.doc_ori_dir 또는 PaddleX 캐시)")
    return settings.env


def _sheet(data):
    doc = open_pdf(data)
    page = doc[0]
    img, size = render_page(page, 2.0), tuple(page.get_size())
    page.close()
    doc.close()
    return img, size


def _red_box(img):
    ys, xs = np.where((img[:, :, 2] > 200) & (img[:, :, 1] < 60) & (img[:, :, 0] < 60))
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def _run_pages(tmp_path, data, settings):
    pdf = open_pdf(data)
    try:
        return list(_pages(pdf, read_info(pdf, "sha256:ab", "a.pdf"), settings, tmp_path))
    finally:
        pdf.close()


def _ch(**flags):
    return {"c": "가", "box": [0, 0, 10, 10], "gen": False, **flags}


def test_is_scan_and_layerless():
    assert not orient.is_scan([_ch()] * 30) and not orient.layerless([_ch()] * 30)
    assert orient.is_scan([]) and orient.layerless([])                # 텍스트 레이어 없음
    assert orient.is_scan([_ch()] * 5) and not orient.layerless([_ch()] * 5)       # 글자 몇 개(표지·간지) — 방향은 보지 않는다
    assert orient.is_scan([_ch(inv=True)] * 30) and orient.layerless([_ch(inv=True)] * 30)   # 투명 OCR 레이어만
    assert orient.is_scan([_ch()] * 25 + [_ch(hid=True)] * 30)        # 보이는 글자가 절반이 안 된다


def _fake_votes(monkeypatch, votes):
    """classify 흉내 — i 번째 부름(여백 i//4, np.rot90(틀, j=i%4))의 답이 원래 그림을 votes[i] 만큼 돌리면 서게. 판정을 여러 번
    부르면(파이프라인) 같은 표를 되풀이한다."""
    calls = itertools.count()

    def classify(img, env):
        i = next(calls) % len(votes)
        return (votes[i] - i % 4) % 4 * 90, 0.9
    monkeypatch.setattr(orient, "classify", classify)


N_VOTES = len(orient.MARGINS) * 4


@pytest.mark.parametrize("turn,votes,k", [
    (0, [0] * N_VOTES, 0),
    (0, [0] * 16 + [2] * 4, 0),                                       # 표의 80%
    (0, [0] * 15 + [2] * 5, None),                                    # 75% — 모른다
    (1, [3] * N_VOTES, 3),                                            # 옆으로 누운 시트 — np.rot90(…, 3) 으로 선다
    (1, [0] * N_VOTES, None),                                         # 모델은 0 이라는데 글줄이 세로다(잉크 투영과 어긋남)
    (2, [2] * 17 + [1, 3, 0], 2),
])
def test_upright_k_votes_and_line_direction(monkeypatch, korean_font, turn, votes, k):
    up, _ = _sheet(report_pdf(korean_font, 1))
    _fake_votes(monkeypatch, votes)
    assert orient.upright_k(np.ascontiguousarray(np.rot90(up, turn)), Env()) == k


@pytest.mark.parametrize("make", [cover_pdf, title_page_pdf, lambda f: body_pdf(f, 0), diagram_slide_pdf],
                         ids=["cover", "title", "heading_and_number", "diagram_slide"])
def test_upright_k_keeps_sparse_pages(monkeypatch, korean_font, make):
    """표가 모두 같아도 글이 적은 쪽(표지·제목 쪽·쪽번호만 있는 쪽·도식 슬라이드)은 돌리지 않는다 — 모델이 이런 쪽을 거꾸로 확신한다."""
    img, _ = _sheet(scanned(make(korean_font), k=2))
    _fake_votes(monkeypatch, [2] * N_VOTES)
    assert orient.upright_k(img, Env()) is None


@pytest.mark.parametrize("make", [slide_pdf, toc_pdf, form_pdf], ids=["slide", "toc", "form"])
def test_upright_k_turns_short_text_pages(monkeypatch, korean_font, make):
    """[회귀 2] 글이 100 자에 못 미쳐도 슬라이드(55)·목차(81)·서식(82)은 글 양 하한(45)을 넘어 표대로 세운다 — 전에는 하한 100 에 막혀
    누운 채 엔진에 갔다."""
    img, _ = _sheet(scanned(make(korean_font), k=2))
    _fake_votes(monkeypatch, [2] * N_VOTES)
    assert orient.upright_k(img, Env()) == 2


@pytest.mark.parametrize("turn,votes,want", [
    (2, [2] * 15 + [0] * 5, (None, (2, 0))),                          # 75% — 모르지만 2 로 기울었다(짝 0 의 3배), 그다음 그대로
    (2, [2] * 10 + [0] * 7 + [1] * 3, (None, (0,))),                  # 10:7 — 짝의 1.5배에 못 미쳐 기울지 않았다
    (2, [2] * 8 + [3] * 8 + [0] * 4, (None, (0,))),                   # 1위가 둘
    (2, [1] * 15 + [3] * 5, (None, (0,))),                            # 옆으로 기울었지만 그 방향 글줄이 세로
    (1, [3] * 15 + [1] * 5, (None, (3,))),                            # 누운 시트 — 그대로(0)는 글줄이 세로라 보지 않는다
    (1, [3] * 10 + [1] * 10, (None, ())),
    (2, [2] * N_VOTES, (2, (2, 0))),
])
def test_judge_tries(monkeypatch, korean_font, turn, votes, want):
    """확신이 없을 때 볼 방향 — 표가 기운 방향(혼자 1위, 180° 짝 표의 1.5배 이상, 그 방향 글줄 가로) 다음 그대로(글줄 가로일 때만)."""
    up, _ = _sheet(report_pdf(korean_font, 1))
    _fake_votes(monkeypatch, votes)
    assert orient.judge(np.ascontiguousarray(np.rot90(up, turn)), Env()) == want


def test_model_dir_and_missing_model(tmp_path):
    assert orient.model_dir(Env(paddlex={"cache_home": str(tmp_path)})) == tmp_path / "official_models" / orient.MODEL
    env = Env(models={"doc_ori_dir": str(tmp_path / "none")})
    assert orient.model_dir(env) == tmp_path / "none"
    with pytest.raises(FileNotFoundError, match="doc_ori_dir"):       # 실행 중에 내려받지 않고 바로 멈춘다
        orient.classify(np.zeros((8, 8, 3), np.uint8), env)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_unrotate_matches_rot90(korean_font, k):
    """np.rot90(그림, k) 로 세운 그림의 상자를 convert.to_page 가 돌리기 전 시트 pt 로 되돌린다 — 실제 렌더 표식으로."""
    sheet, size = _sheet(scanned(marker_pdf(korean_font), k=(4 - k) % 4))    # 이 시트는 np.rot90(…, k) 로 선다
    up = np.ascontiguousarray(np.rot90(sheet, k))
    upright, _ = _sheet(marker_pdf(korean_font))
    assert all(abs(a - b) <= 2 for a, b in zip(_red_box(up), _red_box(upright)))   # 세운 그림이 원래 쪽과 같다
    inp = PageInput(page_no=1, sheet_no=1, scale=2.0, sheet_size=size, rot90_k=k)
    got = to_page(RawPage(inp, up.shape[1], up.shape[0], 0.0, [RawBlock(0, "image", _red_box(up), "")])).blocks[0].bbox
    assert all(abs(a - b / 2) <= 0.5 for a, b in zip(got, _red_box(sheet)))


def test_born_digital_and_off_skip_model(tmp_path, korean_font, settings, monkeypatch):
    monkeypatch.setattr("kdp.pipeline.judge", lambda *a: pytest.fail("방향 모델을 부르지 않는다"))
    for data in (report_pdf(korean_font, 1), cover_pdf(korean_font)):                 # born-digital — 글자 몇 개짜리 표지도
        [(_, inp)] = _run_pages(tmp_path, data, settings)
        assert inp.rot90_k == 0
    off = settings.model_copy(deep=True)
    off.parse.orientation = "off"
    [(img, inp)] = _run_pages(tmp_path, scanned(report_pdf(korean_font, 1), k=1), off)
    assert inp.rot90_k == 0 and img.shape[1] > img.shape[0]


UPRIGHTS = {                                                          # 이 꼴은 네 방향 모두 바르게 세운다
    "report": lambda f: report_pdf(f, 1),
    "scan_source": lambda f: _one(_scan_source_page, f),              # 글이 쪽 위쪽에 몰린 쪽(M3 표본 옆 스캔) — 가운데 자르기로는 못 세웠다
    "letter": lambda f: _one(_letter_page, f),
    "table": table_page_pdf,
    "nup_2x2": lambda f: nup_pdf(f, 2, 2, 4),
    "nup_top_numbers": lambda f: placed_pdf(f, [[1, 2, 3, 4]], 2, 2, number="top"),
    "slide": slide_pdf,                                               # 글 양 55~82 — 전에는 하한 100 에 막혔다(회귀 2)
    "toc": toc_pdf,
    "form": form_pdf,
}
KEEPS = {                                                             # 이 꼴은 틀리게 돌리지 않는다(세우거나 그대로)
    "two_column": two_column_pdf,                                     # 같은 문장이 반복되는 합성 글 — 표가 갈린다
    "full_body": lambda f: body_pdf(f, 33),
    "cover": cover_pdf,
    "title": title_page_pdf,                                          # 모델이 거꾸로 확신하는 제목 쪽 — 글 양 하한으로 막는다
    "diagram_slide": diagram_slide_pdf,                               # 상자 이름표뿐(글 양 26)
}


@pytest.mark.parametrize("k", [0, 1, 2, 3])
@pytest.mark.parametrize("name", list(UPRIGHTS))
def test_model_uprights_scan(korean_font, ori_env, name, k):
    """합성 스캔(글·표·공문·모아찍기) × 네 방향 — 바르게 세운다(바로 선 쪽은 0)."""
    img, _ = _sheet(scanned(UPRIGHTS[name](korean_font), k=k))
    assert orient.upright_k(img, ori_env) == (4 - k) % 4


@pytest.mark.parametrize("k", [0, 1, 2, 3])
@pytest.mark.parametrize("name", list(KEEPS))
def test_model_never_turns_wrong(korean_font, ori_env, name, k):
    """세우기 어려운 쪽(같은 문장 반복·표지·제목 쪽) × 네 방향 — 틀린 각도로는 돌리지 않는다(표지·제목 쪽은 아예 돌리지 않는다)."""
    img, _ = _sheet(scanned(KEEPS[name](korean_font), k=k))
    got = orient.upright_k(img, ori_env)
    assert got in (None, (4 - k) % 4) and (got is None or name not in ("cover", "title", "diagram_slide"))


def test_m3_sample_sideways_sheet(tmp_path, korean_font, settings, ori_env):
    """M3 표본 — 옆으로 스캔한 시트 4(논리 페이지 7)만 np.rot90(…, 3) 으로 세워 세로 그림을 엔진에 넣는다(M3 완료 시험의 prep 몫)."""
    out = _run_pages(tmp_path, m3_sample_pdf(korean_font), settings)
    assert [inp.rot90_k for _, inp in out] == [0] * 6 + [3, 0]
    img, inp = out[6]
    assert inp.sheet_no == 4 and img.shape[0] > img.shape[1]


@pytest.mark.parametrize("k", [0, 1, 2])
def test_model_nup_scan(tmp_path, korean_font, settings, ori_env, k):
    """모아찍기 스캔도 방향 판정으로 세운 뒤 나눈다 — 위 쪽번호 문서를 바로 세워 스캔한 시트를 뒤집지 않는다(전에는 180°)."""
    data = scanned(placed_pdf(korean_font, [[1, 2, 3, 4]], 2, 2, number="top"), k=k)
    out = _run_pages(tmp_path, data, settings)
    assert [inp.rot90_k for _, inp in out] == [(4 - k) % 4] * 4 and [tuple(inp.n_up["cell"]) for _, inp in out] == [
        (0, 0), (1, 0), (0, 1), (1, 1)]


def _marks_in_order(out):
    """논리 페이지 j 에 쪽 j+1 의 색 표식이 바로 선 채(그림 위쪽 절반) 하나씩."""
    for j, (img, _) in enumerate(out):
        ys, xs = np.where((np.abs(img.astype(int) - [round(v * 255) for v in MARK_RGB[j][::-1]]) < 40).all(axis=2))
        if len(xs) <= 20 or ys.mean() > img.shape[0] / 2:
            return False
    return True


@pytest.mark.parametrize("make,k", [(lambda f: body_nup_pdf(f, 25), 1), (lambda f: body_nup_pdf(f, 25), 2),
                                    (lambda f: body_nup_pdf(f, 25), 3), (handout_pdf, 3)],
                         ids=["body25-k1", "body25-k2", "body25-k3", "handout-k3"])
def test_model_unsure_nup_scan_follows_lean(tmp_path, korean_font, settings, monkeypatch, make, k):
    """[회귀 3] 옆·거꾸로 스캔한 아래 쪽번호 모아찍기에서 방향 모델 표가 80% 에 못 미쳐도(표 14~15 : 5~6) 기운 방향의 칸 아래 쪽번호가
    맞으면 그리로 세워 차례대로 나눈다 — 전에는 시트가 누운 채 통째로 엔진에 갔다. 모델 표는 15 : 5 로 고정한다 — 실제 모델 표는
    장치마다 달라 GPU(A40)에서는 같은 시트를 확신했다(2026-10-16)."""
    data = make(korean_font)
    up = (4 - k) % 4
    _fake_votes(monkeypatch, [up] * 15 + [(up + 2) % 4] * 5)
    img, _ = _sheet(scanned(data, k=k))
    assert orient.upright_k(img, Env()) is None                        # 확신은 없다
    out = _run_pages(tmp_path, scanned(data, k=k), settings)
    n = 2 if make is handout_pdf else 4
    assert [inp.rot90_k for _, inp in out] == [(4 - k) % 4] * n and _marks_in_order(out)


def test_weights_fingerprint_expands_home(tmp_path, monkeypatch):
    """models.doc_ori_dir 를 '~/…' 로 적어도 불러 쓰는 그 폴더의 판이 잡히고, 가중치가 바뀌면 판이 바뀐다."""
    monkeypatch.setenv("HOME", str(tmp_path))
    d = tmp_path / "ori"
    d.mkdir()
    (d / "inference.pdiparams").write_bytes(b"1")
    env = Env(models={"doc_ori_dir": "~/ori"})
    assert orient.model_dir(env) == d
    before = _weights_fingerprint("~/ori", orient.MODEL)
    (d / "inference.pdiparams").write_bytes(b"22")
    after = _weights_fingerprint("~/ori", orient.MODEL)
    assert "files" in before and before != after and before["path"] == str(d)


def test_describe_has_orientation_method(settings, monkeypatch):
    """엔진 설명(엔진 해시)에 방향 모델 판과 방향·모아찍기 판정 기준값이 들어간다 — 기준을 바꾸면 해시가 바뀐다."""
    eng = object.__new__(PaddleVLEngine)                              # 모델 없이 describe 만
    eng._settings, eng.hooked, eng._served, eng._snapshot = settings, False, None, "snap"
    eng._ocr = SimpleNamespace(_merged_paddlex_config={
        "pipeline_name": "PaddleOCR-VL-1.6",
        "SubModules": {"LayoutDetection": {"model_name": "PP-DocLayoutV3"}, "VLRecognition": {"model_name": "PaddleOCR-VL-1.6"}}})
    desc = eng.describe()
    assert desc["weights"]["doc_ori"]["path"] == str(orient.model_dir(settings.env))
    assert desc["prep"]["orient"] and desc["prep"]["nup"]
    h = engine_hash(settings, desc)
    monkeypatch.setattr(orient, "AGREE", 0.9)
    assert engine_hash(settings, eng.describe()) != h


def test_parse_rotated_scan(tmp_path, korean_font, settings, ori_env):
    """옆으로 스캔한 쪽: 엔진에는 세운 그림, 정본 쪽 크기·bbox 는 돌리기 전 시트 pt, 재조립이 같다."""
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(scanned(report_pdf(korean_font, 2), k=1))
    doc = parse_pdf(pdf, tmp_path / "out", settings, engine=StubEngine())
    d = tmp_path / "out" / doc_dirname(doc.doc_id)
    side = json.loads((d / "raw" / "p0001.sidecar.json").read_text(encoding="utf-8"))
    assert side["input"]["rot90_k"] == 3 and side["width"] < side["height"]
    w, h = doc.pages[0].size
    assert w > h and not doc.pages[0].born_digital
    num = next(b for b in doc.pages[0].blocks if b.type == "number")  # 세운 쪽 아래 가운데 = 시트 오른쪽 가운데
    assert num.bbox[0] > 0.9 * w and abs((num.bbox[1] + num.bbox[3]) / 2 - h / 2) < 0.05 * h
    assert all(0 <= b.bbox[0] < b.bbox[2] <= w + 1 and 0 <= b.bbox[1] < b.bbox[3] <= h + 1
               for p in doc.pages for b in p.blocks)
    before = (d / "document.json").read_bytes()
    reassemble(d, settings)
    assert (d / "document.json").read_bytes() == before
