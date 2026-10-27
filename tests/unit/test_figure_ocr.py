"""그림 다시 읽기 — 고를 그림·영역·열쇠, 'Spotting:' 응답 풀기, 기록 합치기(같은 줄·제목 줄·잘린 다시 읽기·기록 없음)."""
from kdp.assemble.figures import attach_figure_ocr, ocr_key, ocr_region, ocr_targets, spotting_lines
from kdp.config import FigureOcrCfg
from kdp.schema import Block, Document, Page, ParserInfo, Source

ON = FigureOcrCfg(enabled=True, model="m")


def _doc(*blocks):
    page = Page(page_no=1, sheet_no=1, size=(595, 842), blocks=list(blocks))
    return Document(doc_id="sha256:ab", source=Source(paths=["a.pdf"], pages=1),
                    parser=ParserInfo(kdp_version="0", engine={}, engine_hash="e", assemble_hash="a", run_at="t"),
                    pages=[page])


def _fig(n, typ="chart", text=None, bbox=(70, 100, 520, 400), **kw):
    return Block(id=f"p1.b{n:02d}", type=typ, bbox=bbox, figure_text=text, img_path=f"imgs/p1.b{n:02d}.png", **kw)


def _cache(doc, text, finish="stop"):
    page, b = doc.pages[0], doc.pages[0].blocks[0]
    return {ocr_key(ON, b, ocr_region(page, b, ON.pad_pt)): {"text": text, "finish_reason": finish}}


def test_targets_region_and_key():
    doc = _doc(_fig(1), _fig(2, "image", bbox=(10, 10, 30, 30)), _fig(3, "image", bbox=(0, 700, 595, 842)),
               Block(id="p1.b04", type="text", bbox=(70, 410, 520, 420), text="본문"))
    assert [b.id for _, b in ocr_targets(doc, ON)] == ["p1.b01", "p1.b03"]          # 아이콘(넓이 1% 밑)·글 블록은 빼고
    assert [b.id for _, b in ocr_targets(doc, ON.model_copy(update={"types": ["chart"]}))] == ["p1.b01"]   # 기본 설정 — 그래프만
    page, b = doc.pages[0], doc.pages[0].blocks[0]
    assert ocr_region(page, b, 6) == (64, 94, 526, 406)
    assert ocr_region(page, page.blocks[2], 6) == (0, 694, 595, 842)                 # 시트 안으로 자른다
    key = ocr_key(ON, b, ocr_region(page, b, 6))
    assert key == ocr_key(ON, b.model_copy(update={"figure_text": "다른 글"}), ocr_region(page, b, 6))   # 그림 글은 열쇠에 없다
    assert key != ocr_key(ON.model_copy(update={"prompt": "Spotting:"}), b, ocr_region(page, b, 6))
    assert key != ocr_key(ON, b, ocr_region(page, b, 8))


def test_spotting_lines():
    loc = "".join(f"<|LOC_{i}|>" for i in range(8))
    raw = f"<|TEXT_START|>768(11)<|TEXT_END|><|LOC_BEGIN|>{loc}<|LOC_END|><|TEXT_START|>강도5<|TEXT_END|><|LOC_BEGIN|>{loc}<|LOC_END|>"
    assert spotting_lines(raw) == ["768(11)", "강도5"]
    assert spotting_lines(f"35.4{loc}열경련{loc}") == ["35.4", "열경련"]               # 글 뒤에 위치 토큰만 붙은 꼴
    assert spotting_lines("<|TEXT_START|>짧음<|TEXT_END|><|LOC_BEGIN|><|LOC_1|><|LOC_END|>") == []   # 위치가 모자란 짝은 뺀다


def test_attach_merges_reread_lines():
    """다시 읽은 줄을 앞에, 엔진 글 가운데 같은 줄이 아닌 것을 뒤에 — 한두 글자 다르게 읽은 줄은 다시 읽은 쪽만(실측 '주세' → '추세')."""
    doc = _doc(_fig(1, text="9월 충북 평균기온 변화 주세(1973~2025년)\n(a)"))
    out = attach_figure_ocr(doc, _cache(doc, "9월 충북 평균기온 변화 추세(1973~2025년)\n평균기온 평년 편차\n23.2%"), ON).pages[0].blocks[0]
    assert out.figure_text == "9월 충북 평균기온 변화 추세(1973~2025년)\n평균기온 평년 편차\n23.2%\n(a)"
    assert out.provenance == [{"rule": "figure_ocr", "model": "m", "added": 2, "replaced": 1}]


def test_attach_drops_caption_lines_and_handles_truncated_reread():
    """넓힌 영역에 든 그림 제목 줄은 빼고, 다시 읽기도 한도로 잘렸으면 되풀이 앞 줄만(그래프는 SALVAGE_MIN_LINES 이상일 때만 —
    아니면 엔진 글 그대로), 끝까지 읽었으면 잘림 표시를 지운다. 기록이 없으면 표시만, 꺼져 있으면 그대로."""
    cap = Block(id="p1.b02", type="figure_title", bbox=(70, 402, 520, 414), text="【그림 1】 연도별 9월 충북 평균기온")
    doc = _doc(_fig(1, caption_id="p1.b02", truncated="length"), cap)               # 엔진은 잘렸고 다시 읽기는 끝까지
    out = attach_figure_ocr(doc, _cache(doc, "【그림 1】 연도별 9월 충북 평균기온\n23.1\n17.5"), ON).pages[0].blocks[0]
    assert (out.figure_text, out.truncated) == ("23.1\n17.5", None)

    eng = _doc(_fig(1, text="평균기온\n23.1"))
    loop = "평균기온\nLON 125.4\n203.0\n203.1\n203.2"
    short = attach_figure_ocr(eng, _cache(eng, loop, "length"), ON).pages[0].blocks[0]
    assert short.figure_text == "평균기온\n23.1"
    assert short.provenance == [{"rule": "figure_ocr", "model": "m", "truncated": "length", "added": 0}]
    long = "\n".join([f"값 {i}" for i in range(9)] + ["값 0"])
    salvaged = attach_figure_ocr(eng, _cache(eng, long, "length"), ON).pages[0].blocks[0]
    assert (salvaged.figure_text, salvaged.truncated) == ("\n".join(f"값 {i}" for i in range(9)) + "\n평균기온\n23.1", "length")

    assert attach_figure_ocr(eng, {}, ON).pages[0].blocks[0].provenance == [{"rule": "figure_ocr_missing"}]
    assert attach_figure_ocr(eng, {}, FigureOcrCfg()) is eng


def test_attach_pairs_lines_one_to_one_and_drops_symbol_lines():
    """다시 읽은 줄 하나는 엔진 줄 하나만 대신한다 — 한 글자·숫자만 다른 다른 표지('평년보다 낮음'·'2025년 …')는 지우지 않는다.
    가나·기호뿐인 다시 읽기는 그림 글이 되지 않는다(_noise 와 같은 기준)."""
    doc = _doc(_fig(1, text="평년보다 높음\n평년보다 낮음"))
    out = attach_figure_ocr(doc, _cache(doc, "평년보다 높음"), ON).pages[0].blocks[0]
    assert out.figure_text == "평년보다 높음\n평년보다 낮음" and out.provenance[0]["added"] == 0 and out.provenance[0]["replaced"] == 1
    doc = _doc(_fig(1, text="2025년 온열질환자 수(명)"))
    out = attach_figure_ocr(doc, _cache(doc, "2026년 온열질환자 수(명)"), ON).pages[0].blocks[0]
    assert out.figure_text == "2026년 온열질환자 수(명)\n2025년 온열질환자 수(명)"                  # 숫자가 다르면 다른 줄
    doc = _doc(_fig(1, text="5명 미만\n10~15명 미만"))                    # 실측 test4.pdf 5쪽 지도 범례 — 물결을 다른 글자로 다시 읽음
    out = attach_figure_ocr(doc, _cache(doc, "10∼15명 미만"), ON).pages[0].blocks[0]
    assert out.figure_text == "10∼15명 미만\n5명 미만"                                            # '5명 미만' 은 '15명 미만' 의 끝이 아니다
    doc = _doc(_fig(1, text="평균기온"))                                   # 한자·가나가 섞인 다시 읽은 줄은 뺀다(실측 Spotting: '4月')
    out = attach_figure_ocr(doc, _cache(doc, "4月\n768(11)\n2025年 매개모기"), ON).pages[0].blocks[0]
    assert out.figure_text == "768(11)\n평균기온"
    for junk in ("∅∅∅∅∅∅", "うっ"):
        doc = _doc(_fig(1, "image"))
        out = attach_figure_ocr(doc, _cache(doc, junk), ON).pages[0].blocks[0]
        assert out.figure_text is None and out.provenance == [{"rule": "figure_ocr", "model": "m", "added": 0}]
