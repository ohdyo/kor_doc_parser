"""텍스트 레이어 우선(D13) — 블록 글자 바꾸기, 표 칸 맞추기, 깨진·가려진·보이지 않는 레이어 막기."""
import numpy as np

from kdp.io.pdf import PageInfo, PdfInfo
from kdp.schema import Block, Document, Page, ParserInfo, Source
from kdp.textlayer.extract import _fix_table, _join, apply_text_layer, mark_hidden, mixed_regions, plausible, replaceable

H = 842.0
INFO = PdfInfo(doc_id="sha256:ab", path="a.pdf", n_pages=1,
               pages=[PageInfo(sheet_no=1, size=(595, H), rotation=0, bbox=(0, 0, 595, H), text_chars=0)])
BODY = "감리 대상 사업 전반에 대하여 단계별 이행 여부를 점검하고 보완 조치를 요구한다"
OTHER = "감리 결과 보고 내용을 정리하여 기한 안에 제출한다"
SKIPPED = [{"rule": "vlm_skipped", "reason": "text_layer"}]


def _chars(text, x=72.0, y=100.0, size=10.0, **flags):
    """줄마다 글자 상자(사용자 공간, 좌하단 원점)를 만든다. 공백·줄바꿈은 PDFium 처럼 생성 문자로."""
    out = []
    for n, line in enumerate(text.split("\n")):
        top = H - (y + n * 2 * size)
        cx = x
        for c in line:
            if c == " ":
                out.append({"c": " ", "box": [cx, top - size, cx, top], "gen": True})
                cx += size / 2
                continue
            out.append({"c": c, "box": [cx, top - size, cx + size, top], "gen": False, **flags})
            cx += size
        out.append({"c": "\r", "box": [cx, top - size, cx, top], "gen": True})
        out.append({"c": "\n", "box": [cx, top - size, cx, top], "gen": True})
    return out


def _tuples(chars):
    return [(c["c"], None, c["gen"]) for c in chars]


def _doc(*blocks):
    page = Page(page_no=1, sheet_no=1, size=(595, H), blocks=[
        Block(id=f"p1.b{i + 1:02d}", type=t, bbox=bb, text=x, html=h) for i, (t, bb, x, h) in enumerate(blocks)])
    return Document(doc_id="sha256:ab", source=Source(paths=["a.pdf"], pages=1),
                    parser=ParserInfo(kdp_version="0", engine={}, engine_hash="e", assemble_hash="a", run_at="t"),
                    pages=[page])


def test_join_spaces_bullets_and_lines():
    chars = _tuples(_chars("가 나\n다"))
    assert _join(chars, newline=False) == ("가 나 다", False)
    assert _join(chars, newline=True) == ("가 나\n다", False)
    head = [("", None, False), ("가", None, False), ("�", None, False)]
    assert _join(head, newline=False) == ("•가", False)              # Symbol 글머리표, U+FFFD 는 버림
    box = [("", None, False), ("동", None, False), ("", None, False), ("거", None, False)]
    assert _join(box, newline=False) == ("☑동☐거", False)             # 체크 상자는 기호로
    odd = [("가", None, False), ("", None, False), ("나", None, False)]
    assert _join(odd, newline=False) == ("가나", True)               # 뜻 모를 기호가 줄 중간에서 빠짐


def test_unknown_symbol_at_line_head_is_lost():
    assert _join([("\ue123", None, False), ("가", None, False)], newline=False) == ("가", True)   # 체크 상자일 수 있다


def test_vlm_only_run_and_strict_plausible():
    name = BODY.replace("점검하고", "홍길동 점검하고")                 # 모델 글에만 있는 이름(레이어에 없는 입력란 값)
    assert not replaceable(BODY, name)
    assert replaceable(BODY, BODY.replace("단계별", "닌계별").replace("여부를", "여부롤"))   # 흩어진 오독은 바꾼다
    p = apply_text_layer(_doc(("text", (70, 95, 560, 115), name, None)), {1: _chars(BODY)}, INFO, prefer=True).pages[0]
    assert p.blocks[0].text == name
    assert plausible("똠똠똠똠 결과") and not plausible("똠똠똠똠 결과", strict=True)   # 짧은 블록도 드문 한글을 본다
    assert plausible("監理結果報告") and not plausible("監理結果報告", strict=True)


def test_layer_replaces_close_text_keeps_picture_text():
    wrong = BODY.replace("단계별", "닌계별")                          # 모델이 한 글자 잘못 읽음
    doc = _doc(("text", (70, 95, 560, 115), wrong, None), ("image", (70, 300, 300, 400), None, None))
    doc.pages[0].blocks[1].figure_text = "그림 속 글자"
    p = apply_text_layer(doc, {1: _chars(BODY)}, INFO, prefer=True).pages[0]
    assert p.born_digital and p.text_layer_agreement > 0.95
    assert p.blocks[0].text == BODY and p.blocks[0].provenance[0]["before"] == wrong
    assert p.blocks[1].figure_text == "그림 속 글자"
    off = apply_text_layer(doc, {1: _chars(BODY)}, INFO, prefer=False)
    assert off.pages[0].blocks[0].text == wrong and off.pages[0].text_layer_agreement is not None


def test_layer_missing_line_keeps_vlm():
    vlm = BODY + " 보완 기한은 통보일로부터 삼십일 이내로 한다"         # 레이어에 없는 줄(채운 입력란 등)
    doc = _doc(("text", (70, 95, 560, 135), vlm, None))
    p = apply_text_layer(doc, {1: _chars(BODY)}, INFO, prefer=True).pages[0]
    assert p.blocks[0].text == vlm                                  # 빠진 줄을 지우지 않는다


def test_garbled_or_invisible_layer_not_used():
    garbled = "".join(chr(0x4E00 + (ord(c) % 500)) if not c.isspace() else c for c in BODY)   # 깨진 글자 대응표
    doc = _doc(("text", (70, 95, 560, 115), BODY, None))
    out = apply_text_layer(doc, {1: _chars(garbled)}, INFO, prefer=True)
    assert out.pages[0].text_layer_agreement < 0.8 and out.pages[0].blocks[0].text == BODY
    scan = apply_text_layer(doc, {1: []}, INFO, prefer=True).pages[0]
    assert scan.born_digital is False and scan.text_layer_agreement is None
    ocr = apply_text_layer(doc, {1: _chars(BODY.replace("단계별", "딘계별"), inv=True)}, INFO, prefer=True).pages[0]
    assert ocr.born_digital is False and ocr.blocks[0].text == BODY      # 스캔 PDF 의 투명 OCR 레이어는 정답이 아니다


def test_hidden_chars_not_used():
    chars = _chars("담당자 홍길동 주무관은 감리 결과를 보고한다")
    for c in chars:
        if c["c"] in "홍길동":                                       # 가림 상자 아래 이름
            c["hid"] = True
    shown = "담당자 주무관은 감리 결과를 보고한다"
    p = apply_text_layer(_doc(("text", (70, 95, 560, 115), shown, None)), {1: chars}, INFO, prefer=True).pages[0]
    assert "홍길동" not in p.blocks[0].text


def test_mark_hidden():
    img = np.full((200, 400, 3), 255, np.uint8)
    img[20:40, 20:40] = 0                                            # 꽉 칠한 검은 상자 아래 글자
    img[20:40, 100:120:3] = 0                                        # 획이 있는 보통 글자
    img[20:40, 200:203] = 0                                          # 'l' 처럼 원래 꽉 찬 가는 글자

    def box(x0, x1):
        return [x0 / 2, 100 - 20, x1 / 2, 100 - 10]                  # px = pt × 2, 사용자 공간(좌하단 원점)

    chars = [{"c": "가", "box": box(20, 40), "gen": False}, {"c": "나", "box": box(100, 120), "gen": False},
             {"c": "l", "box": box(200, 203), "gen": False}, {"c": "다", "box": box(300, 320), "gen": False},
             {"c": ".", "box": box(340, 350), "gen": False}, {"c": "라", "box": [180, 89, 181.5, 90], "gen": False}]
    out = mark_hidden(chars, img, (0, 0, 200, 100), 0, 2.0)
    assert [c.get("hid", False) for c in out] == [True, False, False, True, False, False]   # 칠한 상자·빈 자리만
    # 문장부호와 아주 작은 상자는 보지 않는다 — 내장 안 한 글꼴은 상자가 글자 자리와 어긋난다


def test_table_cells_matched_in_order():
    layer = "사업 관리 5건 완료\n사업 관리 6건 완료\n데이터 표준화 점검"
    html = ("<table><tr><td>사업 관리 6건 완료</td><td>사업 관리 6건 왼료</td></tr>"
            "<tr><td>데이터\\n표준화 점검</td><td>O</td></tr></table>")
    fixed, n = _fix_table(html, _tuples(_chars(layer)))
    assert fixed == ("<table><tr><td>사업 관리 5건 완료</td><td>사업 관리 6건 완료</td></tr>"
                     "<tr><td>데이터 표준화 점검</td><td>O</td></tr></table>")
    assert n == 3                                                    # 짧은 칸('O')은 그대로


def test_table_edge_char_errors_and_missing_row():
    layer = "인건비 12,500 11,235\n2024.01.15 1,000\n2024.02.15 2,000\n2024.03.15 3,000"
    html = ("<table><tr><td>인건비</td><td>72,500</td><td>11,236</td></tr>"
            "<tr><td>2024.01.15</td><td>1,000</td></tr><tr><td>2024.03.15</td><td>3,000</td></tr></table>")
    fixed, _ = _fix_table(html, _tuples(_chars(layer)))              # 첫·끝 글자 오독, 2월 행은 모델이 빠뜨림
    assert ">12,500<" in fixed and ">11,235<" in fixed
    assert "<td>2024.03.15</td><td>3,000</td>" in fixed and "2024.02.15" not in fixed


def test_table_with_compat_chars_still_fixed():
    fixed, n = _fix_table("<table><tr><td>대지 면적 1,234㎡</td><td>사업 기긴 2년</td></tr></table>",
                          _tuples(_chars("대지 면적 1,234㎡ 사업 기간 2년")))
    assert n == 1 and "사업 기간 2년" in fixed                      # ㎡ 가 있어도 교정을 끄지 않는다


def test_mixed_regions():
    def row(x0, n, y=100):
        return [(chr(0xAC00 + i), (x0 + 10 * i, y, x0 + 10 * i + 9, y + 10), False) for i in range(n)]

    assert not mixed_regions(row(72, 20) + row(72, 20, y=115))
    assert mixed_regions(row(72, 10) + row(400, 10))                 # 같은 줄에 멀리 떨어진 두 영역


def _skip_doc(sample_text, *extra):
    doc = _doc(("text", (70, 95, 560, 115), sample_text, None), ("text", (70, 135, 560, 155), "", None), *extra)
    doc.pages[0].blocks[1].provenance = list(SKIPPED)
    return doc


def test_skipped_blocks_filled_or_flagged():
    chars = _chars(BODY) + _chars(OTHER, y=140)
    ok = apply_text_layer(_skip_doc(BODY), {1: chars}, INFO, prefer=True).pages[0]
    assert ok.text_layer_agreement > 0.95 and ok.blocks[1].text == OTHER and ok.blocks[1].error is None
    bad = apply_text_layer(_skip_doc("모델이 전혀 다른 내용을 읽어 레이어와 맞지 않는 표본 글"), {1: chars}, INFO, prefer=True).pages[0]
    assert bad.text_layer_agreement < 0.8 and not bad.blocks[1].text and "다시 처리" in bad.blocks[1].error
    mojibake = _chars(BODY) + _chars("¥². °¨¸® ¼öÇà °á°ú", y=140)   # 제목 글꼴만 깨진 쪽
    broken = apply_text_layer(_skip_doc(BODY), {1: mojibake}, INFO, prefer=True).pages[0]
    assert not broken.blocks[1].text and "깨졌다" in broken.blocks[1].error


def test_table_misread_does_not_distrust_page():
    chars = _chars(BODY) + _chars(OTHER, y=140) + _chars("구분 내용 비고 사업관리 요구사항 반영 완료", y=200)
    table = ("table", (70, 195, 560, 215), None, "<table><tr><td>전혀</td><td>다른</td><td>표</td></tr></table>")
    p = apply_text_layer(_skip_doc(BODY, table), {1: chars}, INFO, prefer=True).pages[0]
    assert p.blocks[1].text == OTHER and p.blocks[1].error is None   # 표 오독은 표 갈래가 맡는다


def _cells(html):
    import re
    return re.findall(r"<td>(.*?)</td>", html)


def test_table_alignment_regressions():
    fix = lambda html, layer: _fix_table(html, _tuples(_chars(layer)))[0]   # noqa: E731
    assert _cells(fix("<table><tr><td>사업 관리 6건 완료.</td><td>2024년 3월</td></tr></table>",
                      "사업 관리 6건 완료 2024년 3월")) == ["사업 관리 6건 완료", "2024년 3월"]   # 칸 끝 글자를 더 읽음
    assert _cells(fix("<table><tr><td>1,500원</td><td>2,000</td></tr></table>", "1,500 2,000")) == ["1,500", "2,000"]
    rep = "<table><tr><td>1,600</td><td>1,600</td><td>1,600</td><td>1,700</td></tr></table>"
    assert _cells(fix(rep, "1,500 1,600 1,600 1,700")) == ["1,500", "1,600", "1,600", "1,700"]   # 같은 값이 이어짐
    assert _cells(fix("<table><tr><td>1,234㎡</td></tr></table>", "1,234m²")) == ["1,234m²"]
    jamo = "단"                                       # 풀린 자모 '단'
    assert _cells(fix("<table><tr><td>사업관리단</td></tr></table>", "사업관리" + jamo)) == ["사업관리단"]


def test_table_column_order_and_unknown_symbol():
    html = ("<table><tr><td>가나다라마바</td><td>1,234,567</td></tr>"
            "<tr><td>사아자차카타</td><td>7,654,321</td></tr></table>")
    column_major = "가나다라마바 사아자차카타 1,234,567 7,654,321"            # 레이어가 열 순서
    fixed = _cells(_fix_table(html, _tuples(_chars(column_major)))[0])
    assert fixed == ["가나다라마바", "1,234,567", "사아자차카타", "7,654,321"]   # 맞던 칸을 바꾸지 않는다
    sym = [("", None, False) if c["c"] == "#" else (c["c"], None, c["gen"])
           for c in _chars("사업 관리 6건 완료 # 데이터 표준화 점검")]
    out = _cells(_fix_table("<table><tr><td>사업 관리 6건 왼료</td><td>데이터 표준화 점검</td></tr></table>", sym)[0])
    assert out[0] == "사업 관리 6건 완료"                              # 뜻 모를 기호가 있어도 다른 칸은 고친다
