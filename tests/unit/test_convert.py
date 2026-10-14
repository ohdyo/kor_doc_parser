"""원시 결과 → 정본 변환, 라벨 정책, 순서, JSONL."""
from kdp.assemble.order import assign_order, output_hash
from kdp.classify.blocks import classify_blocks
from kdp.config import Classify
from kdp.convert import to_page
from kdp.engine.base import PageInput, RawBlock, RawPage
from kdp.pipeline import jsonl_rows
from kdp.schema import Document, ParserInfo, Source

TABLE = '<table><tr><td><img src="imgs/img_in_image_box_10_10_50_50.jpg" alt="Image"" />사진</td></tr></table>'


def _raw():
    blocks = [
        RawBlock(0, "header", (100, 20, 900, 60), "사업 계획"),
        RawBlock(1, "text", (100, 100, 900, 200), "첫 단락 이어지는", vlm={"finish_reason": "stop"}),
        RawBlock(2, "text", (100, 205, 600, 300), "", merged_into=1),
        RawBlock(3, "table", (100, 400, 900, 700), TABLE, vlm={"finish_reason": "length"}),
        RawBlock(4, "image", (100, 720, 400, 900), "추진 체계", img="p1.b05.png"),
        RawBlock(5, "number", (480, 1600, 520, 1620), "- 1 -"),
        RawBlock(6, "text", (100, 950, 900, 990), "", vlm={"error": "400: too long"}),
    ]
    inp = PageInput(page_no=1, sheet_no=1, scale=2.0, sheet_size=(595.0, 842.0))
    return RawPage(inp, 1190, 1684, 0.1, blocks, {"imgs/img_in_image_box_10_10_50_50.jpg": "p1.t01.png"})


def _doc(page):
    return Document(doc_id="sha256:ab", source=Source(paths=["a.pdf"], pages=1),
                    parser=ParserInfo(kdp_version="0", engine={}, engine_hash="e", assemble_hash="a", run_at="t"),
                    pages=[page])


def test_to_page():
    page = to_page(_raw())
    b = {x.id: x for x in page.blocks}
    assert b["p1.b02"].bbox == (50.0, 50.0, 450.0, 150.0)          # 병합 대표 = 구성원 합집합(pt)
    assert b["p1.b03"].merged_into == "p1.b02" and not b["p1.b03"].keep
    assert b["p1.b04"].html.startswith('<table><tr><td><img src="imgs/p1.t01.png" alt="Image" />')
    assert b["p1.b04"].truncated == "length"
    assert b["p1.b05"].figure_text == "추진 체계" and b["p1.b05"].img_path == "imgs/p1.b05.png"
    assert b["p1.b07"].error


def test_policy_order_jsonl():
    doc = assign_order(classify_blocks(_doc(to_page(_raw())), Classify()))
    b = {x.id: x for x in doc.pages[0].blocks}
    assert (b["p1.b01"].keep, b["p1.b01"].drop_reason) == (False, "header")
    assert b["p1.b06"].drop_reason == "page_number"
    assert [x.order for x in doc.pages[0].blocks if x.order] == [1, 2, 3]
    rows = list(jsonl_rows(doc))
    assert [r["block_id"] for r in rows] == ["p1.b02", "p1.b04", "p1.b05"]
    assert "html" in rows[1] and "text" not in rows[1]
    assert output_hash(doc) == output_hash(doc.model_copy(deep=True))
