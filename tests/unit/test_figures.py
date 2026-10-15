"""그림을 감싼 문단 잇기(실측 보완)."""
from kdp.assemble.figures import join_wrapped
from kdp.schema import Block, Document, Page, ParserInfo, Source


def _doc(*blocks):
    page = Page(page_no=1, sheet_no=1, size=(595, 842), blocks=[
        Block(id=f"p1.b{i + 1:02d}", type=t, bbox=bb, text=x) for i, (t, bb, x) in enumerate(blocks)])
    return Document(doc_id="sha256:ab", source=Source(paths=["a.pdf"], pages=1),
                    parser=ParserInfo(kdp_version="0", engine={}, engine_hash="e", assemble_hash="a", run_at="t"),
                    pages=[page])


FIG = ("image", (330, 100, 520, 260), None)


def test_wrapped_paragraph_joined():
    doc = join_wrapped(_doc(("text", (72, 100, 320, 250), "시스템은 행정 업무 처리를 지원하며 전자결재와"), FIG,
                            ("text", (72, 262, 520, 300), "문서관리 기능을 하나로 묶어 제공한다.")))
    a, _, b = doc.pages[0].blocks
    assert a.text == "시스템은 행정 업무 처리를 지원하며 전자결재와 문서관리 기능을 하나로 묶어 제공한다."
    assert (b.keep, b.drop_reason, b.merged_into) == (False, "merged", "p1.b01")


def test_not_joined():
    ended = join_wrapped(_doc(("text", (72, 100, 320, 250), "시스템을 구축하였다."), FIG,
                              ("text", (72, 262, 520, 300), "다음은 결과이다.")))
    item = join_wrapped(_doc(("text", (72, 100, 320, 250), "주요 기능은 다음과 같고"), FIG,
                             ("text", (72, 262, 520, 300), "□ 전자결재 기능")))
    far = join_wrapped(_doc(("text", (72, 100, 320, 250), "주요 기능은 다음과 같고"), FIG,
                            ("text", (72, 330, 520, 360), "문서관리 기능")))
    narrow = join_wrapped(_doc(("text", (72, 100, 320, 250), "주요 기능은 다음과 같고"), FIG,
                               ("text", (200, 262, 520, 300), "문서관리 기능")))
    for doc in (ended, item, far, narrow):
        assert all(b.keep for b in doc.pages[0].blocks)


def test_joined_block_takes_union_box_and_member_text_cleared():
    doc = join_wrapped(_doc(("text", (72, 100, 320, 250), "시스템은 행정 업무 처리를 지원하며 전자결재와"), FIG,
                            ("text", (72, 262, 520, 300), "문서관리 기능을 하나로 묶어 제공한다.")))
    a, _, b = doc.pages[0].blocks
    assert a.bbox == (72, 100, 520, 300) and b.text is None             # QC 가 같은 글자를 두 번 세지 않는다
    item = join_wrapped(_doc(("text", (72, 100, 320, 250), "ㅇ 사업명: 전자결재 고도화"), FIG,
                             ("text", (72, 262, 520, 300), "ㅇ 사업기간: 2026. 1. ~ 12.")))
    assert all(x.keep for x in item.pages[0].blocks)
