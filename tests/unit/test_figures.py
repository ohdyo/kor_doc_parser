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
    doc = join_wrapped(_doc(("text", (72, 100, 320, 250), "시스템은 민원 접수 처리를 돕고 일정 관리와"), FIG,
                            ("text", (72, 262, 520, 300), "문서관리 기능을 하나로 묶어 제공한다.")))
    a, _, b = doc.pages[0].blocks
    assert a.text == "시스템은 민원 접수 처리를 돕고 일정 관리와 문서관리 기능을 하나로 묶어 제공한다."
    assert (b.keep, b.drop_reason, b.merged_into) == (False, "merged", "p1.b01")


def test_not_joined():
    ended = join_wrapped(_doc(("text", (72, 100, 320, 250), "시스템을 구축하였다."), FIG,
                              ("text", (72, 262, 520, 300), "다음은 결과이다.")))
    item = join_wrapped(_doc(("text", (72, 100, 320, 250), "주요 기능은 다음과 같고"), FIG,
                             ("text", (72, 262, 520, 300), "□ 민원 접수 기능")))
    far = join_wrapped(_doc(("text", (72, 100, 320, 250), "주요 기능은 다음과 같고"), FIG,
                            ("text", (72, 330, 520, 360), "문서관리 기능")))
    narrow = join_wrapped(_doc(("text", (72, 100, 320, 250), "주요 기능은 다음과 같고"), FIG,
                               ("text", (200, 262, 520, 300), "문서관리 기능")))
    for doc in (ended, item, far, narrow):
        assert all(b.keep for b in doc.pages[0].blocks)


def test_joined_block_takes_union_box_and_member_text_cleared():
    doc = join_wrapped(_doc(("text", (72, 100, 320, 250), "시스템은 민원 접수 처리를 돕고 일정 관리와"), FIG,
                            ("text", (72, 262, 520, 300), "문서관리 기능을 하나로 묶어 제공한다.")))
    a, _, b = doc.pages[0].blocks
    assert a.bbox == (72, 100, 520, 300) and b.text is None             # QC 가 같은 글자를 두 번 세지 않는다
    item = join_wrapped(_doc(("text", (72, 100, 320, 250), "ㅇ 사업명: 민원 접수 고도화"), FIG,
                             ("text", (72, 262, 520, 300), "ㅇ 사업기간: 2026. 1. ~ 12.")))
    assert all(x.keep for x in item.pages[0].blocks)


def test_table_caption_and_appendage_not_joined():
    """표에 이은 제목·출처(caption_id·note_ids)와 표·그림 제목 꼴 줄은 그림 감싼 문단에 합치지 않는다 — 합치면 표의
    caption_id 가 출력에 없는 블록을 가리키고 제목 글이 본문 끝에 붙는다."""
    side = ("text", (70, 100, 300, 250), "○ 단계별 추진 체계: 기획·구축·확산의 3단계로 구성")
    table = ("table", (70, 285, 520, 400), None)
    linked = _doc(side, FIG, ("text", (70, 260, 520, 275), "2026년 단계별 추진 기관과 역할 분담 현황 및 기관별 담당 업무"), table)
    linked.pages[0].blocks[3].caption_id = "p1.b03"
    for doc in (linked, _doc(side, FIG, ("text", (70, 260, 520, 275), "<표 1> 단계별 추진 기관                (단위: 개)"), table),
                _doc(side, FIG, ("text", (70, 260, 520, 275), "표 1. 2026년 단계별 추진 기관과 역할 분담 현황"), table)):
        out = join_wrapped(doc).pages[0].blocks
        assert all(b.keep for b in out) and out[0].text == side[2]
    doc = join_wrapped(_doc(("text", (72, 100, 320, 250), "시스템은 민원 접수 처리를 돕고 일정 관리와"), FIG,
                            ("text", (72, 262, 520, 300), "표 1과 같이 문서관리 기능을 하나로 묶어 제공한다.")))
    assert doc.pages[0].blocks[2].merged_into == "p1.b01"                   # 표를 가리키는 본문은 잇는다
    assert doc.pages[0].blocks[0].provenance == [{"rule": "join_wrapped", "with": "p1.b03", "figure": "p1.b02"}]


def test_body_referring_to_several_tables_and_title_label():
    """'표 1, 2와 같이 …' 로 시작하는 옆 글은 본문이라 잇는다. 검출기가 제목으로 본 아래 줄(문장 꼴이라 text 로 바꾼 제목
    라벨)은 새 문단의 시작이라 잇지 않는다."""
    side = ("text", (72, 100, 320, 250), "표 1, 2와 같이 시스템은 민원 접수 처리를 돕고 기관 내부의 여러 부서가 함께 쓰는")
    below = ("text", (72, 262, 520, 300), "민원 접수와 일정 관리, 게시판 기능을 하나로 묶어 제공한다.")
    assert join_wrapped(_doc(side, FIG, below)).pages[0].blocks[2].merged_into == "p1.b01"
    titled = _doc(("text", (72, 100, 320, 250), "시스템은 민원 접수 처리를 돕고 일정 관리와"), FIG,
                  ("text", (72, 262, 520, 300), "하반기 산출물 보완 현황은 아래와 같음"))
    titled.pages[0].blocks[2].provenance = [{"rule": "title_sentence", "type": "paragraph_title"}]
    assert all(b.keep and b.merged_into is None for b in join_wrapped(titled).pages[0].blocks)
