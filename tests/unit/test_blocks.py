"""블록 라벨 정책(기획서 5.2) — 잡요소 버리기와 글상자(aside_text) 예외."""
from kdp.classify.blocks import classify_blocks, sentence_like, sentences
from kdp.config import Classify
from kdp.schema import Block, Document, Page, ParserInfo, Source


def _doc(*blocks):
    page = Page(page_no=1, sheet_no=1, size=(595, 842), blocks=[
        Block(id=f"p1.b{i + 1:02d}", type=t, bbox=(400, 100 + 60 * i, 550, 150 + 60 * i), text=x)
        for i, (t, x) in enumerate(blocks)])
    return Document(doc_id="sha256:ab", source=Source(paths=["a.pdf"], pages=1),
                    parser=ParserInfo(kdp_version="0", engine={}, engine_hash="e", assemble_hash="a", run_at="t"),
                    pages=[page])


BOX = ("aside_text", "본 사업은 2단계로 나누어 추진한다. 1단계 결과를 보고 2단계 범위를 정한다.")
NOTE = ("aside_text", "※ 참고: 단가는 부가세 포함 금액임")
SUMMARY = ("aside_text", "○ 사업 기간 연장 필요\n○ 예산 추가 확보 요망")
MEMO = ("aside_text", "대외비")
ONE = ("aside_text", "자세한 내용은 붙임을 참고한다.")


def test_aside_kept_when_it_has_content():
    blocks = classify_blocks(_doc(BOX, NOTE, SUMMARY, MEMO, ONE, ("aside_text", "2026. 3. 15. 수정")), Classify()).pages[0].blocks
    assert [(b.keep, b.aside, b.drop_reason) for b in blocks] == [
        (True, True, None), (True, True, None), (True, True, None),          # 글상자·'※ 참고' 박스·옆 요약
        (False, None, "aside"), (False, None, "aside"), (False, None, "aside")]   # 띠지·한 문장·날짜 메모
    assert blocks[0].provenance == [{"rule": "aside_keep"}] and blocks[3].provenance is None


def test_aside_min_sentences_from_config():
    blocks = classify_blocks(_doc(BOX, NOTE), Classify(aside_keep_min_sentences=3)).pages[0].blocks
    assert [b.keep for b in blocks] == [False, True]                         # '※ 참고' 박스는 문장 수와 상관없이


def test_sentences():
    assert sentences(BOX[1]) == 2 and sentences(SUMMARY[1]) == 2 and sentences("정보화 사업 결과를 보고함") == 1
    assert sentences("가. 일시: 2026. 3. 15.") == 1 and sentences("대외비") == 0   # 항목 줄 하나 — '가.'·'2026.' 은 문장 끝이 아니다


def test_label_policy():
    doc = _doc(("header", "결과 보고서"), ("footer", "한국정보공사"), ("number", "- 3 -"), ("seal", None),
               ("content", "1. 개요 ..... 3"), ("footnote", "1) 2025년 기준"), ("text", "본문"))
    doc.pages[0].blocks[6].merged_into = "p1.b01"
    out = classify_blocks(doc, Classify()).pages[0].blocks
    assert [b.drop_reason for b in out] == ["header", "footer", "page_number", "stamp", "toc", None, None]
    assert out[5].keep and out[6].keep                                       # 각주는 살린다(D11), 병합 구성원은 그대로
    assert classify_blocks(doc, Classify(footnote="drop")).pages[0].blocks[5].drop_reason == "footnote"


def test_note_boxes_with_report_bullets_are_kept():
    notes = [("aside_text", x) for x in ("□ 참고: 단가는 부가세 포함 금액임", "■ 참고: 단가는 부가세 포함 금액임",
                                          "참고자료: 단가는 부가세 포함 금액임", "[참고] 단가는 부가세 포함 금액임")]
    blocks = classify_blocks(_doc(*notes, ("aside_text", "참고로 2025년 자료를 썼다")), Classify()).pages[0].blocks
    assert [(b.keep, b.aside) for b in blocks] == [(True, True)] * 4 + [(False, None)]   # '참고로 …' 한 문장은 참고 상자가 아니다


def test_sentence_titles_become_text():
    """문단 제목 라벨인데 문장 꼴(실측: 사내 시험 문서의 '■ … 아래와 같음' 꼴 항목 줄을 paragraph_title 로 잡음) → text, 이전 type 은 provenance."""
    doc = _doc(("paragraph_title", "■ 하반기 과제 보완 현황은 아래와 같음"), ("paragraph_title", "정보화 사업 추진 결과를 다음과 같이 보고합니다"),
               ("paragraph_title", "추진 배경"), ("paragraph_title", "2. 사업 개요"), ("paragraph_title", "개인정보 처리 위탁 및 관리 책임"),
               ("paragraph_title", "나. 개선 완료함"))
    blocks = classify_blocks(doc, Classify()).pages[0].blocks
    assert [b.type for b in blocks] == ["text", "text"] + ["paragraph_title"] * 4      # 명사 제목·짧은 제목은 그대로
    assert blocks[0].provenance == [{"rule": "title_sentence", "type": "paragraph_title"}] and blocks[0].keep
    assert sentence_like("Ⅱ. 사업 추진 결과 목표를 초과 달성하였음") and not sentence_like("2026년 정보화 추진 실적 및 개선 필요")
    assert not sentence_like("국민과 함께 디지털 정부를 만들다")                          # 표어 꼴 장 제목은 그대로


def test_sentence_doc_titles_stay_doc_titles():
    """문서 제목(doc_title)은 문장 꼴이어도 그대로 — 표어·보도자료 제목이 text 가 되면 표지 판정과 문서 메타를 잃는다(M2 회귀)."""
    titles = ("데이터가 이끄는 행정, 국민이 체감한다", "정부, 공공데이터 3만 개 추가 개방한다", "정보화 사업 추진 결과를 다음과 같이 보고합니다")
    blocks = classify_blocks(_doc(*[("doc_title", x) for x in titles]), Classify()).pages[0].blocks
    assert all(sentence_like(x) for x in titles)
    assert [(b.type, b.provenance) for b in blocks] == [("doc_title", None)] * 3


def _pages(*pages):
    return Document(doc_id="sha256:ab", source=Source(paths=["a.pdf"], pages=len(pages)),
                    parser=ParserInfo(kdp_version="0", engine={}, engine_hash="e", assemble_hash="a", run_at="t"),
                    pages=[Page(page_no=n + 1, sheet_no=n + 1, size=(595, 842), blocks=[
                        Block(id=f"p{n + 1}.b{i + 1:02d}", type=t, bbox=bb, text=x) for i, (t, bb, x) in enumerate(bs)])
                        for n, bs in enumerate(pages)])


TAG = ("paragraph_title", (64, 76, 111, 97), "붙임 1")


def test_appendix_tag_takes_title_beside_it():
    doc = classify_blocks(_pages(
        [TAG, ("header", (132, 76, 505, 96), "말라리아 매개모기 발생 현황"), ("text", (60, 120, 520, 200), "본문")],
        [("paragraph_title", (64, 76, 112, 96), "붙임 2"), ("header", (74, 76, 455, 97), "발령 기준 및 조치사항")],
        [("header", (66, 76, 458, 97), "붙임 3 매개모기 특징 및 감시 지점")]), Classify())
    p1, p2, p3 = doc.pages
    assert (p1.blocks[0].type, p1.blocks[0].text, p1.blocks[0].keep) == ("paragraph_title", "붙임 1 말라리아 매개모기 발생 현황", True)
    assert p1.blocks[0].bbox == (64, 76, 505, 97)
    assert (p1.blocks[1].keep, p1.blocks[1].drop_reason, p1.blocks[1].merged_into, p1.blocks[1].text) == \
        (False, "merged", "p1.b01", None)                                    # 쪽 맨 위 제목 상자를 머리말로 잡은 것
    assert p2.blocks[0].text == "붙임 2 발령 기준 및 조치사항" and p2.blocks[1].merged_into == "p2.b01"   # 상자가 꼬리표와 겹친다
    assert (p3.blocks[0].type, p3.blocks[0].keep) == ("paragraph_title", True)                       # 꼬리표와 제목이 한 블록
    assert p3.blocks[0].provenance == [{"rule": "appendix_title", "type": "header"}]


def test_appendix_tag_leaves_running_header_and_list_lines():
    run = ("header", (66, 30, 458, 44), "붙임 1 사업계획서")                 # 쪽마다 찍은 머리말
    doc = classify_blocks(_pages(
        [run, TAG, ("text", (132, 76, 505, 96), "1. 계획서 1부.")],
        [run, ("header", (66, 76, 458, 97), "붙임 2. 명단 1부.  끝.")]), Classify())
    p1, p2 = doc.pages
    assert (p1.blocks[0].keep, p1.blocks[0].drop_reason) == (False, "header")
    assert p1.blocks[1].text == "붙임 1" and p1.blocks[2].keep and p1.blocks[2].merged_into is None   # 번호 항목은 잇지 않는다
    assert (p2.blocks[1].type, p2.blocks[1].keep) == ("header", False)       # 붙임 목록 줄은 제목이 아니다


def test_appendix_title_sentence_stays_root():
    from kdp.assemble.headings import headings
    doc = headings(classify_blocks(_pages(
        [("doc_title", (150, 40, 450, 60), "보도 제목"), ("text", (60, 100, 520, 200), "본문")],
        [("paragraph_title", (64, 76, 111, 97), "붙임 2"), ("paragraph_title", (132, 76, 505, 96), "청년 월세 지원, 이렇게 신청하세요"),
         ("text", (60, 120, 520, 140), "○ 월세 지원 대상을 확대함")]), Classify()))
    tag, _, item = doc.pages[1].blocks
    assert (tag.type, tag.heading_level) == ("paragraph_title", 1)              # 문장 꼴 제목이어도 붙임 뿌리
    assert item.section_path == ["붙임 2 청년 월세 지원, 이렇게 신청하세요"]


def test_appendix_header_seen_before_stays_header():
    doc = classify_blocks(_pages(
        [("paragraph_title", (64, 76, 450, 97), "붙임 1 사업 추진계획"), ("text", (60, 120, 520, 200), "1. 추진 배경")],
        [("header", (66, 30, 458, 44), "붙임 1 사업 추진계획"), ("text", (60, 120, 520, 200), "본문 계속")],
        [("header", (66, 30, 458, 44), "붙임 2 세부 일정 (1/3)"), ("text", (60, 120, 520, 200), "일정")],
        [("header", (66, 30, 458, 44), "붙임 2 세부 일정 (2/3)"), ("text", (60, 120, 520, 200), "일정")]), Classify())
    assert (doc.pages[1].blocks[0].type, doc.pages[1].blocks[0].keep) == ("header", False)   # 앞쪽 제목과 같은 붙임 번호
    assert all(p.blocks[0].type == "header" for p in doc.pages[2:])                          # 숫자만 다른 되풀이 머리말


def test_short_appendix_page_is_not_divider():
    from kdp.classify.pages import classify_pages
    doc = classify_pages(classify_blocks(_pages(
        [("doc_title", (150, 40, 450, 60), "보도 제목"), ("text", (60, 100, 520, 300), "본문 " * 40)],
        [("paragraph_title", (64, 76, 111, 97), "붙임 4"), ("paragraph_title", (132, 76, 505, 96), "관련 누리집 안내"),
         ("text", (60, 120, 520, 140), "질병관리청 누리집 www.kdca.go.kr")]), Classify()), Classify())
    assert doc.pages[1].keep and doc.pages[1].page_class != "divider"            # 합친 꼬리표 제목은 간지 판정에서 두 블록


def test_appendix_tag_leaves_doc_title_captions_and_list_lines():
    for typ, text in [("doc_title", "2026년 감시 결과"), ("text", "<표 1> 시·도별 현황"), ("text", "[ 시·도별 현황 ]"),
                      ("text", "<그림 1> 감시 지점"), ("text", "계획서 1부.  끝.")]:
        doc = classify_blocks(_pages([TAG, (typ, (132, 76, 505, 96), text)]), Classify())
        tag, other = doc.pages[0].blocks
        assert tag.text == "붙임 1" and other.merged_into is None, text


def test_appendix_header_with_earlier_title_stays_header():
    doc = classify_blocks(_pages(
        [("doc_title", (100, 60, 500, 90), "2026년 정보화 사업 추진계획"), ("text", (60, 120, 520, 200), "1. 추진 배경")],
        [("header", (66, 30, 458, 44), "붙임 1 2026년 정보화 사업 추진계획"), ("text", (60, 120, 520, 200), "본문 계속")]), Classify())
    assert (doc.pages[1].blocks[0].type, doc.pages[1].blocks[0].keep) == ("header", False)   # 앞쪽 제목 글과 같으면 되풀이 머리말


def test_appendix_tag_leaves_chapter_title():
    for text in ("Ⅰ. 추진 배경", "제1장 추진 배경"):
        doc = classify_blocks(_pages([TAG, ("paragraph_title", (132, 76, 505, 96), text)]), Classify())
        assert doc.pages[0].blocks[0].text == "붙임 1" and doc.pages[0].blocks[1].merged_into is None, text


def test_instruction_sentence_with_tag_is_not_root():
    doc = classify_blocks(_pages([("paragraph_title", (60, 100, 540, 116), "붙임 1 서식에 따라 작성하여 3. 20.(금)까지 제출하여 주시기 바랍니다.")]),
                          Classify())
    assert doc.pages[0].blocks[0].type == "text"                                  # 꼬리표와 합친 제목만 문장 꼴 예외
