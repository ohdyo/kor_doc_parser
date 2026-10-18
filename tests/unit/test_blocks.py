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


def test_appendix_tag_only_headers_are_not_running():
    # 꼬리표와 제목을 따로 머리말로 잡은 붙임 첫 쪽들 — '붙임 1'·'붙임 2'·'붙임 3' 은 숫자만 다른 되풀이 머리말이 아니다
    # (실측 test3.pdf). 셋째 쪽은 표 바로 위라 제목을 그림 제목 라벨로 잡았다. 넷째 쪽 '붙임 3' 은 앞서 나온 번호라 그대로 머리말
    doc = classify_blocks(_pages(
        [("header", (66, 75, 117, 97), "붙임 1"), ("header", (138, 76, 453, 96), "기온 분포도 및 일별 경향")],
        [("header", (64, 75, 116, 96), "붙임 2"), ("header", (136, 76, 464, 96), "강수량 분포도 및 일별 경향")],
        [("header", (64, 76, 116, 96), "붙임 3"), ("figure_title", (136, 76, 517, 96), "연도별 기온, 강수량 등 정보(1973~2026년)"),
         ("table", (54, 104, 530, 764), None)],
        [("header", (64, 76, 116, 96), "붙임 3"), ("text", (60, 120, 520, 200), "표 계속")]), Classify())
    got = [(p.blocks[0].type, p.blocks[0].keep, p.blocks[0].text, p.blocks[1].merged_into) for p in doc.pages[:3]]
    assert got == [("paragraph_title", True, "붙임 1 기온 분포도 및 일별 경향", "p1.b01"),
                   ("paragraph_title", True, "붙임 2 강수량 분포도 및 일별 경향", "p2.b01"),
                   ("paragraph_title", True, "붙임 3 연도별 기온, 강수량 등 정보(1973~2026년)", "p3.b01")]
    assert (doc.pages[3].blocks[0].type, doc.pages[3].blocks[0].keep) == ("header", False)


def test_box_name_above_table_or_chart_is_title():
    # 표·그래프 바로 위 짧은 □ 이름 줄을 검출기가 글·그림 제목으로 잡았다(실측 test3.pdf 12·13쪽) — 사이 단위 줄은 건너뛴다.
    # 구역의 □ 줄이 모두 이름 줄이면 글 라벨도 제목, 그림 제목 라벨은 늘 제목
    doc = classify_blocks(_pages([
        ("paragraph_title", (64, 75, 352, 97), "붙임 6 9월 기상자료"),
        ("text", (54, 126, 480, 143), "□ 평균기온, 평균 최고기온, 강수량(1973~2026년)"), ("chart", (70, 168, 523, 390), None),
        ("figure_title", (55, 400, 225, 416), "□ 평년 대비 기상요소 값"), ("vision_footnote", (478, 400, 532, 414), "(단위: mm)"),
        ("table", (56, 420, 538, 480), None)]), Classify())
    assert [b.type for b in doc.pages[0].blocks][1:4] == ["paragraph_title", "chart", "paragraph_title"]


def test_box_name_with_box_items_stays_item():
    # 표 위 □ 글 줄과 '□ 문제점'·'□ 개선 방향' □ 항목이 한 구역에 있으면 표 위 줄도 형제 항목이고, 그림 제목 라벨이면 그 그림의
    # 제목이다(검토 지적). 범례·체크 상자처럼 기호가 둘 이상인 줄, '항목: 값'·문장·아래가 표·그림이 아닌 줄은 그대로
    doc = classify_blocks(_pages([
        ("paragraph_title", (60, 60, 300, 76), "1. 추진 현황"),
        ("text", (54, 90, 300, 104), "□ 사업 예산 현황"), ("table", (56, 110, 538, 160), None),
        ("text", (54, 170, 300, 184), "□ 문제점"), ("text", (64, 190, 520, 204), "○ 하반기 집행 부진"),
        ("figure_title", (54, 214, 300, 228), "□ 지역별 집행률"), ("chart", (70, 232, 523, 330), None),
        ("figure_title", (54, 340, 300, 354), "■ 2026년 ■ 평년(1991~2020)"), ("chart", (70, 358, 523, 450), None),
        ("text", (54, 460, 540, 474), "□ 사업 기간: 2026. 1. ~ 12."), ("table", (56, 480, 538, 520), None),
        ("text", (54, 530, 540, 544), "□ 강수량이 평년보다 매우 적었다."), ("table", (56, 550, 538, 590), None),
        ("figure_title", (54, 700, 300, 714), "□ 지역별 현황"), ("text", (60, 720, 520, 760), "본문")]), Classify())
    types = [b.type for b in doc.pages[0].blocks]
    assert (types[1], types[3], types[5]) == ("text", "text", "figure_title")
    assert (types[7], types[9], types[11], types[13]) == ("figure_title", "text", "text", "figure_title")


def test_small_images_in_logo_row_are_logos():
    # 보도자료 끝 로고 줄 — 검출기가 하나만 로고로, 둘은 그림으로 잡았다(실측 test3.pdf). 큰 그림·다른 줄 그림은 그대로
    doc = classify_blocks(_pages([
        ("footer_image", (76, 595, 140, 623), None), ("image", (266, 591, 330, 626), "대체불가 대한민국"),
        ("image", (452, 594, 526, 623), "OPEN 공공누리"), ("image", (100, 640, 500, 780), None),
        ("image", (300, 400, 360, 430), None)]), Classify())
    assert [(b.keep, b.drop_reason) for b in doc.pages[0].blocks] == [
        (False, "logo"), (False, "logo"), (False, "logo"), (True, None), (True, None)]


def test_running_tag_header_without_title_stays_header():
    # 쪽마다 찍은 꼬리표만 든 머리말('붙임 3', 한글 머리말 기능) — 옆에 합칠 제목이 없으면 문서 전체의 뿌리가 되지 않는다(검토 지적)
    doc = classify_blocks(_pages(*[[("header", (66, 30, 118, 44), "붙임 3"), ("paragraph_title", (60, 80, 300, 96), f"{n}. 내용"),
                                    ("text", (60, 120, 520, 200), "본문")] for n in (1, 2, 3)]), Classify())
    assert [(p.blocks[0].type, p.blocks[0].keep) for p in doc.pages] == [("header", False)] * 3


def test_single_tag_only_header_is_appendix_title():
    # 붙임이 하나뿐인 문서 — 꼬리표만 든 머리말('붙임 1')은 같은 줄 옆 제목이 없어도(제목은 아래 줄) 한 쪽에만 나오면 붙임
    # 제목이다. 쪽마다 찍은 꼬리표만 옆 제목을 요구한다(검토 지적)
    doc = classify_blocks(_pages(
        [("paragraph_title", (60, 120, 300, 136), "Ⅱ. 추진 계획"), ("text", (70, 145, 520, 200), "○ 하반기 시스템 전환")],
        [("header", (60, 50, 110, 66), "붙임 1"), ("paragraph_title", (180, 80, 415, 102), "세부 추진 일정"),
         ("text", (70, 120, 520, 140), "○ 1단계: 장비 규격 확정")]), Classify())
    assert (doc.pages[1].blocks[0].type, doc.pages[1].blocks[0].keep) == ("paragraph_title", True)


def test_reference_lines_do_not_block_appendix_titles():
    # 본문의 '<붙임 2 참고>'·'(붙임 2 참조)' 줄과 붙임 목록 줄('붙임 1. … 1부.')은 그 붙임 번호가 이미 나온 것으로 세지 않는다 —
    # 뒤쪽 꼬리표·제목 머리말이 붙임 제목으로 산다(검토 지적)
    for ref in ("<붙임 2 참고>", "(붙임 2 참조)", "붙임  1. 기온 분포도 1부."):
        doc = classify_blocks(_pages(
            [("text", (70, 145, 520, 200), "□ 9월 기온은 평년보다 높았다."), ("text", (70, 205, 200, 219), ref)],
            [("header", (64, 76, 111, 97), "붙임 1"), ("header", (118, 76, 380, 97), "기온 분포도"),
             ("image", (64, 120, 520, 600), None)],
            [("header", (64, 76, 111, 97), "붙임 2"), ("header", (118, 76, 380, 97), "강수량 분포도"),
             ("image", (64, 120, 520, 600), None)]), Classify())
        assert [(p.blocks[0].type, p.blocks[0].text) for p in doc.pages[1:]] == [
            ("paragraph_title", "붙임 1 기온 분포도"), ("paragraph_title", "붙임 2 강수량 분포도")], ref


def test_legend_lines_are_not_box_names():
    # 그래프 바로 위 범례 줄은 □ 이름 줄이 아니다 — 기호가 섞인 범례('■ 수출액 ● 증감률'), 그래프 안쪽에서 시작하는 범례 칸
    # ('■ 일강수량(mm)'), 글자층 글자가 없는 줄(그래프에서 떼어 낸 줄 — tables 가 그림 글로 묶는다)(검토 지적)
    rows = [("figure_title", (60, 100, 500, 114), "■ 수출액(억달러) ● 증감률(%)"),
            ("text", (300, 100, 500, 114), "■ 일강수량(mm)"), ("figure_title", (54, 100, 300, 114), "■ 평년비슷범위")]
    for i, row in enumerate(rows):
        doc = _pages([("paragraph_title", (60, 60, 300, 76), "1. 기온"), row, ("chart", (70, 120, 523, 330), None),
                      ("text", (60, 340, 520, 354), "○ 9월 기온은 평년보다 높았다.")])
        if i == 2:
            doc.pages[0].blocks[1].provenance = [{"rule": "no_layer_chars"}]
        assert classify_blocks(doc, Classify()).pages[0].blocks[1].type == row[0], row[2]
