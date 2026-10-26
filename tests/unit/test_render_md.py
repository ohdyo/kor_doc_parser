"""정본 → 마크다운(기획서 4.2·4.4, 설계 6장)."""
import re

import pytest

from kdp.assemble.order import kept
from kdp.render_md import render_markdown, render_rag_markdown
from kdp.schema import Block, Document, Page, ParserInfo, Source

ANCHOR = re.compile(r'^<a id="([^"]+)"></a>$', re.M)
TABLE = '<table><tr><th>구분</th><th>금액</th></tr><tr><td rowspan="2">합계</td><td>1,500</td></tr></table>'
# (블록 글의 줄, 마크다운 줄) — 위는 이스케이프할 것, 아래는 그대로 둘 것(앵커 줄에 붙은 문단 이어짐 줄이라 다르게 읽히지 않는다)
ESCAPES = [("# 참고", "\\# 참고"), ("1. 관련", "1\\. 관련"), ("1) 관련", "1\\) 관련"), ("01. 개요", "01\\. 개요"),
           ("- 세부", "\\- 세부"), ("* 주: 잠정치", "\\* 주: 잠정치"), ("+ 추가", "\\+ 추가"), ("> 인용", "\\> 인용"),
           ("```", "\\```"), ("~~~", "\\~~~"), ("<Table 1> 현황", "\\<Table 1> 현황"), ("<!-- 메모", "\\<!-- 메모"),
           ("===", "\\==="), ("---", "\\---"), ("가 | 나", "가 | 나"), ("---|---", "\\---|---"),
           ("  - 들여쓴 줄", "  \\- 들여쓴 줄")]
PLAIN = ["2. 추진 일정", "2026. 3. 15. 시행", "10. 기타", "1.5배 증가", "#해시태그", "-5% 감소", "<표 1> 현황",
         "가. 일시", "(1) 개요", "① 첫째", "□ 추진", "ㅇ 사업명", "※ 참고"]


def _page(no, *specs, **kw):
    """specs = (type, text[, 다른 필드]) — id 는 p{no}.b01 부터."""
    return Page(page_no=no, sheet_no=no, size=(595, 842), **kw, blocks=[
        Block(id=f"p{no}.b{i + 1:02d}", type=t, bbox=(70, 100 + 30 * i, 520, 120 + 30 * i), text=x, **(f[0] if f else {}))
        for i, (t, x, *f) in enumerate(specs)])


def _doc(*pages):
    return Document(doc_id="sha256:ab", source=Source(paths=["a.pdf"], pages=len(pages)),
                    parser=ParserInfo(kdp_version="0", engine={}, engine_hash="e", assemble_hash="a", run_at="t"),
                    pages=list(pages))


def _md(*pages):
    return render_markdown(_doc(*pages))


def _chunks(md):
    """빈 줄로 나눈 덩이 — 첫 덩이는 쪽 주석."""
    return md.rstrip("\n").split("\n\n")


def _tbl(x):
    return f"<table><tr><td>{x}</td></tr></table>"


def test_headings_and_unleveled_titles():
    md = _md(_page(1, ("doc_title", "○○ 시스템 고도화\n결과 보고서", {"heading_level": 0}),   # 문서 제목 0 이 '#'
                   ("paragraph_title", "1. 추진 배경", {"heading_level": 1}),
                   ("text", "가. 세부 목표", {"heading_level": 9}),          # 수준이 있으면 제목 — 6 을 넘지 않는다
                   ("paragraph_title", "주요 내용"),                       # 수준 없는 제목은 굵은 글 — '#' 수를 지어내지 않는다
                   ("doc_title", "결과 보고서")))
    assert md == ('<!-- page 1 -->\n\n'
                  '<a id="p1.b01"></a>\n# ○○ 시스템 고도화 결과 보고서\n\n'
                  '<a id="p1.b02"></a>\n## 1. 추진 배경\n\n'
                  '<a id="p1.b03"></a>\n###### 가. 세부 목표\n\n'
                  '<a id="p1.b04"></a>\n**주요 내용**\n\n'
                  '<a id="p1.b05"></a>\n**결과 보고서**\n')


def test_list_items_keep_symbols_and_indent():
    md = _md(_page(1, ("text", "□ 추진 배경", {"list_level": 1}),
                   ("text", "○ 시스템 노후화로\n잦은 장애", {"list_level": 2}),
                   ("text", "- 세부 내용", {"list_level": 3}),
                   ("text", "· 참고 사항", {"list_level": 4}),
                   ("text", "본문 문단")))
    assert _chunks(md)[1:] == ['<a id="p1.b01"></a>\n□ 추진 배경',
                               '<a id="p1.b02"></a>\n  ○ 시스템 노후화로\n  잦은 장애',
                               '<a id="p1.b03"></a>\n    \\- 세부 내용',      # 기호는 원문대로 — 목록 문법으로 읽히지 않게만
                               '<a id="p1.b04"></a>\n      · 참고 사항',
                               '<a id="p1.b05"></a>\n본문 문단']


def test_escape_only_what_markdown_reads_differently():
    text = "\n".join([s for s, _ in ESCAPES] + PLAIN)
    md = _md(_page(1, ("text", "앞 줄\n" + text)))
    assert _chunks(md)[1].split("\n")[2:] == [e for _, e in ESCAPES] + PLAIN
    first = _md(_page(1, ("text", "- 첫 줄도"), ("text", "2. 첫 줄")))  # 블록 첫 줄도 문단 이어짐 줄이다
    assert _chunks(first)[1:] == ['<a id="p1.b01"></a>\n\\- 첫 줄도', '<a id="p1.b02"></a>\n2. 첫 줄']


def test_blank_lines_inside_block_removed():
    md = _md(_page(1, ("text", "첫 문단\n\n2. 둘째 줄  ")))
    assert _chunks(md)[1:] == ['<a id="p1.b01"></a>\n첫 문단\n2. 둘째 줄']   # 빈 줄은 블록 경계 — 블록 안에서는 뺀다


def test_table_html_as_is_and_caption_once():
    md = _md(_page(1, ("figure_title", "<표 1> 사업비 현황"), ("text", "(단위: 천원)"),
                   ("table", None, {"html": TABLE, "caption_id": "p1.b01", "unit": "천원", "header_rows": 1,
                                    "route": "html", "img_path": "imgs/p1.b03.png"}),
                   ("text", "자료: 내부 집계")))
    assert _chunks(md)[1:] == ['<a id="p1.b01"></a>\n<표 1> 사업비 현황', '<a id="p1.b02"></a>\n(단위: 천원)',
                               '<a id="p1.b03"></a>\n' + TABLE, '<a id="p1.b04"></a>\n자료: 내부 집계']
    assert md.count("사업비 현황") == 1 and "imgs/" not in md               # 표 제목은 제자리에 한 번만, 크롭 링크 없음
    failed = _md(_page(1, ("table", None, {"html": TABLE, "route": "image", "route_reason": "failed",
                                           "img_path": "imgs/p1.b01.png", "figure_text": "사진 속 글자"})))
    assert _chunks(failed)[1:] == ['<a id="p1.b01"></a>\n' + TABLE, "![table](imgs/p1.b01.png)",
                                   "```text\n사진 속 글자\n```"]          # 이미지 갈래 표는 HTML 뒤에 크롭도(D18)


def test_figure_link_and_figure_text():
    md = _md(_page(1, ("image", None, {"img_path": "imgs/p1.b01.png", "figure_text": "추진 체계\n\n총괄 PM\n- 개발팀"}),
                   ("chart", None, {"figure_text": "연도|금액\n2025|100"}),
                   ("image", None, {"img_path": "imgs/p1.b03.png"}),
                   ("image", None, {"img_path": "imgs/p1.b04.png", "figure_text": "```\n코드"}),
                   ("table", None, {"img_path": "imgs/p1.b05.png"})))         # HTML 이 없는 표는 크롭만
    assert _chunks(md)[1:] == ['<a id="p1.b01"></a>\n![image](imgs/p1.b01.png)',
                               "```text\n추진 체계\n총괄 PM\n- 개발팀\n```",     # 줄을 지키고 이스케이프하지 않는다
                               '<a id="p1.b02"></a>\n```text\n연도|금액\n2025|100\n```',
                               '<a id="p1.b03"></a>\n![image](imgs/p1.b03.png)',
                               '<a id="p1.b04"></a>\n![image](imgs/p1.b04.png)',
                               "````text\n```\n코드\n````",                    # 글 속 ``` 보다 긴 울타리
                               '<a id="p1.b05"></a>\n![table](imgs/p1.b05.png)']


def test_figure_description_and_text_route():
    """그림 설명(D19)은 그림 링크 바로 뒤 문단으로, 글 갈래 그림의 그림 속 글은 울타리 없이 문단으로(내용이라 이스케이프)."""
    md = _md(_page(1, ("chart", None, {"img_path": "imgs/p1.b01.png", "figure_text": "기온(℃)\n23.1", "route": "image",
                                       "figure_desc": "남서쪽이 23.1℃로 가장 높다."}),
                   ("image", None, {"img_path": "imgs/p1.b02.png", "figure_text": "- 3시간마다 모기기피제 사용\n야간활동 시 밝은색 긴 옷",
                                    "route": "text"})))
    assert _chunks(md)[1:] == ['<a id="p1.b01"></a>\n![chart](imgs/p1.b01.png)', "그림 설명: 남서쪽이 23.1℃로 가장 높다.",
                               "```text\n기온(℃)\n23.1\n```",
                               '<a id="p1.b02"></a>\n![image](imgs/p1.b02.png)',
                               "\\- 3시간마다 모기기피제 사용\n야간활동 시 밝은색 긴 옷"]
    numbered = _md(_page(1, ("image", None, {"img_path": "imgs/p1.b01.png", "route": "text",
                                             "figure_text": "    2. 의심 증상 시 보건소 신고\n3) 귀국 후 진료"})))
    assert _chunks(numbered)[2] == "2\\. 의심 증상 시 보건소 신고\n3) 귀국 후 진료"   # 새 문단 첫 줄은 어느 번호든 목록·들여쓰기는 코드


def test_display_formula():
    md = _md(_page(1, ("display_formula", " $$ E = mc^2 $$ ", {"img_path": "imgs/p1.b01.png"}),   # PaddleX 가 \[ \] 를 바꾼 꼴
                   ("display_formula", "\\frac{a}{b} = \\$5\n\n+ c"),          # 구분자 없는 꼴, \$ 는 글자 $
                   ("formula_number", "(1)")))
    assert _chunks(md)[1:] == ['<a id="p1.b01"></a>\n$$\nE = mc^2\n$$',
                               '<a id="p1.b02"></a>\n$$\n\\frac{a}{b} = \\$5\n+ c\n$$',
                               '<a id="p1.b03"></a>\n(1)']


def test_page_comments_and_dropped_pages_blocks():
    md = _md(_page(1, ("header", "결과 보고서", {"keep": False, "drop_reason": "header"}), ("text", "첫 쪽 본문")),
             _page(2, ("text", "목차 쪽 글"), keep=False, page_class="toc"),   # keep=false 쪽은 블록 keep 과 상관없이 빠진다
             _page(3, ("doc_meta", "문서번호 제2026-1호", {"keep": False, "drop_reason": "meta"}),
                   ("text", "셋째 쪽 본문"), ("number", "- 3 -", {"keep": False, "drop_reason": "page_number"})),
             _page(4, ("header", "결과 보고서", {"keep": False, "drop_reason": "header"})))   # 출력 블록이 없으면 주석도 없다
    assert md == ('<!-- page 1 -->\n\n<a id="p1.b02"></a>\n첫 쪽 본문\n\n'
                  '<!-- page 3 -->\n\n<a id="p3.b02"></a>\n셋째 쪽 본문\n')
    assert render_markdown(_doc()) == ""
    plain = _md(_page(1, ("text", "앞 쪽 끝")), _page(2, ("text", "새 문단")))   # 이어짐이 없으면 쪽 주석으로 가른다
    assert plain == '<!-- page 1 -->\n\n<a id="p1.b01"></a>\n앞 쪽 끝\n\n<!-- page 2 -->\n\n<a id="p2.b01"></a>\n새 문단\n'


def test_page_comment_uses_logical_page_no():
    md = _md(Page(page_no=7, sheet_no=2, size=(595, 842), blocks=[Block(id="p7.b01", type="text", bbox=(0, 0, 1, 1), text="가")]))
    assert md.startswith("<!-- page 7 -->\n\n")


def test_continued_paragraph_joined_across_page():
    md = _md(_page(1, ("text", "시스템은 민원 접수 처리를 돕고 일정 관리와"),
                   ("footnote", "※ 민원 접수: 민원을 온라인으로 받는 것")),       # 사이에 낀 앞 쪽 각주는 이어진 문단 뒤로
             _page(2, ("text", "문서관리 기능을\n하나로 묶어 제공한다.", {"continues_from": "p1.b01"}),
                   ("text", "다음 문단.")))
    assert md == ('<!-- page 1 -->\n\n'
                  '<a id="p1.b01"></a>\n<a id="p2.b01"></a>\n'
                  '시스템은 민원 접수 처리를 돕고 일정 관리와 문서관리 기능을\n하나로 묶어 제공한다.\n\n'
                  '<a id="p1.b02"></a>\n※ 민원 접수: 민원을 온라인으로 받는 것\n\n'
                  '<!-- page 2 -->\n\n<a id="p2.b02"></a>\n다음 문단.\n')


def test_not_joined_without_kept_origin():
    md = _md(_page(1, ("text", "앞 쪽 끝 문장", {"keep": False, "drop_reason": "footer"})),
             _page(2, ("text", "이어지는 글", {"continues_from": "p1.b01"}), ("text", "둘째")))
    assert md == '<!-- page 2 -->\n\n<a id="p2.b01"></a>\n이어지는 글\n\n<a id="p2.b02"></a>\n둘째\n'
    gone = _md(_page(1, ("text", "앞 문단")), _page(2, ("text", "뒤 문단", {"continues_from": "p9.b01"})))
    assert "<!-- page 2 -->\n\n<a id=\"p2.b01\"></a>\n뒤 문단" in gone      # 없는 블록을 가리키면 따로 낸다


def test_continued_table_without_page_comment_between():
    md = _md(_page(1, ("text", "표 앞 글"), ("table", None, {"html": _tbl(1)})),
             _page(2, ("text", "(계속)"), ("table", None, {"html": _tbl(2), "continues_from": "p1.b02"})),
             _page(3, ("table", None, {"html": _tbl(3), "continues_from": "p2.b02"}), ("text", "표 뒤 글")))
    assert md == ('<!-- page 1 -->\n\n<a id="p1.b01"></a>\n표 앞 글\n\n'
                  f'<a id="p1.b02"></a>\n{_tbl(1)}\n\n<a id="p2.b02"></a>\n{_tbl(2)}\n\n<a id="p3.b01"></a>\n{_tbl(3)}\n\n'
                  '<!-- page 2 -->\n\n<a id="p2.b01"></a>\n(계속)\n\n'
                  '<!-- page 3 -->\n\n<a id="p3.b02"></a>\n표 뒤 글\n')
    end = _md(_page(1, ("table", None, {"html": _tbl(1)})), _page(2, ("table", None, {"html": _tbl(2), "continues_from": "p1.b01"})))
    assert end.endswith(f'{_tbl(2)}\n\n<!-- page 2 -->\n')               # 이어진 조각만 있는 쪽도 주석은 하나
    mixed = _md(_page(1, ("text", "표 설명")), _page(2, ("table", None, {"html": _tbl(1), "continues_from": "p1.b01"})))
    assert _chunks(mixed)[1:] == ['<a id="p1.b01"></a>\n표 설명', f'<a id="p2.b01"></a>\n{_tbl(1)}', "<!-- page 2 -->"]


def test_anchor_per_kept_block_in_order_and_deterministic():
    doc = _doc(_page(1, ("paragraph_title", "개요", {"heading_level": 1}), ("text", "본문"),
                     ("header", "머리말", {"keep": False, "drop_reason": "header"}),
                     ("image", None, {"img_path": "imgs/p1.b04.png"}), ("table", None, {"html": TABLE}),
                     ("text", "   "), ("text", " \n ")),                     # 글이 공백뿐이면 출력 블록이 아니다(빈 앵커 없음)
               _page(2, ("display_formula", "x"), ("text", "끝")))
    md = render_markdown(doc)
    assert ANCHOR.findall(md) == [b.id for _, b in kept(doc)] == ["p1.b01", "p1.b02", "p1.b04", "p1.b05", "p2.b01", "p2.b02"]
    assert md == render_markdown(doc.model_copy(deep=True)) and md.endswith("\n") and not md.endswith("\n\n")
    assert "\n\n\n" not in md


def test_commonmark_reads_blocks_as_intended():
    """마크다운 도구가 실제로 이렇게 읽는지 — CommonMark(+GFM 표) 구현으로 확인."""
    markdown_it = pytest.importorskip("markdown_it")
    doc = _doc(_page(1, ("paragraph_title", "Ⅰ. 사업 개요", {"heading_level": 1}),
                     ("text", "\n".join([s for s, _ in ESCAPES] + PLAIN)),
                     ("text", "- 세부 내용", {"list_level": 3}), ("text", "○ 세부", {"list_level": 2}),
                     ("table", None, {"html": TABLE}), ("image", None, {"img_path": "imgs/p1.b06.png", "figure_text": "- 총괄"}),
                     ("display_formula", "E = mc^2"), ("text", "일정 관리와")),
               _page(2, ("text", "문서관리 기능", {"continues_from": "p1.b08"}), ("paragraph_title", "추진 일정")))
    html = markdown_it.MarkdownIt("commonmark").enable("table").render(render_markdown(doc))
    tags = re.findall(r"<(h\d|ol|ul|blockquote|hr|pre|table|thead)\b", html)
    assert tags == ["h2", "table", "pre"]                                  # 글 줄이 목록·제목·인용·구분선·표로 읽히지 않는다
    assert re.findall(r'<a id="([^"]+)"></a>', html) == [b.id for _, b in kept(doc)]
    assert html.count("<!-- page") == 2 and "<strong>추진 일정</strong>" in html
    for s in ["# 참고", "1. 관련", "- 세부 내용", "&lt;Table 1&gt; 현황", "---|---", "2. 추진 일정", "일정 관리와 문서관리 기능"]:
        assert s in html, s


def test_unparsed_table_text_escaped():
    """칸을 못 읽어 html 자리에 평문이 온 표(PaddleX 가 VLM 원문을 그대로 둔다)는 글 블록처럼 이스케이프하고 빈 줄을 뺀다 —
    목록·'#' 제목·블록 경계로 읽히지 않게. 크롭 링크는 그대로 뒤에."""
    plain = "1. 과제 관리 대장 보완\n\n- 보안 접근 통제 점검\n# 비고"
    doc = _doc(_page(1, ("table", None, {"html": plain, "route": "image", "route_reason": "failed",
                                         "img_path": "imgs/p1.b01.png"})))
    md = render_markdown(doc)
    assert _chunks(md)[1:] == ['<a id="p1.b01"></a>\n1\\. 과제 관리 대장 보완\n\\- 보안 접근 통제 점검\n\\# 비고',
                               "![table](imgs/p1.b01.png)"]
    markdown_it = pytest.importorskip("markdown_it")
    html = markdown_it.MarkdownIt("commonmark").render(md)
    assert not re.findall(r"<(h\d|ol|ul)\b", html)


def test_continued_fragment_in_other_section_stays_in_place():
    """이어진 조각의 섹션 경로가 앞 조각과 다르면(사이에 '□ …(계속)' 같은 새 제목) 끌어오지 않는다 — 마크다운 자리가 정본
    섹션 경로와 같다. 경로가 같으면 끌어온다."""
    s1, s2 = ["1. 사업비 집행", "□ 사업비 집행 현황"], ["1. 사업비 집행", "□ 사업비 집행 현황(계속)"]

    def md(p2):
        return _md(_page(1, ("paragraph_title", "□ 사업비 집행 현황", {"heading_level": 2, "section_path": s1}),
                         ("table", None, {"html": _tbl(1), "section_path": s1})),
                   _page(2, ("paragraph_title", "□ 사업비 집행 현황(계속)", {"heading_level": 2, "section_path": p2}),
                         ("table", None, {"html": _tbl(2), "continues_from": "p1.b02", "section_path": p2}),
                         ("text", "집행률은 계획 대비 87% 수준이다.", {"section_path": p2})))
    assert md(s2) == ('<!-- page 1 -->\n\n<a id="p1.b01"></a>\n### □ 사업비 집행 현황\n\n'
                      f'<a id="p1.b02"></a>\n{_tbl(1)}\n\n'
                      '<!-- page 2 -->\n\n<a id="p2.b01"></a>\n### □ 사업비 집행 현황(계속)\n\n'
                      f'<a id="p2.b02"></a>\n{_tbl(2)}\n\n<a id="p2.b03"></a>\n집행률은 계획 대비 87% 수준이다.\n')
    assert ANCHOR.findall(md(s1)) == ["p1.b01", "p1.b02", "p2.b02", "p2.b01", "p2.b03"]


# ---- RAG 적재용 마크다운(D21) ----

def test_rag_markdown_same_words_as_document_md_for_text():
    """글·제목·항목·수식·쪽 넘김 문단만 든 문서는 document.md 와 같은 글이다 — 앵커가 빠지고(쪽 주석은 둔다), 항목을 들여쓰지 않고,
    블록 첫 줄은 새 문단 첫 줄로 막는다(앵커가 없으면 첫 줄이 새 문단을 연다: 어느 번호든 목록, 4칸 들여쓰기는 코드, '[…]:' 는
    링크 정의)."""
    doc = _doc(_page(1, ("doc_title", "결과 보고서", {"heading_level": 0}), ("paragraph_title", "1. 추진 배경", {"heading_level": 1}),
                     ("text", "□ 추진 배경", {"list_level": 1}), ("text", "○ 시스템 노후화로\n잦은 장애", {"list_level": 2}),
                     ("text", "\n".join([s for s, _ in ESCAPES] + PLAIN)), ("display_formula", "E = mc^2"),
                     ("text", "시스템은 민원 접수 처리를 돕고"), ("header", "머리말", {"keep": False, "drop_reason": "header"})),
               _page(2, ("text", "문서관리 기능을 제공한다.", {"continues_from": "p1.b07"}), ("paragraph_title", "주요 내용"),
                     ("text", "- 세부 내용", {"list_level": 3}), ("text", "2. 추진 일정"), ("text", "[표 1]: 현황")))
    rag = render_rag_markdown(doc)

    def plain(md):                                 # 앵커·줄 머리 공백·이스케이프를 떼면 같은 글
        return re.sub(r"\\(?=[^\w\s])", "", re.sub(r"(?m)^ +", "", re.sub(r'<a id="[^"]+"></a>\n', "", md)))

    assert plain(rag) == plain(render_markdown(doc))
    assert "<a id=" not in rag and rag.count("<!-- page") == 2 and "○ 시스템 노후화로\n잦은 장애" in rag   # 들여쓰지 않는다
    assert _chunks(rag)[-3:] == ["\\- 세부 내용", "2\\. 추진 일정", "\\[표 1]: 현황"]
    markdown_it = pytest.importorskip("markdown_it")
    html = markdown_it.MarkdownIt("commonmark").enable("table").render(rag)
    assert re.findall(r"<(h\d|ol|ul|blockquote|hr|pre|table)\b", html) == ["h1", "h2"]   # 글 줄이 목록·코드·표로 읽히지 않는다
    assert "[표 1]: 현황" in html and render_rag_markdown(_doc()) == ""


def test_rag_markdown_captions_and_figures_in_one_block():
    """표·그림 제목은 제자리에서 빠져 표·그림 바로 앞에 붙고, 그림 하나(제목·링크·설명·그림 속 글자)는 빈 줄 없는 한 덩이다.
    링크 이름은 한국어, 그림 속 글자는 울타리 없이 '그림 속 글자:' 줄로(이어지는 줄은 문단 이어짐 줄처럼 이스케이프)."""
    doc = _doc(_page(1, ("figure_title", "<표 1> 사업비 현황"), ("text", "(단위: 천원)"),
                     ("table", None, {"html": TABLE, "caption_id": "p1.b01", "unit": "천원", "header_rows": 1,
                                      "route": "html", "img_path": "imgs/p1.b03.png"}),
                     ("chart", None, {"img_path": "imgs/p1.b04.png", "figure_text": "기온(℃)\n- 23.1", "route": "image",
                                      "caption_id": "p1.b05", "figure_desc": "남서쪽이 23.1℃로 가장 높다."}),
                     ("figure_title", "[그림 1] 평균기온 분포"),
                     ("image", None, {"img_path": "imgs/p1.b06.png"})))
    rag = render_rag_markdown(doc)
    assert _chunks(rag)[1:] == ["(단위: 천원)", "<표 1> 사업비 현황\n" + TABLE,
                                "[그림 1] 평균기온 분포\n![도표](imgs/p1.b04.png)\n그림 설명: 남서쪽이 23.1℃로 가장 높다.\n"
                                "그림 속 글자: 기온(℃)\n\\- 23.1",
                                "![그림](imgs/p1.b06.png)"]
    assert rag.count("사업비 현황") == 1 and rag.count("평균기온 분포") == 1
    markdown_it = pytest.importorskip("markdown_it")
    html = markdown_it.MarkdownIt("commonmark").enable("table").render(rag)
    assert re.findall(r"<(h\d|ol|ul|blockquote|hr|pre|table)\b", html) == ["table"]   # 그림 속 글자 줄이 목록·코드가 되지 않는다
    failed = render_rag_markdown(_doc(_page(1, ("table", None, {"html": TABLE, "route": "image", "route_reason": "photo_in_cell",
                                                                "img_path": "imgs/p1.b01.png", "figure_text": "사진 속 글자"}))))
    assert _chunks(failed)[1:] == ["![표](imgs/p1.b01.png)\n그림 속 글자: 사진 속 글자\n" + TABLE]   # 이미지 갈래 표는 나누지 않는다
    assert "<img" in markdown_it.MarkdownIt("commonmark").render(failed)   # 표 HTML 을 뒤에 — HTML 블록이 링크 줄을 삼키지 않게


def test_rag_markdown_long_table_rows_repeat_header_and_title():
    """긴 표는 행 묶음으로 — 묶음마다 머리행, 뒤 묶음 앞에는 '제목 (계속)'과 단위."""
    def row(i):
        return f"<tr><td>지점{i}</td><td>{i}.5</td></tr>"

    head = "<table><tr><th>지점</th><th>기온</th></tr>"
    html = head + "".join(map(row, range(1, 7))) + "</table>"
    doc = _doc(_page(1, ("figure_title", "<표 2> 지점별 기온"),
                     ("table", None, {"html": html, "caption_id": "p1.b01", "unit": "℃", "header_rows": 1})))
    assert _chunks(render_rag_markdown(doc))[1:] == ["<표 2> 지점별 기온\n" + html]     # 한도 안이면 그대로
    small = len(head) + 2 * len(row(1)) + len("</table>")
    assert _chunks(render_rag_markdown(doc, small))[1:] == [
        "<표 2> 지점별 기온\n" + head + row(1) + row(2) + "</table>",
        "<표 2> 지점별 기온 (계속) (단위: ℃)\n" + head + row(3) + row(4) + "</table>",
        "<표 2> 지점별 기온 (계속) (단위: ℃)\n" + head + row(5) + row(6) + "</table>"]


def test_rag_markdown_continued_table_joined_and_lone_piece_gets_header():
    """쪽 넘김 표는 한 표로 잇고(뒤 조각의 되풀이 머리행은 뺀다), 섹션이 달라 따로 나가는 뒤 조각은 첫 조각의 머리행과
    '제목 (계속)' 을 받는다."""
    a = "<table><tr><th>지점</th><th>기온</th></tr><tr><td>청주</td><td>26.0</td></tr></table>"
    b = "<table><tr><th>지점</th><th>기온</th></tr><tr><td>충주</td><td>25.1</td></tr></table>"
    joined = "<table><tr><th>지점</th><th>기온</th></tr><tr><td>청주</td><td>26.0</td></tr><tr><td>충주</td><td>25.1</td></tr></table>"
    md = render_rag_markdown(_doc(_page(1, ("figure_title", "<표 3> 기온"),
                                        ("table", None, {"html": a, "caption_id": "p1.b01", "header_rows": 1})),
                                  _page(2, ("table", None, {"html": b, "continues_from": "p1.b02", "header_rows": 1}),
                                        ("text", "표 뒤 글"))))
    assert md == "<!-- page 1 -->\n\n<표 3> 기온\n" + joined + "\n\n<!-- page 2 -->\n\n표 뒤 글\n"
    s1, s2 = ["1. 기온"], ["1. 기온(계속)"]
    lone = render_rag_markdown(_doc(
        _page(1, ("figure_title", "<표 3> 기온", {"section_path": s1}),
              ("table", None, {"html": a, "caption_id": "p1.b01", "header_rows": 1, "section_path": s1})),
        _page(2, ("paragraph_title", "1. 기온(계속)", {"heading_level": 1, "section_path": s2}),
              ("table", None, {"html": b, "continues_from": "p1.b02", "header_rows": 1, "section_path": s2}))))
    assert _chunks(lone)[1:] == ["<표 3> 기온\n" + a, "<!-- page 2 -->", "## 1. 기온(계속)", "<표 3> 기온 (계속)\n" + b]


def test_rag_markdown_continued_table_header_from_repeated_rows():
    """첫 조각의 머리행을 못 찾은 쪽 넘김 표(연도 머리글이라 header_rows 없음)도, 뒤 조각이 되풀이한 행(merge_tables 가 단 뒤 조각의
    header_rows)을 머리행으로 받는다 — 따로 나가는 뒤 조각과 긴 표의 뒤 묶음 모두."""
    top = "<tr><td>구분</td><td>2024</td><td>2025</td></tr>"
    a = f"<table>{top}<tr><td>청주</td><td>26.0</td><td>25.1</td></tr></table>"
    b = f"<table>{top}<tr><td>충주</td><td>25.1</td><td>24.0</td></tr></table>"
    s1, s2 = ["1. 기온"], ["1. 기온(계속)"]
    lone = render_rag_markdown(_doc(
        _page(1, ("figure_title", "<표 3> 기온", {"section_path": s1}),
              ("table", None, {"html": a, "caption_id": "p1.b01", "section_path": s1})),
        _page(2, ("paragraph_title", "1. 기온(계속)", {"heading_level": 1, "section_path": s2}),
              ("table", None, {"html": b, "continues_from": "p1.b02", "header_rows": 1, "section_path": s2}))))
    assert _chunks(lone)[-1] == "<표 3> 기온 (계속)\n" + b                 # 칸 이름 행이 남는다
    joined = render_rag_markdown(_doc(_page(1, ("table", None, {"html": a})),
                                      _page(2, ("table", None, {"html": b, "continues_from": "p1.b01", "header_rows": 1}))), len(a))
    assert _chunks(joined)[1:] == [a, b, "<!-- page 2 -->"]                # 뒤 묶음도 칸 이름 행부터


def test_rag_markdown_chain_with_image_piece():
    """쪽 넘김 사슬 가운데 이미지 갈래 조각이 끼어도 뒤 조각들은 '제목 (계속)'을 받고, 표 조각은 사슬 첫 조각의 머리행을 받는다."""
    a = "<table><tr><th>지점</th><th>기온</th></tr><tr><td>청주</td><td>26.0</td></tr></table>"
    c = "<table><tr><td>제천</td><td>24.2</td></tr></table>"
    md = render_rag_markdown(_doc(
        _page(1, ("figure_title", "<표 4> 기온"), ("table", None, {"html": a, "caption_id": "p1.b01", "header_rows": 1})),
        _page(2, ("table", None, {"html": TABLE, "continues_from": "p1.b02", "route": "image", "route_reason": "failed",
                                  "img_path": "imgs/p2.b01.png"})),
        _page(3, ("table", None, {"html": c, "continues_from": "p2.b01"}))))
    assert _chunks(md)[1:] == ["<표 4> 기온\n" + a,
                               "<표 4> 기온 (계속)\n![표](imgs/p2.b01.png)\n" + TABLE,
                               "<표 4> 기온 (계속)\n<table><tr><th>지점</th><th>기온</th></tr><tr><td>제천</td><td>24.2</td></tr></table>",
                               "<!-- page 2 -->", "<!-- page 3 -->"]


def test_rag_markdown_pipe_tables():
    """table_format 'pipe' — 칸을 읽은 표는 파이프 표로(제목·'(계속)'·단위 줄은 HTML 꼴과 같다), 이미지 갈래 표는 HTML 그대로."""
    def row(i):
        return f"<tr><td>지점{i}</td><td>{i}.5</td></tr>"

    html = "<table><tr><th>지점</th><th>기온</th></tr>" + "".join(map(row, range(1, 4))) + "</table>"
    doc = _doc(_page(1, ("figure_title", "<표 2> 지점별 기온"),
                     ("table", None, {"html": html, "caption_id": "p1.b01", "unit": "℃", "header_rows": 1}),
                     ("table", None, {"html": TABLE, "route": "image", "route_reason": "failed", "img_path": "imgs/p1.b03.png"})))
    head = "| 지점 | 기온 |\n|---|---|"
    rag = render_rag_markdown(doc, table_format="pipe")
    assert _chunks(rag)[1:] == ["<표 2> 지점별 기온\n" + head + "\n| 지점1 | 1.5 |\n| 지점2 | 2.5 |\n| 지점3 | 3.5 |",
                                "![표](imgs/p1.b03.png)\n" + TABLE]
    small = len(head) + 1 + len("| 지점1 | 1.5 |")
    assert _chunks(render_rag_markdown(doc, small, "pipe"))[1:4] == [
        "<표 2> 지점별 기온\n" + head + "\n| 지점1 | 1.5 |",
        "<표 2> 지점별 기온 (계속) (단위: ℃)\n" + head + "\n| 지점2 | 2.5 |",
        "<표 2> 지점별 기온 (계속) (단위: ℃)\n" + head + "\n| 지점3 | 3.5 |"]
    assert render_rag_markdown(doc) != rag                                             # 기본은 HTML 그대로
    markdown_it = pytest.importorskip("markdown_it")
    assert markdown_it.MarkdownIt("commonmark").enable("table").render(rag).count("<table>") == 2   # 파이프 표 + 이미지 갈래 HTML 표
