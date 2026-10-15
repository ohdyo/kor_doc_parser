"""정규화(기획서 4.3 규칙 8)."""
from kdp.assemble.normalize import norm_text, normalize
from kdp.schema import Block, Document, Page, ParserInfo, Source


def test_units_spaced_titles_dots():
    assert norm_text("감리 일정 : 2025 년 10 월 13 일 ~ 10 월 17 일, 5 명의 인원, 3 %") == \
        "감리 일정 : 2025년 10월 13일 ~ 10월 17일, 5명의 인원, 3%"
    assert norm_text("3 개선 사항, 2 일정, 1 원칙") == "3 개선 사항, 2 일정, 1 원칙"        # 단위가 아닌 낱말은 그대로
    assert norm_text("[목  차]") == "[목차]" and norm_text("1. 개 요") == "1. 개요" and norm_text("구 분") == "구분"
    assert norm_text("사업 관리") == "사업 관리" and norm_text("또 한 번 더 크게") == "또 한 번 더 크게"
    assert norm_text("효율성ㆍ안정성∙신뢰성") == "효율성·안정성·신뢰성"


def test_latex_and_literal_newline():
    assert norm_text("면적 10m$^{2}$, 온도 25$^{\\circ}$C, A$\\rightarrow$B") == "면적 10m², 온도 25°C, A→B"
    assert norm_text("$\\frac{1}{2}$ 은 그대로") == "$\\frac{1}{2}$ 은 그대로"
    assert norm_text("요구사항\\n보완 필요") == "요구사항 보완 필요"


def test_normalize_blocks_skips_formula():
    page = Page(page_no=1, sheet_no=1, size=(595, 842), blocks=[
        Block(id="p1.b01", type="table", bbox=(0, 0, 1, 1), html="<table><tr><td>개 선\\n현 황</td><td>2026 년</td></tr></table>"),
        Block(id="p1.b02", type="display_formula", bbox=(0, 0, 1, 1), text="$x^{2}$"),
        Block(id="p1.b03", type="text", bbox=(0, 0, 1, 1), text="그대로")])
    doc = normalize(Document(doc_id="sha256:ab", source=Source(paths=["a.pdf"], pages=1),
                             parser=ParserInfo(kdp_version="0", engine={}, engine_hash="e", assemble_hash="a", run_at="t"),
                             pages=[page]))
    b = doc.pages[0].blocks
    assert b[0].html == "<table><tr><td>개선현황</td><td>2026년</td></tr></table>"   # 한 글자씩 띄운 머리칸은 붙인다
    assert b[0].provenance == [{"rule": "normalize"}] and b[1].text == "$x^{2}$" and b[2].provenance is None


def test_latex_commands_paths_and_money_survive():
    assert norm_text("A$\\neq$B, $\\nu$ 값") == "A≠B, $\\nu$ 값"            # 모르는 명령은 그대로
    assert norm_text("D:\\nas\\data") == "D:\\nas\\data"
    assert norm_text("AWS 사용료 $1,200 이고 GCP 사용료 $900 이며") == "AWS 사용료 $1,200 이고 GCP 사용료 $900 이며"
    assert norm_text("문서의\\n보완 $x^{2}$", vlm=False) == "문서의\\n보완 $x^{2}$"   # PDF 글자에서 온 글은 손대지 않는다


def test_strict_vlm_blocks_and_short_math():
    assert norm_text("DB\\nServer 연동", strict=True) == "DB Server 연동"          # 모델 글 블록
    assert norm_text("DB\\nServer 연동") == "DB\\nServer 연동"                     # 표 칸(PDF 글자일 수 있다)
    assert norm_text("총 $N$개 노드, $10$ 개, $\\nu$ 값", strict=True) == "총 N개 노드, 10개, $\\nu$ 값"
    assert norm_text("$5 and $10 each", strict=True) == "$5 and $10 each"       # 금액은 그대로
