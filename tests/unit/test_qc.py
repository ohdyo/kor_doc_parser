"""운영 QC — 빈 쪽 세기(빈 쪽으로 버렸는데 잉크가 있는 쪽은 세고, 공문 메타만 남은 쪽·쪽번호만 있는 진짜 빈 쪽은 세지 않는다).
잉크 값은 합성 PDF 실측(144dpi) — 쪽번호만 0.0, 간지 0.002, 본문 6줄 0.0147, 12줄 0.0375."""
from kdp.config import QCCfg
from kdp.engine.base import PageInput, RawPage
from kdp.qc.checks import run_checks
from kdp.schema import Block, Document, Page, ParserInfo, Source


def _raw(no, ink):
    return RawPage(input=PageInput(page_no=no, sheet_no=no, scale=2.0, sheet_size=(595, 842)),
                   width=1190, height=1684, ink_ratio=ink, blocks=[])


def test_empty_pages_counts_detection_failure_not_meta_only():
    lost = Page(page_no=1, sheet_no=1, size=(595, 842), page_class="blank", keep=False)          # 잉크는 있는데 블록이 없다
    meta = Page(page_no=2, sheet_no=2, size=(595, 842), blocks=[
        Block(id="p2.b01", type="doc_meta", bbox=(50, 50, 500, 100), text="문서번호 정보화-123 시행 2026. 3. 5.",
              keep=False, drop_reason="meta")])
    blank = Page(page_no=3, sheet_no=3, size=(595, 842), page_class="blank", keep=False, blocks=[
        Block(id="p3.b01", type="number", bbox=(290, 800, 305, 810), text="- 3 -", keep=False, drop_reason="page_number")])
    doc = Document(doc_id="sha256:ab", source=Source(paths=["a.pdf"], pages=3),
                   parser=ParserInfo(kdp_version="0", engine={}, engine_hash="e", assemble_hash="a", run_at="t"),
                   pages=[lost, meta, blank])
    qc = run_checks(doc, [_raw(1, 0.05), _raw(2, 0.05), _raw(3, 0.0)], QCCfg()).qc
    assert qc.empty_pages == 1 and "empty_pages" in qc.reasons


def test_blank_page_with_ink_but_only_page_number_is_counted():
    """잉크는 본문 쪽만큼 있는데 검출이 쪽번호만 찾아 빈 쪽으로 버린 쪽 — 버리더라도 빈 출력으로 세어 검수로(M2 는 남기고 검수)."""
    body = Page(page_no=1, sheet_no=1, size=(595, 842), blocks=[
        Block(id="p1.b01", type="text", bbox=(70, 100, 520, 300), text="공공기관 문서를 구조화해 검색 품질을 높이는 사업을 추진한다.")])
    lost = Page(page_no=2, sheet_no=2, size=(595, 842), page_class="blank", keep=False, blocks=[
        Block(id="p2.b01", type="number", bbox=(290, 800, 305, 810), text="- 2 -", keep=False, drop_reason="page_number")])
    doc = Document(doc_id="sha256:ab", source=Source(paths=["a.pdf"], pages=2),
                   parser=ParserInfo(kdp_version="0", engine={}, engine_hash="e", assemble_hash="a", run_at="t"),
                   pages=[body, lost])
    qc = run_checks(doc, [_raw(1, 0.0375), _raw(2, 0.07)], QCCfg()).qc
    assert (qc.status, qc.reasons, qc.empty_pages) == ("review", ["empty_pages"], 1)
    assert run_checks(doc, [_raw(1, 0.0375), _raw(2, 0.0)], QCCfg()).qc.status == "ok"   # 쪽번호만 있는 진짜 빈 쪽
    doc.pages[1].blocks.append(Block(id="p2.b02", type="text", bbox=(150, 400, 450, 420), text="이 페이지는 의도적으로 비워 두었습니다."))
    assert run_checks(doc, [_raw(1, 0.0375), _raw(2, 0.07)], QCCfg()).qc.empty_pages == 0   # 잉크는 '비워 둠' 문구의 것
