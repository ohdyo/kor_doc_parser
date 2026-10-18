"""실제 VLM — configs/env.yaml 의 방식대로(이 개발 서버는 native·CPU 로 쪽당 수 분, GPU 서버는 vLLM). pytest -m slow 로만 돈다."""
import pytest

from kdp.pipeline import parse_pdf
from tests.fixtures.synth import report_pdf


@pytest.mark.slow
def test_real_engine_one_page(tmp_path, korean_font, settings):
    s = settings.model_copy(deep=True)
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(report_pdf(korean_font, pages=1))
    doc = parse_pdf(pdf, tmp_path / "out", s)
    text = " ".join(b.text or "" for p in doc.pages for b in p.blocks if b.order)
    assert "추진 배경" in text
    dropped = {b.drop_reason for p in doc.pages for b in p.blocks if not b.keep}
    assert dropped & {"header", "page_number"}


@pytest.mark.slow
def test_m3_sample_text_absent_present(tmp_path, korean_font, settings):
    """M3 완료 기준(설계 12장) — 공문·간지·모아찍기·옆 스캔·빈 쪽이 섞인 문서에서 버릴 글자는 없고 본문 글자는 있다."""
    from tests.fixtures.synth import m3_sample_pdf

    s = settings.model_copy(deep=True)
    pdf = tmp_path / "m3.pdf"
    pdf.write_bytes(m3_sample_pdf(korean_font))
    doc = parse_pdf(pdf, tmp_path / "out", s)
    pages = {p.page_no: p for p in doc.pages}
    assert len(pages) == 8 and [p.sheet_no for p in doc.pages] == [1, 2, 3, 3, 3, 3, 4, 5]
    assert (pages[2].page_class, pages[2].keep) == ("divider", False) and (pages[8].page_class, pages[8].keep) == ("blank", False)
    assert all(pages[n].n_up for n in (3, 4, 5, 6)) and pages[7].rot90_k
    assert {k: doc.meta.get(k) for k in ("issuer", "title", "doc_no", "date")} == {
        "issuer": "한국정보공사", "title": "정보화 사업 협조 요청", "doc_no": "정보화추진단-1300", "date": "2026-03-05"}
    assert "phone" not in doc.meta                                       # 담당자 전화는 올리지 않는다(D15)

    out = " ".join(" ".join(filter(None, (b.text, b.html, b.figure_text))) for p in doc.pages if p.keep
                   for b in p.blocks if b.keep)
    md = doc.markdown or ""
    for junk in ("정보화추진단-1300", "공개구분", "전결", "단장", "2026년 정보화 사업 추진 계획", "스캔 문서 머리말",
                 "- 1 -", "- 4 -", "- 5 -", "- 6 -"):
        assert junk not in out and junk not in md, junk
    present = ["업무에 참고하시기 바랍니다", "공공 데이터 플랫폼 고도화와 검색 품질 개선", "4. 추진 배경과 목표",
               "종이 문서를 스캔한 쪽도 바로 세워 읽는다"]
    for body in present:
        assert body in out and body in md, body
    assert out.count("본문 12행") == 4                                   # 모아찍기 칸 넷 모두
    assert any("".join((b.section_path or [""])[0].split()).startswith("붙임1") for b in pages[3].blocks if b.keep)   # 간지 제목이 경로 뿌리(띄어쓰기는 모델 따라)
