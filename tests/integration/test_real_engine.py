"""실제 VLM(native, CPU) — 쪽당 수 분. pytest -m slow 로만 돈다."""
import pytest

from kdp.pipeline import parse_pdf
from tests.fixtures.synth import report_pdf


@pytest.mark.slow
def test_real_engine_one_page(tmp_path, korean_font, settings):
    s = settings.model_copy(deep=True)
    s.env.vlm.backend = "native"
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(report_pdf(korean_font, pages=1))
    doc = parse_pdf(pdf, tmp_path / "out", s)
    text = " ".join(b.text or "" for p in doc.pages for b in p.blocks if b.order)
    assert "추진 배경" in text
    dropped = {b.drop_reason for p in doc.pages for b in p.blocks if not b.keep}
    assert dropped & {"header", "page_number"}
