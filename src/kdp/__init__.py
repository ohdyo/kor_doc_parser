"""kdp — 한국어 PDF 를 RAG 용 정형 데이터로 바꾸는 파서."""

__version__ = "0.1.0"


def parse_pdf(pdf_path, out_root=None, settings=None, engine=None):
    from .pipeline import parse_pdf as _parse

    return _parse(pdf_path, out_root, settings, engine)
