"""PDF 열기·쪽 정보·텍스트 레이어(pypdfium2)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from .pdfium_lock import PDFIUM_LOCK


class PdfError(Exception):
    """열 수 없는 PDF(암호·손상)."""


@dataclass
class PageInfo:
    sheet_no: int
    size: tuple[float, float]                  # pt, /Rotate 반영
    rotation: int
    bbox: tuple[float, float, float, float]    # 사용자 공간 l,b,r,t (미디어박스∩크롭박스)
    text_chars: int                            # 공백을 뺀 텍스트 레이어 글자 수


@dataclass
class PdfInfo:
    doc_id: str
    path: str
    n_pages: int
    producer: str | None = None
    toc: list[dict] = field(default_factory=list)
    pages: list[PageInfo] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PdfInfo":
        d = dict(d)
        d["pages"] = [PageInfo(**p) for p in d.get("pages", [])]
        return cls(**d)


def open_pdf(data: bytes) -> pdfium.PdfDocument:
    with PDFIUM_LOCK:
        try:
            doc = pdfium.PdfDocument(data)
        except pdfium.PdfiumError as e:
            raise PdfError(str(e)) from e
        doc.init_forms()        # 안 하면 채운 입력란이 빈칸으로 그려진다
    return doc


def read_info(doc: pdfium.PdfDocument, doc_id: str, path: str) -> PdfInfo:
    with PDFIUM_LOCK:
        meta = doc.get_metadata_dict(skip_empty=True)
        toc = []
        for item in doc.get_toc():
            dest = item.get_dest()
            toc.append({
                "level": item.level,
                "title": item.get_title(),
                "page": dest.get_index() + 1 if dest is not None and dest.get_index() is not None else None,
            })
        pages = []
        for i in range(len(doc)):
            page = doc[i]
            try:
                tp = page.get_textpage()
                n = sum(1 for c in tp.get_text_range() if not c.isspace())
                tp.close()
                pages.append(PageInfo(
                    sheet_no=i + 1,
                    size=tuple(page.get_size()),
                    rotation=page.get_rotation(),
                    bbox=tuple(page.get_bbox()),
                    text_chars=n,
                ))
            finally:
                page.close()
    return PdfInfo(doc_id=doc_id, path=path, n_pages=len(pages),
                   producer=meta.get("Producer"), toc=toc, pages=pages)


def text_chars(page: pdfium.PdfPage) -> list[dict]:
    """글자마다 {c, box(l,b,r,t — 회전 전 사용자 공간), gen(PDFium 이 끼워 넣은 문자)}."""
    out = []
    with PDFIUM_LOCK:
        tp = page.get_textpage()
        try:
            for i in range(tp.count_chars()):
                code = pdfium_c.FPDFText_GetUnicode(tp, i)
                if code == 0:
                    continue
                out.append({
                    "c": chr(code),
                    "box": [round(v, 2) for v in tp.get_charbox(i)],
                    "gen": pdfium_c.FPDFText_IsGenerated(tp, i) == 1,
                })
        finally:
            tp.close()
    return out
