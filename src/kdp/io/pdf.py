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


def _optional(fn, default=None):
    """책갈피·Info 문자열은 참고용이다 — 깨진 UTF-16 이 있어도 문서를 실패시키지 않는다."""
    try:
        return fn()
    except UnicodeDecodeError:
        return default


def read_info(doc: pdfium.PdfDocument, doc_id: str, path: str) -> PdfInfo:
    with PDFIUM_LOCK:
        meta = _optional(lambda: doc.get_metadata_dict(skip_empty=True), {})
        toc = []
        for item in doc.get_toc():
            title = _optional(item.get_title)
            if title is None:
                continue
            dest = item.get_dest()
            index = dest.get_index() if dest is not None else None
            toc.append({"level": item.level, "title": title, "page": index + 1 if index is not None else None})
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


def merge_surrogates(codes: list[int]) -> list[tuple[int, int]]:
    """PDFium 은 BMP 밖 글자(이모지·수식 문자·확장 한자)를 UTF-16 서로게이트 둘로 준다 — (코드, 첫 단위 순번)으로 합친다.
    짝 없는 서로게이트는 U+FFFD 로 바꾼다(그대로 두면 UTF-8 로 쓸 수 없다)."""
    out, i = [], 0
    while i < len(codes):
        code, step = codes[i], 1
        if 0xD800 <= code < 0xDC00 and i + 1 < len(codes) and 0xDC00 <= codes[i + 1] < 0xE000:
            code, step = 0x10000 + ((code - 0xD800) << 10) + (codes[i + 1] - 0xDC00), 2
        elif 0xD800 <= code < 0xE000:
            code = 0xFFFD
        if code:
            out.append((code, i))
        i += step
    return out


_INVISIBLE = {pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE, pdfium_c.FPDF_TEXTRENDERMODE_CLIP}


def _invisible(tp, i: int) -> bool:
    """렌더 모드가 '안 그림'(스캔 PDF 의 투명 OCR 레이어 등)인 글자."""
    obj = pdfium_c.FPDFText_GetTextObject(tp, i)
    return bool(obj) and pdfium_c.FPDFTextObj_GetTextRenderMode(obj) in _INVISIBLE


def text_chars(page: pdfium.PdfPage) -> list[dict]:
    """글자마다 {c, box(l,b,r,t — 회전 전 사용자 공간), gen(PDFium 이 끼워 넣은 문자), inv(안 그리는 글자일 때만)}."""
    with PDFIUM_LOCK:
        tp = page.get_textpage()
        try:
            codes = [pdfium_c.FPDFText_GetUnicode(tp, i) for i in range(tp.count_chars())]
            out = []
            for code, i in merge_surrogates(codes):
                gen = pdfium_c.FPDFText_IsGenerated(tp, i) == 1
                ch = {"c": chr(code), "box": [round(v, 2) for v in tp.get_charbox(i)], "gen": gen}
                if not gen and _invisible(tp, i):
                    ch["inv"] = True
                out.append(ch)
            return out
        finally:
            tp.close()
