"""합성 PDF(reportlab). 한글 TrueType 글꼴을 PDF 에 내장해 어느 서버에서나 같게 그려진다."""
from __future__ import annotations

from io import BytesIO

import pypdfium2 as pdfium
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

FONT = "KoFont"
RED = (100, 700, 150, 750)       # 사용자 공간 l, b, r, t
BLACK = (400, 100, 440, 140)


def _font(path: str) -> str:
    if FONT not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(FONT, path))
    return FONT


def _rect(c, box, rgb):
    c.setFillColorRGB(*rgb)
    c.rect(box[0], box[1], box[2] - box[0], box[3] - box[1], stroke=0, fill=1)


def marker_pdf(font_path: str) -> bytes:
    """빨강·검정 사각형과 한글 한 줄 — 좌표 변환 시험용."""
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    _rect(c, RED, (1, 0, 0))
    _rect(c, BLACK, (0, 0, 0))
    c.setFont(_font(font_path), 16)
    c.drawString(72, 600, "가나다 ABC 2026")
    c.showPage()
    c.save()
    return buf.getvalue()


def report_pdf(font_path: str, pages: int = 2) -> bytes:
    """머리말·제목·본문·꼬리말 쪽번호가 있는 보고서."""
    font = _font(font_path)
    w, h = A4
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for i in range(pages):
        c.setFont(font, 9)
        c.drawString(72, h - 40, "2026년 정보화 사업 추진 계획")
        c.line(72, h - 46, w - 72, h - 46)
        c.setFont(font, 16)
        c.drawString(72, h - 100, f"{i + 1}. 추진 배경과 목표")
        c.setFont(font, 11)
        y = h - 140
        for k in range(12):
            c.drawString(72, y, f"본문 {k + 1}행: 공공기관 문서를 구조화해 검색 품질을 높이는 사업을 추진한다.")
            y -= 20
        c.setFont(font, 10)
        c.drawCentredString(w / 2, 40, f"- {i + 1} -")
        c.showPage()
    c.save()
    return buf.getvalue()


def with_page_box(data: bytes, rotation: int = 0, cropbox=None) -> bytes:
    """모든 쪽에 /Rotate 와 크롭박스를 건 사본."""
    doc = pdfium.PdfDocument(data)
    for i in range(len(doc)):
        page = doc[i]
        page.set_rotation(rotation)
        if cropbox:
            page.set_cropbox(*cropbox)
        page.close()
    out = BytesIO()
    doc.save(out)
    doc.close()
    return out.getvalue()
