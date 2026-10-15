"""합성 PDF(reportlab). 한글 TrueType 글꼴을 PDF 에 내장해 어느 서버에서나 같게 그려진다."""
from __future__ import annotations

from io import BytesIO

import numpy as np
import pypdfium2 as pdfium
from PIL import Image
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

FONT = "KoFont"
RED = (100, 700, 150, 750)       # 사용자 공간 l, b, r, t
BLACK = (400, 100, 440, 140)
MARK_RGB = [(1, 0, 0), (0, 0.6, 0), (0, 0, 1), (1, 0, 1)]   # 모아찍기 논리 쪽 n 의 색 표식 = MARK_RGB[(n - 1) % 4]


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


def _report_page(c, font: str, w: float, h: float, no: int) -> None:
    """보고서 한 쪽 — 머리말·제목·본문 12행·꼬리말 쪽번호."""
    c.setFont(font, 9)
    c.drawString(72, h - 40, "2026년 정보화 사업 추진 계획")
    c.line(72, h - 46, w - 72, h - 46)
    c.setFont(font, 16)
    c.drawString(72, h - 100, f"{no}. 추진 배경과 목표")
    c.setFont(font, 11)
    y = h - 140
    for k in range(12):
        c.drawString(72, y, f"본문 {k + 1}행: 공공기관 문서를 구조화해 검색 품질을 높이는 사업을 추진한다.")
        y -= 20
    c.setFont(font, 10)
    c.drawCentredString(w / 2, 40, f"- {no} -")


def report_pdf(font_path: str, pages: int = 2) -> bytes:
    """머리말·제목·본문·꼬리말 쪽번호가 있는 보고서."""
    font = _font(font_path)
    w, h = A4
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for i in range(pages):
        _report_page(c, font, w, h, i + 1)
        c.showPage()
    c.save()
    return buf.getvalue()


def cover_pdf(font_path: str) -> bytes:
    """표지 — 큰 글자 두 줄뿐(보이는 레이어 글자 20자 미만). 방향 모델이 이런 쪽을 거꾸로 확신하기도 한다."""
    font = _font(font_path)
    w, h = A4
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFont(font, 30)
    c.drawCentredString(w / 2, h * 0.6, "정보화 사업 결과 보고서")
    c.setFont(font, 16)
    c.drawCentredString(w / 2, h * 0.3, "2026. 7.")
    c.showPage()
    c.save()
    return buf.getvalue()


def _hand_pdf(objs: list[bytes]) -> bytes:
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def _stream(data: bytes) -> bytes:
    return b"<< /Length " + str(len(data)).encode() + b" >>\nstream\n" + data + b"\nendstream"


def nonbmp_pdf() -> bytes:
    """ToUnicode 로 'A'→U+1F600(이모지), 'B'→U+20000(확장 한자), 'C'→'C' 인 쪽 — PDFium 이 서로게이트 둘로 준다."""
    cmap = (b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap "
            b"/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def "
            b"/CMapName /Adobe-Identity-UCS def /CMapType 2 def "
            b"1 begincodespacerange <00> <FF> endcodespacerange "
            b"3 beginbfchar <41> <D83DDE00> <42> <D840DC00> <43> <0043> endbfchar "
            b"endcmap CMapName currentdict /CMap defineresource pop end end")
    return _hand_pdf([
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /ToUnicode 6 0 R >>",
        _stream(b"BT /F1 24 Tf 20 100 Td (ABC) Tj ET"),
        _stream(cmap),
    ])


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


def nup_pdf(font_path: str, cols: int, rows: int, pages: int) -> bytes:
    """모아찍기 — 보고서 쪽(오른쪽 위에 쪽마다 다른 색 표식)을 시트 한 장에 cols×rows 칸으로 줄여 왼쪽 위부터 행 우선으로 놓는다.
    2×2 는 A4 세로 시트에 세로 쪽 넷, 2×1 은 A4 가로 시트에 세로 쪽 둘, 1×2 는 A4 세로 시트에 가로 쪽 둘."""
    font = _font(font_path)
    sw, sh = landscape(A4) if (cols, rows) == (2, 1) else A4
    lw, lh = landscape(A4) if (cols, rows) == (1, 2) else A4
    cw, ch = sw / cols, sh / rows
    f = min(cw / lw, ch / lh)
    n = cols * rows
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(sw, sh))
    for i in range(pages):
        col, row = i % n % cols, i % n // cols
        c.saveState()
        c.translate(col * cw + (cw - lw * f) / 2, sh - (row + 1) * ch + (ch - lh * f) / 2)
        c.scale(f, f)
        _report_page(c, font, lw, lh, i + 1)
        _rect(c, (lw - 130, lh - 95, lw - 100, lh - 65), MARK_RGB[i % 4])
        c.restoreState()
        if i % n == n - 1 or i == pages - 1:
            c.showPage()
    c.save()
    return buf.getvalue()


def two_column_pdf(font_path: str, wide: bool = False, number: str = "center") -> bytes:
    """두 단 편집 보통 쪽 — 머리말·제목 없이 두 단 사이 빈 띠가 시트 끝까지 이어진다. 쪽번호는 하나(아래 가운데 또는 오른쪽)."""
    font = _font(font_path)
    w, h = landscape(A4) if wide else A4
    gap = 24
    col_w = (w - 2 * 72 - gap) / 2
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(w, h))
    c.setFont(font, 10)
    for col in range(2):
        for k in range(int((h - 144) // 16)):
            c.drawString(72 + col * (col_w + gap), h - 72 - k * 16, f"두 단 편집 본문 {k + 1}행입니다")
    if number == "center":
        c.drawCentredString(w / 2, 40, "- 3 -")
    else:
        c.drawRightString(w - 72, 40, "- 3 -")
    c.showPage()
    c.save()
    return buf.getvalue()


def scan_pdf(images) -> bytes:
    """글자 레이어 없이 그림만 있는 PDF(스캔) — images = [(BGR 배열, 쪽 크기 pt)]."""
    buf = BytesIO()
    c = canvas.Canvas(buf)
    for img, (w, h) in images:
        c.setPageSize((w, h))
        c.drawImage(ImageReader(Image.fromarray(np.ascontiguousarray(img[:, :, ::-1]))), 0, 0, w, h)
        c.showPage()
    c.save()
    return buf.getvalue()


def scanned(data: bytes, k: int = 0, scale: float = 2.0) -> bytes:
    """data 의 쪽마다 그림으로 그려 np.rot90(그림, k) 로 돌려 넣은 스캔 PDF — 옆·거꾸로 스캔한 쪽."""
    from kdp.io.pdf import open_pdf
    from kdp.io.render import render_page

    doc = open_pdf(data)
    pages = []
    for i in range(len(doc)):
        page = doc[i]
        img = np.ascontiguousarray(np.rot90(render_page(page, scale), k))
        page.close()
        pages.append((img, (img.shape[1] / scale, img.shape[0] / scale)))
    doc.close()
    return scan_pdf(pages)


def _letter_page(c, font: str, w: float, h: float) -> None:
    """공문 한 쪽 — 위 메타(수신·제목), 번호 본문, '끝.', 결재란 표, 아래 메타(시행·전화·공개구분)."""
    c.setFont(font, 18)
    c.drawCentredString(w / 2, h - 80, "한국정보공사")
    c.setFont(font, 11)
    for k, line in enumerate(["수신  각 부서장", "(경유)", "제목  정보화 사업 협조 요청"]):
        c.drawString(72, h - 130 - k * 20, line)
    body = ["1. 관련: 정보화추진단-1234(2026. 3. 2.)",
            "2. 올해 정보화 사업을 다음과 같이 추진하고자 하오니 업무에 참고하시기 바랍니다.",
            "  가. 사업 기간: 2026. 4. ~ 2026. 12.",
            "  나. 사업 내용: 공공 데이터 플랫폼 고도화와 검색 품질 개선",
            "  다. 협조 사항: 부서별 실무자 지정 및 자료 제출"]
    for k, line in enumerate(body):
        c.drawString(72 + (14 if line.startswith("  ") else 0), h - 220 - k * 22, line.strip())
    c.drawString(72, h - 350, "붙임  사업 추진 계획 1부.  끝.")
    x0, y0, cw, rh = 300, 200, 75, 24                       # 결재란 — 2행 3열
    for r, row in enumerate([["담당", "팀장", "단장"], ["○○○", "○○○", "전결 3/5"]]):
        for k, cell in enumerate(row):
            c.rect(x0 + k * cw, y0 - (r + 1) * rh, cw, rh)
            c.drawCentredString(x0 + k * cw + cw / 2, y0 - (r + 1) * rh + 8, cell)
    c.setFont(font, 9)
    c.drawString(72, 110, "시행  정보화추진단-1300 (2026. 3. 5.)        접수")
    c.drawString(72, 94, "우 04520 서울특별시 중구 세종대로 / 전화 02-000-0000 / 공개구분 공개")


def _scan_source_page(c, font: str, w: float, h: float) -> None:
    c.setFont(font, 9)
    c.drawString(72, h - 40, "스캔 문서 머리말")
    c.setFont(font, 16)
    c.drawString(72, h - 100, "5. 스캔 쪽 시험")
    c.setFont(font, 12)
    for k in range(8):
        c.drawString(72, h - 140 - k * 24, f"스캔 본문 {k + 1}행: 종이 문서를 스캔한 쪽도 바로 세워 읽는다.")
    c.setFont(font, 10)
    c.drawCentredString(w / 2, 40, "- 5 -")


def _one(draw, font_path: str) -> bytes:
    font = _font(font_path)
    w, h = A4
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    draw(c, font, w, h)
    c.showPage()
    c.save()
    return buf.getvalue()


def _divider(c, font, w, h):
    c.setFont(font, 28)
    c.drawCentredString(w / 2, h * 0.55, "붙임 1")
    c.setFont(font, 16)
    c.drawCentredString(w / 2, h * 0.55 - 40, "사업 추진 계획")


def _blank(c, font, w, h):
    c.setFont(font, 10)
    c.drawCentredString(w / 2, 40, "- 6 -")


def m3_sample_pdf(font_path: str) -> bytes:
    """M3 확인용 — 공문(메타·결재란), 간지 '붙임 1', 2×2 모아찍기 보고서 4쪽, 옆으로 스캔한 쪽, 쪽번호만 있는 빈 쪽.
    시트 5장 → 논리 페이지 8개."""
    parts = [_one(_letter_page, font_path), _one(_divider, font_path), nup_pdf(font_path, 2, 2, 4),
             scanned(_one(_scan_source_page, font_path), k=1), _one(_blank, font_path)]
    out = pdfium.PdfDocument.new()
    for data in parts:
        src = pdfium.PdfDocument(data)
        out.import_pages(src)
        src.close()
    buf = BytesIO()
    out.save(buf)
    out.close()
    return buf.getvalue()


def _top_number_page(c, font: str, w: float, h: float, no: int) -> None:
    """쪽번호가 위 가운데에 있는 보고서 쪽 — 머리말 없음, 제목·본문 12행."""
    c.setFont(font, 10)
    c.drawCentredString(w / 2, h - 40, f"- {no} -")
    c.setFont(font, 16)
    c.drawString(72, h - 100, f"{no}. 추진 배경과 목표")
    c.setFont(font, 11)
    for k in range(12):
        c.drawString(72, h - 140 - k * 20, f"본문 {k + 1}행: 공공기관 문서를 구조화해 검색 품질을 높이는 사업을 추진한다.")


def _outer_number_page(c, font: str, w: float, h: float, no: int) -> None:
    """보고서 쪽 — 쪽번호가 바깥쪽 아래(홀수 쪽 오른쪽, 짝수 쪽 왼쪽)."""
    _report_page(c, font, w, h, no)
    c.setFillColorRGB(1, 1, 1)
    c.rect(w / 2 - 40, 30, 80, 25, stroke=0, fill=1)                 # 가운데 쪽번호를 지운다
    c.setFillColorRGB(0, 0, 0)
    (c.drawRightString(w - 72, 40, f"- {no} -") if no % 2 else c.drawString(72, 40, f"- {no} -"))


def placed_pdf(font_path: str, sheets, cols: int, rows: int, sheet=A4, page=A4, number: str = "bottom") -> bytes:
    """모아찍기 배치를 마음대로 — sheets = 시트마다 칸(왼쪽 위부터 행 우선)에 놓을 인쇄 쪽번호 [[n, …], …].
    쪽 n 은 보고서 쪽(number: bottom 아래 가운데·top 위 가운데·outer 바깥쪽 아래 쪽번호)이고 오른쪽 위에 MARK_RGB[(n - 1) % 4] 표식.
    세로 우선·소책자·6·8쪽 모아찍기용."""
    font = _font(font_path)
    sw, sh = sheet
    lw, lh = page
    cw, ch = sw / cols, sh / rows
    f = min(cw / lw, ch / lh)
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(sw, sh))
    for nos in sheets:
        for i, no in enumerate(nos):
            c.saveState()
            c.translate(i % cols * cw + (cw - lw * f) / 2, sh - (i // cols + 1) * ch + (ch - lh * f) / 2)
            c.scale(f, f)
            {"bottom": _report_page, "top": _top_number_page, "outer": _outer_number_page}[number](c, font, lw, lh, no)
            _rect(c, (lw - 130, lh - 95, lw - 100, lh - 65), MARK_RGB[(no - 1) % 4])
            c.restoreState()
        c.showPage()
    c.save()
    return buf.getvalue()


def two_tables_pdf(font_path: str) -> bytes:
    """가로 A4 보통 쪽 — 같은 꼴 표 둘을 각 절반 가운데에, 쪽 제목·쪽번호 없이 마지막 행이 '합계 | 183'·'합계 | 195'."""
    font = _font(font_path)
    w, h = landscape(A4)
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(w, h))
    c.setFont(font, 11)
    for t, year in enumerate(["2025년", "2026년"]):
        x = t * w / 2 + (w / 2 - 300) / 2
        rows = [(year, "건수")] + [(f"사업 {k + 1}", str(10 + k + t)) for k in range(12)] + [("합계", str(183 + 12 * t))]
        for r, (a, b) in enumerate(rows):
            y = h - 90 - r * 32
            c.rect(x, y - 24, 200, 32)
            c.rect(x + 200, y - 24, 100, 32)
            c.drawString(x + 10, y - 14, a)
            c.drawRightString(x + 290, y - 14, b)
    c.showPage()
    c.save()
    return buf.getvalue()


def _table_page(c, font: str, w: float, h: float) -> None:
    """표 쪽 — 표 제목, 4열 22행 표(괘선), 단위 주석, 쪽번호."""
    c.setFont(font, 14)
    c.drawString(72, h - 80, "<표 3> 연도별 사업 예산 현황")
    c.setFont(font, 10)
    widths, y0 = [120, 110, 110, 110], h - 100
    rows = [["구분", "2024년", "2025년", "2026년"]] + [[f"사업 {k + 1}", f"{120 + k * 7:,}", f"{140 + k * 9:,}", f"{150 + k * 11:,}"]
                                                     for k in range(20)] + [["합계", "3,120", "3,560", "4,010"]]
    for r, row in enumerate(rows):
        x = 72
        for k, cell in enumerate(row):
            c.rect(x, y0 - (r + 1) * 24, widths[k], 24)
            c.drawCentredString(x + widths[k] / 2, y0 - (r + 1) * 24 + 8, cell)
            x += widths[k]
    c.drawString(72, y0 - (len(rows) + 1) * 24 - 10, "※ 단위: 백만 원, 2026년은 예산안 기준")
    c.drawCentredString(w / 2, 40, "- 7 -")


def table_page_pdf(font_path: str) -> bytes:
    return _one(_table_page, font_path)


def _title_page(c, font: str, w: float, h: float) -> None:
    """제목 쪽 — 왼쪽 위 '대외비', 가운데 큰 제목·부제, 아래 기관명. 방향 모델이 이 쪽을 거꾸로 확신한다(20표 중 16표 이상)."""
    c.setFont(font, 12)
    c.drawString(60, h - 60, "대외비")
    c.setFont(font, 24)
    c.drawCentredString(w / 2, h * 0.58, "공공 데이터 품질 관리 지침")
    c.setFont(font, 13)
    c.drawCentredString(w / 2, h * 0.52, "(2026년 개정판)")
    c.setFont(font, 15)
    c.drawCentredString(w / 2, h * 0.15, "행정정보공유센터")


def title_page_pdf(font_path: str) -> bytes:
    return _one(_title_page, font_path)


def body_pdf(font_path: str, lines: int) -> bytes:
    """본문 쪽 — 제목, 같은 꼴 문장 lines 행, 쪽번호(lines=33 이면 꽉 찬 쪽)."""
    def draw(c, font, w, h):
        c.setFont(font, 16)
        c.drawString(72, h - 80, "Ⅱ. 사업 추진 현황")
        c.setFont(font, 11)
        for k in range(lines):
            c.drawString(72, h - 120 - k * 20, f"{k + 1}행: 공공기관 문서를 구조화해 검색 품질을 높이는 사업을 추진한다. 세부 계획은 다음과 같다.")
        c.setFont(font, 10)
        c.drawCentredString(w / 2, 40, "- 3 -")
    return _one(draw, font_path)


def _one_on(draw, font_path: str, size) -> bytes:
    font = _font(font_path)
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=size)
    draw(c, font, *size)
    c.showPage()
    c.save()
    return buf.getvalue()


def _slide(c, font: str, w: float, h: float) -> None:
    """가로 슬라이드 — 큰 제목(밑줄)·글머리 다섯 줄·오른쪽 아래 슬라이드 번호(글 양 55 남짓)."""
    c.setFont(font, 30)
    c.drawString(50, h - 80, "Ⅲ. 추진 전략")
    c.line(50, h - 92, w - 50, h - 92)
    c.setFont(font, 20)
    for k, t in enumerate(["• 데이터 품질 관리 체계 고도화", "• 기관 간 연계 표준 마련", "• 검색 서비스 개선과 이용자 확대",
                           "• 예산 12억 원, 2026년 하반기 착수", "• 성과 지표: 만족도 85점 이상"]):
        c.drawString(70, h - 150 - k * 48, t)
    c.setFont(font, 12)
    c.drawRightString(w - 40, 25, "7")


def slide_pdf(font_path: str) -> bytes:
    return _one_on(_slide, font_path, landscape(A4))


def _toc(c, font: str, w: float, h: float) -> None:
    """목차 쪽 — 11항목, 점선 리더, 오른쪽 쪽번호(글 양 80 남짓)."""
    c.setFont(font, 20)
    c.drawCentredString(w / 2, h - 90, "목    차")
    c.setFont(font, 12)
    items = ["Ⅰ. 사업 개요", "  1. 추진 배경", "  2. 추진 목표", "Ⅱ. 추진 현황", "  1. 예산 현황", "  2. 일정",
             "Ⅲ. 추진 전략", "  1. 품질 관리", "  2. 연계 표준", "Ⅳ. 기대 효과", "붙임"]
    for k, t in enumerate(items):
        y = h - 150 - k * 34
        c.drawString(90, y, t.strip() if not t.startswith("  ") else "    " + t.strip())
        c.drawString(330, y, "·" * 28)
        c.drawRightString(w - 90, y, str(1 + k * 2))


def toc_pdf(font_path: str) -> bytes:
    return _one(_toc, font_path)


def _form(c, font: str, w: float, h: float) -> None:
    """신청서 서식 — 제목, 이름표 칸 10줄(괘선), 아래 문구(글 양 80 남짓)."""
    c.setFont(font, 18)
    c.drawCentredString(w / 2, h - 70, "정보 공개 청구서")
    c.setFont(font, 10)
    labels = ["청구인 성명", "생년월일", "주소", "전화번호", "청구 내용", "공개 방법", "수령 방법", "수수료 감면", "접수 번호", "처리 기관"]
    for r, lab in enumerate(labels):
        y = h - 110 - r * 50
        c.rect(60, y - 40, 120, 50)
        c.rect(180, y - 40, w - 240, 50)
        c.drawString(70, y - 18, lab)
        c.drawString(190, y - 18, "□ 해당  □ 해당 없음" if r % 3 == 0 else "")
    c.drawString(60, 80, "위와 같이 정보 공개를 청구합니다.        2026년    월    일")


def form_pdf(font_path: str) -> bytes:
    return _one(_form, font_path)


def _diagram_slide(c, font: str, w: float, h: float) -> None:
    """도식 슬라이드 — 제목과 상자 여섯 개의 짧은 이름표뿐(글 양 26 남짓)."""
    c.setFont(font, 26)
    c.drawString(50, h - 70, "추진 체계")
    c.line(50, h - 82, w - 50, h - 82)
    c.setFont(font, 14)
    for x, y, t in [(w / 2 - 90, h - 170, "정보화추진단"), (120, h - 300, "기획팀"), (w / 2 - 90, h - 300, "운영팀"),
                    (w - 300, h - 300, "품질관리팀"), (120, h - 430, "외부 자문단"), (w - 300, h - 430, "관계 기관")]:
        c.rect(x, y, 180, 50)
        c.drawCentredString(x + 90, y + 20, t)
    c.line(w / 2, h - 170, w / 2, h - 250)
    c.line(210, h - 250, w - 210, h - 250)
    c.setFont(font, 11)
    c.drawRightString(w - 40, 25, "6")


def diagram_slide_pdf(font_path: str) -> bytes:
    return _one_on(_diagram_slide, font_path, landscape(A4))


def _grid_pdf(font_path: str, cols: int, rows: int, draw, sheet, page=A4) -> bytes:
    """쪽 cols×rows 를 시트 한 장에 행 우선으로 — 쪽 n(1부터)은 draw(c, font, w, h, n) 와 오른쪽 위 MARK_RGB[(n - 1) % 4] 표식."""
    font = _font(font_path)
    sw, sh = sheet
    lw, lh = page
    cw, ch = sw / cols, sh / rows
    f = min(cw / lw, ch / lh)
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(sw, sh))
    for i in range(cols * rows):
        c.saveState()
        c.translate(i % cols * cw + (cw - lw * f) / 2, sh - (i // cols + 1) * ch + (ch - lh * f) / 2)
        c.scale(f, f)
        draw(c, font, lw, lh, i + 1)
        _rect(c, (lw - 130, lh - 95, lw - 100, lh - 65), MARK_RGB[i % 4])
        c.restoreState()
    c.showPage()
    c.save()
    return buf.getvalue()


def caption_nup_pdf(font_path: str, cols: int, rows: int) -> bytes:
    """가운데 정렬 짧은 표 제목('<표 n>' 10pt)이 있는 보고서 쪽 모아찍기 — 2×2 는 A4 세로, 2×1 은 A4 가로 시트.
    쪽: 본문 8행, 표 제목, 표(3열 6행), 본문 8행, 아래 가운데 쪽번호. 표 제목 줄은 쪽번호 줄과 자리·높이가 같다(쌓인 쪽이 아니다)."""
    def draw(c, font, w, h, no):
        c.setFont(font, 11)
        y = h - 80
        for k in range(8):
            c.drawString(72, y, f"본문 {k + 1}행: 연도별 사업 예산과 추진 체계는 다음과 같이 정리하였다.")
            y -= 20
        y -= 16
        c.setFont(font, 10)
        c.drawCentredString(w / 2, y, f"<표 {no}>")
        y -= 30
        for r in range(6):
            for k in range(3):
                c.rect(130 + k * 112, y - 18, 112, 24)
                c.drawCentredString(186 + k * 112, y - 10, f"{2024 + k}년 {r + 1}")
            y -= 24
        y -= 30
        c.setFont(font, 11)
        for k in range(8):
            c.drawString(72, y, f"본문 {k + 9}행: 집행 실적은 분기마다 점검하고 결과를 보고한다.")
            y -= 20
        c.setFont(font, 10)
        c.drawCentredString(w / 2, 40, f"- {no} -")
    return _grid_pdf(font_path, cols, rows, draw, landscape(A4) if (cols, rows) == (2, 1) else A4)


def body_nup_pdf(font_path: str, lines: int = 25, number: str = "bottom") -> bytes:
    """같은 문장 본문 lines 행 쪽의 2×2 모아찍기(A4 세로) — number: bottom 은 밑줄 머리말 + 아래 가운데 '- n -',
    top 은 머리말 없이 위 가운데 '- n -', right 는 머리말 없이 오른쪽 아래 'n'. 옆·거꾸로 스캔하면 방향 모델 표가 80% 에 못 미치는 꼴."""
    def draw(c, font, w, h, no):
        c.setFont(font, 10)
        if number == "top":
            c.drawCentredString(w / 2, h - 40, f"- {no} -")
        elif number == "bottom":
            c.setFont(font, 9)
            c.drawString(72, h - 40, "2026년 정보화 사업 추진 계획")
            c.line(72, h - 46, w - 72, h - 46)
        c.setFont(font, 16)
        c.drawString(72, h - 100, f"{no}. 추진 배경과 목표")
        c.setFont(font, 11)
        for k in range(lines):
            c.drawString(72, h - 140 - k * 20, f"본문 {k + 1}행: 공공기관 문서를 구조화해 검색 품질을 높이는 사업을 추진한다.")
        c.setFont(font, 10)
        if number == "bottom":
            c.drawCentredString(w / 2, 40, f"- {no} -")
        elif number == "right":
            c.drawRightString(w - 72, 40, str(no))
    return _grid_pdf(font_path, 2, 2, draw, A4)


def handout_pdf(font_path: str) -> bytes:
    """PPT 유인물 — 세로 A4 에 테두리 없는 가로 슬라이드 둘(1×2), 슬라이드 번호 오른쪽 아래."""
    return _grid_pdf(font_path, 1, 2, lambda c, font, w, h, no: _slide(c, font, w, h), A4, landscape(A4))
