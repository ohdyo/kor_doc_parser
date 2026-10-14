"""테스트용 한글 TrueType 글꼴.
reportlab 은 CFF(OTF) 글꼴을 넣지 못하고 일부 TTF 는 넣어도 글자가 깨진다 — 쓸 글자만 뽑아 TrueType 으로 다시 만든다."""
from __future__ import annotations

from pathlib import Path

from fontTools import subset
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTCollection, TTFont, newTable

TEXT = ("".join(chr(c) for c in range(0xAC00, 0xD7A4))        # 한글 음절 전체
        + "".join(chr(c) for c in range(0x20, 0x7F))
        + "·□○◦※①②③④⑤–—‘’“”「」『』〈〉《》")


def _korean_index(path: str) -> int:
    if not path.lower().endswith((".ttc", ".otc")):
        return 0
    for i, f in enumerate(TTCollection(path).fonts):
        if " KR" in (f["name"].getDebugName(4) or ""):
            return i
    return 0


def _cff_to_glyf(font: TTFont) -> None:
    """fontTools 의 otf2ttf 예제와 같은 변환(3차 곡선 → 2차)."""
    order = font.getGlyphOrder()
    glyphs = font.getGlyphSet()
    glyf = newTable("glyf")
    glyf.glyphOrder, glyf.glyphs = order, {}
    for name in order:
        pen = TTGlyphPen(glyphs)
        glyphs[name].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
        glyf.glyphs[name] = pen.glyph()
    font["loca"], font["glyf"] = newTable("loca"), glyf
    del font["CFF "]
    glyf.compile(font)
    for name, g in glyf.glyphs.items():
        if hasattr(g, "xMin"):
            font["hmtx"][name] = (font["hmtx"][name][0], g.xMin)
    maxp = newTable("maxp")
    maxp.tableVersion = 0x00010000
    for k in ("maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs",
              "maxStackElements", "maxSizeOfInstructions", "maxComponentElements"):
        setattr(maxp, k, 0)
    maxp.maxZones = 1
    font["maxp"] = maxp
    font["post"].formatType = 3.0
    font.sfntVersion = "\0\1\0\0"


def make_test_font(src: str, dst: Path) -> Path:
    font = TTFont(src, fontNumber=_korean_index(src))
    opt = subset.Options()
    opt.layout_features, opt.name_IDs = [], ["*"]
    opt.notdef_outline, opt.hinting, opt.desubroutinize = True, False, True
    opt.drop_tables += ["VORG", "vhea", "vmtx", "GSUB", "GPOS", "GDEF"]
    sub = subset.Subsetter(opt)
    sub.populate(text=TEXT)
    sub.subset(font)
    if "CFF " in font:
        _cff_to_glyf(font)
    font.save(dst)
    return dst
