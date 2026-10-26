"""비교군 시험 — 같은 PDF 를 비교 파서로 돌리고, 정답(좌표·라벨)과 텍스트 레이어로 같은 지표를 잰다(기획서 8.1 일부).

    python eval/compare.py test2.pdf --golden eval/golden/test2_pdf.yaml

비교군 출력은 out/<문서 해시>/compare/ 에 두고 다시 쓴다(--rerun 이면 다시 돌린다). 보고에는 개수·비율만 낸다.
"""
from __future__ import annotations

import argparse
import html as htmllib
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
import yaml

from kdp.config import apply_env, load_settings
from kdp.convert import FIGURE_LABELS
from kdp.io.ids import doc_dirname, doc_id_of
from kdp.io.pdf import text_chars
from kdp.io.render import user_box_to_display
from kdp.io.write import read_json, write_json

RUNS = ("textlayer", "docling", "paddle_default", "kdp")
NAMES = {"textlayer": "텍스트 레이어", "docling": "docling", "paddle_default": "PaddleOCR-VL 기본", "kdp": "kdp"}
MIN_SEG = 4                      # 이보다 짧은 레이어 조각은 시험하지 않는다
COVER = 0.5                      # 정답 상자의 이만큼 이상을 덮는 블록도 겹친 것으로 본다(제목·다음 줄을 한 블록으로 묶으면 IoU 가 낮다)
PART = 0.2                       # 검출기가 한 항목을 여러 블록으로 나눴을 때 조각 하나가 덮어야 하는 몫
TITLE_TYPES = {"doc_title", "paragraph_title"}             # PP-DocLayoutV3 제목 라벨(kdp 블록 type 도 같다)
_TAG = re.compile(r"<[^>]+>")
_SKIP = re.compile(r"[\s|#*_`\-–•·ㆍᆞ▪▫◦○●■□◆◇▶►]+")    # 공백·마크업·글머리표(파서마다 따로 떼어 둔다)
_DOTS = re.compile(r"\.{2,}")                              # 목차 점선
_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"', "„": '"', "‟": '"'})   # 표기 차이
_LETTER = re.compile(r"[^\W\d_]")                          # 한글·영문·한자 같은 글자(숫자·기호가 아닌 것)


def norm(s: str) -> str:
    return _DOTS.sub("", _SKIP.sub("", unicodedata.normalize("NFKC", s or "").translate(_QUOTES)))


def strip_html(s: str) -> str:
    return htmllib.unescape(_TAG.sub(" ", s))


def html_grid(s: str) -> list[int] | None:
    """표 HTML 의 행·열 수 — rowspan·colspan 을 펼친 격자 기준."""
    occ = set()
    rows = re.split(r"<tr[\s>]", s, flags=re.I)[1:]
    for r, row in enumerate(rows):
        c = 0
        for attrs in re.findall(r"<t[dh]([^>]*)>", row, flags=re.I):
            rs = int((re.search(r'rowspan="?(\d+)', attrs) or [0, 1])[1])
            cs = int((re.search(r'colspan="?(\d+)', attrs) or [0, 1])[1])
            while (r, c) in occ:
                c += 1
            occ.update((r + i, c + j) for i in range(rs) for j in range(cs))
            c += cs
    return [len(rows), max(c for _, c in occ) + 1] if occ else None


def th_rows(s: str) -> int:
    """앞에서부터 모든 칸이 <th> 인 행 수."""
    n = 0
    for row in re.split(r"<tr[\s>]", s, flags=re.I)[1:]:
        cells = re.findall(r"<t([dh])[\s>]", row, flags=re.I)
        if not cells or any(c.lower() != "h" for c in cells):
            break
        n += 1
    return n


def _blk(kind, bbox, body, text, label, html=None, heading=False, heading_level=None, list_level=None) -> dict:
    """heading = 제목 역할. heading_level·list_level 은 1 이 가장 바깥 수준(문서 제목은 0, 모르면 None)."""
    return {"kind": kind, "bbox": [round(v, 1) for v in bbox] if bbox else None, "body": bool(body), "label": label,
            "text": strip_html(html) if html else (text or ""), "grid": html_grid(html) if html else None,
            "header_rows": th_rows(html) if html else None,
            "heading": heading, "heading_level": heading_level, "list_level": list_level}


# ---- 비교군 실행 ----

def run_textlayer(pdf: Path, settings) -> dict:
    """모델 없음 — PDF 텍스트 레이어를 쪽마다 그대로."""
    t = time.perf_counter()
    doc = pdfium.PdfDocument(str(pdf))
    pages = {}
    for i in range(len(doc)):
        w, h = doc[i].get_size()
        pages[i + 1] = [_blk("text", [0, 0, w, h], True, "".join(c["c"] for c in text_chars(doc[i])), "page")]
    doc.close()
    return {"seconds": time.perf_counter() - t, "pages": pages}


def _docling_struct(item, doc) -> dict:
    """docling 의 제목·목록 — 문서 제목(TitleItem, 마크다운 '#')은 0, 섹션 제목은 SectionHeaderItem.level(기본 파이프라인은
    수준 추정이 꺼져 모두 1, 마크다운 '#' 은 level+1 개). 목록 항목 깊이는 감싼 ListGroup 수(PDF 파이프라인은 중첩이 없어 1)."""
    from docling_core.types.doc import ListGroup, ListItem, SectionHeaderItem, TitleItem

    if isinstance(item, TitleItem | SectionHeaderItem):
        return {"heading": True, "heading_level": item.level if isinstance(item, SectionHeaderItem) else 0}
    depth, ref = 0, item.parent if isinstance(item, ListItem) else None
    while ref is not None:
        node = ref.resolve(doc)
        depth += isinstance(node, ListGroup)
        ref = node.parent
    return {"list_level": depth or None}


def run_docling(pdf: Path, settings) -> dict:
    """docling 기본 파이프라인, OCR 끔(기획서 A.3 과 같은 설정). furniture 층은 본문이 아니다. 기본 마크다운도 남긴다(RAG 하류
    평가 eval/rag_eval.py — D21)."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import ContentLayer, PictureItem, TableItem

    t = time.perf_counter()
    opts = PdfPipelineOptions()
    opts.do_ocr = False
    doc = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}).convert(pdf).document
    pages = {}
    for item, _ in doc.iterate_items(included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}):
        if not item.prov:
            continue
        prov = item.prov[0]
        box = list(prov.bbox.to_top_left_origin(doc.pages[prov.page_no].size.height).as_tuple())
        body = item.content_layer == ContentLayer.BODY
        if isinstance(item, TableItem | PictureItem):
            if isinstance(item, TableItem):
                b = _blk("table", box, body, " ".join(c.text for c in item.data.table_cells), item.label.value)
                b["grid"] = [item.data.num_rows, item.data.num_cols]
                b["header_rows"] = next((i for i, row in enumerate(item.data.grid)
                                         if not row or not all(c.column_header for c in row)), len(item.data.grid))
            else:
                b = _blk("figure", box, body, "", item.label.value)
            if item.captions and (cap := item.captions[0].resolve(doc)).prov:
                cp = cap.prov[0]
                b["caption_bbox"] = list(cp.bbox.to_top_left_origin(doc.pages[cp.page_no].size.height).as_tuple())
        else:
            b = _blk("text", box, body, getattr(item, "text", ""), item.label.value, **_docling_struct(item, doc))
        pages.setdefault(prov.page_no, []).append(b)
    return {"seconds": time.perf_counter() - t, "pages": pages, "markdown": doc.export_to_markdown()}


def paddle_title_level(label: str, content: str) -> int | None:
    """설치본 기본 마크다운의 제목 수준. predict() 는 restructure_pages(relevel_titles)를 부르지 않아 paragraph_title 은
    제목 안 '.' 개수로 '##'·'###'… 이 되고(paddlex format_para_title → format_title), doc_title 은 '#' 하나다.
    '#' 개수 − 1 을 돌려준다 — docling 처럼 섹션 제목이 1, 문서 제목이 0. 제목 라벨이 아니면 None."""
    if label not in TITLE_TYPES:
        return None
    if label == "doc_title":
        return 0
    from paddlex.inference.common.result.converter.markdown_format_funcs import format_para_title

    md = format_para_title(SimpleNamespace(content=content))
    return len(md) - len(md.lstrip("#")) - 1


def run_paddle_default(pdf: Path, settings) -> dict:
    """PaddleOCR-VL 기본 설정 그대로 — 우리 설정 사본·훅·규칙 없이(장치·VLM 서버만 env.yaml). 본문 = 기본 마크다운에 들어가는 블록.
    쪽마다 기본 마크다운(markdown_texts)도 남긴다(RAG 하류 평가 eval/rag_eval.py — D21)."""
    apply_env(settings.env)
    from paddleocr import PaddleOCRVL

    env = settings.env
    kw = {"pipeline_version": settings.parse.paddle_vl.pipeline_version, "device": env.device}
    if env.vlm.backend != "native":
        kw.update(vl_rec_backend=env.vlm.backend, vl_rec_server_url=env.vlm.server_url,
                  vl_rec_api_model_name=env.vlm.api_model_name, vl_rec_api_key=env.vlm.api_key)
    t = time.perf_counter()
    ocr = PaddleOCRVL(**kw)
    ignore = set(ocr._merged_paddlex_config.get("markdown_ignore_labels") or [])
    results = list(ocr.predict(str(pdf)))
    seconds = time.perf_counter() - t
    doc = pdfium.PdfDocument(str(pdf))
    pages = {}
    for i, r in enumerate(results):
        res = r.json.get("res", r.json)
        s = res["width"] / doc[i].get_size()[0]
        blocks = []
        for b in res["parsing_res_list"]:
            label, content = b["block_label"], b.get("block_content") or ""
            box = [v / s for v in b["block_bbox"]]
            kind = "table" if label == "table" else "figure" if label in FIGURE_LABELS else "text"
            lv = paddle_title_level(label, content)
            blocks.append(_blk(kind, box, label not in ignore, content, label, html=content if kind == "table" else None,
                               heading=lv is not None, heading_level=lv))
        pages[i + 1] = blocks
    doc.close()
    return {"seconds": seconds, "pages": pages, "markdown": [r.markdown.get("markdown_texts", "") for r in results]}


def load_kdp(doc_dir: Path) -> dict:
    d = read_json(doc_dir / "document.json")
    boxes = {b["id"]: b["bbox"] for p in d["pages"] for b in p["blocks"]}
    pages = {}
    for p in d["pages"]:
        pages[p["page_no"]] = []
        for b in p["blocks"]:
            blk = _blk("table" if b["type"] == "table" else "figure" if b["type"] in FIGURE_LABELS else "text", b["bbox"],
                       p["keep"] and b["keep"], b.get("text") or b.get("figure_text"), b["type"], html=b.get("html"),
                       heading=b["type"] in TITLE_TYPES or b.get("heading_level") is not None,   # 규칙이 제목으로 올린 글 블록 포함
                       heading_level=b.get("heading_level"), list_level=b.get("list_level"))
            if b.get("caption_id"):
                blk["caption_bbox"] = boxes[b["caption_id"]]
            if any(pv.get("rule") == "join_wrapped" for pv in b.get("provenance") or []):
                blk["joined"] = True
            if b["type"] in FIGURE_LABELS:                 # 그림 갈래·그림 설명(D19)
                blk["route"], blk["desc"] = b.get("route"), b.get("figure_desc") or ""
                pvs = [pv for pv in b.get("provenance") or [] if pv.get("rule") == "figure_desc"]
                blk["desc_dropped"] = sum(len(pv.get("dropped") or []) for pv in pvs)
                blk["desc_filtered"] = sum(len(pv.get(k) or []) for pv in pvs for k in ("copied", "unsure"))
            pages[p["page_no"]].append(blk)
    return {"seconds": read_json(doc_dir / "raw" / "engine.json").get("seconds"), "pages": pages, "meta": d.get("meta") or {}}


RUNNERS = {"textlayer": run_textlayer, "docling": run_docling, "paddle_default": run_paddle_default}


# ---- 정답·지표 ----

def segments(pdf: Path) -> dict[int, list[tuple[str, list]]]:
    """쪽마다 텍스트 레이어 조각 — 레이어에 적힌 순서대로 잇다가 줄이 바뀌거나, 뒤로 가거나, 넓게 떨어지거나,
    세로 테두리선(표 칸 경계)을 넘으면 끊는다."""
    doc = pdfium.PdfDocument(str(pdf))
    out = {}
    for i in range(len(doc)):
        pg = doc[i]
        bb, rot = pg.get_bbox(), pg.get_rotation()
        cs = [(c["c"], user_box_to_display(c["box"], bb, rot)) for c in text_chars(pg)
              if not c["gen"] and not c["c"].isspace()]
        rules = []
        for o in pg.get_objects(max_depth=4):
            if o.type == pdfium_c.FPDF_PAGEOBJ_PATH:
                l, t, r, b = user_box_to_display(o.get_bounds(), bb, rot)
                if r - l < 2.5 and b - t > 5:
                    rules.append(((l + r) / 2, t, b))
        out[i + 1] = _segments(cs, rules)
    doc.close()
    return out


def _segments(cs, rules=()) -> list[tuple[str, list]]:
    # 줄 높이는 보통 글자(한글·숫자·영문)로 잡는다 — 마침표·쉼표는 아래쪽, 따옴표는 위쪽에 붙어 있다
    segs, cur, top, bot, core = [], [], 0.0, 0.0, False
    for ch, b in cs:
        if cur:
            h, prev = b[3] - b[1], cur[-1][1]
            ov = min(bot, b[3]) - max(top, b[1])
            same_line = ov >= 0.5 * min(h, bot - top) if h > 0.5 else top <= (b[1] + b[3]) / 2 <= bot
            gap, line_h, cy = b[0] - prev[2], max(bot - top, h), (b[1] + b[3]) / 2
            ruled = any(prev[2] - 1 <= x <= b[0] + 1 and y0 <= cy <= y1 for x, y0, y1 in rules)
            if not same_line or not -line_h < gap <= 2 * line_h or ruled:
                segs.append(cur)
                cur = []
        if not cur:
            top, bot, core = b[1], b[3], ch.isalnum()
        elif ch.isalnum():
            top, bot = (min(top, b[1]), max(bot, b[3])) if core else (b[1], b[3])
            core = True
        cur.append((ch, b))
    if cur:
        segs.append(cur)
    return [(norm("".join(c for c, _ in s)),
             [min(b[0] for _, b in s), min(b[1] for _, b in s), max(b[2] for _, b in s), max(b[3] for _, b in s)])
            for s in segs]


def _best_dist(p: str, t: str) -> int:
    """p 를 t 의 어느 부분 문자열과 맞출 때 최소 편집 거리."""
    tt = np.frombuffer(t.encode("utf-32-le"), dtype=np.uint32)
    ar = np.arange(len(t) + 1)
    prev = np.zeros(len(t) + 1, dtype=np.int64)
    for i, ch in enumerate(p, 1):
        cur = np.empty_like(prev)
        cur[0] = i
        cur[1:] = np.minimum(prev[:-1] + (tt != ord(ch)), prev[1:] + 1)
        prev = np.minimum.accumulate(cur - ar) + ar
    return int(prev.min())


def present(seg: str, text: str) -> bool:
    """조각이 출력에 있나 — 10자당 1자까지 틀려도 있음(10자 미만은 정확히)."""
    if seg in text:
        return True
    k = len(seg) // 10
    return bool(k and text) and _best_dist(seg, text) <= k


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b) -> float:
    return _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def _iou(a, b) -> float:
    i = _inter(a, b)
    return i / (_area(a) + _area(b) - i) if i else 0.0


def _center_in(box, regions) -> bool:
    x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(r[0] <= x <= r[2] and r[1] <= y <= r[3] for r in regions)


def _over(pages: dict, g: dict) -> list[dict]:
    """정답 상자와 겹치는 출력 본문 블록(IoU 0.5 이상이거나 정답 상자의 COVER 이상을 덮음) — 가장 많이 겹치는 것부터."""
    cands = [b for b in pages.get(g["page"], []) if b["body"] and b["bbox"]
             and (_iou(b["bbox"], g["bbox"]) >= 0.5 or _inter(b["bbox"], g["bbox"]) >= COVER * _area(g["bbox"]))]
    if not cands:                                   # 한 항목이 여러 블록으로 나뉘었으면 함께 COVER 를 덮는 조각들(위에서부터)
        parts = [b for b in pages.get(g["page"], []) if b["body"] and b["bbox"]
                 and _inter(b["bbox"], g["bbox"]) >= PART * _area(g["bbox"])]
        if sum(_inter(b["bbox"], g["bbox"]) for b in parts) >= COVER * _area(g["bbox"]):
            return sorted(parts, key=lambda b: b["bbox"][1])
    return sorted(cands, key=lambda b: -_iou(b["bbox"], g["bbox"]))


def evaluate(segs: dict, golden: dict, run: dict) -> dict:
    pages = {int(k): v for k, v in run["pages"].items()}
    junk, tables, figs = golden.get("junk", []), golden.get("tables", []), golden.get("figures", [])
    body = {p: norm("".join(b["text"] for b in pages.get(p, []) if b["body"])) for p in golden["pages"]}
    text = {"normal": [0, 0]}                       # 본문 쪽: [출력에 있는 조각, 전체 조각]
    leak = {"cover": [0, 0], "toc": [0, 0]}         # 표지·목차 쪽: [본문에 남은 글자, 전체 글자]
    for p, cls in golden["pages"].items():
        regions = [j["bbox"] for j in junk if j["page"] == p]
        mine = [s for s, box in segs.get(p, []) if not _center_in(box, regions)]
        if cls == "normal":
            tests = [s for s in mine if len(s) >= MIN_SEG]
            text["normal"][0] += sum(present(s, body[p]) for s in tests)
            text["normal"][1] += len(tests)
        elif cls in leak:
            ref = Counter("".join(mine))
            leak[cls][0] += sum((ref & Counter(body[p])).values())
            leak[cls][1] += sum(ref.values())
    removed = {}
    for j in junk:
        kept = any(b["body"] and b["bbox"] and _inter(j["bbox"], b["bbox"]) >= 0.5 * _area(j["bbox"])
                   for b in pages.get(j["page"], []))
        r = removed.setdefault(j["kind"], [0, 0])
        r[0] += not kept
        r[1] += 1
    det = grid = head = cap = 0
    for g in tables:
        cands = [b for b in pages.get(g["page"], [])
                 if b["body"] and b["kind"] == "table" and _iou(b["bbox"], g["bbox"]) >= 0.5]
        det += bool(cands)
        grid += any(b["grid"] == [g["rows"], g["cols"]] for b in cands)
        best = max(cands, key=lambda b: _iou(b["bbox"], g["bbox"])) if cands else None
        if best:
            head += (best.get("header_rows") or 0) == g.get("header_rows", 0)
            gc, bc = g.get("caption"), best.get("caption_bbox")
            cap += (gc is None and bc is None) or (gc is not None and bc is not None and
                                                   _center_in(bc, [[gc[0] - 3, gc[1] - 3, gc[2] + 3, gc[3] + 3]]))
    junk_text = sum(len(re.findall(r"\\n(?![A-Za-z])", b["text"])) + len(re.findall(r"\$[^$\n]*[\\^_][^$\n]*\$", b["text"]))
                    for bs in pages.values() for b in bs if b["body"])           # 모델 잡문자 꼴만 — 영문자 앞 \n(경로)·금액 $ 는 세지 않는다
    legacy = [f for f in figs if "kind" not in f]   # 종류 없는 그림 = 그림으로 된 표(test.pdf) — 읽을수록 좋다
    fig_chars = sum(len(norm(b["text"])) for f in legacy for b in pages.get(f["page"], [])
                    if b["body"] and b["bbox"] and not b.get("joined")
                    and _inter(f["bbox"], b["bbox"]) >= 0.5 * _area(b["bbox"])) if legacy else None   # 그림 감싼 문단을 이은 블록은 빼고
    heads, items = golden.get("headings", []), golden.get("lists", [])
    found = level_ok = 0
    for g in heads:                                 # 겹치는 블록이 제목 역할이면 찾음, 가장 많이 겹치는 제목 블록의 수준을 본다
        hs = [b for b in _over(pages, g) if b.get("heading")]
        found += bool(hs)
        level_ok += bool(hs) and hs[0].get("heading_level") == g["level"]
    list_ok = 0
    for g in items:                                 # 가장 많이 겹치는 블록의 list_level — 나뉜 항목은 조각 모두가 맞아야 한다
        cands = _over(pages, g)
        if not cands:
            continue
        whole = _iou(cands[0]["bbox"], g["bbox"]) >= 0.5 or _inter(cands[0]["bbox"], g["bbox"]) >= COVER * _area(g["bbox"])
        list_ok += all(b.get("list_level") == g["level"] for b in (cands[:1] if whole else cands))
    out = {"seconds": run.get("seconds"), "n_pages": len(golden["pages"]) or len(pages), "text": text, "leak": leak,  # 쪽 정답이 없는 그림 전용 정답은 출력 쪽 수
           "junk_removed": removed,
           "tables": {"detected": det, "grid_ok": grid, "header_ok": head, "caption_ok": cap, "total": len(tables)},
           "figure_chars": fig_chars, "junk_text": junk_text,
           "headings": {"found": [found, len(heads)], "level_ok": [level_ok, found]}, "lists": {"level_ok": [list_ok, len(items)]}}
    if kinds := [f for f in figs if "kind" in f]:   # 아래 지표는 정답에 그 정보가 있을 때만 낸다
        out["figures"] = _figures(pages, kinds)
    if any("route" in f or "facts" in f for f in kinds):
        out["figure_desc"] = _figure_desc(pages, kinds)
    if lines := [(t["page"], s) for t in golden.get("image_text", []) for s in map(norm, t["lines"]) if len(s) >= MIN_SEG]:
        out["image_text"] = [sum(present(s, body.get(p) or "") for p, s in lines), len(lines)]
    if entries := golden.get("figure_text"):
        out["figure_text"] = _figure_text(pages, entries, segs)
    if gm := golden.get("meta"):
        m = run.get("meta")
        out["meta"] = None if m is None else [sum(" ".join(str(m.get(k, "")).split()) == " ".join(str(v).split())
                                                  for k, v in gm.items()), len(gm)]
    return out


def _fig_blocks(pages: dict, g: dict) -> list[dict]:
    """정답 그림 자리의 출력 그림 블록 — IoU 0.5 이상 하나, 아니면 대부분 정답 안에 든 조각들(각각 PART 이상)이 함께 COVER 이상."""
    figs = [b for b in pages.get(g["page"], []) if b["body"] and b["bbox"] and b["kind"] == "figure"]
    one = sorted((b for b in figs if _iou(b["bbox"], g["bbox"]) >= 0.5), key=lambda b: -_iou(b["bbox"], g["bbox"]))
    if one:
        return one
    parts = [b for b in figs if _inter(b["bbox"], g["bbox"]) >= max(PART * _area(g["bbox"]), 0.5 * _area(b["bbox"]))]
    return parts if sum(_inter(b["bbox"], g["bbox"]) for b in parts) >= COVER * _area(g["bbox"]) else []


def _figures(pages: dict, figs: list[dict]) -> dict:
    """그림 찾음·제목 연결(제목 있는 그림만)·그래프·사진·지도 속 글자가 글 블록으로 본문에 섞인 글자 수(그림 블록이 품은 글자는 빼고)."""
    found = cap = 0
    for g in figs:
        hits = _fig_blocks(pages, g)
        found += bool(hits)
        if (gc := g.get("caption")) is not None:
            cap += any(b.get("caption_bbox") and _center_in(b["caption_bbox"], [[gc[0] - 3, gc[1] - 3, gc[2] + 3, gc[3] + 3]])
                       for b in hits)
    leak = sum(len(norm(b["text"])) for g in figs if g["kind"] in ("chart", "photo", "map") for b in pages.get(g["page"], [])
               if b["body"] and b["bbox"] and b["kind"] == "text" and _inter(g["bbox"], b["bbox"]) >= 0.5 * _area(b["bbox"]))
    return {"found": [found, len(figs)], "caption_ok": [cap, sum(g.get("caption") is not None for g in figs)], "leak_chars": leak}


def _has(word, sent: str) -> bool:
    """낱말이 (정규화한) 문장에 있나 — 숫자로 시작·끝나는 낱말은 앞뒤가 숫자가 아닐 때만('1일' 은 '21일' 에, '0' 은 '2026' 에
    없다). YAML 은 0.5·2025 를 수로 읽으므로 글로 바꿔 본다."""
    w = norm(str(word))
    for m in re.finditer(re.escape(w), sent):
        if w[0].isdigit() and m.start() and sent[m.start() - 1].isdigit():
            continue
        if w[-1].isdigit() and m.end() < len(sent) and sent[m.end()].isdigit():
            continue
        return True
    return False


def _figure_desc(pages: dict, figs: list[dict]) -> dict:
    """그림 갈래·그림 설명(D19) — 정답 그림 자리의 그림 블록이 모두 정답 갈래인가, 설명할 그림(사실을 적은 그림)에 설명이 있나,
    정답 사실이 설명의 한 문장 안에 있나(사실 = 낱말 무리 목록, 무리마다 낱말 하나라도), 근거 검사로 버린 문장 수, 본문 옮김·확신
    낮음으로 거른 문장 수. 한 정답 그림을
    엔진이 여럿으로 잡았으면(IoU 0.5 이상인 큰 조각과 정답 상자 안에 대부분 든 나머지 조각) 모두 그 그림이다."""
    route = n_route = desc = n_desc = facts = n_facts = dropped = filtered = 0
    for g in figs:
        hits = _fig_blocks(pages, g)
        hits += [b for b in pages.get(g["page"], []) if b["body"] and b["bbox"] and b["kind"] == "figure" and b not in hits
                 and _inter(b["bbox"], g["bbox"]) >= 0.5 * _area(b["bbox"])]
        if "route" in g:
            n_route += 1
            route += bool(hits) and all(b.get("route") == g["route"] for b in hits)
        if "facts" in g:
            n_desc += 1
            text = " ".join(b.get("desc") or "" for b in hits)
            desc += bool(text.strip())
            sents = [norm(x) for x in re.split(r"(?<=[.!?])\s+|\n+", text) if x.strip()]
            n_facts += len(g["facts"])
            facts += sum(any(all(any(_has(w, x) for w in group) for group in fact) for x in sents) for fact in g["facts"])
        dropped += sum(b.get("desc_dropped", 0) for b in hits)
        filtered += sum(b.get("desc_filtered", 0) for b in hits)
    return {"route_ok": [route, n_route], "described": [desc, n_desc], "facts": [facts, n_facts], "dropped": dropped,
            "filtered": filtered}


def _figure_text(pages: dict, entries: list[dict], segs: dict) -> dict:
    """그래프·지도 속 글자 — 정답 그림 자리의 그림 블록 글에 정답 줄(2자 이상)이 있나(읽음), 그 글에서 글자가 든 줄 가운데 정답
    줄에도 그 자리 글자층에도 없는 낱말(2자 이상)이 섞인 줄(오독·지어냄 — 숫자뿐인 낱말과 한 글자는 정답에 넣지 않았으니 보지
    않고, 그림에 붙은 글자층 글('(a)')은 맞는 글이다)."""
    read = total = wrong = lines = 0
    for g in entries:
        hits = _fig_blocks(pages, g)
        text = norm("".join(b["text"] for b in hits))
        want = [s for s in map(norm, g["lines"]) if len(s) >= 2]
        read += sum(present(s, text) for s in want)
        total += len(want)
        ref = "".join(want) + "".join(norm(s) for s, box in segs.get(g["page"], []) if _center_in(box, [g["bbox"]]))
        for line in (x for b in hits for x in b["text"].splitlines()):
            words = [w for w in map(norm, line.split()) if len(w) >= 2 and _LETTER.search(w)]
            lines += bool(words)
            wrong += any(w not in ref for w in words)
    return {"read": [read, total], "wrong_lines": [wrong, lines]}


# ---- 보고 ----

def _frac(v) -> str:
    return f"{v[0]}/{v[1]} ({100 * v[0] / v[1]:.1f}%)" if v[1] else "—"


def render(report: dict) -> str:
    names = list(report)
    rows = [("본문 글자 조각이 출력에 있음", lambda r: _frac(r["text"]["normal"])),
            ("표지 글자가 본문에 남음", lambda r: _frac(r["leak"]["cover"])),
            ("목차 글자가 본문에 남음", lambda r: _frac(r["leak"]["toc"]))]
    kinds = sorted({k for r in report.values() for k in r["junk_removed"]})
    rows += [(f"잡요소 제거 — {k}", lambda r, k=k: _frac(r["junk_removed"].get(k, [0, 0]))) for k in kinds]
    rows += [("표 검출(IoU≥0.5)", lambda r: _frac([r["tables"]["detected"], r["tables"]["total"]])),
             ("표 행·열 일치", lambda r: _frac([r["tables"]["grid_ok"], r["tables"]["total"]])),
             ("표 머리행 일치", lambda r: _frac([r["tables"].get("header_ok", 0), r["tables"]["total"]])),
             ("표 제목 연결", lambda r: _frac([r["tables"].get("caption_ok", 0), r["tables"]["total"]]))]

    def has(key):                                   # 정답에 그 정보가 있을 때만 줄을 낸다
        return any(key in r for r in report.values())

    if has("figures"):
        rows += [("그림 찾음", lambda r: _frac(r["figures"]["found"]) if "figures" in r else "—"),
                 ("그림 제목 연결(제목 있는 그림)", lambda r: _frac(r["figures"]["caption_ok"]) if "figures" in r else "—"),
                 ("그래프·사진·지도 글자가 본문 글로 섞임(글자 수)",
                  lambda r: str(r["figures"]["leak_chars"]) if "figures" in r else "—")]
    rows += [("제목 찾음", lambda r: _frac(r.get("headings", {}).get("found", [0, 0]))),
             ("제목 수준 일치(찾은 제목 중)", lambda r: _frac(r.get("headings", {}).get("level_ok", [0, 0]))),
             ("항목 수준 일치", lambda r: _frac(r.get("lists", {}).get("level_ok", [0, 0]))),
             ("출력 잡문자(글자 그대로의 \\n·LaTeX)", lambda r: str(r.get("junk_text", 0))),
             ("그림 영역에서 읽은 글자 수", lambda r: "—" if r["figure_chars"] is None else str(r["figure_chars"]))]
    if has("image_text"):
        rows.append(("그림 속 글자 읽음(포스터 줄)", lambda r: _frac(r["image_text"]) if "image_text" in r else "—"))
    if has("figure_text"):
        rows += [("그래프·지도 속 글자 읽음", lambda r: _frac(r["figure_text"]["read"]) if "figure_text" in r else "—"),
                 ("그래프·지도 글 중 정답에 없는 줄(오독·지어냄)",
                  lambda r: _frac(r["figure_text"]["wrong_lines"]) if "figure_text" in r else "—")]
    if has("figure_desc"):
        rows += [("그림 갈래 맞음(글만으로 충분 / 그림과 함께)", lambda r: _frac(r["figure_desc"]["route_ok"]) if "figure_desc" in r else "—"),
                 ("그림 설명 있음(설명할 그림)", lambda r: _frac(r["figure_desc"]["described"]) if "figure_desc" in r else "—"),
                 ("그림 설명에 든 정답 사실", lambda r: _frac(r["figure_desc"]["facts"]) if "figure_desc" in r else "—"),
                 ("그림 설명에서 버린 문장(근거 없는 숫자·이름)",
                  lambda r: str(r["figure_desc"]["dropped"]) if "figure_desc" in r else "—"),
                 ("그림 설명에서 거른 문장(본문 옮김·확신 낮음)",
                  lambda r: str(r["figure_desc"].get("filtered", 0)) if "figure_desc" in r else "—")]
    if has("meta"):
        rows.append(("문서 정보 맞음(제목·기관·날짜)", lambda r: _frac(r["meta"]) if r.get("meta") else "—"))
    rows.append(("처리 시간(쪽당)", lambda r: f"{r['seconds']:.0f}초 ({r['seconds'] / r['n_pages']:.1f}초)"
                 if r["seconds"] is not None else "—"))
    out = ["| 지표 | " + " | ".join(NAMES.get(n, n) for n in names) + " |", "|---" * (len(names) + 1) + "|"]
    out += ["| " + label + " | " + " | ".join(fn(report[n]) for n in names) + " |" for label, fn in rows]
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="비교군 시험(개수·비율만 출력)")
    ap.add_argument("pdf")
    ap.add_argument("--golden", required=True)
    ap.add_argument("--runs", default=",".join(RUNS))
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--rerun", action="store_true")
    a = ap.parse_args(argv)
    settings = load_settings()
    pdf = Path(a.pdf).resolve()
    doc_id = doc_id_of(pdf.read_bytes())
    golden = yaml.safe_load(Path(a.golden).read_text(encoding="utf-8"))
    if golden["doc_id"] != doc_id:
        sys.exit("정답 파일의 doc_id 가 PDF 와 다르다")
    doc_dir = Path(a.out or settings.env.paths.out_dir) / doc_dirname(doc_id)
    segs = segments(pdf)
    report = {}
    for name in a.runs.split(","):
        if name == "kdp":
            if not (doc_dir / "document.json").exists():
                print("kdp 출력 없음 — 먼저 kdp parse", file=sys.stderr)
                continue
            run = load_kdp(doc_dir)
        else:
            cache = doc_dir / "compare" / f"{name}.json"
            if a.rerun or not cache.exists():
                try:
                    write_json(cache, RUNNERS[name](pdf, settings))
                except ImportError as e:            # docling 은 파서 env 밖에서 돌린다(설계 9장) — 결과 파일만 있으면 된다
                    print(f"{name} 건너뜀({e.name} 없음) — 설치된 env 에서 --runs {name} 으로 먼저 돌린다", file=sys.stderr)
                    continue
            run = read_json(cache)
            if name == "paddle_default":                   # 구조 지표 전 저장본 — 제목 수준은 라벨·글자로 정해지니 다시 돌리지 않고 채운다
                for b in (b for bs in run["pages"].values() for b in bs if "heading" not in b):
                    lv = paddle_title_level(b["label"], b["text"])
                    b["heading"], b["heading_level"] = lv is not None, lv
        report[name] = evaluate(segs, golden, run)
    write_json(doc_dir / "compare" / "report.json", report, indent=1)
    print(render(report))


if __name__ == "__main__":
    main()
