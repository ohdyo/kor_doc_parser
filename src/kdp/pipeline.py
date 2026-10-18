"""문서 하나 처리 — 단계 순서는 여기서만 정한다(설계 5.1)."""
from __future__ import annotations

import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .assemble.figures import join_wrapped, table_img_names
from .assemble.headings import headings
from .assemble.lists import lists
from .assemble.normalize import normalize
from .assemble.order import OUTPUT_FIELDS, assign_order, kept, output_hash, output_rows
from .assemble.paragraphs import join_paragraphs
from .assemble.tables import merge_tables, tables
from .classify.blocks import classify_blocks
from .classify.pages import classify_pages
from .classify.rules_meta import mark_meta_blocks
from .config import OutputCfg, Settings, assemble_hash, engine_hash, load_settings
from .convert import to_page
from .engine import Engine, PageInput, ReplayEngine, create_engine
from .engine.base import page_stem
from .enrich.meta import promote_meta
from .enrich.pii import detect_pii
from .io.ids import doc_dirname, doc_id_of
from .io.pdf import PdfInfo, open_pdf, read_info, text_chars
from .io.pdfium_lock import PDFIUM_LOCK
from .io.render import render_page, render_region, render_scale, unrotate_box
from .io.write import read_json, read_json_gz, write_json, write_json_gz, write_jsonl, write_text
from .prep import nup
from .prep.orient import is_scan, judge, layerless
from .qc.checks import run_checks
from .render_md import render_markdown
from .schema import Document, ParserInfo, Source
from .textlayer.extract import apply_text_layer, display_chars, mark_hidden

ASSEMBLE_CODE_VERSION = "9"
JSONL_FIELDS = OUTPUT_FIELDS


class RawIncompleteError(RuntimeError):
    """raw/ 에 빠진 쪽이 있거나 엔진 실행 기록이 없다 — 재조립하지 않는다."""


def parse_pdf(pdf_path, out_root=None, settings: Settings | None = None, engine: Engine | None = None) -> Document:
    t0 = time.perf_counter()
    settings = settings or load_settings()
    pdf_path = Path(pdf_path).resolve()
    data = pdf_path.read_bytes()
    doc_id = doc_id_of(data)
    doc_dir = Path(out_root or settings.env.paths.out_dir) / doc_dirname(doc_id)
    raw_dir, tmp_dir = doc_dir / "raw", doc_dir / "raw.tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    own_engine = engine is None
    engine = engine or create_engine(settings)
    try:
        pdf = open_pdf(data)
        try:
            info = read_info(pdf, doc_id, str(pdf_path))
            desc = engine.describe()
            started = datetime.now(timezone.utc).isoformat(timespec="seconds")
            raws = list(engine.run(_pages(pdf, info, settings, tmp_dir), tmp_dir))
        finally:
            with PDFIUM_LOCK:
                pdf.close()
        snapshot = engine.config_snapshot() if hasattr(engine, "config_snapshot") else None
    finally:
        if own_engine:
            engine.close()
    # 엔진이 끝까지 돈 뒤에만 raw/ 를 바꾼다 — 중간에 실패한 실행은 이전 raw/ 를 건드리지 않는다
    write_json(tmp_dir / "doc.json", info.to_dict())
    write_json(tmp_dir / "engine.json", {"engine": desc, "engine_hash": engine_hash(settings, desc), "run_at": started,
                                       "seconds": round(time.perf_counter() - t0, 1)})   # 모델 불러오기 포함
    if snapshot:
        write_text(tmp_dir / "paddlex_config.yaml", snapshot)
    shutil.rmtree(raw_dir, ignore_errors=True)
    tmp_dir.rename(raw_dir)
    return _finish(info, raws, doc_dir, settings)


def reassemble(doc_dir, settings: Settings | None = None) -> Document:
    """raw/ 만으로 판정·조립·출력을 다시 한다(GPU·원본 PDF 불필요)."""
    settings = settings or load_settings()
    raw_dir = Path(doc_dir) / "raw"
    if not (raw_dir / "engine.json").exists():
        raise RawIncompleteError(f"{raw_dir} 에 엔진 실행 기록이 없다")
    info = PdfInfo.from_dict(read_json(raw_dir / "doc.json"))
    raws = list(ReplayEngine(raw_dir).run())
    return _finish(info, raws, Path(doc_dir), settings)


def _pages(pdf, info: PdfInfo, settings: Settings, raw_dir: Path):
    """논리 페이지(설계 1장 prep): 시트 렌더·텍스트 레이어 저장(시트마다 한 번) → 스캔 시트 방향 → 모아찍기 분할(칸마다 다시 렌더).
    방향을 먼저 세운다 — 돌아간 시트를 먼저 나누면 칸 순서가 바뀐다(설계 3.7). page_no 는 문서 전체의 논리 페이지 일련번호."""
    r, parse = settings.parse.render, settings.parse
    page_no, layouts = 0, {}                       # layouts: 세운 시트 크기 → 앞 시트의 모아찍기 배치
    for p in info.pages:
        with PDFIUM_LOCK:
            page = pdf[p.sheet_no - 1]
        try:
            scale = render_scale(p.size, r.dpi, r.max_pixels)
            img = render_page(page, scale)
            chars = mark_hidden(text_chars(page), img, p.bbox, p.rotation, scale)   # 가려진 글자는 정답으로 쓰지 않는다
            write_json_gz(raw_dir / "textlayer" / f"{page_stem(p.sheet_no)}.json.gz", chars)
            scan = is_scan(chars)                  # 레이어로 글을 못 읽는 시트 — 모아찍기를 잉크로 본다
            # 방향은 레이어 글자가 하나도 없는 시트만, 방향 판정으로(설계 3.7) — k None 은 확신 없음, tries 는 그때 볼 방향(기운 방향 → 그대로)
            k, tries = judge(img, settings.env) if parse.orientation == "scan_only" and layerless(chars) else (0, ())
            layout = None
            if parse.nup.enabled:
                layer = None if scan else [(b, c) for c, b, gen in display_chars(chars, p.bbox, p.rotation)
                                           if not gen and not c.isspace()]
                hits = nup.search(p.size, layer, nup.ink_mask(img) if scan else None, layouts, (k,) if k is not None else tries)
                if k is None:                      # 모르면 칸 아래 쪽번호가 맞는 첫 방향으로 세워 나눈다 — 위 표식은 180° 짝의 아래 쪽번호와
                    k = next((t for t in tries if t in hits and hits[t].mark[0] == "bottom"), None)   # 못 가르고, 누운 시트는 보지 않는다
                layout = hits.get(k)
            k = k or 0                             # 그래도 모르면 돌리지도 나누지도 않는다
            img = np.ascontiguousarray(np.rot90(img, k)) if k else img
            size = (p.size[1], p.size[0]) if k % 2 else tuple(p.size)                # 세운 시트 pt
            items = [(img, {"scale": scale})]
            if layout:                             # 칸마다 시트 렌더를 자르지 않고 원래 쪽 기준 dpi 로 다시 그린다(설계 4장)
                layouts[nup.key(size)] = layout
                n, items = layout.cols * layout.rows, []
                for cell, box in nup.cells(size, layout.cols, layout.rows, layout.by_col):
                    region = tuple(round(v, 2) for v in unrotate_box(box, k, size))    # 돌리기 전 시트 표시 pt
                    s = nup.cell_scale((region[2] - region[0], region[3] - region[1]), n, r.dpi, r.max_pixels)
                    sub, origin = render_region(page, s, region, p.size)
                    items.append((np.ascontiguousarray(np.rot90(sub, k)) if k else sub,
                                  {"scale": s, "origin": origin,
                                   "n_up": {"cols": layout.cols, "rows": layout.rows, "cell": cell, "region": region}}))
        finally:
            with PDFIUM_LOCK:
                page.close()
        for im, kw in items:
            page_no += 1
            yield im, PageInput(page_no=page_no, sheet_no=p.sheet_no, sheet_size=tuple(p.size), rotation=p.rotation,
                                rot90_k=k, user_bbox=tuple(p.bbox), **kw)


def _finish(info: PdfInfo, raws, doc_dir: Path, settings: Settings) -> Document:
    missing = set(range(1, info.n_pages + 1)) - {r.input.sheet_no for r in raws}
    if missing:
        raise RawIncompleteError(f"raw/ 에 없는 쪽 {len(missing)}개: {sorted(missing)[:10]}")
    nos = [r.input.page_no for r in raws]                 # 모아찍기 칸 하나가 빠져도 시트는 남는다 — 논리 페이지로도 본다
    per_sheet = Counter(r.input.sheet_no for r in raws)
    if nos != list(range(1, len(nos) + 1)) or any(
            r.input.n_up and per_sheet[r.input.sheet_no] != r.input.n_up["cols"] * r.input.n_up["rows"] for r in raws):
        raise RawIncompleteError(f"raw/ 의 논리 페이지가 이어지지 않는다: {sorted(nos)[:20]}")
    eng = read_json(doc_dir / "raw" / "engine.json")
    doc = Document(
        doc_id=info.doc_id,
        source=Source(paths=[info.path], pages=info.n_pages, producer=info.producer),
        parser=ParserInfo(
            kdp_version=__version__,
            engine=eng["engine"],
            engine_hash=eng["engine_hash"],
            assemble_hash=assemble_hash(settings, ASSEMBLE_CODE_VERSION),
            run_at=eng["run_at"],
        ),
        pages=[to_page(r) for r in raws],
    )
    doc = apply_text_layer(doc, _layers(doc_dir, info), info, prefer=settings.parse.text_layer.mode == "prefer")
    doc.source.born_digital = all(p.born_digital for p in doc.pages)
    doc = normalize(doc)
    doc = classify_blocks(doc, settings.parse.classify)
    doc = mark_meta_blocks(doc)
    doc = classify_pages(doc, settings.parse.classify)
    doc = tables(doc)                                  # 배치용 표를 글로 푼 뒤에 제목·항목을 본다
    doc = merge_tables(doc)                            # 제목보다 먼저 — 버린 '(계속)' 표시가 섹션 경로에 들어가지 않게
    doc = headings(doc, info.toc)
    doc = lists(doc)
    doc = join_wrapped(doc)                            # 같은 쪽 이음이 먼저 — 쪽 넘김 이음이 묶음 대표를 가리키게
    doc = join_paragraphs(doc)
    doc = promote_meta(doc, settings.parse.meta)
    doc = detect_pii(doc)
    doc = assign_order(doc)
    doc = run_checks(doc, raws, settings.parse.qc)
    doc.markdown = render_markdown(doc) if settings.parse.output.write_markdown else None
    doc.output_hash = output_hash(doc)
    write_outputs(doc, doc_dir, settings.parse.output)
    return doc


def _layers(doc_dir: Path, info: PdfInfo) -> dict[int, list[dict]]:
    out = {}
    for p in info.pages:
        f = doc_dir / "raw" / "textlayer" / f"{page_stem(p.sheet_no)}.json.gz"
        if f.exists():
            out[p.sheet_no] = read_json_gz(f)
    return out


def jsonl_rows(doc: Document):
    return output_rows(doc)


def write_outputs(doc: Document, doc_dir: Path, out: OutputCfg) -> None:
    data = doc.model_dump(mode="json", exclude_none=True)
    for page in data["pages"]:                         # 계약상 늘 있는 키(기획서 4.2)
        page.setdefault("n_up", None)
        for b in page["blocks"]:
            b.setdefault("order", None)
    write_json(doc_dir / "document.json", data, indent=1)

    md = doc_dir / "document.md"
    if out.write_markdown and doc.markdown is not None:
        write_text(md, doc.markdown)
    else:
        md.unlink(missing_ok=True)
    jsonl = doc_dir / "blocks.jsonl"
    if out.write_jsonl:
        write_jsonl(jsonl, jsonl_rows(doc))
    else:
        jsonl.unlink(missing_ok=True)
    if out.save_images:
        _copy_kept_images(doc, doc_dir)
    else:
        shutil.rmtree(doc_dir / "imgs", ignore_errors=True)
    if not out.keep_raw:                               # 지우면 reassemble 할 수 없다
        shutil.rmtree(doc_dir / "raw", ignore_errors=True)


def _copy_kept_images(doc: Document, doc_dir: Path) -> None:
    """imgs/ 에는 keep 블록 크롭만 — 로고·도장 같은 버린 크롭은 raw/imgs 에만 남는다."""
    src, dst = doc_dir / "raw" / "imgs", doc_dir / "imgs"
    shutil.rmtree(dst, ignore_errors=True)
    names = []
    for _, b in kept(doc):
        if b.img_path:
            names.append(b.img_path.removeprefix("imgs/"))
        names.extend(table_img_names(b.html))
    for name in names:
        if (src / name).exists():
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / name, dst / name)
