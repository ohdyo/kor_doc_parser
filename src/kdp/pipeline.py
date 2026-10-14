"""문서 하나 처리 — 단계 순서는 여기서만 정한다(설계 5.1)."""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .assemble.figures import table_img_names
from .assemble.order import assign_order, kept, output_hash
from .classify.blocks import classify_blocks
from .config import OutputCfg, Settings, assemble_hash, engine_hash, load_settings
from .convert import to_page
from .engine import Engine, PageInput, ReplayEngine, create_engine
from .engine.base import page_stem
from .io.ids import doc_dirname, doc_id_of
from .io.pdf import PdfInfo, open_pdf, read_info, text_chars
from .io.pdfium_lock import PDFIUM_LOCK
from .io.render import render_page, render_scale
from .io.write import read_json, write_json, write_json_gz, write_jsonl, write_text
from .qc.checks import run_checks
from .schema import Document, ParserInfo, Source

ASSEMBLE_CODE_VERSION = "1"
JSONL_FIELDS = ("type", "bbox", "section_path", "list_level", "text", "html", "continues_from", "header_rows",
                "caption_id", "unit", "note_ids", "img_path", "figure_text", "truncated", "pii")


def parse_pdf(pdf_path, out_root=None, settings: Settings | None = None, engine: Engine | None = None) -> Document:
    settings = settings or load_settings()
    pdf_path = Path(pdf_path)
    data = pdf_path.read_bytes()
    doc_id = doc_id_of(data)
    doc_dir = Path(out_root or settings.env.paths.out_dir) / doc_dirname(doc_id)
    raw_dir = doc_dir / "raw"
    own_engine = engine is None
    engine = engine or create_engine(settings)
    try:
        pdf = open_pdf(data)
        try:
            info = read_info(pdf, doc_id, str(pdf_path))
            desc = engine.describe()
            write_json(raw_dir / "doc.json", info.to_dict())
            write_json(raw_dir / "engine.json", {
                "engine": desc,
                "engine_hash": engine_hash(settings, desc),
                "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            if hasattr(engine, "config_snapshot"):
                write_text(raw_dir / "paddlex_config.yaml", engine.config_snapshot())
            raws = list(engine.run(_pages(pdf, info, settings, raw_dir), raw_dir))
        finally:
            with PDFIUM_LOCK:
                pdf.close()
    finally:
        if own_engine:
            engine.close()
    return _finish(info, raws, doc_dir, settings)


def reassemble(doc_dir, settings: Settings | None = None) -> Document:
    """raw/ 만으로 판정·조립·출력을 다시 한다(GPU·원본 PDF 불필요)."""
    settings = settings or load_settings()
    raw_dir = Path(doc_dir) / "raw"
    info = PdfInfo.from_dict(read_json(raw_dir / "doc.json"))
    raws = list(ReplayEngine(raw_dir).run())
    return _finish(info, raws, Path(doc_dir), settings)


def _pages(pdf, info: PdfInfo, settings: Settings, raw_dir: Path):
    """쪽 렌더 + 텍스트 레이어 저장. 모아찍기 분할·스캔 방향은 M2(prep)."""
    r = settings.parse.render
    for p in info.pages:
        with PDFIUM_LOCK:
            page = pdf[p.sheet_no - 1]
        try:
            write_json_gz(raw_dir / "textlayer" / f"{page_stem(p.sheet_no)}.json.gz", text_chars(page))
            scale = render_scale(p.size, r.dpi, r.max_pixels)
            img = render_page(page, scale)
        finally:
            with PDFIUM_LOCK:
                page.close()
        yield img, PageInput(page_no=p.sheet_no, sheet_no=p.sheet_no, scale=scale,
                             sheet_size=tuple(p.size), rotation=p.rotation)


def _finish(info: PdfInfo, raws, doc_dir: Path, settings: Settings) -> Document:
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
    doc = classify_blocks(doc, settings.parse.classify)
    doc = assign_order(doc)
    doc = run_checks(doc, raws, settings.parse.qc)
    doc.output_hash = output_hash(doc)
    write_outputs(doc, doc_dir, settings.parse.output)
    return doc


def jsonl_rows(doc: Document):
    for page, b in kept(doc):
        row = {"doc_id": doc.doc_id, "block_id": b.id, "page_no": page.page_no, "sheet_no": page.sheet_no}
        for f in JSONL_FIELDS:
            v = getattr(b, f)
            if v is not None:
                row[f] = list(v) if isinstance(v, tuple) else v
        yield row


def write_outputs(doc: Document, doc_dir: Path, out: OutputCfg) -> None:
    write_json(doc_dir / "document.json", doc.model_dump(mode="json", exclude_none=True), indent=1)
    if out.write_jsonl:
        write_jsonl(doc_dir / "blocks.jsonl", jsonl_rows(doc))
    if out.save_images:
        _copy_kept_images(doc, doc_dir)


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
