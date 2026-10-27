"""그림 다시 읽기(테스트_결과 14.10) — 조립이 고른 그림(ocr_targets)을 원본 PDF 에서 더 높은 dpi 로 다시 그려(그림 상자를 조금
넓혀) 엔진과 같은 VLM 서버(PaddleOCR-VL — OpenAI 호환 vLLM)에 되풀이 벌점과 낮은 생성 한도로 다시 읽히고 응답을 raw/figure_ocr.jsonl
에 쌓는다. 엔진은 쪽 렌더(144dpi)에서 그림을 잘라(JPEG) 읽어 작은 값 표지·범례를 놓치거나 한도(4096토큰)까지 되풀이했다. 조립
(attach_figure_ocr)은 이 기록만 읽으므로 재조립은 모델 없이 같은 결과를 낸다. 실패한 그림은 적지 않는다 — 다음 실행이 다시 부른다."""
from __future__ import annotations

import base64
import io
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .assemble.figures import OCR_MIN_PIXELS, ocr_key, ocr_region, ocr_targets, spotting_lines
from .config import Settings
from .io.ids import doc_id_of
from .io.pdf import open_pdf
from .io.pdfium_lock import PDFIUM_LOCK
from .io.render import render_region
from .io.write import dumps
from .schema import Document

CACHE = "figure_ocr.jsonl"
TIMEOUT = 180                 # 그림 하나(초) — 큰 그림을 서버가 묶음으로 처리해도 넉넉하게


def load(raw_dir: Path) -> dict[str, dict]:
    f = Path(raw_dir) / CACHE
    out = {}
    for line in f.read_text(encoding="utf-8").splitlines() if f.exists() else []:
        try:
            r = json.loads(line)
        except ValueError:
            continue                                           # 빈 줄·쓰다 끊긴 줄 — 그 그림은 다음 실행이 다시 부른다
        out[r["key"]] = r
    return out


def scale_for(region, dpi: int, max_pixels: int) -> float:
    """dpi 배율 — 화소가 max_pixels 를 넘으면 배율을 낮춘다(서버가 줄이는 것보다 PDFium 이 그 화소로 바로 그리는 편이 곱다)."""
    w, h = region[2] - region[0], region[3] - region[1]
    return min(dpi / 72, math.sqrt(max_pixels / max(w * h, 1e-6)))


def _png(pdf, page, region, scale: float) -> tuple[bytes, tuple[int, int]]:
    """시트 영역을 scale 로 그려 PNG 로 — 돌린 논리 페이지(rot90_k)는 엔진처럼 세운다. PDFium 은 PDFIUM_LOCK 안에서만 쓴다."""
    from PIL import Image

    with PDFIUM_LOCK:
        sheet = pdf[page.sheet_no - 1]
    try:
        img, _ = render_region(sheet, scale, region, page.size)
    finally:
        with PDFIUM_LOCK:
            sheet.close()
    if page.rot90_k:
        img = np.rot90(img, page.rot90_k)
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img[:, :, ::-1])).save(buf, format="PNG")   # BGR → RGB
    return buf.getvalue(), (img.shape[1], img.shape[0])


def run(doc: Document, doc_dir: Path, settings: Settings, pdf: bytes | Path) -> tuple[int, list[str]]:
    """읽기 기록이 없는 그림만 부른다. (새로 받은 수, 실패한 블록과 사유)를 돌려준다. pdf 는 이 문서의 원본 PDF(바이트 또는 경로 —
    읽을 그림이 있을 때만 읽고 문서 해시를 맞춘다)."""
    cfg, env = settings.parse.figure_ocr, settings.env.vlm
    raw = Path(doc_dir) / "raw"
    cache = load(raw)
    todo = []
    for page, b in ocr_targets(doc, cfg):
        region = ocr_region(page, b, cfg.pad_pt)
        key = ocr_key(cfg, b, region)
        if key not in cache:
            todo.append((key, page, b, region))
    if not todo:
        return 0, []
    if not env.server_url:
        raise RuntimeError("figure_ocr.enabled 인데 configs/env.yaml 의 vlm.server_url 이 없다")
    pdf = pdf if isinstance(pdf, bytes) else Path(pdf).read_bytes()
    if doc_id_of(pdf) != doc.doc_id:
        raise ValueError(f"PDF 가 이 문서({doc.doc_id[:15]})가 아니다")
    import openai

    client = openai.OpenAI(base_url=env.server_url, api_key=env.api_key or "EMPTY", timeout=TIMEOUT, max_retries=1)
    spot = cfg.prompt == "Spotting:"
    source = open_pdf(pdf)

    def call(item):
        key, page, b, region = item
        scale = scale_for(region, cfg.dpi, cfg.max_pixels)
        png, size = _png(source, page, region, scale)
        r = client.chat.completions.create(
            model=cfg.model, temperature=0, max_completion_tokens=cfg.max_tokens,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
                {"type": "text", "text": cfg.prompt}]}],
            extra_body={"skip_special_tokens": not spot, "repetition_penalty": cfg.repetition_penalty,   # Spotting 은 위치 토큰을 읽는다
                        "mm_processor_kwargs": {"min_pixels": OCR_MIN_PIXELS, "max_pixels": cfg.max_pixels}})
        ch = r.choices[0]
        content = ch.message.content or ""
        return {"key": key, "block_id": b.id, "model": cfg.model, "prompt": cfg.prompt, "region": list(region), "dpi": cfg.dpi,
                "scale": round(scale, 4), "size": list(size), "text": "\n".join(spotting_lines(content)) if spot else content,
                **({"raw": content} if spot else {}), "finish_reason": ch.finish_reason,
                "usage": r.usage.model_dump() if r.usage else None}

    new, failed = 0, []
    try:
        with ThreadPoolExecutor(max(1, env.max_concurrency)) as ex, open(raw / CACHE, "a", encoding="utf-8") as f:
            futs = {ex.submit(call, it): it[2] for it in todo}
            for fut in as_completed(futs):                     # 받는 대로 적는다 — 중간에 멈춰도 받은 읽기는 남는다
                try:
                    f.write(dumps(fut.result()) + "\n")
                    f.flush()
                    new += 1
                except Exception as e:                         # 그 그림만 다음 실행에서 다시
                    failed.append(f"{futs[fut].id}: {type(e).__name__}: {str(e)[:200]}")
    finally:
        with PDFIUM_LOCK:
            source.close()
    return new, sorted(failed)
