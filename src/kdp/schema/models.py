"""정본 JSON 스키마(기획서 4.2)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"

PageClass = Literal["normal", "cover", "toc", "blank", "divider", "form_blank", "approval"]
BBox = tuple[float, float, float, float]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NUp(_Model):
    cols: int
    rows: int
    cell: tuple[int, int]
    region: BBox                         # 시트 안 이 논리 페이지 영역(pt)


class Block(_Model):
    id: str
    order: int | None = None
    type: str
    bbox: BBox                           # 시트 pt, 좌상단 원점
    text: str | None = None
    html: str | None = None
    keep: bool = True
    drop_reason: str | None = None
    score: float | None = None
    section_path: list[str] | None = None
    list_level: int | None = None
    header_rows: int | None = None
    caption_id: str | None = None
    unit: str | None = None
    note_ids: list[str] | None = None
    continues_from: str | None = None
    img_path: str | None = None
    figure_text: str | None = None       # 그림·차트 속 인쇄 글자(D14)
    meta: dict[str, str] | None = None
    truncated: Literal["length", "repetition"] | None = None
    trimmed: list[str] | None = None
    pii: list[str] | None = None
    aside: bool | None = None
    merged_into: str | None = None       # 병합 묶음의 대표 블록 — 글자는 대표에만 있다
    error: str | None = None
    provenance: list[dict[str, Any]] | None = None


class Page(_Model):
    page_no: int                         # 논리 페이지 번호
    sheet_no: int                        # 원본 PDF 쪽 번호
    size: tuple[float, float]            # 시트 크기(pt, /Rotate 반영)
    page_class: PageClass = "normal"
    page_class_guess: PageClass | None = None
    keep: bool = True
    form: bool | None = None
    born_digital: bool | None = None
    n_up: NUp | None = None
    blocks: list[Block] = Field(default_factory=list)


class Source(_Model):
    paths: list[str]
    pages: int
    producer: str | None = None
    born_digital: bool | None = None
    acl_ref: str | None = None


class ParserInfo(_Model):
    kdp_version: str
    engine: dict[str, Any]
    engine_hash: str
    assemble_hash: str
    run_at: str                          # 엔진을 돌린 시각(재조립해도 그대로)


class QC(_Model):
    status: Literal["ok", "review", "error"] = "ok"
    reasons: list[str] = Field(default_factory=list)
    block_errors: int = 0
    truncated_blocks: int = 0
    empty_pages: int = 0
    text_layer_agreement: float | None = None
    drop_char_ratio: float | None = None
    repeat_ngram_ratio: float | None = None
    unsupported_char_ratio: float | None = None


class Document(_Model):
    schema_version: str = SCHEMA_VERSION
    doc_id: str
    source: Source
    parser: ParserInfo
    meta: dict[str, Any] = Field(default_factory=dict)
    qc: QC = Field(default_factory=QC)
    output_hash: str | None = None
    near_dup_of: str | None = None
    pages: list[Page] = Field(default_factory=list)
    markdown: str | None = None


def json_schema() -> dict:
    return Document.model_json_schema()
