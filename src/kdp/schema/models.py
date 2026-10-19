"""정본 JSON 스키마(기획서 4.2)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.2"                 # 1.1: 블록 heading_level(M3) · 1.2: 그림 갈래·그림 설명(D19)

PageClass = Literal["normal", "cover", "toc", "blank", "divider", "form_blank", "approval"]
BBox = tuple[float, float, float, float]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NUp(_Model):
    cols: int
    rows: int
    cell: tuple[int, int]                # (열, 행) — 왼쪽 위부터 행 우선
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
    heading_level: int | None = None     # 제목 수준 — 문서 제목 0, 섹션 1 부터(작을수록 큼), 확신이 없으면 비운다(기획서 4.3 규칙 3)
    header_rows: int | None = None
    caption_id: str | None = None
    unit: str | None = None
    note_ids: list[str] | None = None
    continues_from: str | None = None
    img_path: str | None = None
    route: Literal["html", "image", "text"] | None = None   # 표 갈래(D18) html|image — image 면 후속 단계가 그림으로도 다룬다.
                                                        # 그림 갈래(D19) text|image — text 는 글자만으로 뜻이 서는 그림, image 는 그림과 함께 봐야 하는 그림
    route_reason: str | None = None                     # 표 photo_in_cell | failed | visual · 그림 chart | labels | no_text | sentences
    figure_text: str | None = None       # 그림·차트 속 인쇄 글자(D14)
    figure_desc: str | None = None       # 그림 설명(D19) — 그림을 봐야 아는 뜻을 범용 VLM 이 쓴 글, 근거 검사를 통과한 문장만
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
    text_layer_agreement: float | None = None   # 레이어 글자 중 모델 출력에도 있는 비율(8.5)
    n_up: NUp | None = None
    rot90_k: int | None = None           # 스캔을 세운 횟수(np.rot90 의 k) — bbox 는 시트 좌표, 위치 규칙은 세운 쪽 기준(geometry)
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
    undescribed_figures: int = 0         # 설명할 그림인데 설명 응답이 없는 것(D19) — kdp describe 로 채운다


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
