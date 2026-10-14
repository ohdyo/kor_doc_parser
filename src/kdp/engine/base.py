"""엔진 인터페이스와 원시 결과(raw/ 에 저장되는 단위)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Protocol

import numpy as np


@dataclass
class PageInput:
    """논리 페이지 곁기록 — 엔진 결과와 입력 순서로 짝짓는다."""
    page_no: int
    sheet_no: int
    scale: float                                  # 논리 페이지 px / pt
    sheet_size: tuple[float, float]               # 시트 pt
    origin: tuple[float, float] = (0.0, 0.0)      # 시트 안 논리 페이지 원점(pt)
    n_up: dict | None = None                      # {cols, rows, cell, region}
    rotation: int = 0                             # PDF /Rotate
    rot90_k: int = 0                              # prep 에서 np.rot90(img, k) 로 세운 횟수


@dataclass
class RawBlock:
    idx: int                                      # parsing_res_list 안 순번
    label: str
    bbox: tuple[float, float, float, float]       # 엔진 입력 이미지 px
    content: str
    score: float | None = None
    vlm: dict | None = None                       # raw·finish_reason·usage·repetition_truncated·error
    merged_into: int | None = None                # 병합 묶음 대표 블록의 idx
    img: str | None = None                        # raw/imgs 안 크롭 파일 이름


@dataclass
class RawPage:
    input: PageInput
    width: int
    height: int
    ink_ratio: float
    blocks: list[RawBlock]
    table_imgs: dict[str, str] = field(default_factory=dict)   # 표 HTML 안 PaddleX 그림 경로 → raw/imgs 이름


class Engine(Protocol):
    def describe(self) -> dict: ...

    def run(self, pages: Iterable[tuple[np.ndarray, PageInput]], raw_dir: Path) -> Iterator[RawPage]: ...

    def close(self) -> None: ...


def page_stem(page_no: int) -> str:
    return f"p{page_no:04d}"


def raw_page_from_files(page_json: dict, sidecar: dict) -> RawPage:
    """PaddleX 쪽 JSON + 곁기록 → RawPage. 실제 엔진과 ReplayEngine 이 같은 길로 만든다."""
    def key(label, box):
        return label, tuple(round(v) for v in box)

    scores = {key(b["label"], b["coordinate"]): b["score"]
              for b in (page_json.get("layout_det_res") or {}).get("boxes", [])}
    extra = {b["idx"]: b for b in sidecar["blocks"]}
    blocks = []
    for i, b in enumerate(page_json.get("parsing_res_list", [])):
        bbox = tuple(float(v) for v in b["block_bbox"])
        e = extra.get(i, {})
        blocks.append(RawBlock(
            idx=i,
            label=b["block_label"],
            bbox=bbox,
            content=b.get("block_content") or "",
            score=scores.get(key(b["block_label"], bbox)),
            vlm=e.get("vlm"),
            merged_into=e.get("merged_into"),
            img=e.get("img"),
        ))
    inp = dict(sidecar["input"])
    inp["sheet_size"] = tuple(inp["sheet_size"])
    inp["origin"] = tuple(inp["origin"])
    return RawPage(
        input=PageInput(**inp),
        width=sidecar["width"],
        height=sidecar["height"],
        ink_ratio=sidecar["ink_ratio"],
        blocks=blocks,
        table_imgs=sidecar.get("table_imgs", {}),
    )
