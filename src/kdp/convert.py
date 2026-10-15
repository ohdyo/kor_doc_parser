"""엔진 원시 결과 → 정본 Page·Block(좌표는 시트 pt)."""
from __future__ import annotations

from .assemble.figures import fix_table_imgs, split_photo_text
from .engine.base import RawPage
from .io.ids import block_id
from .io.render import px_box_to_pt, unrotate_box
from .schema import Block, NUp, Page

FIGURE_LABELS = {"image", "chart", "header_image", "footer_image", "seal"}


def _r(v: float) -> float:
    return round(v, 2)


def _inside(box, outer) -> bool:
    return box[0] >= outer[0] - 1 and box[1] >= outer[1] - 1 and box[2] <= outer[2] + 1 and box[3] <= outer[3] + 1


def to_page(raw: RawPage) -> Page:
    inp = raw.input

    def to_pt(box):
        if inp.rot90_k:
            box = unrotate_box(box, inp.rot90_k, (raw.width, raw.height))
        return tuple(_r(v) for v in px_box_to_pt(box, inp.scale, inp.origin))

    boxes = {b.idx: to_pt(b.bbox) for b in raw.blocks}
    union = dict(boxes)                                   # 병합 묶음 bbox = 구성원 bbox 의 합집합
    for b in raw.blocks:
        if b.merged_into is not None:
            a, c = union[b.merged_into], boxes[b.idx]
            union[b.merged_into] = (min(a[0], c[0]), min(a[1], c[1]), max(a[2], c[2]), max(a[3], c[3]))

    table_names = {path: v["name"] for path, v in raw.table_imgs.items()}
    blocks = []
    for b in raw.blocks:
        vlm = b.vlm or {}
        figs = vlm.get("figures") or []                   # 표 안 그림의 자체 OCR 결과 — 오류·잘림을 표 블록으로 올린다
        fig_error = next((f["error"] for f in figs if f.get("error")), None)
        blk = Block(
            id=block_id(inp.page_no, b.idx),
            type=b.label,
            bbox=union[b.idx],
            score=_r(b.score) if b.score is not None else None,
            img_path=f"imgs/{b.img}" if b.img else None,
            error=vlm.get("error") or (f"표 안 그림: {fig_error}" if fig_error else None),
        )
        if vlm.get("skipped"):                            # 훅 C 가 VLM 을 건너뛴 블록 — 글자는 조립 때 레이어로 채운다
            blk.provenance = [{"rule": "vlm_skipped", "reason": vlm["skipped"]}]
        if b.merged_into is not None:
            blk.merged_into = block_id(inp.page_no, b.merged_into)
            blk.keep, blk.drop_reason = False, "merged"
        elif b.label == "table":
            blk.html = fix_table_imgs(b.content, table_names) or None
            photos = {table_names[f["path"]]: f.get("raw") for f in figs if f.get("path") in table_names}
            if blk.html and photos:
                blk.html, texts = split_photo_text(blk.html, photos)
                blk.figure_text = "\n".join(texts) or None
            lost = sorted(v["name"] for v in raw.table_imgs.values()     # 표 안에 있는데 HTML 에 자리 표시가 없는 사진
                          if not v.get("in_html") and _inside(v["box"], b.bbox))
            if lost:
                blk.provenance = [{"rule": "table_img_missing", "names": lost}]
        elif b.label in FIGURE_LABELS:
            blk.figure_text = b.content or None
        else:
            blk.text = b.content
        flags = [vlm, *figs]
        if any(f.get("finish_reason") == "length" for f in flags):
            blk.truncated = "length"
        elif any(f.get("repetition_truncated") for f in flags):
            blk.truncated = "repetition"
        blocks.append(blk)

    return Page(
        page_no=inp.page_no,
        sheet_no=inp.sheet_no,
        size=tuple(_r(v) for v in inp.sheet_size),
        n_up=NUp(**inp.n_up) if inp.n_up else None,
        blocks=blocks,
    )
