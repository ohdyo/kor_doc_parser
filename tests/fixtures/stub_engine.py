"""모델 없는 엔진 — 쪽마다 정해 둔 블록을 내고 raw/ 를 실제 엔진과 같은 형식으로 쓴다."""
from __future__ import annotations

from dataclasses import asdict

from kdp.engine.base import page_stem, raw_page_from_files
from kdp.io.write import write_json, write_png


class StubEngine:
    def __init__(self, fail_after: int | None = None):
        self.fail_after = fail_after

    def describe(self) -> dict:
        return {"name": "stub", "code_version": "1"}

    def run(self, pages, raw_dir):
        for n, (img, inp) in enumerate(pages):
            if self.fail_after is not None and n >= self.fail_after:
                raise RuntimeError("stub engine failure")
            h, w = img.shape[:2]
            spec = [
                ("header", (0.1, 0.02, 0.6, 0.05), "사업 계획 보고서"),
                ("paragraph_title", (0.1, 0.08, 0.5, 0.11), f"{inp.page_no}. 추진 배경"),
                ("text", (0.1, 0.15, 0.9, 0.40), "본문 문단입니다."),
                ("table", (0.1, 0.45, 0.9, 0.70), "<table><tr><td>구분</td><td>금액</td></tr></table>"),
                ("number", (0.48, 0.95, 0.52, 0.97), f"- {inp.page_no} -"),
            ]
            blocks = [(label, [round(x1 * w), round(y1 * h), round(x2 * w), round(y2 * h)], text)
                      for label, (x1, y1, x2, y2), text in spec]
            page_json = {
                "parsing_res_list": [{"block_label": l, "block_bbox": b, "block_content": c} for l, b, c in blocks],
                "layout_det_res": {"boxes": [{"label": l, "coordinate": b, "score": 0.9} for l, b, _ in blocks]},
            }
            table_img = f"p{inp.page_no}.b04.png"
            x1, y1, x2, y2 = blocks[3][1]
            write_png(raw_dir / "imgs" / table_img, img[y1:y2, x1:x2])
            sidecar = {
                "input": asdict(inp), "width": w, "height": h, "ink_ratio": 0.05, "table_imgs": {},
                "blocks": [{"idx": i, "vlm": {"finish_reason": "stop"}, "merged_into": None,
                            "img": table_img if label == "table" else None}
                           for i, (label, _, _) in enumerate(blocks)],
            }
            write_json(raw_dir / f"{page_stem(inp.page_no)}.json", page_json)
            write_json(raw_dir / f"{page_stem(inp.page_no)}.sidecar.json", sidecar)
            yield raw_page_from_files(page_json, sidecar)

    def close(self) -> None:
        pass
