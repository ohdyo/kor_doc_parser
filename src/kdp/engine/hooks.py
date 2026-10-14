"""PaddleX 내부에 거는 훅(설계 3.2). paddlex==3.7.2 에 맞춘 것이라 버전을 올리면 계약 테스트부터 본다."""
from __future__ import annotations

import threading

import openai
from paddlex.inference.pipelines.paddleocr_vl import pipeline as _vl
from paddlex.inference.pipelines.paddleocr_vl.uilts import truncate_repetitive_content

BLOCK_ERROR_CODES = {400, 413, 422}       # 그 블록만의 문제(너무 긴 입력 등)
_local = threading.local()


def block_error_reason(e: Exception) -> str | None:
    """블록 오류면 사유를, 서버·연결 오류면 None — 그때는 삼키지 않고 문서 오류로 올린다."""
    if isinstance(e, openai.APIStatusError) and e.status_code in BLOCK_ERROR_CODES:
        return f"{e.status_code}: {str(e)[:200]}"
    return None


def install_response_hook(predictor) -> bool:
    """훅 A: 응답마다 finish_reason·usage 를 남기고 블록 오류는 빈 결과로 바꾼다. 서버 백엔드에만 걸린다."""
    if not hasattr(predictor, "_doc_vlm_genai_collect_responses"):
        return False

    def collect(futures):
        texts, metas = [], []
        for f in futures:
            try:
                r = f.result()
            except Exception as e:
                reason = block_error_reason(e)
                if reason is None:
                    raise
                texts.append("")
                metas.append({"error": reason})
                continue
            ch = r.choices[0]
            texts.append(ch.message.content or "")
            metas.append({"finish_reason": ch.finish_reason,
                          "usage": r.usage.model_dump() if r.usage else None})
        _local.metas = metas
        return texts

    orig_process = predictor.process

    def process(data, **kw):
        _local.metas = None
        out = orig_process(data, **kw)
        out["kdp_meta"] = _local.metas or [None] * len(data)
        return out

    predictor._doc_vlm_genai_collect_responses = collect
    predictor.process = process
    return True


class KdpPipeline(_vl._PaddleOCRVLPipeline):
    """훅 B: 블록마다 VLM 원문·finish_reason·반복 절단 여부·병합 대표를 붙인다(obj.kdp, obj.kdp_merged_into)."""

    def _paddleocr_vl_assemble_parsing_results(self, blocks, batch_dict_by_pixel, id2pixel_key_map,
                                               drop_figures_set, vis_image_labels):
        out = super()._paddleocr_vl_assemble_parsing_results(
            blocks, batch_dict_by_pixel, id2pixel_key_map, drop_figures_set, vis_image_labels)
        results = {}
        for info in batch_dict_by_pixel.values():
            results.update(zip(info["vlm_block_ids"], info["vlm_results"]))
        for i, page_blocks in enumerate(blocks):
            objs = iter(out[0][i])
            j2out, leaders = {}, {}
            for j, blk in enumerate(page_blocks):
                # 표 안 그림은 원본 메서드도 건너뛴다
                if (blk["label"] in vis_image_labels and blk["img"] is not None
                        and _vl.construct_img_path(blk["label"], blk["box"]) in drop_figures_set):
                    continue
                obj = next(objs)
                j2out[j] = len(j2out)
                obj.kdp_merged_into = None
                gid = blk.get("group_id")
                if "merge_aligns" in blk and gid is not None:     # merge_blocks 가 묶은 블록
                    if blk["img"] is not None:
                        leaders[gid] = j
                    elif gid in leaders:
                        obj.kdp_merged_into = j2out[leaders[gid]]
                r = results.get((i, j))
                if r is None:
                    obj.kdp = None
                    continue
                raw = r.get("result") or ""
                min_count = 5000 if blk["label"] == "table" else 50
                obj.kdp = {"raw": raw,
                           "repetition_truncated": truncate_repetitive_content(raw, min_count=min_count) != raw,
                           **(r.get("kdp_meta") or {})}
        return out
