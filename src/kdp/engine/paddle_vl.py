"""1단계 엔진: 설치본 PaddleOCR-VL-1.6 + 훅(설계 3장)."""
from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import numpy as np

from ..config import Settings, apply_env, digest
from ..io.ids import block_id
from ..io.write import write_json, write_png
from .base import PageInput, RawPage, page_stem, raw_page_from_files

CODE_VERSION = "1"
CROP_LABELS = {"table", "chart", "display_formula"}   # PaddleX 가 크롭을 안 주는 것 — 렌더에서 직접 자른다


class EngineError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _wrapper_class():
    from paddleocr import PaddleOCRVL
    from paddleocr._common_args import prepare_common_init_args

    from .hooks import KdpPipeline

    class KdpOCRVL(PaddleOCRVL):
        """래퍼의 설정 병합·predict_iter·close 는 그대로 쓰고 파이프라인만 KdpPipeline 으로."""

        def _create_paddlex_pipeline(self):
            cfg = dict(self._merged_paddlex_config)
            cfg.pop("use_hpip", None)
            cfg.pop("hpi_config", None)
            return KdpPipeline(cfg, **prepare_common_init_args(None, self._common_args))

    return KdpOCRVL


def ink_ratio(img: np.ndarray) -> float:
    return round(float((img[::4, ::4].min(axis=2) < 200).mean()), 4)


def _crop(img: np.ndarray, box) -> np.ndarray | None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    return img[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None


def _weights_fingerprint(model_dir: str | None, model_name: str) -> dict:
    """가중치 판 — 경로·파일 크기·수정 시각(대용량이라 내용 해시는 하지 않는다)."""
    base = Path(os.environ.get("PADDLE_PDX_CACHE_HOME", Path.home() / ".paddlex"))
    d = Path(model_dir) if model_dir else base / "official_models" / model_name
    if not d.is_dir():
        return {"path": str(d), "missing": True}
    files = sorted((p.name, p.stat().st_size, int(p.stat().st_mtime)) for p in d.iterdir() if p.is_file())
    return {"path": str(d), "files": digest(files)}


class PaddleVLEngine:
    def __init__(self, settings: Settings):
        apply_env(settings.env)
        env, p = settings.env, settings.parse.paddle_vl
        kw = dict(
            pipeline_version=p.pipeline_version,
            paddlex_config=str(settings.paddlex_config_path),
            device=env.device,
            vl_rec_backend=env.vlm.backend,
            layout_detection_model_dir=env.models.layout_dir,
            vl_rec_model_dir=env.models.vlm_dir,
            use_doc_orientation_classify=False,     # 방향은 prep 에서 직접(설계 3.7)
            use_doc_unwarping=False,
            use_chart_recognition=p.use_chart_recognition,
            use_seal_recognition=p.use_seal_recognition,
            use_ocr_for_image_block=p.use_ocr_for_image_block,
        )
        if env.vlm.backend != "native":
            kw.update(vl_rec_server_url=env.vlm.server_url,
                      vl_rec_api_model_name=env.vlm.api_model_name,
                      vl_rec_api_key=env.vlm.api_key,
                      vl_rec_max_concurrency=env.vlm.max_concurrency)
        from .hooks import install_response_hook

        self._ocr = _wrapper_class()(**kw)
        self.hooked = install_response_hook(self._ocr.paddlex_pipeline.vl_rec_model)
        self._settings = settings
        self._snapshot = self._config_snapshot()

    def _config_snapshot(self) -> str:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "paddlex.yaml"
            self._ocr.export_paddlex_config_to_yaml(path)
            return path.read_text(encoding="utf-8")

    def config_snapshot(self) -> str:
        return self._snapshot

    def describe(self) -> dict:
        env, p = self._settings.env, self._settings.parse.paddle_vl
        return {
            "name": "paddle_vl",
            "code_version": CODE_VERSION,
            "pipeline_version": p.pipeline_version,
            "layout_model": "PP-DocLayoutV3",
            "vlm": env.vlm.api_model_name if env.vlm.backend != "native" else "PaddleOCR-VL-1.6-0.9B",
            "backend": env.vlm.backend,
            "response_hook": self.hooked,
            "paddleocr": version("paddleocr"),
            "paddlex": version("paddlex"),
            "paddlex_config": digest(self._snapshot),
            "weights": {
                "layout": _weights_fingerprint(env.models.layout_dir, "PP-DocLayoutV3"),
                "vlm": (_weights_fingerprint(env.models.vlm_dir, "PaddleOCR-VL-1.6")
                        if env.vlm.backend == "native" else {"served": env.vlm.api_model_name}),
            },
        }

    def run(self, pages, raw_dir: Path):
        raw_dir = Path(raw_dir)
        size = self._settings.parse.paddle_vl.page_batch
        batch = []
        for item in pages:
            batch.append(item)
            if len(batch) == size:
                yield from self._run_batch(batch, raw_dir)
                batch = []
        if batch:
            yield from self._run_batch(batch, raw_dir)

    def _run_batch(self, batch, raw_dir: Path):
        images = [img for img, _ in batch]          # 반드시 list — 튜플·제너레이터는 결과가 0건이다
        n = 0
        results = self._ocr.predict_iter(images, max_new_tokens=self._settings.parse.paddle_vl.max_new_tokens)
        for res, (img, inp) in zip(results, batch):
            n += 1
            yield self._save(res, img, inp, raw_dir)
        if n != len(batch):
            raise EngineError(f"엔진 결과 {n}건 / 입력 {len(batch)}쪽")

    def _save(self, res, img: np.ndarray, inp: PageInput, raw_dir: Path) -> RawPage:
        page_json = res.json
        page_json = page_json.get("res", page_json)
        imgs_dir = raw_dir / "imgs"
        blocks = res["parsing_res_list"]
        meta = []
        for i, b in enumerate(blocks):
            name = None
            image = getattr(b, "image", None)
            crop = image.get("img") if image else None
            if crop is None and b.label in CROP_LABELS:
                crop = _crop(img, b.bbox)
            if crop is not None:
                name = f"{block_id(inp.page_no, i)}.png"
                write_png(imgs_dir / name, crop)
            meta.append({"idx": i, "vlm": getattr(b, "kdp", None),
                         "merged_into": getattr(b, "kdp_merged_into", None), "img": name})
        table_imgs = {}
        tables = " ".join(b.content or "" for b in blocks if b.label == "table")
        for k, d in enumerate(res.get("imgs_in_doc") or []):
            if d["path"] in tables:
                name = f"p{inp.page_no}.t{k + 1:02d}.png"
                write_png(imgs_dir / name, d["img"])
                table_imgs[d["path"]] = name
        sidecar = {"input": asdict(inp), "width": int(img.shape[1]), "height": int(img.shape[0]),
                   "ink_ratio": ink_ratio(img), "blocks": meta, "table_imgs": table_imgs}
        stem = page_stem(inp.page_no)
        write_json(raw_dir / f"{stem}.json", page_json)
        write_json(raw_dir / f"{stem}.sidecar.json", sidecar)
        return raw_page_from_files(page_json, sidecar)

    def close(self) -> None:
        self._ocr.close()
        if self._settings.env.vlm.backend != "native":
            from paddlex.inference.models.common.genai import close_aio_loop

            close_aio_loop()        # 프로세스 종료 때 이벤트 루프 정리 오류가 나지 않게 미리 닫는다
