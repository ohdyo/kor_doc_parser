"""1단계 엔진: 설치본 PaddleOCR-VL-1.6 + 훅(설계 3장)."""
from __future__ import annotations

import json
import os
import tempfile
import urllib.request
from dataclasses import asdict
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml

from ..config import Settings, apply_env, digest
from ..io.ids import block_id
from ..io.render import user_box_to_display
from ..io.write import read_json_gz, write_json, write_png
from ..prep import nup, orient
from ..textlayer.extract import MIN_PAGE_CHARS, plausible, visible
from .base import PageInput, RawPage, page_stem, raw_page_from_files
from .skip import SkipPlan

CODE_VERSION = "7"                                     # 7: 방향 모름 모아찍기(기운 방향 + 칸 아래 쪽번호)·쌓인 쪽(글자·빈 띠)·쪽번호 값(이어지는 쌍)
CROP_LABELS = {"table", "chart", "display_formula"}   # PaddleX 가 크롭을 안 주는 것 — 렌더에서 직접 자른다
PIPELINE_NAMES = {"v1": "PaddleOCR-VL", "v1.5": "PaddleOCR-VL-1.5", "v1.6": "PaddleOCR-VL-1.6"}


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


def wrapper_kwargs(settings: Settings) -> dict:
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
    return kw


def served_models(url: str | None, api_key: str | None) -> dict:
    """VLM 서버가 올린 모델 — vLLM 은 root(가중치 경로)·max_model_len 도 준다."""
    req = urllib.request.Request((url or "").rstrip("/") + "/models",
                                 headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.load(r)["data"]
    return {m["id"]: {"root": m.get("root"), "max_model_len": m.get("max_model_len")} for m in data}


def ink_ratio(img: np.ndarray) -> float:
    return round(float((img[::4, ::4].min(axis=2) < 200).mean()), 4)


def _crop(img: np.ndarray, box) -> np.ndarray | None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    return img[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None


def _inside(box, outer) -> bool:
    return box[0] >= outer[0] and box[1] >= outer[1] and box[2] <= outer[2] and box[3] <= outer[3]


def _weights_fingerprint(model_dir: str | None, model_name: str) -> dict:
    """가중치 판 — 경로·파일 크기·수정 시각(대용량이라 내용 해시는 하지 않는다). '~' 경로도 불러 쓰는 쪽과 같게 펼친다."""
    base = Path(os.environ.get("PADDLE_PDX_CACHE_HOME", Path.home() / ".paddlex")).expanduser() / "official_models"
    candidates = [Path(model_dir).expanduser()] if model_dir else [base / model_name, base / model_name.rsplit("-", 1)[0]]
    d = next((c for c in candidates if c.is_dir()), candidates[0])
    if not d.is_dir():
        return {"path": str(d), "missing": True}
    files = sorted((p.name, p.stat().st_size, int(p.stat().st_mtime)) for p in d.iterdir() if p.is_file())
    return {"path": str(d), "files": digest(files)}


def _rules(mod) -> str:
    """prep 모듈 기준값(대문자 상수)의 해시 — 방향·모아찍기 기준을 바꾸면 CODE_VERSION 을 올리지 않아도 엔진 해시가 바뀐다."""
    return digest({k: v for k, v in sorted(vars(mod).items()) if k.isupper()})


class PaddleVLEngine:
    def __init__(self, settings: Settings, ocr=None):
        apply_env(settings.env)
        env, p = settings.env, settings.parse.paddle_vl
        self._settings = settings
        self._ocr = ocr or _wrapper_class()(**wrapper_kwargs(settings))
        from .hooks import install_response_hook

        self.hooked = install_response_hook(self._ocr.paddlex_pipeline.vl_rec_model)
        self._ocr.paddlex_pipeline.kdp_chart_ocr = p.use_ocr_for_chart_block
        cfg = self._ocr._merged_paddlex_config
        want = PIPELINE_NAMES.get(p.pipeline_version)
        if want and cfg.get("pipeline_name") != want:      # paddlex_config 를 주면 pipeline_version 은 무시되므로 맞는지 본다
            raise EngineError(f"pipeline_version {p.pipeline_version} 과 {settings.paddlex_config_path.name} 의 "
                              f"pipeline_name {cfg.get('pipeline_name')} 이 다르다")
        self._served = None
        if env.vlm.backend != "native":
            try:
                served = served_models(env.vlm.server_url, env.vlm.api_key)
            except OSError as e:
                raise EngineError(f"VLM 서버에 연결할 수 없다: {env.vlm.server_url} — {e}") from e
            if env.vlm.api_model_name not in served:
                raise EngineError(f"VLM 서버에 '{env.vlm.api_model_name}' 이 없다: {list(served)}")
            self._served = served[env.vlm.api_model_name]
        self._snapshot = self._config_snapshot()

    def _config_snapshot(self) -> str:
        """병합된 PaddleX 설정. 서버 주소·동시성·키는 출력과 상관없고 밖에 남으면 안 되므로 뺀다."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "paddlex.yaml"
            self._ocr.export_paddlex_config_to_yaml(path)
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        genai = cfg.get("SubModules", {}).get("VLRecognition", {}).get("genai_config") or {}
        genai.pop("server_url", None)
        genai.pop("max_concurrency", None)
        (genai.get("client_kwargs") or {}).pop("api_key", None)
        return yaml.safe_dump(cfg, allow_unicode=True, sort_keys=True)

    def config_snapshot(self) -> str:
        return self._snapshot

    def describe(self) -> dict:
        env = self._settings.env
        cfg = self._ocr._merged_paddlex_config
        layout = cfg["SubModules"]["LayoutDetection"]["model_name"]
        vlm = cfg["SubModules"]["VLRecognition"]["model_name"]
        native = env.vlm.backend == "native"
        return {
            "name": "paddle_vl",
            "code_version": CODE_VERSION,
            "pipeline": cfg.get("pipeline_name"),
            "layout_model": layout,
            "vlm": vlm if native else env.vlm.api_model_name,
            "backend": env.vlm.backend,
            "response_hook": self.hooked,
            "vlm_skip": self._skip(),
            "paddleocr": version("paddleocr"),
            "paddlex": version("paddlex"),
            "paddlex_config": digest(self._snapshot),
            "weights": {
                "layout": _weights_fingerprint(env.models.layout_dir, layout),
                # 서버 백엔드도 서버가 읽는 폴더(models.vlm_dir)를 적으면 그 판이 들어간다 — 서빙 이름만으로는 가중치 교체를 모른다
                "vlm": _weights_fingerprint(env.models.vlm_dir, vlm) if native or env.models.vlm_dir else None,
                "served": self._served,
                "doc_ori": (_weights_fingerprint(str(orient.model_dir(env)), orient.MODEL)   # 방향 판정이 불러 쓰는 그 폴더
                            if self._settings.parse.orientation == "scan_only" else None),
            },
            "prep": {"orient": _rules(orient) if self._settings.parse.orientation == "scan_only" else None,
                     "nup": _rules(nup) if self._settings.parse.nup.enabled else None},
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

    def _skip(self) -> list[str]:
        """실제로 건너뛰는 것 — text_layer 는 레이어 우선(prefer)일 때만."""
        p = self._settings.parse
        return [k for k in p.paddle_vl.vlm_skip if k != "text_layer" or p.text_layer.mode == "prefer"]

    def _plan(self, inp: PageInput, raw_dir: Path) -> SkipPlan:
        chars = None
        f = raw_dir / "textlayer" / f"{page_stem(inp.sheet_no)}.json.gz"
        if "text_layer" in self._skip() and inp.user_bbox and not inp.rot90_k and f.exists():
            real = [(c, user_box_to_display(c["box"], inp.user_bbox, inp.rotation))
                    for c in read_json_gz(f) if not c["gen"] and not c["c"].isspace()]
            if inp.n_up:                                             # 모아찍기: 이 논리 페이지 영역의 글자만(textlayer 와 같은 기준)
                x0, y0, x1, y1 = inp.n_up["region"]
                real = [(c, b) for c, b in real if x0 <= (b[0] + b[2]) / 2 <= x1 and y0 <= (b[1] + b[3]) / 2 <= y1]
            seen = [(c, b) for c, b in real if visible(c)]           # 투명 OCR 레이어·가려진 글자는 뺀다
            if len(seen) >= MIN_PAGE_CHARS and len(seen) >= 0.5 * len(real) and plausible("".join(c["c"] for c, _ in seen)):
                (ox, oy), s = inp.origin, inp.scale
                chars = [(((l - ox) * s, (t - oy) * s, (r - ox) * s, (b - oy) * s), c["c"]) for c, (l, t, r, b) in seen]
        return SkipPlan("logo" in self._skip(), chars)

    def _run_batch(self, batch, raw_dir: Path):
        images = [img for img, _ in batch]          # 반드시 list — 튜플·제너레이터는 결과가 0건이다
        n = 0
        pipe = self._ocr.paddlex_pipeline
        pipe.kdp_plans = {id(img): self._plan(inp, raw_dir) for img, inp in batch}
        try:
            results = self._ocr.predict_iter(images, max_new_tokens=self._settings.parse.paddle_vl.max_new_tokens)
            for res, (img, inp) in zip(results, batch):
                n += 1
                yield self._save(res, img, inp, raw_dir)
        finally:
            pipe.kdp_plans = {}
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

        # 표 안 그림은 블록에서 빠진다 — HTML 에 다시 나오지 않아도 크롭은 모두 남긴다(설계 3.6)
        tables = [b for b in blocks if b.label == "table"]
        html = " ".join(b.content or "" for b in tables)
        table_imgs = {}
        for k, d in enumerate(res.get("imgs_in_doc") or []):
            if any(_inside(d["coordinate"], t.bbox) for t in tables):
                name = f"p{inp.page_no}.t{k + 1:02d}.png"
                write_png(imgs_dir / name, d["img"])
                table_imgs[d["path"]] = {"name": name, "label": d["label"],
                                         "box": [int(v) for v in d["coordinate"]], "in_html": d["path"] in html}

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
