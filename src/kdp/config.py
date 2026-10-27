"""설정: configs/env.yaml(머신별 값) + configs/parse/*.yaml(동작) → Settings."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV = ROOT / "configs" / "env.yaml"
DEFAULT_PARSE = ROOT / "configs" / "parse" / "default.yaml"


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")   # 오타 난 키는 바로 오류


# env.yaml
class VlmEnv(_Cfg):
    backend: str = "native"
    server_url: str | None = None
    api_model_name: str | None = None
    api_key: str | None = None
    max_concurrency: int = 32


class DescriberEnv(_Cfg):
    server_url: str | None = None        # 그림 설명 VLM 서버(OpenAI 호환 — D19)
    api_key: str | None = None
    max_concurrency: int = 8


class ModelsEnv(_Cfg):
    layout_dir: str | None = None
    vlm_dir: str | None = None
    doc_ori_dir: str | None = None


class PaddlexEnv(_Cfg):
    cache_home: str | None = None
    disable_model_source_check: bool = True


class PathsEnv(_Cfg):
    out_dir: str = "./out"


class LabelStudioEnv(_Cfg):
    url: str | None = None
    api_key: str | None = None


class TestEnv(_Cfg):
    korean_font: str | None = None


class Env(_Cfg):
    device: str = "cpu"
    vlm: VlmEnv = Field(default_factory=VlmEnv)
    describer: DescriberEnv = Field(default_factory=DescriberEnv)
    models: ModelsEnv = Field(default_factory=ModelsEnv)
    paddlex: PaddlexEnv = Field(default_factory=PaddlexEnv)
    paths: PathsEnv = Field(default_factory=PathsEnv)
    label_studio: LabelStudioEnv = Field(default_factory=LabelStudioEnv)
    test: TestEnv = Field(default_factory=TestEnv)


# parse yaml
class PaddleVL(_Cfg):
    pipeline_version: str = "v1.6"
    paddlex_config: str = "paddlex_vl16.yaml"
    max_new_tokens: int = 4096
    use_ocr_for_image_block: bool = True
    use_ocr_for_chart_block: bool = True  # D14 — 그래프·지도 속 보이는 글자도 그림처럼 'OCR:' 로(표로 읽지 않는다)
    use_chart_recognition: bool = False   # D14 — 그래프 표 읽기는 지어낸 값을 냈다(테스트_결과 6.5)
    use_seal_recognition: bool = False
    page_batch: int = 32
    vlm_skip: list[Literal["logo", "text_layer", "non_figure"]] = Field(default_factory=lambda: ["logo", "text_layer"])   # 훅 C
    # non_figure: 그림(image·chart) 밖은 VLM 을 건너뛴다 — 그림 해석 학습 자료 준비용(configs/parse/figures_only.yaml, 기획서 7.10)


class Render(_Cfg):
    dpi: int = 144
    max_pixels: int = 178_956_970


class Nup(_Cfg):
    enabled: bool = True


class TextLayer(_Cfg):
    mode: Literal["off", "prefer"] = "off"


class Classify(_Cfg):
    page_drop_min_score: float = 0.9
    aside_keep_min_sentences: int = 2
    footnote: Literal["keep", "drop"] = "keep"
    form_blank: Literal["keep_flag", "drop"] = "keep_flag"


class MetaCfg(_Cfg):
    promote: list[str] = Field(default_factory=lambda: ["title", "issuer", "doc_no", "date"])
    promote_person_fields: bool = False


class QCCfg(_Cfg):
    min_text_layer_agreement: float = 0.9
    max_drop_char_ratio: float = 0.4
    max_repeat_ngram_ratio: float = 0.05
    max_unsupported_char_ratio: float = 0.02
    max_block_errors: int = 0


class BatchCfg(_Cfg):
    stop_after_consecutive_failures: int = 20
    doc_retries: int = 1


class DescribeCfg(_Cfg):
    enabled: bool = False                # D19 — 그림과 함께 봐야 하는 그림(그림 갈래 image)을 범용 VLM 이 한국어로 설명한다
    model: str = "skt/A.X-4.0-VL-Light"  # 국내 모델(SK텔레콤, Apache-2.0) — 설명 서버가 올린 이름과 같게
    max_tokens: int = 512
    min_area: float = 0.01               # 논리 페이지 넓이 대비 — 이보다 작은 그림(아이콘)은 설명하지 않는다
    min_logprob: float | None = None     # 설명 문장 토큰의 평균 로그확률이 이보다 낮으면 뺀다(모델이 확신하지 못한 그림 해석) — 모델마다 잰다


class FigureOcrCfg(_Cfg):
    enabled: bool = False                # 그림 다시 읽기(테스트_결과 14.10) — 그림을 원본 PDF 에서 더 높은 dpi 로 다시 그려 그림 속 글자를 다시 읽는다
    model: str = "PaddlePaddle/PaddleOCR-VL-1.6"   # 엔진 VLM 서버(env.yaml vlm.server_url)가 올린 이름과 같게
    prompt: Literal["OCR:", "Spotting:"] = "OCR:"   # Spotting: 은 글자마다 위치를 함께 읽는 물음(작은 글자 — PaddleX 는 2배 키워 보낸다)
    types: list[Literal["image", "chart"]] = Field(default_factory=lambda: ["image", "chart"])   # 다시 읽을 블록 갈래 — 그림(image)
    #                                      갈래 지도·사진은 다시 읽은 줄이 24% 틀려(그래프 chart 는 6%) 기본 설정은 chart 만(테스트_결과 14.10)
    dpi: int = 288                       # 엔진 쪽 렌더(144)의 두 배 — 그림 원본 해상도가 높은 문서(test3·test5 323~600dpi)에서 글자가 또렷해진다
    pad_pt: float = 6.0                  # 그림 상자를 사방으로 넓힐 pt
    max_pixels: int = 1_605_632          # 서버에 보내는 화소 상한(PaddleOCR-VL Spotting 과 같다) — 넘으면 배율을 낮춰 그린다
    max_tokens: int = 1024               # 엔진 4096 보다 낮게 — 'OCR:' 정상 그림 읽기는 342토큰 안(실측), 넘으면 되풀이다. 'Spotting:' 은 글
    #                                      한 덩이마다 위치 토큰 11개가 더 들어(test4 10쪽 그래프 OCR: 630·Spotting: 1,676토큰) 3,072 쯤 준다
    repetition_penalty: float = 1.05     # 되풀이 벌점(엔진은 걸지 않아 4096토큰까지 되풀이했다)
    min_area: float = 0.01               # 논리 페이지 넓이 대비 — 이보다 작은 그림(아이콘)은 다시 읽지 않는다


class OutputCfg(_Cfg):
    write_markdown: bool = True
    write_rag_markdown: bool = True      # D21 — document.rag.md(앵커 없음, 표·그림 제목을 앞에, 긴 표는 머리행 되풀이 행 묶음)
    rag_table_chars: int = 400           # 이보다 긴 표는 행 묶음으로 — 받는 쪽 청크의 절반 안팎(LightRAG 512토큰 ≈ 파이프 표 800자)
    rag_table_format: Literal["html", "pipe"] = "pipe"   # 병합 칸을 푼 파이프 표(머리행 한 줄 — 사용자 결정 2026-10-26), html 이면 표 HTML 그대로
    write_jsonl: bool = True
    save_images: bool = True
    keep_raw: bool = True


class ParseConfig(_Cfg):
    engine: Literal["paddle_vl"] = "paddle_vl"
    paddle_vl: PaddleVL = Field(default_factory=PaddleVL)
    render: Render = Field(default_factory=Render)
    nup: Nup = Field(default_factory=Nup)
    orientation: Literal["off", "scan_only"] = "scan_only"
    text_layer: TextLayer = Field(default_factory=TextLayer)
    classify: Classify = Field(default_factory=Classify)
    meta: MetaCfg = Field(default_factory=MetaCfg)
    qc: QCCfg = Field(default_factory=QCCfg)
    describe: DescribeCfg = Field(default_factory=DescribeCfg)
    figure_ocr: FigureOcrCfg = Field(default_factory=FigureOcrCfg)
    batch: BatchCfg = Field(default_factory=BatchCfg)
    output: OutputCfg = Field(default_factory=OutputCfg)


class Settings(_Cfg):
    env: Env
    parse: ParseConfig
    paddlex_config_path: Path


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings(env_path: str | Path | None = None, parse_path: str | Path | None = None) -> Settings:
    env_file = Path(env_path) if env_path else DEFAULT_ENV
    parse_file = Path(parse_path) if parse_path else DEFAULT_PARSE
    if not env_file.exists():                      # 없으면 조용히 CPU 로 돌지 않게 바로 멈춘다
        raise FileNotFoundError(f"{env_file} 이 없다 — 머신별 값(장치·VLM 서버 등)을 이 파일에 둔다")
    env = Env(**_read_yaml(env_file))
    raw = _read_yaml(parse_file)
    parse = ParseConfig(**raw)
    # 이 파일이 paddlex_config 를 직접 정했을 때만 이 파일 기준, 아니면 기본 설정 폴더 기준
    base = parse_file.parent if "paddlex_config" in (raw.get("paddle_vl") or {}) else DEFAULT_PARSE.parent
    px = Path(parse.paddle_vl.paddlex_config)
    if not px.is_absolute():
        px = (base / px).resolve()
    if not px.exists():
        raise FileNotFoundError(px)
    return Settings(env=env, parse=parse, paddlex_config_path=px)


def apply_env(env: Env) -> None:
    """PaddleX 는 import 할 때 환경 변수를 읽으므로 그 전에 부른다."""
    if env.paddlex.cache_home:
        os.environ["PADDLE_PDX_CACHE_HOME"] = str(Path(env.paddlex.cache_home).expanduser())
    if env.paddlex.disable_model_source_check:
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


def digest(obj) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def engine_hash(s: Settings, engine_desc: dict) -> str:
    """같으면 GPU 를 다시 돌리지 않는다(설계 7장)."""
    p = s.parse
    return digest({
        "render": p.render.model_dump(),
        "nup": p.nup.model_dump(),
        "orientation": p.orientation,
        "paddle_vl": p.paddle_vl.model_dump(exclude={"page_batch", "paddlex_config"}),
        "engine": engine_desc,
    })


def assemble_hash(s: Settings, code_version: str) -> str:
    """이것만 바뀌면 kdp reassemble 로 충분하다(그림 다시 읽기·설명 설정을 바꾸면 kdp figure-ocr·describe 로 빠진 기록을 채운다)."""
    p = s.parse
    return digest({
        "text_layer": p.text_layer.model_dump(),
        "classify": p.classify.model_dump(),
        "meta": p.meta.model_dump(),
        "qc": p.qc.model_dump(),
        "describe": p.describe.model_dump(),
        "figure_ocr": p.figure_ocr.model_dump(),
        "code": code_version,
    })
