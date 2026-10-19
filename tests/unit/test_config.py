import pytest
from pydantic import ValidationError

from kdp.config import DEFAULT_PARSE, ParseConfig, assemble_hash, engine_hash, load_settings


def test_defaults_load(settings, tmp_path):
    assert settings.parse.text_layer.mode == "prefer"       # D13 — M2 에서 켬
    assert settings.paddlex_config_path.exists()
    assert settings.env.vlm.backend in {"native", "vllm-server"}
    f = tmp_path / "off.yaml"
    f.write_text('text_layer: { mode: "off" }\n', encoding="utf-8")
    assert load_settings(parse_path=f).parse.text_layer.mode == "off"   # YAML 의 off 가 False 로 바뀌지 않는다


def test_unknown_key_rejected():
    with pytest.raises(ValidationError):
        ParseConfig(**{"render": {"dpi": 144, "dpii": 200}})


def test_hash_split(settings):
    desc = {"name": "x"}
    other = settings.model_copy(deep=True)
    other.parse.classify.page_drop_min_score = 0.5
    assert engine_hash(settings, desc) == engine_hash(other, desc)
    assert assemble_hash(settings, "1") != assemble_hash(other, "1")
    other.parse.render.dpi = 200
    assert engine_hash(settings, desc) != engine_hash(other, desc)


def test_explicit_missing_env_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "nope.yaml")


def test_partial_override_keeps_default_paddlex_config(tmp_path):
    f = tmp_path / "hi_dpi.yaml"
    f.write_text("render: {dpi: 200}\n", encoding="utf-8")
    s = load_settings(parse_path=f)
    assert s.parse.render.dpi == 200
    assert s.paddlex_config_path == (DEFAULT_PARSE.parent / "paddlex_vl16.yaml").resolve()


def test_figures_only_config_differs_from_default_in_two_places():
    """그림 해석 학습 자료 준비용 설정(기획서 7.10) — 그림 쪽 처리가 실제 파싱과 같으려면 default.yaml 과 두 곳만 달라야 한다."""
    import yaml

    base = yaml.safe_load(DEFAULT_PARSE.read_text(encoding="utf-8"))
    fig = yaml.safe_load((DEFAULT_PARSE.parent / "figures_only.yaml").read_text(encoding="utf-8"))
    assert fig["paddle_vl"].pop("vlm_skip") == base["paddle_vl"].pop("vlm_skip") + ["non_figure"]
    assert fig["describe"].pop("enabled") is False and base["describe"].pop("enabled") is True
    assert fig == base
    assert load_settings(parse_path=DEFAULT_PARSE.parent / "figures_only.yaml").parse.paddle_vl.vlm_skip[-1] == "non_figure"
