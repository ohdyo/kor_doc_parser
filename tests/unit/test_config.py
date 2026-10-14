import pytest
from pydantic import ValidationError

from kdp.config import ParseConfig, assemble_hash, engine_hash, load_settings


def test_defaults_load(settings):
    assert settings.parse.text_layer.mode == "off"          # YAML 의 off 가 False 로 바뀌지 않는다
    assert settings.paddlex_config_path.exists()
    assert settings.env.vlm.backend in {"native", "vllm-server"}


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
