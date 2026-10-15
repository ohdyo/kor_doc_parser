"""실제 검출기(PP-DocLayoutV3, CPU) + 가짜 VLM 서버로 전체 흐름, parse 와 reassemble 출력이 같은지."""
import json

import jsonschema
import pytest

from kdp.config import apply_env
from kdp.engine.paddle_vl import _weights_fingerprint
from kdp.io.ids import doc_dirname
from kdp.pipeline import parse_pdf, reassemble
from kdp.schema import json_schema
from tests.fixtures.fake_vlm import FakeVLM
from tests.fixtures.synth import report_pdf


@pytest.fixture()
def fake_settings(settings):
    apply_env(settings.env)
    if _weights_fingerprint(settings.env.models.layout_dir, "PP-DocLayoutV3").get("missing"):
        pytest.skip("PP-DocLayoutV3 가중치가 없다")
    s = settings.model_copy(deep=True)
    s.env.device = "cpu"
    s.env.vlm.backend = "vllm-server"
    s.env.vlm.api_model_name = "served-name"
    s.env.vlm.max_concurrency = 4
    s.parse.paddle_vl.page_batch = 2
    return s


def test_parse_then_reassemble(tmp_path, korean_font, fake_settings):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(report_pdf(korean_font, pages=3))
    with FakeVLM() as vlm:
        fake_settings.env.vlm.server_url = vlm.url
        doc = parse_pdf(pdf, tmp_path / "out", fake_settings)
        assert vlm.requests and all(r["model"] == "served-name" for r in vlm.requests)

    d = tmp_path / "out" / doc_dirname(doc.doc_id)
    raw = d / "raw"
    for name in ("doc.json", "engine.json", "paddlex_config.yaml", "p0001.json", "p0001.sidecar.json",
                 "textlayer/p0001.json.gz", "p0003.sidecar.json"):
        assert (raw / name).exists(), name
    assert json.loads((raw / "engine.json").read_text())["engine"]["response_hook"] is True

    jsonschema.validate(json.loads((d / "document.json").read_text(encoding="utf-8")), json_schema())
    assert len(doc.pages) == 3 and all(p.blocks for p in doc.pages)
    kept = [b for p in doc.pages for b in p.blocks if b.order]
    assert kept and all(b.text or b.html or b.figure_text or b.img_path for b in kept)
    assert doc.qc.status in {"ok", "review"}

    before = {f: (d / f).read_bytes() for f in ("document.json", "blocks.jsonl")}
    reassemble(d, fake_settings)
    assert {f: (d / f).read_bytes() for f in before} == before
