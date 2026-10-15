"""실제 검출기(PP-DocLayoutV3, CPU) + 가짜 VLM 서버로 전체 흐름, parse 와 reassemble 출력이 같은지."""
import json
import socket

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
    s.parse.paddle_vl.vlm_skip = ["logo"]           # 가짜 VLM 글자는 레이어와 달라 검증 표본이 맞지 않는다 — 훅 C 는 계약 테스트에서
    return s


@pytest.fixture()
def no_internet(monkeypatch):
    """이 컴퓨터 밖으로 나가는 연결을 막고 시도를 기록한다 — 실행 중 외부 다운로드가 없어야 한다(설계 9장)."""
    attempts = []
    local = ("localhost", "127.", "::1")
    real_getaddrinfo, real_connect = socket.getaddrinfo, socket.socket.connect

    def getaddrinfo(host, *a, **k):
        if host and not str(host).startswith(local):
            attempts.append(str(host))
            raise OSError("외부 연결 차단(테스트)")
        return real_getaddrinfo(host, *a, **k)

    def connect(self, address):
        if isinstance(address, tuple) and not str(address[0]).startswith(local):
            attempts.append(str(address[0]))
            raise OSError("외부 연결 차단(테스트)")
        return real_connect(self, address)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", connect)
    return attempts


def test_parse_then_reassemble(tmp_path, korean_font, fake_settings, no_internet):
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
    assert no_internet == []                                         # 글꼴·모델 등 외부 다운로드 시도 없음
