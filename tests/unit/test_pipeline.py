"""모델 없는 전체 흐름(StubEngine): 정본이 JSON Schema 를 통과하고, JSONL·imgs 가 정본과 맞고, 재조립이 같은 결과를 낸다."""
import json

import jsonschema
import pytest

from kdp.io.ids import doc_dirname
from kdp.io.write import read_json_gz
from kdp.pipeline import RawIncompleteError, parse_pdf, reassemble
from kdp.schema import json_schema
from tests.fixtures.stub_engine import StubEngine
from tests.fixtures.synth import nonbmp_pdf, nup_pdf, report_pdf, with_page_box


def _parse(tmp_path, data, engine=None, settings=None):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(data)
    doc = parse_pdf(pdf, tmp_path / "out", settings, engine=engine or StubEngine())
    return pdf, doc, tmp_path / "out" / doc_dirname(doc.doc_id)


def test_outputs_schema_jsonl_and_reassemble(tmp_path, korean_font, settings):
    pdf, doc, d = _parse(tmp_path, with_page_box(report_pdf(korean_font, 3), rotation=90), settings=settings)
    data = json.loads((d / "document.json").read_text(encoding="utf-8"))
    jsonschema.validate(data, json_schema())
    assert data["source"]["paths"] == [str(pdf.resolve())]
    assert data["pages"][0]["size"][0] > data["pages"][0]["size"][1]          # /Rotate 90 → 가로 시트

    blocks = [b for p in data["pages"] for b in p["blocks"]]
    assert all("order" in b for b in blocks) and all("n_up" in p for p in data["pages"])
    assert {b["drop_reason"] for b in blocks if not b["keep"]} == {"header", "page_number"}
    rows = [json.loads(line) for line in (d / "blocks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["block_id"] for r in rows] == [b["id"] for b in blocks if b["order"]]
    assert sorted(p.name for p in (d / "imgs").iterdir()) == ["p1.b04.png", "p2.b04.png", "p3.b04.png"]
    assert not (d / "raw.tmp").exists()

    before = {f: (d / f).read_bytes() for f in ("document.json", "blocks.jsonl")}
    reassemble(d, settings)
    assert {f: (d / f).read_bytes() for f in before} == before


def test_failed_run_keeps_previous_raw(tmp_path, korean_font, settings):
    data = report_pdf(korean_font, 3)
    _, _, d = _parse(tmp_path, data, settings=settings)
    snapshot = {p: p.read_bytes() for p in (d / "raw").rglob("*") if p.is_file()}
    with pytest.raises(RuntimeError, match="stub engine failure"):
        _parse(tmp_path, data, StubEngine(fail_after=1), settings)
    assert {p: p.read_bytes() for p in (d / "raw").rglob("*") if p.is_file()} == snapshot
    reassemble(d, settings)


def test_reassemble_refuses_missing_pages(tmp_path, korean_font, settings):
    _, _, d = _parse(tmp_path, report_pdf(korean_font, 3), settings=settings)
    (d / "raw" / "p0002.sidecar.json").unlink()
    with pytest.raises(RawIncompleteError):
        reassemble(d, settings)


def test_nonbmp_pdf_parses(tmp_path, settings):
    _, doc, d = _parse(tmp_path, nonbmp_pdf(), settings=settings)
    chars = read_json_gz(d / "raw" / "textlayer" / "p0001.json.gz")
    assert [c["c"] for c in chars] == ["\U0001F600", "\U00020000", "C"]
    assert doc.qc.status == "ok"


def test_reassemble_refuses_missing_logical_page(tmp_path, korean_font, settings):
    _, doc, d = _parse(tmp_path, nup_pdf(korean_font, 2, 2, 4), settings=settings)
    assert [p.page_no for p in doc.pages] == [1, 2, 3, 4] and {p.sheet_no for p in doc.pages} == {1}
    for f in ("p0002.json", "p0002.sidecar.json"):                    # 모아찍기 칸 하나만 빠져도 시트는 남는다
        (d / "raw" / f).unlink()
    with pytest.raises(RawIncompleteError):
        reassemble(d, settings)
