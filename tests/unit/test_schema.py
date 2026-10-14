import json

import jsonschema
from typer.testing import CliRunner

from kdp.cli import app
from kdp.schema import Block, Document, Page, ParserInfo, Source, json_schema


def _doc():
    return Document(
        doc_id="sha256:ab",
        source=Source(paths=["a.pdf"], pages=1),
        parser=ParserInfo(kdp_version="0", engine={}, engine_hash="e", assemble_hash="a", run_at="t"),
        pages=[Page(page_no=1, sheet_no=1, size=(595, 842), blocks=[
            Block(id="p1.b01", type="text", bbox=(1, 2, 3, 4), text="본문", order=1),
            Block(id="p1.b02", type="number", bbox=(1, 2, 3, 4), text="- 1 -", keep=False, drop_reason="page_number"),
        ])],
    )


def test_document_validates_against_schema():
    jsonschema.validate(_doc().model_dump(mode="json", exclude_none=True), json_schema())


def test_schema_cli():
    r = CliRunner().invoke(app, ["schema"])
    assert r.exit_code == 0
    assert json.loads(r.output)["title"] == "Document"
