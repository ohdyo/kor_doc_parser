"""kdp 명령(설계 8장): doctor · parse · reassemble · schema. batch·qc·export-ls 는 M4."""
from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional

import typer

from .config import apply_env, load_settings

app = typer.Typer(add_completion=False, help="한국어 PDF → RAG 정형 데이터 파서")

EnvOpt = typer.Option(None, "--env", help="머신별 설정(기본 configs/env.yaml)")
CfgOpt = typer.Option(None, "--config", help="동작 설정(기본 configs/parse/default.yaml)")

PINS = {"paddleocr": "3.7.0", "paddlex": "3.7.2", "pypdfium2": "5.14.0"}
EXCLUDED = ["PyMuPDF", "mineru", "docling"]            # 파서 env 에는 넣지 않는다(설계 9·11장)


def _version(pkg: str) -> str | None:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


@app.command()
def doctor(env: Optional[Path] = EnvOpt, config: Optional[Path] = CfgOpt):
    """환경 점검: 버전 고정·배제 패키지·장치·모델·VLM 서버."""
    s = load_settings(env, config)
    apply_env(s.env)
    rows = []
    for pkg, want in PINS.items():
        have = _version(pkg)
        rows.append(("OK" if have == want else "FAIL", f"{pkg}=={want}", f"설치본 {have}"))
    for pkg in EXCLUDED:
        if _version(pkg):
            rows.append(("WARN", f"{pkg} 설치됨", "파서 env 에는 넣지 않는다"))

    if s.env.device.startswith("gpu"):
        import paddle

        n = paddle.device.cuda.device_count() if paddle.device.is_compiled_with_cuda() else 0
        rows.append(("OK" if n else "FAIL", "GPU", f"{n}개"))
    else:
        rows.append(("INFO", "장치", s.env.device))

    from .engine.paddle_vl import _weights_fingerprint

    for name, d in (("PP-DocLayoutV3", s.env.models.layout_dir), ("PaddleOCR-VL-1.6", s.env.models.vlm_dir)):
        if name.startswith("PaddleOCR-VL") and s.env.vlm.backend != "native":
            continue
        fp = _weights_fingerprint(d, name)
        rows.append(("WARN" if fp.get("missing") else "OK", f"가중치 {name}", fp["path"]))

    if s.env.vlm.backend != "native":
        rows.append(("OK" if _version("openai") else "FAIL", "openai(서버 백엔드)", f"설치본 {_version('openai')}"))
        from .engine.paddle_vl import served_models

        try:
            served = served_models(s.env.vlm.server_url, s.env.vlm.api_key)
            ok = s.env.vlm.api_model_name in served
            rows.append(("OK" if ok else "FAIL", "VLM 모델 이름", f"요청 {s.env.vlm.api_model_name} / 서버 {list(served)}"))
        except OSError as e:
            rows.append(("FAIL", "VLM 서버", f"{s.env.vlm.server_url} — {e}"))

    font = s.env.test.korean_font
    rows.append(("OK" if font and Path(font).exists() else "WARN", "테스트 한글 글꼴", str(font)))

    for status, item, detail in rows:
        typer.echo(f"[{status:4}] {item}  {detail}")
    raise typer.Exit(1 if any(r[0] == "FAIL" for r in rows) else 0)


@app.command()
def parse(pdf: Path, out: Optional[Path] = typer.Option(None, "-o", "--out", help="출력 폴더(기본 env.yaml paths.out_dir)"),
          env: Optional[Path] = EnvOpt, config: Optional[Path] = CfgOpt):
    """PDF 한 건을 처리한다."""
    from .io.ids import doc_dirname
    from .pipeline import parse_pdf

    s = load_settings(env, config)
    doc = parse_pdf(pdf, out, s)
    out_dir = Path(out or s.env.paths.out_dir) / doc_dirname(doc.doc_id)
    typer.echo(f"{doc.doc_id}  qc={doc.qc.status}  → {out_dir}")


@app.command()
def reassemble(path: Path, env: Optional[Path] = EnvOpt, config: Optional[Path] = CfgOpt):
    """raw/ 로 판정·조립만 다시 한다. 문서 폴더 하나 또는 그것들이 든 출력 폴더."""
    from .pipeline import reassemble as _reassemble

    s = load_settings(env, config)
    dirs = [path] if (path / "raw").is_dir() else sorted(p for p in path.iterdir() if (p / "raw").is_dir())
    for d in dirs:
        doc = _reassemble(d, s)
        typer.echo(f"{doc.doc_id}  qc={doc.qc.status}")


@app.command()
def schema():
    """정본 JSON Schema 를 출력한다."""
    from .schema import json_schema

    typer.echo(json.dumps(json_schema(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
