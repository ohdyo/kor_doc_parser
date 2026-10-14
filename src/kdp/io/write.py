"""원자적 쓰기 — 임시 파일에 쓰고 바꿔 끼운다."""
from __future__ import annotations

import gzip
import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


def write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def dumps(obj, indent: int | None = None) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=indent)


def write_text(path, text: str) -> None:
    write_bytes(path, text.encode("utf-8"))


def write_json(path, obj, indent: int | None = None) -> None:
    write_text(path, dumps(obj, indent))


def write_json_gz(path, obj) -> None:
    write_bytes(path, gzip.compress(dumps(obj).encode("utf-8"), mtime=0))


def write_jsonl(path, rows) -> None:
    write_text(path, "".join(dumps(r) + "\n" for r in rows))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_json_gz(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()).decode("utf-8"))


def write_png(path, img) -> None:
    """PIL 이미지 또는 BGR 배열."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(np.ascontiguousarray(img[:, :, ::-1]) if img.ndim == 3 else img)
    buf = BytesIO()
    img.save(buf, format="PNG")
    write_bytes(path, buf.getvalue())
