"""raw/ 를 다시 읽어 같은 RawPage 를 돌려준다 — 재조립·테스트용(GPU·원본 PDF 불필요)."""
from __future__ import annotations

from pathlib import Path

from ..io.write import read_json
from .base import raw_page_from_files


class ReplayEngine:
    def __init__(self, raw_dir: str | Path):
        self.raw_dir = Path(raw_dir)

    def describe(self) -> dict:
        return read_json(self.raw_dir / "engine.json")["engine"]

    def run(self, pages=None, raw_dir=None):
        for side in sorted(self.raw_dir.glob("p*.sidecar.json")):
            stem = side.name.split(".")[0]
            yield raw_page_from_files(read_json(self.raw_dir / f"{stem}.json"), read_json(side))

    def close(self) -> None:
        pass
