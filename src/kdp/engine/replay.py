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
        # page_no 숫자 순 — 이름 글자 순이면 p10000 이 p1000 과 p1001 사이에 와서 parse 와 쪽 순서가 달라진다
        for stem in sorted((s.name.split(".")[0] for s in self.raw_dir.glob("p*.sidecar.json")), key=lambda s: int(s[1:])):
            yield raw_page_from_files(read_json(self.raw_dir / f"{stem}.json"), read_json(self.raw_dir / f"{stem}.sidecar.json"))

    def close(self) -> None:
        pass
