"""두 kdp 출력 폴더의 정본(document.json)을 블록마다 견준다 — 조립 규칙을 바꾼 뒤 무엇이 바뀌었는지(바뀐 블록·칸)만 보인다.

    python eval/block_diff.py <기준 출력 폴더> <새 출력 폴더> [--out block_diff.md]

문서 폴더 이름(문서 해시)이 같은 것끼리, 블록 id 가 같은 것끼리 견준다. 문서 머리(판·해시·QC)는 바뀐 칸 이름만, 파생물(document.md·
document.rag.md·blocks.jsonl)은 바이트가 같은지만 적는다. 첫 줄은 요약이다(서버 진행 기록에 그대로 옮긴다)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FILES = ("document.md", "document.rag.md", "blocks.jsonl")


def _short(v) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    s = " / ".join(s.split("\n"))
    return s if len(s) <= 90 else s[:87] + "…"


def diff(old: Path, new: Path) -> tuple[list[str], int]:
    """(바뀐 곳 줄, 바뀐 블록 수)."""
    out, changed = [], 0
    for d in sorted(p for p in new.iterdir() if (p / "document.json").exists()):
        o = old / d.name
        if not (o / "document.json").exists():
            out.append(f"- {d.name[:8]}: 기준 출력이 없다")
            continue
        a = json.loads((o / "document.json").read_text(encoding="utf-8"))
        b = json.loads((d / "document.json").read_text(encoding="utf-8"))
        head = sorted(k for k in a.keys() | b.keys() if k != "pages" and a.get(k) != b.get(k))
        if a.get("qc") != b.get("qc"):
            head.append(f"qc {a['qc'].get('status')}{a['qc'].get('reasons')} → {b['qc'].get('status')}{b['qc'].get('reasons')}")
        same = [f for f in FILES if (o / f).exists() and (d / f).exists() and (o / f).read_bytes() == (d / f).read_bytes()]
        out.append(f"## {d.name[:8]} — 머리 {', '.join(head) or '같음'} · 파생물 같음: {', '.join(same) or '없음'}")
        blocks_a = {x["id"]: x for p in a["pages"] for x in p["blocks"]}
        blocks_b = {x["id"]: x for p in b["pages"] for x in p["blocks"]}
        for bid in [*blocks_a, *(k for k in blocks_b if k not in blocks_a)]:
            x, y = blocks_a.get(bid), blocks_b.get(bid)
            if x == y:
                continue
            changed += 1
            if x is None or y is None:
                out.append(f"- {bid}: {'새 블록' if x is None else '없어진 블록'}")
                continue
            out += [f"- {bid} {k}: {_short(x.get(k))} → {_short(y.get(k))}" for k in sorted(x.keys() | y.keys()) if x.get(k) != y.get(k)]
    return out, changed


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="두 kdp 출력 폴더의 정본을 블록마다 견준다")
    ap.add_argument("old", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    lines, changed = diff(a.old, a.new)
    text = "\n".join([f"바뀐 블록 {changed}개 — {a.old} → {a.new}", "", *lines]) + "\n"
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
