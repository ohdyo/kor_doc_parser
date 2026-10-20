"""새 시험지(tt 문서) 라벨 → 채점 물음 파일(eval/judged/figure_questions_tt.jsonl — eval/figure_questions.py 가 함께 읽는다).
두 라벨러가 모두 맞는 물음이라 하고 정답이 겹치는 물음만(keys_of 와 같은 기준). 그림은 채점 때 원본 PDF 를 쪽(sheet_no)·bbox(시트 pt,
좌상단 원점)로 다시 잘라 만든다(DPI 144 — 파서와 같다). 보기 물음은 보기를 qid 로 정해 섞은 순서(A)와 거꾸로(B) 둘.
사용: python eval/testset/build_questions.py /hdd_data/raganything/tt_dt PACK_DIR... --out eval/judged/figure_questions_tt.jsonl"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kdp.figure_qa import prompt  # noqa: E402

_spec = importlib.util.spec_from_file_location("figure_questions", ROOT / "eval" / "figure_questions.py")
fq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fq)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def build(tt: Path, packs: list[Path]) -> tuple[list[dict], Counter]:
    names = {r["sha256"][:12]: (r["sha256"], r["name"].removesuffix(".pdf")) for r in _jsonl(ROOT / "eval" / "testset" / "docs.jsonl")}
    rows, stat, docs = [], Counter(), {}
    for pack in packs:
        items = json.load(open(pack / "figures.json", encoding="utf-8"))
        la, lb = ({x["qid"]: x for x in json.load(open(pack / f"labels_{w}.json", encoding="utf-8"))["labels"]} for w in "AB")
        for it in items:
            sha, name = names[it["doc"]]
            if sha not in docs:
                docs[sha] = json.load(open(tt / "parsed" / sha / "document.json", encoding="utf-8"))
            page, b = next((p, b) for p in docs[sha]["pages"] for b in p["blocks"] if b["id"] == it["block"])
            if page.get("rot90_k"):                                   # 세운 스캔 쪽은 시트 그대로 자르면 누운 그림이 된다
                stat["세운 쪽이라 뺌"] += len(it["questions"])
                continue
            for q in it["questions"]:
                ka, kb = la.get(q["qid"]), lb.get(q["qid"])
                if not (ka and kb):
                    stat["라벨 없음"] += 1
                    continue
                row = {"qid": f"{name}:{it['block']}:{q['qtype']}", "doc": name, "block": it["block"], "page": page["sheet_no"],
                       "bbox": b["bbox"], "qtype": q["qtype"], "question": q["question"], **({"x": q["x"]} if q.get("x") else {}),
                       "options_A": None, "options_B": None, "prompt_A": None, "prompt_B": None,
                       "keys": {w: {k: lab[k] for k in ("ok_question", "answer", "note")} for w, lab in (("A", ka), ("B", kb))}}
                if q["options"]:
                    oa = random.Random(row["qid"]).sample(q["options"], len(q["options"]))
                    row.update(options_A=oa, options_B=oa[::-1],
                               prompt_A=prompt(q["qtype"], it["caption"], it["figure_text"], oa, x=q.get("x")),
                               prompt_B=prompt(q["qtype"], it["caption"], it["figure_text"], oa[::-1], x=q.get("x")))
                else:
                    row["prompt_A"] = prompt(q["qtype"], it["caption"], it["figure_text"], x=q.get("x"))
                if fq.keys_of(row) is None:
                    stat["합의 없음·맞지 않는 물음"] += 1
                    continue
                rows.append(row)
                stat["넣음"] += 1
    return rows, stat


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tt", type=Path, help="시험 자료 폴더(parsed/<sha256>/document.json 이 있는 곳)")
    ap.add_argument("packs", nargs="+", type=Path, help="라벨 묶음 폴더(figures.json·labels_A.json·labels_B.json)")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "judged" / "figure_questions_tt.jsonl")
    a = ap.parse_args(argv)
    rows, stat = build(a.tt, a.packs)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(dict(stat), "→", a.out, "· 갈래", dict(Counter(r["qtype"] for r in rows)))


if __name__ == "__main__":
    main()
