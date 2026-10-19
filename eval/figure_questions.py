"""그림 해석 물음 평가(D20, 기획서 7.10) — 실제 문서(test2~test5) 그래프·지도 86물음(eval/judged/figure_questions.jsonl, 두 평가자가 따로
매긴 정답)을 설명 서버에 묻고 채점한다. 원래 A.X 와 튜닝한 A.X 를 같은 크롭(PDF 를 144dpi 로 렌더해 블록 상자로 자름)으로 잰다.
통과 기준(테스트_결과 11.6 과 같다): 확신 문턱 위 답의 정답률 90% 이상이면서 물음의 40% 이상에 답한다 — 개발(test2·test3)에서 이
기준을 넘는 문턱 가운데 가장 많이 답하는 것을 골라, 그 문턱 그대로 확인(test4·test5)도 넘어야 한다(확인 자료로 문턱을 고르지 않는다).
보기 물음은 보기 순서를 거꾸로 해 두 번 묻고 두 답이 같을 때만 답한 것으로 본다. 정답은 두 평가자가 모두 맞다고 본 것만.
사용: python eval/figure_questions.py ask --server http://127.0.0.1:8119/v1 --out ans_base.jsonl
      python eval/figure_questions.py score ans_base.jsonl [ans_tuned.jsonl ...]"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kdp.figure_qa import alts, nkey, same_short  # noqa: E402

QUESTIONS = ROOT / "eval" / "judged" / "figure_questions.jsonl"
DEV, TEST = ("test2", "test3"), ("test4", "test5")
DPI = 144                                     # 파서 렌더 해상도(configs/parse/default.yaml) — 실제 크롭과 같게
PASS_ACC, PASS_COVER = 0.90, 0.40
THRESHOLDS = (-0.02, -0.05, -0.1, -0.2, -0.3, -0.5, -0.7, -1.0, None)


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return [json.loads(x) for x in open(path, encoding="utf-8")]


def keys_of(q: dict) -> set[str] | None:
    """두 평가자가 모두 맞다고 본 정답 — 보기 물음은 두 정답의 겹침, 단답은 상대 정답과도 맞는 표기만(한 평가자만 적은 답은 정답이
    아니다). 둘 다 물음이 맞다고 보고 겹치는 정답이 있을 때만, 아니면 None(물음이 맞지 않음·다툼 — 채점에서 뺀다)."""
    a, b = q["keys"]["A"], q["keys"]["B"]
    if not (a["ok_question"] and b["ok_question"]):
        return None
    sa, sb = alts(a["answer"]), alts(b["answer"])
    keys = sa & sb if q["options_A"] else {x for x in sa if same_short(x, sb)} | {y for y in sb if same_short(y, sa)}
    return keys or None


def crop(pdf_cache: dict, doc: str, page: int, bbox: list[float]) -> bytes:
    import pypdfium2 as pdfium

    if doc not in pdf_cache:
        pdf_cache[doc] = pdfium.PdfDocument(str(ROOT / f"{doc}.pdf"))
    s = DPI / 72
    img = pdf_cache[doc][page - 1].render(scale=s).to_pil()
    x0, y0, x1, y1 = bbox
    buf = io.BytesIO()
    img.crop((int(x0 * s), int(y0 * s), int(x1 * s), int(y1 * s))).convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def ask(a) -> None:
    import openai

    c = openai.OpenAI(base_url=a.server, api_key="EMPTY", timeout=300, max_retries=1)
    pdfs, imgs = {}, {}
    with open(a.out, "w", encoding="utf-8") as out:
        for q in load_questions():
            k = (q["doc"], q["block"])
            if k not in imgs:
                imgs[k] = base64.b64encode(crop(pdfs, q["doc"], q["page"], q["bbox"])).decode()
            for order in ("A", "B") if q["options_A"] else ("A",):
                r = c.chat.completions.create(
                    model=a.model, temperature=0, max_tokens=8 if q["options_A"] else 24, logprobs=True, top_logprobs=5,
                    messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + imgs[k]}},
                                                           {"type": "text", "text": q["prompt_" + order]}]}],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                ch = r.choices[0]
                toks = [[t.token, round(t.logprob, 4)] for t in (ch.logprobs.content or [])] if ch.logprobs else []
                out.write(json.dumps({"qid": q["qid"], "order": order, "text": ch.message.content, "tokens": toks}, ensure_ascii=False) + "\n")
    print("ASK_DONE", a.out)


def parse(q: dict, r: dict, order: str) -> tuple[str | None, float | None]:
    """(답 글, 확신 로그확률) — 보기 물음은 첫 숫자 토큰의 확률로 보기 글을, 단답은 첫 줄에서 맺음말을 떼고 토큰 평균."""
    toks = r.get("tokens") or []
    if q["options_A"]:
        for t, lp in toks:
            m = re.search(r"[1-9]", t)
            if m:
                opts = q["options_" + order]
                i = int(m.group()) - 1
                return (opts[i] if i < len(opts) else None), lp
        return None, None
    text = (r.get("text") or "").strip()
    ans = re.sub(r"(입니다|이다|다)?[.。]?$", "", text.splitlines()[0] if text else "").strip()
    lps = [lp for t, lp in toks if t.strip()]
    return ans, (sum(lps) / len(lps) if lps else None)


def score_rows(questions: list[dict], answers: list[dict]) -> list[dict]:
    by = defaultdict(dict)
    for r in answers:
        by[r["qid"]][r["order"]] = r
    rows = []
    for q in questions:
        keys = keys_of(q)
        if keys is None or "A" not in by[q["qid"]]:
            continue
        ans, conf = parse(q, by[q["qid"]]["A"], "A")
        if q["options_A"]:
            ans_b, conf_b = parse(q, by[q["qid"]].get("B", {}), "B")
            ok = ans is not None and nkey(ans) in keys
            same = ans is not None and ans == ans_b
            conf = min(conf, conf_b) if conf is not None and conf_b is not None else None
        else:
            ok, same = ans is not None and same_short(ans, keys), True
        rows.append({"qid": q["qid"], "doc": q["doc"], "qtype": q["qtype"], "ans": ans, "ok": ok, "same": same, "conf": conf})
    return rows


def auc(pos, neg) -> float:
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def summary(rows: list[dict]) -> dict:
    """정답률·가르는 힘·문턱별(두 순서 같은 답 + 문턱 위) 답한 비율과 정답률·통과 여부."""
    n = len(rows)
    out = {"n": n, "acc": sum(r["ok"] for r in rows) / max(1, n),
           "auc": auc([r["conf"] for r in rows if r["ok"] and r["conf"] is not None],
                      [r["conf"] for r in rows if not r["ok"] and r["conf"] is not None]),
           "by_qtype": {qt: [sum(r["ok"] for r in rows if r["qtype"] == qt), sum(r["qtype"] == qt for r in rows)]
                        for qt in sorted({r["qtype"] for r in rows})}, "curve": []}
    passed = False
    for th in THRESHOLDS:
        keep = [r for r in rows if r["same"] and r["conf"] is not None and (th is None or r["conf"] >= th)]
        acc = sum(r["ok"] for r in keep) / max(1, len(keep))
        cover = len(keep) / max(1, n)
        ok = bool(keep) and acc >= PASS_ACC and cover >= PASS_COVER
        out["curve"].append({"th": th, "answered": len(keep), "cover": cover, "acc": acc, "pass": ok})
        passed |= ok
    out["pass"] = passed                      # 이 자료만 볼 때 기준을 넘는 문턱이 있나 — 판정은 verdict
    return out


def verdict(dev: dict, test: dict) -> dict:
    """통과 판정 — 개발에서 기준을 넘는 문턱 가운데 가장 많이 답하는 것을 골라 확인에 그대로 댄다(두 summary 의 curve 는 같은 THRESHOLDS)."""
    ok = [c for c in dev["curve"] if c["pass"]]
    if not ok:
        return {"th": None, "chosen": False, "pass": False, "test": None}
    best = max(ok, key=lambda c: c["cover"])
    t = next(c for c in test["curve"] if c["th"] == best["th"])
    return {"th": best["th"], "chosen": True, "pass": t["pass"], "test": t}


def score(a) -> None:
    qs = load_questions()
    kept = sum(keys_of(q) is not None for q in qs)
    print(f"물음 {len(qs)} — 채점하는 물음(두 평가자 정답이 같음) {kept}")
    for path in a.answers:
        rows = score_rows(qs, [json.loads(x) for x in open(path, encoding="utf-8")])
        print(f"\n== {path}")
        s2 = {}
        for name, docs in (("개발 test2·test3", DEV), ("확인 test4·test5", TEST)):
            s = s2[docs] = summary([r for r in rows if r["doc"] in docs])
            print(f"  {name}: 정답 {s['acc']:.0%}({s['n']}물음) AUC {s['auc']:.2f}"
                  f" | 갈래별 {', '.join(f'{k} {v[0]}/{v[1]}' for k, v in s['by_qtype'].items())}")
            print("    문턱별 답함/정답률: " + " · ".join(
                f"{'전부' if c['th'] is None else c['th']}: {c['cover']:.0%}/{c['acc']:.0%}{'✓' if c['pass'] else ''}" for c in s["curve"]))
        v = verdict(s2[DEV], s2[TEST])
        print("  판정: " + (f"개발에서 고른 문턱 {'전부' if v['th'] is None else v['th']} → 확인 답함 {v['test']['cover']:.0%}"
                          f"·정답 {v['test']['acc']:.0%} — 통과 {'예' if v['pass'] else '아니오'}" if v["chosen"]
                          else "개발에서 기준을 넘는 문턱이 없다 — 통과 아니오"))
        print("  지도 물음 답 분포(보기 순서 A — 한 방위로 쏠리면 찍는 것):",
              dict(Counter(r["ans"] for r in rows if r["qtype"].startswith("map"))))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ask")
    p.add_argument("--server", default="http://127.0.0.1:8119/v1")
    p.add_argument("--model", default="skt/A.X-4.0-VL-Light")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("score")
    p.add_argument("answers", nargs="+", type=Path)
    a = ap.parse_args(argv)
    (ask if a.cmd == "ask" else score)(a)


if __name__ == "__main__":
    main()
