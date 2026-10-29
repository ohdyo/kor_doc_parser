"""RAG 답 비교(D24) — kor_doc_parser 출력으로 만든 LightRAG 저장소에 같은 질문을 세 가지로 묻고, 같은 LLM(A.X-4.0-Light)의 답을
모은다: LightRAG 기본(mix — 지식그래프 + 조각 벡터, 다시 순위), Naive RAG(naive — 조각 벡터만, 다시 순위), RAG 없음(bypass — 검색 없이
질문만 LLM 에). 답 프롬프트는 LightRAG 기본 그대로(질문 언어로·마크다운·참고 문헌), RAG 없음은 질문만 보낸다.

    python eval/answer_eval.py --work <eval/lightrag_eval.py 의 저장소 폴더> --out <결과 폴더> [--questions <질문 파일>]

- 질문: 기본은 답 비교용 50개(eval/golden/rag_answer_questions.yaml — 사실·표·그림·종합·요약·없음·문서간, 정답 a 와 채점 요소 points).
  검색 평가 질문 72개(eval/golden/rag_questions.yaml)도 같은 꼴로 읽는다. 근거 낱말(evidence)이 있는 한 문서 질문만 '문맥에 근거'를 센다.
- 저장소는 넣지 않고 연다(없으면 멈춘다). 문맥 한도: A.X 문맥 16k(serve_llm.sh --max-model-len 16384) 안에 답 2,048 을 남기게 LightRAG
  기본 한도(전체 30,000 · 엔티티 6,000 · 관계 8,000 토큰)를 같은 비율로 0.4 배(12,000 · 2,400 · 3,200) — LightRAG 토크나이저(o200k)로
  세며, A.X 는 이 문서들에서 o200k 의 약 1.03 배다.
- 결과: <out>/answers.jsonl(질문·모드마다 답·정답·문맥 조각에서 근거의 순위·조각·엔티티·관계 수·걸린 시간) · report.md(모드별 문맥에
  근거·답 길이·시간). 답이 맞는지는 정답과 대조해 따로 판정한다.
- 7B 대 31B 비교(D25): --llm-model·--llm-url 로 답하는 LLM 을 바꾼다. --questions 는 쉼표로 여러 파일. --variant 는 결과에 남길 이름,
  --user-prompt-file 은 답 지시(QueryParam.user_prompt — mix·naive 만), --oracle 은 근거만 주고 묻기(naive — 조각 검색 자리에 질문 문서의
  근거 낱말이 든 조각만 넣는다, 근거 낱말이 있는 한 문서 질문만). 답은 늘 새로 받는다(LightRAG 답 캐시 끔 — 열쇠에 문맥이 들지 않아
  근거만 주고 묻기가 앞 naive 답을 받아 온다). 행마다 LLM·저장소·변형 이름, LLM 에 넘긴 조각 id·문맥 속 수(답 속 수 확인용)·조각 다시
  순위 최고 점수(답하지 않기 문턱을 나중에 흉내 낸다)를 남긴다."""
from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

from lightrag_eval import LLM_MODEL, LLM_URL, RUNS, chunk_rank, open_store, parse_runs, run_label
from rag_eval import NAMES, RR_MODEL, Dense, Reranker, hit, norm

MODES = {"mix": "LightRAG 기본(mix)", "naive": "Naive RAG(naive)", "bypass": "RAG 없음(bypass)"}
QUESTIONS = Path(__file__).resolve().parent / "golden" / "rag_answer_questions.yaml"
REFS = "### References"
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
ORACLE = contextvars.ContextVar("oracle", default=None)   # 근거만 주고 묻기 — 이 물음에 넣을 조각(물음마다 — 동시에 여럿을 묻는다)


def load(paths) -> list[dict]:
    """질문 파일(하나 또는 여럿) — 한 문서(doc)든 여러 문서(docs)든, 근거 낱말 묶음(evidence)은 있으면 정규화해 둔다."""
    out, seen = [], set()
    for path in [paths] if isinstance(paths, (str, Path)) else paths:
        for q in yaml.safe_load(Path(path).read_text(encoding="utf-8"))["questions"]:
            if q["id"] in seen or not q.get("q") or "a" not in q:
                sys.exit(f"질문 {q.get('id')}: id 가 겹치거나 물음·정답이 없다")
            seen.add(q["id"])
            docs = q.get("docs") or [q["doc"]]
            out.append({**q, "docs": docs, "ev": [[norm(x) for x in g] for g in q.get("evidence") or []]})
    return out


def body(answer: str) -> str:
    """참고 문헌 절을 뗀 답 — 길이는 본문만 센다."""
    return answer.split(REFS)[0].strip()


def numbers(texts: list[str]) -> list[str]:
    """글 속 수(쉼표는 뗀다) — 답 속 수가 LLM 에 넘긴 문맥에 있는지 나중에 견준다."""
    return sorted({m.replace(",", "") for t in texts for m in NUM.findall(t or "")})


def context_texts(data: dict) -> list[str]:
    """LLM 에 넘긴 문맥의 글 — 조각 글, 엔티티 이름·설명, 관계 양끝 이름·설명(LightRAG 1.5.7 이 mix 문맥에 넣는 것)."""
    graph = (data.get("entities") or []) + (data.get("relationships") or [])
    return ([c.get("content") or "" for c in data.get("chunks") or []]
            + [x.get(k) or "" for x in graph for k in ("entity_name", "src_id", "tgt_id", "description")])


def oracle_chunks(kv: dict, q: dict, limit: int = 3) -> list[dict]:
    """근거만 주고 묻기 — 질문 문서의 조각 가운데 근거 낱말 묶음이 모두 든 조각(없으면 묶음 하나라도 든 조각)을 문서 차례로 limit 개까지,
    LightRAG 조각 벡터 검색 결과 꼴로. kv 는 저장소의 kv_store_text_chunks.json."""
    doc = q["docs"][0]
    mine = sorted(((cid, c) for cid, c in kv.items() if c.get("file_path") == doc), key=lambda x: x[1].get("chunk_order_index", 0))
    picked = ([x for x in mine if hit(norm(x[1].get("content") or ""), q["ev"])]
              or [x for x in mine if any(hit(norm(x[1].get("content") or ""), [g]) for g in q["ev"])])
    return [{"id": cid, "content": c["content"], "file_path": doc, "created_at": None} for cid, c in picked[:limit]]


def max_rerank(rr, query: str, docs: list[str]) -> float | None:
    """LLM 에 넘긴 조각의 다시 순위 최고 점수 — LightRAG 가 다시 순위를 매기며 채운 캐시에서만 읽는다(모델을 다시 돌리지 않는다)."""
    s = [rr.cache[(query, d)] for d in docs if (query, d) in rr.cache] if rr else []
    return round(max(s), 4) if s else None


async def answer(rag, q: dict, mode: str, a, rr: Reranker | None, oracle: list[dict] | None = None) -> dict:
    from lightrag import QueryParam

    param = QueryParam(mode=mode, top_k=a.top_k, chunk_top_k=a.chunk_top_k, enable_rerank=rr is not None and mode != "bypass",
                       max_total_tokens=a.max_total_tokens, max_entity_tokens=a.max_entity_tokens,
                       max_relation_tokens=a.max_relation_tokens, user_prompt=a.user_prompt if mode != "bypass" else None)
    token = ORACLE.set(oracle)
    t = time.perf_counter()
    try:
        r = await rag.aquery_llm(q["q"], param=param)
    except Exception as e:                     # 그 물음만 실패로 적는다
        r = {"status": "error", "message": f"{type(e).__name__}: {str(e)[:300]}"}
    finally:
        ORACLE.reset(token)
    data = r.get("data") or {}
    chunks = data.get("chunks") or []
    contents = [c.get("content") or "" for c in chunks]
    text = (r.get("llm_response") or {}).get("content") or ""
    scored = bool(q["ev"]) and len(q["docs"]) == 1        # 근거 낱말이 있는 한 문서 질문만 문맥 근거를 센다
    return {"id": q["id"], "docs": q["docs"], "type": q["type"], "q": q["q"], "gold": q["a"], "points": q.get("points") or [],
            "llm": a.llm_model, "variant": a.variant, "mode": mode, "answer": text, "status": r.get("status"),
            "message": None if r.get("status") == "success" else r.get("message"),
            "seconds": round(time.perf_counter() - t, 1), "chunks": len(chunks), "scored": scored,
            "evidence_rank": chunk_rank(chunks, q["docs"][0], q["ev"]) if scored and chunks else None,
            "entities": len(data.get("entities") or []), "relations": len(data.get("relationships") or []),
            "chunk_ids": [c.get("chunk_id") for c in chunks],
            "ctx_numbers": numbers(context_texts(data)),
            "max_rr": max_rerank(rr, q["q"], contents) if mode != "bypass" else None,
            **({"oracle_chunks": len(oracle)} if oracle is not None else {})}


def summarize(rows: list[dict]) -> dict[str, dict]:
    """모드마다 — 답 수·실패·문맥 조각에 근거가 든 질문(근거 낱말이 있는 질문 가운데, 유형별)·평균 조각 수·평균 답 길이(참고 문헌
    뺌)·평균 시간."""
    out, types = {}, list(dict.fromkeys(r["type"] for r in rows))
    for mode in dict.fromkeys(r["mode"] for r in rows):
        rs = [r for r in rows if r["mode"] == mode]
        ctx = Counter(r["type"] for r in rs if r["evidence_rank"] is not None)
        out[mode] = {"n": len(rs), "failed": sum(r["status"] != "success" for r in rs), "scored": sum(r["scored"] for r in rs),
                     "in_context": sum(ctx.values()), "in_context_by_type": {t: ctx[t] for t in types},
                     "chunks": round(sum(r["chunks"] for r in rs) / len(rs), 1),
                     "chars": round(sum(len(body(r["answer"])) for r in rs) / len(rs)),
                     "seconds": round(sum(r["seconds"] for r in rs) / len(rs), 1)}
    return out


def render(summary: dict, label: str, a, rr_name: str | None) -> str:
    types = list(next(iter(summary.values()))["in_context_by_type"]) if summary else []
    how = [f"변형 `{getattr(a, 'variant', 'base')}`"]
    if getattr(a, "user_prompt_file", None):
        how.append(f"답 지시 `{a.user_prompt_file.name}`")
    if getattr(a, "oracle", False):
        how.append("근거만 주고 묻기(조각 검색 자리에 근거 조각만)")
    lines = ["# RAG 답 비교(D24·D25) — LightRAG 기본·Naive RAG·RAG 없음", "",
             f"{' · '.join(how)}. "
             f"저장소 `{label}`, LLM `{a.llm_model}`(temperature {a.temperature}, 답 한도 {a.llm_max_tokens}), 다시 순위 `{rr_name}`(bypass 빼고), "
             f"top_k {a.top_k} · chunk_top_k {a.chunk_top_k}, 문맥 한도 {a.max_total_tokens:,} · 엔티티 {a.max_entity_tokens:,} · 관계 {a.max_relation_tokens:,} 토큰. "
             "'문맥에 근거' = 근거 낱말이 있는 한 문서 질문 가운데 LLM 에 넘긴 조각에 근거 조각이 든 수(유형 칸도 같다). 답이 맞는지는 따로 판정한다.", "",
             "| 모드 | 답 | 실패 | 문맥에 근거 | " + " | ".join(types) + " | 평균 조각 | 평균 답 글자 | 평균 시간(초) |",
             "|---|---|---|---|" + "---|" * len(types) + "---|---|---|"]
    for mode, s in summary.items():
        lines.append(f"| {MODES.get(mode, mode)} | {s['n']} | {s['failed']} | {s['in_context']}/{s['scored']} | "
                     + " | ".join(str(s["in_context_by_type"][t]) for t in types) + f" | {s['chunks']} | {s['chars']} | {s['seconds']} |")
    return "\n".join(lines)


async def run(questions: list[dict], dense: Dense, a, rr: Reranker | None) -> tuple[str, list[dict]]:
    (name, tokens, strategy), = a.runs
    label = run_label(name, tokens, strategy)
    store = label.replace(":", "-")
    d = a.work / store
    if not d.is_dir():
        sys.exit(f"저장소가 없다: {d} — eval/lightrag_eval.py 로 먼저 만든다")
    rag = await open_store(store, d, tokens, dense, a, rr, Counter())
    picked = {}
    if a.oracle:                               # 조각 검색 자리에 근거 조각만 — 근거 조각을 찾은 질문만 묻는다
        kv = json.loads(next(d.rglob("kv_store_text_chunks.json")).read_text(encoding="utf-8"))
        picked = {q["id"]: oracle_chunks(kv, q) for q in questions if q["ev"] and len(q["docs"]) == 1}
        picked = {k: v for k, v in picked.items() if v}
        print(f"근거만 주고 묻기: 근거 조각을 찾은 질문 {len(picked)}/{len(questions)}", file=sys.stderr)
        questions = [q for q in questions if q["id"] in picked]
        search = rag.chunks_vdb.query

        async def query(text, top_k, query_embedding=None, **kw):
            chunks = ORACLE.get()
            return chunks if chunks is not None else await search(text, top_k=top_k, query_embedding=query_embedding, **kw)

        rag.chunks_vdb.query = query
    sem = asyncio.Semaphore(a.concurrency)

    async def one(q, mode):
        async with sem:
            r = await answer(rag, q, mode, a, rr, picked.get(q["id"]))
            print(f"{q['id']} {mode} {r['status']} {r['seconds']}초", file=sys.stderr)
            return {**r, "store": f"{a.work.name}/{label}"}     # 같은 계통 이름의 저장소(7B·31B 그래프)를 폴더로 가른다

    try:
        rows = await asyncio.gather(*(one(q, m) for q in questions for m in a.modes))
    finally:
        await rag.finalize_storages()
    return label, list(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", type=Path, required=True, help="eval/lightrag_eval.py 가 만든 저장소 폴더(--work 와 같게)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--questions", default=str(QUESTIONS), help="질문 파일 — 쉼표로 여럿")
    ap.add_argument("--variant", default="base", help="결과에 남길 변형 이름(최적화 시험 — 답 지시·문맥 한도)")
    ap.add_argument("--user-prompt-file", type=Path, default=None, help="답 지시 파일(LightRAG QueryParam.user_prompt — mix·naive 만)")
    ap.add_argument("--oracle", action="store_true", help="근거만 주고 묻기 — naive, 근거 낱말이 있는 한 문서 질문만")
    ap.add_argument("--runs", default=RUNS, help="저장소 하나 — '계통:청크 토큰[:R|F]'(eval/lightrag_eval.py 와 같다)")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--gleaning", type=int, default=1)
    ap.add_argument("--entity-types", choices=("ko", "default"), default="ko")
    ap.add_argument("--llm-url", default=LLM_URL)
    ap.add_argument("--llm-model", default=LLM_MODEL)
    ap.add_argument("--llm-async", type=int, default=8)
    ap.add_argument("--llm-max-tokens", type=int, default=2048)
    ap.add_argument("--llm-timeout", type=int, default=None, help="LLM 한 번의 시간 한도(초) — 없으면 LightRAG 기본 240")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--chunk-top-k", type=int, default=20)
    ap.add_argument("--max-total-tokens", type=int, default=12000)
    ap.add_argument("--max-entity-tokens", type=int, default=2400)
    ap.add_argument("--max-relation-tokens", type=int, default=3200)
    ap.add_argument("--concurrency", type=int, default=8, help="한꺼번에 묻는 수")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rerank", nargs="?", const=RR_MODEL, default=RR_MODEL, help=f"다시 순위 모델(기본 {RR_MODEL}), none 이면 끈다")
    a = ap.parse_args(argv)
    a.rerank = None if a.rerank == "none" else a.rerank
    a.no_llm = False                           # open_store 가 읽는다 — 답을 받으려면 LLM 을 부른다
    a.query_cache = False                      # open_store 가 읽는다 — 답을 늘 새로 받는다(모듈 설명)
    a.user_prompt = a.user_prompt_file.read_text(encoding="utf-8").strip() if a.user_prompt_file else None
    a.runs, a.modes = parse_runs(a.runs, NAMES), ["naive"] if a.oracle else [m for m in a.modes.split(",") if m]
    if len(a.runs) != 1 or set(a.modes) - MODES.keys():
        sys.exit("저장소는 하나, 모드는 mix·naive·bypass 가운데서")
    import logging

    import lightrag.utils  # noqa: F401  — 불러올 때 lightrag 로그를 INFO 로 두므로 그 뒤에 낮춘다

    logging.getLogger("lightrag").setLevel(logging.WARNING)
    questions = load([Path(p) for p in a.questions.split(",") if p])
    dense = Dense(a.model, a.device)
    rr = Reranker(a.rerank, a.device) if a.rerank else None
    label, rows = asyncio.run(run(questions, dense, a, rr))
    summary = summarize(rows)
    report = render(summary, label, a, rr.name if rr else None)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "answers.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (a.out / "report.json").write_text(json.dumps({"summary": summary, "store": label, "llm": a.llm_model, "variant": a.variant,
                                                   "oracle": a.oracle, "user_prompt": a.user_prompt,
                                                   "reranker": rr.name if rr else None, "max_total_tokens": a.max_total_tokens},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    (a.out / "report.md").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
