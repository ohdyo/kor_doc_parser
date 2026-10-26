"""LightRAG 검색 평가(D21·D22) — 계통·청크 크기마다 LightRAG 저장소를 따로 만들고, 같은 질문을 naive(조각 벡터 검색)·mix(지식
그래프 + 벡터) 모드로 물어 돌려받은 조각 위 5개에 근거가 있는지 잰다. 정답·비교 규칙은 eval/rag_eval.py 와 같다.

    python eval/lightrag_eval.py --kdp <kdp 문서 폴더들> --compare <eval/compare.py -o 폴더> --work <저장소 폴더> --out <결과 폴더>

- 실행(--runs): '계통:청크 토큰[:R|F]' 목록 — 기본은 kdp document.rag.md 하나를 512토큰 R 로(사용자 결정 2026-10-26 — 저장소마다
  LLM 추출에 15분 안팎이라 고른 후보 하나만, 1회차 계통·크기 비교는 `테스트_결과.md` 14장). 넣기 R 은 구조 자르기
  (apipeline_enqueue_documents 에 process_options 'R' — 빈 줄·줄바꿈 경계로 나눠 청크 토큰까지 모은다, langchain-text-splitters),
  F 는 SDK ainsert(고정 토큰 자르기). 겹침은 LightRAG 기본 100토큰. 문서 id 는 문서 해시, file_path 는 문서 이름(test2 …).
- 추출: 언어 Korean, 한국어 엔티티 종류 안내(KO_ENTITY_TYPES — 사람 이름은 뽑지 않는다), 글리닝 1(기본). LightRAG 1.5.7 의
  few-shot 은 실제 예문이 아니라 출력 꼴 틀이라(작은 모델이 예문 엔티티를 베끼지 않게) 그대로 둔다.
- LLM: OpenAI 호환 서버(gpu_server/serve_llm.sh 의 vLLM — 기본 국내 모델 skt/A.X-4.0-Light, Apache-2.0). LightRAG 는 temperature·
  max_tokens 를 넘기지 않으므로(모델 기본 표본 뽑기가 된다) 여기서 0·2,048 로 건다.
- 임베딩: 이 프로세스의 BGE-M3(MIT, 1024차원). 저장소: 벡터 NanoVectorDB(LightRAG 기본, numpy 로 모두 견주는 정확 검색 — 사용자
  결정 2026-10-26)·그래프 NetworkX·나머지 JSON — 모두 <work>/<계통>-<토큰>-<R|F>/ 파일. 실행마다 지우고 새로 만든다(--reuse 면 둔다 —
  글이 바뀌어도 LightRAG 가 같은 문서 id 를 다시 넣지 않아 옛 글로 잰다).
결과: <out>/report.md · report.json · ranks.jsonl."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
import time
from pathlib import Path

from rag_eval import (NAMES, P_TOKENS, QUESTIONS, TYPES, Dense, frac, hit, intro, load_questions, norm, parse_variants,
                      summarize, system_texts)

RUNS = "kdp_rag:512"
MODES = ("naive", "mix")
MODE_NAMES = {"naive": "naive(조각 벡터 검색)", "mix": "mix(지식 그래프 + 벡터)", "local": "local", "global": "global",
              "hybrid": "hybrid"}
LLM_URL, LLM_MODEL = "http://127.0.0.1:8120/v1", "skt/A.X-4.0-Light"
SEP = "<SEP>"                    # LightRAG 가 엔티티·관계의 file_path 여럿을 잇는 구분자(constants.GRAPH_FIELD_SEP)
KO_ENTITY_TYPES = """각 엔티티를 아래 종류 가운데 하나로 분류한다. 맞는 종류가 없으면 `기타`로 쓴다. 종류 이름은 아래 한국어 그대로 쓴다.

- 기관: 정부 부처·청, 지방자치단체, 공공기관, 위원회, 연구소, 병원·학교 같은 조직과 그 부서
- 지역: 시·도, 시·군·구, 읍·면·동, 관측 지점, 해역·하천·산 같은 지리적 장소
- 질병·건강: 질병, 증상, 병원체·매개체(모기 등), 환자 집단, 건강 위험 요인
- 기상·환경: 기온·강수·가뭄·태풍·해수면 온도·기압계 같은 기상·기후 현상과 환경 요인
- 지표·통계: 수치로 나타낸 측정값·통계·순위·비율 — 이름에 대상과 기준 시기를 함께 쓰고, 값과 단위는 설명에 그대로 옮긴다
- 기간·시점: 연도, 월, 주차, 상순·중순·하순, 날짜, 관측·집계 기간
- 정책·제도: 법령, 제도, 경보·주의보, 지침, 감시체계, 사업
- 조치·방법: 예방수칙, 대응 조치, 진단·치료 방법, 분석·관측 방법
- 시설·장소: 작업장·주택·논밭 같은 장소 유형, 시설, 장비, 정보 시스템
- 문서·자료: 보도자료, 보고서, 표·그림, 통계 자료, 누리집

사람 이름(담당자·기관장 같은 개인)은 엔티티로 뽑지 않는다. 직책이 필요하면 기관 엔티티의 설명에 직책만 쓴다."""


def parse_runs(spec: str, names=NAMES) -> list[tuple[str, int, str]]:
    """'계통:토큰[:R|F],…' → [(계통, 토큰, 자르기)] — 자르기는 R(구조 자르기 — LangChain 재귀 분할, 기본) 또는 F(고정 토큰, ainsert)."""
    out = []
    for item in (x.strip() for x in spec.split(",") if x.strip()):
        name, tokens, strategy = (item.split(":") + ["R"])[:3]
        if name not in names or not tokens.isdigit() or strategy not in ("F", "R"):
            sys.exit(f"--runs 가 잘못됐다: {item}")
        out.append((name, int(tokens), strategy))
    return out


def run_label(name: str, tokens: int, strategy: str) -> str:
    return f"{name}:{tokens}:{strategy}"


def chunk_rank(chunks: list[dict], doc: str, evidence: list[list[str]]) -> int | None:
    """돌려받은 조각 차례에서 첫 근거 조각의 순위(1부터) — 질문 문서(file_path)의 조각만 근거다."""
    for k, c in enumerate(chunks, start=1):
        if c.get("file_path") == doc and hit(norm(c.get("content") or ""), evidence):
            return k
    return None


def graph_hit(items: list[dict], doc: str, evidence: list[list[str]]) -> bool:
    """LLM 이 받는 그래프 맥락에 근거가 있다 — 돌려받은 엔티티·관계 가운데 질문 문서에서 나온 것만 보고, 근거 낱말 묶음마다 그
    설명들 어딘가에 있다(그래프는 값은 관계에, 주어는 엔티티에 나눠 담는다). 설명들을 이어 붙이지 않는다(경계에서 글자가 붙지 않게)."""
    descs = [norm(x.get("description") or "") for x in items if doc in (x.get("file_path") or "").split(SEP)]
    return bool(descs) and all(any(a in d for d in descs for a in group) for group in evidence)


def store_stats(d: Path, ids: list[str]) -> dict:
    """조각·엔티티·관계 수, 넣은 문서의 처리 상태 — 저장소 파일에서 센다."""
    out = {}
    if f := next(d.rglob("kv_store_text_chunks.json"), None):
        out["chunks"] = len(json.loads(f.read_text(encoding="utf-8")))
    if f := next(d.rglob("kv_store_doc_status.json"), None):
        rows = json.loads(f.read_text(encoding="utf-8"))
        status = [str((rows.get(i) or {}).get("status", "없음")) for i in ids]   # 다른 행(중복 표시 등)은 세지 않는다
        out["docs"] = {s: status.count(s) for s in sorted(set(status))}
    if f := next(d.rglob("graph_chunk_entity_relation.graphml"), None):
        import networkx as nx

        g = nx.read_graphml(f)
        out["entities"], out["relations"] = g.number_of_nodes(), g.number_of_edges()
    return out


async def open_store(name: str, d: Path, tokens: int, dense: Dense, a) -> object:
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete_if_cache
    from lightrag.utils import EmbeddingFunc

    async def llm(prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kw):
        kw.setdefault("temperature", a.temperature)
        kw.setdefault("max_tokens", a.llm_max_tokens)          # 작은 모델이 되풀이에 빠져도 여기서 멈춘다
        kw.setdefault("extra_body", {"chat_template_kwargs": {"enable_thinking": False}})   # 생각 모드가 있는 모델(Qwen3)만 읽는다
        return await openai_complete_if_cache(a.llm_model, prompt, system_prompt=system_prompt,
                                              history_messages=history_messages or [], keyword_extraction=keyword_extraction,
                                              base_url=a.llm_url, api_key="EMPTY", **kw)

    lock = asyncio.Lock()

    async def embed(texts: list[str]):
        async with lock:                       # GPU 모델 하나를 차례로 — LightRAG 는 임베딩을 여럿 동시에 부른다
            return await asyncio.to_thread(dense.encode, list(texts), P_TOKENS)

    addon = {"language": "Korean"}
    if a.entity_types == "ko":
        addon["entity_types_guidance"] = KO_ENTITY_TYPES
    rag = LightRAG(working_dir=str(d), workspace=name, vector_storage="NanoVectorDBStorage",
                   chunk_token_size=tokens, chunk_overlap_token_size=a.overlap, entity_extract_max_gleaning=a.gleaning,
                   llm_model_func=llm, llm_model_name=a.llm_model, llm_model_max_async=a.llm_async,
                   embedding_func=EmbeddingFunc(embedding_dim=1024, max_token_size=8192, func=embed, model_name="bge-m3"),
                   addon_params=addon)
    await rag.initialize_storages()
    return rag


async def ask(rag, questions: list[dict], modes: list[str], top_k: int, chunk_top_k: int) -> list[dict]:
    from lightrag import QueryParam

    rows = []
    for q in questions:
        for mode in modes:
            r = await rag.aquery_data(q["q"], param=QueryParam(mode=mode, top_k=top_k, chunk_top_k=chunk_top_k,
                                                               enable_rerank=False))
            data = (r.get("data") or {}) if r.get("status") == "success" else {}
            chunks = data.get("chunks") or []
            rank = chunk_rank(chunks, q["doc"], q["ev"])
            rows.append({"mode": mode, "id": q["id"], "doc": q["doc"], "type": q["type"], "rank": rank, "returned": len(chunks),
                         "in_chunks": rank is not None,
                         "in_graph": graph_hit((data.get("entities") or []) + (data.get("relationships") or []), q["doc"], q["ev"]),
                         "status": r.get("status")})
    return rows


async def run(questions: list[dict], texts: dict[str, dict[str, str]], keys: dict[str, str], dense: Dense, a) -> tuple[list, dict]:
    rows, builds = [], {}
    for name, tokens, strategy in a.runs:
        label, docs = run_label(name, tokens, strategy), texts[name]
        store = label.replace(":", "-")
        d = a.work / store
        if not a.reuse:
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
        rag = await open_store(store, d, tokens, dense, a)
        try:
            ids = sorted(docs)
            t = time.perf_counter()
            if strategy == "R":                # SDK ainsert 는 F 만 — 문서마다 process_options 로 R 을 골라 넣는다(서버 넣기 경로와 같다)
                await rag.apipeline_enqueue_documents([docs[i] for i in ids], ids=ids, file_paths=[keys[i] for i in ids],
                                                      process_options=["R"] * len(ids))
                await rag.apipeline_process_enqueue_documents()
            else:
                await rag.ainsert([docs[i] for i in ids], ids=ids, file_paths=[keys[i] for i in ids])
            builds[label] = {"seconds": round(time.perf_counter() - t, 1), **store_stats(d, ids)}
            print(f"{label} 넣기 {builds[label]}", file=sys.stderr)
            rows += [{**r, "run": label} for r in await ask(rag, questions, a.modes, a.top_k, a.chunk_top_k)]
        finally:
            await rag.finalize_storages()
    return rows, builds


def run_name(label: str) -> str:
    name, tokens, strategy = label.split(":")
    return f"{NAMES.get(name, name)} · {int(tokens):,}토큰 · " + ("R 구조 자르기" if strategy == "R" else "F 고정 토큰")


def render(summary: dict, builds: dict, questions: list[dict], labels: list[str], modes: list[str], a, dense_name: str) -> str:
    lines = ["# LightRAG 검색 평가(D21·D22)", "", intro(questions),
             f"LightRAG 1.5.x SDK — 넣기 R(구조 자르기, process_options 'R')·F(고정 토큰, ainsert)는 실행 이름에, 겹침 {a.overlap}, "
             f"언어 Korean, 엔티티 종류 "
             f"{'한국어 안내' if a.entity_types == 'ko' else 'LightRAG 기본'}, 글리닝 {a.gleaning}, LLM `{a.llm_model}`"
             f"(temperature {a.temperature}, max_tokens {a.llm_max_tokens}), 임베딩 `{dense_name}`, 벡터 NanoVectorDB·그래프 NetworkX."
             f" 질문 top_k {a.top_k}·chunk_top_k {a.chunk_top_k}, 다시 순위 매기기 없음.",
             "'돌려받은 조각에' = 돌려받은 조각 어디에든 근거가 있는 질문 수, '그래프 설명에' = 돌려받은 엔티티·관계 가운데 질문 문서에서"
             " 나온 것의 설명(LLM 이 쓴 글)들에 근거 낱말 묶음이 모두(묶음마다 어느 설명에든) 있는 질문 수 — test3·test5 처럼 같은 이름"
             " 엔티티를 함께 가진 문서는 설명이 이미 섞여 있어 다 거르지 못한다.", "", "## 저장소", "",
             "| 실행 | 조각 | 엔티티 | 관계 | 문서 상태 | 넣기(초) |", "|---|---|---|---|---|---|"]
    for label in labels:
        b = builds.get(label, {})
        lines.append(f"| {run_name(label)} | {b.get('chunks', '—')} | {b.get('entities', '—')} | {b.get('relations', '—')} | "
                     f"{', '.join(f'{k} {v}' for k, v in (b.get('docs') or {}).items()) or '—'} | {b.get('seconds', '—')} |")
    for mode in modes:
        lines += ["", f"## {MODE_NAMES.get(mode, mode)}", "",
                  "| 실행 | R@5 | " + " | ".join(TYPES) + " | MRR@10 | 돌려받은 조각에 | 그래프 설명에 |",
                  "|---|---|" + "---|" * len(TYPES) + "---|---|---|"]
        for label in labels:
            x = summary[f"{mode}|{label}"]
            lines.append(f"| {run_name(label)} | {frac(x['r5'], x['n'])} | " + " | ".join(frac(*x["by_type"][t]) for t in TYPES)
                         + f" | {x['mrr']:.3f} | {x['in_chunks']}/{x['n']} | {x['in_graph']}/{x['n']} |")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="LightRAG 검색 평가(D21·D22) — 계통·청크 크기별 저장소, naive·mix 모드 R@5·MRR@10")
    ap.add_argument("--kdp", type=Path, required=True)
    ap.add_argument("--compare", type=Path, default=None)
    ap.add_argument("--questions", type=Path, default=QUESTIONS)
    ap.add_argument("--work", type=Path, required=True, help="LightRAG 저장소 폴더(실행마다 하위 폴더)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", default=RUNS, help="'계통:청크 토큰[:R|F]' 목록 — R 구조 자르기(기본)·F 고정 토큰")   # 그래프는 실행마다 LLM 추출이라 하나만 기본으로
    ap.add_argument("--variant", action="append", help="'이름=폴더' — 다른 설정으로 다시 조립한 kdp 출력을 계통으로(eval/rag_eval.py 와 같다)")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--gleaning", type=int, default=1)
    ap.add_argument("--entity-types", choices=("ko", "default"), default="ko")
    ap.add_argument("--llm-url", default=LLM_URL)
    ap.add_argument("--llm-model", default=LLM_MODEL)
    ap.add_argument("--llm-async", type=int, default=8)
    ap.add_argument("--llm-max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--chunk-top-k", type=int, default=20)
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reuse", action="store_true", help="저장소를 지우지 않는다(같은 글로 다시 물을 때만)")
    a = ap.parse_args(argv)
    variants = parse_variants(a.variant)
    a.runs, a.modes = parse_runs(a.runs, {**NAMES, **variants}), [m for m in a.modes.split(",") if m]
    import lightrag.utils  # noqa: F401  — 불러올 때 lightrag 로그를 INFO 로 두므로 그 뒤에 낮춘다

    logging.getLogger("lightrag").setLevel(logging.WARNING)
    questions = load_questions(a.questions)
    keys = {q["doc_id"]: q["doc"] for q in questions}
    texts = system_texts(a.kdp, a.compare, set(keys), [n for n in dict.fromkeys(n for n, _, _ in a.runs) if n not in variants],
                         {n: p for n, p in variants.items() if n in {r[0] for r in a.runs}})
    a.runs = [r for r in a.runs if r[0] in texts]
    if not a.runs:
        sys.exit("평가할 계통이 없다")
    dense = Dense(a.model, a.device)
    rows, builds = asyncio.run(run(questions, texts, keys, dense, a))
    summary = summarize(rows, key=("mode", "run"))
    for name, x in summary.items():
        x["in_graph"] = sum(r["in_graph"] for r in rows if f"{r['mode']}|{r['run']}" == name)
    labels = [run_label(*r) for r in a.runs]
    report = render(summary, builds, questions, labels, a.modes, a, dense.name)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.md").write_text(report + "\n", encoding="utf-8")
    (a.out / "report.json").write_text(json.dumps({"summary": summary, "builds": builds, "llm": a.llm_model, "embedding": dense.name,
                                                   "runs": labels}, ensure_ascii=False, indent=1), encoding="utf-8")
    (a.out / "ranks.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
