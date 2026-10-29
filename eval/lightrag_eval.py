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
- 다시 순위: 기본으로 교차 인코더 bge-reranker-v2-m3(Apache-2.0)가 묻기마다 후보 조각의 차례를 다시 매긴다(D23 — 넣기·저장소는 그대로,
  테스트_결과 14.10). --rerank none 이면 끈다.
- 임베딩: 이 프로세스의 BGE-M3(MIT, 1024차원). 저장소: 벡터 NanoVectorDB(LightRAG 기본, numpy 로 모두 견주는 정확 검색 — 사용자
  결정 2026-10-26)·그래프 NetworkX·나머지 JSON — 모두 <work>/<계통>-<토큰>-<R|F>/ 파일. 실행마다 지우고 새로 만든다(--reuse 면 둔다 —
  글이 바뀌어도 LightRAG 가 같은 문서 id 를 다시 넣지 않아 옛 글로 잰다).
결과: <out>/report.md · report.json · ranks.jsonl."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import shutil
import sys
import threading
import time
from collections import Counter
from pathlib import Path

from rag_eval import (NAMES, P_TOKENS, QUESTIONS, RR_MODEL, TYPES, Dense, Reranker, frac, hit, intro, load_questions, norm,
                      parse_variants, summarize, system_texts)

RUNS = "kdp_rag:512"
MODES = ("naive", "mix")
MODE_NAMES = {"naive": "naive(조각 벡터 검색)", "mix": "mix(지식 그래프 + 벡터)", "local": "local", "global": "global",
              "hybrid": "hybrid"}
LLM_URL, LLM_MODEL = "http://127.0.0.1:8120/v1", "skt/A.X-4.0-Light"
SEP = "<SEP>"                    # LightRAG 가 엔티티·관계의 file_path 여럿을 잇는 구분자(constants.GRAPH_FIELD_SEP)
NUMBER_ONLY = re.compile(r"[\d.,%℃㎜ ()+\-~∼]+")     # 숫자뿐인 엔티티 이름(2026, 23.1℃, (11) …) — 그래프에 쓸모가 적다
WARN_KEYS = (("truncated", "Token-limit truncation during initial"), ("truncated_gleaning", "Token-limit truncation during gleaning"),
             ("format_errors", "LLM output format error; found "),      # 버린 레코드만 — '…; find LLM use <|#|> …' 는 고쳐 읽은 알림
             ("rerank_failed", "using original chunks"), ("rerank_failed", "Rerank is enabled but no rerank model"))
CACHE_FILE = "kv_store_llm_response_cache.json"   # LightRAG LLM 응답 캐시 — 열쇠는 물음·모델 이름이고 max_tokens 는 들지 않는다
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
        out["number_only"] = sum(bool(NUMBER_ONLY.fullmatch(str(n))) for n in g.nodes)
    return out


class Warnings(logging.Handler):
    """lightrag 경고를 센다 — 넣기: 추출 응답이 출력 토큰 한도(max_tokens)에 잘린 조각(처음 추출·글리닝)과 꼴 오류로 버린
    엔티티·관계 레코드(관계는 따로), 질문: 다시 순위가 실패해 처음 차례로 돌아간 수."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.counts = Counter()

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        self.counts.update(k for k, s in WARN_KEYS if s in msg)
        if WARN_KEYS[2][1] in msg and " on RELATION " in msg:
            self.counts["format_relations"] += 1


async def open_store(name: str, d: Path, tokens: int, dense: Dense, a, rr: Reranker | None = None,
                     calls: Counter | None = None) -> object:
    """저장소를 연다 — rr 이 있으면 교차 인코더를 LightRAG 다시 순위 자리에(생성자로 넘겨야 mix 도 쓴다 — 설정이 생성 때 굳는다)."""
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete_if_cache
    from lightrag.utils import EmbeddingFunc

    async def llm(prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kw):
        if calls is not None:                  # LightRAG 는 캐시에 없을 때만 부른다 — 캐시를 복사해 넣었으면 새로 부른 수
            calls["llm_calls"] += 1
        kw.setdefault("temperature", a.temperature)
        kw.setdefault("max_tokens", a.llm_max_tokens)          # 작은 모델이 되풀이에 빠져도 여기서 멈춘다
        kw.setdefault("extra_body", {"chat_template_kwargs": {"enable_thinking": False}})   # 생각 모드가 있는 모델(Qwen3)만 읽는다
        return await openai_complete_if_cache(a.llm_model, prompt, system_prompt=system_prompt,
                                              history_messages=history_messages or [], keyword_extraction=keyword_extraction,
                                              base_url=a.llm_url, api_key="EMPTY", **kw)

    async def no_llm(*args, **kw):
        raise RuntimeError("--no-llm 인데 LLM 을 불렀다 — mix 키워드가 캐시에 없거나 넣기가 일어났다")

    gpu = threading.Lock()   # GPU 모델(BGE-M3·다시 순위)을 차례로 — 스레드 안에서 잡는다(LightRAG 가 시간 초과로 코루틴을 끊어도 스레드는 돈다)

    def locked(f, *args):
        with gpu:
            return f(*args)

    async def embed(texts: list[str]):     # LightRAG 는 임베딩을 여럿 동시에 부른다
        return await asyncio.to_thread(locked, dense.encode, list(texts), P_TOKENS)

    async def rerank(query: str, documents: list[str], top_n: int | None = None, **_):
        s = await asyncio.to_thread(locked, rr.score, query, list(documents))
        calls["rerank_calls"] += 1
        order = sorted(range(len(s)), key=lambda i: (-s[i], i))[:top_n or None]   # LightRAG 는 돌려준 차례를 그대로 쓴다
        return [{"index": int(i), "relevance_score": float(s[i])} for i in order]

    addon = {"language": "Korean"}
    if a.entity_types == "ko":
        addon["entity_types_guidance"] = KO_ENTITY_TYPES
    rag = LightRAG(working_dir=str(d), workspace=name, vector_storage="NanoVectorDBStorage",
                   chunk_token_size=tokens, chunk_overlap_token_size=a.overlap, entity_extract_max_gleaning=a.gleaning,
                   llm_model_func=no_llm if a.no_llm else llm, llm_model_name=a.llm_model, llm_model_max_async=a.llm_async,
                   enable_llm_cache=getattr(a, "query_cache", True),   # 물음 답·키워드 캐시 — 답 비교는 끈다(추출 캐시는 따로)
                   **({"default_llm_timeout": a.llm_timeout} if getattr(a, "llm_timeout", None) else {}),
                   embedding_func=EmbeddingFunc(embedding_dim=1024, max_token_size=8192, func=embed, model_name="bge-m3"),
                   addon_params=addon, **({"rerank_model_func": rerank, "min_rerank_score": 0.0, "rerank_model_max_async": 1,
                                           "default_rerank_timeout": 120} if rr else {}))   # 거르지 않고 차례만
    await rag.initialize_storages()
    return rag


async def ask(rag, questions: list[dict], modes: list[str], top_k: int, chunk_top_k: int, rerank: bool = False) -> list[dict]:
    from lightrag import QueryParam

    rows = []
    for q in questions:
        for mode in modes:
            r = await rag.aquery_data(q["q"], param=QueryParam(mode=mode, top_k=top_k, chunk_top_k=chunk_top_k,
                                                               enable_rerank=rerank))
            data = (r.get("data") or {}) if r.get("status") == "success" else {}
            chunks = data.get("chunks") or []
            rank = chunk_rank(chunks, q["doc"], q["ev"])
            rows.append({"mode": mode, "id": q["id"], "doc": q["doc"], "type": q["type"], "rank": rank, "returned": len(chunks),
                         "in_chunks": rank is not None,
                         "in_graph": graph_hit((data.get("entities") or []) + (data.get("relationships") or []), q["doc"], q["ev"]),
                         "status": r.get("status")})
    return rows


async def run(questions: list[dict], texts: dict[str, dict[str, str]], keys: dict[str, str], dense: Dense, a,
              rr: Reranker | None = None) -> tuple[list, dict]:
    rows, builds = [], {}
    for name, tokens, strategy in a.runs:
        label, docs = run_label(name, tokens, strategy), texts[name]
        store = label.replace(":", "-")
        d = a.work / store
        if a.query_only and not d.is_dir():
            sys.exit(f"--query-only 인데 저장소가 없다: {d}")
        if not (a.query_only or a.reuse):
            cache = a.llm_cache_from.read_bytes() if a.llm_cache_from else None   # 지우기 전에 읽는다 — 원본이 이 저장소 안일 수 있다
            shutil.rmtree(d, ignore_errors=True)
            if cache is not None:              # 같은 물음은 캐시 응답 그대로 — 잘려서 캐시에 없던 추출·그에 딸린 요약만 새로 부른다
                (d / store).mkdir(parents=True)
                (d / store / CACHE_FILE).write_bytes(cache)
        d.mkdir(parents=True, exist_ok=True)
        calls, warn = Counter(), Warnings()
        rag = await open_store(store, d, tokens, dense, a, rr, calls)
        logging.getLogger("lightrag").addHandler(warn)
        try:
            ids = sorted(docs)
            t = time.perf_counter()
            if a.query_only:                   # 넣지 않는다 — 다시 넣으면 이미 있는 문서 id 가 doc_status 에 'dup-' 실패 행으로 남는다
                builds[label] = store_stats(d, ids)
            else:
                if strategy == "R":            # SDK ainsert 는 F 만 — 문서마다 process_options 로 R 을 골라 넣는다(서버 넣기 경로와 같다)
                    await rag.apipeline_enqueue_documents([docs[i] for i in ids], ids=ids, file_paths=[keys[i] for i in ids],
                                                          process_options=["R"] * len(ids))
                    await rag.apipeline_process_enqueue_documents()
                else:
                    await rag.ainsert([docs[i] for i in ids], ids=ids, file_paths=[keys[i] for i in ids])
                builds[label] = {"seconds": round(time.perf_counter() - t, 1), **store_stats(d, ids), "llm_calls": calls["llm_calls"],
                                 **{k: warn.counts[k] for k in ("truncated", "truncated_gleaning", "format_errors", "format_relations")}}
            print(f"{label} 저장소 {builds[label]}", file=sys.stderr)
            warn.counts.clear()                # 질문 동안의 경고는 따로
            rows += [{**r, "run": label} for r in await ask(rag, questions, a.modes, a.top_k, a.chunk_top_k, rr is not None)]
            if rr:
                builds[label].update(rerank_calls=calls["rerank_calls"], rerank_failed=warn.counts["rerank_failed"])
        finally:
            logging.getLogger("lightrag").removeHandler(warn)
            await rag.finalize_storages()
    return rows, builds


def run_name(label: str) -> str:
    name, tokens, strategy = label.split(":")
    return f"{NAMES.get(name, name)} · {int(tokens):,}토큰 · " + ("R 구조 자르기" if strategy == "R" else "F 고정 토큰")


def render(summary: dict, builds: dict, questions: list[dict], labels: list[str], modes: list[str], a, dense_name: str,
           rr_name: str | None = None) -> str:
    rerank = (f"다시 순위 `{rr_name}`(naive 는 처음 가져온 chunk_top_k 개를, mix 는 벡터·그래프 조각을 합친 후보 전체를 다시 순위 매긴 뒤"
              f" chunk_top_k 개로 자른다 — 거르지 않는다)" if rr_name else "다시 순위 매기기 없음")
    store = ("넣지 않고 있는 저장소에 묻기만(--query-only)" if a.query_only else
             f"LLM 캐시를 `{a.llm_cache_from}` 에서 복사해 넣음(같은 물음은 같은 응답)" if a.llm_cache_from else "")
    lines = ["# LightRAG 검색 평가(D21·D22)", "", intro(questions),
             f"LightRAG 1.5.x SDK — 넣기 R(구조 자르기, process_options 'R')·F(고정 토큰, ainsert)는 실행 이름에, 겹침 {a.overlap}, "
             f"언어 Korean, 엔티티 종류 "
             f"{'한국어 안내' if a.entity_types == 'ko' else 'LightRAG 기본'}, 글리닝 {a.gleaning}, LLM `{a.llm_model}`"
             f"(temperature {a.temperature}, max_tokens {a.llm_max_tokens}), 임베딩 `{dense_name}`, 벡터 NanoVectorDB·그래프 NetworkX."
             f" 질문 top_k {a.top_k}·chunk_top_k {a.chunk_top_k}, {rerank}." + (f" {store}." if store else ""),
             "'돌려받은 조각에' = 돌려받은 조각 어디에든 근거가 있는 질문 수, '그래프 설명에' = 돌려받은 엔티티·관계 가운데 질문 문서에서"
             " 나온 것의 설명(LLM 이 쓴 글)들에 근거 낱말 묶음이 모두(묶음마다 어느 설명에든) 있는 질문 수 — test3·test5 처럼 같은 이름"
             " 엔티티를 함께 가진 문서는 설명이 이미 섞여 있어 다 거르지 못한다.", "", "## 저장소", "",
             "'숫자뿐' = 이름이 숫자·단위·기호뿐인 엔티티, '추출 잘림' = 추출 응답이 max_tokens 에 잘린 조각 수(처음 추출 / 글리닝 —"
             " 이미 넣은 문서를 다시 쓰면(--reuse) 넣기가 없어 0), '꼴 오류' = 칸 수가 맞지 않아 버린 레코드(그 가운데 관계),"
             " 'LLM 호출' = 넣기 동안 캐시에 없어 LLM 을 부른 수(묻기만 하면 —).", "",
             "| 실행 | 조각 | 엔티티(숫자뿐) | 관계 | 추출 잘림 | 꼴 오류(관계) | 문서 상태 | 넣기(초) | LLM 호출 | 다시 순위 호출(실패) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for label in labels:
        b = builds.get(label, {})
        lines.append(f"| {run_name(label)} | {b.get('chunks', '—')} | {b.get('entities', '—')}({b.get('number_only', '—')}) | "
                     f"{b.get('relations', '—')} | {b.get('truncated', '—')} / {b.get('truncated_gleaning', '—')} | "
                     f"{b.get('format_errors', '—')}({b.get('format_relations', '—')}) | "
                     f"{', '.join(f'{k} {v}' for k, v in (b.get('docs') or {}).items()) or '—'} | {b.get('seconds', '—')} | "
                     f"{b.get('llm_calls', '—')} | {b.get('rerank_calls', '—')}({b.get('rerank_failed', '—')}) |")
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
    ap.add_argument("--llm-timeout", type=int, default=None, help="LLM 한 번의 시간 한도(초) — 없으면 LightRAG 기본 240, 31B 처럼 느린 모델은 늘린다")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--chunk-top-k", type=int, default=20)
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reuse", action="store_true", help="저장소를 지우지 않는다(같은 글로 다시 물을 때만)")
    ap.add_argument("--query-only", action="store_true", help="넣지 않고 있는 저장소에 묻기만(지우지도 않는다 — 다시 순위만 바꿔 잴 때)")
    ap.add_argument("--no-llm", action="store_true", help="LLM 을 부르면 오류 — naive 와 캐시된 mix 키워드만 쓰는 --query-only 실행")
    ap.add_argument("--llm-cache-from", type=Path, default=None,
                    help="이 저장소 폴더의 LLM 캐시를 새 저장소에 복사해 넣는다 — 잘려서 캐시에 없던 추출만 새로 부른다(max_tokens 견주기)")
    ap.add_argument("--rerank", nargs="?", const=RR_MODEL, default=RR_MODEL,
                    help=f"교차 인코더 다시 순위 — 모델 이름(기본 {RR_MODEL}, Apache-2.0), none 이면 끈다")
    a = ap.parse_args(argv)
    a.rerank = None if a.rerank == "none" else a.rerank
    if a.no_llm and not a.query_only:
        sys.exit("--no-llm 은 --query-only 와 함께 — 넣기는 LLM 으로 추출한다")
    if a.llm_cache_from and (a.reuse or a.query_only):
        sys.exit("--llm-cache-from 은 저장소를 새로 만들 때만 — --reuse·--query-only 면 지우지 않아 복사하지 않는다")
    if a.llm_cache_from:                       # 저장소 하나만 — 여럿이 든 폴더면 rglob 이 아무것이나 고른다
        found = list(a.llm_cache_from.rglob(CACHE_FILE))
        if len(found) != 1:
            sys.exit(f"{a.llm_cache_from} 에 {CACHE_FILE} 이 {len(found)}개 — 저장소 폴더 하나를 준다")
        a.llm_cache_from = found[0]
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
    rr = Reranker(a.rerank, a.device) if a.rerank else None   # 첫 질문 전에 올린다 — 호출 시간 한도 안에 모델을 올리다 넘기지 않게
    rows, builds = asyncio.run(run(questions, texts, keys, dense, a, rr))
    summary = summarize(rows, key=("mode", "run"))
    for name, x in summary.items():
        x["in_graph"] = sum(r["in_graph"] for r in rows if f"{r['mode']}|{r['run']}" == name)
    labels = [run_label(*r) for r in a.runs]
    report = render(summary, builds, questions, labels, a.modes, a, dense.name, rr.name if rr else None)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.md").write_text(report + "\n", encoding="utf-8")
    (a.out / "report.json").write_text(json.dumps({"summary": summary, "builds": builds, "llm": a.llm_model, "embedding": dense.name,
                                                   "reranker": rr.name if rr else None, "runs": labels,
                                                   "llm_max_tokens": a.llm_max_tokens, "query_only": a.query_only,
                                                   "llm_cache_from": str(a.llm_cache_from) if a.llm_cache_from else None},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    (a.out / "ranks.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
