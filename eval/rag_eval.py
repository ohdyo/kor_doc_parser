"""RAG 하류 평가 축소판(기획서 8.6, D21) — 파서 출력 글을 같은 방식으로 잘라 같은 검색기로 찾고, 질문의 근거가 든 조각이 위 5개
안에 드는지(R@5)와 첫 근거 조각의 순위(MRR@10)를 잰다. 근거 유형(문단·표·그림)별로 따로 센다.

    python eval/rag_eval.py --kdp <kdp 문서 폴더들이 든 폴더> --compare <eval/compare.py -o 폴더> --out <결과 폴더>

계통: 텍스트 레이어(바닥선) · PaddleOCR-VL 기본(같은 엔진 기본 마크다운) · docling(외부 도구 기본 마크다운) · kdp document.md(지금
출력) · kdp document.rag.md(D21). 자르기 — LightRAG: ainsert 가 쓰는 고정 토큰 자르기 그대로(lightrag 패키지 — 받는 쪽이 LightRAG 다,
D22. 사용자가 정한 512토큰과 LightRAG 기본 1,200토큰, 겹침 100), 구조: 빈 줄로 가른 덩이를 제목 줄에서 끊고 800자(512토큰 안팎)까지
모음(제목·빈 줄을 보는 도구), 길이: 800자 창·100자 겹침(시험용 — 패키지 없이). 검색기 — BM25(글자 2-gram, 모델 없음)·BGE-M3(MIT,
밀집 — GPU, LightRAG naive 와 같은 벡터 검색). 네 문서의 조각을 한 말뭉치에 넣고 찾는다(다른 문서 조각은 근거가 아니다). 정답 eval/golden/rag_questions.yaml —
근거 낱말 묶음마다 대안 하나 이상이 든 조각이 근거다. 공백·HTML 태그·주석·마크다운 기호·글머리표는 떼고 비교한다(표기 차이로 계통이
갈리지 않게). LightRAG 저장소로 직접 묻는 평가(그래프 포함)는 eval/lightrag_eval.py.
결과: <out>/report.md(표) · report.json(값) · ranks.jsonl(질문별 순위)."""
from __future__ import annotations

import argparse
import functools
import html as htmllib
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "eval" / "golden" / "rag_questions.yaml"
SYSTEMS = ("textlayer", "paddle_default", "docling", "kdp", "kdp_rag")
NAMES = {"textlayer": "텍스트 레이어", "paddle_default": "PaddleOCR-VL 기본", "docling": "docling",
         "kdp": "kdp document.md", "kdp_rag": "kdp document.rag.md"}
KDP_FILES = {"kdp": "document.md", "kdp_rag": "document.rag.md"}
RETRIEVER_NAMES = {"bm25": "BM25(글자 2-gram)", "bge-m3": "BGE-M3"}
TYPES = ("문단", "표", "그림")
SIZE, OVERLAP = 800, 100         # 구조·길이 자르기의 조각 길이·겹침(글자) — 512토큰 안팎
LR_TOKENS, LR_OVERLAP = 512, 100   # LightRAG 청크(사용자 결정 2026-10-26)·겹침(LightRAG 기본) — 고정 토큰, tiktoken gpt-4o-mini
LR_DEFAULT = 1200                  # LightRAG 기본 청크 — 견주는 값
K, MRR_K = 5, 10
Q_TOKENS, P_TOKENS = 256, 4096   # BGE-M3 에 넣는 질문·조각 최대 토큰(표 HTML 은 글자당 토큰이 많다)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_TAG = re.compile(r"</?[A-Za-z][^<>]*>")           # HTML 태그만 — '<표 1>' 같은 한글 꺾쇠 제목은 글자다
_SKIP = re.compile(r"[\s|#*_`~\\\-‐‑‒–—―−•·‧∙・ㆍᆞ▪▫◦○●■□◆◇▶►∼]+")   # 공백·마크업·글머리표·이음표·가운뎃점
_DOTS = re.compile(r"\.{2,}")                      # 목차 점선
_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"', "„": '"', "‟": '"'})
_WORD = re.compile(r"\w+")
_HEAD = re.compile(r"^#{1,6}[ \t]", re.M)          # 마크다운 제목 줄(앵커 줄 다음 줄이어도)
_BLANK = re.compile(r"\n[ \t]*\n")


# ---- 정답 ----

def norm(s: str) -> str:
    """비교용 글 — NFKC, HTML 주석·태그를 떼고 문자 참조를 푼 뒤 공백·마크업 기호·글머리표·이음표·가운뎃점·따옴표 표기 차이를 없앤다."""
    s = htmllib.unescape(_TAG.sub(" ", _COMMENT.sub(" ", unicodedata.normalize("NFKC", s or ""))))
    return _DOTS.sub("", _SKIP.sub("", s.translate(_QUOTES)))


def hit(text: str, evidence: list[list[str]]) -> bool:
    """text(정규화한 글)에 낱말 묶음마다 대안 하나 이상이 있다."""
    return all(any(a in text for a in group) for group in evidence)


def load_questions(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    out, seen = [], set()
    for q in data["questions"]:
        ev = [[norm(a) for a in group] for group in q["evidence"]]
        if q["type"] not in TYPES or not ev or not all(group and all(group) for group in ev) or q["id"] in seen:
            sys.exit(f"질문 {q['id']}: 유형·근거가 잘못됐거나 id 가 겹친다")
        seen.add(q["id"])
        out.append({**q, "doc_id": data["docs"][q["doc"]], "ev": ev})
    return out


# ---- 계통 글 ----

def kdp_texts(root: Path) -> dict[str, dict[str, str]]:
    """{계통: {doc_id: 글}} — document.json 이 있는 문서 폴더마다 document.md·document.rag.md."""
    out = {name: {} for name in KDP_FILES}
    for d in sorted(p for p in root.iterdir() if (p / "document.json").exists()):
        doc_id = json.loads((d / "document.json").read_text(encoding="utf-8"))["doc_id"]
        for name, f in KDP_FILES.items():
            if (d / f).exists():
                out[name][doc_id] = (d / f).read_text(encoding="utf-8")
    return out


def compare_texts(root: Path) -> dict[str, dict[str, str]]:
    """eval/compare.py 출력(<문서 해시>/compare/<계통>.json)의 글 — 텍스트 레이어는 쪽 글을 잇고, 두 파서는 기본 마크다운(없는 옛
    저장본은 본문 블록 글로 대신하고 알린다)."""
    out = {name: {} for name in ("textlayer", "paddle_default", "docling")}
    for d in sorted(p for p in root.iterdir() if (p / "compare").is_dir()):
        for name in out:
            f = d / "compare" / f"{name}.json"
            if not f.exists():
                continue
            run = json.loads(f.read_text(encoding="utf-8"))
            md = run.get("markdown")
            md = "\n\n".join(md) if isinstance(md, list) else md
            if name == "textlayer" or not md:
                if name != "textlayer":
                    print(f"{NAMES[name]} {d.name[:8]}: 기본 마크다운 없음 — 본문 블록 글로 대신한다", file=sys.stderr)
                pages = sorted(run["pages"].items(), key=lambda kv: int(kv[0]))
                md = "\n\n".join(b["text"] for _, bs in pages for b in bs if b.get("body", True) and b.get("text"))
            out[name]["sha256:" + d.name] = md
    return out


def system_texts(kdp: Path, compare: Path | None, need: set[str], wanted=SYSTEMS) -> dict[str, dict[str, str]]:
    """평가할 계통의 글 — 질문 문서가 하나라도 없는 계통은 알리고 뺀다(질문 수가 다른 칸을 나란히 두지 않게). 말뭉치는 질문 문서만."""
    texts = {**kdp_texts(kdp), **(compare_texts(compare) if compare else {})}
    out = {}
    for s in wanted:
        miss = need - set(texts.get(s, {}))
        if miss:
            print(f"{NAMES[s]} 뺌 — 글이 없는 문서 {len(miss)}개({', '.join(sorted(m[7:15] for m in miss))})", file=sys.stderr)
            continue
        out[s] = {d: t for d, t in texts[s].items() if d in need}
    return out


# ---- 자르기 ----

def chunk_length(text: str, size: int = SIZE, overlap: int = OVERLAP) -> list[str]:
    """size 글자 창을 overlap 만큼 겹쳐 민다 — 구조를 보지 않고 길이로 자르는 도구."""
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        if i + size >= len(text):
            break
        i += size - overlap
    return [c for c in out if c.strip()]


def chunk_structure(text: str, size: int = SIZE, overlap: int = OVERLAP) -> list[str]:
    """빈 줄로 가른 덩이를 차례로 모은다 — 제목 줄이 든 덩이에서 새 조각을 열고, 모은 길이가 size 를 넘기 전에 끊는다.
    size 보다 긴 덩이는 길이로 자른다(마크다운 제목·빈 줄 경계를 보는 도구)."""
    out, cur, n = [], [], 0
    for blk in (b.strip("\n") for b in _BLANK.split(text) if b.strip()):
        if len(blk) > size:
            out += ["\n\n".join(cur)] if cur else []
            cur, n = [], 0
            out += chunk_length(blk, size, overlap)
            continue
        if cur and (_HEAD.search(blk) or n + 2 + len(blk) > size):
            out.append("\n\n".join(cur))
            cur, n = [], 0
        n += len(blk) + (2 if cur else 0)
        cur.append(blk)
    return out + (["\n\n".join(cur)] if cur else [])


@functools.cache
def _tiktoken():
    from lightrag.utils import TiktokenTokenizer

    return TiktokenTokenizer("gpt-4o-mini")


def chunk_lightrag(text: str, size: int = LR_TOKENS, overlap: int = LR_OVERLAP) -> list[str]:
    """LightRAG ainsert 의 고정 토큰 자르기 그대로 — 넣기처럼 글을 먼저 다듬고(sanitize_text_for_encoding: 앞뒤 공백·문자 참조·제어
    문자) lightrag 패키지의 chunking_by_token_size 로(구분 문자 없음)."""
    from lightrag.chunker.token_size import chunking_by_token_size
    from lightrag.utils import sanitize_text_for_encoding

    return [c["content"] for c in chunking_by_token_size(_tiktoken(), sanitize_text_for_encoding(text),
                                                          chunk_overlap_token_size=overlap, chunk_token_size=size)
            if c["content"].strip()]


CHUNKERS: dict[str, tuple[str, Callable[[str], list[str]]]] = {
    "lightrag": (f"LightRAG 자르기({LR_TOKENS}토큰·{LR_OVERLAP} 겹침)", chunk_lightrag),
    "lightrag1200": (f"LightRAG 자르기({LR_DEFAULT:,}토큰 — LightRAG 기본)", functools.partial(chunk_lightrag, size=LR_DEFAULT)),
    "structure": (f"구조 자르기({SIZE:,}자)", chunk_structure),
    "length": (f"길이 자르기({SIZE:,}자·{OVERLAP} 겹침)", chunk_length),
}


# ---- 검색기 ----

def tokens(s: str) -> list[str]:
    """글자 2-gram(낱말 안에서만, 한 글자 낱말은 그대로) — 형태소 분석기 없이도 조사·어미가 붙은 꼴에 덜 흔들린다."""
    out = []
    for w in _WORD.findall(unicodedata.normalize("NFKC", s).lower()):
        out += [w] if len(w) == 1 else [w[i:i + 2] for i in range(len(w) - 1)]
    return out


def bm25_scores(chunks: list[str], queries: list[str], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    """Okapi BM25 — 질문 토큰은 한 번씩만 센다."""
    tfs = [Counter(tokens(c)) for c in chunks]
    lens = [sum(t.values()) for t in tfs]
    avg = sum(lens) / max(len(lens), 1) or 1.0
    df = Counter(w for t in tfs for w in t)
    idf = {w: math.log(1 + (len(chunks) - d + 0.5) / (d + 0.5)) for w, d in df.items()}
    out = np.zeros((len(queries), len(chunks)))
    for qi, q in enumerate(queries):
        for w in set(tokens(q)) & idf.keys():
            for ci, (t, n) in enumerate(zip(tfs, lens)):
                if f := t.get(w):
                    out[qi, ci] += idf[w] * f * (k1 + 1) / (f + k1 * (1 - b + b * n / avg))
    return out


class Dense:
    """BGE-M3 밀집 벡터 — [CLS] 마지막 은닉 상태를 정규화한다(FlagEmbedding 의 dense 와 같다). 같은 글은 한 번만 만든다."""

    def __init__(self, model: str, device: str, batch: int = 8):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch, self.device, self.batch, self.cache = torch, device, batch, {}
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(device).eval()
        if device.startswith("cuda"):
            self.model.half()
        self.name = f"{model}@{getattr(self.model.config, '_commit_hash', None) or '?'}"

    def encode(self, texts: list[str], max_length: int = P_TOKENS) -> np.ndarray:
        todo = sorted({t for t in texts if (t, max_length) not in self.cache}, key=len)   # 길이가 비슷한 것끼리 묶어 덧댐을 줄인다
        for i in range(0, len(todo), self.batch):
            part = todo[i:i + self.batch]
            enc = self.tok(part, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                h = self.model(**enc).last_hidden_state[:, 0]
            for t, v in zip(part, self.torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy()):
                self.cache[(t, max_length)] = v
        return np.stack([self.cache[(t, max_length)] for t in texts]) if texts else np.zeros((0, 1024), np.float32)

    def scores(self, chunks: list[str], queries: list[str]) -> np.ndarray:
        return self.encode(queries, Q_TOKENS) @ self.encode(chunks, P_TOKENS).T


# ---- 평가 ----

def first_rank(scores: np.ndarray, gold: set[int]) -> int | None:
    """점수 순서(같으면 앞 조각 먼저)에서 첫 근거 조각의 순위(1부터)."""
    for k, i in enumerate(sorted(range(len(scores)), key=lambda i: (-scores[i], i)), start=1):
        if i in gold:
            return k
    return None


def evaluate(questions: list[dict], texts: dict[str, dict[str, str]], retrievers: list[str],
             chunkers: dict[str, Callable[[str], list[str]]], dense: Dense | None = None) -> tuple[list[dict], list[dict]]:
    """(질문별 줄, 조각 통계) — 계통·자르기·검색기마다 첫 근거 조각 순위와 근거가 글·조각에 있는지."""
    rows, stats = [], []
    queries = [q["q"] for q in questions]
    for system, docs in texts.items():
        whole = {d: norm(t) for d, t in docs.items()}
        in_text = [hit(whole.get(q["doc_id"], ""), q["ev"]) for q in questions]
        for chunker, cut in chunkers.items():
            chunks = [(d, c) for d, t in sorted(docs.items()) for c in cut(t)]
            normed = [norm(c) for _, c in chunks]
            gold = [{i for i, (d, _) in enumerate(chunks) if d == q["doc_id"] and hit(normed[i], q["ev"])} for q in questions]
            stats.append({"system": system, "chunker": chunker, "chunks": len(chunks),
                          "avg_chars": round(sum(len(c) for _, c in chunks) / max(len(chunks), 1))})
            body = [c for _, c in chunks]
            for retriever in retrievers:
                scores = bm25_scores(body, queries) if retriever == "bm25" else dense.scores(body, queries)
                for q, g, s, t in zip(questions, gold, scores, in_text):
                    rows.append({"retriever": retriever, "chunker": chunker, "system": system, "id": q["id"],
                                 "doc": q["doc"], "type": q["type"], "rank": first_rank(s, g), "in_chunks": bool(g),
                                 "in_text": t})
    return rows, stats


def summarize(rows: list[dict], key=("retriever", "chunker", "system")) -> dict:
    """묶음(기본: 검색기·자르기·계통) → 질문 수·R@5(전체·유형별)·MRR@10·근거가 글/조각에 있는 수."""
    groups = defaultdict(list)
    for r in rows:
        groups["|".join(r[k] for k in key)].append(r)
    out = {}
    for name, rs in groups.items():
        found = [r for r in rs if r["rank"] and r["rank"] <= K]
        out[name] = {
            "n": len(rs), "r5": len(found),
            "by_type": {t: [sum(r["type"] == t for r in found), sum(r["type"] == t for r in rs)] for t in TYPES},
            "mrr": round(sum(1 / r["rank"] for r in rs if r["rank"] and r["rank"] <= MRR_K) / len(rs), 3),
            "in_text": sum(bool(r.get("in_text")) for r in rs), "in_chunks": sum(bool(r.get("in_chunks")) for r in rs)}
    return out


def frac(a: int, b: int) -> str:
    return f"{a / b:.2f} ({a}/{b})" if b else "—"


def intro(questions: list[dict]) -> str:
    n = Counter(q["type"] for q in questions)
    return (f"질문 {len(questions)}개(" + "·".join(f"{t} {n[t]}" for t in TYPES) + f"), 문서 {len({q['doc'] for q in questions})}개."
            f" R@5 = 위 5개 조각 안에 근거 조각이 든 질문 비율, MRR@10 = 첫 근거 조각 순위의 역수 평균(10위 밖은 0).")


def render(summary: dict, stats: list[dict], questions: list[dict], systems: list[str], retrievers: list[str],
           chunkers: list[str], check: tuple[int, int], dense_name: str | None) -> str:
    st = {(s["system"], s["chunker"]): s for s in stats}
    lines = ["# RAG 하류 평가 축소판(D21) — 같은 자르기·같은 검색기", "", intro(questions),
             "검색기: " + ", ".join(RETRIEVER_NAMES[r] + (f" `{dense_name}`" if r == "bge-m3" and dense_name else "")
                                  for r in retrievers) + ". 네 문서 조각을 한 말뭉치로 찾는다.",
             (f"지표 점검 — 텍스트 레이어 글에 근거가 있는 문단·표 질문: {check[0]}/{check[1]} (모두여야 한다 — 아니면 정답이나 비교 규칙이 틀렸다)."
              if check[1] else "지표 점검 못 함 — 텍스트 레이어 계통이 없다(비교군 출력을 확인한다)."),
             "'근거가 글에'·'조각에' = 출력 글 전체·자른 조각 안에 근거가 있는 질문 수(R@5 의 위 한계).", ""]
    for retriever in retrievers:
        for chunker in chunkers:
            lines += [f"## {RETRIEVER_NAMES[retriever]} · {CHUNKERS[chunker][0]}", "",
                      "| 계통 | 조각 수 | 평균 글자 | R@5 | " + " | ".join(TYPES) + " | MRR@10 | 근거가 글에 | 근거가 조각에 |",
                      "|---|---|---|---|" + "---|" * len(TYPES) + "---|---|---|"]
            for system in systems:
                s = summary[f"{retriever}|{chunker}|{system}"]
                c = st[(system, chunker)]
                lines.append(f"| {NAMES[system]} | {c['chunks']} | {c['avg_chars']} | {frac(s['r5'], s['n'])} | "
                             + " | ".join(frac(*s["by_type"][t]) for t in TYPES)
                             + f" | {s['mrr']:.3f} | {s['in_text']}/{s['n']} | {s['in_chunks']}/{s['n']} |")
            lines.append("")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="RAG 하류 평가 축소판(D21) — 계통별 R@5·MRR@10")
    ap.add_argument("--kdp", type=Path, required=True, help="kdp 문서 폴더들이 든 폴더(document.md·document.rag.md)")
    ap.add_argument("--compare", type=Path, default=None, help="eval/compare.py -o 출력 폴더(텍스트 레이어·PaddleOCR-VL 기본·docling)")
    ap.add_argument("--questions", type=Path, default=QUESTIONS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--retrievers", default="bm25,bge-m3")
    ap.add_argument("--chunkers", default="lightrag,lightrag1200,structure")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    questions = load_questions(a.questions)
    retrievers = [r for r in a.retrievers.split(",") if r]
    chunkers = [c for c in a.chunkers.split(",") if c]
    if set(retrievers) - RETRIEVER_NAMES.keys() or set(chunkers) - CHUNKERS.keys():
        sys.exit(f"모르는 검색기·자르기: {sorted((set(retrievers) - RETRIEVER_NAMES.keys()) | (set(chunkers) - CHUNKERS.keys()))}")
    texts = system_texts(a.kdp, a.compare, {q["doc_id"] for q in questions})
    if not texts:
        sys.exit("평가할 계통이 없다")
    dense = Dense(a.model, a.device) if "bge-m3" in retrievers else None
    rows, stats = evaluate(questions, texts, retrievers, {c: CHUNKERS[c][1] for c in chunkers}, dense)
    summary = summarize(rows)
    layer = [r for r in rows if r["system"] == "textlayer" and r["type"] != "그림"
             and r["retriever"] == retrievers[0] and r["chunker"] == chunkers[0]]
    check = (sum(r["in_text"] for r in layer), len(layer))
    report = render(summary, stats, questions, list(texts), retrievers, chunkers, check, dense.name if dense else None)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.md").write_text(report + "\n", encoding="utf-8")
    (a.out / "report.json").write_text(json.dumps({"summary": summary, "stats": stats, "check": check,
                                                   "model": dense.name if dense else None}, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    (a.out / "ranks.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
