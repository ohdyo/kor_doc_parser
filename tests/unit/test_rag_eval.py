"""RAG 하류 평가 축소판(eval/rag_eval.py·eval/lightrag_eval.py, D21) — 정답 비교 규칙, 자르기, BM25, 순위·요약, 질문 파일."""
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

EVAL = Path(__file__).parents[2] / "eval"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EVAL / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                       # lightrag_eval 이 rag_eval 을 이름으로 읽는다
    spec.loader.exec_module(mod)
    return mod


ev = _load("rag_eval")
lr = _load("lightrag_eval")


def test_norm_ignores_markup_and_spacing():
    md = ('<a id="p1.b01"></a>\n<!-- page 3 -->\n<table><tr><td>9월</td><td>21.3℃(+1.2℃) 7위</td></tr></table>\n'
          "1\\. 대구‧경북 &lt;총괄&gt;")
    t = ev.norm(md)
    assert ev.hit(t, [[ev.norm("21.3℃(+1.2℃)")], [ev.norm("9월")]])
    assert ev.hit(t, [[ev.norm("대구·경북")]])                  # 가운뎃점 표기 차이
    assert ev.hit(t, [[ev.norm("<총괄>")]])                     # 문자 참조 — 한글 꺾쇠는 태그가 아니다
    assert not ev.hit(t, [[ev.norm("21.3℃")], [ev.norm("8위")]])
    assert "page" not in t and "b01" not in t


def test_chunkers():
    assert ev.chunk_length("abcdefghij", 4, 1) == ["abcd", "defg", "ghij"]
    text = "# 제목\n\n" + "가" * 30 + "\n\n" + "나" * 30 + "\n\n## 둘째\n\n" + "다" * 80
    assert ev.chunk_structure(text, 70, 10) == ["# 제목\n\n" + "가" * 30 + "\n\n" + "나" * 30, "## 둘째",
                                                *ev.chunk_length("다" * 80, 70, 10)]
    anchored = '<a id="p1.b01"></a>\n## 둘째\n\n본문'                 # 앵커 줄 다음 줄의 제목도 새 조각을 연다
    assert ev.chunk_structure("앞 글\n\n" + anchored, 1000, 10) == ["앞 글", anchored]


def test_bm25_and_rank():
    chunks = ["말라리아 주의보 발령 기준", "충북 9월 평균기온 21.3℃", "대구 경북 강수량 역대 네 번째"]
    s = ev.bm25_scores(chunks, ["충북 평균기온은?", "말라리아 주의보"])
    assert s.shape == (2, 3) and s[0].argmax() == 1 and s[1].argmax() == 0
    assert ev.first_rank(np.array([0.1, 0.5, 0.5]), {2}) == 2          # 같은 점수는 앞 조각 먼저
    assert ev.first_rank(np.array([0.1, 0.5]), set()) is None


def test_evaluate_and_summarize():
    questions = [{"id": "a", "doc": "d1", "doc_id": "sha256:1", "type": "문단", "q": "충북 평균기온", "ev": [[ev.norm("21.3℃")]]},
                 {"id": "b", "doc": "d1", "doc_id": "sha256:1", "type": "그림", "q": "가뭄 지도", "ev": [[ev.norm("가뭄")]]}]
    texts = {"kdp": {"sha256:1": "충북 9월 평균기온 21.3℃\n\n그 밖의 글", "sha256:2": "대구 평균기온 21.3℃"}}
    rows, stats = ev.evaluate(questions, texts, ["bm25"], {"structure": ev.chunk_structure})
    r = {x["id"]: x for x in rows}
    assert r["a"]["rank"] == 1 and r["a"]["in_chunks"] and r["b"]["rank"] is None and not r["b"]["in_text"]
    assert stats == [{"system": "kdp", "chunker": "structure", "chunks": 2, "avg_chars": 18}]   # (24 + 13) / 2
    s = ev.summarize(rows)["bm25|structure|kdp"]
    assert s["r5"] == 1 and s["n"] == 2 and s["mrr"] == 0.5
    assert s["by_type"] == {"문단": [1, 1], "표": [0, 0], "그림": [0, 1]}
    other = [{**questions[0], "doc_id": "sha256:2", "doc": "d2"}]        # 다른 문서의 같은 글은 근거가 아니다
    rows, _ = ev.evaluate(other, {"kdp": {"sha256:1": "21.3℃"}}, ["bm25"], {"length": ev.chunk_length})
    assert rows[0]["rank"] is None


def test_texts_and_main(tmp_path, capsys):
    kdp = tmp_path / "kdp" / "x"
    kdp.mkdir(parents=True)
    (kdp / "document.json").write_text('{"doc_id": "sha256:' + "ab" * 32 + '"}', encoding="utf-8")
    (kdp / "document.md").write_text('<a id="p1.b01"></a>\n충북 9월 평균기온 21.3℃\n', encoding="utf-8")
    (kdp / "document.rag.md").write_text("충북 9월 평균기온 21.3℃\n", encoding="utf-8")
    cmp = tmp_path / "cmp" / ("ab" * 32) / "compare"
    cmp.mkdir(parents=True)
    (cmp / "paddle_default.json").write_text(json.dumps({"pages": {"1": [{"text": "블록 글", "body": True}]},
                                                         "markdown": ["# 1쪽", "2쪽"]}), encoding="utf-8")
    (cmp / "docling.json").write_text(json.dumps({"pages": {"2": [{"text": "둘째", "body": True}],
                                                            "1": [{"text": "첫째", "body": True}, {"text": "머리말", "body": False}]}}),
                                      encoding="utf-8")
    t = ev.compare_texts(tmp_path / "cmp")
    key = "sha256:" + "ab" * 32
    assert t["paddle_default"][key] == "# 1쪽\n\n2쪽" and t["docling"][key] == "첫째\n\n둘째" and t["textlayer"] == {}
    assert "기본 마크다운 없음" in capsys.readouterr().err
    q = tmp_path / "q.yaml"
    q.write_text(f'docs: {{d1: "{key}"}}\nquestions:\n'
                 '  - {id: a, doc: d1, type: 문단, page: 1, q: "충북 평균기온은?", a: "21.3℃", evidence: [["21.3℃"]]}\n',
                 encoding="utf-8")                                     # 흐름 표기 안 '?' 는 따옴표로(PyYAML 이 기호로 읽는다)
    ev.main(["--kdp", str(tmp_path / "kdp"), "--compare", str(tmp_path / "cmp"), "--questions", str(q),
             "--out", str(tmp_path / "out"), "--retrievers", "bm25", "--chunkers", "structure,length"])
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "| kdp document.rag.md |" in report and "텍스트 레이어 뺌" in capsys.readouterr().err
    rows = [json.loads(x) for x in (tmp_path / "out" / "ranks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {(r["system"], r["chunker"], r["rank"]) for r in rows} >= {("kdp", "structure", 1), ("kdp_rag", "length", 1)}
    q.write_text(q.read_text(encoding="utf-8").replace('[["21.3℃"]]', '[["-"]]'), encoding="utf-8")
    with pytest.raises(SystemExit):                                     # 비교 규칙으로 지우면 빈 근거
        ev.load_questions(q)


def test_question_file():
    """질문 파일(eval/golden/rag_questions.yaml) — 유형 셋이 다 있고, 개인정보(전화·이메일)가 없다."""
    qs = ev.load_questions(ev.QUESTIONS)
    assert len(qs) >= 50 and {q["type"] for q in qs} == set(ev.TYPES) and len({q["doc"] for q in qs}) == 4
    text = ev.QUESTIONS.read_text(encoding="utf-8")
    assert not re.search(r"\d{2,4}\s*-\s*\d{3,4}\s*-\s*\d{4}|@\w+\.\w+", text)


def test_lightrag_chunk_rank():
    chunks = [{"file_path": "test3", "content": "충북 강수량 21.6㎜"}, {"file_path": "test5", "content": "대구 강수량 21.6㎜"},
              {"file_path": "test5", "content": "<table><tr><td>대구</td><td>21.6㎜</td></tr></table>"}]
    evidence = [[ev.norm("21.6㎜")], [ev.norm("대구")]]
    assert lr.chunk_rank(chunks, "test5", evidence) == 2 and lr.chunk_rank(chunks, "test2", evidence) is None


def test_lightrag_runs_and_graph_hit():
    assert lr.parse_runs("kdp_rag:512, kdp:1200:F") == [("kdp_rag", 512, "R"), ("kdp", 1200, "F")]   # 넣기 기본은 R(구조 자르기)
    with pytest.raises(SystemExit):
        lr.parse_runs("kdp_rag:512:X")
    evidence = [[ev.norm("17.5")], [ev.norm("23.1")]]
    other = {"file_path": "test3", "description": "충북 17.5% 23.1℃"}                      # 다른 문서의 설명은 보지 않는다
    low = {"file_path": "test5<SEP>test4", "description": "최저 17.5℃"}
    assert not lr.graph_hit([other, low], "test5", evidence)
    assert lr.graph_hit([other, low, {"file_path": "test5", "description": "최고 23.1℃"}], "test5", evidence)   # 묶음마다 어느 설명에든
    assert not lr.graph_hit([{"file_path": "test5", "description": "17."}, {"file_path": "test5", "description": "5 23.1"}],
                            "test5", [[ev.norm("17.523.1")]])                             # 설명 경계에서 글자가 붙지 않는다



def test_variant_system(tmp_path):
    """--variant '이름=폴더' — 다른 설정으로 다시 조립한 kdp 출력의 document.rag.md 를 계통으로 더한다."""
    for folder, text in (("kdp", "가 표"), ("pipe", "가 파이프")):
        d = tmp_path / folder / "x"
        d.mkdir(parents=True)
        (d / "document.json").write_text('{"doc_id": "sha256:1"}', encoding="utf-8")
        (d / "document.md").write_text("md", encoding="utf-8")
        (d / "document.rag.md").write_text(text, encoding="utf-8")
    variants = ev.parse_variants([f"kdp_rag_pipe={tmp_path / 'pipe'}"])
    assert ev.system_texts(tmp_path / "kdp", None, {"sha256:1"}, ("kdp", "kdp_rag"), variants) == {
        "kdp": {"sha256:1": "md"}, "kdp_rag": {"sha256:1": "가 표"}, "kdp_rag_pipe": {"sha256:1": "가 파이프"}}
    with pytest.raises(SystemExit):
        ev.parse_variants(["kdp=x"])                                                   # 이미 있는 계통 이름
    assert lr.parse_runs("kdp_rag_pipe:512:R", {**ev.NAMES, **variants}) == [("kdp_rag_pipe", 512, "R")]
    assert lr.run_label("kdp_rag", 512, "F") == "kdp_rag:512:F" and lr.run_name("kdp_rag_pipe:512:R").endswith("R 구조 자르기")
