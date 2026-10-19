"""그림 해석 튜닝(D20, 기획서 7.10) — 물음 틀이 평가 물음과 같은지, 합성 그림의 정답이 그린 값과 맞는지, 라벨 합치기·평가 채점."""
import importlib.util
import json
import random
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kdp.figure_qa import DIRS, QUESTIONS, TREND, alts, nkey, prompt, same_short

ROOT = Path(__file__).parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


synth = _load("kdp_synth_figures", "data/synth/figures.py")
label = _load("kdp_label_figures", "data/label/figures.py")
fq = _load("kdp_figure_questions", "eval/figure_questions.py")
HEAD_RE = re.compile(r"그림 제목: (.*)\n그림에서 읽은 글자\(틀린 글자가 있을 수 있습니다\): (.*)\n\n물음: ", re.S)


def opts(row):
    """물음 글의 보기('1 북서쪽 2 북쪽 …' — 보기 글에 빈칸이 있을 수 있다)."""
    line = re.search(r"보기: (.*)\n", row["prompt"]).group(1)
    return [t for _, t in re.findall(r"(\d) (.+?)(?= \d |$)", line)]


def test_prompt_matches_eval_questions():
    """학습·라벨 물음이 평가 물음(eval/judged/figure_questions.jsonl)과 글자까지 같다."""
    for q in fq.load_questions():
        cap, text = HEAD_RE.search(q["prompt_A"]).groups()
        assert prompt(q["qtype"], cap, text, q["options_A"]) == q["prompt_A"]
        if q["options_B"]:
            assert prompt(q["qtype"], cap, text, q["options_B"]) == q["prompt_B"]


def test_same_short_is_strict_for_numbers_and_fragments():
    assert same_short("2024년", alts("2024"))
    assert same_short("8월 초순", alts("8월초순|8월 상순"))
    assert same_short("강화군(7)", alts("강화군"))
    assert not same_short("20", alts("2020"))                   # 숫자만인 답은 똑같아야
    assert not same_short("8월", alts("8월 7일|8월초"))           # 뭉뚱그린 답은 아니다
    assert nkey("60~69세") == nkey("60-69")
    assert not same_short("3주차", alts("13주차"))               # 숫자 덩어리를 자르면 다른 답
    assert not same_short("547명", alts("1,547명"))
    assert not same_short("5.2℃", alts("15.2℃"))
    assert nkey("1.5") != nkey("15") and nkey("8월.") == nkey("8월")   # 소수점은 남기고 맺음 마침표는 뗀다
    assert same_short("0～9세", alts("0-9세"))                   # 전각 물결


def test_direction_grid():
    assert synth.direction(0.1, 0.1) == ("북서쪽", True)
    assert synth.direction(0.5, 0.5) == ("가운데", True)
    assert synth.direction(0.9, 0.95) == ("남동쪽", True)
    assert synth.direction(0.34, 0.5)[1] is False                # 가름선 가까이는 정답으로 쓰지 않는다


@pytest.mark.parametrize("trend", TREND)
def test_trend_shapes_hold_after_noise(trend):
    rng, np_rng = random.Random(trend), np.random.default_rng(0)
    got = [v for v in (synth.shape_values(rng, np_rng, trend, 20) for _ in range(60)) if v is not None]
    assert got
    for v in got:
        k = len(v) // 3
        a, b = v[:k].mean(), v[-k:].mean()
        if trend == "늘어남":
            assert b > a
        elif trend == "줄어듦":
            assert a > b
        elif trend == "늘다가 줄어듦":
            assert 0.2 * len(v) <= v.argmax() <= 0.8 * len(v)
        elif trend == "줄다가 늘어남":
            assert 0.2 * len(v) <= v.argmin() <= 0.8 * len(v)


def test_separate_makes_extremes_clear():
    v = synth.separate(np.array([1.0, 5.0, 4.95, 2.0, 1.02]), 0.0)
    assert v.argmax() == 1 and v[1] - 4.95 >= synth.MARGIN * 4.0 - 1e-9
    assert v.argmin() == 0 and 1.02 - v[0] >= synth.MARGIN * 4.0 - 1e-9


def test_questions_answers_point_to_truth():
    rng = random.Random(0)
    chart = {"kind": "chart", "labels": ["1월", "2월", "3월"], "i_max": 2, "i_min": 0, "trend": "늘어남", "value_labels": True,
             "values": ["1", "5", "9"], "caption": "월별 환자 수", "text": "(없음)"}
    for _ in range(20):
        for q in synth.questions(rng, chart):
            want = {"chart_trend": None, "chart_high": "3월", "chart_low": "1월", "chart_maxval": "9", "chart_first": "1월"}[q["qtype"]]
            assert (opts(q)[int(q["answer"]) - 1] == "늘어남") if want is None else q["answer"] == want
    mp = {"kind": "map", "d_hi": "남동쪽", "d_lo": "북쪽", "unit": "일", "unit_text": "(단위: 일)", "caption": "분포도", "text": "(없음)"}
    seen = set()
    for _ in range(30):
        for q in synth.questions(rng, mp):
            if q["qtype"] == "map_unit":
                assert q["answer"] == "일"
                continue
            assert opts(q)[int(q["answer"]) - 1] == (mp["d_hi"] if q["qtype"] == "map_high" else mp["d_lo"])
            seen.add(q["answer"])
    assert len(seen) > 3                                          # 보기 순서를 섞어 정답 번호가 여러 가지다(쏠림 막기)


def test_generated_figures_have_consistent_answers(tmp_path):
    pytest.importorskip("matplotlib")
    synth.main([str(tmp_path), "--charts", "4", "--maps", "4", "--seed", "1"])
    rows = [json.loads(x) for f in ("train.jsonl", "val.jsonl") for x in open(tmp_path / f, encoding="utf-8")]
    assert rows and all((tmp_path / r["image"]).exists() for r in rows)
    for r in rows:
        m = r["meta"]
        if r["qtype"] in ("map_high", "map_low"):
            assert opts(r)[int(r["answer"]) - 1] == (m["d_hi"] if r["qtype"] == "map_high" else m["d_lo"])
        elif r["qtype"] == "chart_high":
            vals = [float(v.replace(",", "")) for v in m["values"]]
            assert vals.count(max(vals)) == 1 and vals.index(max(vals)) == m["i_max"] and r["answer"] == m["labels"][m["i_max"]]
        elif r["qtype"] == "chart_trend":
            assert opts(r)[int(r["answer"]) - 1] == m["trend"]


def test_label_agreement():
    ok = lambda a: {"ok_question": True, "answer": a, "note": ""}  # noqa: E731
    assert label.agreed("map_high", ok("남동쪽."), ok("남동쪽")) == "남동쪽"       # 정해진 보기 글로 되돌린다
    assert label.agreed("map_high", ok("남동쪽|동쪽"), ok("남동쪽")) is None        # 정답이 여럿이면 뺀다
    assert label.agreed("map_high", ok("남쪽"), ok("북쪽")) is None
    assert label.agreed("chart_trend", ok("늘다가줄어듦"), ok("늘다가 줄어듦")) == "늘다가 줄어듦"
    assert label.agreed("chart_high", ok("2024년|2024"), ok("2024")) == "2024년"
    assert label.agreed("chart_high", ok("2024년"), {"ok_question": False, "answer": "", "note": ""}) is None


def test_label_merge_writes_training_rows(tmp_path):
    pack = tmp_path / "pack"
    (pack / "imgs").mkdir(parents=True)
    img = pack / "imgs" / "a.png"
    img.write_bytes(b"x")
    item = {"doc": "d", "block": "p1.b1", "image": str(img), "caption": "제목", "figure_text": "(없음)", "before": "", "after": "",
            "questions": [{"qid": "d:p1.b1:map_high", "qtype": "map_high", "question": QUESTIONS["map_high"][0], "options": DIRS},
                          {"qid": "d:p1.b1:chart_high", "qtype": "chart_high", "question": QUESTIONS["chart_high"][0], "options": None}]}
    (pack / "figures.json").write_text(json.dumps([item], ensure_ascii=False), encoding="utf-8")
    labs = {"labels": [{"qid": "d:p1.b1:map_high", "ok_question": True, "answer": "서쪽", "note": ""},
                       {"qid": "d:p1.b1:chart_high", "ok_question": False, "answer": "", "note": ""}]}
    for who in ("a", "b"):
        (tmp_path / f"{who}.json").write_text(json.dumps(labs, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "data" / "train.jsonl"
    label.main(["merge", str(pack), str(tmp_path / "a.json"), str(tmp_path / "b.json"), "--out", str(out)])
    rows = [json.loads(x) for x in open(out, encoding="utf-8")]
    assert len(rows) == 1 and opts(rows[0])[int(rows[0]["answer"]) - 1] == "서쪽"
    assert (out.parent / rows[0]["image"]).resolve() == img.resolve()


def test_masker_hides_names_and_phones():
    clean = label.masker({"홍길동"})
    assert clean("홍길동 과장(043-123-4567)") == "○○○ 과장(전화번호 지움)"
    assert clean("문의 044‐201‐1234") == "문의 (전화번호 지움)"                    # 붙임표 갈래(파서 pii 와 같은 정규식)


def test_pii_names_catch_heads_and_contact_rows_only():
    """기관장 꼴·담당 부서 표의 전화번호 행 이름만 — 직함이 든 다른 낱말·지명은 남긴다(지어낸 이름으로 시험)."""
    B = lambda **k: SimpleNamespace(**{"type": "paragraph", "text": None, "figure_text": None, "pii": None, "html": None, **k})  # noqa: E731
    blocks = [B(text="질병관리청장(가나다)은 이날 회의를 열었다. 아울러 질병관리청장은 장관회의를 열고 청장년층에서 지원장비를 점검했다."),
              B(text="홍길동 기상청장은 말했다."),
              B(type="figure", figure_text="청주시 / 제천시"),
              B(type="table", html="<table><tr><td>청주시 보건소</td><td>과장</td><td>나라미</td><td>044‐201‐1234</td></tr></table>")]
    assert label.pii_names(SimpleNamespace(pages=[SimpleNamespace(blocks=blocks)])) == {"가나다", "홍길동", "나라미"}


def test_eval_keys_and_order_consistency():
    q = {"qid": "x", "doc": "test4", "qtype": "map_high", "options_A": DIRS, "options_B": DIRS[::-1],
         "keys": {"A": {"ok_question": True, "answer": "남동쪽|동쪽"}, "B": {"ok_question": True, "answer": "남동쪽"}}}
    assert fq.keys_of(q) == {"남동쪽"}                            # 두 평가자 모두 맞다고 본 것만
    good = [{"qid": "x", "order": "A", "text": "9", "tokens": [["9", -0.01]]}, {"qid": "x", "order": "B", "text": "1", "tokens": [["1", -0.02]]}]
    r = fq.score_rows([q], good)[0]
    assert r["ok"] and r["same"] and r["conf"] == -0.02
    biased = [{"qid": "x", "order": "A", "text": "9", "tokens": [["9", -0.01]]}, {"qid": "x", "order": "B", "text": "9", "tokens": [["9", -0.01]]}]
    r = fq.score_rows([q], biased)[0]
    assert r["ok"] and not r["same"]                             # 순서를 바꿔도 9번 — 쏠림이라 답하지 않은 것으로 본다


def test_eval_pass_needs_accuracy_and_coverage():
    hit = {"ok": True, "same": True, "conf": -0.01, "qtype": "map_high"}
    miss = {"ok": False, "same": True, "conf": -0.9, "qtype": "map_high"}
    assert fq.summary([hit] * 9 + [miss])["pass"]                 # 문턱 -0.02 에서 90% 답하고 모두 맞음
    assert not fq.summary([hit] * 3 + [dict(miss, conf=-0.01)] * 7)["pass"]


def test_eval_verdict_applies_dev_threshold_to_test():
    def rows(hits, misses):
        return [{"ok": True, "same": True, "conf": c, "qtype": "map_high"} for c in hits] + \
               [{"ok": False, "same": True, "conf": c, "qtype": "map_high"} for c in misses]
    dev = fq.summary(rows([-0.01] * 9, [-0.9]))                   # -1.0 에서도 90% — 가장 많이 답하는 문턱은 -1.0
    strict_only = fq.summary(rows([-0.01] * 6, [-0.5] * 4))      # -0.3 까지만 넘는다(자기 문턱으로는 통과)
    assert strict_only["pass"] and not fq.verdict(dev, strict_only)["pass"]
    v = fq.verdict(dev, fq.summary(rows([-0.01] * 9, [-0.9])))
    assert v["th"] == -1.0 and v["pass"]
    assert not fq.verdict(fq.summary(rows([-0.01] * 3, [-0.01] * 7)), dev)["chosen"]
