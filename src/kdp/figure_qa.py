"""그림 물음 틀(D20, 기획서 7.10) — 그림 해석 튜닝의 합성 학습 자료(data/synth/figures.py)·실제 그림 라벨(data/label/figures.py)·
평가(eval/figure_questions.py)가 같은 물음 꼴을 쓴다. 평가 물음(eval/judged/figure_questions.jsonl — 테스트_결과 11.6)과 글자까지
같다. 정답 맞춤(nkey·same_short)도 여기 하나로 둔다."""
from __future__ import annotations

import re

DIRS = ["북서쪽", "북쪽", "북동쪽", "서쪽", "가운데", "동쪽", "남서쪽", "남쪽", "남동쪽"]   # 지도 테두리 상자를 3×3 으로 나눈 방위(위가 북)
TREND = ["늘어남", "줄어듦", "늘다가 줄어듦", "줄다가 늘어남", "뚜렷한 흐름 없음"]
HEAD = ("아래 그림을 보고 물음에 답하세요.\n그림 제목: {caption}\n그림에서 읽은 글자(틀린 글자가 있을 수 있습니다): {text}\n\n"
        "물음: {q}\n{tail}")
CHOICE_TAIL = "보기: {opts}\n보기 번호 하나만 답하세요."
SHORT_TAIL = "그림에 적힌 이름(가로축 글자·항목 이름) 그대로 짧게 답하세요. 막대나 선이 여럿이면 그림 제목이 가리키는 주된 값으로 봅니다."
READ_TAIL = "그림에 적힌 그대로 짧게 답하세요."
QUESTIONS = {                         # 물음 갈래 → (물음, 보기 — None 이면 단답)
    "map_high": ("이 지도에서 값이 가장 높은(많은) 곳은 지도의 어느 쪽인가요?", DIRS),
    "map_low": ("이 지도에서 값이 가장 낮은(적은) 곳은 지도의 어느 쪽인가요?", DIRS),
    "chart_high": ("이 그래프에서 값이 가장 큰 때나 항목은 무엇인가요?", None),
    "chart_low": ("이 그래프에서 값이 가장 작은 때나 항목은 무엇인가요?", None),
    "chart_trend": ("이 그래프에서 값의 전체 흐름은 어떤가요?", TREND),
}
READS = {                             # 읽기 물음(잊음 막기 — 그림에 적힌 글자를 그대로)
    "chart_maxval": "이 그래프의 막대 위에 적힌 값 가운데 가장 큰 숫자는 무엇인가요?",   # 축 눈금 숫자는 아니다
    "chart_first": "이 그래프 가로축의 첫 번째 이름은 무엇인가요?",
    "map_unit": "이 지도에 적힌 단위는 무엇인가요?",
}


def figure_text(lines) -> str:
    """그림에서 읽은 글자 칸 — 줄을 ' / ' 로 잇고, 없으면 '(없음)'."""
    out = " / ".join(s.strip() for s in lines if s and s.strip())
    return out or "(없음)"


def prompt(qtype: str, caption: str, text: str, options: list[str] | None = None, question: str | None = None) -> str:
    """물음 하나 — 보기 물음은 options(섞은 순서)를 1부터 번호 매겨 붙인다. question 을 주면 물음 글만 바꾼다(학습용 다른 표현)."""
    if qtype in READS:
        q, tail = question or READS[qtype], READ_TAIL
    else:
        q0, opts = QUESTIONS[qtype]
        q = question or q0
        if opts is None:
            tail = SHORT_TAIL
        else:
            options = options or opts
            assert sorted(options) == sorted(opts), "보기는 정해진 보기를 섞은 것이어야 한다"
            tail = CHOICE_TAIL.format(opts=" ".join(f"{i + 1} {x}" for i, x in enumerate(options)))
    return HEAD.format(caption=caption or "(없음)", text=text or "(없음)", q=q, tail=tail)


def nkey(s: str) -> str:
    """정답 맞춤 열쇠 — 띄어쓰기·따옴표·괄호·가운뎃점과 마침표(소수점은 남긴다 — '1.5' 는 '15' 가 아니다)를 빼고, 물결·줄표를 '-' 로
    맞추며(모델·텍스트층이 여러 글자로 낸다) '년'·'세' 를 뗀다('2024년' = '2024')."""
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", "", s or "")
    s = re.sub(r"[\s\"',·‧()\[\]<>]", "", s)
    s = re.sub(r"[~∼〜～‐‑‒–—−－]", "-", s)
    return s.replace("년", "").replace("세", "")


def alts(s: str) -> set[str]:
    """'|' 로 이은 정답 여럿."""
    return {nkey(a) for a in (s or "").split("|") if nkey(a)}


def _inside(short: str, long_: str) -> bool:
    """short 가 long_ 안에 숫자 덩어리를 자르지 않고 들어 있나 — '3주차' 는 '13주차' 안에 있지 않다."""
    i = long_.find(short)
    while i >= 0:
        j = i + len(short)
        if not (short[0].isdigit() and i and long_[i - 1].isdigit()) and not (short[-1].isdigit() and j < len(long_) and long_[j].isdigit()):
            return True
        i = long_.find(short, i + 1)
    return False


def same_short(answer: str, keys: set[str]) -> bool:
    """단답이 정답 가운데 하나와 같은가 — 같거나, 한쪽이 다른 쪽을 숫자 덩어리를 자르지 않고 품고 짧은 쪽이 긴 쪽의 3분의 2 를 넘으면
    같다('강화군' 과 '강화군(7)'). 숫자만인 답은 똑같아야 한다('20' 은 '2020' 이 아니고, '8월' 은 '8월 초' 가 아니다 — 테스트_결과 11.6 의
    너그러운 채점)."""
    a = nkey(answer)
    if not a:
        return False
    for k in keys:
        if a == k:
            return True
        if a.isdigit() or k.isdigit():
            continue
        short, long_ = sorted((a, k), key=len)
        if len(short) > len(long_) * 2 / 3 and _inside(short, long_):
            return True
    return False
