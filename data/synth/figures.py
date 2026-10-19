"""합성 그래프·지도와 물음·정답(그림 해석 튜닝 D20, 기획서 7.10) — 정답은 그린 값에서 코드가 계산한다.
그래프: 막대·묶음 막대·선·막대+선(두 축)·평년선, 가로축은 해·월·순·주·날짜·항목. 물음은 가장 큰·작은 때나 항목(단답), 전체 흐름(보기),
가장 큰 값 읽기. 지도: 무작위 땅 모양을 시군처럼 나눠 단계 색으로 칠한 단계구분도와 등치선 채움 지도. 물음은 가장 높은·낮은 곳의 방위,
범례 단위 읽기. 평가 문서(test2~test5)의 그림은 본뜨지 않고, 지명·항목 이름은 지어낸다. 물음 꼴은 kdp.figure_qa(평가와 같다).
사용(GPU 서버 kdp-paddle env, matplotlib 필요): python data/synth/figures.py OUT_DIR --charts 1000 --maps 1000 --seed 0
출력: OUT_DIR/imgs/*.png, OUT_DIR/train.jsonl·val.jsonl — 줄마다 {"image", "prompt", "answer", "qtype", "meta"}(image 는 OUT_DIR 기준)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from kdp.figure_qa import DIRS, QUESTIONS, READS, TREND, figure_text, prompt  # noqa: E402

FONT_PATHS = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"]
VAL_EVERY = 20                # 그림 20장마다 1장은 검증용
MARGIN = 0.08                 # 가장 큰·작은 값과 둘째 값의 차이(값 범위에 대한 몫) — 눈으로 가려지게
EDGE = 0.05                   # 지도 방위 가름선(1/3·2/3)에서 이만큼 떨어져야 정답으로 쓴다(테두리 상자 몫)
PARAPHRASE = 0.3              # 물음 표현을 바꿔 쓰는 몫(틀을 외우지 않게) — 나머지는 평가와 같은 물음
READ_SHARE = 0.25             # 읽기 물음을 붙이는 몫(잊음 막기)
OCR_DROP, OCR_TYPO = 0.2, 0.05   # 그림에서 읽은 글자 흉내 — 줄 빠짐·한 글자 틀림
PARAPHRASES = {
    "map_high": ["이 지도에서 가장 높게(많게) 나타난 곳은 어느 쪽인가요?", "지도에서 값이 가장 큰 지역은 지도의 어느 쪽에 있나요?"],
    "map_low": ["이 지도에서 가장 낮게(적게) 나타난 곳은 어느 쪽인가요?", "지도에서 값이 가장 작은 지역은 지도의 어느 쪽에 있나요?"],
    "chart_high": ["그래프에서 가장 높은 값이 나타난 때나 항목은 무엇인가요?", "이 그림에서 값이 제일 큰 것은 어느 때나 항목인가요?"],
    "chart_low": ["그래프에서 가장 낮은 값이 나타난 때나 항목은 무엇인가요?", "이 그림에서 값이 제일 작은 것은 어느 때나 항목인가요?"],
    "chart_trend": ["그래프의 값은 전체적으로 어떻게 변하나요?", "이 그래프 값의 큰 흐름은 무엇인가요?"],
}

# ---------------------------------------------------------------- 이름·값


SYL1 = list("한새별솔가누다미보하온늘마라시청푸") + ["아름", "여울", "구름"]
SYL2 = ["별", "솔", "내", "들", "뫼", "결", "빛", "가람", "바다", "나래", "누리", "미르", "샘", "길", "벌"]
CATS = {                       # 항목 갈래 → (이름들, 순서가 있나)
    "연령": (["0-9세", "10-19세", "20-29세", "30-39세", "40-49세", "50-59세", "60-69세", "70-79세", "80세 이상"], True),
    "장소": (["실외작업장", "논밭", "길가", "운동장", "산", "강가·해변", "비닐하우스", "집", "실내작업장", "주거지 주변", "기타"], False),
    "질환": (["열탈진", "열사병", "열경련", "열실신", "열부종", "기타"], False),
    "품목": (["쌀", "보리", "콩", "감자", "배추", "무", "사과", "배", "포도", "귤", "고추", "마늘"], False),
    "업종": (["제조업", "건설업", "도소매업", "숙박음식점업", "운수업", "정보통신업", "금융업", "교육서비스업", "보건업"], False),
}
MEASURES = [                   # 그래프 값 — (이름, 단위, 꼴, (최소, 최대))
    ("환자 수", "명", "count", (5, 5000)), ("발생 건수", "건", "count", (3, 900)), ("평균기온", "℃", "dec1", (5, 33)),
    ("강수량", "mm", "dec1", (0, 400)), ("발생률", "%", "dec1", (0.1, 40)), ("모기지수", "", "dec2", (0.01, 2.5)),
    ("해수면 온도", "℃", "dec1", (15, 29)), ("폭염일수", "일", "count", (0, 31)), ("판매량", "톤", "count", (10, 3000)),
]
MAP_MEASURES = [               # 지도 값 — (이름, 단위, (최소, 최대), 색표)
    ("평균기온", "℃", (12, 26), "YlOrRd"), ("평년 대비 기온 편차", "℃", (-3, 3), "RdBu_r"), ("강수량", "mm", (0, 300), "Blues"),
    ("강수량 퍼센타일", "", (0, 100), "BrBG"), ("기상가뭄 발생일수", "일", (0, 30), "YlOrBr"), ("인구 10만명당 발생률", "명", (0, 20), "Reds"),
    ("해수면 온도", "℃", (16, 29), "turbo"), ("전년 대비 해수면 온도 편차", "℃", (-3, 3), "RdBu_r"), ("적설량", "cm", (0, 40), "PuBu"),
]


def place_names(rng: random.Random, n: int, suffixes=("시", "군", "구")) -> list[str]:
    out = []
    while len(out) < n:
        s = rng.choice(SYL1) + rng.choice(SYL2) + rng.choice(suffixes)
        if s not in out:
            out.append(s)
    return out


def fmt(v: float, kind: str) -> str:
    return f"{int(round(v)):,}" if kind == "count" else (f"{v:.2f}" if kind == "dec2" else f"{v:.1f}")


def ocr(rng: random.Random, items: list[str]) -> str:
    """그림에서 읽은 글자 흉내 — 줄을 몇 개 빼고, 몇 줄은 한 글자를 빼거나 숫자 하나를 바꾼다."""
    out = []
    for s in items:
        if not s or rng.random() < OCR_DROP:
            continue
        if rng.random() < OCR_TYPO and len(s) > 2:
            i = rng.randrange(len(s))
            s = s[:i] + (str(rng.randrange(10)) if s[i].isdigit() else "") + s[i + 1:]
        out.append(s)
    return figure_text(out)


# ---------------------------------------------------------------- 그래프


def time_labels(rng: random.Random, kind: str) -> tuple[list[str], list[bool]]:
    """가로축 이름과 보일지 — 긴 해 축은 홀수 해만 적는 꼴도 있다."""
    if kind == "year":
        start = rng.randint(1973, 2016)
        n = rng.randint(8, min(54, 2026 - start + 1))
        suf = "년" if rng.random() < 0.2 else ""                         # 한 그래프 안에서는 한 꼴로
        labels = [f"{start + i}{suf}" for i in range(n)]
        step = 2 if n > 20 and rng.random() < 0.5 else 1
        show = [i % step == 0 for i in range(n)]
        return labels, show
    if kind == "month":
        n = rng.randint(6, 12)
        m0 = rng.randint(1, 13 - n)
        return [f"{m0 + i}월" for i in range(n)], [True] * n
    if kind == "xun":
        n = rng.randint(6, 15)
        m = rng.randint(4, 8)
        names = [f"{m + i // 3}월 {('초순', '중순', '하순')[i % 3]}" for i in range(n)]
        return names, [True] * n
    if kind == "week":
        n = rng.randint(8, 25)
        w0 = rng.randint(1, 53 - n)
        suf = rng.choice(["주", "주차"])
        return [f"{w0 + i}{suf}" for i in range(n)], [True] * n
    m = rng.randint(1, 12)                                    # date — 그달 날 수를 넘지 않게
    n = rng.randint(15, min(30, (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)[m - 1]))
    style = rng.choice(["dot", "dash", "il"])
    labels = [(f"{m}.{d}" if style == "dot" else f"{m:02d}-{d:02d}" if style == "dash" else f"{d}일") for d in range(1, n + 1)]
    step = rng.choice([1, 1, 2, 3])
    return labels, [i % step == 0 for i in range(n)]


def shape_values(rng: random.Random, np_rng, trend: str, n: int) -> np.ndarray | None:
    """흐름 꼴의 값(0~1 근처) — 노이즈를 넣은 뒤에도 그 꼴인지 확인하고, 아니면 None."""
    t = np.linspace(0, 1, n)
    if trend in ("늘어남", "줄어듦"):
        base = 0.15 + 0.7 * t ** rng.uniform(0.6, 1.8)
        base = base if trend == "늘어남" else base[::-1]
    elif trend in ("늘다가 줄어듦", "줄다가 늘어남"):
        p, s = rng.uniform(0.3, 0.7), rng.uniform(0.12, 0.28)
        base = 0.15 + 0.75 * np.exp(-((t - p) ** 2) / (2 * s * s))
        base = base if trend == "늘다가 줄어듦" else 1.05 - base
    else:
        base = 0.5 + 0.18 * np.sin(2 * np.pi * rng.uniform(1.5, 3.5) * t + rng.uniform(0, 6.3))
    v = base + np_rng.normal(0, rng.uniform(0.03, 0.07), n)
    k = max(2, n // 3)
    a, mid, b = v[:k].mean(), v[k:n - k].mean() if n - 2 * k > 0 else v.mean(), v[-k:].mean()
    span = v.max() - v.min()
    if span <= 0:
        return None
    a, mid, b = (a - v.min()) / span, (mid - v.min()) / span, (b - v.min()) / span
    i_max, i_min = int(v.argmax()), int(v.argmin())
    ok = {"늘어남": b - a >= 0.4 and b >= mid >= a - 0.05,
          "줄어듦": a - b >= 0.4 and a >= mid >= b - 0.05,
          "늘다가 줄어듦": 0.2 * n <= i_max <= 0.8 * n and mid - a >= 0.3 and mid - b >= 0.3,
          "줄다가 늘어남": 0.2 * n <= i_min <= 0.8 * n and a - mid >= 0.3 and b - mid >= 0.3,
          "뚜렷한 흐름 없음": abs(b - a) < 0.15 and abs(mid - a) < 0.2 and abs(mid - b) < 0.2}[trend]
    return v if ok else None


def separate(v: np.ndarray, lo_floor: float | None) -> np.ndarray:
    """가장 큰 값과 가장 작은 값이 둘째 값과 MARGIN 이상 떨어지게 한다(눈으로 가려지게)."""
    v = v.astype(float).copy()
    span = v.max() - v.min()
    order = np.argsort(v)
    gap = MARGIN * span
    if v[order[-1]] - v[order[-2]] < gap:
        v[order[-1]] = v[order[-2]] + gap
    if v[order[1]] - v[order[0]] < gap:
        v[order[0]] = v[order[1]] - gap
        if lo_floor is not None and v[order[0]] < lo_floor:     # 0 아래로 못 가면 나머지를 올린다
            v += lo_floor - v[order[0]]
    return v


def unique_extremes(vals: list[str]) -> bool:
    """적힌 꼴(반올림 뒤)로도 가장 큰·작은 값이 하나뿐인가."""
    f = [float(x.replace(",", "")) for x in vals]
    return f.count(max(f)) == 1 and f.count(min(f)) == 1


def make_chart(rng: random.Random, np_rng, path: Path, plt) -> dict | None:
    kind = rng.choices(["year", "month", "xun", "week", "date", "cat"], [3, 2, 2, 2, 2, 4])[0]
    name, unit, vkind, (lo, hi) = rng.choice(MEASURES)
    if kind == "cat":
        cat = rng.choice(list(CATS) + ["지역"])
        if cat == "지역":
            labels = place_names(rng, rng.randint(4, 12))
        else:
            pool, ordered = CATS[cat]
            k = rng.randint(4, len(pool))
            labels = pool[:k] if ordered else rng.sample(pool, k)
        show, trend = [True] * len(labels), None
        base = np.array([rng.uniform(0.1, 1.0) for _ in labels])
    else:
        labels, show = time_labels(rng, kind)
        trend = rng.choice(TREND)
        base = None
        for _ in range(30):
            base = shape_values(rng, np_rng, trend, len(labels))
            if base is not None:
                break
        if base is None:
            return None
    a, b = sorted(rng.uniform(lo, hi) for _ in range(2))
    if b - a < 0.25 * (hi - lo):
        a, b = lo, hi
    vals = a + (base - base.min()) / (base.max() - base.min()) * (b - a)
    vals = separate(vals, 0.0 if lo >= 0 else None)
    shown = [fmt(v, vkind) for v in vals]
    if not unique_extremes(shown):
        return None
    vals = np.array([float(s.replace(",", "")) for s in shown])
    i_max, i_min = int(vals.argmax()), int(vals.argmin())

    mode = rng.choices(["single", "compare", "twin", "normal"],
                       [4, 2 if kind != "year" else 0, 2, 1 if vkind != "count" else 0])[0]   # 해 축을 두 해로 묶어 견주는 그림은 없다
    main_bar = rng.random() < (0.75 if mode == "twin" else 0.55)
    year = rng.randint(2018, 2026)
    lab1 = f"{year}년" if mode == "compare" else name
    period = {"year": "연도별", "month": "월별", "xun": "순별", "week": "주별", "date": "일별", "cat": f"{cat}별" if kind == "cat" else ""}[kind]
    if mode == "compare":
        title = f"{period} {name} 현황({year - 1}~{year}년)"
    elif mode == "twin":
        name2, unit2 = rng.choice([p for p in [("평균기온", "℃"), ("평균 최고기온", "℃"), ("강수량", "mm"), ("발생률", "%")] if p[0] != name])
        title = f"{period} {name}{'과' if (ord(name[-1]) - 0xAC00) % 28 else '와'} {name2}"      # 받침 있으면 '과'
    else:
        title = f"{period} {name} {rng.choice(['현황', '추이', '분포', '변화'])}"

    w, h = rng.uniform(6.0, 9.0), rng.uniform(3.2, 5.0)
    fig, ax = plt.subplots(figsize=(w, h), dpi=rng.randint(100, 140))
    x = np.arange(len(labels))
    colors = ["#2f6db5", "#e07b39", "#3a9a5b", "#8e5bb5", "#c94a4a", "#4aa3c9", "#7f7f7f"]
    c1, c2, c3 = rng.sample(colors, 3)
    texts = []
    if main_bar:
        width = 0.38 if mode == "compare" else rng.uniform(0.5, 0.75)
        off = -0.2 if mode == "compare" else 0.0
        bar_colors = [c1] * len(x)
        if rng.random() < 0.15:
            bar_colors[i_max] = "#f2c12e"                      # 가장 큰 막대만 다른 색
        ax.bar(x + off, vals, width, color=bar_colors, label=lab1)
    else:
        ax.plot(x, vals, color=c1, marker="o" if rng.random() < 0.6 else None, ms=3, lw=1.6, label=lab1)
    legend_labels = [lab1]
    if mode == "compare":           # 지난해도 같은 칸이 가장 크고 작으며 올해 막대 안쪽 — 어느 계열로 봐도(모든 막대 가운데로 봐도) 정답이 같다
        gap = MARGIN * (vals.max() - vals.min())
        other = np.clip(vals * np_rng.uniform(0.85, 1.15, len(vals)), vals[i_min] + gap, vals[i_max] - gap)
        other[i_max], other[i_min] = vals[i_max] - gap / 2, vals[i_min] + gap / 2
        if main_bar:
            ax.bar(x + 0.2, other, 0.38, color=c2, label=f"{year - 1}년")
        else:
            ax.plot(x, other, color=c2, ls="--", lw=1.4, label=f"{year - 1}년")
        legend_labels.append(f"{year - 1}년")
    elif mode == "normal":
        ax.plot(x, vals * np_rng.uniform(0.85, 1.15) + np_rng.normal(0, 0.02 * (b - a), len(x)), color="#555555", ls="--",
                lw=1.2, label="평년")
        legend_labels.append("평년")
    elif mode == "twin":
        ax2 = ax.twinx()
        ax2.plot(x, np_rng.uniform(0.3, 1.0) * np.cumsum(np_rng.normal(0, 1, len(x))) + 20, color=c3, marker="o", ms=3, lw=1.4,
                 label=name2)
        ax2.set_ylabel(f"{name2}({unit2})", fontsize=8)
        legend_labels.append(name2)
        texts.append(f"{name2}({unit2})")
    value_labels = main_bar and rng.random() < 0.35
    if value_labels:
        for xi, s in zip(x, shown):
            ax.text(xi + (-0.2 if mode == "compare" else 0), float(s.replace(",", "")), s, ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([lab if sh else "" for lab, sh in zip(labels, show)], fontsize=7,
                       rotation=0 if len(labels) <= 8 else rng.choice([45, 90]))
    if unit:
        ax.set_ylabel(f"{name}({unit})", fontsize=8)
        texts.append(f"{name}({unit})")
    legend_drawn = len(legend_labels) > 1 or rng.random() < 0.3
    if legend_drawn:                # 범례는 위 띠에 — 축 위끝을 30% 올려 비운 자리라 막대 끝·값 글자를 가리지 않는다
        handles = ax.get_legend_handles_labels()
        for a_ in [ax] + ([ax2] if mode == "twin" else []):
            y0, y1 = a_.get_ylim()
            a_.set_ylim(y0, y1 + 0.3 * (y1 - y0))
        if mode == "twin":
            h2 = ax2.get_legend_handles_labels()
            handles = (handles[0] + h2[0], handles[1] + h2[1])
        ax.legend(*handles, fontsize=7, loc=rng.choice(["upper left", "upper right", "upper center"]), ncol=len(handles[1]))
    title_inside = rng.random() < 0.4
    if title_inside:
        ax.set_title(title, fontsize=10)
    if rng.random() < 0.5:
        ax.yaxis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.canvas.draw()
    lo_y, hi_y = sorted(ax.get_ylim())                                     # 보이는 눈금만(로케이터는 범위 밖에도 하나씩 둔다)
    yticks = [t.get_text() for t, v in zip(ax.get_yticklabels(), ax.get_yticks()) if lo_y <= v <= hi_y and t.get_text()]
    fig.savefig(path)
    plt.close(fig)

    items = ([title] if title_inside else []) + (legend_labels if legend_drawn else []) + (shown if value_labels else []) + yticks + \
        [lab for lab, sh in zip(labels, show) if sh] + texts
    caption = title if (not title_inside or rng.random() < 0.6) else "(없음)"
    return {"kind": "chart", "axis": kind, "trend": trend, "labels": labels, "shown": show, "values": shown,
            "i_max": i_max, "i_min": i_min, "mode": mode, "value_labels": value_labels, "caption": caption,
            "text": ocr(rng, items)}


# ---------------------------------------------------------------- 지도


def direction(x: float, y: float) -> tuple[str, bool]:
    """테두리 상자 안의 몫 좌표(0~1, y 는 아래로) → 9방위와 가름선에서 EDGE 이상 떨어졌는지."""
    col, row = min(2, int(x * 3)), min(2, int(y * 3))
    clear = all(abs(v - c) >= EDGE for v in (x, y) for c in (1 / 3, 2 / 3))
    return DIRS[row * 3 + col], clear


def make_map(rng: random.Random, np_rng, path: Path, plt, style: str) -> dict | None:
    """style(단계구분도 choropleth·등치선 contour)은 부르는 쪽이 그림마다 정한다 — 다시 그릴 때 새로 뽑으면 검사가 까다로운 단계구분도가
    줄어든다."""
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.path import Path as MPath

    W, H = rng.randint(360, 520), rng.randint(320, 480)
    th = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    r = 0.42 * min(W, H) * (1 + sum(rng.uniform(0, 0.11 / (k - 1) ** 0.5) * np.cos(k * th + rng.uniform(0, 6.3)) for k in range(2, 7)))
    sx, sy = rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2)
    poly = np.stack([W / 2 + sx * r * np.cos(th), H / 2 + sy * r * np.sin(th)], axis=1)
    X, Y = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    mask = MPath(poly).contains_points(np.stack([X.ravel(), Y.ravel()], axis=1)).reshape(H, W)
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    bw, bh = x1 - x0, y1 - y0

    def rel(px, py):
        return (px - x0) / bw, (py - y0) / bh

    def plant(d):
        """그 방위 칸의 가운데 쪽(칸의 20~80%) 땅 위 한 점."""
        i = DIRS.index(d)
        row, col = divmod(i, 3)
        for _ in range(200):
            px = x0 + bw * (col + rng.uniform(0.2, 0.8)) / 3
            py = y0 + bh * (row + rng.uniform(0.2, 0.8)) / 3
            if mask[int(py), int(px)]:
                return px, py
        return None

    d_hi = rng.choice(DIRS)
    d_lo = rng.choice([d for d in DIRS if d != d_hi])
    hi_c, lo_c = plant(d_hi), plant(d_lo)
    if hi_c is None or lo_c is None:
        return None
    size = min(bw, bh)
    s_hi, s_lo = size * rng.uniform(0.07, 0.16), size * rng.uniform(0.07, 0.16)
    g = lambda c, s: np.exp(-((X - c[0]) ** 2 + (Y - c[1]) ** 2) / (2 * s * s))  # noqa: E731
    F = g(hi_c, s_hi) - g(lo_c, s_lo)
    for _ in range(rng.randint(0, 2)):                                     # 작은 봉우리·골
        c = (rng.uniform(x0, x1), rng.uniform(y0, y1))
        F += rng.uniform(-0.3, 0.3) * g(c, size * rng.uniform(0.1, 0.25))
    F += rng.uniform(-0.15, 0.15) * (X - x0) / bw + rng.uniform(-0.15, 0.15) * (Y - y0) / bh
    Fm = np.where(mask, F, np.nan)

    name, unit, (vlo, vhi), cmap_name = rng.choice(MAP_MEASURES)
    a, b = sorted(rng.uniform(vlo, vhi) for _ in range(2))
    if b - a < 0.4 * (vhi - vlo):
        a, b = vlo, vhi
    fmin, fmax = np.nanmin(Fm), np.nanmax(Fm)
    V = a + (Fm - fmin) / (fmax - fmin) * (b - a)
    n_cells = rng.randint(8, 24)
    seeds = []
    while len(seeds) < n_cells:
        i = rng.randrange(len(xs))
        seeds.append((xs[i] + 0.5, ys[i] + 0.5))
    seeds = np.array(seeds)
    lab = np.full((H, W), -1)
    d2 = (X[..., None] - seeds[:, 0]) ** 2 + (Y[..., None] - seeds[:, 1]) ** 2
    lab[mask] = d2[mask].argmin(axis=1)
    names = place_names(rng, n_cells)
    cmap = plt.get_cmap(cmap_name)
    decimals = 1 if (b - a) < 20 else 0

    if style == "choropleth":
        cell_v = np.array([np.nanmean(V[lab == k]) if (lab == k).any() else np.nan for k in range(n_cells)])
        cent = np.array([(X[lab == k].mean(), Y[lab == k].mean()) if (lab == k).any() else (np.nan, np.nan) for k in range(n_cells)])
        ok = ~np.isnan(cell_v)
        n_bins = rng.randint(5, 7)
        edges = np.linspace(np.nanmin(cell_v), np.nanmax(cell_v), n_bins + 1)
        top = [k for k in range(n_cells) if ok[k] and cell_v[k] >= edges[-2]]
        bot = [k for k in range(n_cells) if ok[k] and cell_v[k] <= edges[1]]
        dirs_top = {direction(*rel(*cent[k])) for k in top}
        dirs_bot = {direction(*rel(*cent[k])) for k in bot}
        if dirs_top != {(d_hi, True)} or dirs_bot != {(d_lo, True)}:                # 가장 진한·옅은 칸이 모두 그 방위 한가운데 쪽에
            return None
        bins = np.clip(np.digitize(cell_v, edges[1:-1]), 0, n_bins - 1)
        colors = cmap(np.linspace(0.12, 0.92, n_bins))
        img = np.ones((H, W, 4))
        for k in range(n_cells):
            if ok[k]:
                img[lab == k] = colors[bins[k]]
        edge_px = mask & ((np.roll(lab, 1, 0) != lab) | (np.roll(lab, 1, 1) != lab))
        img[edge_px] = (0.35, 0.35, 0.35, 1)
        legend = [f"{edges[i]:.{decimals}f}~{edges[i + 1]:.{decimals}f}" for i in range(n_bins)]
    else:
        levels = np.linspace(np.nanmin(V), np.nanmax(V), rng.randint(6, 9))
        legend = [f"{v:.{decimals}f}" for v in levels]

    fig, ax = plt.subplots(figsize=(W / 100 * rng.uniform(1.4, 1.9), H / 100 * rng.uniform(1.4, 1.9)), dpi=rng.randint(100, 130))
    texts = []
    if style == "choropleth":
        ax.imshow(img, interpolation="nearest")
        if rng.random() < 0.5:                                             # 색 막대 / 칸 범례
            sm = plt.cm.ScalarMappable(cmap=ListedColormap(colors), norm=BoundaryNorm(edges, n_bins))
            cb = fig.colorbar(sm, ax=ax, fraction=0.04, ticks=edges)
            cb.ax.set_yticklabels([f"{e:.{decimals}f}" for e in edges], fontsize=7)
            texts += [f"{e:.{decimals}f}" for e in edges]
        else:
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(color=colors[i], label=legend[i]) for i in range(n_bins)][::-1], fontsize=6,   # 지도 밖 — 모서리 칸을 덮지 않게
                      loc="center left", bbox_to_anchor=(1.0, 0.5))
            texts += legend[::-1]
    else:
        cs = ax.contourf(X, Y, V, levels=levels, cmap=cmap)
        if rng.random() < 0.6:
            cl = ax.contour(X, Y, V, levels=levels[1:-1], colors="k", linewidths=0.5)
            ax.clabel(cl, fontsize=6, fmt=f"%.{decimals}f")
            texts += [f"{v:.{decimals}f}" for v in levels[1:-1]]
        if rng.random() < 0.5:
            edge_px = mask & ((np.roll(lab, 1, 0) != lab) | (np.roll(lab, 1, 1) != lab))
            ov = np.zeros((H, W, 4))
            ov[edge_px] = (0.4, 0.4, 0.4, 0.8)
            ax.imshow(ov, interpolation="nearest")
        cb = fig.colorbar(cs, ax=ax, fraction=0.04, format=f"%.{decimals}f")      # 읽은 글자 흉내(legend)와 같은 자릿수
        cb.ax.tick_params(labelsize=7)
        texts += [t for t in legend]
        hi_px = np.unravel_index(np.nanargmax(V), V.shape)
        lo_px = np.unravel_index(np.nanargmin(V), V.shape)
        if direction(*rel(hi_px[1], hi_px[0])) != (d_hi, True) or direction(*rel(lo_px[1], lo_px[0])) != (d_lo, True):
            plt.close(fig)
            return None
    ax.contour(mask.astype(float), levels=[0.5], colors="k", linewidths=1.0)
    show_names = rng.random() < 0.6
    if show_names:
        for k in range(n_cells):
            if (lab == k).sum() > 300:
                cx, cy = X[lab == k].mean(), Y[lab == k].mean()
                ax.text(cx, cy, names[k], fontsize=5, ha="center", va="center")
                texts.append(names[k])
    unit_text = f"(단위: {unit})" if unit else ""
    if unit and rng.random() < 0.7:
        ax.text(1.0, -0.02, unit_text, transform=ax.transAxes, fontsize=7, ha="right", va="top")
        texts.append(unit_text)
    else:
        unit_text = ""
    region = rng.choice(SYL1) + rng.choice(SYL2) + "도"
    year, month = rng.randint(2020, 2026), rng.randint(1, 12)
    title = f"{year}년 {month}월 {region} {name} 분포도"
    title_inside = rng.random() < 0.35
    if title_inside:
        ax.set_title(title, fontsize=9)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    caption = (title + (f"(단위: {unit})" if unit and rng.random() < 0.3 else "")) if (not title_inside or rng.random() < 0.6) else "(없음)"
    return {"kind": "map", "style": style, "measure": name, "unit": unit, "unit_text": unit_text, "d_hi": d_hi, "d_lo": d_lo,
            "caption": caption, "text": ocr(rng, ([title] if title_inside else []) + texts)}


# ---------------------------------------------------------------- 물음


def questions(rng: random.Random, m: dict) -> list[dict]:
    """그림 하나의 물음·정답 — 보기 물음은 보기 순서를 섞고 정답은 보기 번호, 단답은 그림에 적힌 이름."""
    out = []

    def ask(qtype, answer_text):
        q0, opts = QUESTIONS[qtype]
        q = rng.choice(PARAPHRASES[qtype]) if rng.random() < PARAPHRASE else None
        if opts is None:
            out.append({"qtype": qtype, "prompt": prompt(qtype, m["caption"], m["text"], question=q), "answer": answer_text})
        else:
            order = rng.sample(opts, len(opts))
            out.append({"qtype": qtype, "prompt": prompt(qtype, m["caption"], m["text"], order, question=q),
                        "answer": str(order.index(answer_text) + 1)})

    if m["kind"] == "chart":
        ask("chart_high", m["labels"][m["i_max"]])
        if rng.random() < 0.7:
            ask("chart_low", m["labels"][m["i_min"]])
        if m["trend"]:
            ask("chart_trend", m["trend"])
        if m["value_labels"]:                                             # 읽기 — 막대 위에 적힌 가장 큰 값
            out.append({"qtype": "chart_maxval", "prompt": prompt("chart_maxval", m["caption"], m["text"]),
                        "answer": m["values"][m["i_max"]]})
        if rng.random() < READ_SHARE:                                     # 읽기 — 가로축 첫 이름(늘 적혀 있다)
            out.append({"qtype": "chart_first", "prompt": prompt("chart_first", m["caption"], m["text"]),
                        "answer": m["labels"][0]})
    else:
        ask("map_high", m["d_hi"])
        ask("map_low", m["d_lo"])
        if m["unit_text"] and rng.random() < 2 * READ_SHARE:
            out.append({"qtype": "map_unit", "prompt": prompt("map_unit", m["caption"], m["text"]), "answer": m["unit"]})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--charts", type=int, default=1000)
    ap.add_argument("--maps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--font", default=None)
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = a.font or next((p for p in FONT_PATHS if Path(p).exists()), None)
    if not font:
        sys.exit("한글 글꼴이 없다 — --font 로 Noto Sans CJK 같은 글꼴 파일을 준다")
    font_manager.fontManager.addfont(font)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    (a.out / "imgs").mkdir(parents=True, exist_ok=True)
    jobs = [("c", i) for i in range(a.charts)] + [("m", i) for i in range(a.maps)]
    n_img = n_skip = 0
    count = Counter()
    with open(a.out / "train.jsonl", "w", encoding="utf-8") as tr, open(a.out / "val.jsonl", "w", encoding="utf-8") as va:
        for kind, i in jobs:
            for attempt in range(50):                                     # 정답이 분명하지 않은 그림은 버리고 다시 그린다(같은 꼴 안에서)
                rng = random.Random(f"{a.seed}:{kind}:{i}:{attempt}")
                np_rng = np.random.default_rng(rng.randrange(2 ** 32))
                rel = f"imgs/{kind}{i:05d}.png"
                m = (make_chart(rng, np_rng, a.out / rel, plt) if kind == "c" else
                     make_map(rng, np_rng, a.out / rel, plt, ("choropleth", "contour")[i % 2]))
                if m is not None:
                    break
                n_skip += 1
            else:
                count["포기"] += 1
                continue
            f = va if n_img % VAL_EVERY == VAL_EVERY - 1 else tr
            for q in questions(rng, m):
                f.write(json.dumps({"image": rel, **q, "meta": m}, ensure_ascii=False) + "\n")
            n_img += 1
            count.update([m["mode"]] if kind == "c" else [m["style"], "가장 높은 곳 " + m["d_hi"]])
    print(f"그림 {n_img}장(다시 그린 것 {n_skip}) → {a.out}")
    print("갈래 수(쏠림 확인):", dict(sorted(count.items())))


if __name__ == "__main__":
    main()
