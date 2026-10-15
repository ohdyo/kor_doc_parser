"""정규화(기획서 4.3 규칙 8) — 출력은 NFC. NFKC 는 평가에만 쓴다(①→1 같은 변형이 생긴다)."""
from __future__ import annotations

import re
import unicodedata

from ..schema import Document

UNITS = ("백만원", "천원", "만원", "억원", "개월", "년", "월", "일", "시", "분", "초", "명", "개", "건", "원",
         "회", "차", "쪽", "곳", "대", "부", "권", "층", "호", "배", "점", "%")
PARTICLES = ("으로", "까지", "부터", "에서", "에게", "마다", "보다", "이상", "이하", "이내", "미만", "초과",
             "의", "에", "을", "를", "이", "가", "은", "는", "과", "와", "로", "도", "만", "씩", "간", "째", "당")
_UNIT = re.compile(r"(\d)\s+(" + "|".join(UNITS) + r")(?=(?:" + "|".join(PARTICLES) + r")?(?:[^가-힣]|$))")
_DOTS = str.maketrans({"ㆍ": "·", "∙": "·", "・": "·"})
_SPACED = re.compile(r"^(\s*(?:\d{1,2}[.)]|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]{1,3}[.)]|[가나다라마바사아자차카타파하][.)]"
                     r"|[\[(【<〈《「『])?\s*)([가-힣](?:\s+[가-힣]){1,5})(\s*[\])】>〉》」』]?\s*)$")
_MATH = re.compile(r"\$([^$\n]{1,30})\$")
_NL = re.compile(r"\\n(?![A-Za-z])")          # 표 칸: 글자 그대로의 \n — 뒤에 영문자가 오면(경로 등 PDF 글자일 수 있다) 둔다
_NL_ALL = re.compile(r"\\n")                   # 모델 글 블록: 수식 밖의 \n 은 모두 모델 잡문자
_SIMPLE = re.compile(r"[A-Za-z0-9+\-=*/.,()]{1,12}")   # $N$·$10$·$a+b$ 같은 짧은 표기
_SUP = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")
_SUB = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")
_CMDS = {"cdot": "·", "times": "×", "rightarrow": "→", "to": "→", "leftarrow": "←", "Rightarrow": "⇒",
         "leftrightarrow": "↔", "sim": "~", "pm": "±", "geq": "≥", "ge": "≥", "leq": "≤", "le": "≤",
         "approx": "≈", "neq": "≠", "circ": "°", "%": "%", "bullet": "•", "cdots": "⋯", "ldots": "…"}
_CMD = re.compile(r"\\(" + "|".join(sorted(map(re.escape, _CMDS), key=len, reverse=True)) + r")(?![A-Za-z])")
_WRAP = re.compile(r"\\(?:mathrm|mathbf|text|textbf|mathit)\{([^{}]*)\}")
_SCRIPT = re.compile(r"([\^_])\{?([0-9+\-=()n]+)\}?")


def _math(m: re.Match) -> str:
    """짧은 인라인 수식만 글자로 — 다 바꿀 수 없으면 원래대로 둔다. \·^·_ 가 없으면 금액($1,200) 같은 글이다."""
    if not re.search(r"[\\^_]", m.group(1)):
        return m.group(1) if _SIMPLE.fullmatch(m.group(1)) else m.group(0)
    s = _WRAP.sub(r"\1", m.group(1))
    s = _CMD.sub(lambda c: _CMDS[c.group(1)], s)
    s = _SCRIPT.sub(lambda c: c.group(2).translate(_SUP if c.group(1) == "^" else _SUB), s)
    s = re.sub(r"\^\{?([°′″])\}?", r"\1", s).replace("{", "").replace("}", "").strip()
    return m.group(0) if re.search(r"[\\^_$]", s) else s


def _drop_nl(s: str, pat: re.Pattern) -> str:
    """수식($…$) 밖의 글자 그대로의 \n 만 지운다 — $\nu$ 같은 명령은 그대로."""
    return "".join(p if len(p) > 1 and p[0] == p[-1] == "$" else pat.sub(" ", p) for p in re.split(r"(\$[^$\n]{1,30}\$)", s))


def norm_text(s: str, vlm: bool = True, strict: bool = False) -> str:
    """vlm 이면 모델 잡문자(글자 그대로의 \n·LaTeX)도 치운다 — PDF 글자에서 온 글은 건드리지 않는다.
    strict 는 모델이 읽은 글 블록(수식 밖 \n 은 모두 잡문자), 아니면 PDF 글자가 섞일 수 있는 표 칸."""
    s = unicodedata.normalize("NFC", s).translate(_DOTS)
    if vlm:
        s = _drop_nl(_MATH.sub(_math, s), _NL_ALL if strict else _NL)   # 수식 먼저 — \neq 가 \n 치환에 깨지지 않게
    s = _UNIT.sub(r"\1\2", s)
    lines = []
    for ln in s.split("\n"):
        m = _SPACED.match(ln)
        lines.append(m.group(1) + re.sub(r"\s+", "", m.group(2)) + m.group(3) if m else ln)
    return "\n".join(" ".join(ln.split()) if "  " in ln else ln for ln in lines)


def _norm_html(h: str) -> str:
    return re.sub(r">([^<]+)<", lambda m: ">" + norm_text(m.group(1)) + "<", h)


def normalize(doc: Document) -> Document:
    doc = doc.model_copy(deep=True)
    for page in doc.pages:
        for b in page.blocks:
            if b.type in ("display_formula", "inline_formula", "formula_number"):
                continue
            before = (b.text, b.html, b.figure_text)
            vlm = not any(p.get("rule") in ("text_layer", "vlm_skipped") for p in b.provenance or [])
            if b.text:
                b.text = norm_text(b.text, vlm, strict=True)
            if b.html:
                b.html = _norm_html(b.html)
            if b.figure_text:
                b.figure_text = norm_text(b.figure_text, strict=True)
            if (b.text, b.html, b.figure_text) != before:
                b.provenance = (b.provenance or []) + [{"rule": "normalize"}]
    return doc
