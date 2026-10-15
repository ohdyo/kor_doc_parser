"""스캔 시트 방향(기획서 4.3 규칙 12, 설계 3.7) — 텍스트 레이어가 없는 시트만 0·90·180·270 을 정해 np.rot90 으로 세운다.
파이프라인의 문서 전처리(use_doc_orientation_classify)는 끈 채로 둔다 — 문서 펴기가 함께 켜져 좌표를 되돌릴 수 없다.
born-digital 쪽의 /Rotate 는 렌더러가 이미 반영한다.
방향 분류 모델(PP-LCNet_x1_0_doc_ori)의 전처리(짧은 변 256 → 가운데 224 자르기)는 긴 변 양 끝 19% 씩을 버려, 시트를 그대로 물으면
글이 위쪽에 몰린 쪽·꽉 찬 쪽의 답이 갈린다. 그래서 잉크 상자를 여백 다섯 가지로 잘라 흰 정사각형에 놓고 네 방향으로 돌려 묻고(20표),
한 각도가 AGREE 이상이고 그 각도로 세운 쪽에서 잉크 투영의 글줄이 가로이고 글이 MIN_TEXT 이상일 때만 돌린다 — 틀리게 돌리느니 그대로.
합성 스캔 75꼴 × 네 방향(300쪽 — 244쪽 묶음에 슬라이드·목차·서식·사진·도식·간지 더함, 기울기·잡음·저해상 포함): 틀린 회전 0,
돌려야 할 쪽의 65%(표지·제목·간지·빈 쪽을 빼면 79%)를 세운다, CPU 에서 쪽당 0.2~0.3초. 확신이 없을 때는 볼 방향(표가 기운
방향 → 그대로)만 내고, 모아찍기 스캔은 그 차례로 칸 아래 쪽번호가 맞는 첫 방향으로 세워 나눈다(파이프라인). 검출기(PP-DocLayoutV3)를 네 방향에 돌려
견주는 길은 버렸다 — 머리말·쪽번호 이름을 자리 따라 붙여(뒤집힌 쪽 위 작은 줄을 쪽번호로) 위아래를 가르지 못하고, CPU 에서 방향마다 1.5초 걸린다."""
from __future__ import annotations

import os
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import Env, apply_env
from ..textlayer.extract import MIN_PAGE_CHARS, visible

MODEL = "PP-LCNet_x1_0_doc_ori"      # Apache-2.0, PaddleX 공식 모델(Hugging Face PaddlePaddle/PP-LCNet_x1_0_doc_ori)
MARGINS = (0.0, 0.03, 0.05, 0.1, 0.15)   # 잉크 상자 여백(긴 변 비율) — 틀마다 답이 조금씩 갈려 여럿으로 묻는다(여백 × 네 방향 = 20 표)
AGREE = 0.8                          # 한 각도가 표의 이 비율 이상일 때만 — 합성 시험(글 있는 쪽)에서 틀린 각도의 최대 득표율은 0.55
INK = 160                            # 이보다 어두우면 잉크(모아찍기 감지와 같다)
TRIM = 0.01                          # 시트 가장자리(스캐너 그림자)는 보지 않는다
NOISE = 0.003                        # 잉크 상자: 행·열 잉크가 그 길이의 이 비율 이하면 티(쪽번호 줄은 남는다)
DUST = 0.01                          # 글 양: 띠 폭의 이 비율 이하 잉크 행은 빈 행(먼지·티)
RULE = 0.04                          # 시트 긴 변의 이 비율보다 길게 이어진 가로·세로 잉크는 괘선·밑줄 — 글줄을 잴 때 지운다
STRIPS = 4                           # 글줄 방향은 띠 넷으로 나눠 잰다 — 기운 스캔(1°)에서도 줄 사이 빈 행이 남게
LINE_H = 0.004                       # 글 양: 시트 긴 변의 이 비율보다 낮은 잉크 띠는 글줄이 아니다(144dpi 에서 7px, 5pt 글자는 10px)
MIN_TEXT = 45                        # 글줄 길이/높이 합(대략 글자 수)이 이보다 적은 쪽(표지·간지·쪽번호만 있는 쪽)은 돌리지 않는다 —
                                     # 모델이 큰 글자 몇 줄짜리 제목 쪽을 거꾸로 확신한다(합성 제목 쪽 20표 중 19표, 글 양 24~34).
                                     # 슬라이드 55·목차 81·서식 82 는 세운다. 도식 슬라이드 26·사진 쪽 20~37 은 그대로
LEAN = 1.5                           # 모름이어도 표 1위가 180° 짝 방향 표의 이 배수 이상이면 '기운 방향' — 모아찍기 쪽번호 자리와 맞춰 볼 때만 쓴다
                                     # (10:7·11:9 같은 반반은 기울지 않았다고 본다)


def _real(chars: list[dict]) -> list[dict]:
    return [c for c in chars if not c["gen"] and not c["c"].isspace()]


def is_scan(chars: list[dict]) -> bool:
    """텍스트 레이어로 글을 읽을 수 없는 시트 — 보이는 글자가 born-digital 기준(textlayer 와 같다)에 못 미친다."""
    real = _real(chars)
    seen = [c for c in real if visible(c)]
    return len(seen) < MIN_PAGE_CHARS or len(seen) < 0.5 * len(real)


def layerless(chars: list[dict]) -> bool:
    """보이는 레이어 글자가 하나도 없는 시트 — 방향은 이런 시트만 본다. 글자가 몇 개라도 있으면 born-digital 표지·간지일 수 있다."""
    return not any(visible(c) for c in _real(chars))


def model_dir(env: Env) -> Path:
    """models.doc_ori_dir, 없으면 PaddleX 캐시(official_models). 실행 중에는 내려받지 않는다."""
    if env.models.doc_ori_dir:
        return Path(env.models.doc_ori_dir).expanduser()
    home = env.paddlex.cache_home or os.environ.get("PADDLE_PDX_CACHE_HOME") or Path.home() / ".paddlex"
    return Path(home).expanduser() / "official_models" / MODEL


@lru_cache(maxsize=2)
def _model(path: str, device: str):
    """프로세스에서 한 번만 불러온다(문서마다 불러오지 않는다)."""
    from paddlex import create_model

    return create_model(model_name=MODEL, model_dir=path, device=device)


def classify(img: np.ndarray, env: Env) -> tuple[int, float]:
    """(각도, 점수) — 반시계로 각도만큼 돌리면 선다(PaddleX 문서 전처리의 rotate_image 와 같은 약속). img 는 BGR."""
    path = model_dir(env)
    if not path.is_dir():
        raise FileNotFoundError(f"방향 분류 가중치가 없다: {path} — configs/env.yaml 의 models.doc_ori_dir 를 적거나 "
                                f"orientation: off 로 끈다")
    apply_env(env)                   # PaddleX 는 import 할 때 환경 변수를 읽는다
    res = next(iter(_model(str(path), env.device).predict(img)))
    return int(res["label_names"][0]), float(res["scores"][0])


def _ink(img: np.ndarray) -> np.ndarray:
    m = img.min(axis=2) < INK
    h, w = m.shape
    ty, tx = max(1, round(TRIM * h)), max(1, round(TRIM * w))
    m[:ty] = m[-ty:] = False
    m[:, :tx] = m[:, -tx:] = False
    return m


def _view(img: np.ndarray, ink: np.ndarray, margin: float) -> np.ndarray:
    """잉크 상자를 긴 변의 margin 만큼 넓혀 잘라 흰 정사각형 가운데에 놓는다 — 모델이 가운데만 잘라 보지 않게."""
    h, w = ink.shape
    ys = np.flatnonzero(ink.sum(axis=1) > NOISE * w)
    xs = np.flatnonzero(ink.sum(axis=0) > NOISE * h)
    if len(ys) and len(xs):
        p = round(margin * max(h, w))
        img = img[max(0, ys[0] - p):ys[-1] + 1 + p, max(0, xs[0] - p):xs[-1] + 1 + p]
    n = max(img.shape[:2])
    out = np.full((n, n, 3), 255, np.uint8)
    y, x = (n - img.shape[0]) // 2, (n - img.shape[1]) // 2
    out[y:y + img.shape[0], x:x + img.shape[1]] = img
    return out


def _long(m: np.ndarray, n: int) -> np.ndarray:
    """가로로 n 칸 이상 이어진 잉크 화소."""
    c = np.pad(np.cumsum(m, axis=1, dtype=np.int32), ((0, 0), (1, 0)))
    full = (c[:, n:] - c[:, :-n]) == n                       # [x, x+n) 가 모두 잉크
    f = np.pad(np.cumsum(full, axis=1, dtype=np.int32), ((0, 0), (1, 0)))
    x = np.arange(m.shape[1])
    return (f[:, np.minimum(x + 1, full.shape[1])] - f[:, np.clip(x - n + 1, 0, full.shape[1])]) > 0


def _gaps(ink: np.ndarray) -> float:
    """가로 글줄로 볼 때 빈 행 비율 — 세로 띠마다 그 띠의 내용 범위 안에서 재 잉크로 가중 평균한다.
    세로 글줄(옆으로 누운 쪽)이면 행마다 여러 줄을 지나 빈 행이 드물다."""
    w = ink.shape[1]
    blank = weight = 0.0
    for i in range(STRIPS):
        r = ink[:, i * w // STRIPS:(i + 1) * w // STRIPS].sum(axis=1)
        on = np.flatnonzero(r)
        if len(on) > 1:
            blank += float(r.sum()) * float((r[on[0]:on[-1] + 1] == 0).mean())
            weight += float(r.sum())
    return blank / weight if weight else 0.0


def _amount(text: np.ndarray) -> float:
    """가로 글줄의 길이/높이 합(대략 글자 수) — 세로 띠마다 잉크 행이 이어진 띠(글줄)를 센다. 길이는 잉크가 있는 열의 수
    (양 끝 거리로 재면 먼 티 하나가 줄을 늘린다), LINE_H 보다 낮은 띠(티·기운 괘선 조각)는 뺀다."""
    w = text.shape[1]
    low = LINE_H * max(text.shape)
    total = 0.0
    for i in range(STRIPS):
        s = text[:, i * w // STRIPS:(i + 1) * w // STRIPS]
        on = np.flatnonzero(s.sum(axis=1) > DUST * s.shape[1])
        if not len(on):
            continue
        for run in np.split(on, np.flatnonzero(np.diff(on) > 1) + 1):
            if len(run) >= low:
                total += np.count_nonzero(s[run[0]:run[-1] + 1].any(axis=0)) / len(run)
    return total


def judge(img: np.ndarray, env: Env) -> tuple[int | None, tuple[int, ...]]:
    """(k, 모를 때 볼 방향들). k = np.rot90(img, k) 로 세우는 반시계 90° 횟수, 확신이 없으면 None. np.rot90(그림, j) 의 답 a 는 원래
    그림에 a + 90j 를 가리킨다. 표가 한 각도로 모이고(AGREE), 그 각도로 세운 쪽에서 글줄이 가로이고(빈 행 비율이 빈 열 비율 이상)
    글이 MIN_TEXT 이상일 때만 답한다. 모를 때 볼 방향 = 표가 기운 방향(혼자 1위, 180° 짝의 LEAN 배 이상, 글줄 가로) 다음 그대로(0,
    글줄 가로일 때) — 모아찍기 스캔은 이 차례로 칸 아래 쪽번호가 맞는 첫 방향으로 세워 나눈다(파이프라인)."""
    ink = _ink(img)
    votes = Counter()
    for margin in MARGINS:
        view = _view(img, ink, margin)
        for j in range(4):
            angle, _ = classify(np.ascontiguousarray(np.rot90(view, j)), env)
            votes[(angle // 90 + j) % 4] += 1
    (k, n), *rest = votes.most_common()
    up = np.rot90(ink, k)                                     # 세운 쪽의 잉크
    flat = _gaps(up) >= _gaps(up.T)                           # 글줄 방향(잉크 투영)이 모델 답과 맞는다
    tries = [k] if flat and (not rest or rest[0][1] < n) and n >= LEAN * votes[(k + 2) % 4] else []
    if 0 not in tries and _gaps(ink) >= _gaps(ink.T):         # 그대로 둔 시트도 글줄이 가로면 본다 — 누운 시트는 보지 않는다
        tries.append(0)
    if n < AGREE * sum(votes.values()) or not flat:
        return None, tuple(tries)
    rule = max(8, round(RULE * max(ink.shape)))
    return (k if _amount(up & ~(_long(up, rule) | _long(up.T, rule).T)) >= MIN_TEXT else None), tuple(tries)   # 괘선·밑줄을 지운 글의 양


def upright_k(img: np.ndarray, env: Env) -> int | None:
    """세우는 k, 확신이 없으면 None(judge 의 첫째)."""
    return judge(img, env)[0]
