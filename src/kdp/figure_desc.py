"""그림 설명 VLM 부르기(D19) — 조립이 고른 그림(describe_targets)을 OpenAI 호환 서버(vLLM·gpu_server/serve_hf_vlm.py)에 보내고 응답을
토큰 확률과 함께 raw/describe.jsonl 에 쌓는다. 조립(attach_descriptions)은 이 기록만 읽으므로 재조립은 모델 없이 같은 결과를 낸다.
실패한 그림은 적지 않는다 — 다음 실행이 다시 부른다. 새로 파싱하면 raw/ 가 바뀌므로 기록도 함께 없어진다."""
from __future__ import annotations

import base64
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .assemble.figures import desc_key, desc_prompt, describe_targets
from .config import Settings
from .io.write import dumps
from .schema import Document

CACHE = "describe.jsonl"
TIMEOUT = 120                 # 그림 하나(초) — 서버가 묶음으로 처리해도 넉넉하게


def load(raw_dir: Path) -> dict[str, dict]:
    f = Path(raw_dir) / CACHE
    out = {}
    for line in f.read_text(encoding="utf-8").splitlines() if f.exists() else []:
        try:
            r = json.loads(line)
        except ValueError:
            continue                                           # 빈 줄·쓰다 끊긴 줄 — 그 그림은 다음 실행이 다시 부른다
        out[r["key"]] = r
    return out


def run(doc: Document, doc_dir: Path, settings: Settings) -> tuple[int, list[str]]:
    """설명 응답이 없는 그림만 부른다. (새로 받은 수, 실패한 블록과 사유)를 돌려준다."""
    cfg, env = settings.parse.describe, settings.env.describer
    raw = Path(doc_dir) / "raw"
    cache = load(raw)
    todo = []
    for page, b in describe_targets(doc, cfg):
        prompt = desc_prompt(doc, page, b)
        key = desc_key(cfg.model, b, prompt, cfg.max_tokens)
        if key not in cache:
            todo.append((key, b, prompt))
    if not todo:
        return 0, []
    if not env.server_url:
        raise RuntimeError("describe.enabled 인데 configs/env.yaml 의 describer.server_url 이 없다")
    import openai

    client = openai.OpenAI(base_url=env.server_url, api_key=env.api_key or "EMPTY", timeout=TIMEOUT, max_retries=1)

    def call(item):
        key, b, prompt = item
        img = (raw / b.img_path).read_bytes()                  # raw/imgs 에 모든 크롭이 있다(imgs/ 는 살린 것만)
        r = client.chat.completions.create(
            model=cfg.model, temperature=0, max_tokens=cfg.max_tokens, logprobs=True, top_logprobs=0,   # 문장별 확신(min_logprob)
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(img).decode()}},
                {"type": "text", "text": prompt}]}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})   # DNA3.0(Qwen3.5 바탕)은 생각 모드가 기본이다
        ch = r.choices[0]
        lps = [[t.token, round(t.logprob, 4)] for t in ch.logprobs.content] if ch.logprobs and ch.logprobs.content else None
        return {"key": key, "block_id": b.id, "model": cfg.model, "text": ch.message.content or "",
                "finish_reason": ch.finish_reason, "usage": r.usage.model_dump() if r.usage else None, "logprobs": lps}

    new, failed = 0, []
    with ThreadPoolExecutor(max(1, env.max_concurrency)) as ex, open(raw / CACHE, "a", encoding="utf-8") as f:
        futs = {ex.submit(call, it): it[1] for it in todo}
        for fut in as_completed(futs):                         # 받는 대로 적는다 — 중간에 멈춰도 받은 설명은 남는다
            try:
                f.write(dumps(fut.result()) + "\n")
                f.flush()
                new += 1
            except Exception as e:                             # 그 그림만 다음 실행에서 다시
                failed.append(f"{futs[fut].id}: {type(e).__name__}: {str(e)[:200]}")
    return new, sorted(failed)
