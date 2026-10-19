"""그림 설명 VLM(D19)을 transformers 로 띄우는 OpenAI 호환 서버 — vLLM 0.31 에 없는 모델(SK텔레콤 A.X-4.0-VL-Light)용.
파서(src/kdp/figure_desc.py)가 보내는 꼴만 받는다: 사용자 메시지 하나(그림 한 장 data URL + 글), temperature 0(탐욕 생성),
max_tokens, chat_template_kwargs, logprobs·top_logprobs(설명 문장 검증이 '맞음'·'틀림' 확률을 본다). 요청은 하나씩 처리한다(문서
하나에 그림 수십 장 규모). 127.0.0.1 에만 연다.
  tmux new -d -s desc "/workspace/envs/kdp-serve/bin/python /workspace/kor_doc_parser/gpu_server/serve_hf_vlm.py > /workspace/desc.log 2>&1"
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import transformers
from PIL import Image


def load(model_id: str, device: str):
    import transformers.processing_utils as pu

    if not hasattr(pu, "_validate_images_text_input_order"):      # A.X 저장소 코드가 부르는 4.x 내부 함수(5.x 에서 빠짐) — 순서를 그대로 둔다
        pu._validate_images_text_input_order = lambda images, text: (images, text)
    cfg = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    for name in ("text_config", "llm_config", "vision_config"):   # 설정은 flash_attention_2 — 서버에 flash-attn 이 없다
        sub = getattr(cfg, name, None)
        if sub is not None and hasattr(sub, "_attn_implementation"):
            sub._attn_implementation = "sdpa"
    auto = (getattr(cfg, "auto_map", None) or {})
    cls = transformers.AutoModelForCausalLM if "AutoModelForCausalLM" in auto else transformers.AutoModelForImageTextToText
    proc = transformers.AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = cls.from_pretrained(model_id, config=cfg, trust_remote_code=True, dtype=torch.bfloat16).to(device).eval()
    if cfg.model_type == "a.x-4-vl":
        _patch_first_step(type(model))
    return proc, model


def _patch_first_step(cls) -> None:
    """A.X 저장소 코드는 그림을 넣을 첫 단계를 cache_position[0] == 0 으로 가린다(4.49 기준) — 5.x 는 cache_position 을 넘기지
    않아 None 에서 멈춘다. 이번 단계 입력에 그림 자리 토큰이 있을 때만 그림을 넣는 함수로 바꾼다(판 번호에 기대지 않는다)."""
    def prepare(self, input_ids, past_key_values=None, inputs_embeds=None, pixel_values=None, image_sizes=None,
                attention_mask=None, cache_position=None, logits_to_keep=None, is_first_iteration=None, **kwargs):
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            cache_position=cache_position, logits_to_keep=logits_to_keep, is_first_iteration=is_first_iteration, **kwargs)
        ids = model_inputs.get("input_ids")
        if bool((ids == self.config.image_token_index).any()) if ids is not None else bool(is_first_iteration):
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_sizes"] = image_sizes
        return model_inputs

    cls.prepare_inputs_for_generation = prepare


def generate(proc, model, image: Image.Image, prompt: str, max_new_tokens: int, template_kwargs: dict,
             repetition_penalty: float, top_logprobs: int | None = None) -> tuple[str, str, int, int, list | None]:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kwargs)
    inputs = proc(images=[image], text=[text], padding=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,   # A.X 설정은 use_cache false — 켜지 않으면 글자마다 그림 토큰을 다시 계산한다
                             repetition_penalty=repetition_penalty, return_dict_in_generate=True,
                             output_logits=top_logprobs is not None)
    n_in = inputs["input_ids"].shape[1]
    new = out.sequences[0, n_in:]
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    finish = "length" if len(new) >= max_new_tokens and int(new[-1]) not in eos else "stop"
    lps = None
    if top_logprobs is not None:                          # 되풀이 벌점을 매기기 전 값 — 물음에 든 '맞음'·'틀림' 이 깎이지 않게
        tok = proc.tokenizer
        lps = []
        for i, logits in enumerate(out.logits):
            lp = torch.log_softmax(logits[0].float(), dim=-1)
            top = torch.topk(lp, max(1, top_logprobs))
            lps.append({"token": tok.decode([int(new[i])]), "logprob": float(lp[int(new[i])]),
                        "top_logprobs": [{"token": tok.decode([int(t)]), "logprob": float(v)}
                                         for v, t in zip(top.values, top.indices)]})
    return proc.batch_decode([new], skip_special_tokens=True)[0].strip(), finish, n_in, len(new), lps


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="skt/A.X-4.0-VL-Light")
    ap.add_argument("--port", type=int, default=8119)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repetition-penalty", type=float, default=1.05)   # 모델 카드 권장값 — 탐욕 생성의 되풀이를 줄인다
    a = ap.parse_args(argv)
    proc, model = load(a.model, a.device)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                self._send(200, {"object": "list", "data": [{"id": a.model, "object": "model"}]})
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                parts = body["messages"][-1]["content"]
                url = next(p["image_url"]["url"] for p in parts if p.get("type") == "image_url")
                prompt = "\n".join(p["text"] for p in parts if p.get("type") == "text")
                image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
                max_new = int(body.get("max_tokens") or body.get("max_completion_tokens") or 512)
                top_lp = int(body.get("top_logprobs") or 1) if body.get("logprobs") else None
            except (KeyError, StopIteration, ValueError, OSError) as e:
                self._send(400, {"error": {"message": f"bad request: {e}"}})
                return
            try:
                with lock:
                    t0 = time.perf_counter()
                    text, finish, n_in, n_out, lps = generate(proc, model, image, prompt, max_new,
                                                              body.get("chat_template_kwargs") or {}, a.repetition_penalty, top_lp)
                    took = round(time.perf_counter() - t0, 2)
            except Exception as e:                                # 그 그림만 실패 — 파서가 다음 실행에서 다시 부른다
                traceback.print_exc()
                self._send(500, {"error": {"message": f"{type(e).__name__}: {e}"}})
                return
            self._send(200, {"id": "hf", "object": "chat.completion", "created": int(time.time()), "model": a.model,
                             "choices": [{"index": 0, "finish_reason": finish,
                                          "message": {"role": "assistant", "content": text},
                                          "logprobs": {"content": lps} if lps is not None else None}],
                             "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out,
                                       "seconds": took}})

    print(f"serving {a.model} on 127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
