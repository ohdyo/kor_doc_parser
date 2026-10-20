"""ohdyo/A.X-kor-image-caption — skt/A.X-4.0-VL-Light 에 이 LoRA 를 얹어 그림 물음 하나에 답한다(학습·평가와 같은 물음 틀·생성 설정).
확인한 판: torch 2.13 · transformers 5.16.1 · peft 0.20.0 (NVIDIA A40, bf16, GPU 메모리 약 16GB).
  python inference.py 그림.png chart_high --caption "연도별 온열질환자 수" --text "2021 / 2022 / 2023 / 2024"
  python inference.py 그림.png flow_next --x "서류 접수"
물음 갈래: map_high·map_low(9방위 보기), chart_high·chart_low(단답), chart_trend(5보기), flow_first·flow_last·flow_next,
org_top·org_parent(상자 글 단답). 비공개 저장소면 먼저 `hf auth login`(또는 HF_TOKEN)."""
from __future__ import annotations

import argparse

import torch
import transformers
from PIL import Image

BASE = "skt/A.X-4.0-VL-Light"
ADAPTER = "ohdyo/A.X-kor-image-caption"

DIRS = ["북서쪽", "북쪽", "북동쪽", "서쪽", "가운데", "동쪽", "남서쪽", "남쪽", "남동쪽"]   # 지도 테두리 상자를 3×3 으로 나눈 방위(위가 북)
TREND = ["늘어남", "줄어듦", "늘다가 줄어듦", "줄다가 늘어남", "뚜렷한 흐름 없음"]
HEAD = ("아래 그림을 보고 물음에 답하세요.\n그림 제목: {caption}\n그림에서 읽은 글자(틀린 글자가 있을 수 있습니다): {text}\n\n"
        "물음: {q}\n{tail}")
CHOICE_TAIL = "보기: {opts}\n보기 번호 하나만 답하세요."
SHORT_TAIL = "그림에 적힌 이름(가로축 글자·항목 이름) 그대로 짧게 답하세요. 막대나 선이 여럿이면 그림 제목이 가리키는 주된 값으로 봅니다."
DIAG_TAIL = "그림의 상자에 적힌 글 그대로 짧게 답하세요."
QUESTIONS = {
    "map_high": ("이 지도에서 값이 가장 높은(많은) 곳은 지도의 어느 쪽인가요?", DIRS),
    "map_low": ("이 지도에서 값이 가장 낮은(적은) 곳은 지도의 어느 쪽인가요?", DIRS),
    "chart_high": ("이 그래프에서 값이 가장 큰 때나 항목은 무엇인가요?", None),
    "chart_low": ("이 그래프에서 값이 가장 작은 때나 항목은 무엇인가요?", None),
    "chart_trend": ("이 그래프에서 값의 전체 흐름은 어떤가요?", TREND),
    "flow_first": ("이 흐름도에서 처음 단계는 무엇인가요?", None),
    "flow_last": ("이 흐름도에서 마지막 단계는 무엇인가요?", None),
    "flow_next": ("이 흐름도에서 '{x}' 바로 다음 단계는 무엇인가요?", None),
    "org_top": ("이 조직도(체계도)에서 맨 위에 있는 것은 무엇인가요?", None),
    "org_parent": ("이 조직도(체계도)에서 '{x}' 바로 위에 있는 것은 무엇인가요?", None),
}
DIAGRAM = ("flow_first", "flow_last", "flow_next", "org_top", "org_parent")
NEEDS_X = ("flow_next", "org_parent")


def prompt(qtype: str, caption: str = "", text: str = "", options: list[str] | None = None, x: str | None = None) -> str:
    """물음 글 — caption 은 그림 제목, text 는 그림에서 읽은 글자(OCR, 줄은 ' / ' 로), options 는 보기 순서(없으면 기본 순서)."""
    q, opts = QUESTIONS[qtype]
    if qtype in NEEDS_X and not x:
        raise ValueError(f"{qtype} 는 상자 글 x 가 있어야 한다")
    q = q.replace("{x}", x or "")
    if opts is None:
        tail = DIAG_TAIL if qtype in DIAGRAM else SHORT_TAIL
    else:
        options = options or opts
        if sorted(options) != sorted(opts):
            raise ValueError("보기는 정해진 보기를 섞은 것이어야 한다")
        tail = CHOICE_TAIL.format(opts=" ".join(f"{i + 1} {o}" for i, o in enumerate(options)))
    return HEAD.format(caption=caption or "(없음)", text=text or "(없음)", q=q, tail=tail)


def load(adapter: str = ADAPTER, device: str = "cuda"):
    """(프로세서, 모델) — A.X 저장소 코드(transformers 4.49 기준)를 5.x 에서 돌리는 보정 셋을 하고 LoRA 를 합치지 않고 얹는다
    (bf16 무게에 합치면 작은 변화가 반올림으로 사라진다)."""
    import transformers.processing_utils as pu
    from peft import PeftModel

    if not hasattr(pu, "_validate_images_text_input_order"):      # 4.x 내부 함수(5.x 에서 빠짐) — 순서를 그대로 둔다
        pu._validate_images_text_input_order = lambda images, text: (images, text)
    cfg = transformers.AutoConfig.from_pretrained(BASE, trust_remote_code=True)
    for name in ("text_config", "llm_config", "vision_config"):   # 설정은 flash_attention_2 — flash-attn 없이 sdpa 로
        sub = getattr(cfg, name, None)
        if sub is not None and hasattr(sub, "_attn_implementation"):
            sub._attn_implementation = "sdpa"
    auto = getattr(cfg, "auto_map", None) or {}
    cls = transformers.AutoModelForCausalLM if "AutoModelForCausalLM" in auto else transformers.AutoModelForImageTextToText
    proc = transformers.AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
    model = cls.from_pretrained(BASE, config=cfg, trust_remote_code=True, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter).to(device).eval()
    _patch_first_step(type(model.get_base_model()))
    return proc, model


def _patch_first_step(cls) -> None:
    """A.X 저장소 코드는 그림을 넣을 첫 단계를 cache_position[0] == 0 으로 가린다 — 5.x 는 cache_position 을 넘기지 않아 None 에서
    멈춘다. 이번 단계 입력에 그림 자리 토큰이 있을 때만 그림을 넣게 바꾼다."""
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


@torch.inference_mode()
def answer(proc, model, image: Image.Image, text: str, max_new_tokens: int = 24, repetition_penalty: float = 1.05) -> str:
    """탐욕 생성(학습·평가와 같다) — 보기 물음은 보기 번호, 단답은 그림의 글 그대로."""
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    chat = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(images=[image], text=[chat], return_tensors="pt").to(model.device, dtype=torch.bfloat16)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                         repetition_penalty=repetition_penalty)
    return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image")
    ap.add_argument("qtype", choices=sorted(QUESTIONS))
    ap.add_argument("--caption", default="", help="그림 제목")
    ap.add_argument("--text", default="", help="그림에서 읽은 글자(OCR) — 줄은 ' / ' 로 잇는다")
    ap.add_argument("--x", default=None, help="flow_next·org_parent 의 상자 글")
    ap.add_argument("--adapter", default=ADAPTER)
    a = ap.parse_args()
    proc, model = load(a.adapter)
    opts = QUESTIONS[a.qtype][1]
    out = answer(proc, model, Image.open(a.image).convert("RGB"), prompt(a.qtype, a.caption, a.text, x=a.x),
                 max_new_tokens=8 if opts else 24)
    num = next((ch for ch in out if ch.isdigit()), None)
    print(f"{out} → {opts[int(num) - 1]}" if opts and num and 0 < int(num) <= len(opts) else out)


if __name__ == "__main__":
    main()
