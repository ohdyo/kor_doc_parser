"""그림 해석 튜닝(D20, 기획서 7.10) — A.X-4.0-VL-Light 에 LoRA 지도 학습. 물음·그림 토큰은 가리고(-100) 답 토큰과 닫는 표시만 배운다.
ms-swift 4.5.3 에 A.X-4.0-VL 이 없어(2026-10-19 확인) transformers + peft 로 짧게 쓴다(기획서 7.1 의 예외). 모델은 서빙과 같은
보정으로 띄운다(gpu_server/serve_hf_vlm.py 의 load) — 학습과 서빙의 입력이 같다.
GPU 서버 kdp-serve env(transformers 5.16·peft). 설명·PaddleOCR-VL 서버를 내리고 돌린다(A40 46GB 한 장).
사용: python train/figdesc/train_lora.py configs/tune/figdesc_v1.yaml [--max-steps N]
출력: <out>/adapter(LoRA — serve_hf_vlm.py --adapter), <out>/log.jsonl(손실), <out>/val.json(합성 검증 정답률)."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gpu_server"))
sys.path.insert(0, str(ROOT / "src"))
from kdp.figure_qa import QUESTIONS, nkey, same_short  # noqa: E402
from serve_hf_vlm import generate, load  # noqa: E402

END = "<|im_end|>"            # A.X 대화 틀이 답을 닫는 표시 — 이것까지 배워야 생성이 멈춘다
REP_PENALTY = 1.05            # 서버(serve_hf_vlm.py --repetition-penalty) 기본값과 같게 — 평가 물음은 그 서버로 묻는다
TARGETS = r".*language_model\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"   # 언어 모델(Qwen2 28층)만


def read(path: str) -> list[dict]:
    """학습 자료 JSONL(image·prompt·answer) — image 는 JSONL 폴더 기준 상대 경로."""
    base = Path(path).parent
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        rows.append({"image": str(base / r["image"]), "prompt": r["prompt"], "answer": r["answer"], "qtype": r.get("qtype")})
    return rows


def mix(sources: list[dict], rng: random.Random) -> list[dict]:
    """자료를 weight 비율로 섞는다 — 1 은 그대로, 2 는 두 번, 0.5 는 절반을 무작위로."""
    out = []
    for s in sources:
        rows, w = read(s["path"]), float(s.get("weight", 1.0))
        out += rows * int(w)
        if w - int(w):
            out += rng.sample(rows, int(len(rows) * (w - int(w))))
    rng.shuffle(out)
    return out


def encode(proc, row: dict, device: str):
    """(모델 입력, 정답) — 물음 쪽(대화 틀 + 그림 토큰 + 물음)은 -100 으로 가리고 답 + 닫는 표시만 남긴다."""
    image = Image.open(row["image"]).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": row["prompt"]}]}]
    prefix = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = proc(images=[image], text=[prefix + row["answer"] + END], return_tensors="pt")
    head = proc(images=[image], text=[prefix], return_tensors="pt")["input_ids"]
    n = head.shape[1]
    if not torch.equal(full["input_ids"][0, :n], head[0]):           # 답 앞의 토큰이 물음만 넣었을 때와 같아야 가린 자리가 맞다
        raise ValueError("물음 쪽 토큰이 답을 붙이면 달라진다 — 대화 틀·토크나이저 확인")
    labels = full["input_ids"].clone()
    labels[:, :n] = -100
    inputs = {k: full[k].to(device) for k in ("input_ids", "attention_mask")}
    inputs["pixel_values"] = full["pixel_values"].to(device, dtype=torch.bfloat16)
    return inputs, labels.to(device)


def correct(row: dict, out: str) -> bool:
    """합성 검증 채점 — 보기 물음은 첫 숫자(보기 번호), 막대 위 가장 큰 값 읽기는 수 값, 나머지 단답은 figure_qa 의 맞춤."""
    if QUESTIONS.get(row["qtype"], (None, None))[1] is not None:
        return next((ch for ch in out if ch.isdigit()), None) == row["answer"]
    first = out.splitlines()[0] if out else ""
    if row["qtype"] == "chart_maxval":
        m = re.search(r"-?\d[\d,]*(?:\.\d+)?", first)
        return bool(m) and float(m.group().replace(",", "")) == float(row["answer"].replace(",", ""))
    return same_short(first, {nkey(row["answer"])})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", type=Path)
    ap.add_argument("--max-steps", type=int, default=None, help="이만큼 최적화하고 멈춘다(첫 시험 — 시간·메모리 재기)")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    tr = cfg["train"]
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(tr.get("seed", 0))
    torch.manual_seed(tr.get("seed", 0))

    from peft import LoraConfig, get_peft_model

    proc, model = load(cfg["model"], "cuda")
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["rank"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"], bias="none",
                                             target_modules=TARGETS))
    # 기울기 체크포인트는 LoRA 를 씌운 뒤에 언어 모델에만 — 먼저 켜면 peft 가 입력 grad 갈고리를 그림 인코더에도 걸어 얼린 27층의
    # 활성값을 다 들고 역전파한다. use_reentrant=False 라 갈고리 없이도 LoRA 기울기는 흐른다
    lm = model.get_base_model().language_model
    lm.config.use_cache = False
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for p in model.parameters():                                         # LoRA 무게는 fp32 로 — 가중치(bf16)는 얼린다
        if p.requires_grad:
            p.data = p.data.float()
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"학습 무게 {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)

    rows = mix(cfg["data"], rng)
    accum = tr.get("accum", 8)
    total = int(len(rows) * tr.get("epochs", 1)) // accum                # 누적을 다 채우는 만큼만(끝에 남는 몇 개는 쓰지 않는다)
    if a.max_steps or tr.get("max_steps"):
        total = min(total, a.max_steps or tr["max_steps"])
    order = [r for _ in range(math.ceil(tr.get("epochs", 1))) for r in rows][: total * accum]
    warm = max(1, int(total * tr.get("warmup", 0.03)))
    opt = torch.optim.AdamW(params, lr=tr["lr"], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    print(f"자료 {len(rows)}개 · 최적화 {total}번(누적 {accum}) · warmup {warm}", flush=True)

    model.train()
    log = open(out / "log.jsonl", "w", encoding="utf-8")                # 실행마다 새로(시간 재기 시험과 본 학습이 섞이지 않게)
    step, t0, run_loss = 0, time.time(), []
    for i, row in enumerate(order):
        inputs, labels = encode(proc, row, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):                 # LoRA 무게는 fp32, 셈은 bf16
            loss = model(**inputs, labels=labels, use_cache=False).loss
        (loss / accum).backward()
        run_loss.append(float(loss))
        if (i + 1) % accum:
            continue
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % tr.get("log_every", 10) == 0 or step == total:
            rec = {"step": step, "loss": round(sum(run_loss) / len(run_loss), 4), "lr": sched.get_last_lr()[0],
                   "sec": round(time.time() - t0, 1), "mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
            run_loss = []
        if step % tr.get("save_every", 200) == 0:
            model.save_pretrained(out / "adapter")
        if step >= total:
            break
    model.save_pretrained(out / "adapter")

    val = read(cfg["val"])
    rng.shuffle(val)
    model.eval()
    res = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in val[: tr.get("val_n", 200)]:
            text = generate(proc, model, Image.open(row["image"]).convert("RGB"), row["prompt"], 24, {}, REP_PENALTY)[0]
            res.append({"qtype": row["qtype"], "ok": correct(row, text)})
    by = {}
    for r in res:
        by.setdefault(r["qtype"], []).append(r["ok"])
    summary = {"n": len(res), "acc": sum(r["ok"] for r in res) / max(1, len(res)),
               "by_qtype": {k: [sum(v), len(v)] for k, v in sorted(by.items())}, "steps": step, "sec": round(time.time() - t0, 1)}
    (out / "val.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("합성 검증", summary, flush=True)


if __name__ == "__main__":
    main()
