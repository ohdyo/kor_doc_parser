#!/bin/bash
# 서버: 31B LLM 을 FP8 로 바꾼다(D25 — 7B 대 31B) — A40 에 bf16(62.6GB)이 들어가지 않는다. 모델은 websfactory/Webs-Sejong-31B-v7
# (Gemma 4 31B 바탕 한국어 튜닝, Gemma 이용약관 — 사용료·규모 조건 없음, 금지 용도 정책을 따른다. 바꾼 사본은 이 서버 안에서만 쓰고
# 어디에도 올리지 않는다 — 나눠 주면 약관 사본·고지가 따라가야 한다).
# LLM Compressor(Apache-2.0)의 model_free_ptq: 모델을 메모리에 올리지 않고 safetensors 조각마다 선형층 가중치를 FP8(E4M3, 출력 채널별
# 비율)로 바꾼다 — 보정 자료가 필요 없다(FP8_DYNAMIC). 그림 인코더·임베딩·lm_head·정규화 층은 bf16 그대로(Gemma 4 는 이름 끝에 숫자가
# 붙은 정규화 층이 있어 norm 을 모두 뺀다). A40(Ampere)은 FP8 계산 장치가 없어 vLLM 이 가중치만 FP8 인 Marlin 커널(W8A16)로 돌린다.
# CPU 로 바꿔 GPU 를 쓰는 다른 단계와 함께 돌 수 있다. 디스크: bf16 62.6GB + FP8 약 33GB — 끝나면 bf16 을 지운다(KEEP_RAW=1 이면 둔다 —
# rag_7b31b.sh 는 FP8 사본이 뜨지 않을 때 bf16 을 vLLM 온라인 FP8(fp8_per_tensor)로 띄우려고 두었다가 마지막에 지운다).
#   bash /workspace/kor_doc_parser/gpu_server/quantize_fp8.sh      (rag_7b31b.sh 가 배경으로 부른다)
set -euo pipefail
export LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1
SRC=${SRC:-websfactory/Webs-Sejong-31B-v7}
RAW=/workspace/models/$(basename "$SRC")             # 받은 bf16 — 바꾼 뒤 지운다
DST=${DST:-$RAW-FP8}
ENV=/workspace/envs/kdp-quant
if [ -f "$DST/config.json" ] && grep -q '"quantization_config"' "$DST/config.json"; then echo "이미 있음: $DST"; exit 0; fi
free=$(df --output=avail -BG /workspace | tail -1 | tr -dc 0-9)
if [ ! -f "$RAW/model.safetensors.index.json" ] && [ "$free" -lt 100 ]; then
  echo "디스크 부족: /workspace 남은 ${free}GB — bf16 62.6GB 와 FP8 약 33GB 가 함께 들어갈 100GB 이상이 필요하다(서버 볼륨 200GB)"; exit 1
fi
if [ ! -x "$ENV/bin/python" ]; then
  uv venv "$ENV" --python 3.12
  VIRTUAL_ENV="$ENV" uv pip install --torch-backend cpu "llmcompressor==0.14.0.1" "huggingface_hub>=0.34"
fi
mkdir -p /workspace/models
"$ENV/bin/hf" download "$SRC" --local-dir "$RAW" > /dev/null   # 공개 모델 — 토큰 없이
"$ENV/bin/python" - "$RAW" "$DST" <<'EOF'
import sys

from llmcompressor import model_free_ptq

src, dst = sys.argv[1:3]
model_free_ptq(model_stub=src, save_directory=dst, scheme="FP8_DYNAMIC",
               ignore=["lm_head", "re:.*embed.*", "re:.*vision.*", "re:.*norm.*"], device="cpu", max_workers=8)
EOF
for f in "$RAW"/*; do                                 # 가중치 밖의 파일(토크나이저·채팅 틀·처리기 설정)이 없으면 옮긴다
  case $f in *.safetensors|*model.safetensors.index.json) continue ;; esac
  [ -e "$DST/$(basename "$f")" ] || cp "$f" "$DST/"
done
grep -q '"quantization_config"' "$DST/config.json" || { echo "config.json 에 quantization_config 가 없다 — 양자화 실패"; exit 1; }
du -sh "$DST"
[ "${KEEP_RAW:-0}" = 1 ] || rm -rf "$RAW"
echo QUANT_DONE
