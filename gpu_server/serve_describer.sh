#!/bin/bash
# 그림 설명 VLM(D19)을 vLLM 으로 띄운다 — vLLM 이 아는 구조의 국내 모델(디노티시아 DNA3.0, Apache-2.0)용. 서버에서 tmux 안에서 돌린다(README.md).
# vLLM 에 없는 A.X-4.0-VL-Light(SK텔레콤)는 serve_hf_vlm.py 로 띄운다. 둘 다 127.0.0.1:8119 이라 한 번에 하나만 띄운다.
#   tmux new -d -s desc "MODEL=dnotitia/DNA3.0-4B bash /workspace/kor_doc_parser/gpu_server/serve_describer.sh > /workspace/desc.log 2>&1"
# (변수는 따옴표 안에 — 이미 떠 있는 tmux 서버는 밖의 변수를 새 세션에 넘기지 않는다.) 처음에는 가중치(약 10GB)를 받는다.
# 준비되면 curl -s http://127.0.0.1:8119/v1/models 에 모델 이름이 나온다. 파서 설정 describe.model 을 같은 이름으로 둔다.
export LC_ALL=C.UTF-8
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1        # 사용 통계 전송을 끈다(설계 9장)
export VLLM_USE_FLASHINFER_SAMPLER=0               # serve_vlm.sh 와 같은 까닭 — temperature 0 이라 필요 없다
MODEL=${MODEL:-dnotitia/DNA3.0-4B}
# DNA3.0-9B: UTIL=0.55(가중치 17.7GB + 계산 칸) — Qwen3.5 구조는 일부 층이 순환형이라 동시 요청마다 상태 칸이 하나씩 들어
# 기본 동시 수(256)는 몫 0.55 에 들지 않는다(실측 246) → MAX_SEQS 32
exec /workspace/envs/kdp-serve/bin/vllm serve "$MODEL" \
  --served-model-name "$MODEL" \
  --gpu-memory-utilization "${UTIL:-0.35}" \
  --max-num-seqs "${MAX_SEQS:-32}" \
  --max-model-len 8192 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --host 127.0.0.1 --port "${PORT:-8119}"
