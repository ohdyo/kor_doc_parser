#!/bin/bash
# PaddleOCR-VL-1.6 을 vLLM 으로 띄운다 — 서버에서 tmux 안에서 돌린다(README.md).
#   tmux new -d -s vllm "bash /workspace/kor_doc_parser/gpu_server/serve_vlm.sh > /workspace/vllm.log 2>&1"
# 처음에는 허깅페이스에서 가중치(약 1.8GB)를 받는다. 준비되면 curl -s http://127.0.0.1:8118/v1/models 에 모델 이름이 나온다.
export LC_ALL=C.UTF-8
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1        # 사용 통계 전송을 끈다(설계 9장)
export VLLM_USE_FLASHINFER_SAMPLER=0               # flashinfer 표본 뽑기 커널은 첫 사용 때 컴파일(ninja·nvcc)하다 실패했다 — temperature 0 이라 필요 없다
# 그림 설명 서버(D19)를 함께 띄울 때는 UTIL=0.25 — A40 에서 파싱 속도가 같았다(테스트_결과 9.5). 변수는 tmux 따옴표 안에 쓴다
exec /workspace/envs/kdp-serve/bin/vllm serve PaddlePaddle/PaddleOCR-VL-1.6 \
  --served-model-name PaddlePaddle/PaddleOCR-VL-1.6 \
  --trust-remote-code \
  --trust-request-mm-kwargs \
  --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --gpu-memory-utilization "${UTIL:-0.5}" \
  --max-model-len 8192 \
  --host 127.0.0.1 --port 8118
