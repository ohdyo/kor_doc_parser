#!/bin/bash
# LightRAG 의 LLM(D21 — 그래프 추출·mix 질문 키워드)을 vLLM 으로 띄운다 — 127.0.0.1:8120. rag_eval.sh 가 띄우고 내린다.
# 기본은 국내 모델 skt/A.X-4.0-Light(SK텔레콤, 7.3B, Apache-2.0, 문맥 16k). 더 큰 모델은 MODEL=Qwen/Qwen3-14B UTIL=0.8(Apache-2.0 —
# 생각 모드는 eval/lightrag_eval.py 가 요청마다 끈다). 처음에는 가중치(A.X 약 15GB)를 받는다. PaddleOCR-VL 서버는 먼저 내린다.
#   tmux new -d -s llm "bash /workspace/kor_doc_parser/gpu_server/serve_llm.sh > /workspace/llm.log 2>&1"
export LC_ALL=C.UTF-8
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1        # 사용 통계 전송을 끈다(설계 9장)
export VLLM_USE_FLASHINFER_SAMPLER=0               # serve_vlm.sh 와 같은 까닭(README '겪은 문제')
MODEL=${MODEL:-skt/A.X-4.0-Light}
exec /workspace/envs/kdp-serve/bin/vllm serve "$MODEL" \
  --served-model-name "$MODEL" \
  --gpu-memory-utilization "${UTIL:-0.6}" \
  --max-model-len "${LEN:-16384}" \
  --host 127.0.0.1 --port "${PORT:-8120}"
