#!/bin/bash
# 서버: LightRAG 저장소 평가를 고른 실행만 — LLM 서버(serve_llm.sh)를 띄우고 eval/lightrag_eval.py 를 돌린 뒤 내린다(D21·D22).
# rag_eval.sh 를 한 번 돌린 서버에서 그 비교군($W/compare)·kdp 출력을 쓴다. 인자는 lightrag_eval.py 에 그대로 넘긴다.
#   tmux new -d -s lr "bash /workspace/kor_doc_parser/gpu_server/lightrag_one.sh <이름> --runs kdp_rag_pipe:512:R --variant kdp_rag_pipe=/workspace/rag_eval/kdp_pipe"
#   진행: /workspace/rag_eval/status_<이름>.txt, 결과: /workspace/rag_eval/lightrag_report_<이름>/ (저장소는 lightrag_<이름>/)
set -uo pipefail
export LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1
R=/workspace/kor_doc_parser Q=/workspace/envs/kdp-rag/bin W=/workspace/rag_eval
NAME=$1; shift
cd $R
mark() { echo "$(date -u '+%F %T') $*" | tee -a $W/status_$NAME.txt; }
up() { for _ in $(seq 180); do curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null && return 0; sleep 5; done; return 1; }
free_gpu() {
  for _ in $(seq 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ] && return 0; sleep 2; done
  pkill -f VLLM::EngineCore; sleep 5
}

mark 시작
bash gpu_server/serve_llm.sh > $W/llm_$NAME.log 2>&1 &
L=$!
if up 8120; then
  $Q/python eval/lightrag_eval.py --kdp ${KDP:-$W/kdp_html} --compare $W/compare --work $W/lightrag_$NAME --out $W/lightrag_report_$NAME "$@" \
    > $W/lightrag_eval_$NAME.log 2>&1; mark "LightRAG 평가 $?"
else
  mark "LLM 서버 실패"
fi
kill $L; wait $L 2>/dev/null; free_gpu
mark 끝
