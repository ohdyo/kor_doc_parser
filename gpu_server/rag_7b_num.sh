#!/bin/bash
# 서버: 7B 수 옮기기 지시 시험(D25 덧붙임) — rag_7b31b.sh 를 돌린 서버에서. 7B 가 수를 옮길 때 쉼표와 숫자 하나를 끼워 넣는 버릇
# (396 → 3,596)이 답 지시 한 줄(eval/prompts/answer_numbers.txt)로 없어지는지 본다. 같은 저장소(lightrag_final)·같은 질문에
#   7b_num    M naive + 수 지시        7b_or_num  O naive 근거만 + 수 지시
#   tmux new -d -s num "bash /workspace/kor_doc_parser/gpu_server/rag_7b_num.sh"   진행: /workspace/rag_eval/status_cmp.txt
set -uo pipefail
export LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1
R=/workspace/kor_doc_parser Q=/workspace/envs/kdp-rag/bin W=/workspace/rag_eval C=/workspace/rag_eval/cmp M7=skt/A.X-4.0-Light
QM=eval/golden/rag_answer_questions.yaml,eval/golden/rag_answer_questions2.yaml QR=eval/golden/rag_questions.yaml NUM=eval/prompts/answer_numbers.txt
cd $R
mark() { echo "$(date -u '+%F %T') $*" | tee -a $W/status_cmp.txt; }
up() { for _ in $(seq 360); do curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null && return 0; kill -0 "$2" 2>/dev/null || return 1; sleep 5; done; return 1; }
ask() {
  local cell=$1
  shift
  $Q/python eval/answer_eval.py --work $W/lightrag_final --llm-model $M7 --out "$C/$cell" "$@" > "$C/$cell.log" 2>&1
  mark "답 $cell $? — 성공 $(grep -c ' success ' "$C/$cell.log"), 실패 $(grep -cE ' (error|failure) ' "$C/$cell.log")"
}
mark "수 지시 시험 시작"
bash gpu_server/serve_llm.sh > $W/llm_7b_num.log 2>&1 &
L=$!
if up 8120 $L; then
  ask 7b_num --questions $QM --modes naive --variant num --user-prompt-file $NUM
  ask 7b_or_num --questions $QM,$QR --oracle --variant oracle-num --user-prompt-file $NUM
else
  mark "7B 서버 실패(수 지시)"
fi
kill $L; wait $L 2>/dev/null
mark "수 지시 시험 끝"
