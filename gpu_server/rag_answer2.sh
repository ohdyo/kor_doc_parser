#!/bin/bash
# 서버: RAG 답 비교 — 답 비교용 질문 50개(eval/golden/rag_answer_questions.yaml, `테스트_결과.md` 14.11, D24). rag_answer.sh 가 만든 최종
# 저장소($W/lightrag_final)에 넣지 않고, LightRAG 기본(mix)·Naive RAG(naive)·RAG 없음(bypass) 세 가지로 묻는다. LLM 서버(serve_llm.sh)는
# 이 스크립트가 띄우고 내린다.
#   tmux new -d -s ans2 "bash /workspace/kor_doc_parser/gpu_server/rag_answer2.sh"   진행: /workspace/rag_eval/status_ans2.txt
set -uo pipefail
export LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1
R=/workspace/kor_doc_parser P=/workspace/envs/kdp-paddle/bin Q=/workspace/envs/kdp-rag/bin W=/workspace/rag_eval
cd $R
mark() { echo "$(date -u '+%F %T') $*" | tee -a $W/status_ans2.txt; }
up() { for _ in $(seq 180); do curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null && return 0; sleep 5; done; return 1; }
free_gpu() {   # 서버를 내린 뒤 GPU 가 빌 때까지(vLLM 엔진 자식 프로세스가 늦게 내려간다)
  for _ in $(seq 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ] && return 0; sleep 2; done
  pkill -f VLLM::EngineCore; sleep 5
}

mark 시작
$P/python -m pytest -q tests/unit/test_rag_eval.py > $W/pytest_ans2.log 2>&1; mark "평가 코드 시험 $? — $(tail -1 $W/pytest_ans2.log)"
bash gpu_server/serve_llm.sh > $W/llm_ans2.log 2>&1 &
L=$!
if up 8120; then
  $Q/python eval/answer_eval.py --work $W/lightrag_final --questions eval/golden/rag_answer_questions.yaml --out $W/answers50 \
    > $W/answers50.log 2>&1
  mark "답 받기 $? — $(grep -c ' success ' $W/answers50.log) 성공"
else
  mark "LLM 서버 실패"
fi
kill $L; wait $L 2>/dev/null; free_gpu
mark 끝
