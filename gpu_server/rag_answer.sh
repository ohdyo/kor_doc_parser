#!/bin/bash
# 서버: RAG 답 비교(`테스트_결과.md` 14.11, D24) — rag_eval5.sh 를 돌린 서버에서 그 kdp 최종 출력($W/kdp5 — 그래프 다시 읽기 포함)·비교군·
# 2회차 저장소($W/lightrag_rpipe — LLM 캐시 원본)를 쓴다. LLM 서버(serve_llm.sh — A.X-4.0-Light)는 이 스크립트가 띄우고 내린다.
#   1) 평가 코드 시험(tests/unit/test_rag_eval.py)  2) 최종 저장소 — kdp 최종 출력을 구조 자르기 512토큰으로 넣고(2회차 LLM 캐시를 복사해
#      같은 조각은 같은 추출) naive·mix 검색 평가(다시 순위 기본)  3) 같은 저장소에 질문 72개를 세 가지로 묻고 답을 받는다 —
#      LightRAG 기본(mix)·Naive RAG(naive)·RAG 없음(bypass), 같은 LLM·같은 문맥 한도(eval/answer_eval.py — 검색 평가 질문 72개.
#      답 비교용 질문 50개는 rag_answer2.sh)
#   tmux new -d -s ans "bash /workspace/kor_doc_parser/gpu_server/rag_answer.sh"   진행: /workspace/rag_eval/status_ans.txt
set -uo pipefail
export LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1
R=/workspace/kor_doc_parser P=/workspace/envs/kdp-paddle/bin Q=/workspace/envs/kdp-rag/bin W=/workspace/rag_eval
cd $R
mark() { echo "$(date -u '+%F %T') $*" | tee -a $W/status_ans.txt; }
up() { for _ in $(seq 180); do curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null && return 0; sleep 5; done; return 1; }
free_gpu() {   # 서버를 내린 뒤 GPU 가 빌 때까지(vLLM 엔진 자식 프로세스가 늦게 내려간다)
  for _ in $(seq 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ] && return 0; sleep 2; done
  pkill -f VLLM::EngineCore; sleep 5
}

mark 시작
$P/python -m pytest -q tests/unit/test_rag_eval.py > $W/pytest_ans.log 2>&1; mark "평가 코드 시험 $? — $(tail -1 $W/pytest_ans.log)"

bash gpu_server/serve_llm.sh > $W/llm_ans.log 2>&1 &
L=$!
if up 8120; then
  $Q/python eval/lightrag_eval.py --kdp $W/kdp5 --compare $W/compare --work $W/lightrag_final --out $W/lightrag_report_final \
    --llm-cache-from $W/lightrag_rpipe > $W/lightrag_final.log 2>&1
  mark "최종 저장소 $?"
  $Q/python eval/answer_eval.py --work $W/lightrag_final --questions eval/golden/rag_questions.yaml --out $W/answers \
    > $W/answers.log 2>&1
  mark "답 받기 $? — $(grep -c ' success ' $W/answers.log) 성공"
else
  mark "LLM 서버 실패"
fi
kill $L; wait $L 2>/dev/null; free_gpu
mark 끝
