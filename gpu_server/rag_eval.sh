#!/bin/bash
# 서버: RAG 하류 평가 축소판(D21·D22 — 기획서 8.6)을 한 번에 — setup_envs.sh·setup_eval.sh 뒤에 tmux 로 돌린다. 모델 서버는 이
# 스크립트가 띄우고 내린다(먼저 떠 있으면 안 된다).
#   1) 단위 시험  2) 비교군 — 텍스트 레이어(모델 없음)·PaddleOCR-VL 기본(기본 마크다운 포함, 서버를 띄웠다 내린다)·docling(CPU)
#   3) kdp — runs/t1/out_base(test2~5 원래 A.X 파싱) 사본의 raw/ 로 새 코드가 다시 조립: document.rag.md 를 더 쓰고, 정본·document.md·
#      blocks.jsonl 은 원래 출력과 같아야 한다(status.txt 의 '같음')  4) 같은 자르기·같은 검색기 평가(eval/rag_eval.py — 계통 다섯 ×
#      LightRAG R 512·F 512·F 1,200토큰·구조 자르기 × BM25·BGE-M3)  5) LightRAG 저장소 평가(eval/lightrag_eval.py 기본 — kdp
#      document.rag.md 하나를 512토큰 R 로 × naive·mix. 1회차 넷은 --runs kdp_rag:512:F,kdp:512:F,paddle_default:512:F,kdp_rag:1200:F
#      와 표 HTML 설정이었다. LLM 서버 serve_llm.sh 를 띄웠다 내린다)
# 결과는 /workspace/rag_eval(status.txt 에 단계마다 시각·종료 코드). 끝나면 이 컴퓨터에서 pull.sh.
#   tmux new -d -s rag "bash /workspace/kor_doc_parser/gpu_server/rag_eval.sh"
set -uo pipefail
export LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1
R=/workspace/kor_doc_parser P=/workspace/envs/kdp-paddle/bin D=/workspace/envs/kdp-docling/bin Q=/workspace/envs/kdp-rag/bin
W=/workspace/rag_eval
DOCS=(test2 test3 test4 test5)
mkdir -p $W && cd $R
mark() { echo "$(date -u '+%F %T') $*" | tee -a $W/status.txt; }
up() { for _ in $(seq 180); do curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null && return 0; sleep 5; done; return 1; }
free_gpu() {   # 서버를 내린 뒤 GPU 가 빌 때까지(vLLM 엔진 자식 프로세스가 늦게 내려간다)
  for _ in $(seq 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ] && return 0; sleep 2; done
  pkill -f VLLM::EngineCore; sleep 5
}
compare() {    # $1 문서, $2 python, $3 비교군 — eval/compare.py -o 자리가 rag_eval.py --compare 가 읽는 자리다
  $2 eval/compare.py $1.pdf --golden eval/golden/${1}_pdf.yaml --runs $3 -o $W/compare > $W/compare_${1}_$3.log 2>&1
  mark "비교군 $1 $3 $?"
}

mark 시작
$P/python -m pytest -q tests/unit > $W/pytest.log 2>&1; mark "단위 시험 $? — $(tail -1 $W/pytest.log)"

for t in "${DOCS[@]}"; do compare $t $P/python textlayer; done
UTIL=0.5 bash gpu_server/serve_vlm.sh > $W/vllm.log 2>&1 &
V=$!
if up 8118; then
  for t in "${DOCS[@]}"; do compare $t $P/python paddle_default; done
else
  mark "PaddleOCR-VL 서버 실패"
fi
kill $V; wait $V 2>/dev/null; free_gpu
for t in "${DOCS[@]}"; do compare $t $D/python docling; done

rm -rf $W/kdp && cp -r runs/t1/out_base $W/kdp
rm -f $W/kdp/*/document.json $W/kdp/*/document.md $W/kdp/*/document.rag.md $W/kdp/*/blocks.jsonl   # raw/ 만 두고 새로 쓴다
$P/kdp reassemble $W/kdp > $W/reassemble.log 2>&1; rc=$?; mark "다시 조립 $rc"
for d in runs/t1/out_base/*/; do
  id=$(basename "$d")
  for f in document.json document.md blocks.jsonl; do
    if cmp -s "$d/$f" "$W/kdp/$id/$f"; then mark "같음 ${id:0:8} $f"; else mark "다름 ${id:0:8} $f"; fi
  done
done

$Q/python eval/rag_eval.py --kdp $W/kdp --compare $W/compare --out $W/report > $W/rag_eval.log 2>&1; mark "검색 평가 $?"

bash gpu_server/serve_llm.sh > $W/llm.log 2>&1 &
L=$!
if up 8120; then
  $Q/python eval/lightrag_eval.py --kdp $W/kdp --compare $W/compare --work $W/lightrag --out $W/lightrag_report \
    > $W/lightrag_eval.log 2>&1; mark "LightRAG 평가 $?"
else
  mark "LLM 서버 실패"
fi
kill $L; wait $L 2>/dev/null; free_gpu
mark 끝
