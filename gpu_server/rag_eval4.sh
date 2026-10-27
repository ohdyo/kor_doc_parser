#!/bin/bash
# 서버: 개선 후보 셋 시험(`테스트_결과.md` 14.10) — 새로 켠 서버에서 push.sh(runs/rag_eval 기준 자료 함께)·setup_envs.sh·setup_eval.sh 뒤에
# tmux 로. 모델 서버(PaddleOCR-VL·그림 설명 A.X·LightRAG LLM)는 이 스크립트가 차례로 띄우고 내린다(먼저 떠 있으면 안 된다).
#   1) 단위 시험  2) ③ 조립 규칙 — runs/t1/out_base 사본을 다시 읽기 끈 설정으로 다시 조립($W/kdp4)해 2회차 파이프 표 판($W/kdp_pipe)과 블록마다
#      견준다(eval/block_diff.py — 바뀐 그림 글은 잘린 그림 넷이어야 한다)  3) ③ 그림 다시 읽기 — PaddleOCR-VL 서버로 kdp4 사본 둘에
#      kdp figure-ocr(OCR:·Spotting:)  4) 그림 설명 서버(A.X 기본 모델)로 셋 모두 kdp describe(그림 글이 바뀐 그림만 새로 부른다)
#   5) 검색 평가 — kdp4·2회차 파이프 표 판(③ 전 — 6)의 저장소와 같은 글)·다시 읽기 둘 × BM25·BGE-M3·BGE-M3→다시 순위(①) ×
#      LightRAG R·F 512·구조  6) ① 2회차 LightRAG 저장소 사본에
#      묻기만(LLM 없이 — mix 키워드는 캐시) — 다시 순위 없이(2회차 값과 같아야 한다)·다시 순위로  7) ② LLM 서버로 2회차 LLM 캐시를 복사해
#      저장소 둘 — 추출 max_tokens 2048(대조)·4096(잘렸던 추출과 그에 딸린 요약만 새로 부른다)
#   4회차 때 기본 설정(그림 다시 읽기 끔·다시 순위 끔)은 설정 사본·--rerank none 으로 고정했다. 한자·가나 줄 거르기는 4회차 뒤에 넣어
#   지금 코드로 다시 돌리면 OCR·Spot 판의 그림 글이 4회차와 조금 다르다(테스트_결과 14.10).
#   tmux new -d -s rag4 "bash /workspace/kor_doc_parser/gpu_server/rag_eval4.sh"   진행: /workspace/rag_eval/status4.txt
set -uo pipefail
export LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1
R=/workspace/kor_doc_parser P=/workspace/envs/kdp-paddle/bin Q=/workspace/envs/kdp-rag/bin S=/workspace/envs/kdp-serve/bin/python
W=/workspace/rag_eval
cd $R
mark() { echo "$(date -u '+%F %T') $*" | tee -a $W/status4.txt; }
up() { for _ in $(seq 180); do curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null && return 0; sleep 5; done; return 1; }
free_gpu() {   # 서버를 내린 뒤 GPU 가 빌 때까지(vLLM 엔진 자식 프로세스가 늦게 내려간다)
  for _ in $(seq 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ] && return 0; sleep 2; done
  pkill -f VLLM::EngineCore; sleep 5
}

mark 시작
$P/python -m pytest -q tests/unit > $W/pytest4.log 2>&1; mark "단위 시험 $? — $(tail -1 $W/pytest4.log)"

# 설정 사본 셋(설정 폴더에 — paddlex 설정 상대 경로): 그림 다시 읽기 끔(4회차 때 기본) · 켬 — 물음 OCR:(한도 1,024) · Spotting:
# (글 덩이마다 위치 토큰이 붙어 한도 3,072). 4회차는 그림·그래프 블록을 모두 다시 읽었다(types)
sed 's/figure_ocr: { enabled: true/figure_ocr: { enabled: false/' configs/parse/default.yaml > configs/parse/_off.yaml
sed 's/prompt: "Spotting:"/prompt: "OCR:"/; s/types: \[chart\]/types: [image, chart]/; s/max_tokens: 3072/max_tokens: 1024/' \
  configs/parse/default.yaml > configs/parse/_ocr.yaml
sed 's/types: \[chart\]/types: [image, chart]/' configs/parse/default.yaml > configs/parse/_spot.yaml

rm -rf $W/kdp4 && cp -r runs/t1/out_base $W/kdp4
rm -f $W/kdp4/*/document.json $W/kdp4/*/document.md $W/kdp4/*/document.rag.md $W/kdp4/*/blocks.jsonl   # raw/ 만 두고 새로 쓴다
$P/kdp reassemble $W/kdp4 --config configs/parse/_off.yaml > $W/kdp4.log 2>&1; mark "다시 조립 $?"
$P/python eval/block_diff.py $W/kdp_pipe $W/kdp4 --out $W/block_diff4.md > /dev/null 2>&1; mark "블록 견주기 $? — $(head -1 $W/block_diff4.md)"

for v in ocr spot; do rm -rf $W/kdp_$v && cp -r $W/kdp4 $W/kdp_$v; done
UTIL=0.5 bash gpu_server/serve_vlm.sh > $W/vllm4.log 2>&1 &
V=$!
if up 8118; then
  for v in ocr spot; do
    $P/kdp figure-ocr $W/kdp_$v --config configs/parse/_$v.yaml > $W/figure_ocr_$v.log 2>&1; mark "그림 다시 읽기 $v $?"
  done
else
  mark "PaddleOCR-VL 서버 실패"
fi
kill $V; wait $V 2>/dev/null; free_gpu

$S gpu_server/serve_hf_vlm.py > $W/desc4.log 2>&1 &   # 기본 모델 skt/A.X-4.0-VL-Light(공개) — 튜닝 어댑터는 쓰지 않는다
D=$!
if up 8119; then
  $P/kdp describe $W/kdp4 --config configs/parse/_off.yaml > $W/describe_kdp4.log 2>&1; mark "그림 설명 kdp4 $?"
  for v in ocr spot; do
    $P/kdp describe $W/kdp_$v --config configs/parse/_$v.yaml > $W/describe_$v.log 2>&1; mark "그림 설명 $v $?"
  done
else
  mark "그림 설명 서버 실패"
fi
kill $D; wait $D 2>/dev/null; free_gpu
rm -f configs/parse/_off.yaml configs/parse/_ocr.yaml configs/parse/_spot.yaml

$Q/python eval/rag_eval.py --kdp $W/kdp4 --compare $W/compare --variant kdp_rag_pipe=$W/kdp_pipe \
  --variant kdp_rag_ocr=$W/kdp_ocr --variant kdp_rag_spot=$W/kdp_spot \
  --retrievers bm25,bge-m3,bge-m3+rr --chunkers lightrag_r,lightrag,structure --out $W/report4 > $W/rag_eval4.log 2>&1
mark "검색 평가 $?"

rm -rf $W/lightrag_q && cp -r $W/lightrag_rpipe $W/lightrag_q   # 사본에 — 2회차 저장소는 ② 의 캐시 원본으로 그대로 둔다
for n in base rr; do
  $Q/python eval/lightrag_eval.py --kdp $W/kdp_pipe --compare $W/compare --work $W/lightrag_q --out $W/lightrag_report_q$n \
    --runs kdp_rag_pipe:512:R --variant kdp_rag_pipe=$W/kdp_pipe --query-only --no-llm --rerank $([ $n = base ] && echo none) \
    > $W/lightrag_q$n.log 2>&1
  mark "LightRAG 묻기만 $n $?"
done

bash gpu_server/serve_llm.sh > $W/llm4.log 2>&1 &
L=$!
if up 8120; then
  for mt in 2048 4096; do
    $Q/python eval/lightrag_eval.py --kdp $W/kdp_pipe --compare $W/compare --work $W/lightrag_mt$mt --out $W/lightrag_report_mt$mt \
      --runs kdp_rag_pipe:512:R --variant kdp_rag_pipe=$W/kdp_pipe --llm-cache-from $W/lightrag_rpipe --llm-max-tokens $mt --rerank none \
      > $W/lightrag_mt$mt.log 2>&1
    mark "LightRAG 추출 한도 $mt $?"
  done
else
  mark "LLM 서버 실패"
fi
kill $L; wait $L 2>/dev/null; free_gpu
mark 끝
