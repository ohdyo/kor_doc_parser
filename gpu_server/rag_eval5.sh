#!/bin/bash
# 서버: 4회차 뒤 기본값 점검(`테스트_결과.md` 14.10) — rag_eval4.sh 를 돌린 서버에서 그 출력($W/kdp4·kdp_spot·kdp_pipe)과 비교군을 쓴다.
# 그림 다시 읽기 기본 설정(그래프 블록만 Spotting: · 한자·가나 줄 거르기)은 4회차 Spot 기록(raw/figure_ocr.jsonl — 열쇠가 같다)만 읽어
# PaddleOCR-VL 서버를 띄우지 않는다.
#   1) 단위 시험  2) $W/kdp4 사본($W/kdp5)에 4회차 Spot 다시 읽기 기록과 설명 기록(kdp4·kdp_spot)을 넣어 기본 설정으로 다시 조립 —
#      kdp figure-ocr 가 새로 부를 그림이 없어야 한다(네 문서 모두 '새 읽기 0  실패 0')  3) 그림 설명 서버(A.X 기본 모델)로 kdp describe
#      (그림 글이 바뀐 그림만 새로 부른다) 뒤 4회차 Spot 판과 블록마다 견준다(eval/block_diff.py)  4) 검색 평가 — kdp5·2회차 파이프 표 판(③ 전)·
#      4회차 Spot 판 × BM25·BGE-M3·BGE-M3→다시 순위 × LightRAG R·F 512·구조
#   tmux new -d -s rag5 "bash /workspace/kor_doc_parser/gpu_server/rag_eval5.sh"   진행: /workspace/rag_eval/status5.txt
set -uo pipefail
export LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1
R=/workspace/kor_doc_parser P=/workspace/envs/kdp-paddle/bin Q=/workspace/envs/kdp-rag/bin S=/workspace/envs/kdp-serve/bin/python
W=/workspace/rag_eval
cd $R
mark() { echo "$(date -u '+%F %T') $*" | tee -a $W/status5.txt; }
up() { for _ in $(seq 180); do curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null && return 0; sleep 5; done; return 1; }
free_gpu() {   # 서버를 내린 뒤 GPU 가 빌 때까지(vLLM 엔진 자식 프로세스가 늦게 내려간다)
  for _ in $(seq 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ] && return 0; sleep 2; done
  pkill -f VLLM::EngineCore; sleep 5
}

mark 시작
$P/python -m pytest -q tests/unit > $W/pytest5.log 2>&1; mark "단위 시험 $? — $(tail -1 $W/pytest5.log)"

rm -rf $W/kdp5 && cp -r $W/kdp4 $W/kdp5
rm -f $W/kdp5/*/document.json $W/kdp5/*/document.md $W/kdp5/*/document.rag.md $W/kdp5/*/blocks.jsonl   # raw/ 만 두고 새로 쓴다
for d in $W/kdp5/*/; do
  id=$(basename "$d")
  cp $W/kdp_spot/$id/raw/figure_ocr.jsonl "$d/raw/"
  cat $W/kdp_spot/$id/raw/describe.jsonl >> "$d/raw/describe.jsonl"   # 같은 열쇠는 뒤 줄 — 같은 물음의 응답이다
done
$P/kdp reassemble $W/kdp5 > $W/kdp5.log 2>&1; mark "다시 조립 $?"
$P/kdp figure-ocr $W/kdp5 > $W/figure_ocr5.log 2>&1   # 서버가 없어 기록에 없는 그림은 실패로 센다 — 새 읽기 0·실패 0 이어야 한다
mark "다시 읽기 기록 $? — 새 읽기 0·실패 0 인 문서 $(grep -c '새 읽기 0  실패 0 ' $W/figure_ocr5.log)/4"

$S gpu_server/serve_hf_vlm.py > $W/desc5.log 2>&1 &   # 기본 모델 skt/A.X-4.0-VL-Light(공개) — 튜닝 어댑터는 쓰지 않는다
D=$!
if up 8119; then
  $P/kdp describe $W/kdp5 > $W/describe_kdp5.log 2>&1; mark "그림 설명 $? — $(grep -o '새 설명 [0-9]*' $W/describe_kdp5.log | tr '\n' ' ')"
else
  mark "그림 설명 서버 실패"
fi
kill $D; wait $D 2>/dev/null; free_gpu
$P/python eval/block_diff.py $W/kdp_spot $W/kdp5 --out $W/block_diff5.md > /dev/null 2>&1; mark "블록 견주기 $? — $(head -1 $W/block_diff5.md)"

$Q/python eval/rag_eval.py --kdp $W/kdp5 --compare $W/compare --variant kdp_rag_pipe=$W/kdp_pipe --variant kdp_rag_spot=$W/kdp_spot \
  --chunkers lightrag_r,lightrag,structure --out $W/report5 > $W/rag_eval5.log 2>&1
mark "검색 평가 $?"
mark 끝
