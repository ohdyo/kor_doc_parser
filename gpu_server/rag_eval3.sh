#!/bin/bash
# 서버: 기본값을 파이프 표·LightRAG R 전략으로 바꾼 뒤 점검(`테스트_결과.md` 14.9). rag_eval2.sh 를 돌린 서버에서 그 비교군($W/compare)과
# 2회차 파이프 표 판($W/kdp_pipe)을 기준으로 쓴다. LightRAG 저장소는 만들지 않는다 — 같은 글·같은 넣기로 2회차에 만든 R + 파이프 표 저장소가 있다.
#   1) 단위 시험  2) kdp 를 기본 설정으로 다시 조립($W/kdp) — 정본·document.md·blocks.jsonl 은 원래 출력과, document.rag.md 는 2회차
#      파이프 표 판과 바이트까지 같아야 한다(status3.txt 의 '같음')  3) 검색 평가 — 기본 자르기(LightRAG R 512·F 512·F 1,200토큰·구조) × BM25·BGE-M3
#   tmux new -d -s rag3 "bash /workspace/kor_doc_parser/gpu_server/rag_eval3.sh"   진행: /workspace/rag_eval/status3.txt
set -uo pipefail
export LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1
R=/workspace/kor_doc_parser P=/workspace/envs/kdp-paddle/bin Q=/workspace/envs/kdp-rag/bin W=/workspace/rag_eval
cd $R
mark() { echo "$(date -u '+%F %T') $*" | tee -a $W/status3.txt; }
same() {       # $1 기준 파일, $2 새 파일, $3 이름
  if cmp -s "$1" "$2"; then mark "같음 $3"; else mark "다름 $3"; fi
}

mark 시작
$P/python -m pytest -q tests/unit > $W/pytest3.log 2>&1; mark "단위 시험 $? — $(tail -1 $W/pytest3.log)"
rm -rf $W/kdp && cp -r runs/t1/out_base $W/kdp
rm -f $W/kdp/*/document.json $W/kdp/*/document.md $W/kdp/*/document.rag.md $W/kdp/*/blocks.jsonl   # raw/ 만 두고 새로 쓴다
$P/kdp reassemble $W/kdp > $W/kdp.log 2>&1; mark "다시 조립 $?"
for d in runs/t1/out_base/*/; do
  id=$(basename "$d")
  for f in document.json document.md blocks.jsonl; do same "$d/$f" "$W/kdp/$id/$f" "${id:0:8} $f"; done
  same "$W/kdp_pipe/$id/document.rag.md" "$W/kdp/$id/document.rag.md" "2회차 파이프 ${id:0:8} document.rag.md"
done
$Q/python eval/rag_eval.py --kdp $W/kdp --compare $W/compare --out $W/report3 > $W/rag_eval3.log 2>&1; mark "검색 평가 $?"
mark 끝
