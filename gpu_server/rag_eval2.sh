#!/bin/bash
# 서버: RAG 하류 평가 2회차(D21 다음 — 표 꼴·LightRAG 구조 자르기, `테스트_결과.md` 14.7). rag_eval.sh 를 한 번 돌린 서버에서 그 비교군
# ($W/compare)과 1회차 kdp 출력($W/kdp)을 다시 쓴다. LightRAG 저장소는 만들지 않는다 — 검색 평가 뒤 고른 후보 하나만 따로(README).
#   1) 단위 시험  2) kdp 다시 조립 두 벌 — 표 HTML(rag_table_format: html)·파이프 표(기본 설정 — 2회차 때는 html 이 기본이었다).
#      정본·document.md·blocks.jsonl 은 원래 출력과, HTML 판 document.rag.md 는 1회차($W/kdp — 기본값을 바꾸기 전 rag_eval.sh 출력)와
#      같아야 한다(status2.txt 의 '같음')  3) 검색 평가 — 계통 여섯(파이프 표 판을 더함) ×
#      LightRAG F·R 512토큰·구조 자르기 × BM25·BGE-M3
#   tmux new -d -s rag2 "bash /workspace/kor_doc_parser/gpu_server/rag_eval2.sh"   진행: /workspace/rag_eval/status2.txt
set -uo pipefail
export LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1
R=/workspace/kor_doc_parser P=/workspace/envs/kdp-paddle/bin Q=/workspace/envs/kdp-rag/bin W=/workspace/rag_eval
cd $R
mark() { echo "$(date -u '+%F %T') $*" | tee -a $W/status2.txt; }
same() {       # $1 기준 파일, $2 새 파일, $3 이름
  if cmp -s "$1" "$2"; then mark "같음 $3"; else mark "다름 $3"; fi
}
reassemble() { # $1 출력 폴더, $2 설정(없으면 기본) — runs/t1/out_base 사본의 raw/ 만 두고 새로 쓴다
  rm -rf $1 && cp -r runs/t1/out_base $1
  rm -f $1/*/document.json $1/*/document.md $1/*/document.rag.md $1/*/blocks.jsonl
  $P/kdp reassemble $1 ${2:+--config $2} > $1.log 2>&1; mark "다시 조립 $(basename $1) $?"
  for d in runs/t1/out_base/*/; do
    id=$(basename "$d")
    for f in document.json document.md blocks.jsonl; do same "$d/$f" "$1/$id/$f" "$(basename $1) ${id:0:8} $f"; done
  done
}

mark 시작
$P/python -m pytest -q tests/unit > $W/pytest2.log 2>&1; mark "단위 시험 $? — $(tail -1 $W/pytest2.log)"
sed 's/rag_table_format: pipe/rag_table_format: html/' configs/parse/default.yaml > configs/parse/_rag_html.yaml   # 설정 폴더에 — paddlex 설정 상대 경로
reassemble $W/kdp_html configs/parse/_rag_html.yaml
rm -f configs/parse/_rag_html.yaml
for d in $W/kdp/*/; do id=$(basename "$d"); same "$d/document.rag.md" "$W/kdp_html/$id/document.rag.md" "1회차 HTML ${id:0:8} document.rag.md"; done
reassemble $W/kdp_pipe
$Q/python eval/rag_eval.py --kdp $W/kdp_html --compare $W/compare --variant kdp_rag_pipe=$W/kdp_pipe \
  --chunkers lightrag,lightrag_r,structure --out $W/report2 > $W/rag_eval2.log 2>&1; mark "검색 평가 $?"
mark 끝
