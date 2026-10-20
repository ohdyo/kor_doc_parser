#!/bin/bash
# 서버: 그림 해석 튜닝 자료의 그림 자르기(기획서 7.10) — push.sh 가 올린 /workspace/tt_dt/{pdf,tr_dt/pdf} 의 PDF 를 그림만 설정
# (configs/parse/figures_only.yaml — 그림 밖 VLM 생략)으로 파싱해 /workspace/tt_dt/parsed/<sha256>/ 에 둔다. 이미 끝난 문서는 건너뛴다.
# PaddleOCR-VL 서버(serve_vlm.sh)를 먼저 띄운다. 끝나면 이 컴퓨터에서 pull.sh 로 가져와 라벨링한다(서버는 그동안 합성 학습).
# 사용(서버): bash gpu_server/crop_tt.sh
set -uo pipefail
cd "$(dirname "$0")/.."
TT=${TT_DT:-/workspace/tt_dt}
OUT=$TT/parsed
mkdir -p "$OUT/logs"
for f in "$TT"/pdf/*.pdf "$TT"/tr_dt/pdf/*.pdf; do
  [ -e "$f" ] || continue
  id=$(sha256sum "$f" | cut -c1-64)
  [ -f "$OUT/$id/document.json" ] && continue
  n=$(basename "$f" .pdf); s=$(date +%s)
  /workspace/envs/kdp-paddle/bin/kdp parse "$f" -o "$OUT" --config configs/parse/figures_only.yaml > "$OUT/logs/$n.log" 2>&1
  echo "$n 끝 코드=$? $(( $(date +%s) - s ))초" | tee -a "$OUT/logs/_done.txt"
done
echo CROP_DONE
