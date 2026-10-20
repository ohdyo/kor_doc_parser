#!/bin/bash
# GPU 서버 → 이 컴퓨터: 서버 out/ (파싱 결과·raw) 를 out_gpu/ 로 가져온다. 서버를 끄기 전에 돌린다.
# 사용: bash gpu_server/pull.sh <주소> <포트> [ssh 키]
set -euo pipefail
HOST=$1 PORT=$2 KEY=${3:-$HOME/.ssh/id_ed25519}
cd "$(dirname "$0")/.."
rsync -az -e "ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new" "root@$HOST:/workspace/kor_doc_parser/out/" out_gpu/
TT=${TT_DT:-/hdd_data/raganything/tt_dt}             # 그림 해석 튜닝 자료의 그림 자르기 결과(crop_tt.sh) — 서버에 있으면 함께
if ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new "root@$HOST" test -d /workspace/tt_dt/parsed; then
  rsync -az -e "ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new" "root@$HOST:/workspace/tt_dt/parsed/" "$TT/parsed/"
fi
echo PULL_DONE
