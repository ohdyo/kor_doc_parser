#!/bin/bash
# 이 컴퓨터 → GPU 서버: 코드, 공개 시험 문서(test2·test3), 검출기 가중치, 비교군 결과(공개 문서만).
# 사용: bash gpu_server/push.sh <주소> <포트> [ssh 키]      — 코드를 고칠 때마다 다시 돌린다(서버 configs/env.yaml 은 두고)
set -euo pipefail
HOST=$1 PORT=$2 KEY=${3:-$HOME/.ssh/id_ed25519}
cd "$(dirname "$0")/.."
E="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new"
DEST=root@$HOST
PUBLIC=(test2.pdf test3.pdf)                         # 서버에 올리는 PDF — 공개 정부 문서만, 이름으로
$E "$DEST" 'mkdir -p /workspace/kor_doc_parser /root/.paddlex/official_models'
rsync -az -e "$E" --exclude .git --exclude out --exclude out_gpu --exclude '*.pdf' --exclude '*.PDF' \
  --exclude configs/env.yaml --exclude __pycache__ --exclude '*.egg-info' --exclude .pytest_cache ./ "$DEST:/workspace/kor_doc_parser/"
rsync -az -e "$E" "${PUBLIC[@]}" "$DEST:/workspace/kor_doc_parser/"
rsync -az -e "$E" ~/.paddlex/official_models/PP-DocLayoutV3 ~/.paddlex/official_models/PP-LCNet_x1_0_doc_ori \
  "$DEST:/root/.paddlex/official_models/"
for pdf in "${PUBLIC[@]}"; do                        # 비교군 결과(텍스트 레이어·docling·PaddleOCR-VL 기본) — 서버에서 다시 돌리지 않게
  id=$(sha256sum "$pdf" | cut -c1-64)
  [ -d "out/$id/compare" ] && rsync -az -e "$E" --relative out/"$id"/compare/{textlayer,docling,paddle_default}.json \
    "$DEST:/workspace/kor_doc_parser/" || true
done
echo PUSH_DONE
