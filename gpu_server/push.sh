#!/bin/bash
# 이 컴퓨터 → GPU 서버: 코드, 공개 시험 문서(test2~test5), 그림 해석 튜닝의 시험·학습 PDF(tt_dt — 공공누리 공개 문서만, 기획서 7.10),
# 검출기 가중치, 비교군 결과(공개 문서만).
# 사용: bash gpu_server/push.sh <주소> <포트> [ssh 키]      — 코드를 고칠 때마다 다시 돌린다(서버 configs/env.yaml 은 두고)
set -euo pipefail
HOST=$1 PORT=$2 KEY=${3:-$HOME/.ssh/id_ed25519}
cd "$(dirname "$0")/.."
E="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new"
DEST=root@$HOST
PUBLIC=(test2.pdf test3.pdf test4.pdf test5.pdf)     # 서버에 올리는 PDF — 공개 정부 문서만, 이름으로
$E "$DEST" 'mkdir -p /workspace/kor_doc_parser /root/.paddlex/official_models'
rsync -az -e "$E" --exclude .git --exclude out --exclude out_gpu --exclude datasets --exclude runs --exclude '*.pdf' --exclude '*.PDF' \
  --exclude configs/env.yaml --exclude __pycache__ --exclude '*.egg-info' --exclude .pytest_cache ./ "$DEST:/workspace/kor_doc_parser/"
rsync -az -e "$E" "${PUBLIC[@]}" "$DEST:/workspace/kor_doc_parser/"
TT=${TT_DT:-/hdd_data/raganything/tt_dt}             # 시험(pdf/)·학습(tr_dt/pdf/) PDF — 있으면 서버 /workspace/tt_dt 로
for sub in pdf tr_dt/pdf; do
  if [ -d "$TT/$sub" ]; then
    $E "$DEST" "mkdir -p /workspace/tt_dt/$sub"
    rsync -az -e "$E" --include '*.pdf' --exclude '*' "$TT/$sub/" "$DEST:/workspace/tt_dt/$sub/"
  fi
done
SRC=()                                                # 그림 해석 학습 자료 — 합성(synth)·실제 그림 정답(real)·그 그림(label 묶음)
for sub in synth real label; do [ -d "$TT/tr_dt/$sub" ] && SRC+=("$TT/tr_dt/$sub"); done
if [ ${#SRC[@]} -gt 0 ]; then                         # 라벨러의 확대 사본·지시문·몫 파일은 보내지 않는다
  $E "$DEST" "mkdir -p /workspace/tt_dt/tr_dt"
  rsync -az -e "$E" --exclude 'zoom_*' --exclude 'guide_*' --exclude 'figures_p*.json' "${SRC[@]}" "$DEST:/workspace/tt_dt/tr_dt/"
fi
rsync -az -e "$E" ~/.paddlex/official_models/PP-DocLayoutV3 ~/.paddlex/official_models/PP-LCNet_x1_0_doc_ori \
  "$DEST:/root/.paddlex/official_models/"
if [ -d runs/t1/out_base ]; then                     # RAG 하류 평가(D21, rag_eval.sh)의 kdp 기준 출력 — test2~5 원래 A.X 파싱(raw 포함)
  rsync -az -e "$E" --relative runs/t1/out_base "$DEST:/workspace/kor_doc_parser/"
fi
RAG=()                                                # RAG 하류 평가 기준(D21·D22) — 비교군 출력·2회차 파이프 표 출력·2회차 LightRAG 저장소와
for f in compare kdp_pipe lightrag_rpipe lightrag_eval_rpipe.log; do   # 그 로그. 새로 켠 서버에서 다시 만들지 않게(runs/ 는 커밋하지 않는다)
  [ -e "runs/rag_eval/$f" ] && RAG+=("runs/rag_eval/$f")
done
for f in kdp4 kdp_spot; do                            # 4회차 그림 설명·Spotting: 다시 읽기 기록 — 5회차 최종 출력을 다시 만든다(rag_7b31b.sh)
  [ -d "runs/rag_eval4/$f" ] && RAG+=("runs/rag_eval4/$f")
done
if [ ${#RAG[@]} -gt 0 ]; then
  $E "$DEST" 'mkdir -p /workspace/rag_eval'
  rsync -az -e "$E" "${RAG[@]}" "$DEST:/workspace/rag_eval/"
fi
for pdf in "${PUBLIC[@]}"; do                        # 비교군 결과(텍스트 레이어·docling·PaddleOCR-VL 기본) — 서버에서 다시 돌리지 않게
  id=$(sha256sum "$pdf" | cut -c1-64)
  [ -d "out/$id/compare" ] && rsync -az -e "$E" --relative out/"$id"/compare/{textlayer,docling,paddle_default}.json \
    "$DEST:/workspace/kor_doc_parser/" || true
done
echo PUSH_DONE
