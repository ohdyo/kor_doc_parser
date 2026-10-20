#!/bin/bash
# 서버: 그림 해석 튜닝 1차 시범(T1, 기획서 7.10)을 한 번에 — setup_envs.sh 뒤에 tmux 로 돌린다. 설명(A.X)·PaddleOCR-VL 서버는 이
# 스크립트가 띄우고 내린다(먼저 떠 있으면 안 된다 — 학습이 GPU 를 혼자 쓴다).
#   1) 단위 시험(원래 A.X 단계와 함께)  2) 원래 A.X — 채점 물음(eval/figure_questions.py ask)·잊음 검사용 test2~5 파싱(설명 켬)
#   3) 50 step 시험(시간·메모리)  4) 본 학습(configs/tune/figdesc_v1.yaml)  5) 튜닝 A.X — 같은 물음·파싱  6) 채점
# 결과는 /workspace/t1(status.txt 에 단계마다 시각·종료 코드), 어댑터는 /workspace/runs/figdesc_v1/adapter. 끝나면 이 컴퓨터에서 pull.sh.
#   tmux new -d -s t1 "bash /workspace/kor_doc_parser/gpu_server/tune_t1.sh"
set -uo pipefail
export LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1
R=/workspace/kor_doc_parser S=/workspace/envs/kdp-serve/bin/python P=/workspace/envs/kdp-paddle/bin W=/workspace/t1
ADAPTER=/workspace/runs/figdesc_v1/adapter
mkdir -p $W && cd $R
mark() { echo "$(date -u '+%F %T') $*" | tee -a $W/status.txt; }
up() { for _ in $(seq 180); do curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null && return 0; sleep 5; done; return 1; }
free_gpu() {   # 서버를 내린 뒤 GPU 가 빌 때까지(vLLM 엔진 자식 프로세스가 늦게 내려간다)
  for _ in $(seq 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ] && return 0; sleep 2; done
  pkill -f VLLM::EngineCore; sleep 5
}

stage() {      # $1 이름(base·tuned), 나머지는 serve_hf_vlm.py 인자
  local name=$1; shift
  $S gpu_server/serve_hf_vlm.py "$@" > $W/serve_$name.log 2>&1 &
  local d=$! v=
  if up 8119; then                                           # vLLM 은 A.X 가 다 올라간 뒤에 — 뜨면서 남은 메모리를 잰다
    UTIL=0.25 bash gpu_server/serve_vlm.sh > $W/vllm_$name.log 2>&1 &
    v=$!
  fi
  if [ -z "$v" ] || ! up 8118; then mark "$name 서버 실패"; kill $d $v; free_gpu; return 1; fi
  $P/python eval/figure_questions.py ask --out $W/ans_$name.jsonl > $W/ask_$name.log 2>&1; mark "$name 채점 물음 $?"
  for t in test2 test3 test4 test5; do
    $P/kdp parse $t.pdf -o $W/out_$name > $W/parse_${name}_$t.log 2>&1; mark "$name 파싱 $t $?"
  done
  kill $d $v; wait $d $v 2>/dev/null; free_gpu
}

mark 시작
$P/python -m pytest -q tests/unit > $W/pytest.log 2>&1 &
T=$!
stage base || exit 1
wait $T; mark "단위 시험 $? — $(tail -1 $W/pytest.log)"
sed -e 's#/workspace/runs/figdesc_v1#/workspace/runs/figdesc_try#' -e 's/val_n: 200/val_n: 20/' configs/tune/figdesc_v1.yaml > $W/try.yaml
$S train/figdesc/train_lora.py $W/try.yaml --max-steps 50 > $W/try.log 2>&1; rc=$?; mark "50 step 시험 $rc"
[ $rc = 0 ] || exit 1
$S train/figdesc/train_lora.py configs/tune/figdesc_v1.yaml > $W/train.log 2>&1; mark "본 학습 $?"
[ -f $ADAPTER/adapter_model.safetensors ] || exit 1        # 끝의 합성 검증이 멈춰도 어댑터가 있으면 평가는 한다
stage tuned --adapter $ADAPTER || exit 1
$P/python eval/figure_questions.py score $W/ans_base.jsonl $W/ans_tuned.jsonl > $W/score.txt 2>&1; mark "채점 $?"
mark 끝
