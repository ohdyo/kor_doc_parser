#!/bin/bash
# 서버: 7B 대 31B 답 비교(`기획서.md` D25) — 같은 저장소·같은 질문·같은 채점으로 A.X-4.0-Light(7B, bf16)와 Webs-Sejong-31B-v7(31B,
# FP8 — quantize_fp8.sh)의 답을 견준다. 31B 가 높을 것은 알고, 7B 가 최적화나 학습으로 따라갈 수 있는지 판단할 자료를 모은다 — 격차의
# 크기, 격차가 생기는 곳(그래프 만들기 / 답 쓰기, 검색 / 읽기), 학습 없는 최적화로 메우는 몫.
# 새로 켠 서버(/workspace 볼륨 200GB 이상): 이 컴퓨터 push.sh → 서버 setup_envs.sh → setup_eval.sh → 이 스크립트(모델 서버는 스스로 띄우고 내린다).
#   0) 단위 시험  1) 31B 받기·FP8(CPU — 배경에서 2 와 함께)  2) 5회차 최종 출력(kdp5) 다시 만들기 — runs/t1/out_base 에 4회차 설명·
#      Spotting: 다시 읽기 기록을 넣어 기본 설정으로 다시 조립(rag_eval5.sh 와 같은 기록), 그림 설명 서버로 바뀐 그림만 설명
#   3) 31B 미리 띄우기 — KV 자리·생각 모드를 앞에서 확인한다(FP8 사본이 없거나 뜨지 않으면 bf16 을 vLLM 온라인 FP8(fp8_per_tensor)로,
#      그래도 안 되면 31B 칸은 빼고 7B 만 돈다)  4) 7B 저장소(lightrag_final — 2회차 LLM 캐시, rag_answer.sh 와 같다)  5) 7B 답
#   6) 31B 답(7B 저장소) → 31B 저장소 → 31B 답(31B 저장소)  7) 7B 답(31B 저장소)
# 31B vLLM: 몫 0.85·문맥 24,576·--language-model-only(그림·영상 인코더를 올리지 않는다 — 영상 갈래가 켜져 있으면 한 번에 2,496토큰·
#   슬라이딩 층 KV 자리가 커져 A40 에 최대 길이 요청 하나도 못 담는다)·한 번에 1,024토큰·동시 32. 같은 GPU 에 평가 프로세스가 함께 돈다.
# 답(eval/answer_eval.py — 결과 폴더 $C/<칸>). 질문 M = 답 비교 50 + 추가 24(없음 12·종합 8·문서간 4), R = 검색 평가 72,
# O = 근거만 주고 묻기(근거 낱말이 있는 한 문서 질문 — M·R 에서).
#   7b·31b        M  mix·naive·bypass (7B 저장소)          7b_r72·31b_r72  R  naive
#   7b_or·31b_or  O  naive 근거만                           7b_g31·31b_g31  M  mix (31B 저장소)
#   7b_rule M naive 답 지시 · 7b_short M naive 문맥 6,000 · 7b_both·31b_both M naive 둘 다 (eval/prompts/answer_rules.txt)
#   tmux new -d -s cmp "bash /workspace/kor_doc_parser/gpu_server/rag_7b31b.sh"   진행: /workspace/rag_eval/status_cmp.txt
set -uo pipefail
export LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1
R=/workspace/kor_doc_parser P=/workspace/envs/kdp-paddle/bin Q=/workspace/envs/kdp-rag/bin S=/workspace/envs/kdp-serve/bin/python
W=/workspace/rag_eval C=/workspace/rag_eval/cmp
M7=skt/A.X-4.0-Light M31=websfactory/Webs-Sejong-31B-v7 RAW31=/workspace/models/Webs-Sejong-31B-v7 Q31=/workspace/models/Webs-Sejong-31B-v7-FP8
QM=eval/golden/rag_answer_questions.yaml,eval/golden/rag_answer_questions2.yaml QR=eval/golden/rag_questions.yaml
RULES=eval/prompts/answer_rules.txt T31=1200                   # 31B 는 느려 LLM 한 번의 시간 한도를 LightRAG 기본 240초에서 늘린다
cd $R && mkdir -p $C
mark() { echo "$(date -u '+%F %T') $*" | tee -a $W/status_cmp.txt; }
up() {   # 서버가 뜰 때까지(최대 30분) — 서버 프로세스($2)가 죽으면 바로 실패
  for _ in $(seq 360); do
    curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}
free_gpu() {   # 서버를 내린 뒤 GPU 가 빌 때까지(vLLM 엔진 자식 프로세스가 늦게 내려간다)
  for _ in $(seq 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ] && return 0; sleep 2; done
  pkill -f VLLM::EngineCore; sleep 5
}
stop() { kill $L 2>/dev/null; wait $L 2>/dev/null; free_gpu; }
probe() {   # 생각 모드가 새지 않는지 — enable_thinking false 로 짧게 물어 답이 생각 꼴(thought·<|channel>)로 시작하지 않아야 한다
  curl -s http://127.0.0.1:8120/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\": \"$1\", \"temperature\": 0,
    \"max_tokens\": 32, \"messages\": [{\"role\": \"user\", \"content\": \"대한민국의 수도는? 한 낱말로.\"}],
    \"chat_template_kwargs\": {\"enable_thinking\": false}}" > $W/probe.json
  $Q/python - $W/probe.json <<'EOF'
import json
import sys

t = (json.load(open(sys.argv[1]))["choices"][0]["message"]["content"] or "").strip()
print(repr(t[:60]))
sys.exit(1 if not t or t.lower().startswith("thought") or "channel" in t else 0)
EOF
}
start31() {   # 31B 서버 — 모델 폴더 [더할 vLLM 인자…]. 0 준비, 1 서버가 뜨지 않음, 2 생각 모드가 샌다
  local m=$1
  shift
  MODEL=$m NAME=$M31 UTIL=0.85 LEN=24576 EXTRA="--language-model-only --max-num-batched-tokens 1024 --max-num-seqs 32 $*" \
    bash gpu_server/serve_llm.sh > $W/llm_31b.log 2>&1 &
  L=$!
  up 8120 $L || return 1
  probe $M31 >> $W/status_cmp.txt || return 2
}
ask() {   # 칸 저장소 모델 [answer_eval 인자…]
  local cell=$1 work=$2 model=$3
  shift 3
  $Q/python eval/answer_eval.py --work "$work" --llm-model "$model" --out "$C/$cell" "$@" > "$C/$cell.log" 2>&1
  mark "답 $cell $? — 성공 $(grep -c ' success ' "$C/$cell.log"), 실패 $(grep -cE ' (error|failure) ' "$C/$cell.log")"
}

mark 시작
$P/python -m pytest -q tests/unit > $W/pytest_cmp.log 2>&1; mark "단위 시험 $? — $(tail -1 $W/pytest_cmp.log)"

KEEP_RAW=1 bash gpu_server/quantize_fp8.sh > $W/quant.log 2>&1 &   # bf16 은 31B 단계가 끝날 때까지 둔다(FP8 사본이 안 뜰 때 대비)
QP=$!
mark "31B 받기·FP8 시작(배경) — 디스크 /workspace 남은 $(df --output=avail -BG /workspace | tail -1 | tr -dc 0-9)GB"

# 2) 5회차 최종 출력 — rag_eval4.sh 처럼 runs/t1/out_base 사본에, rag_eval5.sh 와 같은 기록(4회차 설명 + Spotting: 다시 읽기·그 설명)
rm -rf $W/kdp5 && cp -r runs/t1/out_base $W/kdp5
rm -f $W/kdp5/*/document.json $W/kdp5/*/document.md $W/kdp5/*/document.rag.md $W/kdp5/*/blocks.jsonl   # raw/ 만 두고 새로 쓴다
for d in $W/kdp5/*/; do
  id=$(basename "$d")
  cp $W/kdp4/$id/raw/describe.jsonl "$d/raw/describe.jsonl"           # 4회차 설명 기록(원래 파싱의 기록을 담는다)
  cat $W/kdp_spot/$id/raw/describe.jsonl >> "$d/raw/describe.jsonl"    # 같은 열쇠는 뒤 줄 — 같은 물음의 응답이다
  cp $W/kdp_spot/$id/raw/figure_ocr.jsonl "$d/raw/"
done
$P/kdp reassemble $W/kdp5 > $W/kdp5.log 2>&1; mark "다시 조립 $?"
$P/kdp figure-ocr $W/kdp5 > $W/figure_ocr5.log 2>&1   # 서버 없이 기록만 — 네 문서 모두 '새 읽기 0  실패 0' 이어야 한다
mark "다시 읽기 기록 $? — 새 읽기 0·실패 0 인 문서 $(grep -c '새 읽기 0  실패 0 ' $W/figure_ocr5.log)/4"
$S gpu_server/serve_hf_vlm.py > $W/desc_cmp.log 2>&1 &   # 그림 설명 기본 모델 skt/A.X-4.0-VL-Light(공개)
L=$!
if up 8119 $L; then
  $P/kdp describe $W/kdp5 > $W/describe_cmp.log 2>&1; mark "그림 설명 $? — $(grep -o '새 설명 [0-9]*' $W/describe_cmp.log | tr '\n' ' ')"
else
  mark "그림 설명 서버 실패"
fi
stop

# 3) 31B 미리 띄우기 — 무인 실행 앞부분에서 KV 자리·생각 모드를 확인하고, 뜨는 길(FP8 사본 / 온라인 FP8)을 정해 둔다
wait $QP; QS=$?; mark "31B FP8 $QS — $(tail -2 $W/quant.log | tr '\n' ' ')"
s=1 Q31M=$Q31 Q31X="" Q31USED=FP8
if [ $QS -eq 0 ] && grep -q '"quantization_config"' $Q31/config.json 2>/dev/null; then
  start31 $Q31M; s=$?
  [ $s -eq 1 ] && cp $W/llm_31b.log $W/llm_31b_fp8.log
  stop
fi
if [ $s -eq 1 ] && [ -f $RAW31/model.safetensors.index.json ]; then   # 다른 길: bf16 을 읽으며 층마다 FP8 로(vLLM 에 든 기능, 가중치 무게는 같다)
  Q31M=$RAW31 Q31X="--quantization fp8_per_tensor" Q31USED=fp8_per_tensor
  start31 $Q31M $Q31X; s=$?
  stop
fi
echo "$Q31USED $s" > $C/quant31.txt
mark "31B 미리 띄우기 $s($Q31USED — 0 준비·1 안 뜸·2 생각 모드가 샘) $(grep -oE 'Available KV cache memory: [0-9.]+ GiB|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' $W/llm_31b.log | tr '\n' ' ')"

# 4)·5) 7B — 저장소와 답
bash gpu_server/serve_llm.sh > $W/llm_7b.log 2>&1 &
L=$!
if up 8120 $L; then
  $Q/python eval/lightrag_eval.py --kdp $W/kdp5 --compare $W/compare --work $W/lightrag_final --out $W/lightrag_report_final \
    --llm-cache-from $W/lightrag_rpipe > $W/lightrag_final.log 2>&1
  mark "7B 저장소 $?"
  ask 7b $W/lightrag_final $M7 --questions $QM
  ask 7b_r72 $W/lightrag_final $M7 --questions $QR --modes naive
  ask 7b_or $W/lightrag_final $M7 --questions $QM,$QR --oracle --variant oracle
  ask 7b_rule $W/lightrag_final $M7 --questions $QM --modes naive --variant rule --user-prompt-file $RULES
  ask 7b_short $W/lightrag_final $M7 --questions $QM --modes naive --variant short --max-total-tokens 6000
  ask 7b_both $W/lightrag_final $M7 --questions $QM --modes naive --variant both --user-prompt-file $RULES --max-total-tokens 6000
else
  mark "7B 서버 실패"
fi
stop

# 6) 31B — 7B 저장소에 답, 31B 저장소 만들기, 31B 저장소에 답
if [ $s -eq 0 ]; then
  start31 $Q31M $Q31X; s=$?
  if [ $s -eq 0 ]; then
    mark "31B 서버 준비($Q31USED)"
    ask 31b $W/lightrag_final $M31 --questions $QM --llm-timeout $T31
    ask 31b_r72 $W/lightrag_final $M31 --questions $QR --modes naive --llm-timeout $T31
    ask 31b_or $W/lightrag_final $M31 --questions $QM,$QR --oracle --variant oracle --llm-timeout $T31
    ask 31b_both $W/lightrag_final $M31 --questions $QM --modes naive --variant both --user-prompt-file $RULES --max-total-tokens 6000 \
      --llm-timeout $T31
    $Q/python eval/lightrag_eval.py --kdp $W/kdp5 --compare $W/compare --work $W/lightrag_31b --out $W/lightrag_report_31b \
      --llm-model $M31 --llm-timeout $T31 > $W/lightrag_31b.log 2>&1
    mark "31B 저장소 $?"
    ask 31b_g31 $W/lightrag_31b $M31 --questions $QM --modes mix --llm-timeout $T31
  else
    mark "31B 서버 다시 띄우기 실패 $s — $W/llm_31b.log·probe.json"
  fi
  stop
else
  mark "31B 칸 건너뜀 — 미리 띄우기 실패 $s($Q31USED, $W/quant.log·llm_31b.log·probe.json)"
fi
rm -rf $RAW31                                         # bf16 원본 — 31B 단계가 끝났으니 지운다(FP8 사본은 둔다)

# 7) 7B — 31B 저장소에 답(그래프만 큰 모델로 만들고 답은 7B 로 하는 경우)
if [ -d $W/lightrag_31b ]; then
  bash gpu_server/serve_llm.sh > $W/llm_7b2.log 2>&1 &
  L=$!
  if up 8120 $L; then
    ask 7b_g31 $W/lightrag_31b $M7 --questions $QM --modes mix
  else
    mark "7B 서버 실패(2)"
  fi
  stop
fi
mark 끝
