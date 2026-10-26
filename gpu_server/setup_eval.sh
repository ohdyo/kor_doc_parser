#!/bin/bash
# RAG 하류 평가(D21 — 기획서 8.6 축소판)용 추가 설치 — setup_envs.sh 다음에 서버에서 한 번(README.md).
#   kdp-rag    : 검색 평가(eval/rag_eval.py·eval/lightrag_eval.py) — lightrag-hku 1.5.7(MIT, 벡터 저장소 nano-vectordb MIT 가 함께 깔린다)·
#                tiktoken(MIT)·openai 라이브러리(Apache-2.0 — 서버 안 vLLM 을 부를 때만, 유료 API 는 쓰지 않는다)·torch·transformers 4.57(BGE-M3, MIT)
#   kdp-docling: 비교군 docling 2.138.0(MIT, CPU torch) + 이 저장소(--no-deps) — docling 은 파서 env 밖에서 돌린다(설계 9장)
# 사용: bash /workspace/kor_doc_parser/gpu_server/setup_eval.sh
set -euo pipefail
export LC_ALL=C.UTF-8
ROOT=/workspace/kor_doc_parser
ENVS=/workspace/envs
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "$DRIVER" -ge 580 ]; then TB=(); else TB=(--torch-backend cu129); fi   # setup_envs.sh 와 같은 기준 — 580 미만은 CUDA 12.9 판
uv venv "$ENVS/kdp-rag" --python 3.12
VIRTUAL_ENV="$ENVS/kdp-rag" uv pip install "${TB[@]}" "torch==2.13.0" "transformers>=4.57,<4.58" "lightrag-hku==1.5.7" \
  "openai>=2,<4" "tiktoken>=0.7" "langchain-text-splitters>=0.3,<2" pyyaml numpy   # LightRAG R 전략(구조 자르기)
uv venv "$ENVS/kdp-docling" --python 3.12
VIRTUAL_ENV="$ENVS/kdp-docling" uv pip install --torch-backend cpu "docling==2.138.0" "pypdfium2==5.14.0" "pydantic>=2.5" \
  "PyYAML>=6.0" "typer>=0.12" numpy Pillow
cd "$ROOT" && VIRTUAL_ENV="$ENVS/kdp-docling" uv pip install --no-deps -e .
# tiktoken 은 처음 쓸 때 o200k_base 표를 받는다 — 여기서 받아 두고, 못 받으면 평가 전에 멈춘다
"$ENVS/kdp-rag/bin/python" -c "import importlib.metadata as m, torch, nano_vectordb, tiktoken; tiktoken.encoding_for_model('gpt-4o-mini'); print('kdp-rag', torch.__version__, torch.cuda.is_available(), m.version('transformers'), m.version('lightrag-hku'), m.version('nano-vectordb'))"
"$ENVS/kdp-docling/bin/python" -c "import importlib.metadata as m, docling, kdp; print('kdp-docling', m.version('docling'))"
echo SETUP_EVAL_DONE
