#!/bin/bash
# GPU 서버 환경 설치 — 서버를 새로 켤 때마다 서버에서 한 번 돌린다(README.md). 버전은 기획서 7.4.
#   kdp-serve : vLLM 서버(vllm 0.31.0 · torch 2.13.0(CUDA 13) · transformers 5.16)
#   kdp-paddle: 파서(paddlepaddle-gpu 3.3.1 cu126 · paddleocr 3.7.0 · paddlex 3.7.2 · 이 저장소)
# 사용: bash /workspace/kor_doc_parser/gpu_server/setup_envs.sh   (먼저 이 컴퓨터에서 push.sh 로 저장소를 보낸다)
set -euo pipefail
export LC_ALL=C.UTF-8
ROOT=/workspace/kor_doc_parser
ENVS=/workspace/envs
command -v uv >/dev/null || pip install -q --break-system-packages uv     # 시스템 pip 는 그냥은 설치를 거부한다
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq fonts-noto-cjk bc >/dev/null   # 테스트 한글 글꼴
uv venv "$ENVS/kdp-serve" --python 3.12
VIRTUAL_ENV="$ENVS/kdp-serve" uv pip install "vllm==0.31.0" "transformers>=5.16,<5.17"
uv venv "$ENVS/kdp-paddle" --python 3.12
VIRTUAL_ENV="$ENVS/kdp-paddle" uv pip install pip
"$ENVS/kdp-paddle/bin/python" -m pip install -q paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
cd "$ROOT" && VIRTUAL_ENV="$ENVS/kdp-paddle" uv pip install -e ".[test]"
cp "$ROOT/gpu_server/env.gpu.yaml" "$ROOT/configs/env.yaml"                # 서버용 설정(GPU·vLLM 서버)
"$ENVS/kdp-serve/bin/python" -c "import vllm, torch; print('kdp-serve', vllm.__version__, torch.__version__, torch.cuda.is_available())"
"$ENVS/kdp-paddle/bin/python" -c "import paddle, paddleocr, paddlex; print('kdp-paddle', paddle.__version__, paddleocr.__version__, paddlex.__version__, paddle.device.is_compiled_with_cuda())" 2>/dev/null
echo SETUP_DONE
