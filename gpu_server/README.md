# GPU 서버 다시 켜기

클라우드 GPU 서버(A40 46GB, Ubuntu 24.04, Python 3.12)는 비용 때문에 껐다 켠다. 켤 때마다 파이썬 환경과 vLLM 을 새로 깔아야 해서 순서와 옵션을 여기 둔다.

서버 주소·포트는 켤 때마다 바뀔 수 있으므로 파일에 적지 않고 명령 인자로 넘긴다. 접속은 `ssh root@<주소> -p <포트> -i ~/.ssh/id_ed25519` 이다.

## 순서

| 어디서 | 명령 | 하는 일 | 걸린 시간(2026-10-16) |
|---|---|---|---|
| 이 컴퓨터 | `bash gpu_server/push.sh <주소> <포트>` | 코드, 공개 시험 문서(test2·test3), 검출기 가중치, 비교군 결과를 보낸다 | 1분 안 |
| 서버 | `bash /workspace/kor_doc_parser/gpu_server/setup_envs.sh` | 환경 두 개 설치, 서버용 설정(`env.gpu.yaml` → `configs/env.yaml`) | 약 10분(10-18 새 서버 7분) |
| 서버 | `tmux new -d -s vllm "bash /workspace/kor_doc_parser/gpu_server/serve_vlm.sh > /workspace/vllm.log 2>&1"` | vLLM 띄우기. 처음에는 가중치(약 1.8GB)를 받는다 | 약 1분 반 |
| 서버 | `curl -s http://127.0.0.1:8118/v1/models` | 모델 이름이 나오면 준비됐다 | — |
| 서버 | `cd /workspace/kor_doc_parser && export PATH=/workspace/envs/kdp-paddle/bin:$PATH && kdp doctor` | 버전·가중치·VLM 서버 점검(모두 OK) | — |
| 서버 | `kdp parse test2.pdf -o out` · `python eval/compare.py test2.pdf --golden eval/golden/test2_pdf.yaml` | 파싱·비교표 | — |
| 이 컴퓨터 | `bash gpu_server/pull.sh <주소> <포트>` | 서버 `out/` 을 이 컴퓨터 `out_gpu/` 로 가져온다 — 서버를 끄기 전에 | — |

코드는 이 컴퓨터에서 고치고 `push.sh` 로 다시 보낸다(서버 `configs/env.yaml` 은 덮어쓰지 않는다). 서버에서 고친 것은 서버를 끄면 사라진다. 시험·파싱은 이 서버에서만 돌린다(사용자 지시 2026-10-18 — 이 컴퓨터 CPU 로는 너무 오래 걸린다).

2026-10-18 새로 켠 서버(같은 A40 46GB, 드라이버 595·CUDA 13.2)에서도 이 순서 그대로 깔렸다.

## 고정 버전 (기획서 7.4)

| 환경 | 패키지 |
|---|---|
| `kdp-serve`(vLLM 서버) | vllm 0.31.0, torch 2.13.0(CUDA 13.0 빌드 — 드라이버 580 이상), transformers 5.16.1 |
| `kdp-paddle`(파서) | paddlepaddle-gpu 3.3.1(cu126 빌드), paddleocr 3.7.0, paddlex 3.7.2, 이 저장소(`pip install -e ".[test]"`) |

## vLLM 옵션 (`serve_vlm.sh`)

| 옵션 | 값 | 까닭 |
|---|---|---|
| 모델 · `--served-model-name` | `PaddlePaddle/PaddleOCR-VL-1.6` | 파서 설정 `vlm.api_model_name` 과 같아야 한다 |
| `--trust-remote-code` | — | 모델 저장소의 코드를 쓴다 |
| `--trust-request-mm-kwargs` | — | 파서가 블록마다 그림 전처리 옵션(mm_processor_kwargs)을 보낸다. vLLM 0.31 은 기본값으로 막아 요청이 모두 400 으로 거절됐다. 서버를 127.0.0.1 에만 열어 클라이언트가 우리 파서뿐이라 켠다 |
| `--max-num-batched-tokens` | 16384 | PaddleOCR-VL 문서 권장 |
| `--no-enable-prefix-caching`, `--mm-processor-cache-gb 0` | — | 블록마다 그림이 달라 캐시가 쓸모없다(문서 권장) |
| `--gpu-memory-utilization` | 0.5 | A40 46GB 중 약 23GB. 검출기(파들)가 같은 GPU 를 쓴다 |
| `--max-model-len` | 8192 | 블록당 생성 한도 4096 + 그림 토큰이 넘지 않게(설계 5장) |
| `--host 127.0.0.1 --port 8118` | — | 서버 안에서만 연다. 파서 설정 `vlm.server_url` 과 같다 |
| `VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1` | — | 사용 통계 전송을 끈다(설계 9장) |
| `VLLM_USE_FLASHINFER_SAMPLER=0` | — | 아래 '겪은 문제' |

## 겪은 문제

- **vLLM 이 뜨다가 멈춤(2026-10-16)**: flashinfer 의 표본 뽑기 커널을 처음 쓸 때 그 자리에서 컴파일하려다 `ninja` 가 없어 실패했다. 파서는 temperature 0 으로 요청하므로 이 커널이 필요 없어 `VLLM_USE_FLASHINFER_SAMPLER=0` 으로 끈다.
- **모든 요청이 400 으로 거절됨(2026-10-16)**: 'Per-request mm_processor_kwargs are disabled by default' — vLLM 0.31 은 요청마다 보내는 그림 전처리 옵션을 기본으로 막는다. `--trust-request-mm-kwargs` 로 연다(설계 14장에 남겨 둔 확인 거리).
- **시스템 pip 설치 거부**: Ubuntu 24.04 의 시스템 pip 는 패키지 설치를 거부한다(externally-managed). 가상환경은 uv 로 만든다(서버 이미지에 `/usr/bin/uv` 가 있다).

## 지킬 것

- 서버에는 공개 문서만 올린다(`push.sh` 는 PDF 를 이름으로만 보낸다). 사내 문서는 올리지 않는다.
- AI Hub 데이터는 이 서버에 두지 않는다 — 해외 클라우드면 국외 반출이다(기획서 D12).
- 서버를 끄기 전에 `pull.sh` 로 결과를 가져온다.
