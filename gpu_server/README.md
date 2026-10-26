# GPU 서버 다시 켜기

클라우드 GPU 서버(A40 46GB, Ubuntu 24.04, Python 3.12)는 비용 때문에 껐다 켠다. 켤 때마다 파이썬 환경과 vLLM 을 새로 깔아야 해서 순서와 옵션을 여기 둔다.

서버 주소·포트는 켤 때마다 바뀔 수 있으므로 파일에 적지 않고 명령 인자로 넘긴다. 접속은 `ssh root@<주소> -p <포트> -i ~/.ssh/id_ed25519` 이다.

## 순서

| 어디서 | 명령 | 하는 일 | 걸린 시간(2026-10-16) |
|---|---|---|---|
| 이 컴퓨터 | `bash gpu_server/push.sh <주소> <포트>` | 코드, 공개 시험 문서(test2·test3), 검출기 가중치, 비교군 결과를 보낸다 | 1분 안 |
| 서버 | `bash /workspace/kor_doc_parser/gpu_server/setup_envs.sh` | 환경 두 개 설치, 서버용 설정(`env.gpu.yaml` → `configs/env.yaml`) | 약 10분(10-18 새 서버 7분) |
| 서버 | `tmux new -d -s vllm "bash /workspace/kor_doc_parser/gpu_server/serve_vlm.sh > /workspace/vllm.log 2>&1"` | vLLM 띄우기. 처음에는 가중치(약 1.8GB)를 받는다. 몫은 기본 0.25(그림 설명 서버와 함께 — 설명을 끄고 혼자 쓰면 `UTIL=0.5` 도 된다) | 약 1분 반 |
| 서버 | `curl -s http://127.0.0.1:8118/v1/models` | 모델 이름이 나오면 준비됐다 | — |
| 서버 | `tmux new -d -s desc "/workspace/envs/kdp-serve/bin/python /workspace/kor_doc_parser/gpu_server/serve_hf_vlm.py > /workspace/desc.log 2>&1"` | 그림 설명 VLM(D19) 띄우기 — 파서 설정 `describe.enabled` 가 기본으로 켜져 있어 `kdp parse` 전에 띄운다(끄면 필요 없다). 국내 모델(사용자 요청): A.X-4.0-VL-Light(SK텔레콤)는 vLLM 에 없는 구조라 이 transformers 서버로, DNA3.0(디노티시아)은 `serve_describer.sh`(vLLM)로 띄운다. 둘 다 127.0.0.1:8119 라 한 번에 하나만. 처음에는 가중치(A.X 약 15GB·DNA3.0-4B 약 10GB)를 받는다. 준비 확인은 `curl -s http://127.0.0.1:8119/v1/models` | — |
| 서버 | `cd /workspace/kor_doc_parser && export PATH=/workspace/envs/kdp-paddle/bin:$PATH && kdp doctor` | 버전·가중치·VLM 서버 점검(모두 OK) | — |
| 서버 | `kdp parse test2.pdf -o out` · `python eval/compare.py test2.pdf --golden eval/golden/test2_pdf.yaml` | 파싱·비교표 | — |
| 서버 | `kdp describe out` | 그림 설명이 빠진 그림만 설명 VLM 에 보내고 다시 조립(설명 모델을 바꿨거나 설명 서버 없이 파싱한 문서) | — |
| 서버 | `tmux new -d -s t1 "bash /workspace/kor_doc_parser/gpu_server/tune_t1.sh"` | 그림 해석 튜닝 1차 시범(D20 — 기획서 7.10) 한 번에: 원래 A.X 채점 물음·test2~5 파싱 → 50 step 시험 → 본 학습 → 튜닝 A.X 같은 물음·파싱 → 채점. 서버 둘은 스크립트가 띄우고 내리므로 먼저 띄우지 않는다. 진행은 `/workspace/t1/status.txt` | — |
| 이 컴퓨터 | `bash gpu_server/pull.sh <주소> <포트>` | 서버 `out/` 을 이 컴퓨터 `out_gpu/` 로 가져온다 — 서버를 끄기 전에 | — |

### RAG 하류 평가(D21 — 기획서 8.6 축소판, LightRAG)

| 어디서 | 명령 | 하는 일 |
|---|---|---|
| 이 컴퓨터 | `bash gpu_server/push.sh <주소> <포트>` | 위와 같다 — `runs/t1/out_base`(test2~5 원래 A.X 파싱, raw 포함)도 보낸다 |
| 서버 | `bash /workspace/kor_doc_parser/gpu_server/setup_envs.sh` | 위와 같다 |
| 서버 | `bash /workspace/kor_doc_parser/gpu_server/setup_eval.sh` | 환경 둘 더: `kdp-rag`(lightrag-hku 1.5.7 — 벡터 저장소 nano-vectordb 함께·tiktoken·BGE-M3 용 torch·transformers 4.57), `kdp-docling`(비교군 docling 2.138.0, CPU) |
| 서버 | `tmux new -d -s rag "bash /workspace/kor_doc_parser/gpu_server/rag_eval.sh"` | 단위 시험 → 비교군(텍스트 레이어·PaddleOCR-VL 기본·docling) → kdp 다시 조립(raw/ 만 두고 새로 쓴다 — document.rag.md 를 더 쓰고, document.json·document.md·blocks.jsonl 이 원래 출력과 같은지 `cmp`) → 같은 자르기·같은 검색기 평가(LightRAG R 512·F 512·F 1,200토큰·구조 자르기 × BM25·BGE-M3) → LightRAG 저장소 평가(기본은 kdp document.rag.md 하나를 512토큰 R 로 × naive·mix — 1회차의 넷은 `테스트_결과.md` 14.5). PaddleOCR-VL·LLM 서버는 스크립트가 띄우고 내리므로 먼저 띄우지 않는다. 진행은 `/workspace/rag_eval/status.txt` |
| 이 컴퓨터 | `bash gpu_server/pull.sh <주소> <포트>` | `runs/rag_eval/` 로 보고(`report/`·`lightrag_report/`)·비교군 출력·다시 조립한 마크다운을 가져온다(raw·LightRAG 저장소는 두고 온다) |

- 2회차(`테스트_결과.md` 14.8): `rag_eval.sh` 를 한 번 돌린 서버에서 `tmux new -d -s rag2 "bash …/gpu_server/rag_eval2.sh"` — kdp 를 표 HTML·파이프 표 두 벌로 다시 조립해 검색 평가(LightRAG F·R 512토큰·구조 자르기). LightRAG 저장소는 고른 실행 하나만 `lightrag_one.sh <이름> --runs kdp_rag_pipe:512:R --variant kdp_rag_pipe=/workspace/rag_eval/kdp_pipe`(LLM 서버를 띄우고 내린다). R 전략에는 `kdp-rag` 에 langchain-text-splitters 가 있어야 한다(setup_eval.sh).
- 기본값을 바꾼 뒤 점검(`테스트_결과.md` 14.9): `tmux new -d -s rag3 "bash …/gpu_server/rag_eval3.sh"` — 단위 시험 → 기본 설정으로 다시 조립(`/workspace/rag_eval/kdp` — 정본·document.md·blocks.jsonl 은 원래 출력과, document.rag.md 는 2회차 파이프 표 판과 같은지 `cmp`) → 검색 평가(기본 자르기). 저장소는 만들지 않는다. 기본 저장소를 새로 만들 때는 `lightrag_one.sh <이름>`(인자 없이 — kdp document.rag.md 하나를 512토큰 R 로).
- LightRAG 의 LLM 은 `serve_llm.sh`(vLLM, 127.0.0.1:8120) — 기본 국내 모델 `skt/A.X-4.0-Light`(7.3B, Apache-2.0, 문맥 16k). 바꾸려면 `MODEL=Qwen/Qwen3-14B UTIL=0.8`(rag_eval.sh 안의 serve_llm.sh 줄)과 `eval/lightrag_eval.py --llm-model` 을 같은 이름으로. LightRAG 는 temperature·max_tokens 를 보내지 않아(그대로면 vLLM 이 1.0 으로 뽑는다) 평가 스크립트가 0·2,048 로 건다. 7B 에 맞춘 설정: 청크 512토큰(사용자 결정), 넣기 R 전략(사용자 결정 2026-10-26), 글리닝 1, 한국어 엔티티 종류 안내(사람 이름은 뽑지 않음), 추출 언어 Korean.
- LightRAG 벡터 저장소는 NanoVectorDB(`NanoVectorDBStorage`, LightRAG 기본·MIT — numpy 로 모두 견주는 정확 검색, 사용자 결정 2026-10-26). 그래프는 NetworkX, 나머지는 JSON — 모두 `/workspace/rag_eval/lightrag/<계통>-<토큰>-<R|F>/` 파일이고 실행마다 지우고 새로 만든다(옛 글이 남지 않게). 다시 돌려도 같은 값이 나오는 것은 naive 와 `rag_eval.py` 다 — mix 는 LLM 이 추출한 그래프라 temperature 0 이어도 동시 요청 묶음에 따라 조금 달라질 수 있다.
- 임베딩은 BGE-M3(MIT)를 평가 프로세스 안에서 쓴다(처음에 약 2.3GB 를 받는다). LightRAG 자르기의 tiktoken 은 처음 쓸 때 o200k_base 표를 받는다(setup_eval.sh 가 미리 받는다).

코드는 이 컴퓨터에서 고치고 `push.sh` 로 다시 보낸다(서버 `configs/env.yaml` 은 덮어쓰지 않는다). 서버에서 고친 것은 서버를 끄면 사라진다. 시험·파싱은 이 서버에서만 돌린다(사용자 지시 2026-10-18 — 이 컴퓨터 CPU 로는 너무 오래 걸린다).

2026-10-18 새로 켠 서버(같은 A40 46GB, 드라이버 595·CUDA 13.2)에서도 이 순서 그대로 깔렸다.

## 고정 버전 (기획서 7.4)

| 환경 | 패키지 |
|---|---|
| `kdp-serve`(vLLM 서버) | vllm 0.31.0, torch 2.13.0, transformers 5.16.1 — 드라이버 580 이상이면 PyPI 판(CUDA 13.0 빌드), 미만이면 vLLM 의 CUDA 12.9 판(GitHub 배포 `+cu129`)·torch `+cu129`·torchcodec 0.17.0 CPU 판(setup_envs.sh 가 드라이버를 보고 고른다) |
| `kdp-paddle`(파서) | paddlepaddle-gpu 3.3.1(cu126 빌드), paddleocr 3.7.0, paddlex 3.7.2, 이 저장소(`pip install -e ".[test]"`) |
| `kdp-rag`(RAG 하류 평가, D21·D22) | lightrag-hku 1.5.7(nano-vectordb 함께), torch 2.13.0(드라이버 기준은 kdp-serve 와 같다), transformers 4.57.x(BGE-M3 — kdp-serve 의 5.16 과 다르다), openai 2~3.x 라이브러리, tiktoken, langchain-text-splitters(R 전략) — setup_eval.sh |
| `kdp-docling`(비교군) | docling 2.138.0(CPU torch), pypdfium2 5.14.0, 이 저장소(`--no-deps`) — setup_eval.sh |

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

그림 설명 VLM(D19)은 국내 모델만 쓴다(사용자 요청 2026-10-19). A.X-4.0-VL-Light(7.7B, Apache-2.0)는 vLLM 0.31 에 없는 구조라 `serve_hf_vlm.py`(transformers, bf16, 요청 하나씩)로 띄운다 — 저장소 코드(trust_remote_code)가 transformers 4.49 기준이라 5.x 에서 빠진 내부 함수 하나를 채우고, 설정의 flash_attention_2 는 sdpa 로 바꿔 띄운다(서버에 flash-attn 없음). DNA3.0-4B·2B(Apache-2.0, Qwen3.5 구조)는 `serve_describer.sh` 로 vLLM 에 띄운다 — `--gpu-memory-utilization 0.35`, `--max-model-len 8192`, `--limit-mm-per-prompt '{"image":1,"video":0}'`(파서는 그림 한 장만 보낸다), `127.0.0.1:8119`. 모델을 바꿀 때는 `tmux kill-session -t desc` 로 앞 서버를 먼저 끄고, 변수는 따옴표 안에 넣는다(`tmux new -d -s desc "MODEL=dnotitia/DNA3.0-2B bash …/serve_describer.sh > /workspace/desc.log 2>&1"` — 이미 떠 있는 tmux 서버는 밖의 변수를 새 세션에 넘기지 않는다). 파서 설정 `describe.model` 도 같은 이름으로 두고 `kdp doctor` 로 확인한다. 파서는 요청마다 생각 모드를 끈다(`enable_thinking: false` — DNA3.0 은 생각 모드가 기본). 파서는 요청마다 토큰 확률(`logprobs`)도 받는다 — 확신이 낮은 설명 문장을 빼는 데 쓴다(`describe.min_logprob`, A.X 로 잰 -0.30 — 모델을 바꾸면 다시 잰다). `serve_hf_vlm.py` 는 `logprobs`·`top_logprobs` 를 받으면 되풀이 벌점 전 값으로 돌려준다. DNA3.0-9B(9.4B, Apache-2.0)도 재 봤다(설명·검증 모두 A.X 와 확신도 거르기보다 낫지 않아 쓰지 않음 — 테스트_결과.md 10장). 띄우려면 `UTIL=0.55 MODEL=dnotitia/DNA3.0-9B` — Qwen3.5 구조는 동시 요청마다 순환형 층의 상태 칸이 들어 기본 동시 수 256 이 몫 0.55 에 들지 않으므로 `serve_describer.sh` 가 `--max-num-seqs 32`(`MAX_SEQS`)로 띄운다. 포트는 `PORT`(기본 8119). 메모리(A40 46GB 에서 잼, 2026-10-19 — 테스트_결과.md 9.5): 파싱만 할 때 PaddleOCR-VL 서버 몫 0.5 는 23.0GB, 검출기(파서 프로세스) 3.6GB 로 최대 26.6GB 다. 몫을 0.25(11.6GB)로 줄여도 test3 파싱 시간이 같았다(36.9 → 37.4초). 몫 0.25 에 A.X 설명 서버(설명 중 16.2GB)를 함께 띄우고 설명을 켜 파싱하면 GPU 최대 31.1GB 로 15GB 가 남는다 — 그래서 `serve_vlm.sh` 의 몫 기본값을 0.25 로 두었다(그림 설명이 기본으로 켜져 있다 — 설명을 끄고 혼자 쓸 때는 `UTIL=0.5` 도 된다, 변수는 tmux 따옴표 안에)(vLLM 은 몫을 '전체 × 비율'로 잡고 뜰 때 남은 메모리가 모자라면 멈춘다). 두 서버를 다 띄운 뒤 `kdp parse` 를 돌린다. 나눠 돌리려면 설명을 끈 채 파싱을 다 하고 PaddleOCR-VL 서버를 내린 뒤 설명 서버만 띄워 `kdp describe out` 로 몰아서 부른다.

## 겪은 문제

- **vLLM 이 뜨다가 멈춤(2026-10-16)**: flashinfer 의 표본 뽑기 커널을 처음 쓸 때 그 자리에서 컴파일하려다 `ninja` 가 없어 실패했다. 파서는 temperature 0 으로 요청하므로 이 커널이 필요 없어 `VLLM_USE_FLASHINFER_SAMPLER=0` 으로 끈다.
- **모든 요청이 400 으로 거절됨(2026-10-16)**: 'Per-request mm_processor_kwargs are disabled by default' — vLLM 0.31 은 요청마다 보내는 그림 전처리 옵션을 기본으로 막는다. `--trust-request-mm-kwargs` 로 연다(설계 14장에 남겨 둔 확인 거리).
- **시스템 pip 설치 거부**: Ubuntu 24.04 의 시스템 pip 는 패키지 설치를 거부한다(externally-managed). 가상환경은 uv 로 만든다(서버 이미지에 `/usr/bin/uv` 가 있다).
- **드라이버 570 서버(2026-10-19)**: PyPI 의 vLLM 0.31.0·torch 2.13.0 은 CUDA 13 빌드라 드라이버 580 이상이 필요하다. 이 서버는 570(CUDA 12.8)이라 vLLM 의 CUDA 12.9 판(`vllm-0.31.0+cu129` 바퀴 파일)과 `uv pip install --torch-backend cu129` 로 torch `+cu129` 를 깔았다(CUDA 12.x 는 같은 대 버전 안에서 낮은 드라이버로도 돈다 — `torch.cuda.is_available()` True). 그래도 vLLM 이 뜨다 'libnvrtc.so.13 을 열 수 없다'며 멈췄다 — PyPI 의 torchcodec 0.17.0 이 CUDA 13 빌드다. 영상·소리를 쓰지 않으므로 CPU 판(`torchcodec==0.17.0+cpu`)으로 바꿔 풀었다. setup_envs.sh 가 드라이버를 보고 둘 다 한다.

## 지킬 것

- 서버에는 공개 문서만 올린다(`push.sh` 는 PDF 를 이름으로만 보낸다). 사내 문서는 올리지 않는다.
- AI Hub 데이터는 이 서버에 두지 않는다 — 해외 클라우드면 국외 반출이다(기획서 D12).
- 서버를 끄기 전에 `pull.sh` 로 결과를 가져온다.
