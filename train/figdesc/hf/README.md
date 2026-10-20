---
license: apache-2.0
base_model: skt/A.X-4.0-VL-Light
library_name: peft
pipeline_tag: image-text-to-text
language:
- ko
tags:
- lora
- peft
- vision-language
- chart
- map
- flowchart
- document-understanding
---

# A.X-kor-image-caption

**SK텔레콤의 공식 모델이 아닙니다.** SK텔레콤이 Apache-2.0 으로 공개한 [skt/A.X-4.0-VL-Light](https://huggingface.co/skt/A.X-4.0-VL-Light) 에 얹어 쓰는 LoRA 어댑터입니다. 개인이 한국어 PDF 파서([kor_doc_parser](https://github.com/ohdyo/kor_doc_parser))의 그림 설명 단계를 위해 튜닝했고, SK텔레콤과는 관계가 없습니다.

> English: An unofficial LoRA adapter for skt/A.X-4.0-VL-Light (Apache-2.0), fine-tuned to answer short interpretive questions about figures in Korean public documents: charts (largest/smallest item, overall trend), choropleth/contour maps (direction of the highest/lowest area), flowcharts (first/last/next step) and org charts (top box, parent box). Not affiliated with SK Telecom.

## 무엇을 하나

원래 A.X 는 그림 속 글자·숫자는 잘 읽지만, '어디가 가장 높은지·무엇이 가장 큰지·흐름이 어떤지' 같은 해석을 자주 틀립니다(아래 평가의 원래 A.X). 이 어댑터는 그 해석을 아래 물음 꼴로 가르쳤습니다.

| 그림 | 물음 갈래 | 답 꼴 |
|---|---|---|
| 그래프 | `chart_high`·`chart_low` — 값이 가장 큰·작은 때나 항목 | 그림에 적힌 이름 그대로(단답) |
| 그래프 | `chart_trend` — 전체 흐름 | 보기 5개(늘어남·줄어듦·늘다가 줄어듦·줄다가 늘어남·뚜렷한 흐름 없음) 가운데 번호 |
| 지도 | `map_high`·`map_low` — 값이 가장 높은·낮은 곳 | 지도 테두리 상자를 3×3 으로 나눈 9방위 보기 가운데 번호 |
| 흐름도·절차도 | `flow_first`·`flow_last`·`flow_next` — 처음·마지막 단계, 'X' 바로 다음 단계 | 상자 글 그대로 |
| 조직도·체계도 | `org_top`·`org_parent` — 맨 위, 'X' 바로 위 | 상자 글 그대로 |

이름에 caption 이 들어가지만 긴 그림 설명(캡션)을 쓰도록 배운 것은 아닙니다. 위 물음에 짧게 답하도록만 배웠고, 파서에서는 이 답으로 그림 설명의 해석을 채울 계획입니다.

## 쓰는 법

이 저장소의 `inference.py` 에 물음 틀·불러오기·생성이 다 있습니다. A.X 원격 코드는 transformers 4.49 기준이라 5.x 에서 돌리려면 보정이 셋 필요한데, `load()` 가 해 줍니다.

```bash
pip install "torch==2.13.*" "transformers==5.16.1" "peft==0.20.0" pillow
hf download ohdyo/A.X-kor-image-caption inference.py --local-dir .
python inference.py 그래프.png chart_high --caption "연도별 온열질환자 수" --text "2021 / 2022 / 2023 / 2024 / (명)"
python inference.py 흐름도.png flow_next --x "서류 접수"
```

```python
from PIL import Image
from inference import answer, load, prompt

proc, model = load()        # skt/A.X-4.0-VL-Light + 이 LoRA(합치지 않고 얹는다), bf16, GPU 메모리 약 16GB
q = prompt("map_high", caption="9월 평균기온 분포", text="23 / 24 / 25 / ℃")
print(answer(proc, model, Image.open("지도.png").convert("RGB"), q, max_new_tokens=8))   # 보기 번호(보기는 prompt 가 붙인 순서)
```

물음 글은 아래 꼴입니다. 학습·평가와 글자까지 같아야 결과도 같습니다. '그림에서 읽은 글자' 는 그림의 OCR 결과를 ' / ' 로 이은 것이고, 없으면 '(없음)' 입니다.

```text
아래 그림을 보고 물음에 답하세요.
그림 제목: 연도별 온열질환자 수
그림에서 읽은 글자(틀린 글자가 있을 수 있습니다): 2021 / 2022 / 2023 / 2024 / (명)

물음: 이 그래프에서 값이 가장 큰 때나 항목은 무엇인가요?
그림에 적힌 이름(가로축 글자·항목 이름) 그대로 짧게 답하세요. 막대나 선이 여럿이면 그림 제목이 가리키는 주된 값으로 봅니다.
```

- 확인한 판: torch 2.13 · transformers 5.16.1 · peft 0.20.0, NVIDIA A40.
- 생성: 탐욕 생성(do_sample=False), repetition_penalty 1.05(원래 모델 카드 권장값), 보기 물음은 최대 8토큰·단답은 24토큰.
- 보기 물음은 보기 순서를 바꿔 두 번 묻고 두 답이 같을 때만 믿는 것을 권합니다(아래 평가 방식).

## 학습 자료

| 갈래 | 그림 | 학습 물음 | 정답을 매긴 방법 |
|---|---|---|---|
| 합성 그래프·지도·흐름도·조직도 | 2,850장(검증 150장은 따로) | 8,103(검증 429) | 그림을 그린 코드가 계산(matplotlib·numpy — 지명·조직 이름은 지어냈고, 항목 이름은 연령대·장소·질환 같은 실제 분류 이름) |
| 실제 공개 문서 그림 | 268장(문서 25개) | 605 — 학습 때 두 번 섞음 | Claude(Anthropic, Opus) 라벨러 둘이 서로의 답을 보지 않고 따로 매겨, 둘이 같게 본 정답만 |

- 합성: 그래프 1,000장(막대·묶음 막대·선·막대+선 두 축·평년선, 가로축은 해·월·순·주·날짜·항목), 지도 1,000장(단계구분도·등치선 분포도, 9방위가 고르게 나오게), 흐름도 500장(가로·세로·줄 바꿈·예/아니오 갈림), 조직도 500장. 이미 잘하는 그림 글자 읽기를 잊지 않도록 읽기 물음(막대 위 가장 큰 값·가로축 첫 이름·범례 단위)을 약 9% 섞었습니다.
- 실제: 정책브리핑(korea.kr)의 정부 보도자료 27개(공공누리 제1유형, 아래 출처)를 파서로 잘라 낸 그림입니다. 로고처럼 뜻 없는 그림은 뺐고, 담당자 이름·전화번호는 라벨 단계부터 가렸습니다. 물음 갈래별 수: chart_high 216 · chart_low 215 · chart_trend 132 · flow_first 11 · flow_last 11 · map_low 6 · map_high 5 · org_top 5 · org_parent 2 · flow_next 2.
- 평가 문서(아래 평가)와 그 연속 보도는 학습 자료에 넣지 않았습니다.

## 학습 설정

| 항목 | 값 |
|---|---|
| 시작 가중치 | skt/A.X-4.0-VL-Light(7.7B — SigLIP 그림 인코더 + Qwen2 계열 28층 언어 모델) |
| LoRA | 언어 모델의 q·k·v·o·gate·up·down, rank 16 · α 32 · dropout 0.05, 학습 무게 40.4M. 그림 인코더·연결부는 얼림 |
| 손실 | 답 토큰과 닫는 표시(`<\|im_end\|>`)만 — 물음·그림 토큰은 가림 |
| 최적화 | AdamW lr 1e-4, warmup 3%, cosine, 배치 1 × 누적 8, 1 에폭(9,313개 · 1,164 step), bf16 autocast(LoRA 무게는 fp32), gradient checkpointing, 시드 0 |
| 해상도 | 추론과 같게(384px 타일 최대 12장 + 썸네일) |
| 자원 | NVIDIA A40 46GB 한 장, 약 2시간 54분(1,164 step · step 당 약 9초), GPU 메모리 최대 20.2GB |

학습 설정(`train_config.yaml`), 손실 기록(`train_log.jsonl`), 합성 검증 결과(`synth_val.json`)를 함께 올렸습니다.

## 평가

실제 공개 정부 문서의 그림에 같은 물음을 원래 A.X 와 이 어댑터에 묻고 채점했습니다. 같은 그림 크롭·같은 물음 글·탐욕 생성이고, 정답은 평가자 둘(Claude 에이전트)이 따로 매겨 둘 다 맞다고 본 것만 씁니다. 정답률은 보기 순서 A 의 답으로 매깁니다. 보기 물음은 보기 순서를 거꾸로 해 한 번 더 묻고, 두 답이 같을 때만 확신 문턱 판정에서 '답함'으로 칩니다. 평가 문서는 학습 자료에 넣지 않았습니다.

| 물음 묶음 | 물음 | 원래 A.X | 이 어댑터 |
|---|---|---|---|
| 개발 — 공개 문서 2개 | 25 | 20% | **40%** |
| 확인 — 공개 문서 2개 | 42 | 43% | **60%** |
| 새 시험지 — 공개 문서 6개(기관 6곳) | 140 | 62% | **81%** |
| 합성 검증 — 학습에 쓰지 않은 합성 그림 | 200 | — | **97.5%** |

| 물음 갈래(실제 문서 세 묶음 합계) | 원래 A.X | 이 어댑터 |
|---|---|---|
| 그래프 — 가장 큰·작은 것, 전체 흐름 | 96/167 (57%) | **135/167 (81%)** |
| 지도 — 가장 높은·낮은 곳의 방위 | 6/31 (19%) | 6/31 (19%) |
| 흐름도·조직도 | 8/9 | 8/9 |

- **그래프 해석이 크게 나아졌습니다.** 새 시험지의 '가장 큰·작은 것' 은 61/98 → 83/98, 전체 흐름은 15/22 → 20/22 입니다.
- **지도 방위는 나아지지 않았습니다.** 합성 지도 검증은 95% 맞혔지만 실제 기상·통계 지도로 옮겨 가지 않았습니다. 원래 A.X 의 '마지막 보기 고르기' 쏠림은 사라졌지만 맞히지는 못하므로, 지도 답은 쓰지 않는 것을 권합니다.
- 답의 확신(답 토큰 평균 로그확률)으로 맞는 답과 틀린 답을 가르는 힘(AUC)이 확인 묶음에서 0.70 → 0.91 로 좋아졌습니다. 확신 -0.05 이상인 답만 쓰면 확인 묶음 물음의 45% 에 답해 95% 를 맞혔습니다(새 시험지는 -0.02 이상에서 67% 답함·91% 정답).
- **미리 정한 통과 기준은 넘지 못했습니다** — 확신 문턱 위 정답률 90% 이상이면서 물음의 40% 이상에 답하되, 문턱은 개발 묶음에서만 고릅니다. 개발 묶음(25물음)은 어느 문턱에서도 정답률이 57% 를 넘지 않았습니다(지도 물음을 빼고 그래프 16물음만 봐도 최고 67%).
- **잊음 검사** — 이 어댑터로 파서의 그림 설명 문장을 쓰게 하고(공개 문서 4개), 문장마다 그림과 대조해 판정했습니다. 틀린 문장은 원래 A.X 8/58(14%) → 이 어댑터 5/54(9%)로 늘지 않았습니다(차이는 3문장이고, 두 쪽의 판정자가 달라 판정 차이가 섞여 있습니다). 다만 맞는 문장은 47 → 40, 그림을 봐야 아는 정답 사실은 7/90 → 8/90 으로, 물음 꼴로 배운 해석이 긴 설명 문장으로 저절로 옮겨 가지는 않았습니다.

## 한계

- 위 물음 꼴에 맞춰 배웠습니다. 다른 물음 꼴이나 긴 설명에서도 나아지는지는 재지 않았습니다.
- 실제 그림 정답도 사람이 아니라 Claude 라벨러 둘이 매긴 것입니다. 둘이 같게 본 것만 썼지만, 같은 모델의 버릇이 섞여 있을 수 있습니다.
- 실제 학습 그림은 막대·선 그래프가 대부분이고 지도·흐름도·조직도는 수십 개뿐입니다. 여러 계열이 겹친 그래프나 점 분포 지도처럼 물음이 맞지 않는 그림은 실제 자료에서 빠졌습니다.
- 지도 방위는 실제 지도에서 나아지지 않았고(위 평가), 답도 3×3 으로 나눈 거친 방위입니다.
- 평가 물음이 수백 개라 정답률 몇 %p 차이는 오차 안에 듭니다.

## 라이선스와 출처

- 어댑터: Apache-2.0(시작 가중치 skt/A.X-4.0-VL-Light 의 라이선스를 따릅니다).
- 실제 그림 학습 자료는 아래 정부 보도자료(공공누리 제1유형 — 출처표시)에서 왔습니다. 정책브리핑 페이지의 표시에는 '텍스트에 한하여'라는 단서가 있어, 그림의 이용 조건은 각 저작권자 기준을 따릅니다. 그림 파일은 이 저장소에 올리지 않았습니다.
- 실제 그림의 학습 정답은 Claude(Anthropic) 라벨러 둘이 매긴 것이라 그 서비스 약관을 따릅니다.

| | 기관 | 제목 | 날짜 |
|---|---|---|---|
| 1 | 기상청 | [폭염은 빨라지고 열대야는 늦게까지 지속, 발생 빈도 증가와 함께 발생 기간도 확대되고 있다](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156772054) | 2026-07-24 |
| 2 | 행정안전부 | [2026년 행정안전통계연보 발간](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156775400) | 2026-08-25 |
| 3 | 성평등가족부 | [경력단절 경험한 여성은 10명 중 6명 (「2025년 여성의 경제활동 및 경력단절 실태조사」 결과 발표)](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156770250) | 2026-07-09 |
| 4 | 국토교통부 | [전국 건축물 총 7,438,642동 / 43억 75백만㎡](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156783582) | 2026-09-30 |
| 5 | 국가데이터처 | [「청년 삶의 질 2025」 보고서 발간](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156735299) | 2025-12-16 |
| 6 | 보건복지부 | [2025년 외국인 환자 유치 200만 돌파 아시아 의료관광 '중심국가' 도약](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156757630) | 2026-04-24 |
| 7 | 기후에너지환경부 | [2030년 재생에너지 100GW 이상 수용하도록 전력망 체계 혁신한다](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156783472) | 2026-09-29 |
| 8 | 기후에너지환경부 | [가습기살균제, 국가 주도 배상체계로 전환](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156736728) | 2025-12-24 |
| 9 | 농림축산식품부 | [농촌을 잇고 지역에 활력을 더하다, 농촌관광벨트 시범모델 개발!](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156770336) | 2026-07-10 |
| 10 | 농림축산식품부 | [농업인 안전재해 최소화를 위한 '농림분야 안전관리 종합대책' 발표](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156765239) | 2026-06-05 |
| 11 | 국토교통부 | [K-UAM '28년 상용화 UAM팀코리아가 실현하겠습니다.](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156713208) | 2025-08-28 |
| 12 | 문화체육관광부 | [2026년 우리 사회 문화 흐름은 '회복에서 적응으로'](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156737589) | 2025-12-31 |
| 13 | 국토교통부 | [도심 내 우수입지의 유휴부지 등을 활용한 6만호 신속 공급](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156741964) | 2026-01-29 |
| 14 | 국토교통부 | [지역주택조합 '희망고문' 멈춘다… "조합원 피해 예방 및 사업 정상화 방안 발표"](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156756665) | — |
| 15 | 고용노동부 | [2026년 고용노동부 업무보고 보도자료](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156734661) | — |
| 16 | 고용노동부 | ["함께 도약하는 노동 있는 산업대전환" - 국민주권정부, 첫번째 「산업전환 고용안정 기본계획」 수립 -](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156770200) | 2026-07-09 |
| 17 | 고용노동부 | [2025년 12월 고용행정 통계로 본 노동시장 동향](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156739308) | 2026-01-12 |
| 18 | 산업통상부 | [2025년 4월 수출입 동향](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156687279) | 2025-05-01 |
| 19 | 산업통상부 | [2026년 8월 수출입동향](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156776348) | 2026-09-01 |
| 20 | 산업통상부 | ['30년 유망소비재 700억불 수출 목표로, 「K-소비재 수출 확대 방안」 마련](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156736681) | 2025-12-24 |
| 21 | 문화체육관광부 | ['케이-컬처', 음식·미용(뷰티)에 이어 캐릭터·공연 등 융합 콘텐츠로 지평 확장](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156751579) | 2026-03-30 |
| 22 | 문화체육관광부 | [대한민국 문화정책 대전환 '문화한국 2035' 발표](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156677526) | — |
| 23 | 문화체육관광부 | ['케이-관광, 세계를 품다', 외래객 3천만 명 시대 앞당긴다](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156745975) | 2026-02-25 |
| 24 | 중소벤처기업부 | [2024년 연간 창업기업동향](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156676794) | — |
| 25 | 해양수산부 | [북극항로 시대로의 대도약, 민생경제 활력, 대한민국 균형성장 실현](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156736548) | 2025-12-23 |
| 26 | 농림축산식품부 | [국민주권정부 출범 6개월, 모두가 함께 만드는 '행복한 농업 ･ 농촌'](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156732946) | — |
| 27 | — | [청년의 잠재력에 국가가 투자합니다! 청년 예산 언박싱(UNBOXING) 2027, 「청년 성장단계별 종합투자 추진전략」 발표](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156775927) | — |
