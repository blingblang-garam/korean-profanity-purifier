# 변형 욕설까지 잡아내는 문맥 보존형 비속어 순화기

기계학습 미니프로젝트 (NLP 모델 구현 및 개선) — 1차 발표 10/13, **최종 발표 12/1**

욕설이 섞인 문장에서 욕설 **위치**를 찾고, 문맥에 맞는 표현을 골라 **문제 구간만** 바꿉니다. 정상 단어나 밈은 그대로 둡니다. 모든 모델은 교재 『밑바닥부터 시작하는 딥러닝 2』의 NumPy 계층(ch05 TimeRNN, ch06 TimeLSTM)으로 직접 구현했습니다.

```
씨발 또 버그네         →  아 진짜 또 버그네      (감탄으로 쓴 욕 → 감탄사)
오늘 씨발 덥네         →  오늘 정말 덥네         (강조로 쓴 욕 → 부사)
시1발 오늘 존나 덥네    →  아 진짜 오늘 정말 덥네  (변형 욕설, 감탄 + 강조)
짱깨들이 또 시끄럽네    →  중국인들이 또 시끄럽네  (비하어 → 중립 표현)
개나리가 피었다         →  개나리가 피었다         (정상 단어는 그대로)
```

## 구동 파이프라인

```
입력 문장
 → ① 자모 분해 (ㅅㅣㅂㅏㄹ, 종성은 _ㄹ 토큰)
 → ② 탐지기: 양방향 LSTM이 자모마다 욕설 확률 (Sigmoid)
 → ③ 밈·비속어 사전 검색: 탐지기가 놓친 비하어 찾기 / 밈·정상 단어는 보호
 → ④ 용법 분류기: 강조·감탄·지칭·모욕 + 강도 규칙 (접두 욕설·반복·느낌표)
 → ⑤ 후보 생성: 용법별 순화어 + 같은 계열 순화어 (조사 받침 맞춤)
 → ⑥ LSTM 언어 모델: 바꾼 자리 주변의 국소 퍼플렉서티로 가장 자연스러운 후보 선택
 → ⑦ 재검사: 바꾼 문장을 탐지기로 다시 확인
 → 출력
```

단계별 중간값은 [`project/architecture_standalone.html`](project/architecture_standalone.html)(내려받아 브라우저로 열기) 또는 데모 서버의 `/arch`에서 애니메이션으로 볼 수 있습니다.

## 학습한 모델

| 모델 | 구조 (교재 계층) | 학습 데이터 |
|---|---|---|
| 탐지기 | 양방향 LSTM (ch06 TimeLSTM × 2) · 비교용 단방향/양방향 RNN (ch05 TimeRNN) | 깨끗한 실제 댓글에 욕설을 끼워 넣은 문장 약 3.6만 쌍 (함정 문장 2,000 포함) |
| 언어 모델 | LSTM (ch06 Rnnlm) · 비교용 RNN (ch05 SimpleRnnlm) | 깨끗한 댓글 9만 문장 (UnSmile·HateSpeech·K-MHaS 3만 + HateSpeech 비라벨 194만 문장 중 필터 통과 6만) |
| 용법 분류기 | 양방향 LSTM + 욕설 구간 평균 + Softmax | 용법을 정해 끼워 넣은 문장 약 2.2만 |
| 밈·비속어 사전 | 학습 없음 (50항목, 자모 편집 거리 검색) | [`project/lexicon/meme_lexicon.tsv`](project/lexicon/meme_lexicon.tsv) |
| 심판 (평가용) | 글자 1~3-gram 나이브 베이즈 | Curse-detection (평가 문장 제외) |

라벨은 사람이 달지 않았습니다. 깨끗한 문장에 욕설을 끼워 넣어 만들기 때문에 위치·용법 라벨이 자동으로 생깁니다.

## 결과

개발 중 한 번도 보지 않은 실제 욕설 댓글 **195문장**(Curse-detection), 시드 3개 평균입니다. J = 욕설 제거(STA) × 원문 유사도(SIM) × 자연스러움(FL).

| 시스템 | J | 비속어 잔존 | 손으로 쓴 25문장 J | 함정 문장 과교정 |
|---|---|---|---|---|
| 금칙어 사전 (기준선) | 0.243 | 0.308 | 0.175 | 0.80 |
| 1차 방식 (양방향 RNN + 문장 퍼플렉서티) | 0.294 | 0.388 | 0.394 | 0.73 |
| + 학습 데이터 개선 · LSTM · 국소 채점 · 계열 후보 | 0.326 | 0.285 | 0.721 | 0.27 |
| + 밈·비속어 사전 검색 | 0.345 | 0.215 | 0.711 | **0.00** |
| **최종: + 용법·강도** | **0.348** | **0.215** | 0.696 | **0.00** |

- 1차 방식 행은 이전 학습 데이터(v2)로 학습한 모델의 점수입니다. 다른 행은 같은 v3 데이터로 학습했습니다.
- 용법 분류기는 J를 거의 올리지 않습니다. 지금 지표에 "말투·감정이 남았는가"를 재는 항목이 없기 때문입니다. 대신 같은 단어가 위치만 다른 문장(오늘 씨발 덥네 / 씨발 오늘 덥네)의 용법을 0.917 맞힙니다(단어만 보는 방식 0.500).
- 남은 한계: 낯선 변형(쫑나)·뜻이 둘인 말(씨방새), 비꼼·반어, 정상 문장 오탐 12%, 특성 보존을 재는 지표 부재.

## 실행

```bash
git clone https://github.com/blingblang-garam/korean-profanity-purifier.git
cd korean-profanity-purifier/project
python3 -m venv .venv && .venv/bin/pip install numpy matplotlib
```

**데모** — 학습된 모델(`saved/final_v3/`의 `.pkl`)이 있어야 합니다. 모델 파일은 용량 때문에 저장소에 올리지 않았으니, 팀에서 받은 압축 파일을 쓰거나 아래 학습을 먼저 실행합니다.

```bash
.venv/bin/python demo.py          # 브라우저 http://localhost:8765 (시스템별 결과 비교), /arch (파이프라인)
.venv/bin/python demo.py --cli    # 터미널에서 한 줄씩
```

**학습·평가** — 데이터셋을 [project/README.md](project/README.md)의 안내대로 `project/data/`에 받은 뒤 실행합니다. 데이터셋 원본은 라이선스 때문에 이 저장소에 포함하지 않았습니다.

```bash
.venv/bin/python final_pipeline.py --data v3 --seeds 0 1 2   # 탐지기·언어 모델 학습 + 평가 E1~E11 (시드당 약 25분, CPU)
.venv/bin/python usage_model.py                               # 용법 분류기 (약 1.5분)
.venv/bin/python learning_curve.py --seed 0                   # 에포크별 정밀도·재현율 곡선
```

## 코드 구성

이 저장소는 교재 코드([WegraLee/deep-learning-from-scratch-2](https://github.com/WegraLee/deep-learning-from-scratch-2), MIT 라이선스)를 그대로 두고, 우리 코드를 `project/` 폴더에 새로 작성한 것입니다. **교재 원본 파일은 수정하지 않았습니다.**

| 위치 | 내용 |
|---|---|
| `common/`, `ch01`~`ch08/`, `dataset/` | 교재 원본 코드 (원본 README는 `README_원본교재.md`) |
| `project/jamo.py`, `layers_ext.py`, `models.py` | 자모 분해, 길이 마스크를 쓰는 양방향 RNN·LSTM, 탐지기·언어 모델 |
| `project/purifier.py`, `lexicon.py`, `usage_model.py` | 순화기 본체, 밈·비속어 사전, 용법 분류기 |
| `project/data_v3.py`, `final_pipeline.py` | 학습 데이터 생성, 최종 학습·평가 |
| `project/demo.py`, `trace_pipeline.py`, `architecture.html` | 데모 서버, 파이프라인 시각화 |
| `project/saved/final_v3/` | 최종 평가 결과(`final_eval_results_s0_1_2.json`)와 그림 |
| `project/train_demo.py`, `experiments.py`, `real_data.py`, `kcbert_fill.py` | 1차 발표 때 코드 (설명은 [project/README.md](project/README.md)) |

## 참조 코드

| 참조 코드 | 데이터셋 | 쓰임 |
|---|---|---|
| [WegraLee/deep-learning-from-scratch-2](https://github.com/WegraLee/deep-learning-from-scratch-2) | PTB | 구현 기반 (ch04 Embedding, ch05 TimeRNN·SimpleRnnlm, ch06 TimeLSTM·Rnnlm) |
| [2runo/Curse-detection](https://github.com/2runo/Curse-detection) | [Curse-detection-data](https://github.com/2runo/Curse-detection-data) | 자모 분해 아이디어, 욕설 문장 평가 |
| [smilegate-ai/korean_unsmile_dataset](https://github.com/smilegate-ai/korean_unsmile_dataset) | UnSmile | 깨끗한 문장, 심판 분류기 |
| [kocohub/korean-hate-speech](https://github.com/kocohub/korean-hate-speech) | Korean HateSpeech | 깨끗한 문장, 비라벨 댓글 (언어 모델) |
| [adlnlp/K-MHaS](https://github.com/adlnlp/K-MHaS) | K-MHaS | 깨끗한 문장 확장 |
| [boychaboy/KOLD](https://github.com/boychaboy/KOLD) | KOLD | 공격 구간 대비 평가 |
| [minkyeongjeon/kda](https://github.com/minkyeongjeon/kda) | K/DA | 비교 대상 (LLM 순화 연구) |
| [beomi/kcbert-base](https://huggingface.co/beomi/kcbert-base) | - | 1차 E8 비교군 (Mask-and-Fill) |

## 팀

양태규 (인공지능융합공학부) · 송용현 (IoT전자공학과)
