# FICR 기대효용 디코더 v3

이 코드는 기존 발전량 예측 모델을 새로 학습하는 것이 아니라, `model_2` 같은 기존 Test 예측값을 받아 **FICR 정산에 더 유리한 방향으로 예측값을 미세 조정하는 후처리 모델**입니다.

[[해당 디코더는 은호님의 model_2의 전체 validation_prediction이 존재하지 않아 제 개인 모델의 val을 썼습니다!
혹시 본인 모델로 실험해보고 싶으신 분은 자신의 val과 test submission 파일을 하단 실행 입력 칸에 설명대로 입력하신 후 사용하시면 됩니다
python baram_qrf3.py `
     --preprocessed-dir [전처리 폴더 이름] ` 
     --base-validation [val 폴더이름/파일명] `
     --postprocess-check [후처리 진행하였다면 후처리 폴더이름/파일명] ` 
     --base-test [test submission 폴더이름/파일명] `
     --decoder ens ` --allow-marginal ` --output-dir out_v3]]

## 전체 흐름

```text
기존 Test 예측값
   ↓
전처리 NWP로 FICR 기대효용 학습
   ↓
기존 예측값 주변에서 더 유리한 값 탐색
   ↓
2024 Validation에서 보정 효과 검증
   ↓
안정적으로 좋아진 Group만 적용
   ↓
최종 submission.csv
```

## 1. FICR을 직접 고려한 예측 보정

일반적인 회귀 모델은 실제 발전량과의 평균적인 오차를 줄이는 방향으로 예측합니다.

하지만 FICR은 오차에 따라 정산금이 계단식으로 결정됩니다.

```text
오차 ≤ 6%       → 4원/kWh
6% < 오차 ≤ 8% → 3원/kWh
오차 > 8%       → 정산금 없음
```

따라서 이 코드는 단순히 가장 평균적인 발전량을 찾는 대신,

```text
"어떤 값을 제출했을 때 기대 정산금이 가장 큰가?"
```

를 기준으로 예측값을 조정합니다.

---

## 2. QRF / Direct / ENS 디코더 추가

FICR 기대효용을 계산하는 방법은 세 가지입니다.

### QRF

Random Forest에서 현재 기상조건과 비슷한 과거 데이터의 실제 발전량 분포를 이용합니다.

```text
현재 NWP
   ↓
비슷한 과거 발전량 분포
   ↓
후보 예측값별 예상 정산금 계산
```

### Direct

발전량 분포를 먼저 구하지 않고 LightGBM으로 후보 예측값별 기대 정산금을 직접 학습합니다.

```text
현재 NWP
   ↓
"0.60을 제출하면 기대 정산금은?"
"0.65를 제출하면?"
...
```

### ENS

QRF와 Direct의 기대효용을 각각 정규화한 뒤 50:50으로 합칩니다.

```text
QRF 효용 50%
+
Direct 효용 50%
```

두 모델이 추천한 예측값을 단순 평균하는 것이 아니라, **후보값별 효용을 합친 뒤 가장 높은 값을 선택**합니다.

---

## 3. 기존 예측 주변에서만 수정

디코더가 기존 모델의 예측을 지나치게 크게 변경하지 않도록 `window`를 둡니다.

예를 들어

```text
기존 예측 CF = 0.60
window = ±0.07
```

이면

```text
0.53 ~ 0.67
```

사이에서만 FICR 기대효용이 가장 높은 값을 찾습니다.

그리고 `blend`를 사용해 기존 예측과 다시 섞습니다.

```text
최종 예측
= blend × 디코더 예측
+ (1 - blend) × 기존 예측
```

예를 들어

```text
Base    = 0.60
Decoder = 0.65
Blend   = 0.7
```

이면 최종 예측은 `0.635`가 됩니다.

검증하는 후보는 다음과 같습니다.

```text
Window
±0.02, ±0.03, ±0.05, ±0.07,
±0.10, ±0.13, ±0.15

Blend
1.0, 0.7, 0.5
```

총 21개 조합을 Group별로 비교합니다.

---

## 4. 반기 검증과 Bootstrap Gate 추가

2024년 전체 성능만 좋아졌다고 바로 적용하지 않습니다.

2024년을 상반기와 하반기로 나누어

```text
H1 Score 개선 > 0
AND
H2 Score 개선 > 0
```

인 후보만 남깁니다.

그중 Score가 가장 높은 후보에 대해 하루 단위 Block Bootstrap을 수행합니다.

```text
95% CI 하한 > 0
→ Strict

평균적으로는 개선되지만 CI가 0 포함
→ Marginal

반기 조건 자체를 통과하지 못함
→ 미적용
```

즉 특정 기간에서만 우연히 좋아진 보정이 Test에 적용되는 것을 막기 위한 장치입니다.

---

## 5. Validation Base와 Test Base 분리

이번 버전의 중요한 변경점은 **Recipe를 검증하는 모델과 실제 적용하는 모델이 달라도 된다는 것**입니다.

현재 구조는 다음과 같습니다.

```text
개인_6 2024 Validation
   ↓
좋은 Window / Blend 선택
   ↓
model_2 Test Prediction
   ↓
선택된 Recipe 적용
```

`model_2`의 동일한 Validation 예측을 다시 만들기 어려운 상황을 위해 추가된 구조입니다.

따라서 `개인_6`에서 디코더의 안정성을 검증하고, 검증된 보정 방법을 `model_2` Test 예측에 적용합니다.

---

## 6. 여러 Test 모델 블렌드 지원

`--extra-test`를 사용하면 디코더를 적용하기 전에 두 Test 모델을 먼저 평균낼 수 있습니다.

예를 들어

```text
model_2 × 0.5
+
개인_6 × 0.5
```

로 새로운 Base Prediction을 만든 뒤 그 결과에 FICR 디코더를 적용합니다.

```text
model_2 ───┐
           ├─ Base Blend → FICR Decoder → 제출
개인_6 ────┘
```

`--blend-test 0.5`를 기본적으로 권장합니다.

---

## 최종 출력

### `submission.csv`

Bootstrap까지 Strict 기준을 통과한 Group만 디코더를 적용한 기본 제출 파일입니다.

### `submission_marginal.csv`

`--allow-marginal`을 사용하면 생성될 수 있으며, Strict 기준은 통과하지 못했지만 개선 가능성이 있는 Marginal Group까지 적용합니다.

### `qrf_report.json`

Group별로 다음 내용을 저장합니다.

```text
기존 Validation 성능
Window × Blend 후보 결과
상·하반기 개선량
Bootstrap CI
Strict / Marginal 판정
최종 선택 Recipe
Test 예측 변화량
```

---

## 핵심 정리

이번에 추가된 부분의 목적은 다음과 같습니다.

```text
기존 모델의 발전량 예측
        ↓
FICR 정산 구조를 직접 고려
        ↓
QRF / Direct로 기대 정산금 계산
        ↓
기존 예측 주변에서만 미세 조정
        ↓
반기 + Bootstrap으로 안정성 확인
        ↓
검증된 Group만 Test에 적용
```

즉 **새로운 발전량 예측 모델을 하나 더 만든 것이 아니라, 기존 모델의 예측 정확도는 최대한 유지하면서 FICR을 더 받을 수 있는 방향으로 제출값을 조정하는 후처리 계층을 추가한 것**입니다.
