# AWS ML Specialty - 5~6주차 학습 계획

## 목표
- 탐색적 데이터 분석 (EDA) 마스터
- 특성 엔지니어링 기법 습득
- AWS 데이터 분석/시각화 도구 활용

---

## 시험 출제 포인트 (Domain 2: EDA - 24%)

| 주제 | 출제 비중 | 핵심 개념 |
|------|---------|----------|
| 데이터 시각화 | 20% | 분포, 상관관계, 이상치 |
| 통계 분석 | 25% | 기술통계, 가설검정 |
| 특성 엔지니어링 | 35% | 변환, 인코딩, 스케일링 |
| 데이터 품질 | 20% | 결측치, 이상치, 불균형 |

---

## 5주차: 탐색적 데이터 분석 (EDA)

### Day 29: EDA 기초 + 기술통계

**학습 목표**: 데이터 이해를 위한 기초 분석

**기술통계 요약**:

| 측정값 | 용도 | 시험 포인트 |
|--------|------|-----------|
| **평균 (Mean)** | 중심 경향 | 이상치에 민감 |
| **중앙값 (Median)** | 중심 경향 | 이상치에 강건 |
| **최빈값 (Mode)** | 가장 빈번한 값 | 범주형에 적합 |
| **분산 (Variance)** | 분산 정도 | 제곱 단위 |
| **표준편차 (Std)** | 분산 정도 | 원래 단위 |
| **왜도 (Skewness)** | 분포 비대칭 | 0=대칭, +우측, -좌측 |
| **첨도 (Kurtosis)** | 꼬리 두께 | 3=정규, >3=뾰족 |

**Python EDA 기본**:
```python
import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('data.csv')

# 기본 정보
print(df.shape)           # (행, 열)
print(df.info())          # 데이터 타입, 결측치
print(df.describe())      # 기술통계

# 상세 통계
print(f"평균: {df['amount'].mean()}")
print(f"중앙값: {df['amount'].median()}")
print(f"왜도: {df['amount'].skew()}")
print(f"첨도: {df['amount'].kurtosis()}")

# 결측치 확인
print(df.isnull().sum())
print(df.isnull().sum() / len(df) * 100)  # 비율
```

**분포 유형 (시험 필수)**:

```
정규분포 (Normal)     우측 왜곡 (Right)     좌측 왜곡 (Left)
     ▲                    ▲                    ▲
    ╱ ╲                  ╱  ╲                 ╱  ╲
   ╱   ╲                ╱    ╲___            ╱    ╲
  ╱     ╲              ╱         ╲       ___╱      ╲
 ╱       ╲            ╱           ╲     ╱           ╲
━━━━━━━━━━━          ━━━━━━━━━━━━━━    ━━━━━━━━━━━━━━
Mean=Median          Mean > Median     Mean < Median
```

---

### Day 30: 데이터 시각화

**학습 목표**: 시각화를 통한 데이터 인사이트 도출

**시각화 유형별 용도**:

| 차트 유형 | 용도 | 변수 유형 |
|----------|------|---------|
| **히스토그램** | 분포 확인 | 수치형 1개 |
| **박스플롯** | 이상치, 분포 | 수치형 1개 |
| **산점도** | 관계 분석 | 수치형 2개 |
| **히트맵** | 상관관계 | 수치형 다수 |
| **바 차트** | 빈도/비교 | 범주형 |
| **파이 차트** | 비율 | 범주형 (5개 이하) |
| **라인 차트** | 시계열 | 시간 + 수치형 |

**시각화 코드**:
```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 히스토그램 - 분포 확인
axes[0, 0].hist(df['amount'], bins=50, edgecolor='black')
axes[0, 0].set_title('Amount Distribution')
axes[0, 0].axvline(df['amount'].mean(), color='red', label='Mean')
axes[0, 0].axvline(df['amount'].median(), color='green', label='Median')
axes[0, 0].legend()

# 2. 박스플롯 - 이상치 확인
axes[0, 1].boxplot(df['amount'])
axes[0, 1].set_title('Amount Boxplot')

# 3. 상관관계 히트맵
corr_matrix = df[['amount', 'age', 'balance', 'transactions']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[1, 0])
axes[1, 0].set_title('Correlation Heatmap')

# 4. 범주별 비교
df.groupby('category')['amount'].mean().plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_title('Average Amount by Category')

plt.tight_layout()
plt.savefig('eda_plots.png')
```

**상관관계 해석**:

| 상관계수 | 해석 |
|---------|------|
| 0.9 ~ 1.0 | 매우 강한 양의 상관 |
| 0.7 ~ 0.9 | 강한 양의 상관 |
| 0.4 ~ 0.7 | 중간 양의 상관 |
| 0.1 ~ 0.4 | 약한 양의 상관 |
| -0.1 ~ 0.1 | 거의 없음 |
| -1.0 ~ -0.1 | 음의 상관 (역방향) |

---

### Day 31: Amazon QuickSight

**학습 목표**: AWS 시각화 도구 활용

**QuickSight 특징**:
- 서버리스 BI 도구
- S3, Athena, Redshift 연동
- SPICE 인메모리 엔진
- ML 인사이트 (이상 탐지, 예측)

**데이터 소스 연결**:
```
1. QuickSight Console → Datasets
2. New dataset → Athena
3. Database: ml_database
4. Table: transactions
5. Import to SPICE (권장) or Direct Query
```

**ML 인사이트 기능**:

| 기능 | 설명 | 용도 |
|------|------|------|
| **Anomaly Detection** | 이상치 자동 탐지 | 모니터링 |
| **Forecasting** | 시계열 예측 | 트렌드 분석 |
| **Auto-narratives** | 자동 설명 생성 | 리포트 |
| **What-if Analysis** | 시나리오 분석 | 의사결정 |

**시험 포인트**:
- SPICE: Super-fast, Parallel, In-memory Calculation Engine
- 사용자당 과금 (Author: $24/월, Reader: $0.30/세션)
- Row-Level Security (RLS) 지원

---

### Day 32: 이상치 탐지 기법

**학습 목표**: 이상치 식별 및 처리 방법

**이상치 탐지 방법**:

| 방법 | 공식 | 적용 시기 |
|------|------|---------|
| **IQR** | Q1-1.5×IQR ~ Q3+1.5×IQR | 일반적 |
| **Z-score** | \|z\| > 3 | 정규분포 가정 |
| **Modified Z-score** | MAD 기반 | 비정규분포 |
| **Isolation Forest** | 트리 기반 | 다변량 |
| **DBSCAN** | 밀도 기반 | 클러스터링 |

**IQR 방법**:
```python
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

outliers, lower, upper = detect_outliers_iqr(df, 'amount')
print(f"이상치 개수: {len(outliers)}")
print(f"범위: {lower:.2f} ~ {upper:.2f}")
```

**Z-score 방법**:
```python
from scipy import stats

def detect_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    outliers = df[z_scores > threshold]
    return outliers

outliers = detect_outliers_zscore(df, 'amount')
```

**이상치 처리 전략**:

| 전략 | 설명 | 적용 시기 |
|------|------|---------|
| **제거** | 이상치 행 삭제 | 데이터 충분, 명확한 오류 |
| **대체** | 중앙값/평균으로 | 데이터 부족 |
| **캡핑** | 상하한 제한 | 극단값 제한 필요 |
| **변환** | 로그, 제곱근 | 왜곡 분포 |
| **분리** | 별도 모델링 | 이상치가 중요한 경우 |

---

### Day 33: 결측치 처리

**학습 목표**: 결측치 패턴 분석 및 처리

**결측치 유형 (시험 필수)**:

| 유형 | 영문 | 설명 | 예시 |
|------|------|------|------|
| **MCAR** | Missing Completely At Random | 완전 무작위 | 입력 실수 |
| **MAR** | Missing At Random | 조건부 무작위 | 특정 그룹 미응답 |
| **MNAR** | Missing Not At Random | 비무작위 | 고소득자 소득 미공개 |

**결측치 시각화**:
```python
import missingno as msno

# 결측치 매트릭스
msno.matrix(df)
plt.savefig('missing_matrix.png')

# 결측치 히트맵 (상관관계)
msno.heatmap(df)
plt.savefig('missing_heatmap.png')

# 결측치 바 차트
msno.bar(df)
plt.savefig('missing_bar.png')
```

**결측치 처리 방법**:

| 방법 | 코드 | 적용 시기 |
|------|------|---------|
| **삭제 (행)** | `df.dropna()` | MCAR, 충분한 데이터 |
| **삭제 (열)** | `df.drop(columns=[])` | 결측 > 50% |
| **평균 대체** | `df.fillna(df.mean())` | 수치형, 정규분포 |
| **중앙값 대체** | `df.fillna(df.median())` | 수치형, 왜곡분포 |
| **최빈값 대체** | `df.fillna(df.mode()[0])` | 범주형 |
| **KNN 대체** | `KNNImputer` | 다변량 관계 |
| **회귀 대체** | `IterativeImputer` | 복잡한 관계 |

**고급 대체 (KNN)**:
```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(
    imputer.fit_transform(df[numeric_columns]),
    columns=numeric_columns
)
```

---

### Day 34-35: 주말 실습 - 종합 EDA

**프로젝트**: 금융 거래 데이터 EDA

```python
# 1. 데이터 로드 및 기본 탐색
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('transactions.csv')

print("=== 데이터 개요 ===")
print(f"Shape: {df.shape}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")

# 2. 기술통계
print("\n=== 기술통계 ===")
print(df.describe())

# 3. 타겟 변수 분포 (불균형 확인)
print("\n=== 타겟 분포 ===")
print(df['is_fraud'].value_counts(normalize=True))

# 4. 시각화
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 금액 분포
axes[0, 0].hist(df['amount'], bins=50)
axes[0, 0].set_title('Amount Distribution')

# 로그 변환 금액
axes[0, 1].hist(np.log1p(df['amount']), bins=50)
axes[0, 1].set_title('Log(Amount) Distribution')

# 범주별 사기 비율
fraud_by_category = df.groupby('category')['is_fraud'].mean()
fraud_by_category.plot(kind='bar', ax=axes[0, 2])
axes[0, 2].set_title('Fraud Rate by Category')

# 상관관계
corr = df[['amount', 'balance', 'age']].corr()
sns.heatmap(corr, annot=True, ax=axes[1, 0])
axes[1, 0].set_title('Correlation')

# 시간대별 거래
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df.groupby('hour')['amount'].count().plot(ax=axes[1, 1])
axes[1, 1].set_title('Transactions by Hour')

# 사기 vs 정상 금액 비교
df.boxplot(column='amount', by='is_fraud', ax=axes[1, 2])
axes[1, 2].set_title('Amount: Fraud vs Normal')

plt.tight_layout()
plt.savefig('eda_complete.png')
```

---

## 6주차: 특성 엔지니어링

### Day 36: 수치형 특성 변환

**학습 목표**: 수치형 데이터 스케일링 및 변환

**스케일링 기법 (시험 필수)**:

| 기법 | 공식 | 범위 | 적용 시기 |
|------|------|------|---------|
| **StandardScaler** | (x-μ)/σ | 제한없음 | 정규분포, 이상치 적음 |
| **MinMaxScaler** | (x-min)/(max-min) | [0, 1] | 신경망, 이상치 없음 |
| **RobustScaler** | (x-Q2)/(Q3-Q1) | 제한없음 | 이상치 많음 |
| **MaxAbsScaler** | x/max(\|x\|) | [-1, 1] | 희소 데이터 |

**스케일링 코드**:
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler (Z-score 정규화)
scaler = StandardScaler()
df['amount_scaled'] = scaler.fit_transform(df[['amount']])

# MinMaxScaler (0-1 정규화)
minmax = MinMaxScaler()
df['amount_minmax'] = minmax.fit_transform(df[['amount']])

# RobustScaler (이상치에 강건)
robust = RobustScaler()
df['amount_robust'] = robust.fit_transform(df[['amount']])
```

**비선형 변환**:

| 변환 | 공식 | 용도 |
|------|------|------|
| **Log** | log(x+1) | 우측 왜곡 완화 |
| **Square Root** | √x | 경미한 왜곡 |
| **Box-Cox** | (x^λ-1)/λ | 최적 λ 탐색 |
| **Yeo-Johnson** | Box-Cox 확장 | 음수 포함 |

```python
from sklearn.preprocessing import PowerTransformer

# Box-Cox (양수만)
pt_boxcox = PowerTransformer(method='box-cox')
df['amount_boxcox'] = pt_boxcox.fit_transform(df[['amount']] + 1)

# Yeo-Johnson (음수 포함)
pt_yeo = PowerTransformer(method='yeo-johnson')
df['amount_yeo'] = pt_yeo.fit_transform(df[['amount']])
```

---

### Day 37: 범주형 특성 인코딩

**학습 목표**: 범주형 데이터 수치화

**인코딩 기법 비교**:

| 기법 | 특징 | 적용 시기 | 주의사항 |
|------|------|---------|---------|
| **Label Encoding** | 정수 할당 | 트리 모델, 순서형 | 순서 의미 부여 |
| **One-Hot Encoding** | 이진 벡터 | 선형 모델, 비순서형 | 고차원화 |
| **Target Encoding** | 타겟 평균 | 고카디널리티 | 과적합 위험 |
| **Frequency Encoding** | 빈도 할당 | 빈도 중요 시 | 충돌 가능 |
| **Binary Encoding** | 이진 표현 | 고카디널리티 | 트리 모델 |

**인코딩 코드**:
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import category_encoders as ce

# Label Encoding
le = LabelEncoder()
df['category_le'] = le.fit_transform(df['category'])

# One-Hot Encoding
df_onehot = pd.get_dummies(df, columns=['category'], prefix='cat')

# Target Encoding (category_encoders)
te = ce.TargetEncoder(cols=['category'])
df['category_te'] = te.fit_transform(df['category'], df['is_fraud'])

# Frequency Encoding
freq = df['category'].value_counts(normalize=True)
df['category_freq'] = df['category'].map(freq)
```

**고카디널리티 처리**:
```python
# 상위 N개 + 기타
def reduce_cardinality(df, column, top_n=10):
    top_categories = df[column].value_counts().nlargest(top_n).index
    df[column + '_reduced'] = df[column].apply(
        lambda x: x if x in top_categories else 'Other'
    )
    return df

df = reduce_cardinality(df, 'merchant', top_n=20)
```

---

### Day 38: 특성 생성 기법

**학습 목표**: 새로운 특성 생성

**특성 생성 유형**:

| 유형 | 예시 | 용도 |
|------|------|------|
| **수학적 조합** | A+B, A×B, A/B | 비율, 합계 |
| **다항 특성** | x², x³, x₁×x₂ | 비선형 관계 |
| **시간 특성** | 요일, 시간, 월 | 주기성 |
| **집계 특성** | 평균, 합계, 카운트 | 그룹별 통계 |
| **윈도우 특성** | 이동평균, 누적합 | 시계열 |
| **텍스트 특성** | 길이, 단어수 | NLP |

**특성 생성 코드**:
```python
# 1. 수학적 조합
df['amount_per_age'] = df['amount'] / df['age']
df['balance_ratio'] = df['balance'] / (df['amount'] + 1)

# 2. 다항 특성
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['amount', 'balance']])

# 3. 시간 특성
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['month'] = df['timestamp'].dt.month

# 4. 집계 특성 (사용자별 통계)
user_stats = df.groupby('user_id').agg({
    'amount': ['mean', 'std', 'max', 'count'],
    'is_fraud': 'mean'
}).reset_index()
user_stats.columns = ['user_id', 'user_avg_amount', 'user_std_amount',
                       'user_max_amount', 'user_tx_count', 'user_fraud_rate']
df = df.merge(user_stats, on='user_id', how='left')

# 5. 윈도우 특성 (시계열)
df = df.sort_values(['user_id', 'timestamp'])
df['rolling_mean_7d'] = df.groupby('user_id')['amount'].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean()
)
df['cumsum_amount'] = df.groupby('user_id')['amount'].cumsum()
```

---

### Day 39: 특성 선택

**학습 목표**: 중요 특성 선택 기법

**특성 선택 방법**:

| 방법 | 유형 | 설명 | 도구 |
|------|------|------|------|
| **분산 기반** | Filter | 낮은 분산 제거 | VarianceThreshold |
| **상관관계** | Filter | 높은 상관 제거 | corr() |
| **카이제곱** | Filter | 범주형-타겟 관계 | chi2 |
| **상호정보** | Filter | 비선형 관계 | mutual_info_classif |
| **RFE** | Wrapper | 재귀적 제거 | RFE |
| **L1 정규화** | Embedded | Lasso | SelectFromModel |
| **트리 중요도** | Embedded | 특성 중요도 | RandomForest |

**특성 선택 코드**:
```python
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, chi2,
    mutual_info_classif, RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier

# 1. 분산 기반 (낮은 분산 제거)
selector = VarianceThreshold(threshold=0.01)
X_var = selector.fit_transform(X)

# 2. 상관관계 기반 (다중공선성 제거)
def remove_correlated_features(df, threshold=0.9):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop)

X_uncorr = remove_correlated_features(X, threshold=0.9)

# 3. SelectKBest (상호정보)
selector = SelectKBest(score_func=mutual_info_classif, k=10)
X_best = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

# 4. 트리 기반 중요도
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(importance.head(20))

# 5. RFE (재귀적 특성 제거)
rfe = RFE(estimator=rf, n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)
```

---

### Day 40: SageMaker Data Wrangler

**학습 목표**: AWS 노코드 특성 엔지니어링

**Data Wrangler 기능**:

| 기능 | 설명 |
|------|------|
| **데이터 가져오기** | S3, Athena, Redshift, Snowflake |
| **데이터 탐색** | 히스토그램, 상관관계, 타겟 분석 |
| **변환** | 300+ 내장 변환 |
| **특성 저장소** | Feature Store 연동 |
| **내보내기** | S3, Pipeline, Feature Store |

**주요 변환**:

```
수치형:
- 정규화 (Min-max, Z-score)
- 로그 변환
- Binning (등간격, 분위수)
- 수학 연산

범주형:
- One-hot 인코딩
- Ordinal 인코딩
- Target 인코딩

시간:
- 시간 추출 (년, 월, 일, 시)
- 시간 차이 계산

텍스트:
- 토큰화
- TF-IDF
- 임베딩 (BERT)
```

**Data Wrangler → SageMaker Pipeline**:
```python
# Data Wrangler 플로우를 Pipeline으로 내보내기
# 1. Data Wrangler에서 Export → Pipeline 선택
# 2. 생성된 노트북 실행

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep

# Data Wrangler Processing Step
processing_step = ProcessingStep(
    name="DataWrangler",
    processor=data_wrangler_processor,
    inputs=[...],
    outputs=[...]
)

pipeline = Pipeline(
    name="feature-engineering-pipeline",
    steps=[processing_step]
)

pipeline.upsert(role_arn=role)
pipeline.start()
```

---

### Day 41: SageMaker Feature Store

**학습 목표**: 특성 저장 및 재사용

**Feature Store 개념**:

```
┌─────────────────────────────────────────────────────────────┐
│                 SageMaker Feature Store                      │
├──────────────────────────┬──────────────────────────────────┤
│     Online Store         │        Offline Store             │
│    (실시간 조회)          │       (배치 학습)                 │
├──────────────────────────┼──────────────────────────────────┤
│ • 저지연 (<10ms)         │ • S3 저장                        │
│ • 최신 값만              │ • 전체 이력                       │
│ • 실시간 추론용          │ • 모델 학습용                     │
└──────────────────────────┴──────────────────────────────────┘
```

**Feature Store 코드**:
```python
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_definition import (
    FeatureDefinition, FeatureTypeEnum
)

# Feature Group 정의
feature_group = FeatureGroup(
    name='user-transaction-features',
    sagemaker_session=session
)

# 특성 정의
feature_definitions = [
    FeatureDefinition(feature_name='user_id', feature_type=FeatureTypeEnum.STRING),
    FeatureDefinition(feature_name='avg_amount', feature_type=FeatureTypeEnum.FRACTIONAL),
    FeatureDefinition(feature_name='tx_count', feature_type=FeatureTypeEnum.INTEGRAL),
    FeatureDefinition(feature_name='fraud_rate', feature_type=FeatureTypeEnum.FRACTIONAL),
    FeatureDefinition(feature_name='event_time', feature_type=FeatureTypeEnum.FRACTIONAL),
]

# Feature Group 생성
feature_group.load_feature_definitions(data_frame=df)
feature_group.create(
    s3_uri=f's3://{bucket}/feature-store/',
    record_identifier_name='user_id',
    event_time_feature_name='event_time',
    role_arn=role,
    enable_online_store=True
)

# 데이터 수집
feature_group.ingest(data_frame=df, max_workers=3, wait=True)

# 온라인 조회 (실시간 추론)
record = feature_group.get_record(record_identifier_value_as_string='USER001')

# 오프라인 쿼리 (Athena)
query = feature_group.athena_query()
query.run(
    query_string='SELECT * FROM "feature_group_table" LIMIT 100',
    output_location=f's3://{bucket}/query-results/'
)
query.wait()
df_result = query.as_dataframe()
```

---

### Day 42: 주말 종합 실습

**프로젝트**: 완전한 특성 엔지니어링 파이프라인

**체크리스트**:
- [ ] 데이터 로드 및 EDA
- [ ] 결측치 분석 및 처리
- [ ] 이상치 탐지 및 처리
- [ ] 수치형 스케일링 (StandardScaler)
- [ ] 범주형 인코딩 (Target Encoding)
- [ ] 시간 특성 생성
- [ ] 집계 특성 생성 (사용자별)
- [ ] 특성 선택 (Random Forest 중요도)
- [ ] Feature Store 저장
- [ ] 학습/검증 데이터 분할

---

## 5-6주차 퀴즈 (자가 점검)

### Q1. MCAR, MAR, MNAR의 차이점은?
<details>
<summary>정답 보기</summary>

- **MCAR**: 결측이 완전 무작위 (다른 변수와 무관)
- **MAR**: 결측이 관측된 변수에 의존 (조건부 무작위)
- **MNAR**: 결측이 결측값 자체에 의존 (비무작위)
</details>

### Q2. StandardScaler vs MinMaxScaler 선택 기준은?
<details>
<summary>정답 보기</summary>

- **StandardScaler**: 정규분포 가정, 이상치 적을 때, 트리 외 모델
- **MinMaxScaler**: 범위 고정 필요, 이상치 없을 때, 신경망
</details>

### Q3. 고카디널리티 범주형 변수 처리 방법 3가지는?
<details>
<summary>정답 보기</summary>

1. Target Encoding (타겟 평균)
2. Frequency Encoding (빈도)
3. 상위 N개 + 기타 그룹화
</details>

### Q4. SageMaker Feature Store의 Online vs Offline Store 차이는?
<details>
<summary>정답 보기</summary>

- **Online**: 저지연 조회(<10ms), 최신 값, 실시간 추론용
- **Offline**: S3 저장, 전체 이력, 모델 학습용
</details>

### Q5. 특성 선택에서 Filter vs Wrapper 방법의 차이는?
<details>
<summary>정답 보기</summary>

- **Filter**: 모델 독립적, 통계 기반, 빠름 (분산, 상관관계)
- **Wrapper**: 모델 종속적, 성능 기반, 느림 (RFE)
</details>

---

## 핵심 요약표

| 주제 | 핵심 기법 | 시험 빈도 |
|------|---------|---------|
| **결측치** | MCAR/MAR/MNAR, KNN Imputer | ★★★★★ |
| **이상치** | IQR, Z-score, Isolation Forest | ★★★★ |
| **스케일링** | Standard, MinMax, Robust | ★★★★★ |
| **인코딩** | One-hot, Target, Label | ★★★★★ |
| **특성 생성** | 다항, 집계, 시간 | ★★★★ |
| **특성 선택** | RFE, 트리 중요도 | ★★★★ |
| **QuickSight** | SPICE, ML 인사이트 | ★★★ |
| **Data Wrangler** | 노코드 변환, 내보내기 | ★★★ |
| **Feature Store** | Online/Offline, Athena | ★★★★ |

---

## 다음 주 예고 (7-9주차)

- SageMaker 내장 알고리즘 심화
- 딥러닝 프레임워크 (TensorFlow, PyTorch)
- 모델 선택 및 평가 지표
- 하이퍼파라미터 최적화 심화
