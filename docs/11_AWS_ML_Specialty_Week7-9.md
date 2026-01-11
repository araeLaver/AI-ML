# AWS ML Specialty - 7~9주차 학습 계획

## 목표
- SageMaker 내장 알고리즘 완벽 마스터
- 딥러닝 프레임워크 활용
- 모델 평가 및 하이퍼파라미터 최적화

---

## 시험 출제 포인트 (Domain 3: Modeling - 36%)

| 주제 | 출제 비중 | 핵심 내용 |
|------|---------|----------|
| 알고리즘 선택 | 30% | 문제 유형별 적합 알고리즘 |
| 모델 학습 | 25% | 하이퍼파라미터, 분산 학습 |
| 모델 평가 | 25% | 평가 지표, 과적합/과소적합 |
| 하이퍼파라미터 튜닝 | 20% | HPO 전략, Bayesian |

**이 도메인이 시험의 36%로 가장 중요!**

---

## 7주차: SageMaker 내장 알고리즘 (지도학습)

### Day 43: 알고리즘 선택 프레임워크

**학습 목표**: 문제 유형별 최적 알고리즘 선택

**알고리즘 선택 의사결정 트리**:

```
문제 유형?
│
├── 지도학습
│   ├── 분류
│   │   ├── 이진 분류 → XGBoost, Linear Learner, Factorization Machines
│   │   ├── 다중 분류 → XGBoost, Linear Learner, KNN
│   │   └── 텍스트 분류 → BlazingText
│   │
│   ├── 회귀
│   │   ├── 수치 예측 → XGBoost, Linear Learner
│   │   └── 시계열 예측 → DeepAR
│   │
│   └── 이미지/비디오
│       ├── 이미지 분류 → Image Classification
│       ├── 객체 탐지 → Object Detection
│       └── 시맨틱 분할 → Semantic Segmentation
│
├── 비지도학습
│   ├── 클러스터링 → K-Means
│   ├── 차원 축소 → PCA
│   ├── 이상 탐지 → Random Cut Forest, IP Insights
│   └── 토픽 모델링 → LDA, NTM
│
└── 강화학습 → RL (Coach, Ray)
```

**시험 빈출 알고리즘 TOP 10**:

| 순위 | 알고리즘 | 용도 | 시험 빈도 |
|------|---------|------|---------|
| 1 | **XGBoost** | 분류/회귀 | ★★★★★ |
| 2 | **Linear Learner** | 분류/회귀 | ★★★★★ |
| 3 | **BlazingText** | 텍스트 분류/임베딩 | ★★★★ |
| 4 | **DeepAR** | 시계열 예측 | ★★★★ |
| 5 | **Random Cut Forest** | 이상 탐지 | ★★★★ |
| 6 | **K-Means** | 클러스터링 | ★★★★ |
| 7 | **PCA** | 차원 축소 | ★★★ |
| 8 | **Factorization Machines** | 추천/희소 데이터 | ★★★ |
| 9 | **Image Classification** | 이미지 분류 | ★★★ |
| 10 | **Object Detection** | 객체 탐지 | ★★★ |

---

### Day 44: XGBoost 심화

**학습 목표**: XGBoost 하이퍼파라미터 완벽 이해

**XGBoost 특징**:
- Gradient Boosting 기반
- 정형 데이터 최강자
- 분류/회귀 모두 지원
- 내장 결측치 처리
- CSV, LibSVM, Parquet 지원

**핵심 하이퍼파라미터 (시험 필수)**:

| 파라미터 | 설명 | 기본값 | 범위 |
|---------|------|-------|------|
| **num_round** | 부스팅 라운드 수 | 필수 | 10-1000 |
| **max_depth** | 트리 최대 깊이 | 6 | 1-10 |
| **eta (learning_rate)** | 학습률 | 0.3 | 0.01-0.3 |
| **subsample** | 샘플링 비율 | 1 | 0.5-1 |
| **colsample_bytree** | 열 샘플링 비율 | 1 | 0.5-1 |
| **min_child_weight** | 최소 리프 가중치 | 1 | 1-10 |
| **gamma** | 분할 최소 손실 감소 | 0 | 0-5 |
| **alpha (L1)** | L1 정규화 | 0 | 0-1 |
| **lambda (L2)** | L2 정규화 | 1 | 0-1 |

**목적 함수 (objective)**:

| 값 | 문제 유형 |
|---|---------|
| `binary:logistic` | 이진 분류 (확률) |
| `multi:softmax` | 다중 분류 (클래스) |
| `multi:softprob` | 다중 분류 (확률) |
| `reg:squarederror` | 회귀 (MSE) |
| `reg:logistic` | 회귀 (로지스틱) |

**XGBoost 학습 코드**:
```python
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

role = sagemaker.get_execution_role()
session = sagemaker.Session()
bucket = session.default_bucket()

# XGBoost 컨테이너
container = sagemaker.image_uris.retrieve(
    framework='xgboost',
    region=session.boto_region_name,
    version='1.5-1'
)

# 하이퍼파라미터 설정
hyperparameters = {
    'objective': 'binary:logistic',
    'num_round': 100,
    'max_depth': 5,
    'eta': 0.2,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'eval_metric': 'auc'
}

# Estimator 생성
xgb = Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    hyperparameters=hyperparameters,
    output_path=f's3://{bucket}/models/xgboost/'
)

# 학습 데이터 설정
train_input = TrainingInput(
    s3_data=f's3://{bucket}/data/train/',
    content_type='csv'
)
validation_input = TrainingInput(
    s3_data=f's3://{bucket}/data/validation/',
    content_type='csv'
)

# 학습 시작
xgb.fit({
    'train': train_input,
    'validation': validation_input
})
```

**과적합 방지 (시험 포인트)**:
```
과적합 징후: train_auc ↑, validation_auc ↓

해결책:
1. max_depth 감소 (6→3)
2. eta 감소 (0.3→0.1)
3. min_child_weight 증가
4. subsample, colsample_bytree 감소 (1→0.8)
5. alpha, lambda 정규화 증가
6. early_stopping_rounds 설정
```

---

### Day 45: Linear Learner

**학습 목표**: Linear Learner 활용법

**Linear Learner 특징**:
- 선형 모델 (로지스틱/선형 회귀)
- 대용량 데이터에 효율적
- 자동 모델 튜닝
- SGD 기반 학습
- RecordIO-Protobuf 권장

**핵심 하이퍼파라미터**:

| 파라미터 | 설명 | 값 |
|---------|------|---|
| **predictor_type** | 분류/회귀 | binary_classifier, multiclass_classifier, regressor |
| **num_models** | 병렬 모델 수 | auto (32) |
| **loss** | 손실 함수 | auto, logistic, squared_loss, hinge |
| **optimizer** | 최적화 알고리즘 | auto, sgd, adam |
| **learning_rate** | 학습률 | auto |
| **l1** | L1 정규화 | auto |
| **wd (L2)** | L2 정규화 | auto |
| **normalize_data** | 데이터 정규화 | true/false |
| **normalize_label** | 레이블 정규화 | auto |

**Linear Learner 코드**:
```python
from sagemaker import LinearLearner

ll = LinearLearner(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    predictor_type='binary_classifier',
    num_models=32,           # 자동 모델 튜닝
    loss='auto',
    optimizer='auto',
    normalize_data=True,
    normalize_label='auto',
    output_path=f's3://{bucket}/models/linear-learner/'
)

# RecordIO 형식 권장
ll.fit({'train': train_recordio, 'validation': validation_recordio})
```

**XGBoost vs Linear Learner 선택**:

| 상황 | 선택 |
|------|------|
| 비선형 관계 | XGBoost |
| 선형 관계 | Linear Learner |
| 해석 가능성 필요 | Linear Learner |
| 대용량 데이터 | Linear Learner (빠름) |
| 특성 상호작용 중요 | XGBoost |
| 기본 성능 우선 | XGBoost |

---

### Day 46: BlazingText

**학습 목표**: 텍스트 분류 및 Word2Vec

**BlazingText 모드**:

| 모드 | 용도 | 입력 형식 |
|------|------|---------|
| **supervised** | 텍스트 분류 | `__label__class text` |
| **skipgram** | Word2Vec 임베딩 | 텍스트 |
| **cbow** | Word2Vec 임베딩 | 텍스트 |

**텍스트 분류 (supervised)**:
```python
# 입력 데이터 형식
# __label__positive This movie is great!
# __label__negative Terrible experience...

from sagemaker import BlazingText

bt = BlazingText(
    role=role,
    instance_count=1,
    instance_type='ml.c5.xlarge',
    output_path=f's3://{bucket}/models/blazingtext/'
)

bt.set_hyperparameters(
    mode='supervised',
    epochs=10,
    learning_rate=0.05,
    vector_dim=100,
    min_count=2,
    word_ngrams=2  # bigram
)

bt.fit({'train': train_channel})
```

**Word2Vec (skipgram/cbow)**:
```python
bt.set_hyperparameters(
    mode='skipgram',        # 또는 'cbow'
    vector_dim=100,
    window_size=5,
    negative_samples=5,
    min_count=5,
    epochs=5
)
```

**시험 포인트**:
- `supervised` = fastText 기반 텍스트 분류
- `skipgram` = 희귀 단어에 강함
- `cbow` = 빈번한 단어에 강함, 빠름

---

### Day 47: DeepAR (시계열 예측)

**학습 목표**: 시계열 예측 알고리즘

**DeepAR 특징**:
- RNN 기반 시계열 예측
- 여러 관련 시계열 동시 학습
- 확률적 예측 (분위수)
- JSON Lines 입력

**입력 데이터 형식**:
```json
{"start": "2024-01-01 00:00:00", "target": [10, 20, 30, 40, 50]}
{"start": "2024-01-01 00:00:00", "target": [15, 25, 35, 45, 55], "cat": [0]}
{"start": "2024-01-01 00:00:00", "target": [12, 22, 32, 42, 52], "dynamic_feat": [[1,2,3,4,5]]}
```

**핵심 하이퍼파라미터**:

| 파라미터 | 설명 | 권장값 |
|---------|------|-------|
| **context_length** | 입력 시퀀스 길이 | 예측 길이와 유사 |
| **prediction_length** | 예측 길이 | 필수 |
| **time_freq** | 시간 간격 | H(시간), D(일), W(주), M(월) |
| **num_layers** | RNN 레이어 수 | 2-3 |
| **num_cells** | 셀 수 | 40-100 |
| **likelihood** | 분포 유형 | gaussian, negative-binomial, student-t |

**DeepAR 코드**:
```python
from sagemaker import DeepAR

deepar = DeepAR(
    role=role,
    instance_count=1,
    instance_type='ml.c5.xlarge',
    output_path=f's3://{bucket}/models/deepar/'
)

deepar.set_hyperparameters(
    time_freq='H',              # 시간 단위
    context_length=72,          # 과거 72시간
    prediction_length=24,       # 미래 24시간 예측
    num_layers=2,
    num_cells=40,
    likelihood='gaussian',
    epochs=100
)

deepar.fit({
    'train': train_channel,
    'test': test_channel
})
```

**예측 결과 (분위수)**:
```python
# 추론 결과
{
    "predictions": [
        {
            "quantiles": {
                "0.1": [10.2, 11.3, ...],   # 10% 분위수
                "0.5": [15.0, 16.5, ...],   # 중앙값
                "0.9": [20.1, 22.3, ...]    # 90% 분위수
            }
        }
    ]
}
```

---

### Day 48-49: 주말 실습 - 지도학습 종합

**프로젝트**: 금융 사기 탐지 모델 비교

```python
# 1. XGBoost
xgb = Estimator(
    image_uri=xgb_container,
    hyperparameters={
        'objective': 'binary:logistic',
        'num_round': 100,
        'max_depth': 5,
        'eta': 0.1,
        'scale_pos_weight': 10  # 불균형 처리
    }
)

# 2. Linear Learner
ll = LinearLearner(
    predictor_type='binary_classifier',
    positive_example_weight_mult='balanced'  # 불균형 처리
)

# 3. 성능 비교
models = {'XGBoost': xgb, 'Linear Learner': ll}
results = {}

for name, model in models.items():
    model.fit({'train': train_data, 'validation': val_data})
    predictor = model.deploy(instance_type='ml.m5.large', initial_instance_count=1)
    predictions = predictor.predict(test_data)
    results[name] = calculate_metrics(predictions, y_test)
    predictor.delete_endpoint()

print(pd.DataFrame(results))
```

---

## 8주차: 비지도학습 + 딥러닝

### Day 50: K-Means + PCA

**학습 목표**: 클러스터링 및 차원 축소

**K-Means 핵심**:

| 파라미터 | 설명 | 기본값 |
|---------|------|-------|
| **k** | 클러스터 수 | 필수 |
| **init_method** | 초기화 방법 | random |
| **epochs** | 반복 횟수 | 10 |
| **mini_batch_size** | 배치 크기 | 5000 |

```python
from sagemaker import KMeans

kmeans = KMeans(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    k=10,
    init_method='kmeans++',
    epochs=20
)

kmeans.fit({'train': train_recordio})
```

**PCA (Principal Component Analysis)**:

| 파라미터 | 설명 |
|---------|------|
| **num_components** | 주성분 수 |
| **algorithm_mode** | regular (정확) / randomized (빠름) |
| **subtract_mean** | 평균 제거 |

```python
from sagemaker import PCA

pca = PCA(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    num_components=10,
    algorithm_mode='regular',
    subtract_mean=True
)

pca.fit({'train': train_recordio})
```

**PCA 사용 시기 (시험 포인트)**:
- 고차원 데이터 시각화 (2-3D)
- 다중공선성 제거
- 학습 속도 개선
- 노이즈 제거

---

### Day 51: Random Cut Forest (이상 탐지)

**학습 목표**: 비지도 이상 탐지

**RCF 특징**:
- Isolation Forest 변형
- 스트리밍 데이터 지원
- 비지도 학습 (라벨 불필요)
- 이상 점수 출력 (높을수록 이상)

**핵심 하이퍼파라미터**:

| 파라미터 | 설명 | 기본값 |
|---------|------|-------|
| **num_trees** | 트리 수 | 50 |
| **num_samples_per_tree** | 트리당 샘플 수 | 256 |
| **feature_dim** | 특성 차원 | 필수 |

```python
from sagemaker import RandomCutForest

rcf = RandomCutForest(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    num_trees=100,
    num_samples_per_tree=256
)

rcf.fit({'train': train_recordio})

# 추론
predictor = rcf.deploy(instance_type='ml.m5.large', initial_instance_count=1)
results = predictor.predict(test_data)

# 결과: 이상 점수 (anomaly_score)
# 점수가 높을수록 이상 가능성 높음
```

**Kinesis Analytics에서 RCF**:
```sql
-- 실시간 이상 탐지
CREATE OR REPLACE STREAM "ANOMALY_STREAM" (
    anomaly_score DOUBLE,
    value DOUBLE
);

CREATE OR REPLACE PUMP "ANOMALY_PUMP" AS
INSERT INTO "ANOMALY_STREAM"
SELECT STREAM
    ANOMALY_SCORE,
    value
FROM TABLE(
    RANDOM_CUT_FOREST(
        CURSOR(SELECT STREAM value FROM "SOURCE_STREAM"),
        100,    -- numberOfTrees
        256,    -- subSampleSize
        100000  -- timeDecay
    )
);
```

---

### Day 52: SageMaker 딥러닝 프레임워크

**학습 목표**: TensorFlow, PyTorch 활용

**지원 프레임워크**:

| 프레임워크 | 사용 시기 |
|-----------|---------|
| **TensorFlow** | 프로덕션, 모바일, TFX |
| **PyTorch** | 연구, 유연성, NLP |
| **MXNet** | 분산 학습, Gluon |
| **Hugging Face** | NLP, Transformers |

**TensorFlow Estimator**:
```python
from sagemaker.tensorflow import TensorFlow

tf_estimator = TensorFlow(
    entry_point='train.py',
    source_dir='src',
    role=role,
    instance_count=1,
    instance_type='ml.p3.2xlarge',  # GPU
    framework_version='2.12',
    py_version='py310',
    hyperparameters={
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001
    }
)

tf_estimator.fit({'train': train_data, 'validation': val_data})
```

**train.py 예시**:
```python
import argparse
import tensorflow as tf
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    return parser.parse_args()

def main():
    args = parse_args()

    # 데이터 로드
    train_data = load_data(args.train)

    # 모델 정의
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )

    # 학습
    model.fit(
        train_data,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # 저장
    model.save(os.path.join(args.model_dir, '1'))

if __name__ == '__main__':
    main()
```

**PyTorch Estimator**:
```python
from sagemaker.pytorch import PyTorch

pytorch_estimator = PyTorch(
    entry_point='train.py',
    source_dir='src',
    role=role,
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    framework_version='2.0',
    py_version='py310',
    hyperparameters={
        'epochs': 10,
        'batch_size': 32
    }
)

pytorch_estimator.fit({'train': train_data})
```

---

### Day 53: Hugging Face on SageMaker

**학습 목표**: Transformers 모델 학습/배포

```python
from sagemaker.huggingface import HuggingFace

huggingface_estimator = HuggingFace(
    entry_point='train.py',
    source_dir='src',
    role=role,
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    transformers_version='4.28',
    pytorch_version='2.0',
    py_version='py310',
    hyperparameters={
        'model_name': 'bert-base-uncased',
        'epochs': 3,
        'batch_size': 16
    }
)

huggingface_estimator.fit({'train': train_data})
```

**train.py (Hugging Face)**:
```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import load_from_disk

def main():
    # 모델 및 토크나이저
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 데이터
    dataset = load_from_disk('/opt/ml/input/data/train')

    # 학습 설정
    training_args = TrainingArguments(
        output_dir='/opt/ml/model',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model('/opt/ml/model')

if __name__ == '__main__':
    main()
```

---

### Day 54-55: 주말 실습 - 딥러닝 파이프라인

**프로젝트**: 이미지 분류 (TensorFlow)

**체크리스트**:
- [ ] 데이터 준비 (S3 업로드)
- [ ] TensorFlow 학습 스크립트 작성
- [ ] SageMaker Estimator 설정
- [ ] GPU 인스턴스 학습
- [ ] 모델 배포 및 추론
- [ ] 리소스 정리

---

## 9주차: 모델 평가 + 하이퍼파라미터 튜닝

### Day 56: 모델 평가 지표

**학습 목표**: 평가 지표 완벽 이해

**분류 평가 지표 (시험 필수)**:

| 지표 | 공식 | 용도 |
|------|------|------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | 균형 데이터 |
| **Precision** | TP/(TP+FP) | FP 비용 높을 때 |
| **Recall (Sensitivity)** | TP/(TP+FN) | FN 비용 높을 때 |
| **F1 Score** | 2×(P×R)/(P+R) | P와 R 균형 |
| **AUC-ROC** | ROC 곡선 아래 면적 | 전체 성능 |
| **AUC-PR** | PR 곡선 아래 면적 | 불균형 데이터 |

**혼동 행렬**:
```
              예측
            Pos   Neg
실제  Pos   TP    FN
      Neg   FP    TN

TP: True Positive (정탐)
TN: True Negative (정상)
FP: False Positive (오탐, Type I)
FN: False Negative (미탐, Type II)
```

**불균형 데이터 처리 (시험 포인트)**:

| 방법 | 설명 | 적용 |
|------|------|------|
| **Oversampling** | 소수 클래스 증가 | SMOTE |
| **Undersampling** | 다수 클래스 감소 | Random |
| **Class Weight** | 클래스별 가중치 | 모델 파라미터 |
| **Threshold Tuning** | 임계값 조정 | 후처리 |

**회귀 평가 지표**:

| 지표 | 공식 | 특징 |
|------|------|------|
| **MSE** | Σ(y-ŷ)²/n | 큰 오차 패널티 |
| **RMSE** | √MSE | 원래 스케일 |
| **MAE** | Σ\|y-ŷ\|/n | 이상치에 강건 |
| **MAPE** | Σ\|y-ŷ\|/y × 100 | 상대적 오차 |
| **R²** | 1 - SS_res/SS_tot | 설명력 |

---

### Day 57: 과적합 vs 과소적합

**학습 목표**: 편향-분산 트레이드오프

```
과소적합 (High Bias)          과적합 (High Variance)
─────────────────────         ─────────────────────
• Train Error 높음            • Train Error 낮음
• Val Error 높음              • Val Error 높음
• 모델이 너무 단순             • 모델이 너무 복잡

해결책:                        해결책:
• 복잡한 모델 사용             • 더 많은 데이터
• 특성 추가                    • 정규화 (L1, L2)
• 정규화 감소                  • Dropout
                              • Early stopping
                              • 교차 검증
```

**학습 곡선 분석**:
```python
import matplotlib.pyplot as plt

# 학습 곡선
train_sizes = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
train_scores = []
val_scores = []

for size in train_sizes:
    model.fit(X_train[:int(len(X_train)*size)], y_train[:int(len(y_train)*size)])
    train_scores.append(model.score(X_train, y_train))
    val_scores.append(model.score(X_val, y_val))

plt.plot(train_sizes, train_scores, label='Train')
plt.plot(train_sizes, val_scores, label='Validation')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.legend()
plt.title('Learning Curve')
```

**시험 포인트**:
- Train/Val 모두 높음 → **과소적합** → 복잡도 증가
- Train 높음, Val 낮음 → **과적합** → 정규화, 데이터 증가
- Train/Val 모두 낮음 → **과소적합** → 특성 추가, 복잡한 모델

---

### Day 58: SageMaker Automatic Model Tuning

**학습 목표**: 자동 하이퍼파라미터 최적화

**HPO 전략 비교**:

| 전략 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **Grid Search** | 모든 조합 탐색 | 완전 탐색 | 비용 높음 |
| **Random Search** | 무작위 탐색 | 빠름, 병렬화 | 최적 보장 안됨 |
| **Bayesian** | 과거 결과 활용 | 효율적 | 순차적 |
| **Hyperband** | 조기 종료 | 빠름 | 복잡한 설정 |

**SageMaker HPO 코드**:
```python
from sagemaker.tuner import (
    HyperparameterTuner,
    IntegerParameter,
    ContinuousParameter,
    CategoricalParameter
)

# 하이퍼파라미터 범위 정의
hyperparameter_ranges = {
    'max_depth': IntegerParameter(3, 10),
    'eta': ContinuousParameter(0.01, 0.3),
    'subsample': ContinuousParameter(0.5, 1.0),
    'colsample_bytree': ContinuousParameter(0.5, 1.0),
    'min_child_weight': IntegerParameter(1, 10),
    'alpha': ContinuousParameter(0, 1),
    'gamma': ContinuousParameter(0, 5)
}

# 목표 메트릭
objective_metric_name = 'validation:auc'
objective_type = 'Maximize'

# Tuner 생성
tuner = HyperparameterTuner(
    estimator=xgb,
    objective_metric_name=objective_metric_name,
    objective_type=objective_type,
    hyperparameter_ranges=hyperparameter_ranges,
    strategy='Bayesian',        # 또는 'Random'
    max_jobs=50,                # 총 실험 수
    max_parallel_jobs=5,        # 동시 실행 수
    early_stopping_type='Auto'  # 조기 종료
)

# 튜닝 시작
tuner.fit({
    'train': train_input,
    'validation': validation_input
})

# 결과 확인
tuner.wait()
best_job = tuner.best_training_job()
print(f"Best job: {best_job}")

# 최적 하이퍼파라미터
tuner.best_estimator()
```

**Warm Start (전이 학습)**:
```python
from sagemaker.tuner import WarmStartConfig, WarmStartTypes

# 이전 튜닝 작업 활용
warm_start_config = WarmStartConfig(
    warm_start_type=WarmStartTypes.TRANSFER_LEARNING,
    parents=['previous-tuning-job-name']
)

tuner = HyperparameterTuner(
    ...,
    warm_start_config=warm_start_config
)
```

---

### Day 59: 교차 검증 + 앙상블

**학습 목표**: 모델 검증 및 결합

**K-Fold 교차 검증**:
```python
from sklearn.model_selection import KFold, cross_val_score

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='roc_auc')

print(f"AUC: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

**Stratified K-Fold (불균형 데이터)**:
```python
from sklearn.model_selection import StratifiedKFold

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skfold, scoring='roc_auc')
```

**앙상블 기법**:

| 기법 | 설명 | 적용 |
|------|------|------|
| **Bagging** | 병렬 학습 + 평균 | Random Forest |
| **Boosting** | 순차 학습 + 가중 | XGBoost, AdaBoost |
| **Stacking** | 메타 모델 학습 | 다양한 모델 결합 |
| **Voting** | 다수결 | 분류 |
| **Blending** | 가중 평균 | 회귀 |

```python
from sklearn.ensemble import VotingClassifier, StackingClassifier

# Voting
voting = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('lr', lr_model)
    ],
    voting='soft'  # 확률 기반
)

# Stacking
stacking = StackingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('rf', rf_model)
    ],
    final_estimator=LogisticRegression(),
    cv=5
)
```

---

### Day 60-61: 주말 종합 실습

**프로젝트**: 완전한 모델링 파이프라인

```
데이터 → 전처리 → 특성 엔지니어링 → 모델 학습 → HPO → 평가 → 배포
```

**체크리스트**:
- [ ] 데이터 분할 (Train/Val/Test)
- [ ] 기본 모델 학습 (XGBoost)
- [ ] 평가 지표 계산 (AUC, F1, Precision, Recall)
- [ ] 하이퍼파라미터 튜닝 (20+ jobs)
- [ ] 최적 모델 선택
- [ ] 교차 검증
- [ ] 테스트 셋 최종 평가
- [ ] 모델 배포

---

## 7-9주차 퀴즈 (자가 점검)

### Q1. XGBoost에서 과적합 해결 방법 3가지는?
<details>
<summary>정답 보기</summary>

1. max_depth 감소
2. eta (learning_rate) 감소
3. subsample, colsample_bytree 감소 (0.8)
4. alpha, lambda 정규화 증가
5. early_stopping_rounds 설정
</details>

### Q2. BlazingText의 3가지 모드는?
<details>
<summary>정답 보기</summary>

1. **supervised**: 텍스트 분류 (fastText 기반)
2. **skipgram**: Word2Vec (희귀 단어에 강함)
3. **cbow**: Word2Vec (빈번 단어에 강함, 빠름)
</details>

### Q3. 불균형 데이터에서 Accuracy 대신 사용할 지표는?
<details>
<summary>정답 보기</summary>

- **AUC-ROC**: 전체 성능 측정
- **AUC-PR**: 불균형에 민감
- **F1 Score**: Precision과 Recall 균형
- **Recall**: FN 비용이 높을 때 (사기 탐지)
</details>

### Q4. SageMaker HPO에서 Bayesian vs Random 선택 기준은?
<details>
<summary>정답 보기</summary>

- **Bayesian**: 적은 실험으로 최적화, 순차적 실행
- **Random**: 빠른 탐색, 완전 병렬화 가능, 파라미터 영향 파악
</details>

### Q5. DeepAR 입력 데이터 형식과 출력은?
<details>
<summary>정답 보기</summary>

- **입력**: JSON Lines (`{"start": "...", "target": [...]}`)
- **출력**: 분위수 예측 (0.1, 0.5, 0.9 등)
</details>

---

## 핵심 알고리즘 요약표

| 알고리즘 | 문제 유형 | 입력 형식 | 핵심 파라미터 |
|---------|---------|---------|------------|
| **XGBoost** | 분류/회귀 | CSV, LibSVM | max_depth, eta, num_round |
| **Linear Learner** | 분류/회귀 | RecordIO | predictor_type, num_models |
| **BlazingText** | 텍스트 | Text | mode, vector_dim |
| **DeepAR** | 시계열 | JSON Lines | prediction_length, context_length |
| **K-Means** | 클러스터링 | RecordIO | k, init_method |
| **PCA** | 차원 축소 | RecordIO | num_components |
| **RCF** | 이상 탐지 | RecordIO | num_trees, num_samples_per_tree |
| **Image Classification** | 이미지 | RecordIO | num_classes, num_layers |

---

## 다음 주 예고 (10-11주차)

- 모델 배포 전략 (Real-time, Batch, Async)
- SageMaker MLOps (Pipelines, Model Registry)
- 모니터링 (Model Monitor, Clarify)
- 보안 및 비용 최적화
