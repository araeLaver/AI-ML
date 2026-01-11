# AWS ML Specialty - 1~2주차 학습 계획

## 목표
- AWS 클라우드 ML 서비스 전체 이해
- SageMaker 핵심 기능 숙달
- 첫 ML 파이프라인 구축

---

## 1주차: AWS ML 서비스 개요 + SageMaker 기초

### Day 1: AWS ML 서비스 전체 조망

**학습 목표**: AWS ML 서비스 종류와 용도 파악

| 서비스 | 용도 | 시험 출제 빈도 |
|--------|------|---------------|
| **SageMaker** | ML 전체 플랫폼 | ★★★★★ |
| **Comprehend** | NLP (감정분석, 엔티티) | ★★★★ |
| **Rekognition** | 이미지/비디오 분석 | ★★★★ |
| **Lex** | 대화형 챗봇 | ★★★ |
| **Polly** | 텍스트 → 음성 (TTS) | ★★ |
| **Transcribe** | 음성 → 텍스트 (STT) | ★★★ |
| **Translate** | 기계 번역 | ★★ |
| **Forecast** | 시계열 예측 | ★★★ |
| **Personalize** | 추천 시스템 | ★★★ |
| **Textract** | 문서 OCR | ★★★ |
| **Kendra** | 지능형 검색 | ★★ |

**실습**:
```bash
# AWS CLI 설치 확인
aws --version

# AWS 계정 설정
aws configure

# ML 서비스 리전 확인
aws sagemaker list-endpoints --region us-east-1
```

**학습 자료**:
- AWS Skill Builder: "AWS Machine Learning Foundations"
- 시간: 2시간

---

### Day 2: SageMaker 아키텍처 이해

**학습 목표**: SageMaker 구성요소 완벽 이해

```
┌─────────────────────────────────────────────────────────────┐
│                    Amazon SageMaker                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Studio    │  │  Notebook   │  │  Autopilot  │         │
│  │   (IDE)     │  │  Instance   │  │  (AutoML)   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Processing  │  │  Training   │  │   Tuning    │         │
│  │   Jobs      │  │   Jobs      │  │   Jobs      │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Endpoint   │  │   Batch     │  │   Model     │         │
│  │ (Real-time) │  │  Transform  │  │  Registry   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Model     │  │  Feature    │  │  Pipelines  │         │
│  │   Monitor   │  │   Store     │  │   (MLOps)   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

**핵심 개념**:

| 구성요소 | 설명 | 비용 발생 |
|---------|------|---------|
| **Notebook Instance** | Jupyter 환경 | 실행 중일 때 |
| **Training Job** | 모델 학습 | 학습 시간 |
| **Processing Job** | 전처리/후처리 | 처리 시간 |
| **Endpoint** | 실시간 추론 | 24/7 상시 |
| **Batch Transform** | 배치 추론 | 처리 시간 |

**학습 자료**:
- AWS 공식 문서: SageMaker Developer Guide
- 시간: 2시간

---

### Day 3: SageMaker Studio 설정 + 첫 노트북

**학습 목표**: SageMaker Studio 환경 구축

**실습 1: Studio 도메인 생성**
```
1. AWS Console → SageMaker → Studio
2. Quick setup 선택
3. IAM Role: Create new role
4. S3 버킷: Any S3 bucket
5. 생성 완료 (약 5분 소요)
```

**실습 2: 첫 노트북 생성**
```python
# SageMaker Python SDK 기본
import sagemaker
from sagemaker import get_execution_role

# 세션 및 역할 설정
session = sagemaker.Session()
role = get_execution_role()
bucket = session.default_bucket()
region = session.boto_region_name

print(f"Role: {role}")
print(f"Bucket: {bucket}")
print(f"Region: {region}")
```

**비용 주의사항**:
- Studio 사용 후 반드시 **앱 종료**
- 불필요한 노트북 인스턴스 **중지**
- 예상 비용: 일 $1-2 (ml.t3.medium 기준)

---

### Day 4: S3 + IAM for ML

**학습 목표**: ML 워크로드용 S3/IAM 설정

**S3 버킷 구조 (Best Practice)**:
```
s3://my-ml-bucket/
├── data/
│   ├── raw/              # 원본 데이터
│   ├── processed/        # 전처리된 데이터
│   └── features/         # 피처 데이터
├── models/
│   ├── training/         # 학습 산출물
│   └── artifacts/        # 모델 아티팩트
├── code/
│   └── scripts/          # 학습 스크립트
└── output/
    ├── predictions/      # 예측 결과
    └── reports/          # 리포트
```

**IAM 정책 (SageMaker 실행 역할)**:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-ml-bucket",
        "arn:aws:s3:::my-ml-bucket/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "*"
    }
  ]
}
```

**실습**: S3 버킷 생성 및 샘플 데이터 업로드
```python
import boto3

s3 = boto3.client('s3')

# 버킷 생성
bucket_name = 'my-ml-bucket-12345'  # 고유한 이름 필요
s3.create_bucket(Bucket=bucket_name)

# 데이터 업로드
s3.upload_file('train.csv', bucket_name, 'data/raw/train.csv')
```

---

### Day 5: SageMaker Built-in 알고리즘 개요

**학습 목표**: 주요 내장 알고리즘 이해

| 알고리즘 | 용도 | 입력 형식 | 시험 빈도 |
|---------|------|---------|---------|
| **XGBoost** | 분류/회귀 | CSV, LibSVM | ★★★★★ |
| **Linear Learner** | 분류/회귀 | RecordIO, CSV | ★★★★ |
| **K-Means** | 클러스터링 | RecordIO | ★★★ |
| **PCA** | 차원 축소 | RecordIO | ★★★ |
| **Factorization Machines** | 추천, 희소 데이터 | RecordIO | ★★★ |
| **BlazingText** | 텍스트 분류, Word2Vec | Text | ★★★★ |
| **Seq2Seq** | 번역, 요약 | RecordIO | ★★ |
| **DeepAR** | 시계열 예측 | JSON | ★★★★ |
| **Object Detection** | 객체 탐지 | RecordIO | ★★★ |
| **Image Classification** | 이미지 분류 | RecordIO | ★★★ |
| **Semantic Segmentation** | 이미지 분할 | RecordIO | ★★ |
| **Random Cut Forest** | 이상 탐지 | RecordIO | ★★★★ |
| **IP Insights** | IP 이상 탐지 | CSV | ★★ |

**핵심 암기 포인트**:
```
XGBoost: 정형 데이터 분류/회귀의 기본 선택
Linear Learner: 대용량 데이터, 빠른 학습
BlazingText: Word2Vec + 텍스트 분류 (fastText 기반)
DeepAR: 시계열 예측 (여러 시계열 동시 학습)
Random Cut Forest: 비지도 이상 탐지
```

---

### Day 6-7: 주말 실습 - 첫 ML 파이프라인

**프로젝트**: XGBoost로 이진 분류 모델 학습

```python
import sagemaker
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator

# 설정
role = get_execution_role()
session = sagemaker.Session()
bucket = session.default_bucket()

# XGBoost 컨테이너 이미지
container = sagemaker.image_uris.retrieve(
    framework='xgboost',
    region=session.boto_region_name,
    version='1.5-1'
)

# 하이퍼파라미터
hyperparameters = {
    'max_depth': 5,
    'eta': 0.2,
    'objective': 'binary:logistic',
    'num_round': 100,
    'eval_metric': 'auc'
}

# Estimator 생성
xgb = Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    hyperparameters=hyperparameters,
    output_path=f's3://{bucket}/models/'
)

# 데이터 입력 설정
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

print(f"Model artifact: {xgb.model_data}")
```

---

## 2주차: SageMaker 심화 + 데이터 준비

### Day 8: 데이터 형식 (RecordIO, CSV, Protobuf)

**학습 목표**: SageMaker 데이터 형식 이해

| 형식 | 장점 | 단점 | 사용 알고리즘 |
|------|------|------|-------------|
| **CSV** | 간단, 범용 | 느린 로딩 | XGBoost, Linear Learner |
| **RecordIO** | 빠른 스트리밍 | 변환 필요 | 대부분 내장 알고리즘 |
| **Protobuf** | 압축, 빠름 | 복잡한 변환 | 내장 알고리즘 |
| **JSON Lines** | 유연함 | 파싱 오버헤드 | DeepAR |
| **Parquet** | 컬럼 기반, 압축 | 일부만 지원 | Processing Job |

**RecordIO 변환 코드**:
```python
import io
import numpy as np
from sagemaker.amazon.common import write_numpy_to_dense_tensor

def convert_to_recordio(features, labels):
    buf = io.BytesIO()
    write_numpy_to_dense_tensor(buf, features, labels)
    buf.seek(0)
    return buf.read()

# 사용 예시
features = np.array([[1.0, 2.0], [3.0, 4.0]])
labels = np.array([0, 1])
recordio_data = convert_to_recordio(features, labels)
```

---

### Day 9: SageMaker Processing

**학습 목표**: Processing Job으로 전처리 자동화

```python
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker import get_execution_role

role = get_execution_role()

# SKLearn 프로세서 생성
sklearn_processor = ScriptProcessor(
    framework_version='1.0-1',
    role=role,
    instance_type='ml.m5.xlarge',
    instance_count=1,
    command=['python3']
)

# Processing Job 실행
sklearn_processor.run(
    code='preprocessing.py',
    inputs=[
        ProcessingInput(
            source='s3://bucket/data/raw/',
            destination='/opt/ml/processing/input'
        )
    ],
    outputs=[
        ProcessingOutput(
            source='/opt/ml/processing/output',
            destination='s3://bucket/data/processed/'
        )
    ]
)
```

**preprocessing.py 예시**:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# 데이터 로드
input_path = '/opt/ml/processing/input/data.csv'
df = pd.read_csv(input_path)

# 전처리
scaler = StandardScaler()
X = scaler.fit_transform(df.drop('target', axis=1))
y = df['target'].values

# Train/Test 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 저장
output_path = '/opt/ml/processing/output'
pd.DataFrame(X_train).to_csv(f'{output_path}/train.csv', index=False)
pd.DataFrame(X_test).to_csv(f'{output_path}/test.csv', index=False)
```

---

### Day 10: 하이퍼파라미터 튜닝

**학습 목표**: 자동 하이퍼파라미터 최적화

```python
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter

# 튜닝할 하이퍼파라미터 범위
hyperparameter_ranges = {
    'eta': ContinuousParameter(0.01, 0.3),
    'max_depth': IntegerParameter(3, 10),
    'min_child_weight': ContinuousParameter(1, 10),
    'subsample': ContinuousParameter(0.5, 1.0),
}

# 목표 메트릭
objective_metric_name = 'validation:auc'

# 튜너 생성
tuner = HyperparameterTuner(
    estimator=xgb,
    objective_metric_name=objective_metric_name,
    hyperparameter_ranges=hyperparameter_ranges,
    max_jobs=20,           # 총 실험 수
    max_parallel_jobs=4,   # 동시 실행 수
    strategy='Bayesian'    # 또는 'Random'
)

# 튜닝 시작
tuner.fit({
    'train': train_input,
    'validation': validation_input
})

# 최적 하이퍼파라미터
tuner.best_training_job()
```

**튜닝 전략**:
| 전략 | 설명 | 사용 시기 |
|------|------|---------|
| **Random** | 무작위 탐색 | 파라미터 영향 파악 |
| **Bayesian** | 이전 결과 기반 탐색 | 최적화 수렴 |
| **Hyperband** | 조기 종료 기반 | 대규모 탐색 |

---

### Day 11: 모델 배포 (Endpoint)

**학습 목표**: 실시간 추론 엔드포인트 생성

```python
# 모델 배포
predictor = xgb.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    serializer=sagemaker.serializers.CSVSerializer(),
    deserializer=sagemaker.deserializers.JSONDeserializer()
)

# 추론 테스트
test_data = '0.5,1.2,3.4,2.1'
response = predictor.predict(test_data)
print(f"Prediction: {response}")

# 엔드포인트 삭제 (비용 절감!)
predictor.delete_endpoint()
```

**배포 옵션**:
| 옵션 | 설명 | 사용 시기 |
|------|------|---------|
| **Real-time Endpoint** | 상시 운영 | 실시간 응답 필요 |
| **Serverless Inference** | 자동 스케일링 | 간헐적 트래픽 |
| **Batch Transform** | 대량 배치 처리 | 정기 배치 작업 |
| **Async Inference** | 비동기 처리 | 긴 추론 시간 |

---

### Day 12: Batch Transform

**학습 목표**: 대량 데이터 배치 추론

```python
from sagemaker.transformer import Transformer

# Transformer 생성
transformer = xgb.transformer(
    instance_count=1,
    instance_type='ml.m5.large',
    output_path=f's3://{bucket}/output/predictions/'
)

# 배치 추론 실행
transformer.transform(
    data=f's3://{bucket}/data/test/',
    content_type='text/csv',
    split_type='Line'
)

transformer.wait()
print(f"Predictions: {transformer.output_path}")
```

---

### Day 13-14: 주말 종합 실습

**미니 프로젝트: End-to-End ML 파이프라인**

```
1. S3에 데이터 업로드
2. Processing Job으로 전처리
3. XGBoost로 모델 학습
4. 하이퍼파라미터 튜닝
5. 최적 모델 배포
6. 추론 테스트
7. 리소스 정리
```

**체크리스트**:
- [ ] S3 버킷 구조 설정
- [ ] IAM 역할 생성
- [ ] Processing Job 실행
- [ ] Training Job 실행
- [ ] HPO 실행 (5개 job)
- [ ] Endpoint 배포
- [ ] 추론 테스트
- [ ] 리소스 삭제

---

## 1-2주차 퀴즈 (자가 점검)

### Q1. SageMaker에서 XGBoost 학습 시 권장 데이터 형식은?
<details>
<summary>정답 보기</summary>
CSV 또는 LibSVM. XGBoost는 CSV를 기본 지원하며, RecordIO 변환 없이 사용 가능.
</details>

### Q2. 실시간 추론 vs 배치 추론의 차이점은?
<details>
<summary>정답 보기</summary>
- 실시간 (Endpoint): 상시 운영, ms 단위 응답, 24/7 비용 발생
- 배치 (Transform): 대량 처리, 처리 시간만 비용 발생
</details>

### Q3. 하이퍼파라미터 튜닝에서 Bayesian vs Random 전략 차이는?
<details>
<summary>정답 보기</summary>
- Random: 무작위 탐색, 병렬화 용이
- Bayesian: 이전 결과 기반 탐색, 더 적은 실험으로 최적화 가능
</details>

### Q4. SageMaker Processing Job의 용도는?
<details>
<summary>정답 보기</summary>
데이터 전처리, 후처리, 평가 등 학습 외 컴퓨팅 작업을 위한 관리형 인프라 제공.
</details>

### Q5. DeepAR 알고리즘의 주요 용도와 입력 형식은?
<details>
<summary>정답 보기</summary>
- 용도: 시계열 예측 (여러 관련 시계열 동시 학습 가능)
- 입력: JSON Lines 형식
</details>

---

## 비용 관리 팁

| 리소스 | 시간당 비용 (예상) | 관리 방법 |
|--------|------------------|---------|
| Studio App | $0.05 | 사용 후 앱 종료 |
| Notebook (ml.t3.medium) | $0.05 | 사용 후 중지 |
| Training (ml.m5.large) | $0.12 | 작업 완료 후 자동 종료 |
| Endpoint (ml.m5.large) | $0.12 | 테스트 후 즉시 삭제 |

**일일 예상 비용**: $2-5 (적극적 학습 시)

---

## 다음 주 예고 (3-4주차)

- AWS Glue ETL
- Kinesis 실시간 스트리밍
- Athena SQL 쿼리
- Data Pipeline 구축
