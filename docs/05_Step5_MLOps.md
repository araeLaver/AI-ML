# Step 5: MLOps + 모델 서빙 (2-3개월)

## 목표
> ML 모델 배포/운영/모니터링 역량 (김다운님 핵심 차별화 영역)

## 왜 MLOps가 김다운님에게 최적인가?

```
┌─────────────────────────────────────────────────────────────────────┐
│                  김다운님 기존 경험 → MLOps 매핑                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  데이터 파이프라인 (헥토데이터 3년)  →  ML 파이프라인                │
│  배치 처리 시스템                    →  모델 학습 파이프라인          │
│  실시간 모니터링                     →  모델 성능 모니터링            │
│  Docker/Linux                        →  컨테이너화/오케스트레이션     │
│  REST API 설계                       →  Model Serving API            │
│  ESB/시스템 연계                     →  ML 시스템 통합                │
│                                                                     │
│  기존 역량의 90%가 그대로 활용됨!                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## MLOps 전체 파이프라인

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           MLOps Lifecycle                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│    데이터 수집  →  데이터 검증  →  피처 엔지니어링  →  모델 학습           │
│         │              │               │                │                 │
│         ↓              ↓               ↓                ↓                 │
│    [Data Lake]   [Great Exp.]    [Feature Store]   [MLflow]              │
│                                                         │                 │
│                                                         ↓                 │
│    모니터링  ←  A/B 테스트  ←  모델 배포  ←  모델 레지스트리              │
│         │           │            │              │                         │
│         ↓           ↓            ↓              ↓                         │
│    [Grafana]   [Platform]   [K8s/Docker]   [MLflow Registry]             │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 학습 순서

### Week 1-2: 모델 서빙 기초 (FastAPI)

**김다운님의 REST API 경험이 직접 활용되는 영역**

#### FastAPI ML 서빙

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ML Model API")

# 모델 로드
model = joblib.load("models/fraud_detector.pkl")

class TransactionInput(BaseModel):
    amount: float
    time_hour: int
    location_distance: float
    previous_avg_amount: float

class PredictionOutput(BaseModel):
    is_fraud: bool
    probability: float
    risk_level: str

@app.post("/predict", response_model=PredictionOutput)
async def predict(transaction: TransactionInput):
    """이상 거래 예측"""
    try:
        # 입력 데이터 변환
        features = np.array([[
            transaction.amount,
            transaction.time_hour,
            transaction.location_distance,
            transaction.previous_avg_amount
        ]])

        # 예측
        probability = model.predict_proba(features)[0][1]
        is_fraud = probability > 0.5

        # 리스크 레벨 결정
        if probability > 0.8:
            risk_level = "HIGH"
        elif probability > 0.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return PredictionOutput(
            is_fraud=is_fraud,
            probability=round(probability, 4),
            risk_level=risk_level
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy", "model_version": "1.0.0"}

# 배치 예측 엔드포인트
class BatchInput(BaseModel):
    transactions: list[TransactionInput]

@app.post("/predict/batch")
async def predict_batch(batch: BatchInput):
    """배치 예측"""
    results = []
    for tx in batch.transactions:
        result = await predict(tx)
        results.append(result)
    return {"predictions": results}
```

#### 비동기 처리 (대용량 요청)

```python
import asyncio
from fastapi import BackgroundTasks

# 비동기 모델 추론
async def async_predict(features):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, model.predict_proba, features)
    return result

# 백그라운드 작업 (재학습 등)
@app.post("/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(retrain_model)
    return {"message": "Retraining started in background"}

async def retrain_model():
    # 재학습 로직
    pass
```

---

### Week 3-4: MLflow (실험 관리)

#### MLflow 기본 사용

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# MLflow 서버 연결
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("fraud-detection")

# 실험 실행
with mlflow.start_run(run_name="rf_baseline"):
    # 하이퍼파라미터 로깅
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5
    }
    mlflow.log_params(params)

    # 모델 학습
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # 예측 및 평가
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 메트릭 로깅
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
    mlflow.log_metrics(metrics)

    # 모델 저장
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="fraud-detector"
    )

    # 아티팩트 로깅 (그래프, 데이터 등)
    mlflow.log_artifact("confusion_matrix.png")

    print(f"Run ID: {mlflow.active_run().info.run_id}")
```

#### 모델 레지스트리

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 모델 버전 관리
model_name = "fraud-detector"

# 스테이징으로 전환
client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Staging"
)

# 프로덕션으로 전환
client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Production"
)

# 프로덕션 모델 로드
model_uri = f"models:/{model_name}/Production"
model = mlflow.sklearn.load_model(model_uri)
```

---

### Week 5-6: Docker + Kubernetes

**김다운님의 Docker 경험 확장**

#### ML 모델 Dockerize

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 모델 및 코드 복사
COPY models/ ./models/
COPY app/ ./app/

# 환경 변수
ENV MODEL_PATH=/app/models/fraud_detector.pkl
ENV PORT=8000

# 포트 노출
EXPOSE 8000

# 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/fraud_detector.pkl
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    volumes:
      - ./models:/app/models
    depends_on:
      - mlflow

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.9.0
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0 --port 5000
    volumes:
      - ./mlruns:/mlflow/mlruns

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

#### Kubernetes 배포

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-detector
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fraud-detector
  template:
    metadata:
      labels:
        app: fraud-detector
    spec:
      containers:
      - name: fraud-detector
        image: your-registry/fraud-detector:v1.0.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 3
---
apiVersion: v1
kind: Service
metadata:
  name: fraud-detector-service
spec:
  selector:
    app: fraud-detector
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

### Week 7-8: 클라우드 AI 서비스

#### AWS SageMaker

```python
import sagemaker
from sagemaker.sklearn import SKLearn

# SageMaker 세션
session = sagemaker.Session()
role = "arn:aws:iam::xxx:role/SageMakerRole"

# 학습 스크립트
sklearn_estimator = SKLearn(
    entry_point="train.py",
    source_dir="src",
    role=role,
    instance_type="ml.m5.large",
    framework_version="1.2-1",
    py_version="py3",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 10
    }
)

# 학습 실행
sklearn_estimator.fit({"train": "s3://bucket/train", "test": "s3://bucket/test"})

# 엔드포인트 배포
predictor = sklearn_estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium"
)

# 추론
result = predictor.predict(data)
```

#### GCP Vertex AI

```python
from google.cloud import aiplatform

# 초기화
aiplatform.init(project="your-project", location="us-central1")

# 모델 업로드
model = aiplatform.Model.upload(
    display_name="fraud-detector",
    artifact_uri="gs://bucket/model",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest"
)

# 엔드포인트 배포
endpoint = model.deploy(
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=5
)

# 예측
response = endpoint.predict(instances=[{"features": [1.0, 2.0, 3.0]}])
```

---

### Week 9-10: 모니터링 및 CI/CD

#### 모델 모니터링

```python
from prometheus_client import Counter, Histogram, start_http_server
import time

# 메트릭 정의
PREDICTION_COUNT = Counter(
    'prediction_total',
    'Total predictions',
    ['model_version', 'result']
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction latency'
)

PREDICTION_SCORE = Histogram(
    'prediction_score',
    'Prediction probability distribution',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# 예측 시 메트릭 수집
@PREDICTION_LATENCY.time()
async def predict_with_metrics(features):
    result = model.predict_proba(features)
    probability = result[0][1]

    # 메트릭 기록
    PREDICTION_COUNT.labels(
        model_version="1.0.0",
        result="fraud" if probability > 0.5 else "normal"
    ).inc()

    PREDICTION_SCORE.observe(probability)

    return probability
```

#### 드리프트 감지

```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def detect_drift(reference_data, current_data):
    """데이터 드리프트 감지"""
    column_mapping = ColumnMapping(
        target='is_fraud',
        numerical_features=['amount', 'time_hour', 'location_distance']
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )

    # 드리프트 점수 추출
    drift_score = report.as_dict()['metrics'][0]['result']['dataset_drift']

    if drift_score:
        # 알림 발송 및 재학습 트리거
        trigger_retrain()

    return report
```

#### CI/CD for ML (GitHub Actions)

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 0 * * 0'  # 매주 일요일 재학습

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/

  train:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Train model
        run: python scripts/train.py
      - name: Evaluate model
        run: python scripts/evaluate.py
      - name: Upload model
        if: success()
        run: python scripts/upload_to_registry.py

  deploy:
    needs: train
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging
        run: kubectl apply -f k8s/staging/
      - name: Run integration tests
        run: pytest tests/integration/
      - name: Deploy to production
        if: success()
        run: kubectl apply -f k8s/production/
```

---

## 실습 프로젝트

### 프로젝트: End-to-End MLOps 파이프라인

**김다운님의 세 번째 포트폴리오 프로젝트**

```
mlops_pipeline/
├── src/
│   ├── data/
│   │   ├── ingestion.py      # 데이터 수집
│   │   ├── validation.py     # 데이터 검증
│   │   └── preprocessing.py  # 전처리
│   ├── training/
│   │   ├── train.py          # 학습
│   │   └── evaluate.py       # 평가
│   ├── serving/
│   │   ├── api.py            # FastAPI
│   │   └── predictor.py      # 예측 로직
│   └── monitoring/
│       ├── metrics.py        # 메트릭 수집
│       └── drift.py          # 드리프트 감지
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml             # 오토스케일링
├── mlflow/
├── tests/
├── .github/workflows/
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## 체크리스트

### 모델 서빙
- [ ] FastAPI ML API 구현
- [ ] 배치 예측 엔드포인트
- [ ] 비동기 처리

### MLflow
- [ ] 실험 추적
- [ ] 모델 레지스트리
- [ ] 모델 버전 관리

### 컨테이너화
- [ ] Docker 이미지 빌드
- [ ] docker-compose 환경
- [ ] K8s 배포

### 클라우드 AI
- [ ] AWS SageMaker 또는 GCP Vertex AI
- [ ] 엔드포인트 배포

### 모니터링
- [ ] Prometheus 메트릭
- [ ] Grafana 대시보드
- [ ] 드리프트 감지

### CI/CD
- [ ] GitHub Actions 파이프라인
- [ ] 자동 테스트
- [ ] 자동 배포

---

## 다음 단계
Step 5 완료 후 → **Step 6: Fine-tuning + 고급 최적화** 로 진행
