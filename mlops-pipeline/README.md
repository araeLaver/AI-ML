# MLOps Pipeline - Fraud Detection

End-to-End MLOps 파이프라인 프로젝트: 이상 거래 탐지 시스템

## Overview

이 프로젝트는 ML 모델의 전체 생명주기(학습 → 배포 → 모니터링)를 다루는 MLOps 파이프라인입니다.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          MLOps Pipeline                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Data Pipeline        Training Pipeline       Serving Pipeline         │
│   ┌──────────┐        ┌──────────────┐        ┌──────────────┐         │
│   │ Ingestion│   →    │   Training   │   →    │   FastAPI    │         │
│   │ Validation│        │   Evaluation │        │   Prediction │         │
│   │ Preprocess│        │   MLflow     │        │   Batch API  │         │
│   └──────────┘        └──────────────┘        └──────────────┘         │
│         │                    │                       │                  │
│         ↓                    ↓                       ↓                  │
│   ┌─────────────────────────────────────────────────────────────┐      │
│   │                    Monitoring Layer                          │      │
│   │   Prometheus  │  Grafana  │  Drift Detection  │  Alerting   │      │
│   └─────────────────────────────────────────────────────────────┘      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Features

| 기능 | 설명 |
|------|------|
| **데이터 파이프라인** | 데이터 수집, 검증, 전처리 자동화 |
| **모델 학습** | Random Forest, Gradient Boosting, Logistic Regression |
| **실험 관리** | MLflow로 하이퍼파라미터, 메트릭, 모델 버전 관리 |
| **모델 서빙** | FastAPI 기반 실시간/배치 예측 API |
| **모니터링** | Prometheus 메트릭, 데이터/모델 드리프트 감지 |
| **CI/CD** | GitHub Actions 자동 테스트/배포 파이프라인 |
| **컨테이너화** | Docker, Kubernetes 배포 지원 |

## Tech Stack

- **ML Framework**: scikit-learn, pandas, numpy
- **Experiment Tracking**: MLflow
- **API Framework**: FastAPI, Uvicorn
- **Monitoring**: Prometheus, Grafana, Evidently
- **Containerization**: Docker, Kubernetes
- **CI/CD**: GitHub Actions
- **UI**: Streamlit

## Quick Start

### 1. 환경 설정

```bash
# 저장소 클론
cd mlops-pipeline

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 생성 및 모델 학습

```bash
# 샘플 데이터 생성
python scripts/generate_data.py

# 모델 학습
python scripts/train.py

# 모델 평가
python scripts/evaluate.py
```

### 3. API 서버 실행

```bash
# FastAPI 서버 실행
uvicorn src.serving.api:app --reload --port 8000

# API 문서: http://localhost:8000/docs
```

### 4. Streamlit UI 실행

```bash
streamlit run app/streamlit_app.py
```

## Docker 실행

```bash
# 전체 스택 실행 (API + MLflow + Prometheus + Grafana)
docker-compose up -d

# 서비스 접속
# - API: http://localhost:8000
# - Streamlit: http://localhost:8501
# - MLflow: http://localhost:5000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin123)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | 헬스 체크 |
| `/model/info` | GET | 모델 정보 조회 |
| `/predict` | POST | 단일 거래 예측 |
| `/predict/batch` | POST | 배치 예측 |
| `/model/threshold` | POST | 임계값 업데이트 |
| `/metrics` | GET | Prometheus 메트릭 |

### 예측 요청 예시

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 150000,
    "time_hour": 14,
    "location_distance": 5.0,
    "previous_avg_amount": 100000
  }'
```

### 응답 예시

```json
{
  "is_fraud": false,
  "probability": 0.0823,
  "risk_level": "MINIMAL",
  "threshold": 0.5,
  "model_version": "random_forest",
  "latency_ms": 12.34
}
```

## Project Structure

```
mlops-pipeline/
├── src/
│   ├── data/               # 데이터 파이프라인
│   │   ├── ingestion.py    # 데이터 수집
│   │   ├── validation.py   # 데이터 검증
│   │   └── preprocessing.py# 전처리
│   ├── training/           # 학습 파이프라인
│   │   ├── train.py        # 모델 학습
│   │   └── evaluate.py     # 모델 평가
│   ├── serving/            # 서빙 파이프라인
│   │   ├── api.py          # FastAPI 서버
│   │   └── predictor.py    # 예측 로직
│   └── monitoring/         # 모니터링
│       ├── metrics.py      # Prometheus 메트릭
│       └── drift.py        # 드리프트 감지
├── app/
│   └── streamlit_app.py    # 웹 데모 UI
├── k8s/                    # Kubernetes 설정
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml
├── tests/                  # 테스트
├── .github/workflows/      # CI/CD
├── docker-compose.yml
├── Dockerfile
├── prometheus.yml
└── requirements.txt
```

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 0.98+ |
| Precision | 0.85+ |
| Recall | 0.80+ |
| ROC AUC | 0.95+ |
| F1 Score | 0.82+ |

## Monitoring

### Grafana Dashboard

Docker Compose 실행 시 자동으로 프로비저닝되는 대시보드:

- **URL**: http://localhost:3000
- **계정**: admin / admin123
- **대시보드**: MLOps - Fraud Detection

| 패널 | 설명 |
|------|------|
| Total Predictions | 총 예측 요청 수 |
| Fraud Rate | 최근 1000건 기준 이상거래 비율 |
| P95 Latency | 95 백분위 응답 시간 |
| Active Connections | 현재 활성 연결 수 |
| Prediction Rate | 예측 결과별 초당 요청 |
| Latency Percentiles | P50/P90/P99 응답 시간 추이 |
| API Request Rate | 엔드포인트별 요청률 |
| HTTP Status Codes | HTTP 상태 코드 분포 |
| Latency Heatmap | 레이턴시 분포 히트맵 |

### Prometheus Metrics

- `fraud_detection_prediction_total`: 총 예측 수
- `fraud_detection_prediction_latency_seconds`: 예측 지연 시간
- `fraud_detection_prediction_probability`: 확률 분포
- `fraud_detection_fraud_rate`: 실시간 이상 거래 비율
- `fraud_detection_api_requests_total`: API 요청 수
- `fraud_detection_active_connections`: 활성 연결 수

### Alert Rules

| Alert | 조건 | 심각도 |
|-------|------|--------|
| HighFraudRate | Fraud Rate > 30% (5분간) | Critical |
| HighPredictionLatency | P95 > 500ms (5분간) | Warning |
| HighAPIErrorRate | 5xx 에러율 > 5% (5분간) | Critical |
| APIDown | API 1분간 미응답 | Critical |
| NoPredictions | 10분간 예측 없음 | Warning |
| PredictionDistributionShift | 확률 분포 급변 | Warning |

### Drift Detection

- **데이터 드리프트**: PSI (Population Stability Index) 기반
- **예측 드리프트**: KS Test 기반
- **성능 드리프트**: 메트릭 변화 감지

## CI/CD Pipeline

```
Push to main
    │
    ├── Test Job
    │   ├── Linting (ruff)
    │   └── Unit Tests (pytest)
    │
    ├── Build Job
    │   └── Docker Image → GHCR
    │
    ├── Deploy Staging
    │   └── Smoke Tests
    │
    └── Deploy Production
        └── Health Check
```

## Testing

```bash
# 전체 테스트
pytest tests/ -v

# 커버리지 포함
pytest tests/ -v --cov=src --cov-report=html
```

## License

MIT License
