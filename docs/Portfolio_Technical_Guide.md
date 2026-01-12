# AI/ML 포트폴리오 기술 문서

> **개발자**: 김다운 (백엔드 9년 → AI/ML Engineer 전환)
> **GitHub**: https://github.com/araeLaver/AI-ML
> **최종 업데이트**: 2026-01-12

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [Step 1-2: Financial Analysis](#2-step-1-2-financial-analysis)
3. [Step 3: Finance RAG API](#3-step-3-finance-rag-api)
4. [Step 4: Code Review Agent](#4-step-4-code-review-agent)
5. [Step 5: MLOps Pipeline](#5-step-5-mlops-pipeline)
6. [Step 6: Financial Fine-tuning](#6-step-6-financial-fine-tuning)
7. [포트폴리오 웹사이트](#7-포트폴리오-웹사이트)
8. [기술 스택 총정리](#8-기술-스택-총정리)

---

## 1. 프로젝트 개요

### 1.1 목적
백엔드 개발자가 AI/ML 엔지니어로 전환하기 위한 **6단계 학습 로드맵**과 **실전 프로젝트** 모음

### 1.2 학습 로드맵

```
Step 1-2: ML 기초 + LLM API
    ↓
Step 3: RAG (검색 증강 생성)
    ↓
Step 4: AI Agent (자율 에이전트)
    ↓
Step 5: MLOps (운영/배포)
    ↓
Step 6: Fine-tuning (모델 미세조정)
```

### 1.3 전체 구조

```
AI-ML/
├── financial-analysis/      # Step 1-2: ML + LLM 기초 (302KB)
├── finance-rag-api/         # Step 3: RAG 시스템 (1.6GB)
├── code-review-agent/       # Step 4: AI Agent (526MB)
├── mlops-pipeline/          # Step 5: MLOps (456KB)
├── financial-finetuning/    # Step 6: Fine-tuning (583KB)
├── portfolio/               # 포트폴리오 웹 (409MB)
├── portfolio-dashboard/     # 대시보드 (21KB)
└── docs/                    # 학습 문서 (256KB)
```

### 1.4 테스트 현황

| 프로젝트 | 테스트 수 | 상태 |
|:---|:---:|:---:|
| financial-analysis | 20 | ✅ Pass |
| finance-rag-api | 53 | ✅ Pass |
| code-review-agent | 55 | ✅ Pass |
| mlops-pipeline | 40 | ✅ Pass |
| financial-finetuning | 21 | ✅ Pass |
| **총계** | **189** | ✅ |

---

## 2. Step 1-2: Financial Analysis

### 2.1 프로젝트 소개
금융 이상거래 탐지 시스템 - ML 모델과 LLM을 결합한 분석 플랫폼

### 2.2 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │  Data Layer  │    │   ML Layer   │    │ LLM Layer │ │
│  │              │    │              │    │           │ │
│  │ • NumPy      │───▶│ • sklearn    │───▶│ • OpenAI  │ │
│  │ • Pandas     │    │ • RandomForest│   │ • Claude  │ │
│  │ • 전처리     │    │ • IsolationF │    │ • Prompts │ │
│  └──────────────┘    └──────────────┘    └───────────┘ │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 2.3 핵심 기능

| 기능 | 설명 | 기술 |
|:---|:---|:---|
| 데이터 전처리 | 금융 데이터 정제, 특성 추출 | NumPy, Pandas |
| 이상 탐지 | 비정상 거래 패턴 탐지 | Random Forest, Isolation Forest |
| LLM 분석 | 탐지 결과 자연어 해석 | OpenAI/Claude API |
| 프롬프트 기법 | 다양한 프롬프트 전략 | Zero-shot, Few-shot, CoT |

### 2.4 디렉토리 구조

```
financial-analysis/
├── app/
│   └── streamlit_app.py      # 웹 UI
├── src/
│   ├── data/                 # 데이터 처리
│   │   ├── loader.py         # 데이터 로딩
│   │   └── preprocessor.py   # 전처리
│   ├── models/               # ML 모델
│   │   ├── anomaly_detector.py
│   │   └── ensemble.py
│   └── llm/                  # LLM 연동
│       ├── openai_client.py
│       ├── claude_client.py
│       └── prompts.py
├── tests/                    # 테스트 (20개)
├── requirements.txt
└── docker-compose.yml
```

### 2.5 사용 방법

```bash
# 1. 환경 설정
cd financial-analysis
pip install -r requirements.txt

# 2. 환경 변수 설정
export OPENAI_API_KEY="your-key"
# 또는
export ANTHROPIC_API_KEY="your-key"

# 3. Streamlit 실행
streamlit run app/streamlit_app.py

# 4. Docker로 실행
docker-compose up streamlit
```

### 2.6 주요 코드 예시

```python
# 이상 탐지 모델 사용
from src.models.anomaly_detector import AnomalyDetector

detector = AnomalyDetector(model_type="isolation_forest")
detector.fit(train_data)
predictions = detector.predict(test_data)

# LLM 분석 요청
from src.llm.openai_client import analyze_transaction

result = analyze_transaction(
    transaction_data=suspicious_tx,
    prompt_type="chain_of_thought"
)
```

---

## 3. Step 3: Finance RAG API

### 3.1 프로젝트 소개
금융 문서 기반 RAG(Retrieval-Augmented Generation) Q&A 시스템 - LLM 환각(hallucination) 방지

### 3.2 아키텍처

```
┌────────────────────────────────────────────────────────────────┐
│                        Client Layer                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │  Streamlit   │    │   FastAPI    │    │  REST API    │     │
│  │   (Demo)     │    │   Server     │    │  Endpoints   │     │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘     │
└─────────┼───────────────────┼───────────────────┼──────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
┌────────────────────────────────────────────────────────────────┐
│                        RAG Service                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Query Processing                       │  │
│  │  1. Query → Embedding → Vector Search → Retrieved Docs   │  │
│  │  2. Retrieved Docs + Query → LLM → Answer                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │  Embedding   │    │   ChromaDB   │    │    Ollama    │     │
│  │   Service    │    │  (VectorDB)  │    │    (LLM)     │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
└────────────────────────────────────────────────────────────────┘
```

### 3.3 핵심 기능

| 기능 | 설명 | 기술 |
|:---|:---|:---|
| 문서 업로드 | PDF/TXT 파일 처리 | PyPDF, 청킹 |
| 벡터 저장 | 문서 임베딩 및 저장 | ChromaDB |
| 시맨틱 검색 | 의미 기반 유사 문서 검색 | Embedding, Cosine Similarity |
| RAG 응답 | 검색 결과 기반 답변 생성 | Ollama (llama3.2) |
| 실시간 차트 | 주식/환율 데이터 시각화 | yfinance, Plotly |

### 3.4 디렉토리 구조

```
finance-rag-api/
├── app/
│   └── streamlit_app.py      # 데모 UI
├── src/
│   ├── main.py               # FastAPI 엔트리포인트
│   ├── api/
│   │   └── routes.py         # API 라우트
│   ├── rag/
│   │   ├── embeddings.py     # 임베딩 서비스
│   │   ├── vectorstore.py    # ChromaDB 연동
│   │   ├── retriever.py      # 문서 검색
│   │   ├── generator.py      # 답변 생성
│   │   ├── realtime_data.py  # 실시간 시세
│   │   └── charts.py         # 차트 생성
│   └── utils/
│       └── document_loader.py
├── data/
│   └── chroma_db/            # 벡터 DB 저장소
├── tests/                    # 테스트 (53개)
└── requirements.txt
```

### 3.5 사용 방법

```bash
# 1. 환경 설정
cd finance-rag-api
pip install -r requirements.txt

# 2. Ollama 설치 및 모델 다운로드
ollama pull llama3.2

# 3. FastAPI 서버 실행
uvicorn src.main:app --reload --port 8000

# 4. Streamlit 데모 실행 (별도 터미널)
streamlit run app/streamlit_app.py
```

### 3.6 API 엔드포인트

| Method | Endpoint | 설명 |
|:---:|:---|:---|
| POST | `/api/documents/upload` | 문서 업로드 |
| GET | `/api/documents` | 문서 목록 조회 |
| POST | `/api/query` | RAG 질의 |
| GET | `/api/stock/{symbol}` | 주식 시세 조회 |
| GET | `/api/exchange/{pair}` | 환율 조회 |

### 3.7 주요 코드 예시

```python
# RAG 질의 예시
import requests

response = requests.post(
    "http://localhost:8000/api/query",
    json={
        "question": "삼성전자의 2024년 실적은?",
        "top_k": 3
    }
)

result = response.json()
print(result["answer"])
print(result["sources"])      # 출처 문서
print(result["confidence"])   # 신뢰도 점수
```

---

## 4. Step 4: Code Review Agent

### 4.1 프로젝트 소개
AI 기반 코드 리뷰 에이전트 - ReAct 패턴으로 자율적 분석 수행

### 4.2 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Sources                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  GitHub PR   │    │  API Request │    │  CLI Input   │      │
│  │   Webhook    │    │              │    │              │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
└─────────┼───────────────────┼───────────────────┼───────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Orchestrator                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  ReAct Loop (Reasoning + Acting)          │  │
│  │                                                            │  │
│  │   Thought → Action → Observation → Thought → ... → Answer │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Specialist Agents                      │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │   │
│  │  │Security │  │Perform- │  │  Style  │  │Synthesis│    │   │
│  │  │ Agent   │  │ance     │  │  Agent  │  │  Agent  │    │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘    │   │
│  └───────┼────────────┼────────────┼────────────┼──────────┘   │
│          ▼            ▼            ▼            ▼               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                       Tools                               │   │
│  │  • AST Parser    • Complexity Analyzer   • Security Scan │   │
│  │  • Code Metrics  • Style Checker         • Git Diff      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         LLM Backend                              │
│  ┌──────────────┐    ┌──────────────┐                          │
│  │   OpenAI     │    │    Ollama    │                          │
│  │  (GPT-4o)    │    │  (llama3.2)  │                          │
│  └──────────────┘    └──────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 핵심 기능

| 기능 | 설명 | 기술 |
|:---|:---|:---|
| 보안 분석 | SQL Injection, XSS 등 취약점 탐지 | AST, 패턴 매칭 |
| 성능 분석 | 복잡도, 메모리 누수 탐지 | Cyclomatic Complexity |
| 스타일 분석 | 코딩 컨벤션 검사 | Linter 통합 |
| 종합 리포트 | 전체 분석 결과 요약 | LLM Synthesis |
| GitHub 연동 | PR 자동 리뷰 코멘트 | Webhook, GitHub API |

### 4.4 디렉토리 구조

```
code-review-agent/
├── main.py                   # FastAPI 엔트리포인트
├── api/
│   └── webhook.py            # GitHub Webhook 처리
├── agents/
│   ├── __init__.py           # ReviewOrchestrator
│   ├── security_agent.py     # 보안 분석 에이전트
│   ├── performance_agent.py  # 성능 분석 에이전트
│   ├── style_agent.py        # 스타일 분석 에이전트
│   └── synthesis_agent.py    # 종합 에이전트
├── tools/
│   ├── ast_parser.py         # AST 분석 도구
│   ├── complexity.py         # 복잡도 분석
│   ├── security_scanner.py   # 보안 스캐너
│   └── code_metrics.py       # 코드 메트릭
├── tests/                    # 테스트 (55개)
└── requirements.txt
```

### 4.5 사용 방법

```bash
# 1. 환경 설정
cd code-review-agent
pip install -r requirements.txt

# 2. 환경 변수 설정
export OPENAI_API_KEY="your-key"
# 또는
export OLLAMA_URL="http://localhost:11434"

# 3. 서버 실행
uvicorn main:app --reload --port 8001

# 4. GitHub Webhook 설정 (선택)
# Repository Settings → Webhooks → Add webhook
# Payload URL: https://your-server/api/webhook/github
# Events: Pull requests
```

### 4.6 API 엔드포인트

| Method | Endpoint | 설명 |
|:---:|:---|:---|
| GET | `/` | 서비스 정보 |
| GET | `/health` | 헬스 체크 |
| POST | `/api/quick-check` | 빠른 코드 체크 (LLM 미사용) |
| POST | `/api/review` | 전체 코드 리뷰 |
| POST | `/api/review/async` | 비동기 코드 리뷰 |
| GET | `/api/review/{job_id}` | 리뷰 결과 조회 |
| POST | `/api/webhook/github` | GitHub Webhook |

### 4.7 주요 코드 예시

```python
# 코드 리뷰 요청
import requests

code = """
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
"""

response = requests.post(
    "http://localhost:8001/api/review",
    json={
        "code": code,
        "language": "python",
        "check_types": ["security", "performance", "style"]
    }
)

result = response.json()
print(result["security"])      # 보안 이슈 (SQL Injection 탐지)
print(result["performance"])   # 성능 이슈
print(result["final_report"])  # 종합 리포트
```

---

## 5. Step 5: MLOps Pipeline

### 5.1 프로젝트 소개
ML 모델의 전체 생명주기 관리 - 데이터 버전관리부터 배포까지

### 5.2 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                      MLOps Pipeline                              │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │   Data   │──▶│  Train   │──▶│ Evaluate │──▶│  Deploy  │    │
│  │ Version  │   │  Model   │   │  Model   │   │  Model   │    │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘    │
│       │              │              │              │            │
│       ▼              ▼              ▼              ▼            │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │   DVC    │   │  MLflow  │   │  MLflow  │   │  Docker  │    │
│  │ Storage  │   │ Tracking │   │ Registry │   │   K8s    │    │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CI/CD Pipeline                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Push → Test → Train → Evaluate → Register → Deploy        │ │
│  │                                                             │ │
│  │  GitHub Actions / Jenkins                                   │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 핵심 기능

| 기능 | 설명 | 기술 |
|:---|:---|:---|
| 데이터 버전관리 | 학습 데이터 버전 추적 | DVC |
| 실험 추적 | 하이퍼파라미터, 메트릭 기록 | MLflow Tracking |
| 모델 레지스트리 | 모델 버전 관리 | MLflow Registry |
| 자동화 파이프라인 | 학습-평가-배포 자동화 | GitHub Actions |
| 컨테이너화 | 모델 서빙 환경 표준화 | Docker |

### 5.4 디렉토리 구조

```
mlops-pipeline/
├── src/
│   ├── data/
│   │   ├── prepare.py        # 데이터 준비
│   │   └── validate.py       # 데이터 검증
│   ├── train/
│   │   ├── train.py          # 모델 학습
│   │   └── config.py         # 학습 설정
│   ├── evaluate/
│   │   └── evaluate.py       # 모델 평가
│   └── serve/
│       └── predict.py        # 추론 서비스
├── pipelines/
│   ├── training_pipeline.py  # 학습 파이프라인
│   └── inference_pipeline.py # 추론 파이프라인
├── configs/
│   └── config.yaml           # 설정 파일
├── .dvc/                     # DVC 설정
├── .github/
│   └── workflows/
│       └── mlops.yml         # CI/CD 워크플로우
├── docker/
│   └── Dockerfile            # 컨테이너 설정
├── tests/                    # 테스트 (40개)
├── dvc.yaml                  # DVC 파이프라인
└── requirements.txt
```

### 5.5 사용 방법

```bash
# 1. 환경 설정
cd mlops-pipeline
pip install -r requirements.txt

# 2. DVC 초기화 (이미 설정됨)
dvc init

# 3. 데이터 다운로드
dvc pull

# 4. MLflow 서버 실행
mlflow server --host 0.0.0.0 --port 5000

# 5. 학습 파이프라인 실행
dvc repro

# 6. 또는 Python으로 직접 실행
python src/train/train.py
```

### 5.6 DVC 파이프라인

```yaml
# dvc.yaml
stages:
  prepare:
    cmd: python src/data/prepare.py
    deps:
      - src/data/prepare.py
      - data/raw
    outs:
      - data/processed

  train:
    cmd: python src/train/train.py
    deps:
      - src/train/train.py
      - data/processed
    params:
      - train.epochs
      - train.learning_rate
    outs:
      - models/model.pkl
    metrics:
      - metrics.json:
          cache: false

  evaluate:
    cmd: python src/evaluate/evaluate.py
    deps:
      - src/evaluate/evaluate.py
      - models/model.pkl
      - data/processed
    metrics:
      - evaluation.json:
          cache: false
```

### 5.7 주요 코드 예시

```python
# MLflow 실험 추적
import mlflow

with mlflow.start_run():
    # 하이퍼파라미터 로깅
    mlflow.log_params({
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 32
    })

    # 모델 학습
    model = train_model(X_train, y_train)

    # 메트릭 로깅
    mlflow.log_metrics({
        "accuracy": 0.95,
        "f1_score": 0.93
    })

    # 모델 저장
    mlflow.sklearn.log_model(model, "model")
```

---

## 6. Step 6: Financial Fine-tuning

### 6.1 프로젝트 소개
금융 도메인 특화 LLM Fine-tuning - LoRA/QLoRA를 활용한 효율적 미세조정

### 6.2 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                    Fine-tuning Pipeline                          │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Dataset    │    │   Training   │    │  Inference   │      │
│  │  Preparation │───▶│   Process    │───▶│   Engine     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ • Instruction│    │ • LoRA       │    │ • Model Load │      │
│  │ • Formatting │    │ • QLoRA      │    │ • Streaming  │      │
│  │ • Augment    │    │ • PEFT       │    │ • Batch      │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Base Models                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Llama 2    │    │    Mistral   │    │  Phi-2       │      │
│  │   7B/13B     │    │     7B       │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 핵심 기능

| 기능 | 설명 | 기술 |
|:---|:---|:---|
| 데이터셋 준비 | Instruction 형식 변환 | Alpaca Format |
| LoRA 학습 | 저랭크 어댑터 학습 | PEFT, LoRA |
| QLoRA 학습 | 4bit 양자화 + LoRA | BitsAndBytes |
| 모델 병합 | 어댑터 → 전체 모델 | merge_and_unload |
| 추론 엔진 | 스트리밍, 배치 추론 | Transformers |

### 6.4 디렉토리 구조

```
financial-finetuning/
├── src/
│   ├── data/
│   │   ├── instructions.py   # 금융 Instruction 데이터
│   │   ├── dataset.py        # 데이터셋 클래스
│   │   └── augmentation.py   # 데이터 증강
│   ├── training/
│   │   ├── lora_trainer.py   # LoRA 트레이너
│   │   ├── qlora_trainer.py  # QLoRA 트레이너
│   │   └── config.py         # 학습 설정
│   ├── inference/
│   │   ├── inference_engine.py  # 추론 엔진
│   │   └── server.py         # FastAPI 서버
│   └── utils/
│       └── model_utils.py    # 모델 유틸리티
├── configs/
│   ├── lora_config.yaml      # LoRA 설정
│   └── training_config.yaml  # 학습 설정
├── notebooks/
│   └── fine_tuning_demo.ipynb
├── tests/                    # 테스트 (21개)
└── requirements.txt
```

### 6.5 사용 방법

```bash
# 1. 환경 설정
cd financial-finetuning
pip install -r requirements.txt

# 2. GPU 확인 (권장)
python -c "import torch; print(torch.cuda.is_available())"

# 3. 데이터셋 준비
python src/data/dataset.py

# 4. LoRA Fine-tuning 실행
python src/training/lora_trainer.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --output_dir "./outputs/lora" \
    --epochs 3

# 5. 추론 서버 실행
uvicorn src.inference.server:app --port 8002
```

### 6.6 LoRA 설정 예시

```yaml
# configs/lora_config.yaml
lora:
  r: 16                    # LoRA rank
  lora_alpha: 32           # Alpha parameter
  lora_dropout: 0.05       # Dropout
  target_modules:          # 적용 레이어
    - q_proj
    - v_proj
    - k_proj
    - o_proj

training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  num_train_epochs: 3
  learning_rate: 2e-4
  fp16: true

quantization:             # QLoRA용
  load_in_4bit: true
  bnb_4bit_compute_dtype: float16
  bnb_4bit_quant_type: nf4
```

### 6.7 주요 코드 예시

```python
# LoRA Fine-tuning
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

# 베이스 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,  # QLoRA
    device_map="auto"
)

# LoRA 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

# PEFT 모델 생성
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 학습 파라미터 확인

# 학습
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args
)
trainer.train()

# 추론
from src.inference.inference_engine import FinancialLLMInference

inference = FinancialLLMInference(model_path="./outputs/lora")
response = inference.generate(
    instruction="삼성전자 주가 전망을 분석해주세요.",
    input_text="현재가: 71,500원, PER: 13.5"
)
```

---

## 7. 포트폴리오 웹사이트

### 7.1 프로젝트 소개
Next.js 기반 포트폴리오 웹사이트 - 프로젝트 소개 및 기술 스택 시각화

### 7.2 기술 스택

| 분류 | 기술 |
|:---|:---|
| Framework | Next.js 16 |
| UI | React 18, TailwindCSS |
| Animation | Framer Motion |
| Icons | Lucide React |
| Language | TypeScript |

### 7.3 디렉토리 구조

```
portfolio/
├── app/
│   ├── layout.tsx            # 레이아웃
│   ├── page.tsx              # 메인 페이지
│   └── globals.css           # 전역 스타일
├── components/
│   ├── Hero.tsx              # 히어로 섹션
│   ├── Projects.tsx          # 프로젝트 섹션
│   ├── Skills.tsx            # 기술 스택
│   └── Contact.tsx           # 연락처
├── public/                   # 정적 파일
├── package.json
├── tailwind.config.js
└── tsconfig.json
```

### 7.4 사용 방법

```bash
# 1. 의존성 설치
cd portfolio
npm install

# 2. 개발 서버 실행
npm run dev

# 3. 빌드
npm run build

# 4. 프로덕션 실행
npm start
```

### 7.5 배포 (Vercel)

```bash
# Vercel CLI 설치
npm i -g vercel

# 배포
vercel

# 프로덕션 배포
vercel --prod
```

---

## 8. 기술 스택 총정리

### 8.1 언어 & 프레임워크

| 분류 | 기술 |
|:---|:---|
| **Backend** | Python 3.13, FastAPI, Streamlit |
| **Frontend** | Next.js 16, React 18, TypeScript |
| **ML/DL** | NumPy, Pandas, scikit-learn, PyTorch |
| **LLM** | OpenAI API, Claude API, Ollama, LangChain |

### 8.2 AI/ML 특화

| 분류 | 기술 |
|:---|:---|
| **RAG** | ChromaDB, FAISS, Embeddings |
| **Agent** | LangChain, ReAct Pattern |
| **Fine-tuning** | LoRA, QLoRA, PEFT, Transformers |
| **MLOps** | DVC, MLflow, Docker |

### 8.3 인프라 & DevOps

| 분류 | 기술 |
|:---|:---|
| **Container** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions |
| **Version Control** | Git, DVC |
| **Deployment** | Vercel (Frontend) |

### 8.4 테스트 & 품질

| 분류 | 기술 |
|:---|:---|
| **Testing** | pytest (189개 테스트) |
| **Coverage** | pytest-cov |
| **Linting** | Black, isort, flake8 |

---

## 부록: 빠른 실행 가이드

### 전체 프로젝트 테스트

```bash
# 모든 프로젝트 테스트 실행
cd /c/Develop/workspace/11.AI-ML

# financial-analysis
cd financial-analysis && pytest && cd ..

# finance-rag-api
cd finance-rag-api && pytest && cd ..

# code-review-agent
cd code-review-agent && pytest && cd ..

# mlops-pipeline
cd mlops-pipeline && pytest && cd ..

# financial-finetuning
cd financial-finetuning && pytest && cd ..
```

### Docker로 전체 실행

```bash
# 개별 프로젝트
cd financial-analysis && docker-compose up -d
cd finance-rag-api && docker-compose up -d
cd code-review-agent && docker-compose up -d
```

---

## 연락처

- **GitHub**: https://github.com/araeLaver
- **Email**: (이메일 추가)
- **LinkedIn**: (LinkedIn 추가)

---

*이 문서는 2026-01-12에 마지막으로 업데이트되었습니다.*
