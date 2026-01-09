# AI/ML 학습 로드맵 & 포트폴리오

백엔드 개발자의 AI/ML 전환을 위한 6단계 학습 로드맵과 실전 프로젝트입니다.

## 프로젝트 구조

```
AI-ML/
├── financial-analysis/     # Step 1-2: ML + LLM 기초
├── finance-rag-api/        # Step 3: RAG 시스템
├── code-review-agent/      # Step 4: AI Agent
├── mlops-pipeline/         # Step 5: MLOps 파이프라인
├── financial-finetuning/   # Step 6: LLM Fine-tuning
├── docs/                   # 학습 가이드
└── portfolio/              # 포트폴리오 웹사이트
```

## 학습 로드맵 (6단계)

| Step | 프로젝트 | 핵심 기술 | 상태 |
|------|---------|----------|------|
| 1-2 | [financial-analysis](./financial-analysis/) | NumPy, Pandas, scikit-learn, LLM API | 완료 |
| 3 | [finance-rag-api](./finance-rag-api/) | RAG, ChromaDB, Ollama | 완료 |
| 4 | [code-review-agent](./code-review-agent/) | AI Agent, ReAct, LangGraph | 완료 |
| 5 | [mlops-pipeline](./mlops-pipeline/) | DVC, MLflow, Docker, CI/CD | 완료 |
| 6 | [financial-finetuning](./financial-finetuning/) | LoRA, QLoRA, PEFT | 완료 |

---

## Step 1-2: Financial Analysis (ML + LLM)

금융 이상거래 탐지 시스템 - ML 모델과 LLM을 결합한 분석 플랫폼

### 주요 기능
- NumPy/Pandas 기반 금융 데이터 전처리
- scikit-learn ML 이상거래 탐지 (Random Forest, Isolation Forest)
- OpenAI/Claude API 연동
- Zero-shot, Few-shot, Chain-of-Thought 프롬프트
- Function Calling (Tool Use)

### 빠른 시작
```bash
cd financial-analysis
pip install -r requirements.txt
streamlit run app/streamlit_app.py
# 또는 Docker
docker-compose up streamlit
```

---

## Step 3: Finance RAG API

금융 문서 기반 RAG Q&A 시스템 - LLM 환각 방지

### 주요 기능
- 문서 검색 기반 Q&A (환각 방지)
- PDF/텍스트 업로드 및 자동 청킹
- 답변 출처와 신뢰도 점수
- 스트리밍 응답

### 아키텍처
```
User → Streamlit → FastAPI → RAG Service → Ollama LLM
                                    ↓
                               ChromaDB
```

### 빠른 시작
```bash
cd finance-rag-api
pip install -r requirements.txt
ollama pull llama3.2
uvicorn src.main:app --reload
# 웹 데모
streamlit run app/streamlit_app.py
```

---

## Step 4: Code Review Agent

AI 기반 코드 리뷰 에이전트 - ReAct 패턴 적용

### 주요 기능
- GitHub PR 자동 리뷰
- ReAct 패턴 (Reasoning + Acting)
- 코드 분석 도구 (AST, Complexity)
- 보안 취약점 탐지

### 아키텍처
```
PR Event → Agent Orchestrator → Tools (AST, Security, Complexity)
                    ↓
               LLM Reasoner → Review Comments
```

### 빠른 시작
```bash
cd code-review-agent
pip install -r requirements.txt
streamlit run app/demo_app.py
```

---

## Step 5: MLOps Pipeline

이상거래 탐지 모델 운영 파이프라인 - 엔드투엔드 MLOps

### 주요 기능
- DVC 데이터 버전 관리
- MLflow 실험 추적
- Docker 컨테이너화
- GitHub Actions CI/CD
- 모델 레지스트리

### 파이프라인
```
Data Versioning (DVC) → Training → Experiment Tracking (MLflow)
         ↓                                    ↓
    CI/CD Pipeline ← Model Registry ← Model Evaluation
```

### 빠른 시작
```bash
cd mlops-pipeline
pip install -r requirements.txt
# DVC 초기화 및 실행
dvc repro
# MLflow UI
mlflow ui
```

---

## Step 6: Financial Finetuning

금융 도메인 특화 LLM Fine-tuning - LoRA/QLoRA

### 주요 기능
- 100+ 금융 도메인 Instruction 데이터셋
- LoRA/QLoRA (Parameter-Efficient Fine-Tuning)
- 4-bit 양자화 학습
- FastAPI 추론 서버
- 스트리밍 응답

### 데이터셋 카테고리
| 카테고리 | 샘플 수 |
|---------|--------|
| 이상거래 탐지 | 15 |
| 투자 분석 | 20 |
| 금융 상품 설명 | 16 |
| 리스크 평가 | 15 |
| 시장 분석 | 17 |
| 금융 용어 설명 | 17 |

### 빠른 시작
```bash
cd financial-finetuning
pip install -r requirements.txt
# 학습
python -m src.training.train_lora --config configs/training_config.yaml
# 데모
streamlit run app/streamlit_app.py
# Docker (GPU)
docker-compose up streamlit
```

---

## 기술 스택 요약

| 분야 | 기술 |
|------|------|
| ML/DL | scikit-learn, PyTorch, Transformers |
| LLM | OpenAI, Claude, Ollama, PEFT |
| Vector DB | ChromaDB |
| MLOps | DVC, MLflow, Docker, GitHub Actions |
| Backend | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Testing | pytest |

## 전체 테스트

```bash
# 각 프로젝트별 테스트
cd financial-analysis && pytest tests/ -v
cd finance-rag-api && pytest tests/ -v
cd code-review-agent && pytest tests/ -v
cd mlops-pipeline && pytest tests/ -v
cd financial-finetuning && pytest tests/ -v
```

## 요구사항

- Python 3.11+
- Docker & Docker Compose
- Ollama (Step 3)
- NVIDIA GPU (Step 6 학습 시, 16GB+ VRAM)

## 라이선스

MIT License

## 연락처

- **GitHub**: https://github.com/araeLaver
