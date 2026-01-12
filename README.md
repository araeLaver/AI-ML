# AI/ML 학습 로드맵 & 포트폴리오

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-189%20passed-brightgreen.svg)](#테스트-현황)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗%20Demo-Live-orange.svg)](https://huggingface.co/spaces/araeLaver/finance-rag-demo)

> 백엔드 개발자(9년)의 AI/ML Engineer 전환을 위한 6단계 학습 로드맵과 실전 프로젝트

## 🎯 Live Demo

| 프로젝트 | 데모 링크 | 설명 |
|:---|:---:|:---|
| **Finance RAG** | [🤗 HuggingFace Spaces](https://huggingface.co/spaces/araeLaver/finance-rag-demo) | 금융 RAG Q&A 데모 |
| **Portfolio** | Coming Soon | 포트폴리오 웹사이트 |

---

## 📁 프로젝트 구조

```
AI-ML/
├── financial-analysis/      # Step 1-2: ML + LLM 기초 (20 tests)
├── finance-rag-api/         # Step 3: RAG 시스템 (53 tests)
├── code-review-agent/       # Step 4: AI Agent (55 tests)
├── mlops-pipeline/          # Step 5: MLOps 파이프라인 (40 tests)
├── financial-finetuning/    # Step 6: LLM Fine-tuning (21 tests)
├── portfolio/               # 포트폴리오 웹사이트 (Next.js)
├── huggingface-spaces/      # HuggingFace 배포용
└── docs/                    # 기술 문서 & 학습 가이드
```

---

## 🗺️ 학습 로드맵 (6단계)

```
Step 1-2        Step 3         Step 4         Step 5         Step 6
   │               │              │              │              │
   ▼               ▼              ▼              ▼              ▼
┌──────┐      ┌──────┐      ┌──────┐      ┌──────┐      ┌──────┐
│  ML  │ ───▶ │ RAG  │ ───▶ │Agent │ ───▶ │MLOps │ ───▶ │Fine- │
│ +LLM │      │System│      │      │      │      │      │tuning│
└──────┘      └──────┘      └──────┘      └──────┘      └──────┘
```

| Step | 프로젝트 | 핵심 기술 | 테스트 | 상태 |
|:---:|:---|:---|:---:|:---:|
| 1-2 | [financial-analysis](./financial-analysis/) | NumPy, Pandas, scikit-learn, LLM API | 20 | ✅ |
| 3 | [finance-rag-api](./finance-rag-api/) | RAG, ChromaDB, Ollama | 53 | ✅ |
| 4 | [code-review-agent](./code-review-agent/) | AI Agent, ReAct, LangGraph | 55 | ✅ |
| 5 | [mlops-pipeline](./mlops-pipeline/) | DVC, MLflow, Docker, CI/CD | 40 | ✅ |
| 6 | [financial-finetuning](./financial-finetuning/) | LoRA, QLoRA, PEFT | 21 | ✅ |

---

## 🧪 테스트 현황

```
financial-analysis ████████████████████ 20/20 (100%)
finance-rag-api    █████████████████████████████████████████████████████ 53/53 (100%)
code-review-agent  ███████████████████████████████████████████████████████ 55/55 (100%)
mlops-pipeline     ████████████████████████████████████████ 40/40 (100%)
financial-finetuning ████████████████████ 21/21 (100%)
────────────────────────────────────────────────────────────────────
Total: 189 tests passed ✅
```

---

## 📊 Step 1-2: Financial Analysis

금융 이상거래 탐지 시스템 - ML 모델과 LLM을 결합한 분석 플랫폼

### 아키텍처
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Data Layer │───▶│  ML Models  │───▶│  LLM Layer  │
│ NumPy/Pandas│    │  sklearn    │    │ OpenAI/Claude│
└─────────────┘    └─────────────┘    └─────────────┘
```

### 주요 기능
- 🔢 NumPy/Pandas 기반 금융 데이터 전처리
- 🤖 scikit-learn ML 이상거래 탐지 (Random Forest, Isolation Forest)
- 💬 OpenAI/Claude API 연동
- 📝 Zero-shot, Few-shot, Chain-of-Thought 프롬프트

### 빠른 시작
```bash
cd financial-analysis
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## 🔍 Step 3: Finance RAG API

금융 문서 기반 RAG Q&A 시스템 - LLM 환각 방지

### 아키텍처
```
User ──▶ Streamlit ──▶ FastAPI ──▶ RAG Service ──▶ Ollama LLM
                                        │
                                   ChromaDB
```

### 주요 기능
- 📄 문서 검색 기반 Q&A (환각 방지)
- 📤 PDF/텍스트 업로드 및 자동 청킹
- 📊 답변 출처와 신뢰도 점수
- ⚡ 스트리밍 응답

### 빠른 시작
```bash
cd finance-rag-api
pip install -r requirements.txt
ollama pull llama3.2
uvicorn src.main:app --reload
streamlit run app/streamlit_app.py
```

---

## 🤖 Step 4: Code Review Agent

AI 기반 코드 리뷰 에이전트 - ReAct 패턴 적용

### 아키텍처
```
PR Event ──▶ Agent Orchestrator ──▶ Tools (AST, Security, Complexity)
                    │
               LLM Reasoner ──▶ Review Comments
```

### 주요 기능
- 🔐 보안 취약점 탐지 (SQL Injection, XSS)
- ⚡ 성능 분석 (Complexity)
- 🎨 스타일 검사
- 🐙 GitHub PR 자동 리뷰

### 빠른 시작
```bash
cd code-review-agent
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

---

## ⚙️ Step 5: MLOps Pipeline

이상거래 탐지 모델 운영 파이프라인 - 엔드투엔드 MLOps

### 아키텍처
```
Data (DVC) ──▶ Training ──▶ MLflow Tracking
     │                            │
CI/CD Pipeline ◀── Model Registry ◀── Evaluation
```

### 주요 기능
- 📦 DVC 데이터 버전 관리
- 📈 MLflow 실험 추적
- 🐳 Docker 컨테이너화
- 🔄 GitHub Actions CI/CD

### 빠른 시작
```bash
cd mlops-pipeline
pip install -r requirements.txt
dvc repro
mlflow ui
```

---

## 🎓 Step 6: Financial Finetuning

금융 도메인 특화 LLM Fine-tuning - LoRA/QLoRA

### 아키텍처
```
Dataset ──▶ LoRA/QLoRA Training ──▶ Inference Engine
  │              │                        │
Alpaca      PEFT + 4-bit            FastAPI Server
Format      Quantization            + Streaming
```

### 주요 기능
- 📚 100+ 금융 도메인 Instruction 데이터셋
- 🔧 LoRA/QLoRA (Parameter-Efficient Fine-Tuning)
- 💾 4-bit 양자화 학습
- 🚀 FastAPI 추론 서버

### 빠른 시작
```bash
cd financial-finetuning
pip install -r requirements.txt
python -m src.training.train_lora --config configs/training_config.yaml
```

---

## 🛠️ 기술 스택

| 분야 | 기술 |
|:---|:---|
| **ML/DL** | NumPy, Pandas, scikit-learn, PyTorch, Transformers |
| **LLM** | OpenAI API, Claude API, Ollama, LangChain |
| **RAG** | ChromaDB, Embeddings |
| **Fine-tuning** | LoRA, QLoRA, PEFT, BitsAndBytes |
| **MLOps** | DVC, MLflow, Docker, GitHub Actions |
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | Streamlit, Next.js |
| **Testing** | pytest (189 tests) |

---

## 📚 문서

| 문서 | 설명 |
|:---|:---|
| [Portfolio_Technical_Guide.md](./docs/Portfolio_Technical_Guide.md) | 전체 프로젝트 기술 문서 |
| [AWS_ML_Study_Notion_Template.md](./docs/AWS_ML_Study_Notion_Template.md) | AWS ML Specialty 학습 템플릿 |
| [08-13_AWS_ML_Specialty_*.md](./docs/) | AWS 12주 학습 가이드 |

---

## 🚀 빠른 시작

### 전체 테스트 실행
```bash
# 모든 프로젝트 테스트
cd financial-analysis && pytest && cd ..
cd finance-rag-api && pytest && cd ..
cd code-review-agent && pytest && cd ..
cd mlops-pipeline && pytest && cd ..
cd financial-finetuning && pytest && cd ..
```

### 요구사항
- Python 3.11+
- Docker & Docker Compose
- Ollama (Step 3)
- NVIDIA GPU (Step 6 학습 시, 16GB+ VRAM 권장)

---

## 👤 Author

**김다운 (Kim Dawoon)**
- 백엔드 개발자 9년 → AI/ML Engineer 전환 중
- GitHub: [@araeLaver](https://github.com/araeLaver)

---

## 📄 License

MIT License - 자유롭게 사용하세요!
