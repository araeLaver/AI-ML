# AI/ML 학습 & 포트폴리오

백엔드 개발자의 AI/ML 전환을 위한 학습 자료와 실전 프로젝트입니다.

## 프로젝트 구성

```
AI-ML/
├── docs/                # 학습 로드맵 및 단계별 가이드
├── finance-rag-api/     # RAG 기반 금융 Q&A 시스템 (핵심 프로젝트)
└── portfolio/           # 포트폴리오 웹사이트 (Next.js)
```

## 핵심 프로젝트: Finance RAG API

금융 문서 기반 **RAG (Retrieval-Augmented Generation)** Q&A 시스템

### 왜 만들었나?

LLM의 환각(Hallucination) 문제를 해결하기 위해 RAG 패턴을 적용했습니다.
금융 도메인은 잘못된 정보가 실제 손실로 이어질 수 있어, 검증된 문서 기반 답변이 필수입니다.

### 주요 기능

- 금융 문서 검색 기반 Q&A (환각 방지)
- PDF/텍스트 파일 업로드 및 자동 청킹
- 답변 출처와 신뢰도 점수 제공
- 스트리밍 응답 지원
- Streamlit 웹 데모

### 기술 스택

| 구분 | 기술 |
|------|------|
| Backend | FastAPI, Python 3.11 |
| LLM | Ollama (llama3.2) |
| Vector DB | ChromaDB |
| Frontend | Streamlit |
| DevOps | Docker, pytest |

### 아키텍처

```
User → Streamlit/Swagger → FastAPI → RAG Service → Ollama LLM
                                          ↓
                                     ChromaDB (Vector Store)
```

### 빠른 시작

```bash
cd finance-rag-api

# 환경 설정
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt

# Ollama 모델 준비
ollama pull llama3.2

# 서버 실행
uvicorn src.main:app --reload --port 8000

# 웹 데모 실행
streamlit run app/streamlit_app.py
```

### API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/v1/query` | RAG 질의 |
| POST | `/api/v1/upload` | 문서 업로드 |
| GET | `/api/v1/documents` | 문서 목록 |
| GET | `/api/v1/health` | 헬스체크 |

### 테스트

```bash
pytest tests/ -v  # 35개 테스트 케이스
```

## 학습 로드맵

`docs/` 디렉토리에 단계별 학습 가이드가 있습니다.

1. **Step 1**: Python AI 기초 (NumPy, Pandas)
2. **Step 2**: LLM API & 프롬프트 엔지니어링
3. **Step 3**: RAG 시스템 구축 ← 현재 프로젝트
4. **Step 4**: AI Agent 개발
5. **Step 5**: MLOps
6. **Step 6**: Fine-tuning

## 기술적 특징

### 환각 방지 설계

```python
# 프롬프트 엔지니어링으로 문서 외 답변 차단
"문서에서 찾을 수 없는 정보는
'제공된 문서에서 관련 정보를 찾을 수 없습니다'라고 답변하세요."
```

### 신뢰도 점수

벡터 유사도 기반으로 답변 신뢰도를 High/Medium/Low로 분류합니다.

### 계층화된 아키텍처

```
API Layer (routes, schemas, security)
    ↓
Service Layer (rag_service)
    ↓
Data Layer (vectorstore, document_loader)
```

## 개발 환경

- Python 3.11+
- Ollama (로컬 LLM)
- Docker & Docker Compose

## 라이선스

MIT License

## 연락처

- **GitHub**: https://github.com/araeLaver
