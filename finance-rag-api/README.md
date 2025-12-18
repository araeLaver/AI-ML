# Finance RAG API

금융 문서 기반 **RAG (Retrieval-Augmented Generation)** Q&A 시스템입니다.

LLM이 금융 문서를 검색하여 **근거 있는 답변**을 생성하고, **환각(Hallucination)을 방지**합니다.

## 주요 기능

- **RAG 질의**: 금융 문서를 검색하여 LLM이 답변 생성
- **문서 관리**: PDF/텍스트 파일 업로드 및 자동 청킹
- **출처 제공**: 답변의 근거 문서와 관련도 점수 명시
- **환각 방지**: 검색된 문서에 없는 내용은 답변하지 않음

## 기술 스택

| 구분 | 기술 |
|------|------|
| **LLM** | Ollama (llama3.2) - 로컬 실행 |
| **Vector DB** | ChromaDB - 임베딩 저장/검색 |
| **Framework** | FastAPI - 비동기 API 서버 |
| **Embedding** | Ollama Embeddings |
| **Testing** | pytest - 35개 테스트 케이스 |
| **Container** | Docker + Docker Compose |

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        Finance RAG API                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  │  Client  │──▶│ FastAPI  │──▶│   RAG    │──▶│  Ollama  │     │
│  │ (Swagger)│   │ (Routes) │   │ Service  │   │  (LLM)   │     │
│  └──────────┘   └──────────┘   └────┬─────┘   └──────────┘     │
│                                     │                           │
│                              ┌──────▼──────┐                    │
│                              │  ChromaDB   │                    │
│                              │ (Vector DB) │                    │
│                              └─────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

### 핵심 흐름

```
1. 질문 입력 → 2. 벡터 검색 → 3. 컨텍스트 구성 → 4. LLM 답변 생성 → 5. 출처와 함께 반환
```

## 빠른 시작

### 사전 요구사항

- Python 3.11+
- Ollama 설치 및 실행

### 설치

```bash
# 저장소 클론
git clone https://github.com/your-username/finance-rag-api.git
cd finance-rag-api

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# Ollama 모델 다운로드
ollama pull llama3.2:latest
```

### 환경 설정

```bash
# .env 파일 생성
cp .env.example .env

# 필요시 설정 수정
# - OLLAMA_API_URL: Ollama 서버 주소
# - LLM_MODEL: 사용할 모델
```

### 실행

```bash
# 개발 서버 실행
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# 또는
make dev
```

### API 문서

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API 엔드포인트

### 공개 API

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/api/v1/health` | 헬스체크 |
| GET | `/api/v1/stats` | 시스템 통계 |
| GET | `/api/v1/documents` | 문서 목록 조회 |

### 보호 API (Rate Limited)

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/v1/query` | RAG 질의 |
| POST | `/api/v1/documents` | 문서 추가 |
| DELETE | `/api/v1/documents` | 전체 문서 삭제 |
| POST | `/api/v1/upload` | 파일 업로드 |

### 사용 예시

```bash
# 질의
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "ETF가 뭔가요?", "top_k": 3}'

# 파일 업로드
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@document.pdf" \
  -F "source_name=투자가이드"
```

## Docker 배포

### Docker Compose 실행

```bash
# 전체 스택 실행 (API + Ollama)
docker-compose up -d

# 로그 확인
docker-compose logs -f api

# 종료
docker-compose down
```

### 개발 모드

```bash
# 소스 코드 마운트 + 핫 리로드
docker-compose -f docker-compose.dev.yml up -d
```

## 테스트

```bash
# 전체 테스트 실행
pytest tests/ -v

# 커버리지 포함
pytest tests/ -v --cov=src --cov-report=html
```

### 테스트 범위

- **API 테스트** (11개): 모든 엔드포인트 검증
- **Document Loader 테스트** (16개): 청킹, PDF 파싱
- **Vector Store 테스트** (8개): 검색, 필터링

## 프로젝트 구조

```
finance-rag-api/
├── src/
│   ├── api/                    # API Layer
│   │   ├── routes.py           # 엔드포인트 정의
│   │   ├── schemas.py          # Pydantic 스키마
│   │   ├── security.py         # 인증/Rate Limiting
│   │   ├── middleware.py       # 요청 로깅
│   │   └── exception_handlers.py
│   ├── rag/                    # RAG Core
│   │   ├── rag_service.py      # RAG 오케스트레이션
│   │   ├── vectorstore.py      # ChromaDB 래퍼
│   │   └── document_loader.py  # 문서 파싱/청킹
│   ├── core/                   # 공통 모듈
│   │   ├── config.py           # 환경 설정
│   │   ├── logging.py          # 로깅 시스템
│   │   └── exceptions.py       # 커스텀 예외
│   └── main.py                 # FastAPI 앱
├── tests/                      # 테스트
├── data/                       # 샘플 문서
├── docker-compose.yml          # 프로덕션 배포
├── Dockerfile                  # 멀티스테이지 빌드
└── Makefile                    # 편의 명령어
```

## 포트폴리오 하이라이트

### 1. 환각 방지 프롬프트 엔지니어링

```python
# 검색된 문서에 없는 내용은 답변 거부
"반드시 위 문서들을 바탕으로만 답변하세요.
문서에서 찾을 수 없는 정보는 '제공된 문서에서
관련 정보를 찾을 수 없습니다'라고 답변하세요."
```

### 2. 신뢰도 점수 시스템

```python
# 검색 결과의 관련도에 따른 신뢰도 분류
if avg_relevance >= 0.7:
    return "high"
elif avg_relevance >= 0.5:
    return "medium"
else:
    return "low"
```

### 3. 지능형 문서 청킹

```python
# LangChain 스타일 재귀적 텍스트 분할
# 문장 경계를 유지하면서 의미 단위 분할
separators = ["\n\n", "\n", ". ", " "]
```

### 4. 계층화된 예외 처리

```python
# Spring @ExceptionHandler 패턴 적용
class RAGException(Exception):
    error_code: ErrorCode
    message: str
    detail: Optional[str]
    status_code: int
```

## 향후 개선 계획

- [ ] JWT 기반 사용자 인증
- [ ] Redis 기반 Rate Limiting
- [ ] 스트리밍 응답 지원
- [ ] 멀티 모델 지원 (GPT-4, Claude)
- [ ] 문서 버전 관리
- [ ] 사용자별 문서 격리

## 라이선스

MIT License

## 연락처

- **개발자**: 김다운
- **Email**: your-email@example.com
- **GitHub**: https://github.com/your-username
