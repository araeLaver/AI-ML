# Finance RAG API

금융 문서 기반 **RAG (Retrieval-Augmented Generation)** Q&A 시스템입니다.

하이브리드 검색, Re-ranking, 멀티턴 대화를 지원하며, LLM이 금융 문서를 검색하여 **근거 있는 답변**을 생성하고 **환각(Hallucination)을 방지**합니다.

## 데모

**Live Demo**: [Streamlit App](http://localhost:8502)

![Finance RAG Demo](docs/demo-preview.png)

## 주요 기능

### Core RAG
- **RAG 질의**: 금융 문서를 검색하여 LLM이 답변 생성
- **문서 관리**: PDF/텍스트 파일 업로드 및 자동 청킹
- **출처 제공**: 답변의 근거 문서와 관련도 점수 명시
- **환각 방지**: 검색된 문서에 없는 내용은 답변하지 않음

### Advanced RAG (v2.0)
- **하이브리드 검색**: Vector + BM25 + RRF 알고리즘
- **Re-ranking**: Cross-Encoder / LLM 기반 재정렬
- **멀티턴 대화**: 대화 히스토리 및 엔티티 추적
- **RAG 평가**: RAGAS 스타일 평가 지표
- **스마트 청킹**: 시맨틱/슬라이딩 윈도우 청킹
- **실시간 스트리밍**: LLM 응답 스트리밍

## 기술 스택

| 구분 | 기술 |
|------|------|
| **LLM** | Groq (LLaMA 3.1 8B) - 초고속 추론 |
| **Vector DB** | ChromaDB - 임베딩 저장/검색 |
| **Framework** | FastAPI - 비동기 API 서버 |
| **UI** | Streamlit - 인터랙티브 데모 |
| **Search** | Hybrid (Vector + BM25 + RRF) |
| **Embedding** | Sentence Transformers |
| **Testing** | pytest - 35개 테스트 케이스 |

## 아키텍처

```
┌────────────────────────────────────────────────────────────────────────┐
│                         Finance RAG System                              │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    ┌──────────────────────────────────────────────────┐  │
│  │ Streamlit│    │                  RAG Pipeline                     │  │
│  │    UI    │───▶│  ┌─────────┐  ┌─────────┐  ┌─────────┐          │  │
│  └──────────┘    │  │ Hybrid  │─▶│Re-Ranker│─▶│   LLM   │          │  │
│                  │  │ Search  │  │         │  │ (Groq)  │          │  │
│  ┌──────────┐    │  └────┬────┘  └─────────┘  └─────────┘          │  │
│  │ FastAPI  │───▶│       │                                          │  │
│  │   API    │    │  ┌────┴────┐                                     │  │
│  └──────────┘    │  │ Vector  │  ┌─────────┐                        │  │
│                  │  │   DB    │  │  BM25   │                        │  │
│                  │  └─────────┘  └─────────┘                        │  │
│                  └──────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      Supporting Modules                           │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │  │
│  │  │Evaluation│  │Chunking  │  │Conversa- │  │Financial │         │  │
│  │  │ (RAGAS) │  │Strategies│  │tion Mem  │  │  Data    │         │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

### RAG 파이프라인 흐름

```
1. 질문 입력
      │
      ▼
2. 참조 해결 (대명사 → 엔티티)
      │
      ▼
3. 하이브리드 검색
   ├── Vector Search (의미 기반)
   └── BM25 Search (키워드 기반)
      │
      ▼
4. RRF Fusion (검색 결과 통합)
      │
      ▼
5. Re-Ranking (정밀 재정렬)
      │
      ▼
6. LLM 답변 생성 (스트리밍)
      │
      ▼
7. 출처 + 신뢰도와 함께 반환
```

## 빠른 시작

### 사전 요구사항

- Python 3.11+
- Groq API Key ([여기서 발급](https://console.groq.com))

### 설치

```bash
# 저장소 클론
git clone https://github.com/araeLaver/AI-ML.git
cd AI-ML/finance-rag-api

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 환경 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집
GROQ_API_KEY=your_groq_api_key_here
```

### 실행

```bash
# Streamlit 데모 실행
streamlit run app/streamlit_app.py --server.port 8502

# 또는 FastAPI 서버 실행
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 접속

- **Streamlit Demo**: http://localhost:8502
- **FastAPI Swagger**: http://localhost:8000/docs

## 프로젝트 구조

```
finance-rag-api/
├── app/
│   └── streamlit_app.py        # Streamlit 데모 UI
├── src/
│   ├── api/                    # API Layer
│   │   ├── routes.py           # 엔드포인트 정의
│   │   ├── schemas.py          # Pydantic 스키마
│   │   ├── security.py         # 인증/Rate Limiting
│   │   └── middleware.py       # 요청 로깅
│   ├── rag/                    # RAG Core
│   │   ├── rag_service.py      # RAG 오케스트레이션
│   │   ├── vectorstore.py      # ChromaDB 래퍼
│   │   ├── document_loader.py  # 문서 파싱
│   │   ├── llm_provider.py     # LLM 추상화 (Groq)
│   │   ├── hybrid_search.py    # 하이브리드 검색
│   │   ├── reranker.py         # Re-ranking
│   │   ├── chunking.py         # 청킹 전략
│   │   ├── evaluation.py       # RAG 평가
│   │   ├── conversation.py     # 대화 메모리
│   │   └── financial_data.py   # 금융 데이터
│   ├── core/                   # 공통 모듈
│   │   ├── config.py           # 환경 설정
│   │   ├── logging.py          # 로깅 시스템
│   │   └── exceptions.py       # 커스텀 예외
│   └── main.py                 # FastAPI 앱
├── tests/                      # 테스트
├── data/                       # 샘플 문서
├── docker-compose.yml          # 프로덕션 배포
└── requirements.txt            # 의존성
```

## 핵심 모듈 상세

### 1. 하이브리드 검색 (`hybrid_search.py`)

벡터 검색과 키워드 검색을 결합하여 검색 품질을 극대화합니다.

```python
# Vector Search: 의미적 유사성
# BM25 Search: 키워드 정확도
# RRF Fusion: 두 결과를 통합

class HybridSearcher:
    def search(self, query: str, top_k: int = 5):
        # 1. 벡터 검색
        vector_results = self.vector_search(query)

        # 2. BM25 검색
        bm25_results = self.bm25_search(query)

        # 3. RRF로 통합
        return self.rrf_fusion(vector_results, bm25_results)
```

**왜 하이브리드 검색인가?**
- 벡터 검색: "회사 실적" → "기업 성과" 매칭 (의미)
- BM25 검색: "삼성전자" → 정확히 "삼성전자" 매칭 (키워드)
- 결합: 두 장점을 모두 활용

### 2. Re-Ranking (`reranker.py`)

초기 검색 결과를 더 정교하게 재정렬합니다.

```python
# Two-Stage Retrieval
# Stage 1: 빠른 검색 (Bi-Encoder) - 후보 추출
# Stage 2: 정밀 평가 (Cross-Encoder) - 최종 선정

class ReRanker:
    def rerank(self, query: str, documents: List[Dict]):
        # Cross-Encoder로 쿼리-문서 쌍 평가
        # 관련성 점수 재계산
        # 상위 문서 반환
```

**Re-Ranker 종류:**
| 타입 | 설명 | 특징 |
|------|------|------|
| KeywordReranker | 키워드 매칭 기반 | 빠름, 기본 옵션 |
| LLMReranker | LLM이 관련성 평가 | 정확, 비용 발생 |
| CrossEncoderReranker | 전용 모델 사용 | 균형 잡힌 선택 |

### 3. 멀티턴 대화 (`conversation.py`)

대화 히스토리를 관리하여 자연스러운 대화를 지원합니다.

```python
# 문제: 단일 턴 RAG의 한계
# "삼성전자 실적 알려줘" → "9조원입니다"
# "그 회사 주가는?" → ??? (문맥 없음)

# 해결: 대화 메모리
class ConversationMemory:
    def resolve_references(self, query: str) -> str:
        # "그 회사" → "삼성전자"로 변환
        # 엔티티 추적으로 대명사 해결
```

**메모리 전략:**
- Window Memory: 최근 N개 턴 유지
- Entity Memory: 언급된 엔티티 추적
- Reference Resolution: 대명사 해결

### 4. RAG 평가 (`evaluation.py`)

RAGAS 스타일의 평가 지표로 RAG 품질을 측정합니다.

```python
class RAGEvaluator:
    def evaluate_all(self, input: RAGEvaluationInput):
        return {
            "faithfulness": self.evaluate_faithfulness(),      # 충실도
            "answer_relevancy": self.evaluate_relevancy(),     # 답변 관련성
            "context_precision": self.evaluate_precision(),    # 컨텍스트 정밀도
            "context_recall": self.evaluate_recall()           # 컨텍스트 재현율
        }
```

**평가 지표:**
| 지표 | 설명 | 개선 방향 |
|------|------|----------|
| Faithfulness | 답변이 컨텍스트에 기반하는지 | 프롬프트 강화 |
| Answer Relevancy | 답변이 질문과 관련있는지 | 프롬프트 개선 |
| Context Precision | 검색 문서가 관련있는지 | Re-ranking 적용 |
| Context Recall | 필요한 정보가 검색되었는지 | top_k 증가 |

### 5. 스마트 청킹 (`chunking.py`)

문서를 의미 단위로 분할하여 검색 품질을 높입니다.

```python
# 청킹 전략
strategies = {
    "fixed": FixedSizeChunker,       # 고정 크기
    "semantic": SemanticChunker,     # 의미 단위
    "sliding": SlidingWindowChunker, # 슬라이딩 윈도우
    "recursive": RecursiveChunker    # 재귀적 분할
}
```

**청킹 비교:**
| 전략 | 특징 | 적합한 경우 |
|------|------|------------|
| Fixed | 단순, 빠름 | 균일한 문서 |
| Semantic | 의미 보존 | 논문, 리포트 |
| Sliding | 오버랩 | 연속적 문맥 |
| Recursive | 경계 유지 | 구조화된 문서 |

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
  -d '{"question": "삼성전자 3분기 실적은?", "top_k": 3}'

# 응답
{
  "answer": "삼성전자 2024년 3분기 영업이익은 9조 1,834억원입니다...",
  "sources": [
    {"title": "삼성전자 실적 보고서", "relevance": 0.92}
  ],
  "confidence": "high"
}
```

## Streamlit UI

bpco.kr에서 영감을 받은 미니멀 디자인입니다.

### 디자인 특징

- **크림/베이지 톤**: 눈의 피로를 줄이는 따뜻한 색상
- **Space Mono 폰트**: 모노스페이스로 기술적 느낌
- **아웃라인 타이포그래피**: FINANCE / RAG_ 히어로 텍스트
- **무한 스크롤 마퀴**: 핵심 기능 키워드 흐름
- **섹션 넘버링**: 01, 02, 03 체계적 구성
- **호버 인터랙션**: 오렌지 악센트 효과
- **맥 스타일 터미널**: 데모 영역 UI

### 섹션 구성

1. **Hero**: 프로젝트 소개 + SCROLL DOWN 힌트
2. **Marquee**: 핵심 기능 키워드 흐름
3. **Stats**: 주요 수치 (5+ 소스, 3 알고리즘, <2s 응답)
4. **Features**: Hybrid_Search, Re_Ranking, Multi_Turn
5. **Demo**: 실제 작동하는 채팅 인터페이스
6. **Tech Stack**: 사용 기술 태그
7. **Footer**: 제작자 정보

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

## Docker 배포

```bash
# 전체 스택 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f api

# 종료
docker-compose down
```

## 포트폴리오 하이라이트

### 1. 환각 방지 프롬프트 엔지니어링

```python
"반드시 위 문서들을 바탕으로만 답변하세요.
문서에서 찾을 수 없는 정보는
'제공된 문서에서 관련 정보를 찾을 수 없습니다'라고 답변하세요."
```

### 2. 하이브리드 검색 + RRF

```python
# RRF (Reciprocal Rank Fusion)
# 각 검색 결과의 순위를 결합하여 최종 점수 계산
score = sum(1 / (k + rank) for rank in ranks)
```

### 3. Two-Stage Retrieval

```python
# Stage 1: Bi-Encoder (빠른 후보 추출)
candidates = vector_search(query, top_k=100)

# Stage 2: Cross-Encoder (정밀 평가)
final = rerank(query, candidates, top_k=5)
```

### 4. 대화 컨텍스트 유지

```python
# 엔티티 추적 + 대명사 해결
"그 회사 주가는?" → "삼성전자 주가는?"
```

## 향후 개선 계획

- [ ] JWT 기반 사용자 인증
- [ ] Redis 기반 캐싱/Rate Limiting
- [ ] 벡터 DB 확장 (Pinecone, Weaviate)
- [ ] 더 많은 금융 데이터 소스
- [ ] 차트/그래프 시각화
- [ ] 음성 입력 지원

## 라이선스

MIT License

## 연락처

- **개발자**: 김다운
- **GitHub**: https://github.com/araeLaver
- **Email**: araelaver@gmail.com
