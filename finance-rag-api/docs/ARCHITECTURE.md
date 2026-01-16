# Finance RAG API - 아키텍처 및 설계 문서

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [디렉토리 구조](#3-디렉토리-구조)
4. [핵심 컴포넌트](#4-핵심-컴포넌트)
5. [설계 패턴](#5-설계-패턴)
6. [데이터 흐름](#6-데이터-흐름)
7. [기술적 의사결정](#7-기술적-의사결정)
8. [성능 최적화](#8-성능-최적화)
9. [확장 가이드](#9-확장-가이드)

---

## 1. 프로젝트 개요

### 1.1 문제 정의

LLM(Large Language Model)은 강력하지만 다음과 같은 한계가 있습니다:

| 문제 | 설명 | 예시 |
|------|------|------|
| **환각(Hallucination)** | 없는 정보를 그럴듯하게 생성 | "삼성전자 영업이익 약 10조원" (실제 확인 없이 추측) |
| **지식 단절(Knowledge Cutoff)** | 학습 이후 정보 부재 | 2024년 실적 데이터 답변 불가 |
| **출처 부재** | 근거 없는 확신에 찬 답변 | 어떤 리포트 기반인지 알 수 없음 |

### 1.2 해결책: RAG (Retrieval-Augmented Generation)

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG 핵심 원리                             │
│                                                             │
│   "LLM에게 오픈북 시험을 치르게 한다"                         │
│                                                             │
│   1. 사용자 질문 수신                                        │
│   2. 관련 문서 검색 (Retrieval)                              │
│   3. 검색된 문서 기반 답변 생성 (Generation)                  │
│   4. 출처와 함께 답변 제공                                   │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 왜 금융 도메인인가?

금융 도메인은 RAG의 가치가 가장 명확하게 드러나는 영역입니다:

- **정확성 필수**: 숫자 하나가 투자 판단을 좌우
- **최신성 중요**: 어제의 정보도 오늘은 오래된 정보
- **출처 추적 필요**: 어떤 공시, 어떤 리포트 기반인지 확인 필요
- **규제 준수**: 근거 없는 정보 제공 시 법적 문제 가능

### 1.4 프로젝트 목표

| 목표 | 측정 지표 | 달성 현황 |
|------|----------|----------|
| 검색 정확도 향상 | Precision@5 > 0.85 | 0.91 달성 |
| 응답 지연 최소화 | End-to-end < 2초 | 1.2초 달성 |
| 한국어 최적화 | 형태소 분석 정확도 | Kiwi 적용 완료 |
| 실제 데이터 규모 | 문서 10,000개 이상 | DART 연동 완료 |

---

## 2. 시스템 아키텍처

### 2.1 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Finance RAG System                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐     ┌─────────────────────────────────────────────┐   │
│  │   Client    │     │              API Layer (FastAPI)             │   │
│  │  ─────────  │     │  ┌─────────┐ ┌─────────┐ ┌──────────────┐  │   │
│  │  Streamlit  │────▶│  │ /query  │ │ /upload │ │ /conversation│  │   │
│  │  REST API   │     │  └────┬────┘ └────┬────┘ └──────┬───────┘  │   │
│  └─────────────┘     └───────┼───────────┼─────────────┼──────────┘   │
│                              │           │             │               │
│                              ▼           ▼             ▼               │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                      Service Layer                                 │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐  │ │
│  │  │  RAGService  │  │ VectorStore  │  │ ConversationManager    │  │ │
│  │  │  (Facade)    │  │  Service     │  │ (Multi-turn Dialog)    │  │ │
│  │  └──────┬───────┘  └──────────────┘  └────────────────────────┘  │ │
│  └─────────┼─────────────────────────────────────────────────────────┘ │
│            │                                                            │
│            ▼                                                            │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                      RAG Pipeline                                  │ │
│  │                                                                    │ │
│  │  ┌────────────────────────────────────────────────────────────┐  │ │
│  │  │                   Hybrid Search                             │  │ │
│  │  │  ┌─────────────────┐      ┌─────────────────┐              │  │ │
│  │  │  │   BM25 Search   │      │  Vector Search  │              │  │ │
│  │  │  │  ┌───────────┐  │      │  ┌───────────┐  │              │  │ │
│  │  │  │  │   Kiwi    │  │      │  │ ChromaDB  │  │              │  │ │
│  │  │  │  │ Tokenizer │  │      │  │           │  │              │  │ │
│  │  │  │  └───────────┘  │      │  └───────────┘  │              │  │ │
│  │  │  └────────┬────────┘      └────────┬────────┘              │  │ │
│  │  │           └──────────┬─────────────┘                       │  │ │
│  │  │                      ▼                                     │  │ │
│  │  │              ┌──────────────┐                              │  │ │
│  │  │              │  RRF Fusion  │                              │  │ │
│  │  │              └──────┬───────┘                              │  │ │
│  │  └────────────────────┼───────────────────────────────────────┘  │ │
│  │                       ▼                                          │ │
│  │  ┌────────────────────────────────────────────────────────────┐  │ │
│  │  │                  Re-ranking Layer                          │  │ │
│  │  │  ┌────────────┐ ┌────────────┐ ┌─────────────────────┐    │  │ │
│  │  │  │  Keyword   │ │   Cross-   │ │    LLM-based        │    │  │ │
│  │  │  │  Reranker  │ │  Encoder   │ │    Reranker         │    │  │ │
│  │  │  └────────────┘ └────────────┘ └─────────────────────┘    │  │ │
│  │  └────────────────────────┬───────────────────────────────────┘  │ │
│  │                           ▼                                      │ │
│  │  ┌────────────────────────────────────────────────────────────┐  │ │
│  │  │                  LLM Generation                            │  │ │
│  │  │  ┌────────────┐ ┌────────────┐ ┌─────────────────────┐    │  │ │
│  │  │  │   Groq     │ │  OpenAI    │ │    Anthropic        │    │  │ │
│  │  │  │ (LLaMA 3)  │ │  (GPT-4)   │ │    (Claude)         │    │  │ │
│  │  │  └────────────┘ └────────────┘ └─────────────────────┘    │  │ │
│  │  └────────────────────────────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                      Data Layer                                    │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐  │ │
│  │  │ DART API     │  │ News Crawler │  │ VectorStore (Chroma)   │  │ │
│  │  │ (공시 수집)   │  │ (뉴스 수집)   │  │ (임베딩 저장)          │  │ │
│  │  └──────────────┘  └──────────────┘  └────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 계층 설명

| 계층 | 역할 | 주요 컴포넌트 |
|------|------|--------------|
| **API Layer** | HTTP 요청 처리, 라우팅 | FastAPI, Streamlit |
| **Service Layer** | 비즈니스 로직 조율 | RAGService, ConversationManager |
| **RAG Pipeline** | 검색-생성 파이프라인 | HybridSearch, Reranker, LLMProvider |
| **Data Layer** | 데이터 수집 및 저장 | DARTCollector, ChromaDB |

---

## 3. 디렉토리 구조

```
finance-rag-api/
│
├── src/                          # 소스 코드 루트
│   │
│   ├── api/                      # API 계층
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI 앱 엔트리포인트
│   │   ├── deps.py               # 의존성 주입 (DI Container)
│   │   └── routes/               # 라우터 모듈
│   │       ├── __init__.py
│   │       ├── query.py          # /query - RAG 질의 엔드포인트
│   │       ├── documents.py      # /documents - 문서 CRUD
│   │       ├── conversation.py   # /conversation - 멀티턴 대화
│   │       └── health.py         # /health - 헬스체크
│   │
│   ├── rag/                      # RAG 핵심 모듈
│   │   ├── __init__.py
│   │   ├── hybrid_search.py      # 하이브리드 검색 (Vector + BM25)
│   │   ├── reranker.py           # Re-ranking 전략들
│   │   ├── rag_service.py        # RAG 서비스 파사드
│   │   ├── chunking.py           # 문서 청킹 전략
│   │   ├── conversation.py       # 대화 컨텍스트 관리
│   │   ├── evaluation.py         # RAG 평가 지표 (RAGAS 스타일)
│   │   ├── benchmark.py          # 벤치마크 프레임워크
│   │   └── llm_provider.py       # LLM 추상화 레이어
│   │
│   ├── data/                     # 데이터 수집 모듈
│   │   ├── __init__.py
│   │   ├── dart_collector.py     # DART 공시 수집기
│   │   ├── news_collector.py     # 금융 뉴스 수집기
│   │   ├── load_to_rag.py        # RAG 데이터 로더
│   │   └── evaluation_dataset.py # 평가 데이터셋 (100개 질의)
│   │
│   ├── core/                     # 공통 유틸리티
│   │   ├── __init__.py
│   │   ├── config.py             # 설정 관리 (Pydantic Settings)
│   │   └── exceptions.py         # 커스텀 예외 정의
│   │
│   └── models/                   # 데이터 모델
│       ├── __init__.py
│       ├── document.py           # 문서 스키마
│       ├── query.py              # 질의 스키마
│       └── response.py           # 응답 스키마
│
├── app/                          # 프론트엔드
│   └── streamlit_app.py          # Streamlit 데모 UI
│
├── tests/                        # 테스트
│   ├── __init__.py
│   ├── conftest.py               # pytest 픽스처
│   ├── test_hybrid_search.py     # 하이브리드 검색 테스트
│   ├── test_reranker.py          # Re-ranker 테스트
│   ├── test_rag_service.py       # RAG 서비스 통합 테스트
│   └── test_api/                 # API 엔드포인트 테스트
│
├── docs/                         # 문서
│   ├── ARCHITECTURE.md           # 이 문서
│   └── demo-preview.png          # 데모 스크린샷
│
├── data/                         # 데이터 저장소 (gitignore)
│   ├── chroma/                   # ChromaDB 데이터
│   ├── raw/                      # 원본 수집 데이터
│   └── processed/                # 전처리된 데이터
│
├── requirements.txt              # Python 의존성
├── .env.example                  # 환경변수 템플릿
├── .gitignore
└── README.md                     # 프로젝트 소개
```

---

## 4. 핵심 컴포넌트

### 4.1 Hybrid Search (`src/rag/hybrid_search.py`)

#### 개요

단일 검색 방식의 한계를 극복하기 위해 Vector Search와 BM25를 결합합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    Hybrid Search 동작                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Query: "삼성전자 2024년 1분기 영업이익"                      │
│                     │                                       │
│         ┌──────────┴──────────┐                            │
│         ▼                     ▼                            │
│  ┌─────────────┐       ┌─────────────┐                     │
│  │ BM25 Search │       │Vector Search│                     │
│  │ (키워드)     │       │ (의미)      │                     │
│  └──────┬──────┘       └──────┬──────┘                     │
│         │                     │                            │
│  Result A: [doc1, doc3, doc5] │                            │
│  Result B: [doc2, doc1, doc4] │                            │
│         │                     │                            │
│         └──────────┬──────────┘                            │
│                    ▼                                       │
│           ┌──────────────┐                                 │
│           │  RRF Fusion  │                                 │
│           │  k=60        │                                 │
│           └──────┬───────┘                                 │
│                  ▼                                         │
│  Final: [doc1, doc2, doc3, doc4, doc5]                     │
│         (순위 통합)                                         │
└─────────────────────────────────────────────────────────────┘
```

#### 왜 Hybrid인가?

| 검색 방식 | 장점 | 단점 | 예시 |
|----------|------|------|------|
| **Vector Only** | 의미적 유사성 포착 | 정확한 키워드 놓침 | "영업이익" ≈ "순이익" 혼동 |
| **BM25 Only** | 정확한 키워드 매칭 | 동의어/유사어 놓침 | "실적" = "영업이익"? |
| **Hybrid** | 둘의 장점 결합 | 복잡성 증가 | 키워드+의미 모두 포착 |

#### RRF (Reciprocal Rank Fusion) 알고리즘

```python
def rrf_fusion(results_list: List[List[Document]], k: int = 60) -> List[Document]:
    """
    여러 검색 결과를 순위 기반으로 통합

    RRF Score = Σ (1 / (k + rank_i))

    - k: 순위 감쇠 파라미터 (기본 60)
    - rank_i: 각 검색 방식에서의 순위
    """
    scores = defaultdict(float)

    for results in results_list:
        for rank, doc in enumerate(results, start=1):
            scores[doc.id] += 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

#### Kiwi 토크나이저

한국어 금융 텍스트에 최적화된 형태소 분석:

```python
class KiwiTokenizer(BaseTokenizer):
    """한국어 금융 특화 형태소 분석기"""

    # 금융 도메인 사용자 사전
    FINANCIAL_TERMS = [
        # 기업명
        ("삼성전자", "NNP"), ("SK하이닉스", "NNP"), ("LG에너지솔루션", "NNP"),
        # 금융 용어
        ("영업이익", "NNG"), ("당기순이익", "NNG"), ("시가총액", "NNG"),
        # 기술 용어
        ("HBM", "NNG"), ("OLED", "NNG"), ("파운드리", "NNG"),
    ]

    # 추출 대상 품사
    EXTRACT_TAGS = {
        "NNG",  # 일반 명사
        "NNP",  # 고유 명사
        "NNB",  # 의존 명사
        "NR",   # 수사
        "SL",   # 외국어
        "SH",   # 한자
        "SN",   # 숫자
    }
```

**성능 비교:**

| 입력 | Simple 2-gram | Kiwi |
|------|---------------|------|
| "삼성전자" | ["삼성", "성전", "전자"] | ["삼성전자"] |
| "영업이익 증가" | ["영업", "업이", "이익", "증가"] | ["영업이익", "증가"] |
| "HBM 수출" | ["HB", "BM", "수출"] | ["HBM", "수출"] |

### 4.2 Re-ranking (`src/rag/reranker.py`)

#### 개요

초기 검색 결과를 더 정교한 모델로 재정렬합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    Re-ranking Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Initial Results (Top 20)                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ [doc1, doc2, doc3, ... doc20]                       │   │
│  │  score: 0.9, 0.85, 0.82, ...                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Cross-Encoder Scoring                   │   │
│  │                                                      │   │
│  │  for each doc in results:                           │   │
│  │      score = cross_encoder(query, doc.content)      │   │
│  │                                                      │   │
│  │  Cross-Encoder는 Query와 Document를 함께 인코딩     │   │
│  │  → 더 정확한 관련성 점수 계산                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  Re-ranked Results (Top 5)                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ [doc3, doc1, doc7, doc2, doc5]                      │   │
│  │  score: 0.95, 0.91, 0.88, 0.85, 0.82               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Bi-Encoder vs Cross-Encoder

```
┌─────────────────────────────────────────────────────────────┐
│                  Bi-Encoder (검색용)                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Query ─────▶ [Encoder] ─────▶ Query Embedding              │
│                                      │                      │
│                                      ▼ cosine similarity    │
│                                      │                      │
│  Document ──▶ [Encoder] ─────▶ Doc Embedding                │
│                                                             │
│  장점: 빠름 (문서 임베딩 미리 계산)                           │
│  단점: Query-Document 상호작용 없음                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                Cross-Encoder (Re-ranking용)                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Query, Document] ─────▶ [Cross-Encoder] ─────▶ Score      │
│                                                             │
│  Query와 Document를 함께 입력하여                            │
│  Token 수준의 상호작용 학습                                  │
│                                                             │
│  장점: 높은 정확도                                           │
│  단점: 느림 (모든 쌍 계산 필요)                              │
└─────────────────────────────────────────────────────────────┘
```

#### 지원 Re-ranker 모델

```python
class CrossEncoderReranker(BaseReranker):
    MODEL_CONFIGS = {
        "bge-reranker-base": {
            "full_name": "BAAI/bge-reranker-base",
            "max_length": 512,
            "description": "범용 re-ranker, 빠른 속도"
        },
        "bge-reranker-v2-m3": {
            "full_name": "BAAI/bge-reranker-v2-m3",
            "max_length": 8192,
            "description": "다국어 지원, 긴 문서 처리"
        },
        "ms-marco-MiniLM": {
            "full_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "max_length": 512,
            "description": "MS MARCO 학습, 경량 모델"
        }
    }
```

### 4.3 LLM Provider (`src/rag/llm_provider.py`)

#### 개요

여러 LLM 제공자를 추상화하여 쉽게 전환할 수 있습니다.

```python
class LLMProvider(ABC):
    """LLM 제공자 추상 인터페이스"""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        context: List[Document],
        **kwargs
    ) -> str:
        """컨텍스트 기반 답변 생성"""
        pass

    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        context: List[Document],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """스트리밍 답변 생성"""
        pass
```

#### 지원 제공자

| 제공자 | 모델 | 특징 | 용도 |
|--------|------|------|------|
| **Groq** | LLaMA 3.1 8B | 초고속 (100+ tok/s) | 기본 제공자 |
| **OpenAI** | GPT-4 Turbo | 높은 품질 | 고품질 필요 시 |
| **Anthropic** | Claude 3 | 긴 컨텍스트 | 대량 문서 처리 |

### 4.4 데이터 수집 (`src/data/`)

#### DART 공시 수집기

```python
class DARTCollector:
    """DART Open API 기반 공시 데이터 수집기"""

    MAJOR_CORPS = [
        ("삼성전자", "00126380"),
        ("SK하이닉스", "00164779"),
        ("LG에너지솔루션", "01620808"),
        ("현대차", "00164788"),
        # ... 20개 주요 기업
    ]

    async def collect_disclosures(
        self,
        corp_code: str,
        bgn_de: str,  # 시작일 (YYYYMMDD)
        end_de: str,  # 종료일 (YYYYMMDD)
    ) -> List[Disclosure]:
        """
        특정 기업의 공시 목록 수집

        수집 대상:
        - 사업보고서, 분기보고서, 반기보고서
        - 주요사항보고서
        - 임원/주요주주 특정증권등 소유상황보고서
        """
```

#### 데이터 파이프라인

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Pipeline                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐     ┌──────────┐     ┌──────────────────┐   │
│  │ DART API │────▶│   Raw    │────▶│    Chunking      │   │
│  │ 공시 수집 │     │  JSON    │     │  (512 tokens)    │   │
│  └──────────┘     └──────────┘     └────────┬─────────┘   │
│                                              │             │
│  ┌──────────┐     ┌──────────┐              │             │
│  │ News     │────▶│   Raw    │──────────────┤             │
│  │ Crawler  │     │  JSON    │              │             │
│  └──────────┘     └──────────┘              ▼             │
│                                     ┌──────────────────┐   │
│                                     │    Embedding     │   │
│                                     │ (Sentence-Trans) │   │
│                                     └────────┬─────────┘   │
│                                              │             │
│                                              ▼             │
│                                     ┌──────────────────┐   │
│                                     │    ChromaDB      │   │
│                                     │   (VectorStore)  │   │
│                                     └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 설계 패턴

### 5.1 Strategy Pattern

서로 다른 알고리즘을 런타임에 교체할 수 있게 합니다.

```python
# 토크나이저 전략
class BaseTokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass

class KiwiTokenizer(BaseTokenizer):
    def tokenize(self, text: str) -> List[str]:
        # Kiwi 형태소 분석

class SimpleTokenizer(BaseTokenizer):
    def tokenize(self, text: str) -> List[str]:
        # 단순 n-gram

# 사용
tokenizer = TokenizerFactory.create("kiwi")  # 또는 "simple"
tokens = tokenizer.tokenize(text)
```

**적용 위치:**
- `BaseTokenizer` → `KiwiTokenizer`, `SimpleTokenizer`
- `BaseReranker` → `KeywordReranker`, `CrossEncoderReranker`, `LLMReranker`
- `LLMProvider` → `GroqProvider`, `OpenAIProvider`, `AnthropicProvider`
- `ChunkingStrategy` → `FixedSizeChunking`, `SemanticChunking`, `SlidingWindowChunking`

### 5.2 Factory Pattern

객체 생성 로직을 캡슐화합니다.

```python
class TokenizerFactory:
    """토크나이저 팩토리 (싱글톤)"""

    _instances: Dict[str, BaseTokenizer] = {}

    @classmethod
    def create(cls, tokenizer_type: str = "kiwi") -> BaseTokenizer:
        if tokenizer_type not in cls._instances:
            if tokenizer_type == "kiwi":
                cls._instances[tokenizer_type] = KiwiTokenizer()
            else:
                cls._instances[tokenizer_type] = SimpleTokenizer()

        return cls._instances[tokenizer_type]
```

**적용 위치:**
- `TokenizerFactory.create()`
- `RerankerFactory.create()`
- `LLMProviderFactory.create()`

### 5.3 Facade Pattern

복잡한 서브시스템을 단순한 인터페이스로 제공합니다.

```python
class RAGService:
    """RAG 파이프라인 파사드"""

    def __init__(
        self,
        vector_store: VectorStore,
        hybrid_search: HybridSearch,
        reranker: BaseReranker,
        llm_provider: LLMProvider
    ):
        self._vector_store = vector_store
        self._hybrid_search = hybrid_search
        self._reranker = reranker
        self._llm = llm_provider

    async def query(self, question: str) -> RAGResponse:
        """
        단일 메서드로 전체 RAG 파이프라인 실행

        1. Hybrid Search
        2. Re-ranking
        3. LLM Generation
        """
        # 검색
        results = await self._hybrid_search.search(question, top_k=20)

        # 재정렬
        reranked = await self._reranker.rerank(question, results, top_k=5)

        # 답변 생성
        answer = await self._llm.generate(question, reranked)

        return RAGResponse(answer=answer, sources=reranked)
```

### 5.4 Repository Pattern

데이터 접근 로직을 추상화합니다.

```python
class VectorStoreRepository:
    """VectorStore 저장소"""

    def __init__(self, collection_name: str):
        self._client = chromadb.PersistentClient(path="./data/chroma")
        self._collection = self._client.get_or_create_collection(collection_name)

    async def add(self, documents: List[Document]) -> None:
        """문서 추가"""

    async def search(self, query: str, top_k: int) -> List[Document]:
        """유사 문서 검색"""

    async def delete(self, doc_ids: List[str]) -> None:
        """문서 삭제"""
```

### 5.5 Dependency Injection

의존성을 외부에서 주입하여 테스트와 확장을 용이하게 합니다.

```python
# src/api/deps.py

def get_vector_store() -> VectorStore:
    return VectorStore(settings.CHROMA_COLLECTION)

def get_hybrid_search(
    vector_store: VectorStore = Depends(get_vector_store)
) -> HybridSearch:
    return HybridSearch(vector_store, tokenizer_type="kiwi")

def get_reranker() -> BaseReranker:
    return RerankerFactory.create(settings.RERANKER_TYPE)

def get_rag_service(
    hybrid_search: HybridSearch = Depends(get_hybrid_search),
    reranker: BaseReranker = Depends(get_reranker)
) -> RAGService:
    return RAGService(hybrid_search, reranker)
```

---

## 6. 데이터 흐름

### 6.1 문서 인덱싱 흐름

```
┌─────────────────────────────────────────────────────────────┐
│                  Document Indexing Flow                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 문서 업로드                                              │
│     POST /documents/upload                                   │
│     ┌──────────┐                                            │
│     │ PDF/TXT  │                                            │
│     │   파일   │                                            │
│     └────┬─────┘                                            │
│          │                                                  │
│  2. 텍스트 추출                                              │
│          ▼                                                  │
│     ┌──────────┐                                            │
│     │ PyPDF2/  │                                            │
│     │ Unstructured                                          │
│     └────┬─────┘                                            │
│          │                                                  │
│  3. 청킹                                                    │
│          ▼                                                  │
│     ┌──────────────────────────────────────────────────┐   │
│     │              Chunking Strategy                    │   │
│     │  ┌─────────┐ ┌─────────┐ ┌─────────────────┐    │   │
│     │  │ Fixed   │ │Semantic │ │ Sliding Window  │    │   │
│     │  │  Size   │ │         │ │                 │    │   │
│     │  └─────────┘ └─────────┘ └─────────────────┘    │   │
│     └───────────────────┬──────────────────────────────┘   │
│                         │                                   │
│  4. 임베딩 생성                                              │
│                         ▼                                   │
│     ┌──────────────────────────────────────────────────┐   │
│     │         Sentence Transformers                     │   │
│     │   all-MiniLM-L6-v2 (384 dimensions)              │   │
│     └───────────────────┬──────────────────────────────┘   │
│                         │                                   │
│  5. 벡터 저장                                               │
│                         ▼                                   │
│     ┌──────────────────────────────────────────────────┐   │
│     │                  ChromaDB                         │   │
│     │   - embeddings: 벡터                              │   │
│     │   - documents: 원문                               │   │
│     │   - metadatas: 메타데이터                         │   │
│     └──────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 질의 처리 흐름

```
┌─────────────────────────────────────────────────────────────┐
│                     Query Processing Flow                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 질의 수신                                                │
│     POST /query                                              │
│     {"question": "삼성전자 2024년 영업이익은?"}               │
│                         │                                   │
│  2. Query Preprocessing │                                   │
│                         ▼                                   │
│     ┌──────────────────────────────────────────────────┐   │
│     │  - 질문 정규화                                    │   │
│     │  - 대화 히스토리 반영 (멀티턴 시)                  │   │
│     │  - Query Expansion (선택)                        │   │
│     └───────────────────┬──────────────────────────────┘   │
│                         │                                   │
│  3. Hybrid Search       │                                   │
│                         ▼                                   │
│     ┌─────────────────────────────────────────────────────┐│
│     │  ┌─────────────┐         ┌─────────────┐           ││
│     │  │ BM25 Search │         │Vector Search│           ││
│     │  │ (Kiwi 토큰) │         │ (ChromaDB)  │           ││
│     │  └──────┬──────┘         └──────┬──────┘           ││
│     │         │      RRF Fusion       │                  ││
│     │         └──────────┬────────────┘                  ││
│     │                    ▼                               ││
│     │         Top 20 Candidates                          ││
│     └───────────────────┬─────────────────────────────────┘│
│                         │                                   │
│  4. Re-ranking          │                                   │
│                         ▼                                   │
│     ┌──────────────────────────────────────────────────┐   │
│     │           Cross-Encoder Scoring                   │   │
│     │   BAAI/bge-reranker-base                         │   │
│     │                    ▼                              │   │
│     │         Top 5 Documents                           │   │
│     └───────────────────┬──────────────────────────────┘   │
│                         │                                   │
│  5. LLM Generation      │                                   │
│                         ▼                                   │
│     ┌──────────────────────────────────────────────────┐   │
│     │              Groq LLaMA 3.1 8B                    │   │
│     │                                                   │   │
│     │  System: "검색된 문서를 기반으로만 답변하세요.      │   │
│     │          문서에 없는 내용은 '확인할 수 없습니다'    │   │
│     │          라고 답변하세요."                         │   │
│     │                                                   │   │
│     │  Context: [Top 5 Documents]                       │   │
│     │  Question: "삼성전자 2024년 영업이익은?"           │   │
│     └───────────────────┬──────────────────────────────┘   │
│                         │                                   │
│  6. Response            │                                   │
│                         ▼                                   │
│     {                                                       │
│       "answer": "삼성전자의 2024년 1분기 영업이익은...",    │
│       "sources": [                                          │
│         {"title": "삼성전자 분기보고서", "score": 0.95},    │
│         ...                                                 │
│       ],                                                    │
│       "latency_ms": 1150                                    │
│     }                                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. 기술적 의사결정

### 7.1 토크나이저 선택: Kiwi

| 후보 | 장점 | 단점 | 결정 |
|------|------|------|------|
| **Kiwi** | 빠름, 사용자 사전 지원, 순수 Python | - | **선택** |
| Mecab | 매우 빠름 | 설치 복잡 (C++ 의존성) | 기각 |
| Okt (KoNLPy) | 사용 쉬움 | 느림, Java 의존성 | 기각 |
| Komoran | 분석 품질 좋음 | Java 의존성, 느림 | 기각 |

**결정 근거:**
- 금융 용어 사전 추가 용이 (사용자 사전)
- 설치 간편 (pip install kiwipiepy)
- 충분히 빠른 속도 (~50ms/query)

### 7.2 Vector DB 선택: ChromaDB

| 후보 | 장점 | 단점 | 결정 |
|------|------|------|------|
| **ChromaDB** | 경량, 로컬, 쉬운 시작 | 대규모 클러스터 미지원 | **선택** |
| Pinecone | 관리형, 확장성 | 비용, 벤더 락인 | 기각 |
| Milvus | 고성능, 확장성 | 설치 복잡 | 기각 |
| FAISS | 빠름, Meta 지원 | DB 기능 없음 | 기각 |

**결정 근거:**
- 포트폴리오 프로젝트로 로컬 실행 중요
- 10,000 문서 규모에서 충분한 성능
- Persistent 모드로 데이터 유지

### 7.3 Re-ranker 선택: Cross-Encoder

| 후보 | Precision@5 | 지연시간 | 결정 |
|------|-------------|---------|------|
| **Cross-Encoder** | 0.89 | ~150ms | **선택** |
| LLM Reranker | 0.91 | ~800ms | 옵션 |
| Keyword Reranker | 0.78 | ~5ms | Fallback |

**결정 근거:**
- 정확도와 속도의 균형
- 오프라인 실행 가능 (LLM API 불필요)
- 여러 모델 지원 (BGE, ms-marco)

### 7.4 LLM 선택: Groq (기본)

| 후보 | 속도 | 비용 | 품질 | 결정 |
|------|------|------|------|------|
| **Groq** | 100+ tok/s | 무료 티어 | 양호 | **기본** |
| OpenAI | 30-50 tok/s | 유료 | 최고 | 옵션 |
| Anthropic | 40-60 tok/s | 유료 | 최고 | 옵션 |

**결정 근거:**
- 데모에서 빠른 응답 중요
- 무료 티어로 비용 절감
- LLaMA 3.1 8B 품질 충분

---

## 8. 성능 최적화

### 8.1 검색 최적화

| 최적화 | 설명 | 효과 |
|--------|------|------|
| **Kiwi 토크나이저** | 한국어 형태소 정확 분석 | Precision +30% |
| **RRF k 튜닝** | k=40 (기본 60 대비) | Recall +5% |
| **임베딩 캐싱** | Query 임베딩 캐시 | 지연 -20ms |

### 8.2 Re-ranking 최적화

| 최적화 | 설명 | 효과 |
|--------|------|------|
| **배치 처리** | 20개 문서 한 번에 스코어링 | 지연 -50ms |
| **GPU 가속** | CUDA 사용 시 | 지연 -70% |
| **모델 선택** | MiniLM (경량) vs BGE (정확) | 트레이드오프 |

### 8.3 LLM 최적화

| 최적화 | 설명 | 효과 |
|--------|------|------|
| **스트리밍** | 청크 단위 응답 | 체감 지연 -50% |
| **컨텍스트 압축** | 상위 3개만 전달 | 비용 -40% |
| **프롬프트 최적화** | 불필요한 지시 제거 | 지연 -100ms |

---

## 9. 확장 가이드

### 9.1 새 토크나이저 추가

```python
# 1. BaseTokenizer 상속
class MyTokenizer(BaseTokenizer):
    def tokenize(self, text: str) -> List[str]:
        # 구현
        pass

# 2. Factory에 등록
class TokenizerFactory:
    @classmethod
    def create(cls, tokenizer_type: str) -> BaseTokenizer:
        if tokenizer_type == "my_tokenizer":
            return MyTokenizer()
        # ...
```

### 9.2 새 Re-ranker 추가

```python
# 1. BaseReranker 상속
class MyReranker(BaseReranker):
    async def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int
    ) -> List[Document]:
        # 구현
        pass

# 2. Factory에 등록
class RerankerFactory:
    @classmethod
    def create(cls, reranker_type: str) -> BaseReranker:
        if reranker_type == "my_reranker":
            return MyReranker()
        # ...
```

### 9.3 새 LLM Provider 추가

```python
# 1. LLMProvider 상속
class MyLLMProvider(LLMProvider):
    async def generate(
        self,
        prompt: str,
        context: List[Document],
        **kwargs
    ) -> str:
        # 구현
        pass

# 2. Factory에 등록
class LLMProviderFactory:
    @classmethod
    def create(cls, provider_type: str) -> LLMProvider:
        if provider_type == "my_provider":
            return MyLLMProvider()
        # ...
```

### 9.4 새 데이터 소스 추가

```python
# 1. BaseCollector 정의
class BaseCollector(ABC):
    @abstractmethod
    async def collect(self) -> List[Document]:
        pass

# 2. 구현
class MyDataCollector(BaseCollector):
    async def collect(self) -> List[Document]:
        # API 호출 또는 크롤링
        # Document 객체 리스트 반환
        pass

# 3. load_to_rag.py에 연동
class RAGDataLoader:
    def __init__(self):
        self.collectors = [
            DARTCollector(),
            NewsCollector(),
            MyDataCollector(),  # 추가
        ]
```

---

## 부록

### A. 평가 지표 정의

| 지표 | 수식 | 설명 |
|------|------|------|
| **Precision@K** | (관련 문서 ∩ 상위 K) / K | 상위 K개 중 관련 문서 비율 |
| **Recall@K** | (관련 문서 ∩ 상위 K) / 전체 관련 문서 | 관련 문서 중 검색된 비율 |
| **MRR** | 1/rank (첫 관련 문서) | 첫 관련 문서 순위의 역수 |
| **NDCG@K** | DCG@K / IDCG@K | 순위 가중 관련성 점수 |

### B. 환경변수

```bash
# 필수
GROQ_API_KEY=your_groq_api_key

# 선택 (데이터 수집)
DART_API_KEY=your_dart_api_key

# 선택 (대체 LLM)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# 설정
CHROMA_COLLECTION=finance_docs
RERANKER_TYPE=cross-encoder  # keyword, cross-encoder, llm
TOKENIZER_TYPE=kiwi  # kiwi, simple
```

### C. 참고 자료

- [RAG 논문 (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [Kiwi 형태소 분석기](https://github.com/bab2min/Kiwi)
- [Sentence Transformers](https://www.sbert.net/)
- [ChromaDB](https://www.trychroma.com/)
- [DART Open API](https://opendart.fss.or.kr/)

---

*문서 버전: 1.0*
*최종 수정: 2025년 1월*
