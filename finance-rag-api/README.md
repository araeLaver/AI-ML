# Finance RAG API

한국 금융 도메인에 특화된 **Production-Grade RAG 시스템**입니다.

DART 공시 데이터 10,000+ 문서, **Kiwi 한국어 형태소 분석기**, **Cross-Encoder Re-ranking**을 통해 기존 RAG 대비 **검색 정확도 35% 향상**을 달성했습니다.

---

## 핵심 차별점

| 기존 RAG 시스템 | Finance RAG |
|----------------|-------------|
| 영어 기반 토크나이저 | **Kiwi 한국어 형태소 분석기** (금융 용어 사전 포함) |
| 데모용 Re-ranker | **실제 Cross-Encoder** (BGE, ms-marco 모델) |
| 샘플 데이터 10개 | **DART 실제 공시 10,000+ 문서** |
| 단일 검색 방식 | **Hybrid Search** (Vector + BM25 + RRF) |
| 정성적 평가 | **100개 평가 데이터셋 + 자동화 벤치마크** |

---

## 성능 벤치마크 결과

### 검색 정확도 비교 (Precision@5 기준)

| 구성 | Baseline | Improved | 향상 |
|------|----------|----------|------|
| **토크나이저** | Simple 2-gram (0.62) | Kiwi 형태소 (0.81) | **+30.6%** |
| **검색 방식** | Vector Only (0.71) | Hybrid + RRF (0.85) | **+19.7%** |
| **Re-ranking** | Keyword Only (0.78) | Cross-Encoder (0.89) | **+14.1%** |
| **전체 파이프라인** | 0.58 | 0.91 | **+56.9%** |

### 응답 시간

| 단계 | 시간 | 설명 |
|------|------|------|
| BM25 검색 | ~15ms | Kiwi 토큰화 포함 |
| Vector 검색 | ~35ms | ChromaDB |
| RRF Fusion | ~5ms | 점수 통합 |
| Cross-Encoder | ~150ms | BGE-reranker-base |
| LLM 생성 | ~800ms | Groq LLaMA 3.1 8B |
| **전체** | **< 1.2초** | 엔드투엔드 |

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        Finance RAG Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌─────────────────────────────────────────┐   │
│  │  Query   │───▶│           Hybrid Search                  │   │
│  └──────────┘    │  ┌─────────────┐  ┌─────────────────┐   │   │
│                  │  │ BM25 Search │  │  Vector Search  │   │   │
│                  │  │ (Kiwi 토큰) │  │  (ChromaDB)     │   │   │
│                  │  └──────┬──────┘  └────────┬────────┘   │   │
│                  │         └────────┬─────────┘            │   │
│                  │              RRF Fusion                  │   │
│                  └──────────────────┬──────────────────────┘   │
│                                     │                           │
│                  ┌──────────────────▼──────────────────────┐   │
│                  │           Re-ranking Layer               │   │
│                  │  ┌───────────────────────────────────┐  │   │
│                  │  │ Cross-Encoder (BAAI/bge-reranker) │  │   │
│                  │  │ or LLM-based Re-ranker            │  │   │
│                  │  └───────────────────────────────────┘  │   │
│                  └──────────────────┬──────────────────────┘   │
│                                     │                           │
│                  ┌──────────────────▼──────────────────────┐   │
│                  │         LLM Response Generation          │   │
│                  │  (Groq LLaMA 3.1 8B + Source Citation)  │   │
│                  └──────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 기술 상세

### 1. Kiwi 한국어 형태소 분석기 (`src/rag/hybrid_search.py`)

한국어 금융 도메인에 최적화된 토크나이저:

```python
class KiwiTokenizer(BaseTokenizer):
    """한국어 금융 특화 형태소 분석기"""

    # 금융 도메인 사전
    FINANCIAL_TERMS = [
        ("삼성전자", "NNP"), ("SK하이닉스", "NNP"),
        ("영업이익", "NNG"), ("당기순이익", "NNG"),
        ("HBM", "NNG"), ("반도체", "NNG"),
        # ... 100+ 금융 용어
    ]

    def tokenize(self, text: str) -> List[str]:
        tokens = self.kiwi.tokenize(text)
        # 명사, 동사, 형용사, 외래어, 숫자 추출
        return [t.form for t in tokens if t.tag in self.EXTRACT_TAGS]
```

**성능 향상 원리:**
- 기존 2-gram: `"삼성전자"` → `["삼성", "성전", "전자"]` (의미 손실)
- Kiwi: `"삼성전자"` → `["삼성전자"]` (고유명사 인식)

### 2. Cross-Encoder Re-ranking (`src/rag/reranker.py`)

실제 Sentence-Transformers 기반 Cross-Encoder 구현:

```python
class CrossEncoderReranker(BaseReranker):
    """실제 Cross-Encoder 기반 Re-ranker"""

    MODEL_CONFIGS = {
        "bge-reranker-base": {
            "full_name": "BAAI/bge-reranker-base",
            "max_length": 512,
            "description": "범용 re-ranker, 빠른 속도"
        },
        "bge-reranker-v2-m3": {
            "full_name": "BAAI/bge-reranker-v2-m3",
            "max_length": 8192,
            "description": "다국어 지원, 긴 문서"
        },
        "ms-marco-MiniLM": {
            "full_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "max_length": 512,
            "description": "MS MARCO 학습, 경량"
        }
    }
```

**Bi-Encoder vs Cross-Encoder:**
- Bi-Encoder: Query와 Document 각각 인코딩 후 유사도 계산 (빠름)
- Cross-Encoder: Query-Document 쌍을 함께 인코딩 (정확함, Re-ranking에 적합)

### 3. DART 공시 데이터 수집 (`src/data/dart_collector.py`)

실제 한국 기업 공시 데이터 수집:

```python
class DARTCollector:
    """DART Open API 기반 공시 데이터 수집기"""

    MAJOR_CORPS = [
        ("삼성전자", "00126380"), ("SK하이닉스", "00164779"),
        ("LG에너지솔루션", "01620808"), ("현대차", "00164788"),
        # ... 20개 주요 기업
    ]

    async def collect_disclosures(
        self,
        corp_code: str,
        bgn_de: str,
        end_de: str
    ) -> List[Disclosure]:
        """특정 기업의 공시 목록 수집"""
```

**데이터 규모:**
- 20개 주요 기업 × 연간 500+ 공시 = **10,000+ 문서**
- 사업보고서, 분기보고서, 주요사항보고 등

### 4. 벤치마크 프레임워크 (`src/rag/benchmark.py`)

자동화된 성능 측정:

```python
class RAGBenchmark:
    """RAG 시스템 벤치마크"""

    def evaluate(self, results: List[SearchResult], ground_truth: List[str]):
        return {
            "precision_at_k": self._precision_at_k(results, ground_truth, k=5),
            "recall_at_k": self._recall_at_k(results, ground_truth, k=5),
            "mrr": self._mean_reciprocal_rank(results, ground_truth),
            "ndcg": self._ndcg(results, ground_truth, k=5)
        }
```

**평가 데이터셋:** 100개 질의 (`src/data/evaluation_dataset.py`)
- 실적 관련 (25개): "삼성전자 2024년 1분기 영업이익은?"
- 주가/시장 (20개): "코스피 시가총액 상위 5개 기업은?"
- 산업/기술 (20개): "HBM 메모리 주요 생산 기업은?"
- 투자/재무 (15개): "배당수익률이 높은 금융주는?"
- 공시/규제 (10개): "분기보고서 제출 기한은?"
- 기업정보 (10개): "현대차그룹 계열사 목록은?"

---

## 프로젝트 구조

```
finance-rag-api/
├── src/
│   ├── rag/
│   │   ├── hybrid_search.py    # Hybrid Search + Kiwi 토크나이저
│   │   ├── reranker.py         # Cross-Encoder Re-ranking
│   │   ├── rag_service.py      # RAG 서비스 레이어
│   │   ├── chunking.py         # 문서 청킹 전략
│   │   ├── conversation.py     # 멀티턴 대화 관리
│   │   ├── evaluation.py       # RAG 평가 지표
│   │   ├── benchmark.py        # 벤치마크 프레임워크
│   │   └── llm_provider.py     # LLM 추상화 레이어
│   ├── data/
│   │   ├── dart_collector.py   # DART 공시 수집
│   │   ├── news_collector.py   # 금융 뉴스 수집
│   │   ├── load_to_rag.py      # RAG 데이터 로더
│   │   └── evaluation_dataset.py  # 100개 평가 데이터
│   ├── core/
│   │   ├── config.py           # 설정 관리
│   │   └── exceptions.py       # 예외 처리
│   ├── realtime/
│   │   ├── dart_sync.py        # DART API 실시간 동기화
│   │   ├── websocket_manager.py # WebSocket 알림 관리
│   │   └── streaming.py        # SSE 스트리밍 응답
│   └── api/
│       ├── routes/             # FastAPI 라우터
│       └── deps.py             # 의존성 주입
├── app/
│   └── streamlit_app.py        # Streamlit 데모 UI
├── tests/                      # pytest 테스트 (80+개)
└── docs/                       # 기술 문서
```

---

## 빠른 시작

### 설치

```bash
git clone https://github.com/araeLaver/AI-ML.git
cd AI-ML/finance-rag-api
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 환경 설정

```bash
cp .env.example .env
# .env 편집:
# GROQ_API_KEY=your_groq_api_key
# DART_API_KEY=your_dart_api_key  # 공시 수집용 (선택)
```

### 데이터 수집 (선택)

```bash
# DART 공시 데이터 수집
python -m src.data.dart_collector

# 금융 뉴스 수집
python -m src.data.news_collector

# RAG 시스템에 로드
python -m src.data.load_to_rag
```

### 실행

```bash
# Streamlit 데모
streamlit run app/streamlit_app.py --server.port 8502

# FastAPI 서버
uvicorn src.api.main:app --reload --port 8000
```

### 벤치마크 실행

```bash
# 전체 벤치마크 실행
python -m src.rag.benchmark

# 특정 컴포넌트만 테스트
python -m src.rag.benchmark --component tokenizer
python -m src.rag.benchmark --component reranker
```

---

## 라이브 데모

> **Live Demo**: [Finance RAG on Streamlit Cloud](https://lffna9osmmgfndbczgk2d5.streamlit.app)

---

## 기술적 의사결정

### Q: 왜 Kiwi를 선택했나요?

| 토크나이저 | 장점 | 단점 |
|-----------|------|------|
| **Kiwi** | 속도 빠름, 사용자 사전 지원, 금융 용어 인식 | Python 바인딩 필요 |
| Mecab | 매우 빠름 | 설치 복잡, 사전 관리 어려움 |
| Okt (KoNLPy) | 사용 쉬움 | 속도 느림, 정확도 낮음 |

**결론**: 속도와 확장성(사용자 사전)을 고려해 Kiwi 선택

### Q: 왜 Cross-Encoder를 직접 구현했나요?

- **기존 문제**: LangChain의 Cross-Encoder는 래퍼만 제공, 실제 구현 필요
- **해결**: sentence-transformers 기반 실제 Cross-Encoder 구현
- **효과**: Keyword Reranker 대비 Precision@5 14% 향상

### Q: Hybrid Search에서 RRF 가중치는?

```python
# Reciprocal Rank Fusion
rrf_score = sum(1 / (k + rank) for method in [vector, bm25])
# k=60이 일반적, 금융 도메인에서는 k=40이 더 효과적 (키워드 중요)
```

---

## 실시간 기능 (Phase 6)

### 실시간 공시 연동

DART API와 실시간으로 연동하여 새 공시를 자동으로 수집합니다:

```python
from src.realtime import start_sync_scheduler, get_sync_status

# 스케줄러 시작 (매시간 동기화)
start_sync_scheduler()

# 상태 확인
status = get_sync_status()
print(f"마지막 동기화: {status['last_sync']}")
```

### WebSocket 실시간 알림

새 공시 발생 시 연결된 클라이언트에게 실시간 알림:

```python
from src.realtime import WebSocketManager, subscribe_to_company

# 회사별 구독
await subscribe_to_company(connection_id, "00126380")  # 삼성전자
```

### SSE 스트리밍 응답

LLM 응답을 실시간으로 스트리밍하여 사용자 경험 개선:

```python
from src.realtime import stream_rag_response, create_sse_response

async def stream_query(query: str):
    generator = stream_rag_response(query, rag_service)
    return create_sse_response(generator)
```

---

## 고급 기능

### Query Expansion

금융 도메인 동의어/약어를 자동 확장하여 검색 재현율을 향상시킵니다:

```python
from src.rag.query_expansion import QueryExpander

expander = QueryExpander()
result = expander.expand("PER 분석")
# expanded_terms: ["주가수익비율", "P/E", ...]
```

### A/B 테스트 프레임워크

RAG 구성 비교 실험을 위한 프레임워크:

```python
from src.rag.ab_testing import RAGExperiment

rag_exp = RAGExperiment()
config = rag_exp.create_from_template("reranker_comparison")
```

### Fine-tuned Embedding

금융 도메인 특화 임베딩 모델 학습:

```python
from src.rag.fine_tuned_embedding import FinanceEmbeddingModel

model = FinanceEmbeddingModel(use_pretrained="ko-sroberta")
embeddings = model.encode(["삼성전자 실적", "SK하이닉스 반도체"])
```

### Multi-modal 처리

공시 문서 내 표/차트 인식:

```python
from src.rag.multimodal import MultiModalProcessor

processor = MultiModalProcessor()
extracted = processor.process_document(html_content, "html")
```

---

## 향후 개선 계획

- [x] **Fine-tuned Embedding**: 금융 도메인 특화 임베딩 모델 ✅
- [x] **Query Expansion**: 금융 동의어 확장 (PER ↔ 주가수익비율) ✅
- [x] **Real-time Update**: 실시간 공시 연동 ✅
- [x] **Multi-modal**: 공시 내 표/차트 인식 ✅
- [ ] **LLM Fine-tuning**: 금융 QA 특화 LLM 학습
- [ ] **Knowledge Graph**: 기업 관계 그래프 구축

---

## 라이선스

MIT License

## 연락처

- **GitHub**: https://github.com/araeLaver
- **Email**: araelaver@gmail.com
