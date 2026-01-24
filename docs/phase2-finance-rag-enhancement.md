# Phase 2: Finance RAG 고도화 상세 계획

> **기간**: Q1 2026 Week 3-6
> **목표**: 검색 정확도 추가 20% 향상, 사용자 경험 개선

---

## 현재 상태 (v1.2.0)

### 구현 완료
- ✅ Hybrid Search (Vector + BM25 + RRF)
- ✅ Kiwi 한국어 형태소 분석기
- ✅ Cross-Encoder Re-ranking
- ✅ DART 공시 데이터 10,000+ 문서
- ✅ 벤치마크 프레임워크 (100개 평가 데이터셋)

### 성능 현황
| 메트릭 | 현재 값 |
|--------|---------|
| Precision@5 | 0.91 |
| 전체 응답시간 | < 1.2초 |
| 테스트 통과율 | 53/53 (100%) |

---

## Task 1: Fine-tuned Embedding

### 1.1 목표
금융 도메인 특화 임베딩 모델을 학습하여 검색 정확도 향상

### 1.2 현재 문제점
```
현재: 범용 임베딩 모델 (intfloat/e5-base 또는 sentence-transformers)
문제: 금융 용어 간 의미적 유사도가 부정확
  - "PER" vs "주가수익비율" → 낮은 유사도
  - "영업이익" vs "operating profit" → 낮은 유사도
  - "삼성전자 실적" vs "삼성전자 2024년 1분기 영업이익" → 중간 유사도
```

### 1.3 구현 계획

#### Step 1: 학습 데이터 준비
```
finance-rag-api/
├── data/
│   └── embedding_training/
│       ├── query_document_pairs.jsonl    # 쿼리-문서 쌍 10,000+
│       ├── financial_synonyms.jsonl       # 금융 동의어 쌍
│       └── hard_negatives.jsonl           # 어려운 부정 샘플
```

**데이터 구조:**
```json
// query_document_pairs.jsonl
{
  "query": "삼성전자 2024년 1분기 영업이익",
  "positive": "삼성전자_2024Q1_사업보고서.txt: 영업이익 6조 6,000억원...",
  "negative": "SK하이닉스_2024Q1_사업보고서.txt: 영업이익 2조 8,000억원..."
}

// financial_synonyms.jsonl
{
  "term1": "PER",
  "term2": "주가수익비율",
  "similarity": 1.0
}
```

**데이터 수집 방법:**
1. 기존 평가 데이터셋 확장 (100 → 1,000+)
2. DART 공시 제목-본문 쌍 추출
3. LLM 기반 쿼리 자동 생성
4. 금융 용어 사전에서 동의어 쌍 추출

#### Step 2: 모델 선택 및 학습
```python
# src/rag/embedding_trainer.py

from sentence_transformers import SentenceTransformer, losses
from peft import LoraConfig, get_peft_model

class FinancialEmbeddingTrainer:
    """금융 도메인 특화 임베딩 학습"""

    BASE_MODELS = [
        "intfloat/e5-base",           # 다국어 지원
        "BAAI/bge-base-en-v1.5",      # 높은 성능
        "jhgan/ko-sbert-sts",         # 한국어 특화
    ]

    def __init__(self, base_model: str = "intfloat/e5-base"):
        self.model = SentenceTransformer(base_model)

    def apply_lora(self, r: int = 16, alpha: int = 32):
        """LoRA 적용으로 효율적 학습"""
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_lin", "v_lin"],
            lora_dropout=0.1,
        )
        self.model = get_peft_model(self.model, lora_config)

    def train(self, train_dataset, eval_dataset):
        """학습 실행"""
        train_loss = losses.MultipleNegativesRankingLoss(self.model)

        self.model.fit(
            train_objectives=[(train_dataset, train_loss)],
            epochs=3,
            warmup_steps=100,
            evaluator=eval_dataset,
            output_path="models/finance-embedding-v1",
        )
```

#### Step 3: 평가 메트릭
```python
# src/rag/embedding_evaluation.py

class EmbeddingEvaluator:
    """임베딩 모델 평가"""

    METRICS = [
        "MRR",           # Mean Reciprocal Rank
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "NDCG@10",       # Normalized Discounted Cumulative Gain
    ]

    def evaluate(self, model, test_queries, relevant_docs):
        results = {}
        for metric in self.METRICS:
            results[metric] = self._compute_metric(
                model, test_queries, relevant_docs, metric
            )
        return results

    def compare_models(self, baseline, finetuned, test_data):
        """기존 모델 vs Fine-tuned 비교"""
        baseline_results = self.evaluate(baseline, *test_data)
        finetuned_results = self.evaluate(finetuned, *test_data)

        return {
            "baseline": baseline_results,
            "finetuned": finetuned_results,
            "improvement": {
                k: (finetuned_results[k] - baseline_results[k]) / baseline_results[k] * 100
                for k in self.METRICS
            }
        }
```

### 1.4 예상 결과
| 메트릭 | 현재 | 목표 | 향상 |
|--------|------|------|------|
| MRR | 0.75 | 0.85 | +13% |
| Recall@5 | 0.82 | 0.92 | +12% |
| 동의어 유사도 | 0.45 | 0.90 | +100% |

### 1.5 파일 구조
```
finance-rag-api/
├── src/rag/
│   ├── embedding.py              # 기존 (수정)
│   ├── embedding_trainer.py      # 신규
│   └── embedding_evaluation.py   # 신규
├── data/embedding_training/
│   ├── query_document_pairs.jsonl
│   ├── financial_synonyms.jsonl
│   └── hard_negatives.jsonl
├── models/
│   └── finance-embedding-v1/     # 학습된 모델
└── scripts/
    ├── prepare_training_data.py
    └── train_embedding.py
```

---

## Task 2: Query Expansion

### 2.1 목표
금융 동의어 사전 기반 쿼리 자동 확장

### 2.2 현재 문제점
```
현재: 사용자 쿼리 그대로 검색
문제: 동의어/약어 사용 시 검색 누락
  - "PER 높은 기업" → "주가수익비율" 포함 문서 누락
  - "삼전 실적" → "삼성전자" 포함 문서 누락
```

### 2.3 구현 계획

#### Step 1: 금융 동의어 사전 구축
```python
# src/rag/financial_dictionary.py

FINANCIAL_SYNONYMS = {
    # 재무 지표
    "PER": ["주가수익비율", "P/E ratio", "Price to Earnings"],
    "PBR": ["주가순자산비율", "P/B ratio", "Price to Book"],
    "ROE": ["자기자본이익률", "Return on Equity"],
    "ROA": ["총자산이익률", "Return on Assets"],
    "EPS": ["주당순이익", "Earnings Per Share"],
    "BPS": ["주당순자산", "Book value Per Share"],
    "EBITDA": ["감가상각전영업이익", "세전영업이익"],
    "EBIT": ["영업이익", "Operating Income"],

    # 기업 약어
    "삼전": ["삼성전자", "Samsung Electronics"],
    "하닉": ["SK하이닉스", "SK Hynix"],
    "네카라쿠배": ["네이버", "카카오", "라인", "쿠팡", "배달의민족"],

    # 산업 용어
    "HBM": ["고대역폭메모리", "High Bandwidth Memory"],
    "AI반도체": ["인공지능반도체", "AI칩", "NPU"],
    "2차전지": ["배터리", "리튬이온배터리", "EV배터리"],

    # 공시 용어
    "사업보고서": ["연간보고서", "Annual Report"],
    "분기보고서": ["분기실적", "Quarterly Report"],
    "반기보고서": ["반기실적", "Semi-annual Report"],

    # 시장 용어
    "상장": ["IPO", "기업공개"],
    "유상증자": ["증자", "신주발행"],
    "무상증자": ["무상주", "주식배당"],

    # ... 200+ 항목
}

# 역방향 매핑 자동 생성
REVERSE_SYNONYMS = {}
for key, synonyms in FINANCIAL_SYNONYMS.items():
    for syn in synonyms:
        REVERSE_SYNONYMS[syn.lower()] = key
```

#### Step 2: 쿼리 확장 로직
```python
# src/rag/query_expander.py

from typing import List, Set
from .financial_dictionary import FINANCIAL_SYNONYMS, REVERSE_SYNONYMS

class QueryExpander:
    """금융 도메인 쿼리 확장"""

    def __init__(self, kiwi_tokenizer):
        self.tokenizer = kiwi_tokenizer
        self.synonyms = FINANCIAL_SYNONYMS
        self.reverse = REVERSE_SYNONYMS

    def expand(self, query: str, max_expansions: int = 3) -> List[str]:
        """쿼리 확장"""
        tokens = self.tokenizer.tokenize(query)
        expanded_queries = [query]

        for token in tokens:
            # 정방향 검색 (PER → 주가수익비율)
            if token.upper() in self.synonyms:
                for syn in self.synonyms[token.upper()][:max_expansions]:
                    expanded = query.replace(token, syn)
                    expanded_queries.append(expanded)

            # 역방향 검색 (주가수익비율 → PER)
            elif token.lower() in self.reverse:
                canonical = self.reverse[token.lower()]
                expanded = query.replace(token, canonical)
                expanded_queries.append(expanded)

        return list(set(expanded_queries))

    def get_synonyms(self, term: str) -> Set[str]:
        """특정 용어의 동의어 반환"""
        result = {term}

        if term.upper() in self.synonyms:
            result.update(self.synonyms[term.upper()])
        if term.lower() in self.reverse:
            canonical = self.reverse[term.lower()]
            result.add(canonical)
            result.update(self.synonyms.get(canonical, []))

        return result
```

#### Step 3: 검색 파이프라인 통합
```python
# src/rag/hybrid_search.py (수정)

class HybridSearcher:
    def __init__(self, ...):
        self.query_expander = QueryExpander(self.tokenizer)

    def search(self, query: str, top_k: int = 10) -> List[Document]:
        # 1. 쿼리 확장
        expanded_queries = self.query_expander.expand(query)

        # 2. 모든 확장 쿼리로 검색
        all_results = []
        for exp_query in expanded_queries:
            results = self._search_single(exp_query, top_k)
            all_results.extend(results)

        # 3. 중복 제거 및 점수 통합
        merged = self._merge_results(all_results)

        # 4. Re-ranking
        return self.reranker.rerank(query, merged, top_k)
```

### 2.4 예상 결과
| 쿼리 유형 | 현재 Recall | 목표 Recall |
|-----------|-------------|-------------|
| 약어 사용 (PER, ROE) | 0.65 | 0.90 |
| 기업 약어 (삼전) | 0.40 | 0.95 |
| 영문 혼용 | 0.55 | 0.85 |

---

## Task 3: Redis 캐싱

### 3.1 목표
자주 검색하는 쿼리 캐싱으로 응답 시간 단축

### 3.2 현재 문제점
```
현재: 모든 쿼리마다 전체 파이프라인 실행
문제: 동일/유사 쿼리 반복 시 불필요한 연산
  - 벡터 검색: ~35ms
  - Cross-Encoder: ~150ms
  - LLM 생성: ~800ms
```

### 3.3 구현 계획

#### Step 1: 캐싱 레이어 설계
```python
# src/cache/redis_cache.py

import redis
import hashlib
import json
from typing import Optional, Any
from datetime import timedelta

class RAGCache:
    """RAG 파이프라인 캐싱"""

    CACHE_LAYERS = {
        "embedding": timedelta(days=7),      # 임베딩 결과
        "search": timedelta(hours=1),        # 검색 결과
        "rerank": timedelta(hours=1),        # Re-ranking 결과
        "response": timedelta(minutes=30),   # LLM 응답
    }

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)

    def _make_key(self, layer: str, query: str, **kwargs) -> str:
        """캐시 키 생성"""
        data = {"query": query, **kwargs}
        hash_input = json.dumps(data, sort_keys=True)
        hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        return f"rag:{layer}:{hash_value}"

    def get(self, layer: str, query: str, **kwargs) -> Optional[Any]:
        """캐시 조회"""
        key = self._make_key(layer, query, **kwargs)
        data = self.redis.get(key)
        return json.loads(data) if data else None

    def set(self, layer: str, query: str, value: Any, **kwargs):
        """캐시 저장"""
        key = self._make_key(layer, query, **kwargs)
        ttl = self.CACHE_LAYERS[layer]
        self.redis.setex(key, ttl, json.dumps(value))

    def invalidate(self, pattern: str = "rag:*"):
        """캐시 무효화"""
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)
```

#### Step 2: 파이프라인 통합
```python
# src/rag/rag_service.py (수정)

class RAGService:
    def __init__(self, ..., cache: Optional[RAGCache] = None):
        self.cache = cache

    async def query(self, query: str, use_cache: bool = True) -> RAGResponse:
        # 1. 응답 캐시 확인
        if use_cache and self.cache:
            cached = self.cache.get("response", query)
            if cached:
                return RAGResponse(**cached, from_cache=True)

        # 2. 검색 캐시 확인
        if use_cache and self.cache:
            search_results = self.cache.get("search", query)
        else:
            search_results = None

        if not search_results:
            search_results = await self.searcher.search(query)
            if self.cache:
                self.cache.set("search", query, search_results)

        # 3. Re-ranking (캐시)
        # 4. LLM 생성
        # 5. 응답 캐싱

        response = await self._generate_response(query, search_results)
        if self.cache:
            self.cache.set("response", query, response.dict())

        return response
```

#### Step 3: 캐시 통계 API
```python
# src/api/routes.py (추가)

@router.get("/cache/stats")
async def get_cache_stats():
    """캐시 통계 조회"""
    return {
        "total_keys": cache.redis.dbsize(),
        "memory_usage": cache.redis.info()["used_memory_human"],
        "hit_rate": cache.get_hit_rate(),
        "layers": {
            layer: cache.redis.keys(f"rag:{layer}:*").__len__()
            for layer in RAGCache.CACHE_LAYERS
        }
    }
```

### 3.4 예상 결과
| 시나리오 | 현재 | 캐시 적용 후 |
|----------|------|-------------|
| 동일 쿼리 | 1.2초 | < 50ms |
| 유사 쿼리 | 1.2초 | < 300ms |
| Cache Hit Rate | 0% | 예상 40% |

---

## Task 4: 멀티턴 대화

### 4.1 목표
대화 컨텍스트 유지로 자연스러운 후속 질문 처리

### 4.2 현재 문제점
```
현재: 각 쿼리가 독립적
문제: 후속 질문 시 컨텍스트 손실
  - User: "삼성전자 2024년 1분기 실적 알려줘"
  - Bot: "삼성전자 2024년 1분기 영업이익은 6조 6,000억원..."
  - User: "SK하이닉스랑 비교해줘"  ← "1분기 실적" 컨텍스트 손실
```

### 4.3 구현 계획

#### Step 1: 대화 세션 관리
```python
# src/rag/conversation.py (확장)

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
import uuid

@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime
    sources: Optional[List[str]] = None

@dataclass
class ConversationSession:
    session_id: str
    messages: List[Message]
    context: dict  # 추출된 컨텍스트 (기업명, 기간, 지표 등)
    created_at: datetime

class ConversationManager:
    """멀티턴 대화 관리"""

    def __init__(self, cache: RAGCache):
        self.cache = cache
        self.max_history = 10

    def create_session(self) -> str:
        """새 세션 생성"""
        session_id = str(uuid.uuid4())
        session = ConversationSession(
            session_id=session_id,
            messages=[],
            context={},
            created_at=datetime.now()
        )
        self._save_session(session)
        return session_id

    def add_message(self, session_id: str, role: str, content: str, sources: List[str] = None):
        """메시지 추가"""
        session = self._load_session(session_id)
        session.messages.append(Message(
            role=role,
            content=content,
            timestamp=datetime.now(),
            sources=sources
        ))

        # 컨텍스트 업데이트
        if role == "user":
            session.context = self._extract_context(content, session.context)

        self._save_session(session)

    def _extract_context(self, query: str, prev_context: dict) -> dict:
        """쿼리에서 컨텍스트 추출"""
        context = prev_context.copy()

        # 기업명 추출
        companies = self._extract_companies(query)
        if companies:
            context["companies"] = companies

        # 기간 추출
        period = self._extract_period(query)
        if period:
            context["period"] = period

        # 지표 추출
        metrics = self._extract_metrics(query)
        if metrics:
            context["metrics"] = metrics

        return context
```

#### Step 2: 쿼리 재작성
```python
# src/rag/query_rewriter.py

class QueryRewriter:
    """컨텍스트 기반 쿼리 재작성"""

    def rewrite(self, query: str, context: dict) -> str:
        """컨텍스트를 반영하여 쿼리 재작성"""

        # 대명사/생략 해결
        rewritten = query

        # "비교해줘" → "삼성전자와 SK하이닉스 2024년 1분기 실적 비교"
        if "비교" in query and "companies" in context:
            if len(context["companies"]) >= 2:
                companies = " ".join(context["companies"])
                rewritten = f"{companies} {context.get('period', '')} 실적 비교"

        # "그러면 ROE는?" → "삼성전자 2024년 1분기 ROE"
        if self._is_follow_up(query):
            if "companies" in context:
                company = context["companies"][0]
                period = context.get("period", "")
                rewritten = f"{company} {period} {query}"

        return rewritten

    def _is_follow_up(self, query: str) -> bool:
        """후속 질문 여부 판단"""
        follow_up_patterns = [
            "그러면", "그럼", "그건", "이건",
            "그리고", "또", "다른",
            "비교", "차이",
        ]
        return any(p in query for p in follow_up_patterns)
```

### 4.4 API 수정
```python
# src/api/routes.py (수정)

@router.post("/query")
async def query(
    request: QueryRequest,
    session_id: Optional[str] = None
):
    """RAG 쿼리 (세션 지원)"""

    # 세션 관리
    if session_id:
        session = conversation_manager.get_session(session_id)
        rewritten_query = query_rewriter.rewrite(
            request.query,
            session.context
        )
    else:
        rewritten_query = request.query

    # RAG 실행
    response = await rag_service.query(rewritten_query)

    # 세션 업데이트
    if session_id:
        conversation_manager.add_message(session_id, "user", request.query)
        conversation_manager.add_message(session_id, "assistant", response.answer, response.sources)

    return response
```

---

## 구현 일정

```
Week 3 ─────────────────────────────────────────────
        │
        ├── Fine-tuned Embedding 데이터 준비
        └── 금융 동의어 사전 구축 (200+ 항목)

Week 4 ─────────────────────────────────────────────
        │
        ├── Embedding 모델 학습 및 평가
        └── Query Expansion 구현

Week 5 ─────────────────────────────────────────────
        │
        ├── Redis 캐싱 레이어 구현
        └── 통합 테스트

Week 6 ─────────────────────────────────────────────
        │
        ├── 멀티턴 대화 구현
        ├── 벤치마크 및 성능 측정
        └── 문서화
```

---

## 성공 기준

| 메트릭 | 현재 | 목표 | 측정 방법 |
|--------|------|------|----------|
| Precision@5 | 0.91 | 0.95 | 100개 평가 데이터셋 |
| 동의어 검색 성공률 | 65% | 95% | 금융 용어 테스트셋 |
| 평균 응답시간 | 1.2초 | 0.8초 | 캐시 적용 시 |
| 캐시 히트율 | 0% | 40% | Redis 통계 |
| 후속 질문 정확도 | N/A | 85% | 멀티턴 테스트셋 |

---

## 필요 리소스

### 인프라
- Redis 서버 (캐싱)
- GPU (임베딩 학습, 선택)

### 외부 API
- OpenAI API (데이터 생성)
- HuggingFace (모델 호스팅)

### 예상 비용
| 항목 | 비용 |
|------|------|
| OpenAI API (데이터 생성) | ~$50 |
| HuggingFace Pro (모델 호스팅) | $9/월 |
| Redis Cloud (Free tier) | $0 |

---

## 관련 문서

- [Finance RAG 아키텍처](./finance-rag-architecture.md)
- [벤치마크 가이드](./finance-rag-benchmark.md)
- [CHANGELOG](../CHANGELOG.md)
- [ROADMAP](../ROADMAP.md)
