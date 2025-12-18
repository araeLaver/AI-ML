# Step 3: RAG (검색 증강 생성) 시스템 (2-3개월)

## 목표
> RAG 파이프라인 설계 및 구축 능력 확보

## 왜 RAG가 가장 중요한가?

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG의 핵심 가치                           │
├─────────────────────────────────────────────────────────────┤
│  LLM 환각(Hallucination) 최대 70% 감소                      │
│  응답 정확도 40-60% 향상                                    │
│  파인튜닝 없이 도메인 지식 적용 가능                         │
│  실시간 최신 데이터 반영 가능                                │
│  비용 효율적 (파인튜닝 대비 10배 이상 저렴)                  │
└─────────────────────────────────────────────────────────────┘
```

**김다운님 강점**: 헥토데이터에서 데이터 파이프라인 경험 → RAG 파이프라인에 직접 적용

---

## RAG 아키텍처 이해

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG 전체 흐름                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [문서]  →  [청킹]  →  [임베딩]  →  [벡터DB 저장]                   │
│     │                                       │                       │
│     │         ┌────────────────────────────┘                       │
│     │         ↓                                                     │
│  [질의]  →  [질의 임베딩]  →  [유사도 검색]  →  [관련 문서]         │
│                                                    │                │
│                                                    ↓                │
│                              [프롬프트 = 질의 + 관련 문서]          │
│                                                    │                │
│                                                    ↓                │
│                                               [LLM 응답]            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 학습 순서

### Week 1-2: 임베딩 모델 이해

#### 임베딩이란?
텍스트를 수치 벡터로 변환하여 의미적 유사성을 계산 가능하게 함

```python
from openai import OpenAI

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    """텍스트를 임베딩 벡터로 변환"""
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

# 예시
text1 = "금융 거래 이상 탐지"
text2 = "이상 금융 거래 발견"
text3 = "오늘 날씨가 좋습니다"

emb1 = get_embedding(text1)
emb2 = get_embedding(text2)
emb3 = get_embedding(text3)

# 유사도 계산
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"text1 vs text2: {cosine_similarity(emb1, emb2):.4f}")  # 높음 (~0.95)
print(f"text1 vs text3: {cosine_similarity(emb1, emb3):.4f}")  # 낮음 (~0.30)
```

#### 주요 임베딩 모델 비교

| 모델 | 차원 | 특징 | 비용 |
|------|------|------|------|
| OpenAI text-embedding-3-small | 1536 | 가성비 최고 | $0.02/1M tokens |
| OpenAI text-embedding-3-large | 3072 | 최고 성능 | $0.13/1M tokens |
| Sentence-BERT | 768 | 오픈소스, 무료 | 무료 |
| BGE (BAAI) | 1024 | 한국어 성능 좋음 | 무료 |
| Cohere embed | 1024 | 다국어 지원 | 유료 |

#### 오픈소스 임베딩 사용

```python
from sentence_transformers import SentenceTransformer

# 한국어에 좋은 모델
model = SentenceTransformer('BAAI/bge-m3')

texts = ["금융 거래 이상 탐지", "이상 금융 거래 발견"]
embeddings = model.encode(texts)
```

---

### Week 3-4: 벡터 데이터베이스

#### ChromaDB (로컬 개발용, 입문자 추천)

```python
import chromadb
from chromadb.utils import embedding_functions

# 클라이언트 생성
client = chromadb.PersistentClient(path="./chroma_db")

# OpenAI 임베딩 함수
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-api-key",
    model_name="text-embedding-3-small"
)

# 컬렉션 생성
collection = client.create_collection(
    name="financial_docs",
    embedding_function=openai_ef
)

# 문서 추가
collection.add(
    documents=[
        "금융 거래에서 이상 패턴을 탐지하는 방법",
        "신용 점수 계산의 핵심 요소",
        "리스크 관리의 기본 원칙"
    ],
    metadatas=[
        {"category": "fraud", "date": "2024-01"},
        {"category": "credit", "date": "2024-02"},
        {"category": "risk", "date": "2024-03"}
    ],
    ids=["doc1", "doc2", "doc3"]
)

# 검색
results = collection.query(
    query_texts=["이상 거래를 어떻게 찾나요?"],
    n_results=2
)
print(results['documents'])
```

#### Pinecone (프로덕션용 클라우드)

```python
from pinecone import Pinecone, ServerlessSpec

# 초기화
pc = Pinecone(api_key="your-api-key")

# 인덱스 생성
pc.create_index(
    name="financial-rag",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

index = pc.Index("financial-rag")

# 벡터 업서트
index.upsert(
    vectors=[
        {"id": "doc1", "values": embedding, "metadata": {"text": "원본 텍스트"}},
    ]
)

# 검색
results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
```

#### 벡터 DB 비교

| DB | 특징 | 용도 | 비용 |
|------|------|------|------|
| ChromaDB | 간단, 로컬 | 개발/프로토타입 | 무료 |
| Pinecone | 관리형, 확장성 | 프로덕션 | 종량제 |
| Milvus | 오픈소스, 고성능 | 대규모 | 무료(셀프호스팅) |
| Weaviate | GraphQL, 하이브리드 검색 | 복잡한 쿼리 | 무료/유료 |
| pgvector | PostgreSQL 확장 | 기존 PG 사용자 | 무료 |

---

### Week 5-6: 청킹 전략

#### 청킹이 중요한 이유
- 너무 작으면: 컨텍스트 부족
- 너무 크면: 노이즈 증가, 비용 증가
- 최적의 청크 사이즈: 보통 500-1000 토큰

#### 청킹 방법

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter
)

# 1. 재귀적 문자 분할 (가장 많이 사용)
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,  # 오버랩으로 컨텍스트 유지
    separators=["\n\n", "\n", ".", " ", ""]
)

# 2. 토큰 기반 분할
token_splitter = TokenTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# 3. 시맨틱 청킹 (의미 단위로 분할)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

semantic_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile"
)
```

#### 금융 문서 특화 청킹

```python
def financial_document_chunker(text: str) -> list:
    """금융 문서 특화 청킹"""
    chunks = []

    # 섹션별 분리 (목차, 요약, 상세 내용 등)
    sections = text.split("\n## ")

    for section in sections:
        # 각 섹션을 적절한 크기로 분할
        if len(section) > 1500:
            sub_chunks = recursive_splitter.split_text(section)
            chunks.extend(sub_chunks)
        else:
            chunks.append(section)

    return chunks
```

---

### Week 7-8: 검색 최적화

#### Hybrid Search (키워드 + 시맨틱)

```python
from rank_bm25 import BM25Okapi

class HybridSearch:
    def __init__(self, documents, embeddings):
        # BM25 (키워드 검색)
        tokenized = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

        # 벡터 검색
        self.embeddings = embeddings
        self.documents = documents

    def search(self, query, alpha=0.5, top_k=5):
        # BM25 점수
        bm25_scores = self.bm25.get_scores(query.split())

        # 벡터 유사도 점수
        query_emb = get_embedding(query)
        vector_scores = [
            cosine_similarity(query_emb, emb)
            for emb in self.embeddings
        ]

        # 하이브리드 점수 (가중 평균)
        hybrid_scores = [
            alpha * bm25 + (1-alpha) * vec
            for bm25, vec in zip(bm25_scores, vector_scores)
        ]

        # 상위 K개 반환
        top_indices = np.argsort(hybrid_scores)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]
```

#### Re-Ranker (검색 결과 재정렬)

```python
from sentence_transformers import CrossEncoder

# Cross-Encoder 기반 Re-Ranker
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_results(query, documents, top_k=3):
    """검색 결과를 재정렬"""
    pairs = [[query, doc] for doc in documents]
    scores = reranker.predict(pairs)

    ranked_indices = np.argsort(scores)[::-1][:top_k]
    return [documents[i] for i in ranked_indices]
```

#### Query Expansion (쿼리 확장)

```python
def expand_query(query: str) -> str:
    """LLM을 사용한 쿼리 확장"""
    prompt = f"""
    다음 질문을 검색에 최적화된 형태로 확장해주세요.
    동의어와 관련 키워드를 포함해주세요.

    원본 질문: {query}

    확장된 검색 쿼리:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content
```

---

### Week 9-10: LangChain/LlamaIndex 활용

#### LangChain RAG 구현

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 임베딩 및 벡터스토어
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# 프롬프트 템플릿
prompt_template = """
당신은 금융 전문가입니다. 아래 참고 자료를 바탕으로 질문에 답변하세요.
확실하지 않은 내용은 "확인이 필요합니다"라고 말해주세요.

참고 자료:
{context}

질문: {question}

답변:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# RAG 체인 구성
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o", temperature=0),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# 질의
result = qa_chain.invoke({"query": "이상 거래 탐지 방법은?"})
print(result["result"])
print("출처:", result["source_documents"])
```

#### LlamaIndex RAG 구현

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI

# 설정
Settings.llm = OpenAI(model="gpt-4o", temperature=0)

# 문서 로드 및 인덱스 생성
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# 쿼리 엔진
query_engine = index.as_query_engine(similarity_top_k=3)

# 질의
response = query_engine.query("금융 리스크 관리 방법은?")
print(response)
```

---

## 실습 프로젝트

### 프로젝트: 금융 문서 RAG 시스템

**김다운님의 첫 번째 포트폴리오 프로젝트**

```
financial_rag/
├── app/
│   ├── __init__.py
│   ├── embeddings.py       # 임베딩 처리
│   ├── vectorstore.py      # 벡터 DB 관리
│   ├── chunker.py          # 문서 청킹
│   ├── retriever.py        # 검색 로직
│   ├── chain.py            # RAG 체인
│   └── api.py              # FastAPI 엔드포인트
├── data/
│   └── financial_docs/     # 금융 문서들
├── tests/
├── docker-compose.yml
├── requirements.txt
└── main.py
```

**구현 기능**:
1. PDF/문서 업로드 및 청킹
2. 벡터 DB 저장
3. 하이브리드 검색
4. 출처 표시가 포함된 답변 생성
5. FastAPI 기반 REST API

---

## 체크리스트

### 임베딩
- [ ] OpenAI 임베딩 API 사용
- [ ] 오픈소스 임베딩 모델 사용 (Sentence-BERT, BGE)
- [ ] 코사인 유사도 계산

### 벡터 DB
- [ ] ChromaDB 설치 및 사용
- [ ] 문서 추가, 검색, 삭제
- [ ] 메타데이터 필터링

### 청킹
- [ ] 재귀적 문자 분할
- [ ] 청크 사이즈 최적화
- [ ] 오버랩 설정

### 검색 최적화
- [ ] Hybrid Search 구현
- [ ] Re-Ranker 적용
- [ ] Query Expansion

### 프레임워크
- [ ] LangChain RAG 파이프라인
- [ ] LlamaIndex 기본 사용

---

## 다음 단계
Step 3 완료 후 → **Step 4: AI Agent 개발** 으로 진행
