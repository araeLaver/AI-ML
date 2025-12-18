# -*- coding: utf-8 -*-
"""
03. RAG 기초 - 포트폴리오의 핵심

[RAG란?]
Retrieval-Augmented Generation
= 검색(Retrieval) + 생성(Generation)
= 문서 검색 후 LLM이 답변 생성

[왜 RAG가 중요한가?]
- LLM 환각(Hallucination) 해결
- 최신 정보 / 내부 문서 활용 가능
- 기업에서 가장 많이 쓰는 AI 패턴

[포트폴리오 핵심]
이 실습 = 금융 문서 RAG 시스템의 기반

실행: python notebooks/03_rag_basics.py
"""

import ollama
import chromadb
from chromadb.utils import embedding_functions


def step_1_embedding_concept():
    """
    Step 1: 임베딩(Embedding) 개념 이해

    [핵심] 텍스트 → 숫자 벡터 변환
    - "왕" - "남자" + "여자" = "여왕" (벡터 연산 가능!)
    - 의미가 비슷한 문장 → 벡터도 비슷
    """
    print("=" * 60)
    print("Step 1: 임베딩(Embedding) 개념")
    print("=" * 60)

    # Ollama 임베딩 사용 (무료, 로컬)
    text1 = "How to invest in stocks?"
    text2 = "What is the best way to buy shares?"
    text3 = "What is the weather today?"

    # 임베딩 생성 (Ollama SDK는 Pydantic 모델 반환)
    emb1 = ollama.embed(model='llama3.2', input=text1).embeddings[0]
    emb2 = ollama.embed(model='llama3.2', input=text2).embeddings[0]
    emb3 = ollama.embed(model='llama3.2', input=text3).embeddings[0]

    # 코사인 유사도 계산
    def cosine_similarity(a, b):
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b)

    sim_12 = cosine_similarity(emb1, emb2)
    sim_13 = cosine_similarity(emb1, emb3)

    print(f"\n문장 1: '{text1}'")
    print(f"문장 2: '{text2}'")
    print(f"문장 3: '{text3}'")
    print(f"\n임베딩 차원: {len(emb1)}")
    print(f"\n[유사도 비교]")
    print(f"  문장1 vs 문장2 (비슷한 의미): {sim_12:.4f}")
    print(f"  문장1 vs 문장3 (다른 의미):   {sim_13:.4f}")
    print("\n→ 의미가 비슷하면 유사도가 높음!")


def step_2_vector_db():
    """
    Step 2: 벡터 DB (ChromaDB)

    [핵심] 임베딩을 저장하고 유사한 것을 검색하는 DB
    - SQL DB: SELECT WHERE column = value
    - Vector DB: SELECT WHERE embedding ≈ query_embedding
    """
    print("\n" + "=" * 60)
    print("Step 2: 벡터 DB (ChromaDB)")
    print("=" * 60)

    # ChromaDB 클라이언트 생성 (인메모리)
    client = chromadb.Client()

    # 커스텀 임베딩 함수 (Ollama 사용)
    class OllamaEmbedding(embedding_functions.EmbeddingFunction):
        def __call__(self, input):
            embeddings = []
            for text in input:
                result = ollama.embed(model='llama3.2', input=text)
                embeddings.append(result.embeddings[0])
            return embeddings

    # 컬렉션 생성 (테이블과 유사)
    collection = client.create_collection(
        name="finance_docs",
        embedding_function=OllamaEmbedding()
    )

    # 금융 문서 샘플 데이터
    documents = [
        "ETF는 Exchange Traded Fund의 약자로, 주식처럼 거래되는 펀드입니다.",
        "채권은 정부나 기업이 발행하는 빚 증서로, 만기에 원금을 돌려받습니다.",
        "주식은 기업의 소유권 일부를 나타내며, 배당과 시세차익을 기대할 수 있습니다.",
        "P2P 대출은 개인 간 직접 대출로, 중개 플랫폼을 통해 이루어집니다.",
        "리츠(REITs)는 부동산 투자 신탁으로, 부동산에 간접 투자할 수 있습니다.",
    ]

    # 문서 추가
    collection.add(
        documents=documents,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )

    print(f"\n[저장된 문서 수]: {collection.count()}")

    # 검색 테스트
    query = "주식처럼 사고 팔 수 있는 펀드가 뭐야?"
    results = collection.query(
        query_texts=[query],
        n_results=2
    )

    print(f"\n[검색 쿼리]: {query}")
    print(f"\n[검색 결과]:")
    for i, doc in enumerate(results['documents'][0]):
        distance = results['distances'][0][i]
        print(f"  {i+1}. (거리: {distance:.4f}) {doc}")

    return client, collection


def step_3_rag_pipeline(client, collection):
    """
    Step 3: RAG 파이프라인

    [전체 흐름]
    1. 사용자 질문 입력
    2. 질문을 임베딩으로 변환
    3. 벡터 DB에서 유사한 문서 검색
    4. 검색된 문서 + 질문을 LLM에 전달
    5. LLM이 문서 기반으로 답변 생성
    """
    print("\n" + "=" * 60)
    print("Step 3: RAG 파이프라인 (검색 → 생성)")
    print("=" * 60)

    def rag_query(question: str, collection, top_k: int = 3):
        """RAG 파이프라인 함수"""

        # 1. 벡터 DB에서 관련 문서 검색
        results = collection.query(
            query_texts=[question],
            n_results=top_k
        )

        retrieved_docs = results['documents'][0]
        context = "\n".join([f"- {doc}" for doc in retrieved_docs])

        # 2. 프롬프트 구성
        prompt = f"""다음 문서를 참고하여 질문에 답하세요.
문서에 없는 내용은 "해당 정보가 없습니다"라고 답하세요.

[참고 문서]
{context}

[질문]
{question}

[답변]"""

        # 3. LLM으로 답변 생성
        response = ollama.chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.3}
        )

        return {
            'answer': response.message.content,
            'sources': retrieved_docs
        }

    # 테스트 질문들
    questions = [
        "ETF가 뭐야?",
        "부동산에 간접 투자하는 방법은?",
        "비트코인이 뭐야?",  # 문서에 없는 질문
    ]

    for q in questions:
        print(f"\n[질문] {q}")
        result = rag_query(q, collection)
        print(f"[답변] {result['answer']}")
        print(f"[출처] {len(result['sources'])}개 문서 참조")


def step_4_chunking():
    """
    Step 4: 문서 청킹(Chunking)

    [핵심] 긴 문서를 작은 조각으로 나눔
    - LLM 컨텍스트 제한 대응
    - 검색 정확도 향상
    """
    print("\n" + "=" * 60)
    print("Step 4: 문서 청킹(Chunking)")
    print("=" * 60)

    # 긴 문서 샘플
    long_document = """
    금융 투자의 기본 원칙

    첫째, 분산 투자가 중요합니다. 모든 자산을 한 곳에 투자하면 위험이 집중됩니다.
    주식, 채권, 부동산 등 다양한 자산군에 나누어 투자해야 합니다.
    이를 통해 한 자산의 손실을 다른 자산의 이익으로 상쇄할 수 있습니다.

    둘째, 장기 투자를 권장합니다. 단기적인 시장 변동에 흔들리지 마세요.
    역사적으로 주식 시장은 장기적으로 우상향했습니다.
    복리 효과를 누리려면 최소 5-10년 이상의 투자 기간이 필요합니다.

    셋째, 비용을 최소화하세요. 수수료와 세금은 수익을 갉아먹습니다.
    ETF나 인덱스 펀드는 액티브 펀드보다 비용이 낮습니다.
    비용 1%의 차이가 30년 후에는 큰 금액 차이를 만듭니다.

    넷째, 자신만의 투자 원칙을 세우세요. 남의 말에 휩쓸리지 마세요.
    자신의 재무 상황과 위험 허용도를 파악하세요.
    정해진 원칙에 따라 일관되게 투자하세요.
    """

    def simple_chunking(text, chunk_size=200, overlap=50):
        """간단한 문자 기반 청킹"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
        return chunks

    def sentence_chunking(text, sentences_per_chunk=3):
        """문장 단위 청킹 (더 의미있는 단위)"""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = ' '.join(sentences[i:i+sentences_per_chunk])
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks

    print("\n[원본 문서 길이]:", len(long_document), "자")

    print("\n[문자 기반 청킹 (200자, 50자 오버랩)]")
    char_chunks = simple_chunking(long_document, 200, 50)
    for i, chunk in enumerate(char_chunks[:3]):
        print(f"  청크 {i+1}: {chunk[:60]}...")

    print("\n[문장 기반 청킹 (3문장씩)]")
    sent_chunks = sentence_chunking(long_document, 3)
    for i, chunk in enumerate(sent_chunks[:3]):
        print(f"  청크 {i+1}: {chunk[:60]}...")

    print(f"\n→ 문자 청킹: {len(char_chunks)}개, 문장 청킹: {len(sent_chunks)}개")
    print("→ 실무에서는 RecursiveCharacterTextSplitter (LangChain) 사용")


def step_5_complete_example():
    """
    Step 5: 완전한 RAG 예제 (금융 문서)

    [포트폴리오 미리보기]
    실제 금융 문서를 사용한 Q&A 시스템
    """
    print("\n" + "=" * 60)
    print("Step 5: 완전한 RAG 예제 (금융 도메인)")
    print("=" * 60)

    # 금융 문서 샘플 (실제로는 PDF에서 추출)
    finance_docs = [
        {
            "id": "policy_001",
            "source": "투자 가이드 2024",
            "content": "신규 투자자는 먼저 비상금을 확보해야 합니다. 최소 3-6개월 생활비를 예금으로 보유하세요. 투자는 여유 자금으로만 해야 합니다."
        },
        {
            "id": "policy_002",
            "source": "투자 가이드 2024",
            "content": "초보자에게 추천하는 투자 상품은 ETF입니다. 특히 S&P 500 추종 ETF는 미국 대형주에 분산 투자할 수 있어 리스크가 낮습니다."
        },
        {
            "id": "policy_003",
            "source": "위험 관리 매뉴얼",
            "content": "레버리지 ETF는 초보자에게 적합하지 않습니다. 일일 수익률의 2-3배를 추종하므로 손실도 확대됩니다. 전문 투자자만 활용하세요."
        },
        {
            "id": "market_001",
            "source": "시장 분석 리포트",
            "content": "2024년 금리 인하 기대로 채권 가격 상승이 예상됩니다. 채권 ETF나 채권형 펀드 비중 확대를 고려하세요."
        },
        {
            "id": "market_002",
            "source": "시장 분석 리포트",
            "content": "AI 반도체 수요 증가로 관련 종목 강세가 지속되고 있습니다. 다만 밸류에이션이 높아 단기 조정 가능성에 유의하세요."
        },
    ]

    # ChromaDB 설정
    client = chromadb.Client()

    class OllamaEmbedding(embedding_functions.EmbeddingFunction):
        def __call__(self, input):
            embeddings = []
            for text in input:
                result = ollama.embed(model='llama3.2', input=text)
                embeddings.append(result.embeddings[0])
            return embeddings

    collection = client.create_collection(
        name="finance_qa",
        embedding_function=OllamaEmbedding()
    )

    # 문서 저장
    collection.add(
        documents=[doc["content"] for doc in finance_docs],
        ids=[doc["id"] for doc in finance_docs],
        metadatas=[{"source": doc["source"]} for doc in finance_docs]
    )

    print(f"\n[저장된 문서]: {collection.count()}개")

    def finance_rag(question: str):
        """금융 RAG 시스템"""
        # 검색
        results = collection.query(
            query_texts=[question],
            n_results=2,
            include=["documents", "metadatas", "distances"]
        )

        # 컨텍스트 구성
        context_parts = []
        for i, doc in enumerate(results['documents'][0]):
            source = results['metadatas'][0][i]['source']
            context_parts.append(f"[{source}] {doc}")

        context = "\n".join(context_parts)

        # LLM 답변 생성
        response = ollama.chat(
            model='llama3.2',
            messages=[
                {
                    'role': 'system',
                    'content': '''당신은 금융 전문 상담사입니다.
제공된 문서만을 기반으로 답변하세요.
문서에 없는 내용은 "해당 정보가 문서에 없습니다"라고 답하세요.
답변 마지막에 참조한 문서를 표시하세요.'''
                },
                {
                    'role': 'user',
                    'content': f"[참고 문서]\n{context}\n\n[질문] {question}"
                }
            ],
            options={'temperature': 0.2}
        )

        return {
            'question': question,
            'answer': response.message.content,
            'sources': [m['source'] for m in results['metadatas'][0]]
        }

    # 테스트
    test_questions = [
        "초보자가 투자를 시작하려면 뭘 먼저 해야 해?",
        "레버리지 ETF가 뭐야? 투자해도 돼?",
        "지금 채권 투자하기 좋은 시기야?"
    ]

    for q in test_questions:
        print(f"\n{'─' * 50}")
        result = finance_rag(q)
        print(f"[Q] {result['question']}")
        print(f"[A] {result['answer']}")
        print(f"[출처] {', '.join(result['sources'])}")


def summary():
    """학습 요약"""
    print("\n" + "=" * 60)
    print("RAG 핵심 정리")
    print("=" * 60)

    print("""
    [RAG 파이프라인]

    사용자 질문
         │
         ▼
    ┌─────────────┐
    │  임베딩 변환 │  ← 질문을 벡터로
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │  벡터 DB    │  ← 유사 문서 검색
    │  검색       │
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │  프롬프트   │  ← 검색 결과 + 질문
    │  구성       │
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │  LLM 답변   │  ← 문서 기반 생성
    │  생성       │
    └─────────────┘
         │
         ▼
    최종 답변 + 출처

    [면접 대비 핵심]

    Q: "RAG의 장점은?"
    A: "LLM 환각 감소, 최신 정보 활용, 출처 명시 가능"

    Q: "청킹 전략은?"
    A: "문서 특성에 따라 선택. 의미 단위 보존이 중요.
        일반: 500-1000자, 오버랩 10-20%"

    Q: "검색 정확도 개선 방법은?"
    A: "Hybrid Search (키워드 + 벡터), Re-ranking,
        Query Expansion, 도메인 특화 임베딩"
    """)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  RAG 기초 실습")
    print("  검색 증강 생성의 모든 것")
    print("=" * 60)

    step_1_embedding_concept()
    client, collection = step_2_vector_db()
    step_3_rag_pipeline(client, collection)
    step_4_chunking()
    step_5_complete_example()
    summary()

    print("\n[다음 단계] FastAPI로 서비스화 → 포트폴리오 완성")
