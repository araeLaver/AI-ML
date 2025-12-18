# -*- coding: utf-8 -*-
"""
RAG 서비스 모듈 - 핵심 비즈니스 로직

[백엔드 개발자 관점]
- Service Layer 패턴
- 비즈니스 로직 캡슐화
- 의존성 주입으로 테스트 용이

[면접 포인트]
Q: "RAG 시스템의 핵심 컴포넌트는?"
A: "검색기(Retriever), 생성기(Generator), 프롬프트 엔지니어링입니다.
    검색기는 벡터 DB로 관련 문서를 찾고,
    생성기는 LLM으로 답변을 생성합니다.
    프롬프트는 환각을 줄이는 핵심입니다."
"""

import ollama
from typing import Dict, Any, List, Optional, Generator
from dataclasses import dataclass

from .vectorstore import VectorStoreService


@dataclass
class RAGResponse:
    """RAG 응답 데이터 클래스"""
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: str


class RAGService:
    """
    RAG 파이프라인 서비스

    전체 흐름:
    1. 사용자 질문 수신
    2. 벡터 DB에서 관련 문서 검색 (Retrieval)
    3. 검색된 문서 + 질문으로 프롬프트 구성
    4. LLM으로 답변 생성 (Generation)
    5. 출처와 함께 응답 반환
    """

    # RAG 시스템 프롬프트 (환각 방지 + 출처 명시)
    SYSTEM_PROMPT = """당신은 금융 전문 상담 AI입니다.

역할:
- 제공된 문서만을 기반으로 정확하게 답변합니다
- 금융 용어를 쉽게 설명합니다
- 투자 조언이 아닌 정보 제공임을 명시합니다

규칙:
1. 문서에 없는 내용은 "해당 정보가 제공된 문서에 없습니다"라고 답하세요
2. 추측하거나 지어내지 마세요
3. 숫자나 수치는 문서 그대로 인용하세요
4. 답변 마지막에 참조 문서를 표시하세요

주의:
- 이 정보는 투자 권유가 아닙니다
- 실제 투자 결정은 전문가와 상담하세요"""

    def __init__(
        self,
        vectorstore: VectorStoreService,
        llm_model: str = "llama3.2",
        top_k: int = 3,
        temperature: float = 0.2
    ):
        self.vectorstore = vectorstore
        self.llm_model = llm_model
        self.top_k = top_k
        self.temperature = temperature

    def query(
        self,
        question: str,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> RAGResponse:
        """
        RAG 질의 처리

        Args:
            question: 사용자 질문
            filter_metadata: 문서 필터 (예: {"source": "투자 가이드"})

        Returns:
            RAGResponse: 답변, 출처, 신뢰도 포함
        """
        # Step 1: 관련 문서 검색
        search_results = self.vectorstore.search(
            query=question,
            top_k=self.top_k,
            filter_metadata=filter_metadata
        )

        documents = search_results["documents"]
        metadatas = search_results["metadatas"]
        distances = search_results["distances"]

        # 검색 결과 없음
        if not documents:
            return RAGResponse(
                question=question,
                answer="관련 문서를 찾을 수 없습니다. 다른 질문을 해주세요.",
                sources=[],
                confidence="low"
            )

        # Step 2: 컨텍스트 구성
        context_parts = []
        sources = []

        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            source_name = meta.get("source", f"문서 {i+1}") if meta else f"문서 {i+1}"
            context_parts.append(f"[{source_name}]\n{doc}")
            # 거리를 유사도로 변환 (0~1 범위로 정규화)
            # ChromaDB 거리는 L2 또는 코사인이며, 값이 작을수록 유사
            relevance = max(0.0, min(1.0, 1 - dist / 2))  # 정규화
            sources.append({
                "source": source_name,
                "content_preview": doc[:100] + "..." if len(doc) > 100 else doc,
                "relevance_score": round(relevance, 4)
            })

        context = "\n\n".join(context_parts)

        # Step 3: 프롬프트 구성
        user_prompt = f"""[참고 문서]
{context}

[사용자 질문]
{question}

[답변]"""

        # Step 4: LLM 답변 생성
        response = ollama.chat(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": self.temperature}
        )

        answer = response.message.content

        # Step 5: 신뢰도 판단 (평균 유사도 기반)
        avg_relevance = sum(max(0, min(1, 1 - d / 2)) for d in distances) / len(distances)
        confidence = "high" if avg_relevance > 0.6 else "medium" if avg_relevance > 0.4 else "low"

        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            confidence=confidence
        )

    def add_documents(
        self,
        documents: List[str],
        source: str,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        문서 추가

        Args:
            documents: 문서 내용 리스트
            source: 문서 출처 (예: "투자 가이드 2024")
            additional_metadata: 추가 메타데이터

        Returns:
            추가된 문서 수
        """
        ids = []
        metadatas = []

        base_count = self.vectorstore.get_document_count()

        for i, doc in enumerate(documents):
            doc_id = f"{source.replace(' ', '_')}_{base_count + i}"
            ids.append(doc_id)

            meta = {"source": source}
            if additional_metadata:
                meta.update(additional_metadata)
            metadatas.append(meta)

        self.vectorstore.add_documents(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

        return len(documents)

    def get_stats(self) -> Dict[str, Any]:
        """시스템 통계 조회"""
        return {
            "total_documents": self.vectorstore.get_document_count(),
            "llm_model": self.llm_model,
            "top_k": self.top_k,
            "temperature": self.temperature
        }

    def query_stream(
        self,
        question: str,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        스트리밍 RAG 질의 - 답변을 토큰 단위로 반환

        Yields:
            dict: {"type": "token|source|done", "content": ...}
        """
        # Step 1: 관련 문서 검색
        search_results = self.vectorstore.search(
            query=question,
            top_k=self.top_k,
            filter_metadata=filter_metadata
        )

        documents = search_results["documents"]
        metadatas = search_results["metadatas"]
        distances = search_results["distances"]

        # 검색 결과 없음
        if not documents:
            yield {"type": "token", "content": "관련 문서를 찾을 수 없습니다. 다른 질문을 해주세요."}
            yield {"type": "done", "confidence": "low", "sources": []}
            return

        # Step 2: 컨텍스트 구성 및 출처 정보
        context_parts = []
        sources = []

        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            source_name = meta.get("source", f"문서 {i+1}") if meta else f"문서 {i+1}"
            context_parts.append(f"[{source_name}]\n{doc}")
            relevance = max(0.0, min(1.0, 1 - dist / 2))
            sources.append({
                "source": source_name,
                "content_preview": doc[:100] + "..." if len(doc) > 100 else doc,
                "relevance_score": round(relevance, 4)
            })

        # 먼저 출처 정보 전송
        yield {"type": "sources", "content": sources}

        context = "\n\n".join(context_parts)

        # Step 3: 프롬프트 구성
        user_prompt = f"""[참고 문서]
{context}

[사용자 질문]
{question}

[답변]"""

        # Step 4: LLM 스트리밍 답변 생성
        stream = ollama.chat(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": self.temperature},
            stream=True
        )

        for chunk in stream:
            if chunk.message.content:
                yield {"type": "token", "content": chunk.message.content}

        # Step 5: 신뢰도 계산 및 완료
        avg_relevance = sum(max(0, min(1, 1 - d / 2)) for d in distances) / len(distances)
        confidence = "high" if avg_relevance > 0.6 else "medium" if avg_relevance > 0.4 else "low"

        yield {"type": "done", "confidence": confidence}
