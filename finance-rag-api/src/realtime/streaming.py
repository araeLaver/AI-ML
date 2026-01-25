# -*- coding: utf-8 -*-
"""
SSE 스트리밍 응답 모듈

LLM 응답을 실시간으로 스트리밍하여 사용자 경험을 개선합니다.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Optional

try:
    from fastapi import Response
    from fastapi.responses import StreamingResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


class StreamEventType(Enum):
    """스트림 이벤트 유형"""
    START = "start"  # 스트림 시작
    CHUNK = "chunk"  # 텍스트 청크
    SOURCE = "source"  # 소스 문서
    METADATA = "metadata"  # 메타데이터
    ERROR = "error"  # 오류
    END = "end"  # 스트림 종료


@dataclass
class StreamingConfig:
    """스트리밍 설정

    Attributes:
        chunk_size: 청크 크기 (문자 수)
        chunk_delay_ms: 청크 간 지연 (밀리초)
        include_sources: 소스 문서 포함 여부
        include_metadata: 메타데이터 포함 여부
        heartbeat_interval_s: 하트비트 간격 (초)
    """
    chunk_size: int = 10
    chunk_delay_ms: int = 50
    include_sources: bool = True
    include_metadata: bool = True
    heartbeat_interval_s: float = 15.0


@dataclass
class StreamChunk:
    """스트림 청크

    Attributes:
        event_type: 이벤트 유형
        data: 청크 데이터
        chunk_id: 청크 ID
        timestamp: 타임스탬프
    """
    event_type: StreamEventType
    data: Any
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)

    def to_sse(self) -> str:
        """SSE 형식으로 변환

        Returns:
            SSE 형식 문자열
        """
        event = self.event_type.value
        data_str = json.dumps(self.data, ensure_ascii=False) if isinstance(self.data, dict) else str(self.data)

        return f"event: {event}\ndata: {data_str}\nid: {self.chunk_id}\n\n"


class StreamingRAGService:
    """스트리밍 RAG 서비스

    RAG 응답을 SSE 스트림으로 제공합니다.
    """

    def __init__(
        self,
        rag_service: Optional[Any] = None,
        config: Optional[StreamingConfig] = None,
    ):
        """
        Args:
            rag_service: RAGService 인스턴스
            config: 스트리밍 설정
        """
        self._rag_service = rag_service
        self.config = config or StreamingConfig()

        # 통계
        self._total_streams = 0
        self._total_chunks_sent = 0

    @property
    def rag_service(self):
        """RAGService 인스턴스 (lazy loading)"""
        if self._rag_service is None:
            try:
                from ..rag.rag_service import RAGService
                self._rag_service = RAGService()
            except Exception:
                pass
        return self._rag_service

    @property
    def stats(self) -> dict[str, Any]:
        """통계 정보"""
        return {
            "total_streams": self._total_streams,
            "total_chunks_sent": self._total_chunks_sent,
        }

    async def stream_query(
        self,
        query: str,
        top_k: int = 5,
        **kwargs,
    ) -> AsyncGenerator[StreamChunk, None]:
        """쿼리 응답 스트리밍

        Args:
            query: 검색 쿼리
            top_k: 검색 결과 수
            **kwargs: 추가 파라미터

        Yields:
            StreamChunk 객체
        """
        self._total_streams += 1
        stream_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # 시작 이벤트
        yield StreamChunk(
            event_type=StreamEventType.START,
            data={
                "stream_id": stream_id,
                "query": query,
                "timestamp": datetime.now().isoformat(),
            },
        )
        self._total_chunks_sent += 1

        try:
            # RAG 쿼리 실행
            if self.rag_service:
                response = await self._execute_rag_query(query, top_k, **kwargs)
            else:
                # 테스트/폴백 응답
                response = {
                    "answer": f"'{query}'에 대한 응답입니다. RAG 서비스가 설정되지 않았습니다.",
                    "sources": [],
                    "metadata": {"fallback": True},
                }

            answer = response.get("answer", "")
            sources = response.get("sources", [])
            metadata = response.get("metadata", {})

            # 응답 텍스트 스트리밍
            chunk_size = self.config.chunk_size
            delay = self.config.chunk_delay_ms / 1000.0

            for i in range(0, len(answer), chunk_size):
                chunk_text = answer[i:i + chunk_size]

                yield StreamChunk(
                    event_type=StreamEventType.CHUNK,
                    data={"text": chunk_text, "index": i // chunk_size},
                )
                self._total_chunks_sent += 1

                if delay > 0:
                    await asyncio.sleep(delay)

            # 소스 문서 전송
            if self.config.include_sources and sources:
                yield StreamChunk(
                    event_type=StreamEventType.SOURCE,
                    data={"sources": sources},
                )
                self._total_chunks_sent += 1

            # 메타데이터 전송
            if self.config.include_metadata:
                metadata["duration_ms"] = int((time.time() - start_time) * 1000)
                metadata["stream_id"] = stream_id

                yield StreamChunk(
                    event_type=StreamEventType.METADATA,
                    data=metadata,
                )
                self._total_chunks_sent += 1

        except Exception as e:
            yield StreamChunk(
                event_type=StreamEventType.ERROR,
                data={"error": str(e), "type": type(e).__name__},
            )
            self._total_chunks_sent += 1

        # 종료 이벤트
        yield StreamChunk(
            event_type=StreamEventType.END,
            data={
                "stream_id": stream_id,
                "duration_ms": int((time.time() - start_time) * 1000),
            },
        )
        self._total_chunks_sent += 1

    async def _execute_rag_query(
        self,
        query: str,
        top_k: int,
        **kwargs,
    ) -> dict[str, Any]:
        """RAG 쿼리 실행"""
        if hasattr(self.rag_service, "query_async"):
            response = await self.rag_service.query_async(query, top_k=top_k, **kwargs)
        elif hasattr(self.rag_service, "query"):
            # 동기 메서드를 비동기로 래핑
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.rag_service.query(query, top_k=top_k, **kwargs),
            )
        else:
            response = {"answer": "RAG service not available", "sources": []}

        # RAGResponse 객체 처리
        if hasattr(response, "answer"):
            return {
                "answer": response.answer,
                "sources": getattr(response, "sources", []),
                "metadata": getattr(response, "metadata", {}),
            }

        return response

    async def stream_with_context(
        self,
        query: str,
        context: list[str],
        **kwargs,
    ) -> AsyncGenerator[StreamChunk, None]:
        """컨텍스트와 함께 스트리밍

        검색 없이 주어진 컨텍스트로 응답을 생성합니다.

        Args:
            query: 질문
            context: 컨텍스트 문서 목록
            **kwargs: 추가 파라미터

        Yields:
            StreamChunk 객체
        """
        # 컨텍스트를 RAG 서비스에 전달
        kwargs["context"] = context
        async for chunk in self.stream_query(query, **kwargs):
            yield chunk


async def stream_rag_response(
    query: str,
    rag_service: Optional[Any] = None,
    config: Optional[StreamingConfig] = None,
    **kwargs,
) -> AsyncGenerator[str, None]:
    """RAG 응답 SSE 스트리밍 (편의 함수)

    Args:
        query: 검색 쿼리
        rag_service: RAGService 인스턴스
        config: 스트리밍 설정
        **kwargs: 추가 파라미터

    Yields:
        SSE 형식 문자열
    """
    streaming_service = StreamingRAGService(rag_service, config)

    async for chunk in streaming_service.stream_query(query, **kwargs):
        yield chunk.to_sse()


def create_sse_response(
    generator: AsyncGenerator[str, None],
    headers: Optional[dict[str, str]] = None,
) -> Any:
    """SSE StreamingResponse 생성

    Args:
        generator: SSE 문자열 생성기
        headers: 추가 헤더

    Returns:
        StreamingResponse 객체
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required for SSE responses")

    default_headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # nginx 버퍼링 비활성화
    }

    if headers:
        default_headers.update(headers)

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers=default_headers,
    )


# LLM 스트리밍 헬퍼
class LLMStreamingHelper:
    """LLM 응답 스트리밍 헬퍼

    다양한 LLM 백엔드의 스트리밍 응답을 통합 처리합니다.
    """

    @staticmethod
    async def stream_openai(
        client: Any,
        messages: list[dict],
        model: str = "gpt-4",
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """OpenAI 스트리밍

        Args:
            client: OpenAI 클라이언트
            messages: 메시지 목록
            model: 모델 이름

        Yields:
            텍스트 청크
        """
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                **kwargs,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"[Error: {str(e)}]"

    @staticmethod
    async def stream_anthropic(
        client: Any,
        messages: list[dict],
        model: str = "claude-3-sonnet-20240229",
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Anthropic Claude 스트리밍

        Args:
            client: Anthropic 클라이언트
            messages: 메시지 목록
            model: 모델 이름

        Yields:
            텍스트 청크
        """
        try:
            async with client.messages.stream(
                model=model,
                messages=messages,
                max_tokens=4096,
                **kwargs,
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            yield f"[Error: {str(e)}]"

    @staticmethod
    async def stream_ollama(
        base_url: str,
        prompt: str,
        model: str = "llama2",
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Ollama 스트리밍

        Args:
            base_url: Ollama 서버 URL
            prompt: 프롬프트
            model: 모델 이름

        Yields:
            텍스트 청크
        """
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": True, **kwargs},
                ) as response:
                    async for line in response.content:
                        if line:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                            if data.get("done"):
                                break

        except Exception as e:
            yield f"[Error: {str(e)}]"


# FastAPI 라우터 예제
if FASTAPI_AVAILABLE:
    from fastapi import APIRouter, Query

    def create_streaming_router(rag_service: Optional[Any] = None) -> APIRouter:
        """스트리밍 API 라우터 생성

        Args:
            rag_service: RAGService 인스턴스

        Returns:
            FastAPI 라우터
        """
        router = APIRouter(prefix="/stream", tags=["streaming"])
        streaming_service = StreamingRAGService(rag_service)

        @router.get("/query")
        async def stream_query_endpoint(
            q: str = Query(..., description="검색 쿼리"),
            top_k: int = Query(5, description="검색 결과 수"),
        ):
            """RAG 쿼리 스트리밍 엔드포인트"""
            async def generate():
                async for chunk in streaming_service.stream_query(q, top_k=top_k):
                    yield chunk.to_sse()

            return create_sse_response(generate())

        @router.get("/health")
        async def streaming_health():
            """스트리밍 서비스 상태"""
            return {
                "status": "ok",
                "stats": streaming_service.stats,
            }

        return router
