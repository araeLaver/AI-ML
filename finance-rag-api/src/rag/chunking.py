# -*- coding: utf-8 -*-
"""
청킹 전략 모듈

[설계 의도]
- 다양한 청킹 전략 비교 실험 가능
- 문서 특성에 따른 최적 전략 선택
- 포트폴리오에서 "왜 이 전략을 선택했는가" 설명 가능

[청킹 전략별 특징]
1. Fixed Size: 단순하지만 문맥 단절 가능
2. Sentence: 문장 단위 보존, 한국어에 적합
3. Recursive: 구조적 분할, 복잡한 문서에 적합
4. Semantic: 의미 기반, 고품질이지만 느림
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Chunk:
    """청크 데이터 클래스"""
    text: str
    index: int
    metadata: Dict[str, Any]
    start_char: int
    end_char: int


class BaseChunker(ABC):
    """청킹 전략 추상 클래스"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """텍스트를 청크로 분할"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """전략 이름"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """전략 설명"""
        pass


class FixedSizeChunker(BaseChunker):
    """
    고정 크기 청킹

    장점: 구현 간단, 예측 가능한 청크 크기
    단점: 문맥 단절, 문장 중간 분할 가능
    적합: 균일한 구조의 문서, 빠른 처리 필요시
    """

    @property
    def name(self) -> str:
        return "fixed_size"

    @property
    def description(self) -> str:
        return f"고정 {self.chunk_size}자 단위 분할 (오버랩 {self.chunk_overlap}자)"

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        chunks = []
        start = 0
        index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            chunks.append(Chunk(
                text=chunk_text.strip(),
                index=index,
                metadata=metadata or {},
                start_char=start,
                end_char=min(end, len(text))
            ))

            start = end - self.chunk_overlap
            index += 1

        return chunks


class SentenceChunker(BaseChunker):
    """
    문장 단위 청킹

    장점: 문장 완결성 보장, 자연스러운 분할
    단점: 청크 크기 불균일, 긴 문장 처리 어려움
    적합: 한국어 문서, 서술형 텍스트
    """

    # 한국어 문장 종결 패턴
    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?。])\s+|(?<=다\.)\s*|(?<=요\.)\s*|(?<=니다\.)\s*')

    @property
    def name(self) -> str:
        return "sentence"

    @property
    def description(self) -> str:
        return f"문장 단위 분할 (목표 {self.chunk_size}자)"

    def _split_sentences(self, text: str) -> List[str]:
        """문장 분리"""
        sentences = self.SENTENCE_ENDINGS.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        sentences = self._split_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_start = 0
        index = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # 단일 문장이 chunk_size 초과하면 그대로 하나의 청크로
            if sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append(Chunk(
                        text=' '.join(current_chunk),
                        index=index,
                        metadata=metadata or {},
                        start_char=chunk_start,
                        end_char=chunk_start + current_length
                    ))
                    index += 1
                    chunk_start += current_length

                chunks.append(Chunk(
                    text=sentence,
                    index=index,
                    metadata=metadata or {},
                    start_char=chunk_start,
                    end_char=chunk_start + sentence_length
                ))
                index += 1
                chunk_start += sentence_length
                current_chunk = []
                current_length = 0
                continue

            # 청크 크기 초과시 새 청크 시작
            if current_length + sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append(Chunk(
                        text=' '.join(current_chunk),
                        index=index,
                        metadata=metadata or {},
                        start_char=chunk_start,
                        end_char=chunk_start + current_length
                    ))
                    index += 1
                    chunk_start += current_length

                # 오버랩 처리: 마지막 문장 유지
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_sentences = []
                    overlap_length = 0
                    for s in reversed(current_chunk):
                        if overlap_length + len(s) <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_length += len(s)
                        else:
                            break
                    current_chunk = overlap_sentences
                    current_length = overlap_length
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space

        # 마지막 청크 처리
        if current_chunk:
            chunks.append(Chunk(
                text=' '.join(current_chunk),
                index=index,
                metadata=metadata or {},
                start_char=chunk_start,
                end_char=len(text)
            ))

        return chunks


class RecursiveChunker(BaseChunker):
    """
    재귀적 청킹 (LangChain RecursiveCharacterTextSplitter 스타일)

    장점: 구조적 분할, 단락/섹션 보존
    단점: 구현 복잡, 구분자 의존
    적합: 구조화된 문서, 마크다운, 공시 문서
    """

    # 분할 우선순위: 단락 > 줄바꿈 > 문장 > 공백 > 글자
    SEPARATORS = ["\n\n", "\n", "。", ".", "!", "?", " ", ""]

    @property
    def name(self) -> str:
        return "recursive"

    @property
    def description(self) -> str:
        return f"재귀적 분할 (단락→문장→단어, 목표 {self.chunk_size}자)"

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """재귀적으로 텍스트 분할"""
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # 문자 단위 분할
            return list(text)

        splits = text.split(separator)

        result = []
        for split in splits:
            if len(split) <= self.chunk_size:
                result.append(split)
            else:
                # 더 작은 구분자로 재분할
                result.extend(self._split_text(split, remaining_separators))

        return result

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        splits = self._split_text(text, self.SEPARATORS)

        chunks = []
        current_chunk = []
        current_length = 0
        index = 0
        char_position = 0
        chunk_start = 0

        for split in splits:
            split_length = len(split)

            if current_length + split_length > self.chunk_size and current_chunk:
                chunk_text = ''.join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text.strip(),
                    index=index,
                    metadata=metadata or {},
                    start_char=chunk_start,
                    end_char=chunk_start + len(chunk_text)
                ))
                index += 1
                chunk_start += len(chunk_text)

                # 오버랩 처리
                if self.chunk_overlap > 0:
                    overlap_text = chunk_text[-self.chunk_overlap:]
                    current_chunk = [overlap_text]
                    current_length = len(overlap_text)
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(split)
            current_length += split_length

        if current_chunk:
            chunk_text = ''.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text.strip(),
                index=index,
                metadata=metadata or {},
                start_char=chunk_start,
                end_char=len(text)
            ))

        return chunks


class SemanticChunker(BaseChunker):
    """
    의미 기반 청킹

    장점: 의미 단위 보존, 높은 검색 품질
    단점: 임베딩 필요, 처리 속도 느림
    적합: 고품질 검색 필요시, 사전 처리 가능시

    [알고리즘]
    1. 문장 단위로 분할
    2. 각 문장 임베딩 계산
    3. 인접 문장 간 유사도 계산
    4. 유사도가 임계값 미만인 지점에서 분할
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.5
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.similarity_threshold = similarity_threshold
        self._embedder = None

    @property
    def name(self) -> str:
        return "semantic"

    @property
    def description(self) -> str:
        return f"의미 기반 분할 (유사도 임계값 {self.similarity_threshold})"

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        간소화된 의미 기반 청킹
        (실제 임베딩 없이 휴리스틱 사용 - 데모용)
        """
        # 실제 구현에서는 임베딩 모델 사용
        # 여기서는 단락 구조 + 키워드 기반 휴리스틱 사용

        # 단락 분할
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        current_chunk = []
        current_length = 0
        index = 0
        chunk_start = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_length = len(para)

            # 섹션 헤더 감지 (■, #, [, 1. 등으로 시작)
            is_new_section = bool(re.match(r'^[■#\[0-9]', para))

            # 새 섹션이거나 크기 초과시 청크 분리
            if (is_new_section and current_chunk) or \
               (current_length + para_length > self.chunk_size and current_chunk):
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    index=index,
                    metadata=metadata or {},
                    start_char=chunk_start,
                    end_char=chunk_start + len(chunk_text)
                ))
                index += 1
                chunk_start += len(chunk_text) + 2  # +2 for \n\n
                current_chunk = []
                current_length = 0

            current_chunk.append(para)
            current_length += para_length

        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                index=index,
                metadata=metadata or {},
                start_char=chunk_start,
                end_char=len(text)
            ))

        return chunks


# 전략 레지스트리
CHUNKING_STRATEGIES = {
    "fixed_size": FixedSizeChunker,
    "sentence": SentenceChunker,
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
}


def get_chunker(
    strategy: str = "recursive",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    **kwargs
) -> BaseChunker:
    """청킹 전략 팩토리"""
    if strategy not in CHUNKING_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(CHUNKING_STRATEGIES.keys())}")

    return CHUNKING_STRATEGIES[strategy](
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs
    )


def compare_strategies(
    text: str,
    strategies: Optional[List[str]] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> Dict[str, Any]:
    """
    여러 청킹 전략 비교

    Returns:
        각 전략별 청크 수, 평균 길이, 분포 등
    """
    if strategies is None:
        strategies = list(CHUNKING_STRATEGIES.keys())

    results = {}

    for strategy_name in strategies:
        chunker = get_chunker(strategy_name, chunk_size, chunk_overlap)
        chunks = chunker.chunk(text)

        lengths = [len(c.text) for c in chunks]

        results[strategy_name] = {
            "description": chunker.description,
            "num_chunks": len(chunks),
            "avg_length": sum(lengths) / len(lengths) if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "total_chars": sum(lengths),
            "chunks_preview": [c.text[:100] + "..." for c in chunks[:3]]
        }

    return results
