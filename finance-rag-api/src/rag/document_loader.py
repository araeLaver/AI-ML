# -*- coding: utf-8 -*-
"""
문서 로더 모듈 - PDF 파싱 및 청킹

[백엔드 개발자 관점]
- 파일 I/O 처리 패턴
- 스트림 처리 (대용량 파일 대응)
- 전략 패턴 (다양한 파일 형식 지원)

[포트폴리오 포인트]
- 실제 금융 문서(PDF) 처리 능력 시연
- 청킹 전략의 이해와 구현
- 메타데이터 추출 및 관리
"""

import re
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, BinaryIO
from pathlib import Path
from datetime import datetime

from pypdf import PdfReader


@dataclass
class Document:
    """문서 청크 데이터 클래스"""
    content: str
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        """콘텐츠 기반 고유 ID 생성"""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:12]
        source = self.metadata.get("source", "unknown")
        chunk_idx = self.metadata.get("chunk_index", 0)
        return f"{source}_{chunk_idx}_{content_hash}"


@dataclass
class ChunkingConfig:
    """청킹 설정"""
    chunk_size: int = 500          # 청크 크기 (문자 수)
    chunk_overlap: int = 100       # 오버랩 크기
    min_chunk_size: int = 100      # 최소 청크 크기
    separator: str = "\n\n"        # 기본 구분자


class TextSplitter(ABC):
    """텍스트 분할기 인터페이스"""

    @abstractmethod
    def split(self, text: str) -> List[str]:
        """텍스트를 청크로 분할"""
        pass


class RecursiveTextSplitter(TextSplitter):
    """
    재귀적 텍스트 분할기

    LangChain의 RecursiveCharacterTextSplitter와 유사한 구현
    - 문단 → 문장 → 단어 순으로 분할 시도
    - 의미 단위 보존 우선
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        # 분할 우선순위: 문단 > 줄바꿈 > 문장 > 공백
        self.separators = ["\n\n", "\n", ". ", " "]

    def split(self, text: str) -> List[str]:
        """텍스트를 청크로 분할"""
        return self._split_recursive(text, self.separators)

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """재귀적으로 텍스트 분할"""
        if not text.strip():
            return []

        # 현재 텍스트가 청크 크기 이하면 그대로 반환
        if len(text) <= self.config.chunk_size:
            return [text.strip()] if text.strip() else []

        # 적절한 구분자 찾기
        separator = separators[0] if separators else ""
        for sep in separators:
            if sep in text:
                separator = sep
                break

        # 구분자로 분할
        if separator:
            splits = text.split(separator)
        else:
            # 구분자가 없으면 강제 분할
            return self._force_split(text)

        # 청크 병합
        chunks = []
        current_chunk = ""

        for split in splits:
            split = split.strip()
            if not split:
                continue

            # 현재 청크에 추가해도 크기 초과하지 않으면 추가
            potential = current_chunk + separator + split if current_chunk else split

            if len(potential) <= self.config.chunk_size:
                current_chunk = potential
            else:
                # 현재 청크 저장
                if current_chunk:
                    chunks.append(current_chunk)

                # 새 split이 청크 크기 초과하면 재귀 분할
                if len(split) > self.config.chunk_size:
                    next_seps = separators[1:] if len(separators) > 1 else []
                    sub_chunks = self._split_recursive(split, next_seps)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split

        # 마지막 청크 저장
        if current_chunk:
            chunks.append(current_chunk)

        # 오버랩 적용
        return self._apply_overlap(chunks)

    def _force_split(self, text: str) -> List[str]:
        """강제로 고정 크기로 분할"""
        chunks = []
        for i in range(0, len(text), self.config.chunk_size - self.config.chunk_overlap):
            chunk = text[i:i + self.config.chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """청크 간 오버랩 적용"""
        if len(chunks) <= 1 or self.config.chunk_overlap == 0:
            return chunks

        overlapped = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
            else:
                # 이전 청크의 끝부분을 현재 청크 앞에 추가
                prev_end = chunks[i-1][-self.config.chunk_overlap:]
                overlapped.append(prev_end + " " + chunk)

        return overlapped


class PDFLoader:
    """
    PDF 문서 로더

    [기능]
    - PDF 텍스트 추출
    - 페이지별 메타데이터
    - 자동 청킹
    """

    def __init__(
        self,
        splitter: Optional[TextSplitter] = None,
        chunking_config: Optional[ChunkingConfig] = None
    ):
        self.splitter = splitter or RecursiveTextSplitter(chunking_config)

    def load_from_path(self, file_path: str, source_name: Optional[str] = None) -> List[Document]:
        """파일 경로에서 PDF 로드"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        source = source_name or path.stem

        with open(path, "rb") as f:
            return self._load(f, source, str(path))

    def load_from_bytes(self, file_bytes: BinaryIO, filename: str, source_name: Optional[str] = None) -> List[Document]:
        """바이트 스트림에서 PDF 로드 (업로드 파일용)"""
        source = source_name or Path(filename).stem
        return self._load(file_bytes, source, filename)

    def _load(self, file: BinaryIO, source: str, filename: str) -> List[Document]:
        """PDF 파일 로드 및 청킹"""
        try:
            reader = PdfReader(file)
        except Exception as e:
            raise ValueError(f"PDF 파일을 읽을 수 없습니다: {e}")

        documents = []
        all_text = []

        # 페이지별 텍스트 추출
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text:
                # 텍스트 정제
                text = self._clean_text(text)
                all_text.append(text)

        # 전체 텍스트 결합
        full_text = "\n\n".join(all_text)

        if not full_text.strip():
            raise ValueError("PDF에서 텍스트를 추출할 수 없습니다.")

        # 청킹
        chunks = self.splitter.split(full_text)

        # Document 객체 생성
        for idx, chunk in enumerate(chunks):
            if len(chunk) < 50:  # 너무 짧은 청크 제외
                continue

            doc = Document(
                content=chunk,
                metadata={
                    "source": source,
                    "filename": filename,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "total_pages": len(reader.pages),
                    "loaded_at": datetime.now().isoformat(),
                    "file_type": "pdf"
                }
            )
            documents.append(doc)

        return documents

    def _clean_text(self, text: str) -> str:
        """텍스트 정제"""
        # 연속 공백 제거
        text = re.sub(r'\s+', ' ', text)
        # 연속 줄바꿈 정리
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 앞뒤 공백 제거
        text = text.strip()
        return text


class TextLoader:
    """
    텍스트 파일 로더

    .txt, .md 등 일반 텍스트 파일 처리
    """

    def __init__(
        self,
        splitter: Optional[TextSplitter] = None,
        chunking_config: Optional[ChunkingConfig] = None
    ):
        self.splitter = splitter or RecursiveTextSplitter(chunking_config)

    def load_from_path(self, file_path: str, source_name: Optional[str] = None) -> List[Document]:
        """파일 경로에서 텍스트 로드"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        source = source_name or path.stem

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        return self._process(text, source, str(path))

    def load_from_string(self, text: str, source_name: str) -> List[Document]:
        """문자열에서 직접 로드"""
        return self._process(text, source_name, source_name)

    def _process(self, text: str, source: str, filename: str) -> List[Document]:
        """텍스트 처리 및 청킹"""
        chunks = self.splitter.split(text)

        documents = []
        for idx, chunk in enumerate(chunks):
            if len(chunk) < 50:
                continue

            doc = Document(
                content=chunk,
                metadata={
                    "source": source,
                    "filename": filename,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "loaded_at": datetime.now().isoformat(),
                    "file_type": "text"
                }
            )
            documents.append(doc)

        return documents


class DocumentLoaderFactory:
    """
    문서 로더 팩토리

    파일 확장자에 따라 적절한 로더 반환
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf": PDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
    }

    @classmethod
    def get_loader(cls, filename: str, chunking_config: Optional[ChunkingConfig] = None):
        """파일명에 따른 로더 반환"""
        ext = Path(filename).suffix.lower()

        if ext not in cls.SUPPORTED_EXTENSIONS:
            supported = ", ".join(cls.SUPPORTED_EXTENSIONS.keys())
            raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}. 지원 형식: {supported}")

        loader_class = cls.SUPPORTED_EXTENSIONS[ext]
        return loader_class(chunking_config=chunking_config)

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """지원하는 확장자 목록"""
        return list(cls.SUPPORTED_EXTENSIONS.keys())
