# -*- coding: utf-8 -*-
"""
Document Loader Module

PDF/TXT 파일 업로드 및 청킹 지원
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, BinaryIO
from datetime import datetime


@dataclass
class Document:
    """문서 청크 데이터 클래스"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

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


class TextSplitter:
    """
    재귀적 텍스트 분할기

    의미 단위를 보존하면서 텍스트 분할
    - 문단 > 줄바꿈 > 문장 > 공백 순으로 분할 시도
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        # 분할 우선순위
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


class DocumentLoader:
    """
    문서 로더 (PDF/TXT)

    Streamlit 업로드 파일을 처리하여 청킹된 문서 리스트 반환
    """

    SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md"]

    def __init__(self, chunking_config: Optional[ChunkingConfig] = None):
        self.splitter = TextSplitter(chunking_config)
        self.config = chunking_config or ChunkingConfig()

    def load_from_uploaded_file(
        self,
        uploaded_file,
        source_name: Optional[str] = None
    ) -> List[Document]:
        """
        Streamlit 업로드 파일에서 문서 로드

        Args:
            uploaded_file: Streamlit UploadedFile 객체
            source_name: 소스 이름 (없으면 파일명 사용)

        Returns:
            Document 리스트
        """
        filename = uploaded_file.name
        source = source_name or filename.rsplit(".", 1)[0]

        # 확장자에 따라 처리
        if filename.lower().endswith(".pdf"):
            return self._load_pdf(uploaded_file, source, filename)
        elif filename.lower().endswith((".txt", ".md")):
            return self._load_text(uploaded_file, source, filename)
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다. 지원 형식: {self.SUPPORTED_EXTENSIONS}")

    def load_from_text(self, text: str, source_name: str) -> List[Document]:
        """
        텍스트 문자열에서 문서 로드

        Args:
            text: 텍스트 내용
            source_name: 소스 이름

        Returns:
            Document 리스트
        """
        return self._process_text(text, source_name, source_name)

    def _load_pdf(self, file, source: str, filename: str) -> List[Document]:
        """PDF 파일 로드"""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf 패키지가 필요합니다: pip install pypdf")

        try:
            reader = PdfReader(file)
        except Exception as e:
            raise ValueError(f"PDF 파일을 읽을 수 없습니다: {e}")

        # 페이지별 텍스트 추출
        all_text = []
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text:
                text = self._clean_text(text)
                all_text.append(text)

        # 전체 텍스트 결합
        full_text = "\n\n".join(all_text)

        if not full_text.strip():
            raise ValueError("PDF에서 텍스트를 추출할 수 없습니다.")

        return self._process_text(full_text, source, filename, file_type="pdf", total_pages=len(reader.pages))

    def _load_text(self, file, source: str, filename: str) -> List[Document]:
        """텍스트 파일 로드"""
        try:
            text = file.read().decode("utf-8")
        except UnicodeDecodeError:
            try:
                file.seek(0)
                text = file.read().decode("cp949")
            except Exception as e:
                raise ValueError(f"파일 인코딩을 처리할 수 없습니다: {e}")

        return self._process_text(text, source, filename, file_type="text")

    def _process_text(
        self,
        text: str,
        source: str,
        filename: str,
        file_type: str = "text",
        total_pages: int = 0
    ) -> List[Document]:
        """텍스트 처리 및 청킹"""
        # 청킹
        chunks = self.splitter.split(text)

        # Document 객체 생성
        documents = []
        for idx, chunk in enumerate(chunks):
            if len(chunk) < self.config.min_chunk_size:
                continue

            doc = Document(
                content=chunk,
                metadata={
                    "source": source,
                    "filename": filename,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "loaded_at": datetime.now().isoformat(),
                    "file_type": file_type,
                }
            )
            if total_pages > 0:
                doc.metadata["total_pages"] = total_pages

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

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """지원하는 확장자 목록"""
        return cls.SUPPORTED_EXTENSIONS.copy()
