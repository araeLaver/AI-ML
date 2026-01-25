# -*- coding: utf-8 -*-
"""
Multi-modal 처리 모듈

공시 문서 내 표, 차트, 이미지를 인식하고 처리합니다.
"""

import base64
import io
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class ContentType(Enum):
    """콘텐츠 유형"""
    TEXT = "text"
    TABLE = "table"
    CHART = "chart"
    IMAGE = "image"


@dataclass
class ExtractedContent:
    """추출된 콘텐츠

    Attributes:
        content_type: 콘텐츠 유형
        content: 추출된 내용
        metadata: 메타데이터
        confidence: 추출 신뢰도
        position: 문서 내 위치 정보
    """
    content_type: ContentType
    content: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    position: Optional[dict[str, int]] = None

    def to_text(self) -> str:
        """텍스트로 변환"""
        if self.content_type == ContentType.TEXT:
            return str(self.content)
        elif self.content_type == ContentType.TABLE:
            return self._table_to_text()
        elif self.content_type == ContentType.CHART:
            return self._chart_to_text()
        elif self.content_type == ContentType.IMAGE:
            return self._image_to_text()
        return ""

    def _table_to_text(self) -> str:
        """표를 텍스트로 변환"""
        if isinstance(self.content, list):
            # 2D 리스트 형태의 표
            lines = []
            for row in self.content:
                lines.append(" | ".join(str(cell) for cell in row))
            return "\n".join(lines)
        return str(self.content)

    def _chart_to_text(self) -> str:
        """차트 정보를 텍스트로 변환"""
        if isinstance(self.content, dict):
            parts = []
            if "title" in self.content:
                parts.append(f"차트 제목: {self.content['title']}")
            if "type" in self.content:
                parts.append(f"차트 유형: {self.content['type']}")
            if "data" in self.content:
                data = self.content["data"]
                if isinstance(data, dict):
                    for key, value in data.items():
                        parts.append(f"{key}: {value}")
            return "\n".join(parts)
        return str(self.content)

    def _image_to_text(self) -> str:
        """이미지 설명을 텍스트로 변환"""
        if isinstance(self.content, dict):
            return self.content.get("description", "[이미지]")
        return "[이미지]"


@dataclass
class TableData:
    """표 데이터

    Attributes:
        headers: 헤더 행
        rows: 데이터 행들
        caption: 표 캡션
        metadata: 메타데이터
    """
    headers: list[str]
    rows: list[list[Any]]
    caption: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "headers": self.headers,
            "rows": self.rows,
            "caption": self.caption,
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        """마크다운 표로 변환"""
        lines = []

        # 캡션
        if self.caption:
            lines.append(f"**{self.caption}**\n")

        # 헤더
        lines.append("| " + " | ".join(self.headers) + " |")
        lines.append("|" + "|".join(["---"] * len(self.headers)) + "|")

        # 행
        for row in self.rows:
            cells = [str(cell) for cell in row]
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        """pandas DataFrame으로 변환"""
        if PANDAS_AVAILABLE:
            return pd.DataFrame(self.rows, columns=self.headers)
        return None


@dataclass
class ChartData:
    """차트 데이터

    Attributes:
        chart_type: 차트 유형 (bar, line, pie 등)
        title: 차트 제목
        labels: 레이블 목록
        values: 값 목록
        series: 시리즈 데이터 (복수 시리즈용)
        metadata: 메타데이터
    """
    chart_type: str
    title: str = ""
    labels: list[str] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    series: dict[str, list[float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "chart_type": self.chart_type,
            "title": self.title,
            "labels": self.labels,
            "values": self.values,
            "series": self.series,
            "metadata": self.metadata,
        }

    def to_text(self) -> str:
        """텍스트 설명으로 변환"""
        parts = [f"[{self.chart_type} 차트]"]

        if self.title:
            parts.append(f"제목: {self.title}")

        if self.labels and self.values:
            parts.append("데이터:")
            for label, value in zip(self.labels, self.values):
                parts.append(f"  - {label}: {value}")

        if self.series:
            for series_name, series_values in self.series.items():
                parts.append(f"{series_name}: {series_values}")

        return "\n".join(parts)


class TableExtractor:
    """표 추출기

    HTML, 텍스트, 이미지에서 표를 추출합니다.
    """

    def __init__(self):
        """초기화"""
        # 재무제표 헤더 패턴
        self.financial_headers = [
            "구분", "항목", "과목",
            "당기", "전기", "당분기", "전분기",
            "금액", "비율", "증감",
            "자산", "부채", "자본",
            "매출", "영업이익", "당기순이익",
        ]

    def extract_from_html(self, html: str) -> list[TableData]:
        """HTML에서 표 추출

        Args:
            html: HTML 문자열

        Returns:
            추출된 표 목록
        """
        tables = []

        # 간단한 HTML 표 파싱
        table_pattern = r'<table[^>]*>(.*?)</table>'
        matches = re.findall(table_pattern, html, re.DOTALL | re.IGNORECASE)

        for match in matches:
            table = self._parse_html_table(match)
            if table:
                tables.append(table)

        return tables

    def _parse_html_table(self, table_html: str) -> Optional[TableData]:
        """HTML 표 파싱"""
        headers = []
        rows = []

        # 헤더 추출
        header_pattern = r'<th[^>]*>(.*?)</th>'
        header_matches = re.findall(header_pattern, table_html, re.DOTALL | re.IGNORECASE)
        if header_matches:
            headers = [self._clean_html(h) for h in header_matches]

        # 행 추출
        row_pattern = r'<tr[^>]*>(.*?)</tr>'
        row_matches = re.findall(row_pattern, table_html, re.DOTALL | re.IGNORECASE)

        for row_html in row_matches:
            cell_pattern = r'<td[^>]*>(.*?)</td>'
            cells = re.findall(cell_pattern, row_html, re.DOTALL | re.IGNORECASE)
            if cells:
                rows.append([self._clean_html(c) for c in cells])

        if not headers and rows:
            headers = rows.pop(0)

        if headers or rows:
            return TableData(headers=headers, rows=rows)

        return None

    def _clean_html(self, text: str) -> str:
        """HTML 태그 제거"""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_from_text(self, text: str) -> list[TableData]:
        """텍스트에서 표 추출

        파이프(|) 또는 탭으로 구분된 표를 인식합니다.

        Args:
            text: 텍스트

        Returns:
            추출된 표 목록
        """
        tables = []

        # 파이프 구분 표
        pipe_tables = self._extract_pipe_table(text)
        tables.extend(pipe_tables)

        # 탭 구분 표
        tab_tables = self._extract_tab_table(text)
        tables.extend(tab_tables)

        return tables

    def _extract_pipe_table(self, text: str) -> list[TableData]:
        """파이프 구분 표 추출"""
        tables = []
        lines = text.split('\n')

        current_table: list[list[str]] = []

        for line in lines:
            if '|' in line:
                # 파이프로 구분된 행
                cells = [c.strip() for c in line.split('|')]
                cells = [c for c in cells if c]  # 빈 셀 제거
                if cells:
                    current_table.append(cells)
            else:
                # 표 끝
                if len(current_table) >= 2:
                    headers = current_table[0]
                    # 구분선 제외
                    rows = [
                        row for row in current_table[1:]
                        if not all(c in ['-', '---', '----'] for c in row)
                    ]
                    if rows:
                        tables.append(TableData(headers=headers, rows=rows))
                current_table = []

        # 마지막 표 처리
        if len(current_table) >= 2:
            headers = current_table[0]
            rows = [
                row for row in current_table[1:]
                if not all(c in ['-', '---', '----'] for c in row)
            ]
            if rows:
                tables.append(TableData(headers=headers, rows=rows))

        return tables

    def _extract_tab_table(self, text: str) -> list[TableData]:
        """탭 구분 표 추출"""
        tables = []
        lines = text.split('\n')

        current_table: list[list[str]] = []

        for line in lines:
            if '\t' in line:
                cells = [c.strip() for c in line.split('\t')]
                cells = [c for c in cells if c]
                if cells:
                    current_table.append(cells)
            else:
                if len(current_table) >= 2:
                    headers = current_table[0]
                    rows = current_table[1:]
                    tables.append(TableData(headers=headers, rows=rows))
                current_table = []

        if len(current_table) >= 2:
            headers = current_table[0]
            rows = current_table[1:]
            tables.append(TableData(headers=headers, rows=rows))

        return tables

    def is_financial_table(self, table: TableData) -> bool:
        """재무제표 여부 확인"""
        all_text = " ".join(table.headers + [str(c) for row in table.rows for c in row])
        matches = sum(1 for h in self.financial_headers if h in all_text)
        return matches >= 2


class ChartExtractor:
    """차트 추출기

    이미지에서 차트 데이터를 추출합니다.
    """

    def __init__(self):
        """초기화"""
        self.chart_types = ["bar", "line", "pie", "area", "scatter"]

    def analyze_chart_image(
        self,
        image: Union[str, bytes, Any],
    ) -> Optional[ChartData]:
        """차트 이미지 분석

        Args:
            image: 이미지 (경로, 바이트, PIL Image)

        Returns:
            차트 데이터
        """
        # 이미지 로드
        img = self._load_image(image)
        if img is None:
            return None

        # 차트 유형 감지 (간단한 휴리스틱)
        chart_type = self._detect_chart_type(img)

        # 차트 데이터 추출 (실제로는 OCR + ML 필요)
        chart_data = ChartData(
            chart_type=chart_type,
            title="[차트]",
            metadata={"analyzed": True},
        )

        return chart_data

    def _load_image(self, image: Union[str, bytes, Any]) -> Optional[Any]:
        """이미지 로드"""
        if not PIL_AVAILABLE:
            return None

        if isinstance(image, str):
            if Path(image).exists():
                return Image.open(image)
            elif image.startswith("data:image"):
                # base64 데이터 URI
                base64_data = image.split(",")[1]
                return Image.open(io.BytesIO(base64.b64decode(base64_data)))
        elif isinstance(image, bytes):
            return Image.open(io.BytesIO(image))
        elif PIL_AVAILABLE and isinstance(image, Image.Image):
            return image

        return None

    def _detect_chart_type(self, img: Any) -> str:
        """차트 유형 감지"""
        # 실제로는 이미지 분류 모델 사용
        # 여기서는 기본값 반환
        return "bar"


class MultiModalProcessor:
    """멀티모달 처리기

    문서에서 텍스트, 표, 차트, 이미지를 통합 처리합니다.
    """

    def __init__(self):
        """초기화"""
        self.table_extractor = TableExtractor()
        self.chart_extractor = ChartExtractor()

        # 통계
        self._processed_count = 0
        self._extracted_tables = 0
        self._extracted_charts = 0

    @property
    def stats(self) -> dict[str, Any]:
        """통계 정보"""
        return {
            "processed_count": self._processed_count,
            "extracted_tables": self._extracted_tables,
            "extracted_charts": self._extracted_charts,
        }

    def process_document(
        self,
        content: str,
        content_type: str = "text",
    ) -> list[ExtractedContent]:
        """문서 처리

        Args:
            content: 문서 내용
            content_type: 콘텐츠 유형 (text, html)

        Returns:
            추출된 콘텐츠 목록
        """
        self._processed_count += 1
        extracted: list[ExtractedContent] = []

        # 표 추출
        if content_type == "html":
            tables = self.table_extractor.extract_from_html(content)
        else:
            tables = self.table_extractor.extract_from_text(content)

        for table in tables:
            self._extracted_tables += 1
            extracted.append(ExtractedContent(
                content_type=ContentType.TABLE,
                content=table.to_dict(),
                metadata={
                    "is_financial": self.table_extractor.is_financial_table(table),
                    "row_count": len(table.rows),
                    "col_count": len(table.headers),
                },
            ))

        # 텍스트 추출 (표 제외 부분)
        text_content = self._extract_text_without_tables(content, content_type)
        if text_content.strip():
            extracted.append(ExtractedContent(
                content_type=ContentType.TEXT,
                content=text_content,
            ))

        return extracted

    def _extract_text_without_tables(
        self,
        content: str,
        content_type: str,
    ) -> str:
        """표를 제외한 텍스트 추출"""
        if content_type == "html":
            # HTML 태그 제거
            text = re.sub(r'<table[^>]*>.*?</table>', '', content, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        else:
            # 파이프/탭 테이블 라인 제거
            lines = content.split('\n')
            non_table_lines = [
                line for line in lines
                if not ('|' in line and line.count('|') >= 2)
                and not ('\t' in line and line.count('\t') >= 2)
            ]
            return '\n'.join(non_table_lines)

    def process_image(
        self,
        image: Union[str, bytes, Any],
    ) -> list[ExtractedContent]:
        """이미지 처리

        Args:
            image: 이미지

        Returns:
            추출된 콘텐츠 목록
        """
        extracted: list[ExtractedContent] = []

        # 차트 분석
        chart_data = self.chart_extractor.analyze_chart_image(image)
        if chart_data:
            self._extracted_charts += 1
            extracted.append(ExtractedContent(
                content_type=ContentType.CHART,
                content=chart_data.to_dict(),
                metadata={"chart_type": chart_data.chart_type},
            ))

        return extracted

    def to_searchable_text(
        self,
        contents: list[ExtractedContent],
    ) -> str:
        """검색 가능한 텍스트로 변환

        Args:
            contents: 추출된 콘텐츠 목록

        Returns:
            통합된 검색용 텍스트
        """
        parts = []

        for content in contents:
            text = content.to_text()
            if text:
                parts.append(text)

        return "\n\n".join(parts)


class MultiModalEmbedding:
    """멀티모달 임베딩

    텍스트와 이미지를 함께 임베딩합니다.
    """

    def __init__(self):
        """초기화"""
        self._text_model: Optional[Any] = None
        self._image_model: Optional[Any] = None

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """텍스트 임베딩

        Args:
            texts: 텍스트 목록

        Returns:
            임베딩 벡터
        """
        # 간단한 해시 기반 임베딩 (실제로는 모델 사용)
        import hashlib
        embeddings = []
        for text in texts:
            hash_val = hashlib.md5(text.encode()).hexdigest()
            embedding = [int(hash_val[i:i+2], 16) / 255.0 for i in range(0, 32, 2)]
            embeddings.append(embedding * 24)  # 384 차원
        return np.array(embeddings)

    def encode_image(self, images: list[Any]) -> np.ndarray:
        """이미지 임베딩

        Args:
            images: 이미지 목록

        Returns:
            임베딩 벡터
        """
        # 간단한 임의 임베딩 (실제로는 CLIP 등 사용)
        embeddings = []
        for _ in images:
            embedding = np.random.randn(384).tolist()
            embeddings.append(embedding)
        return np.array(embeddings)

    def encode_multimodal(
        self,
        texts: list[str],
        images: list[Any],
    ) -> np.ndarray:
        """멀티모달 임베딩

        텍스트와 이미지를 결합하여 임베딩합니다.

        Args:
            texts: 텍스트 목록
            images: 이미지 목록

        Returns:
            결합된 임베딩 벡터
        """
        text_emb = self.encode_text(texts) if texts else np.zeros((0, 384))
        image_emb = self.encode_image(images) if images else np.zeros((0, 384))

        # 평균으로 결합
        if len(text_emb) > 0 and len(image_emb) > 0:
            combined = (text_emb.mean(axis=0) + image_emb.mean(axis=0)) / 2
            return combined.reshape(1, -1)
        elif len(text_emb) > 0:
            return text_emb.mean(axis=0).reshape(1, -1)
        elif len(image_emb) > 0:
            return image_emb.mean(axis=0).reshape(1, -1)
        else:
            return np.zeros((1, 384))


# 편의 함수
def extract_tables(text: str) -> list[TableData]:
    """표 추출 (편의 함수)

    Args:
        text: 텍스트

    Returns:
        추출된 표 목록
    """
    extractor = TableExtractor()
    return extractor.extract_from_text(text)


def process_document(
    content: str,
    content_type: str = "text",
) -> list[ExtractedContent]:
    """문서 처리 (편의 함수)

    Args:
        content: 문서 내용
        content_type: 콘텐츠 유형

    Returns:
        추출된 콘텐츠 목록
    """
    processor = MultiModalProcessor()
    return processor.process_document(content, content_type)
