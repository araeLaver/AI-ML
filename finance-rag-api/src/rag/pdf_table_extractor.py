# -*- coding: utf-8 -*-
"""
PDF 표 추출 모듈

Camelot, pdfplumber를 활용하여 PDF에서 표를 추출합니다.
재무제표, 사업보고서 등 금융 문서 특화.
"""

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from .multimodal import TableData


@dataclass
class PDFTableExtractionConfig:
    """PDF 표 추출 설정

    Attributes:
        flavor: Camelot 추출 방식 ('lattice' 또는 'stream')
        pages: 추출할 페이지 ('all' 또는 '1,2,3' 또는 '1-5')
        line_scale: 라인 감지 스케일 (lattice)
        edge_tol: 엣지 허용 오차 (stream)
        row_tol: 행 허용 오차 (stream)
        strip_text: 공백 제거
        split_text: 줄바꿈으로 분할
    """
    flavor: str = "lattice"
    pages: str = "all"
    line_scale: int = 40
    edge_tol: int = 50
    row_tol: int = 2
    strip_text: bool = True
    split_text: bool = False


@dataclass
class ExtractedTable:
    """추출된 표 정보

    Attributes:
        table_data: 표 데이터
        page_number: 페이지 번호
        extraction_method: 추출 방식
        accuracy: 추출 정확도 (0-100)
        bbox: 표 위치 (x1, y1, x2, y2)
        is_financial: 재무제표 여부
    """
    table_data: TableData
    page_number: int
    extraction_method: str
    accuracy: float = 0.0
    bbox: Optional[tuple[float, float, float, float]] = None
    is_financial: bool = False


class PDFTableExtractor:
    """PDF 표 추출기

    Camelot과 pdfplumber를 사용하여 PDF에서 표를 추출합니다.
    """

    # 재무제표 키워드
    FINANCIAL_KEYWORDS = [
        "재무상태표", "손익계산서", "포괄손익계산서",
        "현금흐름표", "자본변동표", "이익잉여금처분계산서",
        "자산총계", "부채총계", "자본총계",
        "매출액", "영업이익", "당기순이익",
        "유동자산", "비유동자산", "유동부채", "비유동부채",
    ]

    # 숫자 패턴 (금액)
    AMOUNT_PATTERN = re.compile(r'^[\d,\.\-\(\)]+$')

    def __init__(self, config: Optional[PDFTableExtractionConfig] = None):
        """초기화

        Args:
            config: 추출 설정
        """
        self.config = config or PDFTableExtractionConfig()
        self._stats = {
            "files_processed": 0,
            "tables_extracted": 0,
            "financial_tables": 0,
        }

    @property
    def stats(self) -> dict[str, Any]:
        """통계 정보"""
        return self._stats.copy()

    def extract_tables(
        self,
        pdf_path: Union[str, Path],
        pages: Optional[str] = None,
    ) -> list[ExtractedTable]:
        """PDF에서 표 추출

        Args:
            pdf_path: PDF 파일 경로
            pages: 추출할 페이지 (None이면 설정값 사용)

        Returns:
            추출된 표 목록
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        pages = pages or self.config.pages
        extracted_tables: list[ExtractedTable] = []

        # Camelot 우선 사용 (더 정확한 표 추출)
        if CAMELOT_AVAILABLE:
            try:
                camelot_tables = self._extract_with_camelot(str(pdf_path), pages)
                extracted_tables.extend(camelot_tables)
            except Exception as e:
                # Camelot 실패 시 pdfplumber로 폴백
                if PDFPLUMBER_AVAILABLE:
                    pdfplumber_tables = self._extract_with_pdfplumber(str(pdf_path), pages)
                    extracted_tables.extend(pdfplumber_tables)
        elif PDFPLUMBER_AVAILABLE:
            pdfplumber_tables = self._extract_with_pdfplumber(str(pdf_path), pages)
            extracted_tables.extend(pdfplumber_tables)

        # 통계 업데이트
        self._stats["files_processed"] += 1
        self._stats["tables_extracted"] += len(extracted_tables)
        self._stats["financial_tables"] += sum(1 for t in extracted_tables if t.is_financial)

        return extracted_tables

    def extract_tables_from_bytes(
        self,
        pdf_bytes: bytes,
        pages: Optional[str] = None,
    ) -> list[ExtractedTable]:
        """바이트에서 표 추출

        Args:
            pdf_bytes: PDF 바이트
            pages: 추출할 페이지

        Returns:
            추출된 표 목록
        """
        # 임시 파일 대신 pdfplumber 직접 사용
        if not PDFPLUMBER_AVAILABLE:
            return []

        pages = pages or self.config.pages
        extracted_tables: list[ExtractedTable] = []

        try:
            pdf_stream = io.BytesIO(pdf_bytes)
            with pdfplumber.open(pdf_stream) as pdf:
                page_nums = self._parse_pages(pages, len(pdf.pages))

                for page_num in page_nums:
                    if page_num <= len(pdf.pages):
                        page = pdf.pages[page_num - 1]
                        tables = page.extract_tables()

                        for i, table in enumerate(tables):
                            if table and len(table) > 0:
                                table_data = self._convert_to_table_data(table)
                                if table_data:
                                    is_financial = self._is_financial_table(table_data)
                                    extracted_tables.append(ExtractedTable(
                                        table_data=table_data,
                                        page_number=page_num,
                                        extraction_method="pdfplumber",
                                        accuracy=85.0,
                                        is_financial=is_financial,
                                    ))

            self._stats["files_processed"] += 1
            self._stats["tables_extracted"] += len(extracted_tables)
            self._stats["financial_tables"] += sum(1 for t in extracted_tables if t.is_financial)

        except Exception:
            pass

        return extracted_tables

    def _extract_with_camelot(
        self,
        pdf_path: str,
        pages: str,
    ) -> list[ExtractedTable]:
        """Camelot으로 표 추출"""
        extracted_tables: list[ExtractedTable] = []

        # Lattice 방식 (선이 있는 표)
        try:
            tables = camelot.read_pdf(
                pdf_path,
                pages=pages,
                flavor="lattice",
                line_scale=self.config.line_scale,
                strip_text="\n" if self.config.strip_text else "",
            )

            for table in tables:
                df = table.df
                table_data = self._dataframe_to_table_data(df)

                if table_data and len(table_data.rows) > 0:
                    is_financial = self._is_financial_table(table_data)
                    extracted_tables.append(ExtractedTable(
                        table_data=table_data,
                        page_number=table.page,
                        extraction_method="camelot-lattice",
                        accuracy=table.accuracy,
                        bbox=table._bbox if hasattr(table, '_bbox') else None,
                        is_financial=is_financial,
                    ))
        except Exception:
            pass

        # Lattice에서 추출 실패 시 Stream 방식 시도
        if not extracted_tables:
            try:
                tables = camelot.read_pdf(
                    pdf_path,
                    pages=pages,
                    flavor="stream",
                    edge_tol=self.config.edge_tol,
                    row_tol=self.config.row_tol,
                    strip_text="\n" if self.config.strip_text else "",
                )

                for table in tables:
                    df = table.df
                    table_data = self._dataframe_to_table_data(df)

                    if table_data and len(table_data.rows) > 0:
                        is_financial = self._is_financial_table(table_data)
                        extracted_tables.append(ExtractedTable(
                            table_data=table_data,
                            page_number=table.page,
                            extraction_method="camelot-stream",
                            accuracy=table.accuracy,
                            bbox=table._bbox if hasattr(table, '_bbox') else None,
                            is_financial=is_financial,
                        ))
            except Exception:
                pass

        return extracted_tables

    def _extract_with_pdfplumber(
        self,
        pdf_path: str,
        pages: str,
    ) -> list[ExtractedTable]:
        """pdfplumber로 표 추출"""
        extracted_tables: list[ExtractedTable] = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_nums = self._parse_pages(pages, len(pdf.pages))

                for page_num in page_nums:
                    if page_num <= len(pdf.pages):
                        page = pdf.pages[page_num - 1]
                        tables = page.extract_tables()

                        for i, table in enumerate(tables):
                            if table and len(table) > 0:
                                table_data = self._convert_to_table_data(table)
                                if table_data:
                                    is_financial = self._is_financial_table(table_data)
                                    bbox = page.find_tables()[i].bbox if i < len(page.find_tables()) else None
                                    extracted_tables.append(ExtractedTable(
                                        table_data=table_data,
                                        page_number=page_num,
                                        extraction_method="pdfplumber",
                                        accuracy=85.0,  # pdfplumber는 정확도 미제공
                                        bbox=bbox,
                                        is_financial=is_financial,
                                    ))
        except Exception:
            pass

        return extracted_tables

    def _parse_pages(self, pages: str, total_pages: int) -> list[int]:
        """페이지 문자열 파싱"""
        if pages == "all":
            return list(range(1, total_pages + 1))

        result = []
        for part in pages.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                result.extend(range(int(start), int(end) + 1))
            else:
                result.append(int(part))

        return [p for p in result if 1 <= p <= total_pages]

    def _dataframe_to_table_data(self, df) -> Optional[TableData]:
        """DataFrame을 TableData로 변환"""
        if df is None or df.empty:
            return None

        # 첫 행을 헤더로 사용
        headers = [str(h).strip() for h in df.iloc[0].tolist()]
        rows = []

        for i in range(1, len(df)):
            row = [str(cell).strip() if cell else "" for cell in df.iloc[i].tolist()]
            if any(cell for cell in row):  # 빈 행 제외
                rows.append(row)

        if not headers and not rows:
            return None

        return TableData(headers=headers, rows=rows)

    def _convert_to_table_data(self, table: list[list]) -> Optional[TableData]:
        """2D 리스트를 TableData로 변환"""
        if not table or len(table) < 2:
            return None

        # 첫 행을 헤더로
        headers = [str(cell).strip() if cell else "" for cell in table[0]]
        rows = []

        for row in table[1:]:
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            if any(cell for cell in cleaned_row):
                rows.append(cleaned_row)

        if not headers and not rows:
            return None

        return TableData(headers=headers, rows=rows)

    def _is_financial_table(self, table_data: TableData) -> bool:
        """재무제표 여부 판단"""
        # 모든 텍스트 결합
        all_text = " ".join(table_data.headers)
        for row in table_data.rows:
            all_text += " " + " ".join(str(cell) for cell in row)

        # 재무제표 키워드 검사
        keyword_count = sum(1 for kw in self.FINANCIAL_KEYWORDS if kw in all_text)

        # 숫자 셀 비율 검사
        total_cells = len(table_data.headers) + sum(len(row) for row in table_data.rows)
        number_cells = 0
        for row in table_data.rows:
            for cell in row:
                if self.AMOUNT_PATTERN.match(str(cell).replace(",", "").replace(" ", "")):
                    number_cells += 1

        number_ratio = number_cells / total_cells if total_cells > 0 else 0

        # 키워드 2개 이상 또는 숫자 비율 50% 이상
        return keyword_count >= 2 or number_ratio >= 0.5


class FinancialStatementParser:
    """재무제표 파서

    추출된 표를 구조화된 재무제표로 변환합니다.
    """

    # 재무상태표 항목 매핑
    BALANCE_SHEET_ITEMS = {
        "자산총계": "total_assets",
        "유동자산": "current_assets",
        "비유동자산": "non_current_assets",
        "부채총계": "total_liabilities",
        "유동부채": "current_liabilities",
        "비유동부채": "non_current_liabilities",
        "자본총계": "total_equity",
    }

    # 손익계산서 항목 매핑
    INCOME_STATEMENT_ITEMS = {
        "매출액": "revenue",
        "영업이익": "operating_income",
        "당기순이익": "net_income",
        "매출총이익": "gross_profit",
        "영업비용": "operating_expenses",
    }

    def parse_financial_statement(
        self,
        table: ExtractedTable,
    ) -> Optional[dict[str, Any]]:
        """재무제표 파싱

        Args:
            table: 추출된 표

        Returns:
            구조화된 재무제표 데이터
        """
        if not table.is_financial:
            return None

        table_data = table.table_data
        result = {
            "type": self._detect_statement_type(table_data),
            "items": {},
            "periods": [],
            "raw_data": table_data.to_dict(),
        }

        # 기간 추출 (헤더에서)
        for header in table_data.headers[1:]:
            period = self._extract_period(header)
            if period:
                result["periods"].append(period)

        # 항목 추출
        for row in table_data.rows:
            if len(row) < 2:
                continue

            item_name = row[0].strip()
            values = row[1:]

            # 항목 매핑
            mapped_name = self._map_item_name(item_name, result["type"])
            if mapped_name:
                result["items"][mapped_name] = {
                    "original_name": item_name,
                    "values": [self._parse_amount(v) for v in values],
                }

        return result

    def _detect_statement_type(self, table_data: TableData) -> str:
        """재무제표 유형 감지"""
        all_text = " ".join(table_data.headers)
        for row in table_data.rows:
            all_text += " " + " ".join(str(cell) for cell in row)

        if "재무상태표" in all_text or "자산총계" in all_text:
            return "balance_sheet"
        elif "손익계산서" in all_text or "매출액" in all_text:
            return "income_statement"
        elif "현금흐름표" in all_text:
            return "cash_flow"
        elif "자본변동표" in all_text:
            return "equity_changes"

        return "unknown"

    def _map_item_name(self, name: str, statement_type: str) -> Optional[str]:
        """항목명 매핑"""
        name = name.strip()

        if statement_type == "balance_sheet":
            for korean, english in self.BALANCE_SHEET_ITEMS.items():
                if korean in name:
                    return english
        elif statement_type == "income_statement":
            for korean, english in self.INCOME_STATEMENT_ITEMS.items():
                if korean in name:
                    return english

        return None

    def _extract_period(self, header: str) -> Optional[str]:
        """기간 추출"""
        # 연도 패턴 (2023년, 2023.12, 제55기 등)
        year_match = re.search(r'(\d{4})(?:년|\.)', header)
        if year_match:
            return year_match.group(1)

        quarter_match = re.search(r'제?(\d+)(?:기|분기)', header)
        if quarter_match:
            return f"기{quarter_match.group(1)}"

        return None

    def _parse_amount(self, value: str) -> Optional[float]:
        """금액 파싱"""
        if not value:
            return None

        # 괄호는 음수
        is_negative = "(" in value or ")" in value

        # 숫자만 추출
        cleaned = re.sub(r'[^\d.]', '', value)
        if not cleaned:
            return None

        try:
            amount = float(cleaned)
            return -amount if is_negative else amount
        except ValueError:
            return None


# 편의 함수
def extract_tables_from_pdf(
    pdf_path: Union[str, Path],
    pages: str = "all",
) -> list[ExtractedTable]:
    """PDF에서 표 추출 (편의 함수)

    Args:
        pdf_path: PDF 파일 경로
        pages: 추출할 페이지

    Returns:
        추출된 표 목록
    """
    extractor = PDFTableExtractor()
    return extractor.extract_tables(pdf_path, pages)


def extract_financial_statements(
    pdf_path: Union[str, Path],
) -> list[dict[str, Any]]:
    """재무제표 추출 (편의 함수)

    Args:
        pdf_path: PDF 파일 경로

    Returns:
        구조화된 재무제표 목록
    """
    extractor = PDFTableExtractor()
    parser = FinancialStatementParser()

    tables = extractor.extract_tables(pdf_path)
    statements = []

    for table in tables:
        if table.is_financial:
            parsed = parser.parse_financial_statement(table)
            if parsed:
                statements.append(parsed)

    return statements
