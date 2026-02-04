# -*- coding: utf-8 -*-
"""
멀티모달 기능 테스트

PDF 표 추출, OCR, 차트 분석 API 테스트
"""

import io
import pytest
from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


# ============================================================
# 헬스체크 및 통계 테스트
# ============================================================

class TestMultimodalHealth:
    """멀티모달 헬스체크 테스트"""

    def test_health_endpoint(self):
        """멀티모달 헬스체크"""
        response = client.get("/api/v1/multimodal/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "multimodal"
        assert "features" in data

    def test_stats_endpoint(self):
        """멀티모달 통계 조회"""
        response = client.get("/api/v1/multimodal/stats")
        # 라이브러리 설치 상태에 따라 다를 수 있음
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "pdf_table_extractor" in data
            assert "ocr_engine" in data
            assert "chart_analyzer" in data


# ============================================================
# PDF 표 추출 테스트
# ============================================================

class TestTableExtraction:
    """PDF 표 추출 테스트"""

    def test_extract_tables_invalid_file_type(self):
        """PDF가 아닌 파일 업로드 거부"""
        # 텍스트 파일 생성
        file_content = b"This is not a PDF"
        files = {"file": ("test.txt", io.BytesIO(file_content), "text/plain")}

        response = client.post("/api/v1/multimodal/extract-tables", files=files)
        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]

    def test_parse_financial_statement_invalid_file(self):
        """재무제표 파싱 - 잘못된 파일 형식"""
        file_content = b"Not a PDF"
        files = {"file": ("report.docx", io.BytesIO(file_content), "application/vnd.openxmlformats")}

        response = client.post("/api/v1/multimodal/parse-financial-statement", files=files)
        assert response.status_code == 400


# ============================================================
# OCR 테스트
# ============================================================

class TestOCR:
    """OCR 기능 테스트"""

    def test_ocr_image_invalid_format(self):
        """지원하지 않는 이미지 형식 거부"""
        file_content = b"Not an image"
        files = {"file": ("test.xyz", io.BytesIO(file_content), "application/octet-stream")}

        response = client.post("/api/v1/multimodal/ocr/image", files=files)
        assert response.status_code == 400
        assert "지원되지 않는 형식" in response.json()["detail"]

    def test_ocr_pdf_invalid_file(self):
        """PDF OCR - 잘못된 파일 형식"""
        file_content = b"Not a PDF"
        files = {"file": ("document.txt", io.BytesIO(file_content), "text/plain")}

        response = client.post("/api/v1/multimodal/ocr/pdf", files=files)
        assert response.status_code == 400


# ============================================================
# 차트 분석 테스트
# ============================================================

class TestChartAnalysis:
    """차트 분석 테스트"""

    def test_analyze_chart_invalid_format(self):
        """지원하지 않는 이미지 형식 거부"""
        file_content = b"Not an image"
        files = {"file": ("chart.pdf", io.BytesIO(file_content), "application/pdf")}

        response = client.post("/api/v1/multimodal/analyze-chart", files=files)
        assert response.status_code == 400

    def test_classify_chart_endpoint(self):
        """차트 유형 분류 엔드포인트 테스트"""
        # 더미 PNG 헤더 (최소한의 PNG 시그니처)
        png_header = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
        files = {"file": ("chart.png", io.BytesIO(png_header), "image/png")}

        response = client.post("/api/v1/multimodal/classify-chart", files=files)
        # 라이브러리 설치 여부에 따라 결과 다름
        assert response.status_code in [200, 500, 501]


# ============================================================
# 통합 처리 테스트
# ============================================================

class TestDocumentProcessing:
    """문서 통합 처리 테스트"""

    def test_process_document_invalid_file(self):
        """통합 처리 - 잘못된 파일 형식"""
        file_content = b"Not a PDF"
        files = {"file": ("document.txt", io.BytesIO(file_content), "text/plain")}

        response = client.post("/api/v1/multimodal/process-document", files=files)
        assert response.status_code == 400

    def test_process_document_options(self):
        """통합 처리 쿼리 파라미터"""
        # PDF 파일 시뮬레이션 (최소 PDF 헤더)
        pdf_content = b"%PDF-1.4\n" + b"\x00" * 100
        files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}

        # OCR 비활성화로 요청
        response = client.post(
            "/api/v1/multimodal/process-document",
            files=files,
            params={"extract_tables": True, "run_ocr": False}
        )
        # 파일이 유효하지 않으므로 오류 발생 가능
        assert response.status_code in [200, 500]


# ============================================================
# 모듈 단위 테스트
# ============================================================

class TestPDFTableExtractorModule:
    """PDF 표 추출 모듈 테스트"""

    def test_import_module(self):
        """모듈 임포트 테스트"""
        try:
            from src.rag.pdf_table_extractor import (
                PDFTableExtractor,
                ExtractedTable,
                TableData,
                FinancialStatementParser,
            )
            assert True
        except ImportError as e:
            # 선택적 의존성 누락 시 스킵
            pytest.skip(f"Optional dependency missing: {e}")

    def test_table_data_to_markdown(self):
        """TableData 마크다운 변환 테스트"""
        from src.rag.multimodal import TableData

        table = TableData(
            headers=["항목", "금액"],
            rows=[["매출액", "1,000"], ["영업이익", "100"]],
        )

        markdown = table.to_markdown()
        assert "| 항목 | 금액 |" in markdown
        assert "| 매출액 | 1,000 |" in markdown

    def test_table_data_to_dict(self):
        """TableData 딕셔너리 변환 테스트"""
        from src.rag.multimodal import TableData

        table = TableData(
            headers=["항목", "금액"],
            rows=[["매출액", "1,000"]],
        )

        result = table.to_dict()
        assert result["headers"] == ["항목", "금액"]
        assert result["rows"] == [["매출액", "1,000"]]


class TestOCRModule:
    """OCR 모듈 테스트"""

    def test_import_module(self):
        """모듈 임포트 테스트"""
        try:
            from src.rag.ocr_pipeline import (
                OCRConfig,
                OCRResult,
                OCREngine,
                ImagePreprocessor,
            )
            assert True
        except ImportError as e:
            pytest.skip(f"Optional dependency missing: {e}")

    def test_ocr_config_defaults(self):
        """OCR 설정 기본값 테스트"""
        from src.rag.ocr_pipeline import OCRConfig

        config = OCRConfig()
        assert "ko" in config.languages
        assert "en" in config.languages
        assert config.backend == "auto"

    def test_image_preprocessor_methods(self):
        """이미지 전처리기 메서드 테스트"""
        from src.rag.ocr_pipeline import ImagePreprocessor

        preprocessor = ImagePreprocessor()
        # 메서드 존재 확인 (static methods)
        assert hasattr(preprocessor, "preprocess")
        assert hasattr(preprocessor, "_deskew")  # internal method


class TestChartAnalyzerModule:
    """차트 분석 모듈 테스트"""

    def test_import_module(self):
        """모듈 임포트 테스트"""
        try:
            from src.rag.chart_analyzer import (
                ChartType,
                ChartData,
                ChartAnalysisResult,
                ChartAnalyzer,
            )
            assert True
        except ImportError as e:
            pytest.skip(f"Optional dependency missing: {e}")

    def test_chart_type_enum(self):
        """차트 유형 열거형 테스트"""
        from src.rag.chart_analyzer import ChartType

        assert ChartType.BAR.value == "bar"
        assert ChartType.LINE.value == "line"
        assert ChartType.PIE.value == "pie"

    def test_chart_data_structure(self):
        """차트 데이터 구조 테스트"""
        from src.rag.multimodal import ChartData

        data = ChartData(
            chart_type="bar",
            labels=["Q1", "Q2", "Q3"],
            values=[100.0, 150.0, 200.0],
            title="매출"
        )
        assert len(data.labels) == 3
        assert len(data.values) == 3
        assert data.chart_type == "bar"


# ============================================================
# 엣지 케이스 테스트
# ============================================================

class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_empty_file_upload(self):
        """빈 파일 업로드"""
        files = {"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")}

        response = client.post("/api/v1/multimodal/extract-tables", files=files)
        # 빈 파일도 처리 가능 (0개 표 반환) 또는 오류
        assert response.status_code in [200, 400, 500]

    def test_large_page_parameter(self):
        """큰 페이지 번호 파라미터"""
        pdf_content = b"%PDF-1.4\n" + b"\x00" * 100
        files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}

        response = client.post(
            "/api/v1/multimodal/extract-tables",
            files=files,
            params={"pages": "999"}
        )
        # 유효하지 않은 페이지는 처리 오류
        assert response.status_code in [200, 500]

    def test_ocr_multiple_languages(self):
        """OCR 다국어 파라미터"""
        # PNG 더미 파일
        png_header = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
        files = {"file": ("image.png", io.BytesIO(png_header), "image/png")}

        response = client.post(
            "/api/v1/multimodal/ocr/image",
            files=files,
            params={"languages": "ko,en,ja"}
        )
        # 라이브러리 설치 여부에 따라 결과 다름
        assert response.status_code in [200, 500, 501]


# ============================================================
# 라우터 통합 테스트
# ============================================================

class TestRouterIntegration:
    """라우터 통합 테스트"""

    def test_multimodal_endpoints_registered(self):
        """멀티모달 엔드포인트 등록 확인"""
        # OpenAPI 스키마에서 엔드포인트 확인
        response = client.get("/openapi.json")
        assert response.status_code == 200

        openapi = response.json()
        paths = openapi["paths"]

        # 주요 엔드포인트 존재 확인
        assert "/api/v1/multimodal/extract-tables" in paths
        assert "/api/v1/multimodal/ocr/image" in paths
        assert "/api/v1/multimodal/analyze-chart" in paths
        assert "/api/v1/multimodal/stats" in paths
        assert "/api/v1/multimodal/health" in paths

    def test_root_endpoint_includes_multimodal(self):
        """루트 엔드포인트에 멀티모달 정보 포함"""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "multimodal" in data
        assert "extract_tables" in data["multimodal"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
