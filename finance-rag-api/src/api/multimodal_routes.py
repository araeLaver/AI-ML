# -*- coding: utf-8 -*-
"""
멀티모달 API 라우터

PDF 표 추출, OCR, 차트 분석 API 엔드포인트
"""

import base64
import io
import tempfile
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..core.logging import get_logger

logger = get_logger(__name__)

# 라우터 생성
multimodal_router = APIRouter(tags=["Multimodal"])


# ============================================================
# 스키마 정의
# ============================================================

class TableExtractionResponse(BaseModel):
    """표 추출 응답"""
    success: bool = Field(..., description="성공 여부")
    tables: list[dict] = Field(..., description="추출된 표 목록")
    total_tables: int = Field(..., description="총 표 수")
    financial_tables: int = Field(..., description="재무제표 수")
    message: str = Field(..., description="결과 메시지")


class OCRResponse(BaseModel):
    """OCR 응답"""
    success: bool = Field(..., description="성공 여부")
    text: str = Field(..., description="추출된 텍스트")
    confidence: float = Field(..., description="신뢰도")
    language: Optional[str] = Field(None, description="감지된 언어")
    processing_time: float = Field(..., description="처리 시간 (초)")


class ChartAnalysisResponse(BaseModel):
    """차트 분석 응답"""
    success: bool = Field(..., description="성공 여부")
    chart_type: str = Field(..., description="차트 유형")
    confidence: float = Field(..., description="신뢰도")
    title: str = Field("", description="차트 제목")
    labels: list[str] = Field(default_factory=list, description="레이블")
    values: list[float] = Field(default_factory=list, description="데이터 값")
    colors: list[str] = Field(default_factory=list, description="사용된 색상")


class MultimodalStatsResponse(BaseModel):
    """멀티모달 통계 응답"""
    pdf_table_extractor: dict = Field(..., description="PDF 표 추출 통계")
    ocr_engine: dict = Field(..., description="OCR 통계")
    chart_analyzer: dict = Field(..., description="차트 분석 통계")


# ============================================================
# PDF 표 추출 API
# ============================================================

@multimodal_router.post(
    "/extract-tables",
    response_model=TableExtractionResponse,
    summary="PDF에서 표 추출",
    description="PDF 파일에서 표를 추출합니다. 재무제표를 자동으로 인식합니다."
)
async def extract_tables_from_pdf(
    file: UploadFile = File(..., description="PDF 파일"),
    pages: str = Query("all", description="추출할 페이지 ('all', '1,2,3', '1-5')"),
):
    """PDF에서 표 추출 API"""
    # 파일 검증
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 지원합니다.")

    try:
        from ..rag.pdf_table_extractor import PDFTableExtractor

        extractor = PDFTableExtractor()

        # 파일 읽기
        content = await file.read()

        # 표 추출
        tables = extractor.extract_tables_from_bytes(content, pages)

        # 응답 생성
        table_dicts = []
        financial_count = 0

        for table in tables:
            table_dict = {
                "page": table.page_number,
                "method": table.extraction_method,
                "accuracy": table.accuracy,
                "is_financial": table.is_financial,
                "headers": table.table_data.headers,
                "rows": table.table_data.rows,
                "markdown": table.table_data.to_markdown(),
            }
            table_dicts.append(table_dict)

            if table.is_financial:
                financial_count += 1

        logger.info(f"Extracted {len(tables)} tables from {file.filename}")

        return TableExtractionResponse(
            success=True,
            tables=table_dicts,
            total_tables=len(tables),
            financial_tables=financial_count,
            message=f"{file.filename}에서 {len(tables)}개의 표를 추출했습니다.",
        )

    except ImportError as e:
        raise HTTPException(
            status_code=501,
            detail="PDF 표 추출 라이브러리가 설치되지 않았습니다. (camelot-py 또는 pdfplumber 필요)"
        )
    except Exception as e:
        logger.error(f"Table extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@multimodal_router.post(
    "/parse-financial-statement",
    summary="재무제표 파싱",
    description="PDF에서 재무제표를 추출하고 구조화합니다."
)
async def parse_financial_statement(
    file: UploadFile = File(..., description="PDF 파일"),
):
    """재무제표 파싱 API"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 지원합니다.")

    try:
        from ..rag.pdf_table_extractor import PDFTableExtractor, FinancialStatementParser

        extractor = PDFTableExtractor()
        parser = FinancialStatementParser()

        content = await file.read()
        tables = extractor.extract_tables_from_bytes(content)

        statements = []
        for table in tables:
            if table.is_financial:
                parsed = parser.parse_financial_statement(table)
                if parsed:
                    statements.append(parsed)

        return JSONResponse({
            "success": True,
            "statements": statements,
            "total_count": len(statements),
            "message": f"{len(statements)}개의 재무제표를 파싱했습니다.",
        })

    except Exception as e:
        logger.error(f"Financial statement parsing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# OCR API
# ============================================================

@multimodal_router.post(
    "/ocr/image",
    response_model=OCRResponse,
    summary="이미지 OCR",
    description="이미지에서 텍스트를 추출합니다. 한글/영어 지원."
)
async def ocr_image(
    file: UploadFile = File(..., description="이미지 파일"),
    languages: str = Query("ko,en", description="인식 언어 (쉼표 구분)"),
):
    """이미지 OCR API"""
    # 파일 검증
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"지원되지 않는 형식: {ext}")

    try:
        from ..rag.ocr_pipeline import OCREngine, OCRConfig

        lang_list = [lang.strip() for lang in languages.split(",")]
        config = OCRConfig(languages=lang_list)
        engine = OCREngine(config)

        content = await file.read()
        result = engine.recognize(content)

        logger.info(f"OCR processed: {file.filename}, confidence: {result.confidence:.2f}")

        return OCRResponse(
            success=True,
            text=result.text,
            confidence=result.confidence,
            language=result.language,
            processing_time=result.processing_time,
        )

    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="OCR 라이브러리가 설치되지 않았습니다. (easyocr 또는 pytesseract 필요)"
        )
    except Exception as e:
        logger.error(f"OCR error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@multimodal_router.post(
    "/ocr/pdf",
    summary="PDF OCR",
    description="스캔된 PDF에서 텍스트를 추출합니다."
)
async def ocr_pdf(
    file: UploadFile = File(..., description="PDF 파일"),
    pages: str = Query("0", description="처리할 페이지 (쉼표 구분, 0-indexed)"),
    languages: str = Query("ko,en", description="인식 언어"),
):
    """PDF OCR API"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 지원합니다.")

    try:
        from ..rag.ocr_pipeline import PDFOCRProcessor, OCRConfig

        lang_list = [lang.strip() for lang in languages.split(",")]
        config = OCRConfig(languages=lang_list)
        processor = PDFOCRProcessor(config)

        # 페이지 번호 파싱
        page_nums = [int(p.strip()) for p in pages.split(",")]

        content = await file.read()
        results = processor.process_pdf_bytes(content, page_nums)

        # 응답 생성
        page_results = []
        total_text = []

        for i, result in enumerate(results):
            page_results.append({
                "page": page_nums[i] if i < len(page_nums) else i,
                "text": result.text,
                "confidence": result.confidence,
                "processing_time": result.processing_time,
            })
            total_text.append(result.text)

        return JSONResponse({
            "success": True,
            "pages": page_results,
            "total_pages": len(results),
            "combined_text": "\n\n".join(total_text),
        })

    except Exception as e:
        logger.error(f"PDF OCR error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 차트 분석 API
# ============================================================

@multimodal_router.post(
    "/analyze-chart",
    response_model=ChartAnalysisResponse,
    summary="차트 이미지 분석",
    description="차트 이미지의 유형을 분류하고 데이터를 추출합니다."
)
async def analyze_chart_image(
    file: UploadFile = File(..., description="차트 이미지"),
):
    """차트 분석 API"""
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"지원되지 않는 형식: {ext}")

    try:
        from ..rag.chart_analyzer import ChartAnalyzer

        analyzer = ChartAnalyzer()

        content = await file.read()
        result = analyzer.analyze(content)

        logger.info(f"Chart analyzed: {file.filename}, type: {result.chart_type.value}")

        return ChartAnalysisResponse(
            success=True,
            chart_type=result.chart_type.value,
            confidence=result.confidence,
            title=result.title,
            labels=result.labels,
            values=result.chart_data.values if result.chart_data else [],
            colors=result.colors,
        )

    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="차트 분석 라이브러리가 설치되지 않았습니다. (PIL, opencv-python 필요)"
        )
    except Exception as e:
        logger.error(f"Chart analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@multimodal_router.post(
    "/classify-chart",
    summary="차트 유형 분류",
    description="차트 이미지의 유형만 분류합니다."
)
async def classify_chart(
    file: UploadFile = File(..., description="차트 이미지"),
):
    """차트 유형 분류 API"""
    try:
        from ..rag.chart_analyzer import classify_chart_type

        content = await file.read()
        chart_type, confidence = classify_chart_type(content)

        return JSONResponse({
            "success": True,
            "chart_type": chart_type.value,
            "confidence": confidence,
        })

    except Exception as e:
        logger.error(f"Chart classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 통합 처리 API
# ============================================================

@multimodal_router.post(
    "/process-document",
    summary="문서 통합 처리",
    description="PDF 문서를 종합 분석합니다 (표 추출 + OCR + 차트 감지)."
)
async def process_document_multimodal(
    file: UploadFile = File(..., description="PDF 파일"),
    extract_tables: bool = Query(True, description="표 추출 여부"),
    run_ocr: bool = Query(False, description="OCR 실행 여부 (스캔 문서용)"),
):
    """문서 통합 처리 API"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 지원합니다.")

    result = {
        "success": True,
        "filename": file.filename,
        "tables": [],
        "ocr_text": "",
        "summary": {},
    }

    content = await file.read()

    # 표 추출
    if extract_tables:
        try:
            from ..rag.pdf_table_extractor import PDFTableExtractor

            extractor = PDFTableExtractor()
            tables = extractor.extract_tables_from_bytes(content)

            result["tables"] = [
                {
                    "page": t.page_number,
                    "is_financial": t.is_financial,
                    "accuracy": t.accuracy,
                    "markdown": t.table_data.to_markdown(),
                }
                for t in tables
            ]
            result["summary"]["total_tables"] = len(tables)
            result["summary"]["financial_tables"] = sum(1 for t in tables if t.is_financial)

        except Exception as e:
            result["summary"]["table_extraction_error"] = str(e)

    # OCR (선택적)
    if run_ocr:
        try:
            from ..rag.ocr_pipeline import PDFOCRProcessor

            processor = PDFOCRProcessor()
            ocr_results = processor.process_pdf_bytes(content, [0])  # 첫 페이지만

            if ocr_results:
                result["ocr_text"] = ocr_results[0].text
                result["summary"]["ocr_confidence"] = ocr_results[0].confidence

        except Exception as e:
            result["summary"]["ocr_error"] = str(e)

    return JSONResponse(result)


# ============================================================
# 통계 및 상태 API
# ============================================================

@multimodal_router.get(
    "/stats",
    response_model=MultimodalStatsResponse,
    summary="멀티모달 통계",
    description="멀티모달 처리 통계를 조회합니다."
)
async def get_multimodal_stats():
    """멀티모달 통계 조회 API"""
    stats = {
        "pdf_table_extractor": {"available": False},
        "ocr_engine": {"available": False},
        "chart_analyzer": {"available": False},
    }

    # PDF 표 추출기
    try:
        from ..rag.pdf_table_extractor import PDFTableExtractor, CAMELOT_AVAILABLE, PDFPLUMBER_AVAILABLE
        stats["pdf_table_extractor"] = {
            "available": True,
            "camelot_available": CAMELOT_AVAILABLE,
            "pdfplumber_available": PDFPLUMBER_AVAILABLE,
        }
    except ImportError:
        pass

    # OCR 엔진
    try:
        from ..rag.ocr_pipeline import EASYOCR_AVAILABLE, TESSERACT_AVAILABLE
        stats["ocr_engine"] = {
            "available": True,
            "easyocr_available": EASYOCR_AVAILABLE,
            "tesseract_available": TESSERACT_AVAILABLE,
        }
    except ImportError:
        pass

    # 차트 분석기
    try:
        from ..rag.chart_analyzer import CV2_AVAILABLE, PIL_AVAILABLE
        stats["chart_analyzer"] = {
            "available": True,
            "opencv_available": CV2_AVAILABLE,
            "pil_available": PIL_AVAILABLE,
        }
    except ImportError:
        pass

    return MultimodalStatsResponse(**stats)


@multimodal_router.get(
    "/health",
    summary="멀티모달 헬스체크",
    description="멀티모달 서비스 상태를 확인합니다."
)
async def multimodal_health():
    """멀티모달 헬스체크"""
    return {
        "status": "ok",
        "service": "multimodal",
        "features": {
            "table_extraction": True,
            "ocr": True,
            "chart_analysis": True,
        },
    }
