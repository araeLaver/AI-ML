# FastAPI Inference Server
"""
Fine-tuned 금융 LLM API 서버

Features:
- 금융 도메인 특화 API 엔드포인트
- Pydantic 기반 요청/응답 검증
- 에러 처리 및 로깅
- 헬스 체크 및 모니터링
"""

import os
import sys
import logging
import time
from typing import Optional, List
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, validator

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import FinancialLLMInference, InferenceError, ModelNotLoadedError, GenerationError

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 전역 모델 인스턴스
inference_engine: Optional[FinancialLLMInference] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 라이프사이클 관리"""
    global inference_engine

    # 시작 시 모델 로드 (환경변수로 설정 가능)
    base_model = os.getenv("BASE_MODEL", "beomi/Llama-3-Open-Ko-8B")
    adapter_path = os.getenv("ADAPTER_PATH", None)
    load_model = os.getenv("LOAD_MODEL_ON_START", "false").lower() == "true"

    if load_model:
        print(f"Loading model on startup: {base_model}")
        inference_engine = FinancialLLMInference(
            base_model=base_model,
            adapter_path=adapter_path,
        )
        inference_engine.load()
    else:
        # 지연 로딩을 위한 설정만 저장
        inference_engine = FinancialLLMInference(
            base_model=base_model,
            adapter_path=adapter_path,
        )
        print("Model configured for lazy loading")

    yield

    # 종료 시 정리
    if inference_engine and inference_engine.model is not None:
        del inference_engine.model
        print("Model unloaded")


# FastAPI 앱 생성
app = FastAPI(
    title="Financial LLM API",
    description="Fine-tuned 금융 도메인 LLM 추론 API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Enums
class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# Request/Response 모델
class GenerateRequest(BaseModel):
    instruction: str = Field(..., description="지시사항", min_length=1, max_length=4096)
    input_text: str = Field(default="", description="입력 텍스트 (선택)", max_length=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1, le=200)
    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)

    @validator("instruction")
    def instruction_not_empty(cls, v):
        if not v.strip():
            raise ValueError("instruction cannot be empty or whitespace only")
        return v


class GenerateResponse(BaseModel):
    response: str
    instruction: str
    input_text: str
    generation_time_ms: Optional[float] = None


class ModelInfoResponse(BaseModel):
    base_model: str
    adapter_path: Optional[str]
    load_in_4bit: bool
    device: Optional[str]
    dtype: Optional[str]
    status: str
    cache_size: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# 금융 특화 요청/응답 모델
class FraudDetectionRequest(BaseModel):
    transaction_info: str = Field(..., description="거래 정보", min_length=10, max_length=4096)
    transaction_amount: Optional[float] = Field(None, ge=0, description="거래 금액")
    transaction_time: Optional[str] = Field(None, description="거래 시간")
    location: Optional[str] = Field(None, description="거래 위치")

    @validator("transaction_info")
    def validate_transaction_info(cls, v):
        if not v.strip():
            raise ValueError("transaction_info cannot be empty")
        return v


class FraudDetectionResponse(BaseModel):
    analysis: str
    transaction_info: str
    risk_level: Optional[RiskLevel] = None
    risk_score: Optional[int] = Field(None, ge=0, le=100)
    generation_time_ms: Optional[float] = None


class InvestmentAnalysisRequest(BaseModel):
    query: str = Field(..., description="투자 질문", min_length=5, max_length=4096)
    investment_type: Optional[str] = Field(None, description="투자 유형 (주식, 채권, ETF 등)")
    risk_tolerance: Optional[str] = Field(None, description="위험 성향 (보수적, 중립, 공격적)")

    @validator("query")
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("query cannot be empty")
        return v


class InvestmentAnalysisResponse(BaseModel):
    analysis: str
    query: str
    investment_type: Optional[str] = None
    generation_time_ms: Optional[float] = None


class ProductExplanationRequest(BaseModel):
    product_info: str = Field(..., description="상품 정보", min_length=5, max_length=4096)
    product_type: Optional[str] = Field(None, description="상품 유형")
    target_audience: Optional[str] = Field(None, description="대상 고객층")

    @validator("product_info")
    def validate_product_info(cls, v):
        if not v.strip():
            raise ValueError("product_info cannot be empty")
        return v


class ProductExplanationResponse(BaseModel):
    explanation: str
    product_info: str
    product_type: Optional[str] = None
    generation_time_ms: Optional[float] = None


# API 엔드포인트
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스 체크"""
    model_loaded = inference_engine is not None and inference_engine.model is not None
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """모델 정보 조회"""
    if inference_engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not initialized")

    info = inference_engine.get_model_info()
    return ModelInfoResponse(
        base_model=inference_engine.base_model_name,
        adapter_path=inference_engine.adapter_path,
        load_in_4bit=inference_engine.load_in_4bit,
        device=info.get("device"),
        dtype=info.get("dtype"),
        status=info.get("status", "loaded"),
    )


@app.post("/model/load")
async def load_model():
    """모델 로드 (지연 로딩)"""
    if inference_engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not initialized")

    if inference_engine.model is None:
        inference_engine.load()
        return {"status": "loaded", "message": "Model loaded successfully"}

    return {"status": "already_loaded", "message": "Model was already loaded"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """텍스트 생성"""
    if inference_engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not initialized")

    try:
        response = inference_engine.generate(
            instruction=request.instruction,
            input_text=request.input_text,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_new_tokens=request.max_new_tokens,
            repetition_penalty=request.repetition_penalty,
        )

        return GenerateResponse(
            response=response,
            instruction=request.instruction,
            input_text=request.input_text,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """스트리밍 텍스트 생성"""
    if inference_engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not initialized")

    def stream_generator():
        try:
            for token in inference_engine.generate_stream(
                instruction=request.instruction,
                input_text=request.input_text,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                max_new_tokens=request.max_new_tokens,
                repetition_penalty=request.repetition_penalty,
            ):
                yield token
        except Exception as e:
            yield f"\n[Error: {str(e)}]"

    return StreamingResponse(
        stream_generator(),
        media_type="text/plain",
    )


# 금융 특화 엔드포인트
@app.post("/financial/fraud-detection", response_model=FraudDetectionResponse)
async def detect_fraud(request: FraudDetectionRequest):
    """
    이상 거래 탐지

    거래 정보를 분석하여 이상 거래 여부를 판단하고 위험 수준을 평가합니다.
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    start_time = time.time()

    # 추가 컨텍스트 구성
    context_parts = [request.transaction_info]
    if request.transaction_amount:
        context_parts.append(f"거래금액: {request.transaction_amount:,.0f}원")
    if request.transaction_time:
        context_parts.append(f"거래시간: {request.transaction_time}")
    if request.location:
        context_parts.append(f"거래위치: {request.location}")

    full_context = "\n".join(context_parts)

    instruction = "다음 금융 거래 정보를 분석하여 이상 거래 여부를 판단하고, 위험 수준(LOW/MEDIUM/HIGH/CRITICAL)과 리스크 점수(0-100)를 포함하여 근거를 설명해주세요."

    try:
        response = inference_engine.generate(
            instruction=instruction,
            input_text=full_context,
        )

        elapsed_ms = (time.time() - start_time) * 1000

        # 위험 수준 파싱 시도
        risk_level = None
        risk_score = None
        response_upper = response.upper()
        for level in RiskLevel:
            if level.value in response_upper:
                risk_level = level
                break

        return FraudDetectionResponse(
            analysis=response,
            transaction_info=request.transaction_info,
            risk_level=risk_level,
            risk_score=risk_score,
            generation_time_ms=elapsed_ms,
        )
    except GenerationError as e:
        logger.error(f"Fraud detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/financial/investment-analysis", response_model=InvestmentAnalysisResponse)
async def analyze_investment(request: InvestmentAnalysisRequest):
    """
    투자 분석

    투자 관련 질문에 대해 전문적인 분석과 리스크 요인을 설명합니다.
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    start_time = time.time()

    # 추가 컨텍스트 구성
    context_parts = [request.query]
    if request.investment_type:
        context_parts.append(f"투자 유형: {request.investment_type}")
    if request.risk_tolerance:
        context_parts.append(f"위험 성향: {request.risk_tolerance}")

    full_context = "\n".join(context_parts)

    instruction = "금융 전문가로서 다음 투자 관련 질문에 답변해주세요. 객관적인 분석과 함께 리스크 요인도 설명해주세요."

    try:
        response = inference_engine.generate(
            instruction=instruction,
            input_text=full_context,
        )

        elapsed_ms = (time.time() - start_time) * 1000

        return InvestmentAnalysisResponse(
            analysis=response,
            query=request.query,
            investment_type=request.investment_type,
            generation_time_ms=elapsed_ms,
        )
    except GenerationError as e:
        logger.error(f"Investment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/financial/product-explanation", response_model=ProductExplanationResponse)
async def explain_product(request: ProductExplanationRequest):
    """
    금융 상품 설명

    금융 상품에 대해 이해하기 쉽게 설명하고 장단점을 분석합니다.
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    start_time = time.time()

    # 추가 컨텍스트 구성
    context_parts = [request.product_info]
    if request.product_type:
        context_parts.append(f"상품 유형: {request.product_type}")
    if request.target_audience:
        context_parts.append(f"대상 고객층: {request.target_audience}")

    full_context = "\n".join(context_parts)

    instruction = "다음 금융 상품에 대해 일반 고객이 이해하기 쉽게 설명해주세요. 장단점과 적합한 투자자 유형도 포함해주세요."

    try:
        response = inference_engine.generate(
            instruction=instruction,
            input_text=full_context,
        )

        elapsed_ms = (time.time() - start_time) * 1000

        return ProductExplanationResponse(
            explanation=response,
            product_info=request.product_info,
            product_type=request.product_type,
            generation_time_ms=elapsed_ms,
        )
    except GenerationError as e:
        logger.error(f"Product explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 에러 핸들러
@app.exception_handler(InferenceError)
async def inference_error_handler(request: Request, exc: InferenceError):
    """추론 에러 핸들러"""
    logger.error(f"Inference error: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InferenceError",
            detail=str(exc),
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    """일반 에러 핸들러"""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=type(exc).__name__,
            detail=str(exc),
        ).dict(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
