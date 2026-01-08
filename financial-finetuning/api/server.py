# FastAPI Inference Server
"""
Fine-tuned 금융 LLM API 서버
"""

import os
import sys
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import FinancialLLMInference


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


# Request/Response 모델
class GenerateRequest(BaseModel):
    instruction: str = Field(..., description="지시사항")
    input_text: str = Field(default="", description="입력 텍스트 (선택)")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1)
    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)


class GenerateResponse(BaseModel):
    response: str
    instruction: str
    input_text: str


class ModelInfoResponse(BaseModel):
    base_model: str
    adapter_path: Optional[str]
    load_in_4bit: bool
    device: Optional[str]
    dtype: Optional[str]
    status: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


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
@app.post("/financial/fraud-detection")
async def detect_fraud(transaction_info: str):
    """이상 거래 탐지"""
    if inference_engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not initialized")

    instruction = "다음 금융 거래 정보를 분석하여 이상 거래 여부를 판단하고, 위험 수준과 근거를 설명해주세요."

    response = inference_engine.generate(
        instruction=instruction,
        input_text=transaction_info,
    )

    return {
        "analysis": response,
        "transaction_info": transaction_info,
    }


@app.post("/financial/investment-analysis")
async def analyze_investment(query: str):
    """투자 분석"""
    if inference_engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not initialized")

    instruction = "금융 전문가로서 다음 투자 관련 질문에 답변해주세요. 객관적인 분석과 함께 리스크 요인도 설명해주세요."

    response = inference_engine.generate(
        instruction=instruction,
        input_text=query,
    )

    return {
        "analysis": response,
        "query": query,
    }


@app.post("/financial/product-explanation")
async def explain_product(product_info: str):
    """금융 상품 설명"""
    if inference_engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not initialized")

    instruction = "다음 금융 상품에 대해 일반 고객이 이해하기 쉽게 설명해주세요. 장단점과 적합한 투자자 유형도 포함해주세요."

    response = inference_engine.generate(
        instruction=instruction,
        input_text=product_info,
    )

    return {
        "explanation": response,
        "product_info": product_info,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
