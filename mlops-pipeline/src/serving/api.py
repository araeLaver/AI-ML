"""
FastAPI ML Serving API
- 실시간 예측 엔드포인트
- 배치 예측 엔드포인트
- 헬스 체크
- 모델 정보
"""

import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging

from .predictor import FraudPredictor

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 전역 predictor 인스턴스
predictor: Optional[FraudPredictor] = None


# --- Pydantic Models ---
class TransactionInput(BaseModel):
    """단일 거래 입력"""

    amount: float = Field(..., description="거래 금액", ge=0)
    time_hour: int = Field(..., description="거래 시간 (0-23)", ge=0, le=23)
    location_distance: float = Field(..., description="평소 위치와의 거리 (km)", ge=0)
    previous_avg_amount: float = Field(..., description="이전 평균 거래 금액", ge=0)
    day_of_week: Optional[int] = Field(0, description="요일 (0=월, 6=일)", ge=0, le=6)
    merchant_category: Optional[int] = Field(0, description="가맹점 카테고리", ge=0)
    transaction_count_1h: Optional[int] = Field(1, description="최근 1시간 거래 횟수", ge=0)
    transaction_count_24h: Optional[int] = Field(5, description="최근 24시간 거래 횟수", ge=0)
    is_weekend: Optional[int] = Field(0, description="주말 여부 (0/1)", ge=0, le=1)
    is_night: Optional[int] = Field(0, description="야간 여부 (0/1)", ge=0, le=1)
    device_change: Optional[int] = Field(0, description="디바이스 변경 여부 (0/1)", ge=0, le=1)


class PredictionOutput(BaseModel):
    """예측 결과"""

    is_fraud: bool = Field(..., description="이상 거래 여부")
    probability: float = Field(..., description="이상 거래 확률")
    risk_level: str = Field(..., description="위험 수준 (MINIMAL/LOW/MEDIUM/HIGH/CRITICAL)")
    threshold: float = Field(..., description="적용된 임계값")
    model_version: str = Field(..., description="모델 버전")
    latency_ms: float = Field(..., description="처리 시간 (ms)")


class BatchInput(BaseModel):
    """배치 입력"""

    transactions: List[TransactionInput] = Field(..., description="거래 목록")


class BatchOutput(BaseModel):
    """배치 출력"""

    predictions: List[Dict[str, Any]] = Field(..., description="예측 결과 목록")
    total_count: int = Field(..., description="전체 건수")
    fraud_count: int = Field(..., description="이상 거래 건수")


class HealthResponse(BaseModel):
    """헬스 체크 응답"""

    status: str
    model_loaded: bool
    model_version: str


class ModelInfoResponse(BaseModel):
    """모델 정보 응답"""

    is_loaded: bool
    model_version: str
    threshold: float
    required_features: List[str]


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    global predictor

    # 시작 시 모델 로드
    model_path = os.getenv("MODEL_PATH", "models/fraud_detector.pkl")
    preprocessor_path = os.getenv("PREPROCESSOR_PATH", "models/preprocessor.pkl")
    threshold = float(os.getenv("THRESHOLD", "0.5"))

    try:
        predictor = FraudPredictor(threshold=threshold)
        if os.path.exists(model_path):
            predictor.load_model(model_path)
            logger.info(f"모델 로드 완료: {model_path}")
        else:
            logger.warning(f"모델 파일 없음: {model_path}")

        if os.path.exists(preprocessor_path):
            predictor.load_preprocessor(preprocessor_path)
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        predictor = FraudPredictor(threshold=threshold)

    yield

    # 종료 시 정리
    logger.info("애플리케이션 종료")


# --- FastAPI App ---
app = FastAPI(
    title="Fraud Detection API",
    description="이상 거래 탐지 ML 서빙 API",
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


# --- Endpoints ---
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """헬스 체크"""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.is_loaded if predictor else False,
        model_version=predictor.model_version if predictor else "unknown",
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """모델 정보 조회"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="모델이 초기화되지 않았습니다")

    info = predictor.get_model_info()
    return ModelInfoResponse(**info)


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(transaction: TransactionInput):
    """단일 거래 예측"""
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")

    try:
        data = transaction.model_dump()
        result = predictor.predict(data)
        return PredictionOutput(**result)
    except Exception as e:
        logger.error(f"예측 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchOutput, tags=["Prediction"])
async def predict_batch(batch: BatchInput):
    """배치 예측"""
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")

    if len(batch.transactions) > 1000:
        raise HTTPException(status_code=400, detail="최대 1000건까지 처리 가능합니다")

    try:
        data_list = [tx.model_dump() for tx in batch.transactions]

        # 비동기 처리
        loop = asyncio.get_event_loop()
        predictions = await loop.run_in_executor(None, predictor.predict_batch, data_list)

        fraud_count = sum(1 for p in predictions if p["is_fraud"])

        return BatchOutput(
            predictions=predictions,
            total_count=len(predictions),
            fraud_count=fraud_count,
        )
    except Exception as e:
        logger.error(f"배치 예측 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/threshold", tags=["Model"])
async def update_threshold(threshold: float):
    """임계값 업데이트"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="모델이 초기화되지 않았습니다")

    if not 0 <= threshold <= 1:
        raise HTTPException(status_code=400, detail="임계값은 0과 1 사이여야 합니다")

    predictor.update_threshold(threshold)
    return {"message": f"임계값이 {threshold}로 업데이트되었습니다"}


@app.post("/model/reload", tags=["Model"])
async def reload_model(background_tasks: BackgroundTasks):
    """모델 재로드 (백그라운드)"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="모델이 초기화되지 않았습니다")

    def reload():
        model_path = os.getenv("MODEL_PATH", "models/fraud_detector.pkl")
        predictor.load_model(model_path)

    background_tasks.add_task(reload)
    return {"message": "모델 재로드가 백그라운드에서 시작되었습니다"}


# 루트 엔드포인트
@app.get("/", tags=["System"])
async def root():
    """API 정보"""
    return {
        "name": "Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
