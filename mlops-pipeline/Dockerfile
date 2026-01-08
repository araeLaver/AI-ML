# MLOps Pipeline - Fraud Detection
FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY src/ ./src/
COPY models/ ./models/
COPY scripts/ ./scripts/

# 환경 변수
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/fraud_detector.pkl
ENV PREPROCESSOR_PATH=/app/models/preprocessor.pkl
ENV PORT=8000

# 포트 노출
EXPOSE 8000

# 헬스 체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 실행
CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
