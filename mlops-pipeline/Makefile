.PHONY: test lint train evaluate build deploy-staging deploy-production validate-model clean install

PYTHON := python
PYTEST := $(PYTHON) -m pytest
PIP := $(PYTHON) -m pip
RUFF := $(PYTHON) -m ruff
REGISTRY := ghcr.io
IMAGE_NAME := fraud-detector
TAG := latest

## 의존성 설치
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

## 테스트 실행
test:
	$(PYTEST) tests/ -v --tb=short

## 린트 실행
lint:
	$(RUFF) check src/ tests/

## 모델 학습
train:
	$(PYTHON) scripts/generate_data.py
	$(PYTHON) scripts/train.py

## 모델 평가
evaluate:
	$(PYTHON) scripts/evaluate.py

## 모델 검증 게이트
validate-model:
	$(PYTHON) -c "from src.cicd.pipeline import ModelValidationGate; print('Validation gate ready')"

## Docker 이미지 빌드
build:
	docker build -t $(REGISTRY)/$(IMAGE_NAME):$(TAG) .

## Staging 배포
deploy-staging:
	kubectl apply -f k8s/deployment.yaml -n staging
	kubectl apply -f k8s/service.yaml -n staging

## Production 배포
deploy-production:
	kubectl apply -f k8s/deployment.yaml -n production
	kubectl apply -f k8s/service.yaml -n production
	kubectl apply -f k8s/hpa.yaml -n production

## 클린업
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage coverage.xml
