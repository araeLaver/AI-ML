# Finance RAG API - Makefile
# Common development and deployment commands

.PHONY: help install dev test docker-build docker-up docker-down docker-logs clean lint

# Default target
help:
	@echo "Finance RAG API - Available Commands"
	@echo "======================================"
	@echo ""
	@echo "Development:"
	@echo "  make install     - Install dependencies"
	@echo "  make dev         - Run development server"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run linter"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-up     - Start all services"
	@echo "  make docker-down   - Stop all services"
	@echo "  make docker-logs   - View logs"
	@echo "  make docker-dev    - Start in dev mode"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean       - Clean cache and temp files"
	@echo "  make pull-model  - Pull Ollama model"

# ==========================================================================
# Development
# ==========================================================================

install:
	pip install -r requirements.txt

dev:
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src/ tests/
	black src/ tests/ --check

format:
	black src/ tests/
	isort src/ tests/

# ==========================================================================
# Docker - Production
# ==========================================================================

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-restart:
	docker-compose restart

# ==========================================================================
# Docker - Development
# ==========================================================================

docker-dev:
	docker-compose -f docker-compose.dev.yml up -d

docker-dev-down:
	docker-compose -f docker-compose.dev.yml down

docker-dev-logs:
	docker-compose -f docker-compose.dev.yml logs -f

# ==========================================================================
# Ollama Model Management
# ==========================================================================

pull-model:
	ollama pull llama3.2:latest

pull-embedding:
	ollama pull nomic-embed-text:latest

# ==========================================================================
# Cleanup
# ==========================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov/ 2>/dev/null || true

# ==========================================================================
# Database
# ==========================================================================

reset-db:
	rm -rf db/*
	@echo "Database reset complete"
