# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Fine-tuned Embedding: 금융 도메인 특화 임베딩 모델
- Real-time Update: DART API 실시간 공시 연동
- Multi-modal: 공시 내 표/차트 인식
- Redis 캐싱: 자주 검색하는 쿼리 캐싱

### Added
- Query Expansion 기능 구현
  - 금융 동의어 사전 200+ 항목 (재무지표, 기업약어, 산업용어 등)
  - QueryExpander 클래스 (쿼리 확장 및 정규화)
  - HybridSearcher 통합 (자동 쿼리 확장)
  - 테스트 33개 추가 (100% 통과)
- CI/CD 파이프라인 구축
  - GitHub Actions 테스트 자동화 (ci.yml)
  - Docker 이미지 빌드 자동화 (docker.yml)
  - Codecov 커버리지 연동
- Phase 2 상세 계획 문서 (docs/phase2-finance-rag-enhancement.md)

---

## [1.2.0] - 2026-01-21

### Added
- Premium UI 기능 - 세션 관리, PDF/CSV 내보내기 (`bce79d7`)
- Finance RAG Demo Production-grade 업그레이드 (`f2048b9`)
- Production-grade RAG 개선 (`04d2b25`)
  - Kiwi 한국어 형태소 분석기 도입
  - Cross-Encoder Re-ranking 추가
  - 벤치마크 프레임워크 구현

### Changed
- Finance RAG 아키텍처 문서 추가 (`eebd60e`)

### Fixed
- HuggingFace Space 링크 수정 (`cd7bd6d`)

---

## [1.1.0] - 2026-01-12

### Added
- HuggingFace Inference API 기반 RAG 데모 (`3d2a622`)
- HuggingFace Spaces 데모 및 자동 동기화 CI/CD (`1173705`, `f9a4bb4`)
- Grafana 대시보드 및 Alert Rules (`0f99d02`)
- AWS ML Specialty 12주 학습 가이드 (`02b2a84`)
- 포트폴리오 기술 문서 (`9313404`)

### Changed
- 포트폴리오 정리 - 테스트/Docker 추가 및 README 개편 (`3553f5f`)
- Vercel 배포 설정 추가 (`f90e2cb`)

### Fixed
- 테스트 실패 수정 (`ca02899`)

---

## [1.0.0] - 2026-01-08

### Added
- **Step 1-2: Financial Analysis** - ML + LLM 기초
  - NumPy/Pandas 데이터 처리
  - scikit-learn 모델 (Random Forest, Isolation Forest)
  - OpenAI/Claude API 연동
  - 프롬프트 엔지니어링 (Zero-shot, Few-shot, CoT)

- **Step 3: Finance RAG API** - 검색 증강 생성
  - Hybrid Search (Vector + BM25 + RRF)
  - ChromaDB 벡터 데이터베이스
  - DART 공시 데이터 (10,000+ 문서)

- **Step 4: Code Review Agent** - AI Agent (`c1e974d`)
  - 3개 특화 에이전트 (Security, Performance, Style)
  - GitHub PR 자동 리뷰
  - LangChain/LangGraph 기반

- **Step 5: MLOps Pipeline**
  - DVC 데이터 버전 관리
  - MLflow 실험 추적
  - Prometheus/Grafana 모니터링
  - Kubernetes 배포 설정

- **Step 6: Financial Fine-tuning**
  - LoRA/QLoRA Fine-tuning
  - 100+ 금융 도메인 데이터셋
  - 4-bit 양자화 추론

- **포트폴리오**
  - Next.js 기반 포트폴리오 웹사이트
  - Streamlit 대시보드

### Changed
- 2025 트렌드 UI 적용 - 에디토리얼/미니멀 디자인 (`9533e53`)
- Streamlit UI 전면 개편 (`9070b87`)
- UI 전면 리디자인 (bpco.kr 스타일) (`63ff9d2`)

---

## [0.2.0] - 2025-12-19

### Added
- 고급 RAG 기능 추가 (`8208db8`)
- 클라우드 배포 지원 및 포트폴리오 데모 강화 (`27c9849`)
- README 문서화 (`5831e23`, `b4118c9`)

---

## [0.1.0] - 2025-12-18

### Added
- Initial commit: AI/ML 학습 및 포트폴리오 프로젝트 (`01c0824`)
- Dev Container 설정 (`636fc14`)

---

## Summary

| Version | Date | Highlights |
|---------|------|------------|
| 1.2.0 | 2026-01-21 | Production-grade RAG, Premium UI |
| 1.1.0 | 2026-01-12 | HuggingFace Spaces, AWS ML Guide |
| 1.0.0 | 2026-01-08 | Step 1-6 완성, 189개 테스트 통과 |
| 0.2.0 | 2025-12-19 | 고급 RAG, 클라우드 배포 |
| 0.1.0 | 2025-12-18 | 프로젝트 시작 |

---

## Test Coverage

- **Total Tests**: 189
- **Pass Rate**: 100%

| Project | Tests |
|---------|-------|
| financial-analysis | 20 |
| finance-rag-api | 53 |
| code-review-agent | 55 |
| mlops-pipeline | 40 |
| financial-finetuning | 21 |
