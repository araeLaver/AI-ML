# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Real-time Update: DART API 실시간 공시 연동
- Multi-modal: 공시 내 표/차트 인식
- Ray Tune 하이퍼파라미터 자동 튜닝
- vLLM 추론 최적화
- A/B 테스트 프레임워크

---

## [1.6.0] - 2026-01-25

### Added
- **Fine-tuning 강화 (Phase 5 완료)**
  - 데이터셋 확장기 (`src/data/dataset_augmentor.py`)
    - DatasetAugmentor: 템플릿/수치 변형 기반 데이터 증강
    - SyntheticDataGenerator: 카테고리별 합성 데이터 생성
    - AugmentationConfig: 목표 개수, 변형 비율 설정
    - AugmentedSample 데이터클래스
    - expand_dataset() 편의 함수
    - 20개 테스트 추가 (100% 통과)
  - DPO 학습 모듈 (`src/training/dpo_trainer.py`)
    - FinancialDPOTrainer: TRL DPOTrainer 래퍼
    - PreferenceDataset: 선호도 페어 데이터셋 관리
    - PreferencePair: 선호/비선호 응답 페어
    - FinancialPreferenceGenerator: 3가지 전략 기반 페어 생성
      - quality_degradation: 품질 저하 (구조/키워드 제거)
      - length_variation: 길이 변형 (요약/축약)
      - factual_errors: 사실 오류 주입
    - DPOTrainingConfig: YAML 설정 로드, LoRA/QLoRA 지원
    - 24개 테스트 추가 (100% 통과)
  - 테스트 53개 추가 (21 → 74개, 100% 통과)

---

## [1.5.0] - 2026-01-25

### Added
- **MLOps Pipeline 고도화 (Phase 4 완료)**
  - Feast Feature Store 통합 (`src/features/feast_store.py`)
    - FeastFeatureStore: 피처 그룹 중앙 관리
    - FeatureDefinition, FeatureGroup 데이터클래스
    - 로컬 파일 기반 폴백 (Feast 미설치 환경 지원)
    - 히스토리컬 피처 조회 및 포인트-인-타임 조인
    - 피처 통계 계산 및 버전 해시 생성
    - 26개 테스트 추가 (100% 통과)
  - Great Expectations 데이터 검증 (`src/data/ge_expectations.py`)
    - GreatExpectationsValidator: Expectation 기반 검증
    - ExpectationResult, ValidationReport 데이터클래스
    - 사기 탐지용 사전 정의 Expectation Suite
    - 스키마 검증 및 자동 데이터 프로파일링
    - 8가지 Expectation 타입 지원
    - 29개 테스트 추가 (100% 통과)
  - Airflow DAG 빌더 (`src/orchestration/airflow_dags.py`)
    - AirflowDAGBuilder: DAG 생성 및 관리
    - DAGConfig, TaskDefinition, RetryPolicy 데이터클래스
    - 사기 탐지 파이프라인 DAG 템플릿
    - 자동 재학습 DAG 템플릿
    - 로컬 시뮬레이션 모드 (Airflow 미설치 환경)
    - DAG Python 코드 내보내기 기능
    - 29개 테스트 추가 (100% 통과)
  - 테스트 84개 추가 (40 → 124개, 100% 통과)

---

## [1.4.0] - 2026-01-25

### Added
- **Code Review Agent 확장 (Phase 3 완료)**
  - OWASP Top 10 보안 에이전트 (`agents/owasp_agent.py`)
    - 10개 OWASP 2021 카테고리 (A01~A10) 완전 지원
    - OWASPAgent: LLM 기반 심층 분석
    - OWASPStaticAnalyzer: 패턴 기반 정적 분석 (LLM 불필요)
    - Python, JavaScript, Java, Go 언어별 취약점 패턴
  - Java/Go 언어 지원 강화 (`tools/code_analyzer.py`)
    - LanguageConfig 데이터클래스로 6개 언어 설정 체계화
    - 새 추출 메서드: extract_interfaces(), extract_structs()
    - get_full_analysis(): 통합 분석 메서드
    - 언어별 보안/품질 이슈 패턴 (Java, Go, Rust, TypeScript)
  - 커스텀 YAML 규칙 시스템 (`tools/rules_loader.py`)
    - Rule, RuleSet 데이터클래스
    - RulesLoader: YAML 파일/디렉토리/문자열 로딩
    - CustomRulesAnalyzer: 커스텀 규칙 기반 분석
    - 샘플 규칙 파일: security.yaml (18개), quality.yaml (15개)
  - 테스트 93개 추가 (55 → 148개, 100% 통과)

---

## [1.3.0] - 2026-01-24

### Added
- **Finance RAG 고도화 (Phase 2 완료)**
  - Redis 캐싱 기능 구현
    - CacheEntry/CacheStats: 캐시 엔트리 및 통계 데이터클래스
    - InMemoryCache: LRU 퇴거 정책, TTL 기반 만료, 스레드 안전
    - RedisCache: 분산 환경 지원, 인메모리 폴백
    - CacheService: 쿼리 결과 자동 캐싱, MD5 키 생성
    - CachedRAGService: RAG 서비스 래퍼 (자동 캐싱 적용)
    - 테스트 59개 추가 (100% 통과)
  - 멀티턴 대화 기능 구현
    - ConversationManager: 세션 기반 대화 히스토리 관리
    - ContextResolver: 대명사/참조 해결 ("그 회사" → "삼성전자")
    - MultiTurnRAGService: 컨텍스트 기반 RAG 통합 서비스
    - 테스트 47개 추가 (100% 통과)
  - Fine-tuned Embedding 기능 구현
    - TrainingDataGenerator: 동의어, 쿼리-문서, 금융 컨텍스트 데이터 생성
    - FinancialEmbeddingTrainer: LoRA 기반 효율적 학습
    - EmbeddingEvaluator: MRR, Recall@K, NDCG@K 평가 메트릭
    - 테스트 34개 추가 (100% 통과)
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
| 1.6.0 | 2026-01-25 | Phase 5 완료: Dataset Augmentor, DPO Trainer |
| 1.5.0 | 2026-01-25 | Phase 4 완료: Feast, Great Expectations, Airflow |
| 1.4.0 | 2026-01-25 | Phase 3 완료: OWASP, Java/Go, YAML 규칙 |
| 1.3.0 | 2026-01-24 | Phase 2 완료: Redis, 멀티턴, Fine-tuning |
| 1.2.0 | 2026-01-21 | Production-grade RAG, Premium UI |
| 1.1.0 | 2026-01-12 | HuggingFace Spaces, AWS ML Guide |
| 1.0.0 | 2026-01-08 | Step 1-6 완성, 189개 테스트 통과 |
| 0.2.0 | 2025-12-19 | 고급 RAG, 클라우드 배포 |
| 0.1.0 | 2025-12-18 | 프로젝트 시작 |

---

## Test Coverage

- **Total Tests**: 590+
- **Pass Rate**: 100%

| Project | Tests |
|---------|-------|
| financial-analysis | 20 |
| finance-rag-api | 225 |
| code-review-agent | 148 |
| mlops-pipeline | 124 |
| financial-finetuning | 74 |
