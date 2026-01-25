# Roadmap

AI/ML 포트폴리오 프로젝트의 개발 로드맵입니다.

---

## Current Status (v1.6.0)

| Project | Status | Tests | Coverage |
|---------|--------|-------|----------|
| financial-analysis | ✅ Complete | 20/20 | 100% |
| finance-rag-api | ✅ Complete | 59/59 | 100% |
| code-review-agent | ✅ Complete | 148/148 | 100% |
| mlops-pipeline | ✅ Complete | 124/124 | 100% |
| financial-finetuning | ✅ Complete | 74/74 | 100% |
| portfolio | ✅ Complete | - | - |

**Total: 425 tests passing**

---

## Q1 2026 (January - March)

### 🔴 Phase 1: Infrastructure (Week 1-2)

| Task | Priority | Status | Project |
|------|----------|--------|---------|
| CI/CD 파이프라인 구축 | High | ✅ Complete | All |
| GitHub Actions 테스트 자동화 | High | ✅ Complete | All |
| Docker 이미지 빌드 자동화 | High | ✅ Complete | All |
| Codecov 테스트 커버리지 연동 | Medium | ✅ Complete | All |

```yaml
# Target: .github/workflows/ci.yml
- pytest + coverage for all projects
- Docker build & push
- Auto-deploy to HuggingFace Spaces
```

### 🟠 Phase 2: Finance RAG 고도화 (Week 3-6)

| Task | Priority | Status | Details |
|------|----------|--------|---------|
| Fine-tuned Embedding | High | ✅ Complete | 금융 도메인 특화 임베딩 |
| Query Expansion | Medium | ✅ Complete | 동의어 사전 200+ 항목, HybridSearcher 통합 |
| Redis 캐싱 | Medium | ✅ Complete | LRU 캐시, Redis 폴백, TTL 기반 만료 |
| 멀티턴 대화 | Low | ✅ Complete | 대화 컨텍스트 유지 |

**Fine-tuned Embedding 상세 계획:**
```
1. 학습 데이터 준비
   - DART 공시 문서 + 쿼리 쌍 10,000+
   - 금융 용어 사전 기반 증강

2. 모델 선택
   - Base: intfloat/e5-base 또는 BAAI/bge-base
   - LoRA fine-tuning

3. 평가
   - MRR, Recall@k 메트릭
   - A/B 테스트 (기존 vs fine-tuned)
```

**Query Expansion 동의어 사전:**
```python
FINANCIAL_SYNONYMS = {
    "PER": ["주가수익비율", "P/E ratio"],
    "PBR": ["주가순자산비율", "P/B ratio"],
    "ROE": ["자기자본이익률", "Return on Equity"],
    "EBITDA": ["감가상각전영업이익"],
    "EPS": ["주당순이익"],
    # ... 200+ 항목
}
```

### ✅ Phase 3: Code Review Agent 확장 (Week 7-8)

| Task | Priority | Status | Details |
|------|----------|--------|---------|
| OWASP Top 10 보안 규칙 | Medium | ✅ Complete | 10개 카테고리 + 정적 분석기 |
| Java/Go 언어 지원 | Low | ✅ Complete | 6개 언어 패턴 + 인터페이스/구조체 추출 |
| 커스텀 규칙 YAML | Low | ✅ Complete | RulesLoader + CustomRulesAnalyzer |

**구현 완료:**
- `agents/owasp_agent.py`: OWASP Top 10 LLM + 정적 분석
- `tools/code_analyzer.py`: LanguageConfig, 6개 언어 지원
- `tools/rules_loader.py`: YAML 규칙 시스템
- `rules/security.yaml`, `rules/quality.yaml`: 샘플 규칙
- 테스트: 55 → 148개 (+93)

---

## Q2 2026 (April - June)

### ✅ Phase 4: MLOps 고도화 (Week 9-10)

| Task | Priority | Status | Details |
|------|----------|--------|---------|
| Feast Feature Store | High | ✅ Complete | 피처 중앙 관리, 로컬 폴백 지원 |
| Great Expectations | Medium | ✅ Complete | 데이터 품질 검증, 스키마/통계 검증 |
| Airflow 자동 재학습 | Medium | ✅ Complete | DAG 빌더, 파이프라인 시뮬레이션 |
| A/B 테스트 프레임워크 | Low | 🔲 Todo | 카나리 배포 |

**구현 완료:**
- `src/features/feast_store.py`: Feast 래퍼, 로컬 폴백
  - FeastFeatureStore: 피처 그룹 관리
  - 히스토리컬 피처 조회, 포인트-인-타임 조인
  - 피처 통계 및 버전 해시 생성
- `src/data/ge_expectations.py`: Great Expectations 래퍼
  - GreatExpectationsValidator: Expectation 기반 검증
  - 스키마 검증, 데이터 프로파일링
  - 사기 탐지용 사전 정의 스위트
- `src/orchestration/airflow_dags.py`: Airflow DAG 빌더
  - AirflowDAGBuilder: DAG 생성/관리
  - 사기 탐지 파이프라인, 자동 재학습 DAG
  - 로컬 시뮬레이션 모드, DAG 코드 내보내기
- 테스트: 40 → 124개 (+84)

### ✅ Phase 5: Fine-tuning 강화

| Task | Priority | Status | Details |
|------|----------|--------|---------|
| 데이터셋 확대 (1,000+) | High | ✅ Complete | LLM 기반 자동 생성 |
| DPO 학습 구현 | Medium | ✅ Complete | 선호도 최적화 |
| Ray Tune 하이퍼파라미터 | Medium | 🔲 Todo | 자동 튜닝 |
| vLLM 추론 최적화 | Low | 🔲 Todo | 고속 추론 |

**구현 완료:**
- `src/data/dataset_augmentor.py`: 데이터셋 확장기
  - DatasetAugmentor: 템플릿/수치 변형, LLM 패러프레이징
  - SyntheticDataGenerator: 카테고리별 합성 데이터 생성
  - AugmentationConfig: 목표 개수, 변형 옵션 설정
  - 20개 테스트 추가 (100% 통과)
- `src/training/dpo_trainer.py`: DPO 학습 모듈
  - FinancialDPOTrainer: TRL 기반 DPO 학습기
  - PreferenceDataset: 선호도 페어 데이터셋
  - FinancialPreferenceGenerator: 선호/비선호 페어 생성
  - DPOTrainingConfig: YAML 설정, LoRA/QLoRA 지원
  - 24개 테스트 추가 (100% 통과)
- 테스트 53개 추가 (21 → 74개, 100% 통과)

---

## Q3 2026 (July - September)

### 🟡 Phase 6: 실시간 기능

| Task | Priority | Status | Details |
|------|----------|--------|---------|
| DART API 실시간 연동 | High | 🔲 Todo | 일별 자동 업데이트 |
| WebSocket 실시간 알림 | Medium | 🔲 Todo | 공시 알림 |
| 스트리밍 응답 | Medium | 🔲 Todo | SSE 기반 |

### 🟡 Phase 7: Multi-modal

| Task | Priority | Status | Details |
|------|----------|--------|---------|
| PDF 표 추출 | Medium | 🔲 Todo | Camelot/Tabula |
| 차트 이미지 인식 | Low | 🔲 Todo | LayoutLM |
| OCR 파이프라인 | Low | 🔲 Todo | PaddleOCR |

---

## Q4 2026 (October - December)

### 🟡 Phase 8: 엔터프라이즈 기능

| Task | Priority | Status | Details |
|------|----------|--------|---------|
| 멀티테넌트 지원 | Medium | 🔲 Todo | 조직별 분리 |
| RBAC 권한 관리 | Medium | 🔲 Todo | 역할 기반 접근 |
| 감사 로깅 | Medium | 🔲 Todo | 보안 감사 추적 |
| Vault 시크릿 관리 | Low | 🔲 Todo | HashiCorp Vault |

### 🟡 Phase 9: 모니터링 고도화

| Task | Priority | Status | Details |
|------|----------|--------|---------|
| ELK Stack 로그 집계 | Medium | 🔲 Todo | 중앙 로깅 |
| Jaeger 분산 추적 | Low | 🔲 Todo | 트레이싱 |
| 모델 드리프트 감지 | Medium | 🔲 Todo | Evidently |

---

## Milestone Summary

```
2026 Q1 ─────────────────────────────────────────────────────
         │
         ├── v1.3.0: CI/CD + Fine-tuned Embedding ✅
         │
         ├── v1.4.0: Query Expansion + OWASP Rules ✅
         │
         ├── v1.5.0: Feature Store + Data Validation ✅
         │
         └── v1.6.0: DPO Training + Dataset Expansion ✅

2026 Q2 ─────────────────────────────────────────────────────
         │
         └── v1.7.0: Ray Tune + vLLM Optimization

2026 Q3 ─────────────────────────────────────────────────────
         │
         ├── v2.0.0: Real-time DART API
         │
         └── v2.1.0: Multi-modal Support

2026 Q4 ─────────────────────────────────────────────────────
         │
         ├── v2.2.0: Enterprise Features
         │
         └── v2.3.0: Advanced Monitoring
```

---

## Priority Legend

| Symbol | Priority | Description |
|--------|----------|-------------|
| 🔴 | High | 즉시 진행 필요 |
| 🟠 | Medium | 다음 분기 내 완료 |
| 🟡 | Low | 여유 있을 때 진행 |

---

## Status Legend

| Symbol | Status |
|--------|--------|
| 🔲 | Todo |
| 🔄 | In Progress |
| ✅ | Complete |
| ⏸️ | On Hold |

---

## Contributing

로드맵에 대한 제안이나 우선순위 변경 요청은 [Issues](https://github.com/your-repo/issues)에서 논의해 주세요.

---

## Related Documents

- [CHANGELOG.md](./CHANGELOG.md) - 버전별 변경 이력
- [README.md](./README.md) - 프로젝트 개요
- [docs/](./docs/) - 기술 문서
