# TODO List

AI/ML 포트폴리오 신규 기능 구현 목록입니다.

---

## 우선순위 높음 (High Priority)

### Phase 5 완료: Fine-tuning 강화

- [x] **Ray Tune 하이퍼파라미터 최적화**
  - 프로젝트: `financial-finetuning`
  - 설명: 분산 하이퍼파라미터 튜닝 자동화
  - 기술: Ray Tune, Optuna 통합
  - 예상 파일: `src/training/hyperparameter_tuner.py`

- [x] **vLLM 추론 최적화**
  - 프로젝트: `financial-finetuning`
  - 설명: 고속 LLM 추론 서버 구축
  - 기술: vLLM, PagedAttention
  - 예상 파일: `src/inference/vllm_server.py`

### Phase 9: 모니터링 고도화

- [ ] **Evidently 모델 드리프트 감지**
  - 프로젝트: `mlops-pipeline`
  - 설명: 데이터/모델 드리프트 자동 감지
  - 기술: Evidently AI, PSI, KS Test
  - 예상 파일: `src/monitoring/evidently_monitor.py`

---

## 우선순위 중간 (Medium Priority)

### Phase 6: 실시간 기능

- [ ] **스트리밍 응답 (SSE)**
  - 프로젝트: `finance-rag-api`
  - 설명: LLM 응답 실시간 스트리밍
  - 기술: Server-Sent Events, FastAPI StreamingResponse
  - 예상 파일: `src/api/streaming.py`

- [ ] **DART API 실시간 연동**
  - 프로젝트: `finance-rag-api`
  - 설명: 일별 공시 자동 수집 및 인덱싱
  - 기술: APScheduler, DART OpenAPI
  - 예상 파일: `src/data/dart_realtime.py`

- [ ] **WebSocket 실시간 알림**
  - 프로젝트: `finance-rag-api`
  - 설명: 신규 공시 실시간 푸시 알림
  - 기술: FastAPI WebSocket
  - 예상 파일: `src/api/websocket.py`

### Phase 7: Multi-modal

- [ ] **PDF 표 추출**
  - 프로젝트: `finance-rag-api`
  - 설명: 재무제표 PDF에서 표 데이터 추출
  - 기술: Camelot, Tabula-py
  - 예상 파일: `src/data/pdf_extractor.py`

### Phase 4: MLOps 완료

- [ ] **A/B 테스트 프레임워크**
  - 프로젝트: `mlops-pipeline`
  - 설명: 카나리 배포 및 모델 비교
  - 기술: Feature Flags, 통계적 유의성 검정
  - 예상 파일: `src/serving/ab_testing.py`

---

## 우선순위 낮음 (Low Priority)

### Phase 7: Multi-modal 확장

- [ ] **차트 이미지 인식**
  - 프로젝트: `finance-rag-api`
  - 설명: 금융 차트 이미지 분석
  - 기술: LayoutLM, DocVQA
  - 예상 파일: `src/multimodal/chart_analyzer.py`

- [ ] **OCR 파이프라인**
  - 프로젝트: `finance-rag-api`
  - 설명: 스캔 문서 텍스트 추출
  - 기술: PaddleOCR, EasyOCR
  - 예상 파일: `src/multimodal/ocr_pipeline.py`

### Phase 8: 엔터프라이즈 기능

- [ ] **멀티테넌트 지원**
  - 프로젝트: `finance-rag-api`
  - 설명: 조직별 데이터 분리
  - 기술: PostgreSQL Row-Level Security
  - 예상 파일: `src/auth/multi_tenant.py`

- [ ] **RBAC 권한 관리**
  - 프로젝트: `finance-rag-api`
  - 설명: 역할 기반 접근 제어
  - 기술: Casbin, JWT
  - 예상 파일: `src/auth/rbac.py`

- [ ] **감사 로깅**
  - 프로젝트: All
  - 설명: 보안 감사 추적
  - 기술: Structured Logging, Audit Trail
  - 예상 파일: `src/logging/audit.py`

- [ ] **Vault 시크릿 관리**
  - 프로젝트: All
  - 설명: 중앙 시크릿 관리
  - 기술: HashiCorp Vault
  - 예상 파일: `src/config/vault_client.py`

### Phase 9: 모니터링 확장

- [ ] **ELK Stack 로그 집계**
  - 프로젝트: All
  - 설명: 중앙 로그 관리
  - 기술: Elasticsearch, Logstash, Kibana
  - 예상 파일: `docker-compose.elk.yml`

- [ ] **Jaeger 분산 추적**
  - 프로젝트: All
  - 설명: 마이크로서비스 트레이싱
  - 기술: Jaeger, OpenTelemetry
  - 예상 파일: `src/tracing/jaeger_config.py`

---

## 완료된 항목

### Phase 1-5 (v1.0.0 ~ v1.6.0)

- [x] CI/CD 파이프라인 구축
- [x] GitHub Actions 테스트 자동화
- [x] Docker 이미지 빌드 자동화
- [x] Fine-tuned Embedding (금융 도메인)
- [x] Query Expansion (동의어 사전 200+)
- [x] Redis 캐싱
- [x] 멀티턴 대화
- [x] OWASP Top 10 보안 규칙
- [x] Java/Go 언어 지원 (6개 언어)
- [x] 커스텀 규칙 YAML
- [x] Feast Feature Store
- [x] Great Expectations 데이터 검증
- [x] Airflow 자동 재학습 DAG
- [x] DPO 학습 구현
- [x] 데이터셋 확대 (1,000+ 샘플)

---

## 진행 상황

| 카테고리 | 완료 | 진행중 | 대기 | 총계 |
|:---|:---:|:---:|:---:|:---:|
| High Priority | 2 | 0 | 1 | 3 |
| Medium Priority | 0 | 0 | 5 | 5 |
| Low Priority | 0 | 0 | 8 | 8 |
| **Total** | **2** | **0** | **14** | **16** |

---

## 다음 단계

1. ~~**Ray Tune 하이퍼파라미터 최적화** 구현~~ (완료)
2. ~~**vLLM 추론 최적화** 구현~~ (완료)
3. **Evidently 드리프트 감지** 구현

---

*마지막 업데이트: 2026-02-11*
