# -*- coding: utf-8 -*-
"""
자동 학습 개선 모듈

[기능]
- 피드백 기반 재훈련 트리거
- 학습 데이터 선택
- 모델 업데이트
- 점진적 학습
"""

import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .collector import FeedbackData, FeedbackSentiment, FeedbackType
from .storage import FeedbackQuery, FeedbackStorage
from .analyzer import FeedbackAnalyzer, QualityMetrics

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """재훈련 트리거 유형"""
    THRESHOLD = "threshold"  # 임계값 기반
    SCHEDULED = "scheduled"  # 스케줄 기반
    MANUAL = "manual"  # 수동
    DRIFT = "drift"  # 드리프트 감지


@dataclass
class TrainingJob:
    """훈련 작업"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trigger_type: TriggerType = TriggerType.THRESHOLD
    status: str = "pending"  # pending, running, completed, failed
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    data_count: int = 0
    metrics_before: Optional[Dict[str, Any]] = None
    metrics_after: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "trigger_type": self.trigger_type.value,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "data_count": self.data_count,
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
            "error": self.error,
        }


class RetrainTrigger:
    """
    재훈련 트리거

    자동 재훈련 조건 판단
    """

    def __init__(
        self,
        storage: FeedbackStorage,
        satisfaction_threshold: float = 0.7,
        min_feedback_count: int = 100,
        check_interval_hours: int = 24,
    ):
        self.storage = storage
        self.analyzer = FeedbackAnalyzer(storage)
        self.satisfaction_threshold = satisfaction_threshold
        self.min_feedback_count = min_feedback_count
        self.check_interval_hours = check_interval_hours
        self._last_check: Optional[float] = None
        self._last_metrics: Optional[QualityMetrics] = None

    def should_retrain(self) -> Tuple[bool, Optional[str]]:
        """재훈련 필요 여부 판단"""
        # 최소 피드백 수 확인
        total_count = self.storage.count()
        if total_count < self.min_feedback_count:
            return False, f"피드백 수 부족: {total_count}/{self.min_feedback_count}"

        # 최근 메트릭 계산
        now = time.time()
        metrics = self.analyzer.calculate_metrics(
            start_time=now - 86400 * 7,  # 최근 7일
            end_time=now,
        )

        # 만족도 임계값 확인
        if metrics.satisfaction_rate < self.satisfaction_threshold:
            return True, f"만족도 저하: {metrics.satisfaction_rate:.1%} < {self.satisfaction_threshold:.1%}"

        # 급격한 품질 저하 감지
        if self._last_metrics:
            satisfaction_drop = self._last_metrics.satisfaction_rate - metrics.satisfaction_rate
            if satisfaction_drop > 0.1:  # 10% 이상 저하
                return True, f"급격한 만족도 저하: {satisfaction_drop:.1%}"

        self._last_metrics = metrics
        self._last_check = now

        return False, None

    def check_drift(self) -> Dict[str, Any]:
        """데이터 드리프트 감지"""
        now = time.time()

        # 최근 기간과 이전 기간 비교
        recent_metrics = self.analyzer.calculate_metrics(
            start_time=now - 86400 * 7,
            end_time=now,
        )
        previous_metrics = self.analyzer.calculate_metrics(
            start_time=now - 86400 * 14,
            end_time=now - 86400 * 7,
        )

        drift_detected = False
        drift_details = {}

        # 만족도 변화
        if previous_metrics.satisfaction_rate > 0:
            satisfaction_change = (
                recent_metrics.satisfaction_rate - previous_metrics.satisfaction_rate
            ) / previous_metrics.satisfaction_rate

            drift_details["satisfaction_change"] = satisfaction_change
            if abs(satisfaction_change) > 0.15:  # 15% 이상 변화
                drift_detected = True

        # 부정 피드백 비율 변화
        if previous_metrics.total_feedback > 0:
            prev_neg_rate = previous_metrics.negative_count / previous_metrics.total_feedback
            curr_neg_rate = recent_metrics.negative_count / recent_metrics.total_feedback if recent_metrics.total_feedback > 0 else 0

            neg_change = curr_neg_rate - prev_neg_rate
            drift_details["negative_rate_change"] = neg_change
            if neg_change > 0.1:  # 부정 비율 10% 이상 증가
                drift_detected = True

        return {
            "drift_detected": drift_detected,
            "recent_metrics": recent_metrics.to_dict(),
            "previous_metrics": previous_metrics.to_dict(),
            "details": drift_details,
        }


class DataSelector:
    """
    학습 데이터 선택기

    피드백 기반 학습 데이터 선택
    """

    def __init__(self, storage: FeedbackStorage):
        self.storage = storage

    def select_training_data(
        self,
        max_samples: int = 1000,
        include_corrections: bool = True,
        include_positive: bool = True,
        include_negative: bool = True,
    ) -> List[Dict[str, Any]]:
        """학습 데이터 선택"""
        training_data = []

        # 1. 수정 제안 (가장 높은 품질)
        if include_corrections:
            corrections = self.storage.query(FeedbackQuery(
                feedback_types=[FeedbackType.CORRECTION],
                limit=max_samples // 3,
            ))
            for fb in corrections:
                training_data.append({
                    "type": "correction",
                    "query": fb.query,
                    "response": fb.value,  # 수정된 응답
                    "original_response": fb.response,
                    "priority": 1.0,
                })

        # 2. 긍정 피드백 (확인된 좋은 응답)
        if include_positive:
            positive = self.storage.query(FeedbackQuery(
                sentiments=[FeedbackSentiment.POSITIVE],
                feedback_types=[FeedbackType.THUMBS_UP, FeedbackType.RATING],
                limit=max_samples // 3,
            ))
            for fb in positive:
                if fb.feedback_type == FeedbackType.RATING and fb.value and fb.value < 4:
                    continue  # 4점 미만 제외

                training_data.append({
                    "type": "positive",
                    "query": fb.query,
                    "response": fb.response,
                    "priority": 0.8,
                })

        # 3. 부정 피드백 (피해야 할 응답)
        if include_negative:
            negative = self.storage.query(FeedbackQuery(
                sentiments=[FeedbackSentiment.NEGATIVE],
                limit=max_samples // 3,
            ))
            for fb in negative:
                training_data.append({
                    "type": "negative",
                    "query": fb.query,
                    "response": fb.response,
                    "priority": 0.5,
                    "avoid": True,  # 이 응답은 피해야 함
                })

        # 중복 제거 (쿼리 기반)
        seen_queries = set()
        unique_data = []
        for item in training_data:
            query_hash = hashlib.md5(item["query"].encode()).hexdigest()
            if query_hash not in seen_queries:
                seen_queries.add(query_hash)
                unique_data.append(item)

        # 우선순위로 정렬
        unique_data.sort(key=lambda x: x["priority"], reverse=True)

        return unique_data[:max_samples]

    def select_hard_examples(self, limit: int = 100) -> List[Dict[str, Any]]:
        """어려운 예제 선택 (부정 피드백이 많은)"""
        negative = self.storage.query(FeedbackQuery(
            sentiments=[FeedbackSentiment.NEGATIVE],
            limit=10000,
        ))

        # 쿼리별 부정 피드백 수 계산
        query_negative_counts: Dict[str, List[FeedbackData]] = {}
        for fb in negative:
            if fb.query not in query_negative_counts:
                query_negative_counts[fb.query] = []
            query_negative_counts[fb.query].append(fb)

        # 부정 피드백이 많은 쿼리 선택
        hard_examples = []
        for query, feedbacks in sorted(
            query_negative_counts.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )[:limit]:
            hard_examples.append({
                "query": query,
                "negative_count": len(feedbacks),
                "responses": [fb.response for fb in feedbacks[:3]],
                "reasons": [fb.metadata.get("reason") for fb in feedbacks if fb.metadata.get("reason")],
            })

        return hard_examples


class ModelUpdater(ABC):
    """모델 업데이터 인터페이스"""

    @abstractmethod
    def update(self, training_data: List[Dict[str, Any]]) -> bool:
        """모델 업데이트"""
        pass

    @abstractmethod
    def get_current_version(self) -> str:
        """현재 모델 버전"""
        pass

    @abstractmethod
    def rollback(self, version: str) -> bool:
        """이전 버전으로 롤백"""
        pass


class DummyModelUpdater(ModelUpdater):
    """더미 모델 업데이터 (테스트용)"""

    def __init__(self):
        self._version = "1.0.0"
        self._history: List[str] = []

    def update(self, training_data: List[Dict[str, Any]]) -> bool:
        logger.info(f"Updating model with {len(training_data)} samples")
        self._history.append(self._version)
        parts = self._version.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        self._version = ".".join(parts)
        return True

    def get_current_version(self) -> str:
        return self._version

    def rollback(self, version: str) -> bool:
        if version in self._history:
            self._version = version
            return True
        return False


class IncrementalLearner:
    """
    점진적 학습기

    온라인/증분 학습 지원
    """

    def __init__(
        self,
        storage: FeedbackStorage,
        model_updater: Optional[ModelUpdater] = None,
        batch_size: int = 100,
        update_interval_hours: int = 24,
    ):
        self.storage = storage
        self.model_updater = model_updater or DummyModelUpdater()
        self.data_selector = DataSelector(storage)
        self.batch_size = batch_size
        self.update_interval_hours = update_interval_hours
        self._last_update_time: Optional[float] = None
        self._processed_ids: set = set()

    def process_new_feedback(self) -> Optional[TrainingJob]:
        """새 피드백 처리 및 학습"""
        # 마지막 업데이트 이후 새 피드백 조회
        start_time = self._last_update_time or (time.time() - 86400 * 7)

        new_feedback = self.storage.query(FeedbackQuery(
            start_time=start_time,
            limit=10000,
        ))

        # 이미 처리된 피드백 제외
        unprocessed = [fb for fb in new_feedback if fb.id not in self._processed_ids]

        if len(unprocessed) < self.batch_size:
            return None

        # 훈련 데이터 준비
        training_data = self.data_selector.select_training_data(
            max_samples=len(unprocessed),
        )

        if not training_data:
            return None

        # 훈련 작업 생성
        job = TrainingJob(
            trigger_type=TriggerType.THRESHOLD,
            data_count=len(training_data),
        )

        try:
            job.status = "running"
            job.started_at = time.time()

            # 모델 업데이트
            success = self.model_updater.update(training_data)

            if success:
                job.status = "completed"
                job.completed_at = time.time()

                # 처리된 피드백 ID 기록
                for fb in unprocessed:
                    self._processed_ids.add(fb.id)

                self._last_update_time = time.time()
            else:
                job.status = "failed"
                job.error = "Model update failed"

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error(f"Incremental learning failed: {e}")

        return job


class LearningPipeline:
    """
    학습 파이프라인

    전체 피드백 기반 학습 프로세스 관리
    """

    def __init__(
        self,
        storage: FeedbackStorage,
        model_updater: Optional[ModelUpdater] = None,
        auto_retrain: bool = True,
        retrain_threshold: float = 0.7,
    ):
        self.storage = storage
        self.model_updater = model_updater or DummyModelUpdater()
        self.analyzer = FeedbackAnalyzer(storage)
        self.trigger = RetrainTrigger(
            storage,
            satisfaction_threshold=retrain_threshold,
        )
        self.data_selector = DataSelector(storage)
        self.auto_retrain = auto_retrain
        self._jobs: List[TrainingJob] = []

    def check_and_retrain(self) -> Optional[TrainingJob]:
        """재훈련 조건 확인 및 실행"""
        if not self.auto_retrain:
            return None

        should_retrain, reason = self.trigger.should_retrain()

        if not should_retrain:
            logger.debug(f"Retrain not needed: {reason}")
            return None

        logger.info(f"Triggering retrain: {reason}")
        return self.run_training_job(TriggerType.THRESHOLD)

    def run_training_job(
        self,
        trigger_type: TriggerType = TriggerType.MANUAL,
        max_samples: int = 1000,
    ) -> TrainingJob:
        """훈련 작업 실행"""
        job = TrainingJob(trigger_type=trigger_type)

        try:
            # 현재 메트릭 기록
            job.metrics_before = self.analyzer.calculate_metrics().to_dict()

            # 훈련 데이터 선택
            training_data = self.data_selector.select_training_data(max_samples)
            job.data_count = len(training_data)

            if not training_data:
                job.status = "failed"
                job.error = "No training data available"
                return job

            # 모델 업데이트
            job.status = "running"
            job.started_at = time.time()

            success = self.model_updater.update(training_data)

            if success:
                job.status = "completed"
                job.completed_at = time.time()
                # 훈련 후 메트릭은 시간이 지나야 측정 가능
            else:
                job.status = "failed"
                job.error = "Model update failed"

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error(f"Training job failed: {e}")

        self._jobs.append(job)
        return job

    def get_job_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """훈련 작업 히스토리"""
        return [job.to_dict() for job in self._jobs[-limit:]]

    def get_pipeline_status(self) -> Dict[str, Any]:
        """파이프라인 상태"""
        should_retrain, reason = self.trigger.should_retrain()

        return {
            "auto_retrain_enabled": self.auto_retrain,
            "model_version": self.model_updater.get_current_version(),
            "should_retrain": should_retrain,
            "retrain_reason": reason,
            "current_metrics": self.analyzer.calculate_metrics().to_dict(),
            "total_jobs": len(self._jobs),
            "recent_jobs": self.get_job_history(5),
        }
