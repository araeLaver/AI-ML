# -*- coding: utf-8 -*-
"""
능동 학습 (Active Learning) 모듈

[기능]
- 불확실성 샘플링
- 다양성 샘플링
- 쿼리 전략
- 라벨링 큐 관리
"""

import hashlib
import logging
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SamplingStrategy(Enum):
    """샘플링 전략"""
    UNCERTAINTY = "uncertainty"  # 불확실성 기반
    DIVERSITY = "diversity"  # 다양성 기반
    HYBRID = "hybrid"  # 혼합
    RANDOM = "random"  # 랜덤


@dataclass
class UnlabeledSample:
    """레이블이 없는 샘플"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    response: str = ""
    confidence: float = 0.0  # 모델 신뢰도
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "response": self.response[:200],
            "confidence": round(self.confidence, 3),
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass
class LabeledSample:
    """레이블된 샘플"""
    sample: UnlabeledSample
    label: str  # 정답 레이블 또는 수정된 응답
    labeler_id: Optional[str] = None
    labeled_at: float = field(default_factory=time.time)
    quality_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample": self.sample.to_dict(),
            "label": self.label[:200],
            "labeler_id": self.labeler_id,
            "labeled_at": self.labeled_at,
            "quality_score": self.quality_score,
        }


class QueryStrategy(ABC):
    """쿼리 전략 인터페이스"""

    @abstractmethod
    def select(
        self,
        samples: List[UnlabeledSample],
        n: int,
    ) -> List[UnlabeledSample]:
        """샘플 선택"""
        pass


class UncertaintySampler(QueryStrategy):
    """
    불확실성 샘플링

    모델이 불확실한 샘플을 우선 선택
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def select(
        self,
        samples: List[UnlabeledSample],
        n: int,
    ) -> List[UnlabeledSample]:
        """불확실한 샘플 선택"""
        # 신뢰도가 낮은 순으로 정렬
        sorted_samples = sorted(samples, key=lambda x: x.confidence)

        # 임계값 이하만 선택
        uncertain = [s for s in sorted_samples if s.confidence < self.threshold]

        return uncertain[:n]

    def score(self, sample: UnlabeledSample) -> float:
        """불확실성 점수 (높을수록 불확실)"""
        # 1 - 신뢰도 = 불확실성
        return 1.0 - sample.confidence


class DiversitySampler(QueryStrategy):
    """
    다양성 샘플링

    다양한 샘플을 선택하여 데이터 편향 방지
    """

    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold

    def select(
        self,
        samples: List[UnlabeledSample],
        n: int,
    ) -> List[UnlabeledSample]:
        """다양한 샘플 선택 (Greedy 다양성)"""
        if not samples:
            return []

        selected: List[UnlabeledSample] = []

        # 첫 번째 샘플은 랜덤 선택
        remaining = list(samples)
        first = random.choice(remaining)
        selected.append(first)
        remaining.remove(first)

        # 나머지는 기존 선택과 가장 다른 샘플 선택
        while len(selected) < n and remaining:
            best_sample = None
            best_min_sim = float("inf")

            for sample in remaining:
                # 기존 선택된 샘플들과의 최소 거리
                min_sim = min(
                    self._similarity(sample, s) for s in selected
                )

                if min_sim < best_min_sim:
                    best_min_sim = min_sim
                    best_sample = sample

            if best_sample:
                selected.append(best_sample)
                remaining.remove(best_sample)
            else:
                break

        return selected

    def _similarity(
        self,
        sample1: UnlabeledSample,
        sample2: UnlabeledSample,
    ) -> float:
        """샘플 간 유사도"""
        if sample1.embedding and sample2.embedding:
            return self._cosine_similarity(sample1.embedding, sample2.embedding)

        # 텍스트 기반 유사도 (Jaccard)
        words1 = set(sample1.query.lower().split())
        words2 = set(sample2.query.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float],
    ) -> float:
        """코사인 유사도"""
        if len(vec1) != len(vec2):
            return 0.0

        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)


class HybridSampler(QueryStrategy):
    """
    하이브리드 샘플링

    불확실성 + 다양성 결합
    """

    def __init__(
        self,
        uncertainty_weight: float = 0.6,
        diversity_weight: float = 0.4,
    ):
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        self.uncertainty_sampler = UncertaintySampler()
        self.diversity_sampler = DiversitySampler()

    def select(
        self,
        samples: List[UnlabeledSample],
        n: int,
    ) -> List[UnlabeledSample]:
        """하이브리드 샘플 선택"""
        n_uncertainty = int(n * self.uncertainty_weight)
        n_diversity = n - n_uncertainty

        # 불확실성 기반 선택
        uncertain = self.uncertainty_sampler.select(samples, n_uncertainty)

        # 나머지에서 다양성 기반 선택
        remaining = [s for s in samples if s not in uncertain]
        diverse = self.diversity_sampler.select(remaining, n_diversity)

        return uncertain + diverse


class ActiveLearner:
    """
    능동 학습기

    레이블링 효율 최대화를 위한 샘플 선택 관리
    """

    def __init__(
        self,
        strategy: SamplingStrategy = SamplingStrategy.HYBRID,
        batch_size: int = 10,
    ):
        self.strategy = strategy
        self.batch_size = batch_size

        # 샘플러 초기화
        if strategy == SamplingStrategy.UNCERTAINTY:
            self.sampler = UncertaintySampler()
        elif strategy == SamplingStrategy.DIVERSITY:
            self.sampler = DiversitySampler()
        elif strategy == SamplingStrategy.HYBRID:
            self.sampler = HybridSampler()
        else:
            self.sampler = None

        # 샘플 풀
        self._unlabeled_pool: List[UnlabeledSample] = []
        self._labeled_samples: List[LabeledSample] = []
        self._labeling_queue: List[UnlabeledSample] = []

    def add_samples(self, samples: List[UnlabeledSample]) -> int:
        """레이블 없는 샘플 추가"""
        added = 0
        for sample in samples:
            if not self._is_duplicate(sample):
                self._unlabeled_pool.append(sample)
                added += 1
        return added

    def add_sample(
        self,
        query: str,
        response: str,
        confidence: float,
        embedding: Optional[List[float]] = None,
        **metadata,
    ) -> UnlabeledSample:
        """단일 샘플 추가"""
        sample = UnlabeledSample(
            query=query,
            response=response,
            confidence=confidence,
            embedding=embedding,
            metadata=metadata,
        )
        self._unlabeled_pool.append(sample)
        return sample

    def _is_duplicate(self, sample: UnlabeledSample) -> bool:
        """중복 확인"""
        query_hash = hashlib.md5(sample.query.encode()).hexdigest()
        for existing in self._unlabeled_pool + [ls.sample for ls in self._labeled_samples]:
            existing_hash = hashlib.md5(existing.query.encode()).hexdigest()
            if query_hash == existing_hash:
                return True
        return False

    def select_for_labeling(self, n: Optional[int] = None) -> List[UnlabeledSample]:
        """레이블링할 샘플 선택"""
        n = n or self.batch_size

        if self.sampler:
            selected = self.sampler.select(self._unlabeled_pool, n)
        else:
            # 랜덤 선택
            selected = random.sample(
                self._unlabeled_pool,
                min(n, len(self._unlabeled_pool)),
            )

        # 큐에 추가
        for sample in selected:
            if sample not in self._labeling_queue:
                self._labeling_queue.append(sample)

        return selected

    def get_next_for_labeling(self) -> Optional[UnlabeledSample]:
        """다음 레이블링 대상"""
        if not self._labeling_queue:
            self.select_for_labeling()

        if self._labeling_queue:
            return self._labeling_queue[0]
        return None

    def submit_label(
        self,
        sample_id: str,
        label: str,
        labeler_id: Optional[str] = None,
        quality_score: Optional[float] = None,
    ) -> bool:
        """레이블 제출"""
        # 샘플 찾기
        sample = None
        for s in self._labeling_queue:
            if s.id == sample_id:
                sample = s
                break

        if not sample:
            for s in self._unlabeled_pool:
                if s.id == sample_id:
                    sample = s
                    break

        if not sample:
            return False

        # 레이블된 샘플로 이동
        labeled = LabeledSample(
            sample=sample,
            label=label,
            labeler_id=labeler_id,
            quality_score=quality_score,
        )
        self._labeled_samples.append(labeled)

        # 풀에서 제거
        if sample in self._unlabeled_pool:
            self._unlabeled_pool.remove(sample)
        if sample in self._labeling_queue:
            self._labeling_queue.remove(sample)

        logger.info(f"Labeled sample {sample_id}")
        return True

    def skip_sample(self, sample_id: str) -> bool:
        """샘플 건너뛰기"""
        for sample in self._labeling_queue:
            if sample.id == sample_id:
                self._labeling_queue.remove(sample)
                # 뒤로 이동
                self._labeling_queue.append(sample)
                return True
        return False

    def get_training_data(self) -> List[Dict[str, Any]]:
        """학습 데이터 생성"""
        return [
            {
                "query": ls.sample.query,
                "response": ls.label,  # 레이블된 응답
                "original_response": ls.sample.response,
                "confidence": ls.sample.confidence,
                "quality_score": ls.quality_score,
            }
            for ls in self._labeled_samples
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """통계"""
        unlabeled_confidences = [s.confidence for s in self._unlabeled_pool]
        avg_confidence = (
            sum(unlabeled_confidences) / len(unlabeled_confidences)
            if unlabeled_confidences else 0.0
        )

        return {
            "unlabeled_pool_size": len(self._unlabeled_pool),
            "labeling_queue_size": len(self._labeling_queue),
            "labeled_count": len(self._labeled_samples),
            "avg_unlabeled_confidence": round(avg_confidence, 3),
            "strategy": self.strategy.value,
        }

    def export_labeled(self, filepath: str) -> int:
        """레이블된 데이터 내보내기"""
        import json

        data = [ls.to_dict() for ls in self._labeled_samples]

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return len(data)

    def get_labeling_progress(self) -> Dict[str, Any]:
        """레이블링 진행 상황"""
        total = len(self._unlabeled_pool) + len(self._labeled_samples)
        labeled = len(self._labeled_samples)

        return {
            "total_samples": total,
            "labeled": labeled,
            "unlabeled": len(self._unlabeled_pool),
            "in_queue": len(self._labeling_queue),
            "progress": labeled / total if total > 0 else 0.0,
        }

    def prioritize_low_confidence(self, threshold: float = 0.3) -> int:
        """낮은 신뢰도 샘플 우선순위 높이기"""
        low_conf = [s for s in self._unlabeled_pool if s.confidence < threshold]

        # 큐 앞에 추가
        for sample in low_conf:
            if sample not in self._labeling_queue:
                self._labeling_queue.insert(0, sample)

        return len(low_conf)
