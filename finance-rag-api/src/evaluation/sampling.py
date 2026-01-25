# -*- coding: utf-8 -*-
"""
평가 샘플링 모듈

[기능]
- 계층화 샘플링
- 다양성 샘플링
- 시간 기반 샘플링
"""

import logging
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

from .ragas import EvaluationSample

logger = logging.getLogger(__name__)

T = TypeVar("T")


class EvaluationSampler(ABC):
    """샘플러 기본 클래스"""

    @abstractmethod
    def sample(
        self,
        data: List[EvaluationSample],
        n: int,
    ) -> List[EvaluationSample]:
        """샘플 선택"""
        pass


class RandomSampler(EvaluationSampler):
    """무작위 샘플러"""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def sample(
        self,
        data: List[EvaluationSample],
        n: int,
    ) -> List[EvaluationSample]:
        """무작위 샘플링"""
        if n >= len(data):
            return data[:]

        return self._rng.sample(data, n)


class StratifiedSampler(EvaluationSampler):
    """
    계층화 샘플러

    특정 속성별로 균등하게 샘플링
    """

    def __init__(
        self,
        stratify_by: Callable[[EvaluationSample], str],
        seed: Optional[int] = None,
    ):
        self.stratify_by = stratify_by
        self._rng = random.Random(seed)

    def sample(
        self,
        data: List[EvaluationSample],
        n: int,
    ) -> List[EvaluationSample]:
        """계층화 샘플링"""
        if n >= len(data):
            return data[:]

        # 계층별 그룹화
        strata: Dict[str, List[EvaluationSample]] = defaultdict(list)
        for sample in data:
            key = self.stratify_by(sample)
            strata[key].append(sample)

        # 각 계층에서 비례적으로 샘플링
        result = []
        n_strata = len(strata)

        if n_strata == 0:
            return []

        # 기본 할당: 각 계층에 최소 1개
        base_per_stratum = max(1, n // n_strata)
        remaining = n

        for key, samples in strata.items():
            take = min(base_per_stratum, len(samples), remaining)
            result.extend(self._rng.sample(samples, take))
            remaining -= take

            if remaining <= 0:
                break

        # 남은 개수가 있으면 추가 샘플링
        if remaining > 0:
            all_remaining = []
            for samples in strata.values():
                selected_ids = {s.id for s in result}
                remaining_samples = [s for s in samples if s.id not in selected_ids]
                all_remaining.extend(remaining_samples)

            if all_remaining:
                additional = self._rng.sample(
                    all_remaining,
                    min(remaining, len(all_remaining))
                )
                result.extend(additional)

        return result

    @staticmethod
    def by_question_type(sample: EvaluationSample) -> str:
        """질문 유형별 계층화"""
        question = sample.question.lower()

        if any(kw in question for kw in ["what", "무엇", "뭐"]):
            return "what"
        elif any(kw in question for kw in ["why", "왜"]):
            return "why"
        elif any(kw in question for kw in ["how", "어떻게"]):
            return "how"
        elif any(kw in question for kw in ["when", "언제"]):
            return "when"
        elif any(kw in question for kw in ["who", "누구"]):
            return "who"
        elif any(kw in question for kw in ["비교", "compare", "vs"]):
            return "comparison"
        else:
            return "other"

    @staticmethod
    def by_answer_length(sample: EvaluationSample) -> str:
        """답변 길이별 계층화"""
        length = len(sample.answer)

        if length < 100:
            return "short"
        elif length < 500:
            return "medium"
        else:
            return "long"


class DiversitySampler(EvaluationSampler):
    """
    다양성 샘플러

    최대한 다양한 샘플 선택
    """

    def __init__(
        self,
        similarity_func: Optional[Callable[[EvaluationSample, EvaluationSample], float]] = None,
        seed: Optional[int] = None,
    ):
        self.similarity_func = similarity_func or self._default_similarity
        self._rng = random.Random(seed)

    def sample(
        self,
        data: List[EvaluationSample],
        n: int,
    ) -> List[EvaluationSample]:
        """다양성 최대화 샘플링 (Maximal Marginal Relevance 스타일)"""
        if n >= len(data):
            return data[:]

        if not data:
            return []

        # 첫 샘플: 무작위 선택
        selected = [self._rng.choice(data)]
        remaining = [s for s in data if s.id != selected[0].id]

        # 나머지: 선택된 것과 가장 다른 것 선택
        while len(selected) < n and remaining:
            best_sample = None
            best_diversity = -1

            for candidate in remaining:
                # 기존 선택된 것들과의 최대 유사도 (낮을수록 좋음)
                max_sim = max(
                    self.similarity_func(candidate, s)
                    for s in selected
                )
                diversity = 1 - max_sim

                if diversity > best_diversity:
                    best_diversity = diversity
                    best_sample = candidate

            if best_sample:
                selected.append(best_sample)
                remaining = [s for s in remaining if s.id != best_sample.id]

        return selected

    def _default_similarity(
        self,
        s1: EvaluationSample,
        s2: EvaluationSample,
    ) -> float:
        """기본 유사도: 자카드 유사도"""
        words1 = set(s1.question.lower().split())
        words2 = set(s2.question.lower().split())

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0


class TemporalSampler(EvaluationSampler):
    """
    시간 기반 샘플러

    특정 시간 범위 또는 패턴에 따라 샘플링
    """

    def __init__(
        self,
        timestamp_func: Callable[[EvaluationSample], float],
        seed: Optional[int] = None,
    ):
        self.timestamp_func = timestamp_func
        self._rng = random.Random(seed)

    def sample(
        self,
        data: List[EvaluationSample],
        n: int,
        recent_bias: float = 0.5,  # 최근 데이터 비율
    ) -> List[EvaluationSample]:
        """시간 기반 샘플링"""
        if n >= len(data):
            return data[:]

        # 시간순 정렬
        sorted_data = sorted(data, key=self.timestamp_func, reverse=True)

        # 최근 절반과 과거 절반으로 분리
        mid_point = len(sorted_data) // 2
        recent = sorted_data[:mid_point]
        older = sorted_data[mid_point:]

        # 비율에 따라 샘플링
        n_recent = int(n * recent_bias)
        n_older = n - n_recent

        result = []

        if recent and n_recent > 0:
            result.extend(self._rng.sample(recent, min(n_recent, len(recent))))

        if older and n_older > 0:
            result.extend(self._rng.sample(older, min(n_older, len(older))))

        return result

    def sample_by_interval(
        self,
        data: List[EvaluationSample],
        n: int,
    ) -> List[EvaluationSample]:
        """시간 간격으로 균등 샘플링"""
        if n >= len(data):
            return data[:]

        sorted_data = sorted(data, key=self.timestamp_func)

        if n == 1:
            return [sorted_data[len(sorted_data) // 2]]

        # 균등한 간격으로 선택
        step = len(sorted_data) / (n - 1) if n > 1 else 1
        indices = [int(i * step) for i in range(n)]
        indices = [min(i, len(sorted_data) - 1) for i in indices]

        return [sorted_data[i] for i in indices]


class PrioritySampler(EvaluationSampler):
    """
    우선순위 기반 샘플러

    특정 조건의 샘플을 우선 선택
    """

    def __init__(
        self,
        priority_func: Callable[[EvaluationSample], float],
        seed: Optional[int] = None,
    ):
        self.priority_func = priority_func
        self._rng = random.Random(seed)

    def sample(
        self,
        data: List[EvaluationSample],
        n: int,
        top_priority_ratio: float = 0.5,
    ) -> List[EvaluationSample]:
        """우선순위 기반 샘플링"""
        if n >= len(data):
            return data[:]

        # 우선순위순 정렬
        sorted_data = sorted(
            data,
            key=self.priority_func,
            reverse=True
        )

        # 상위 우선순위와 나머지 분리
        n_top = int(n * top_priority_ratio)
        n_rest = n - n_top

        top_candidates = sorted_data[:max(n_top * 2, n)]
        rest_candidates = sorted_data[len(top_candidates):]

        result = []

        # 상위 우선순위에서 선택
        if top_candidates and n_top > 0:
            result.extend(
                self._rng.sample(top_candidates, min(n_top, len(top_candidates)))
            )

        # 나머지에서 무작위 선택
        if rest_candidates and n_rest > 0:
            result.extend(
                self._rng.sample(rest_candidates, min(n_rest, len(rest_candidates)))
            )

        return result


class ComposedSampler(EvaluationSampler):
    """
    복합 샘플러

    여러 샘플러를 조합
    """

    def __init__(
        self,
        samplers: List[tuple[EvaluationSampler, float]],  # (sampler, ratio)
    ):
        self.samplers = samplers

    def sample(
        self,
        data: List[EvaluationSample],
        n: int,
    ) -> List[EvaluationSample]:
        """복합 샘플링"""
        if n >= len(data):
            return data[:]

        result = []
        selected_ids: Set[str] = set()

        for sampler, ratio in self.samplers:
            n_from_sampler = int(n * ratio)
            if n_from_sampler == 0:
                continue

            # 이미 선택된 것 제외
            available = [s for s in data if s.id not in selected_ids]
            if not available:
                break

            samples = sampler.sample(available, n_from_sampler)
            for s in samples:
                if s.id not in selected_ids:
                    result.append(s)
                    selected_ids.add(s.id)

        return result[:n]


@dataclass
class SamplingReport:
    """샘플링 결과 리포트"""
    total_samples: int
    sampled_count: int
    sampling_ratio: float
    distribution: Dict[str, int]
    sampling_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "sampled_count": self.sampled_count,
            "sampling_ratio": round(self.sampling_ratio, 4),
            "distribution": self.distribution,
            "sampling_time_ms": round(self.sampling_time_ms, 2),
        }


def create_sampling_report(
    original_data: List[EvaluationSample],
    sampled_data: List[EvaluationSample],
    stratify_func: Optional[Callable[[EvaluationSample], str]] = None,
    sampling_time_ms: float = 0.0,
) -> SamplingReport:
    """샘플링 결과 리포트 생성"""
    distribution = {}

    if stratify_func:
        for sample in sampled_data:
            key = stratify_func(sample)
            distribution[key] = distribution.get(key, 0) + 1

    return SamplingReport(
        total_samples=len(original_data),
        sampled_count=len(sampled_data),
        sampling_ratio=len(sampled_data) / len(original_data) if original_data else 0,
        distribution=distribution,
        sampling_time_ms=sampling_time_ms,
    )
