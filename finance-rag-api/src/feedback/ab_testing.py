# -*- coding: utf-8 -*-
"""
A/B 테스팅 모듈

[기능]
- 실험 관리
- 트래픽 분배
- 통계 분석
- 결과 리포트
"""

import hashlib
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .collector import FeedbackData, FeedbackSentiment
from .storage import FeedbackQuery, FeedbackStorage

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """실험 상태"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


@dataclass
class Variant:
    """실험 변형"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    weight: float = 0.5  # 트래픽 비율
    config: Dict[str, Any] = field(default_factory=dict)

    # 결과
    impressions: int = 0
    conversions: int = 0
    total_feedback: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    total_rating: float = 0.0
    rating_count: int = 0

    @property
    def conversion_rate(self) -> float:
        return self.conversions / self.impressions if self.impressions > 0 else 0.0

    @property
    def satisfaction_rate(self) -> float:
        if self.total_feedback == 0:
            return 0.0
        return self.positive_feedback / self.total_feedback

    @property
    def avg_rating(self) -> float:
        return self.total_rating / self.rating_count if self.rating_count > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "weight": self.weight,
            "config": self.config,
            "impressions": self.impressions,
            "conversions": self.conversions,
            "conversion_rate": round(self.conversion_rate, 4),
            "satisfaction_rate": round(self.satisfaction_rate, 4),
            "avg_rating": round(self.avg_rating, 2),
            "total_feedback": self.total_feedback,
        }


@dataclass
class Experiment:
    """실험"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    hypothesis: str = ""
    status: ExperimentStatus = ExperimentStatus.DRAFT
    variants: List[Variant] = field(default_factory=list)

    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None

    # 설정
    target_sample_size: int = 1000
    min_confidence: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "status": self.status.value,
            "variants": [v.to_dict() for v in self.variants],
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "target_sample_size": self.target_sample_size,
            "min_confidence": self.min_confidence,
            "total_impressions": sum(v.impressions for v in self.variants),
        }


class StatisticalAnalyzer:
    """
    통계 분석기

    A/B 테스트 결과 분석
    """

    @staticmethod
    def calculate_z_score(
        p1: float,
        p2: float,
        n1: int,
        n2: int,
    ) -> float:
        """Z-score 계산"""
        if n1 == 0 or n2 == 0:
            return 0.0

        # Pooled proportion
        p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)

        if p_pool == 0 or p_pool == 1:
            return 0.0

        # Standard error
        se = (p_pool * (1 - p_pool) * (1/n1 + 1/n2)) ** 0.5

        if se == 0:
            return 0.0

        return (p1 - p2) / se

    @staticmethod
    def z_to_p_value(z: float) -> float:
        """Z-score를 p-value로 변환 (two-tailed)"""
        # 간단한 근사
        import math

        if abs(z) > 6:
            return 0.0

        # 표준 정규 분포 CDF 근사
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        sign = 1 if z >= 0 else -1
        z = abs(z) / math.sqrt(2)

        t = 1.0 / (1.0 + p * z)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z * z)

        cdf = 0.5 * (1.0 + sign * y)

        # Two-tailed p-value
        return 2 * (1 - cdf)

    def compare_variants(
        self,
        control: Variant,
        treatment: Variant,
        metric: str = "conversion_rate",
    ) -> Dict[str, Any]:
        """두 변형 비교"""
        if metric == "conversion_rate":
            p1 = control.conversion_rate
            p2 = treatment.conversion_rate
            n1 = control.impressions
            n2 = treatment.impressions
        elif metric == "satisfaction_rate":
            p1 = control.satisfaction_rate
            p2 = treatment.satisfaction_rate
            n1 = control.total_feedback
            n2 = treatment.total_feedback
        else:
            raise ValueError(f"Unknown metric: {metric}")

        z_score = self.calculate_z_score(p1, p2, n1, n2)
        p_value = self.z_to_p_value(z_score)

        # 효과 크기 (상대적 차이)
        relative_lift = (p2 - p1) / p1 if p1 > 0 else 0.0

        # 신뢰 구간 (95%)
        if n2 > 0:
            se = ((p2 * (1 - p2)) / n2) ** 0.5
            ci_lower = p2 - 1.96 * se
            ci_upper = p2 + 1.96 * se
        else:
            ci_lower = ci_upper = 0.0

        return {
            "control": {
                "name": control.name,
                "value": round(p1, 4),
                "samples": n1,
            },
            "treatment": {
                "name": treatment.name,
                "value": round(p2, 4),
                "samples": n2,
            },
            "z_score": round(z_score, 3),
            "p_value": round(p_value, 4),
            "is_significant": p_value < 0.05,
            "relative_lift": round(relative_lift, 4),
            "confidence_interval": [round(ci_lower, 4), round(ci_upper, 4)],
            "winner": treatment.name if p_value < 0.05 and relative_lift > 0 else (
                control.name if p_value < 0.05 and relative_lift < 0 else None
            ),
        }

    def calculate_sample_size(
        self,
        baseline_rate: float,
        min_detectable_effect: float,
        power: float = 0.8,
        alpha: float = 0.05,
    ) -> int:
        """필요 샘플 크기 계산"""
        import math

        # Z-scores for alpha and power
        z_alpha = 1.96 if alpha == 0.05 else 2.576  # 95% or 99%
        z_beta = 0.84 if power == 0.8 else 1.28  # 80% or 90%

        p1 = baseline_rate
        p2 = baseline_rate * (1 + min_detectable_effect)

        # Pooled standard deviation
        p_avg = (p1 + p2) / 2
        pooled_sd = math.sqrt(2 * p_avg * (1 - p_avg))

        # Sample size per group
        effect_size = abs(p2 - p1)
        if effect_size == 0:
            return 0

        n = 2 * ((z_alpha + z_beta) * pooled_sd / effect_size) ** 2

        return int(math.ceil(n))


class ABTestManager:
    """
    A/B 테스트 관리자

    실험 생성, 실행, 분석
    """

    def __init__(
        self,
        storage: Optional[FeedbackStorage] = None,
        sticky_assignment: bool = True,
    ):
        self.storage = storage
        self.sticky_assignment = sticky_assignment
        self.analyzer = StatisticalAnalyzer()
        self._experiments: Dict[str, Experiment] = {}
        self._assignments: Dict[str, Dict[str, str]] = {}  # user_id -> {exp_id: variant_id}

    def create_experiment(
        self,
        name: str,
        variants: List[Dict[str, Any]],
        description: str = "",
        hypothesis: str = "",
        target_sample_size: int = 1000,
    ) -> Experiment:
        """실험 생성"""
        exp = Experiment(
            name=name,
            description=description,
            hypothesis=hypothesis,
            target_sample_size=target_sample_size,
        )

        # 변형 추가
        total_weight = sum(v.get("weight", 1.0) for v in variants)
        for v in variants:
            variant = Variant(
                name=v["name"],
                description=v.get("description", ""),
                weight=v.get("weight", 1.0) / total_weight,
                config=v.get("config", {}),
            )
            exp.variants.append(variant)

        self._experiments[exp.id] = exp
        logger.info(f"Created experiment: {name} ({exp.id})")
        return exp

    def start_experiment(self, experiment_id: str) -> bool:
        """실험 시작"""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return False

        if exp.status != ExperimentStatus.DRAFT:
            return False

        exp.status = ExperimentStatus.RUNNING
        exp.started_at = time.time()
        logger.info(f"Started experiment: {exp.name}")
        return True

    def stop_experiment(self, experiment_id: str) -> bool:
        """실험 중지"""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return False

        exp.status = ExperimentStatus.COMPLETED
        exp.ended_at = time.time()
        logger.info(f"Stopped experiment: {exp.name}")
        return True

    def get_variant(
        self,
        experiment_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[Variant]:
        """사용자에게 변형 할당"""
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status != ExperimentStatus.RUNNING:
            return None

        # Sticky assignment 확인
        if self.sticky_assignment and user_id:
            if user_id in self._assignments:
                variant_id = self._assignments[user_id].get(experiment_id)
                if variant_id:
                    for v in exp.variants:
                        if v.id == variant_id:
                            return v

        # 새 할당
        variant = self._assign_variant(exp, user_id)

        # Sticky assignment 저장
        if self.sticky_assignment and user_id and variant:
            if user_id not in self._assignments:
                self._assignments[user_id] = {}
            self._assignments[user_id][experiment_id] = variant.id

        return variant

    def _assign_variant(
        self,
        experiment: Experiment,
        user_id: Optional[str] = None,
    ) -> Optional[Variant]:
        """변형 할당"""
        if not experiment.variants:
            return None

        if user_id:
            # 결정적 할당 (동일 사용자는 항상 같은 변형)
            hash_input = f"{experiment.id}:{user_id}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            normalized = (hash_value % 10000) / 10000
        else:
            # 랜덤 할당
            normalized = random.random()

        cumulative = 0.0
        for variant in experiment.variants:
            cumulative += variant.weight
            if normalized < cumulative:
                return variant

        return experiment.variants[-1]

    def record_impression(
        self,
        experiment_id: str,
        variant_id: str,
    ) -> None:
        """노출 기록"""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return

        for variant in exp.variants:
            if variant.id == variant_id:
                variant.impressions += 1
                break

    def record_conversion(
        self,
        experiment_id: str,
        variant_id: str,
    ) -> None:
        """전환 기록"""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return

        for variant in exp.variants:
            if variant.id == variant_id:
                variant.conversions += 1
                break

    def record_feedback(
        self,
        experiment_id: str,
        variant_id: str,
        feedback: FeedbackData,
    ) -> None:
        """피드백 기록"""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return

        for variant in exp.variants:
            if variant.id == variant_id:
                variant.total_feedback += 1

                if feedback.sentiment == FeedbackSentiment.POSITIVE:
                    variant.positive_feedback += 1
                elif feedback.sentiment == FeedbackSentiment.NEGATIVE:
                    variant.negative_feedback += 1

                if feedback.value and isinstance(feedback.value, (int, float)):
                    variant.total_rating += feedback.value
                    variant.rating_count += 1
                break

    def get_results(self, experiment_id: str) -> Dict[str, Any]:
        """실험 결과"""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return {}

        results = exp.to_dict()

        # 변형 비교
        if len(exp.variants) >= 2:
            control = exp.variants[0]
            comparisons = []

            for treatment in exp.variants[1:]:
                comparison = self.analyzer.compare_variants(
                    control, treatment, "satisfaction_rate"
                )
                comparisons.append(comparison)

            results["comparisons"] = comparisons

        # 완료 여부
        total_impressions = sum(v.impressions for v in exp.variants)
        results["progress"] = min(total_impressions / exp.target_sample_size, 1.0)
        results["is_complete"] = total_impressions >= exp.target_sample_size

        return results

    def get_recommendation(self, experiment_id: str) -> Dict[str, Any]:
        """실험 결과 기반 추천"""
        results = self.get_results(experiment_id)
        if not results:
            return {"recommendation": "experiment_not_found"}

        exp = self._experiments.get(experiment_id)

        # 충분한 데이터 확인
        if results.get("progress", 0) < 0.5:
            return {
                "recommendation": "continue",
                "reason": "샘플 수가 부족합니다.",
                "progress": results.get("progress", 0),
            }

        # 승자 확인
        comparisons = results.get("comparisons", [])
        for comp in comparisons:
            if comp.get("is_significant") and comp.get("winner"):
                winner_name = comp["winner"]
                winner_variant = None
                for v in exp.variants:
                    if v.name == winner_name:
                        winner_variant = v
                        break

                return {
                    "recommendation": "implement_winner",
                    "winner": winner_name,
                    "config": winner_variant.config if winner_variant else {},
                    "lift": comp.get("relative_lift", 0),
                    "confidence": 1 - comp.get("p_value", 1),
                }

        # 유의미한 차이 없음
        if results.get("is_complete"):
            return {
                "recommendation": "no_significant_difference",
                "reason": "실험이 완료되었지만 유의미한 차이가 없습니다.",
            }

        return {
            "recommendation": "continue",
            "reason": "아직 유의미한 결과가 나오지 않았습니다.",
            "progress": results.get("progress", 0),
        }

    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
    ) -> List[Dict[str, Any]]:
        """실험 목록"""
        experiments = []
        for exp in self._experiments.values():
            if status is None or exp.status == status:
                experiments.append(exp.to_dict())

        return sorted(experiments, key=lambda x: x["created_at"], reverse=True)
