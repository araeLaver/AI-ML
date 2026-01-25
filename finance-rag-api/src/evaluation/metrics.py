# -*- coding: utf-8 -*-
"""
평가 메트릭 유틸리티

[기능]
- 메트릭 집계
- 신뢰 구간 계산
- 통계 검정
- 효과 크기
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceInterval:
    """신뢰 구간"""
    mean: float
    lower: float
    upper: float
    confidence_level: float  # 0.95 for 95%

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean": round(self.mean, 4),
            "lower": round(self.lower, 4),
            "upper": round(self.upper, 4),
            "confidence_level": self.confidence_level,
            "margin_of_error": round((self.upper - self.lower) / 2, 4),
        }


class MetricAggregator:
    """
    메트릭 집계기

    다양한 통계량 계산
    """

    @staticmethod
    def mean(values: List[float]) -> float:
        """평균"""
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def variance(values: List[float], sample: bool = True) -> float:
        """분산"""
        if not values:
            return 0.0

        n = len(values)
        if n == 1:
            return 0.0

        mean = MetricAggregator.mean(values)
        sq_diff_sum = sum((x - mean) ** 2 for x in values)

        if sample:
            return sq_diff_sum / (n - 1)  # 표본 분산
        return sq_diff_sum / n  # 모분산

    @staticmethod
    def std_dev(values: List[float], sample: bool = True) -> float:
        """표준편차"""
        return math.sqrt(MetricAggregator.variance(values, sample))

    @staticmethod
    def median(values: List[float]) -> float:
        """중앙값"""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        n = len(sorted_values)

        if n % 2 == 0:
            return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
        return sorted_values[n // 2]

    @staticmethod
    def percentile(values: List[float], p: float) -> float:
        """백분위수"""
        if not values:
            return 0.0
        if not 0 <= p <= 100:
            raise ValueError("Percentile must be between 0 and 100")

        sorted_values = sorted(values)
        n = len(sorted_values)
        k = (n - 1) * p / 100
        f = math.floor(k)
        c = math.ceil(k)

        if f == c:
            return sorted_values[int(k)]

        return sorted_values[int(f)] * (c - k) + sorted_values[int(c)] * (k - f)

    @staticmethod
    def quartiles(values: List[float]) -> Dict[str, float]:
        """사분위수"""
        return {
            "q1": MetricAggregator.percentile(values, 25),
            "q2": MetricAggregator.percentile(values, 50),  # median
            "q3": MetricAggregator.percentile(values, 75),
        }

    @staticmethod
    def confidence_interval(
        values: List[float],
        confidence_level: float = 0.95,
    ) -> ConfidenceInterval:
        """신뢰 구간 계산 (t-분포 근사)"""
        n = len(values)
        if n < 2:
            mean = MetricAggregator.mean(values) if values else 0
            return ConfidenceInterval(mean, mean, mean, confidence_level)

        mean = MetricAggregator.mean(values)
        std_err = MetricAggregator.std_dev(values) / math.sqrt(n)

        # t-value 근사 (degrees of freedom = n-1)
        # 95% 신뢰수준의 경우 t ≈ 2.0 for large n
        t_values = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576,
        }
        t = t_values.get(confidence_level, 1.96)

        margin = t * std_err

        return ConfidenceInterval(
            mean=mean,
            lower=mean - margin,
            upper=mean + margin,
            confidence_level=confidence_level,
        )

    @staticmethod
    def aggregate(values: List[float]) -> Dict[str, Any]:
        """종합 통계량"""
        if not values:
            return {"count": 0}

        quartiles = MetricAggregator.quartiles(values)
        ci = MetricAggregator.confidence_interval(values)

        return {
            "count": len(values),
            "mean": round(MetricAggregator.mean(values), 4),
            "std_dev": round(MetricAggregator.std_dev(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "median": round(MetricAggregator.median(values), 4),
            "q1": round(quartiles["q1"], 4),
            "q3": round(quartiles["q3"], 4),
            "ci_lower": round(ci.lower, 4),
            "ci_upper": round(ci.upper, 4),
        }


class StatisticalTest:
    """
    통계 검정

    두 그룹 간 유의미한 차이 검정
    """

    @staticmethod
    def t_test(
        group_a: List[float],
        group_b: List[float],
        paired: bool = False,
    ) -> Dict[str, Any]:
        """
        t-검정

        두 그룹의 평균 차이 검정
        """
        if not group_a or not group_b:
            return {"error": "Empty groups"}

        if paired:
            if len(group_a) != len(group_b):
                return {"error": "Paired t-test requires equal group sizes"}
            return StatisticalTest._paired_t_test(group_a, group_b)
        else:
            return StatisticalTest._independent_t_test(group_a, group_b)

    @staticmethod
    def _independent_t_test(
        group_a: List[float],
        group_b: List[float],
    ) -> Dict[str, Any]:
        """독립 표본 t-검정"""
        n_a, n_b = len(group_a), len(group_b)
        mean_a = MetricAggregator.mean(group_a)
        mean_b = MetricAggregator.mean(group_b)
        var_a = MetricAggregator.variance(group_a)
        var_b = MetricAggregator.variance(group_b)

        # Welch's t-test (불균등 분산 가정)
        se = math.sqrt(var_a / n_a + var_b / n_b)

        if se == 0:
            return {
                "t_statistic": 0,
                "degrees_of_freedom": n_a + n_b - 2,
                "mean_difference": mean_b - mean_a,
                "significant": False,
            }

        t_stat = (mean_b - mean_a) / se

        # Welch-Satterthwaite 자유도
        numerator = (var_a / n_a + var_b / n_b) ** 2
        denominator = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        df = numerator / denominator if denominator != 0 else n_a + n_b - 2

        # 유의성 판단 (간략화된 critical value)
        # 실제로는 t-분포 테이블 또는 scipy.stats 사용
        critical_t_95 = 1.96  # 근사값

        return {
            "t_statistic": round(t_stat, 4),
            "degrees_of_freedom": round(df, 2),
            "mean_difference": round(mean_b - mean_a, 4),
            "standard_error": round(se, 4),
            "significant": abs(t_stat) > critical_t_95,
            "p_value_approx": "< 0.05" if abs(t_stat) > critical_t_95 else ">= 0.05",
        }

    @staticmethod
    def _paired_t_test(
        group_a: List[float],
        group_b: List[float],
    ) -> Dict[str, Any]:
        """대응 표본 t-검정"""
        differences = [b - a for a, b in zip(group_a, group_b)]
        n = len(differences)

        mean_diff = MetricAggregator.mean(differences)
        std_diff = MetricAggregator.std_dev(differences)
        se = std_diff / math.sqrt(n)

        if se == 0:
            return {
                "t_statistic": 0,
                "degrees_of_freedom": n - 1,
                "mean_difference": mean_diff,
                "significant": False,
            }

        t_stat = mean_diff / se
        critical_t_95 = 1.96

        return {
            "t_statistic": round(t_stat, 4),
            "degrees_of_freedom": n - 1,
            "mean_difference": round(mean_diff, 4),
            "standard_error": round(se, 4),
            "significant": abs(t_stat) > critical_t_95,
        }

    @staticmethod
    def wilcoxon_signed_rank(
        group_a: List[float],
        group_b: List[float],
    ) -> Dict[str, Any]:
        """
        Wilcoxon 부호순위 검정

        비모수적 대응 표본 검정
        """
        if len(group_a) != len(group_b):
            return {"error": "Groups must have equal size"}

        differences = [(b - a, i) for i, (a, b) in enumerate(zip(group_a, group_b))]

        # 0 제외
        differences = [(d, i) for d, i in differences if d != 0]

        if not differences:
            return {"W_statistic": 0, "significant": False}

        # 절대값으로 순위 매기기
        abs_diffs = sorted([(abs(d), d > 0, i) for d, i in differences])

        # 순위 할당
        ranks = {}
        i = 0
        while i < len(abs_diffs):
            # 동점 처리
            j = i
            while j < len(abs_diffs) and abs_diffs[j][0] == abs_diffs[i][0]:
                j += 1

            avg_rank = (i + 1 + j) / 2
            for k in range(i, j):
                idx = abs_diffs[k][2]
                ranks[idx] = avg_rank
            i = j

        # W+ (양의 순위 합)과 W- (음의 순위 합)
        w_plus = sum(ranks[i] for d, i in differences if d > 0)
        w_minus = sum(ranks[i] for d, i in differences if d < 0)
        w_stat = min(w_plus, w_minus)

        n = len(differences)
        # 유의성 판단 (간략화)
        # 실제로는 Wilcoxon 테이블 사용
        significant = w_stat < n * (n + 1) / 4 * 0.5  # 대략적인 기준

        return {
            "W_statistic": w_stat,
            "W_plus": w_plus,
            "W_minus": w_minus,
            "n_pairs": n,
            "significant": significant,
        }


class EffectSize:
    """
    효과 크기 계산

    차이의 실질적 의미 평가
    """

    @staticmethod
    def cohens_d(
        group_a: List[float],
        group_b: List[float],
    ) -> Dict[str, Any]:
        """
        Cohen's d

        두 그룹 평균 차이의 효과 크기
        """
        if not group_a or not group_b:
            return {"d": 0, "interpretation": "undefined"}

        mean_a = MetricAggregator.mean(group_a)
        mean_b = MetricAggregator.mean(group_b)
        var_a = MetricAggregator.variance(group_a)
        var_b = MetricAggregator.variance(group_b)

        # 풀링된 표준편차
        n_a, n_b = len(group_a), len(group_b)
        pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 1

        d = (mean_b - mean_a) / pooled_std

        # 해석
        abs_d = abs(d)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        return {
            "d": round(d, 4),
            "interpretation": interpretation,
            "pooled_std": round(pooled_std, 4),
        }

    @staticmethod
    def hedges_g(
        group_a: List[float],
        group_b: List[float],
    ) -> Dict[str, Any]:
        """
        Hedges' g

        소표본에 대한 보정된 Cohen's d
        """
        result = EffectSize.cohens_d(group_a, group_b)

        n = len(group_a) + len(group_b)
        # 보정 계수 (소표본 보정)
        correction = 1 - (3 / (4 * n - 9)) if n > 2 else 1

        g = result["d"] * correction

        return {
            "g": round(g, 4),
            "interpretation": result["interpretation"],
            "correction_factor": round(correction, 4),
        }

    @staticmethod
    def glass_delta(
        control: List[float],
        treatment: List[float],
    ) -> Dict[str, Any]:
        """
        Glass's delta

        대조군의 표준편차로 정규화
        """
        if not control or not treatment:
            return {"delta": 0, "interpretation": "undefined"}

        mean_control = MetricAggregator.mean(control)
        mean_treatment = MetricAggregator.mean(treatment)
        std_control = MetricAggregator.std_dev(control)

        if std_control == 0:
            return {"delta": 0, "interpretation": "undefined"}

        delta = (mean_treatment - mean_control) / std_control

        abs_delta = abs(delta)
        if abs_delta < 0.2:
            interpretation = "negligible"
        elif abs_delta < 0.5:
            interpretation = "small"
        elif abs_delta < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        return {
            "delta": round(delta, 4),
            "interpretation": interpretation,
        }

    @staticmethod
    def common_language_effect_size(
        group_a: List[float],
        group_b: List[float],
    ) -> Dict[str, Any]:
        """
        Common Language Effect Size (CLES)

        무작위로 선택된 B 값이 A 값보다 클 확률
        """
        if not group_a or not group_b:
            return {"cles": 0.5}

        count = 0
        total = 0

        for a in group_a:
            for b in group_b:
                total += 1
                if b > a:
                    count += 1
                elif b == a:
                    count += 0.5

        cles = count / total if total > 0 else 0.5

        return {
            "cles": round(cles, 4),
            "interpretation": f"{round(cles * 100, 1)}% probability that B > A",
        }
