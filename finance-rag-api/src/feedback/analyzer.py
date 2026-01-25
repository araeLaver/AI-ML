# -*- coding: utf-8 -*-
"""
피드백 분석 모듈

[기능]
- 품질 메트릭 계산
- 트렌드 분석
- 감정 분석
- 이상 탐지
"""

import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from .collector import FeedbackData, FeedbackSentiment, FeedbackType
from .storage import FeedbackQuery, FeedbackStorage

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """품질 메트릭"""
    satisfaction_rate: float = 0.0  # 만족도 (긍정 비율)
    avg_rating: float = 0.0  # 평균 평점
    nps_score: float = 0.0  # Net Promoter Score
    resolution_rate: float = 0.0  # 해결률 (수정 불필요 비율)
    engagement_rate: float = 0.0  # 참여율 (피드백 제공 비율)

    total_feedback: int = 0
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "satisfaction_rate": round(self.satisfaction_rate, 3),
            "avg_rating": round(self.avg_rating, 2),
            "nps_score": round(self.nps_score, 1),
            "resolution_rate": round(self.resolution_rate, 3),
            "engagement_rate": round(self.engagement_rate, 3),
            "total_feedback": self.total_feedback,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "neutral_count": self.neutral_count,
        }


@dataclass
class TrendData:
    """트렌드 데이터"""
    period: str  # 기간 (일/주/월)
    start_time: float
    end_time: float
    metrics: QualityMetrics
    change_from_previous: Optional[Dict[str, float]] = None


class FeedbackAnalyzer:
    """
    피드백 분석기

    품질 메트릭 및 트렌드 분석
    """

    def __init__(self, storage: FeedbackStorage):
        self.storage = storage

    def calculate_metrics(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        user_id: Optional[str] = None,
    ) -> QualityMetrics:
        """품질 메트릭 계산"""
        query = FeedbackQuery(
            start_time=start_time,
            end_time=end_time,
            user_id=user_id,
            limit=100000,
        )
        feedback_list = self.storage.query(query)

        if not feedback_list:
            return QualityMetrics()

        metrics = QualityMetrics(total_feedback=len(feedback_list))

        ratings = []
        nps_scores = []

        for fb in feedback_list:
            # 감정별 카운트
            if fb.sentiment == FeedbackSentiment.POSITIVE:
                metrics.positive_count += 1
            elif fb.sentiment == FeedbackSentiment.NEGATIVE:
                metrics.negative_count += 1
            else:
                metrics.neutral_count += 1

            # 평점 수집
            if fb.feedback_type == FeedbackType.RATING and fb.value:
                ratings.append(fb.value)
                # NPS: 1-5 스케일을 0-10으로 변환
                nps_scores.append((fb.value - 1) * 2.5)

        # 만족도
        if metrics.total_feedback > 0:
            metrics.satisfaction_rate = metrics.positive_count / metrics.total_feedback

        # 평균 평점
        if ratings:
            metrics.avg_rating = sum(ratings) / len(ratings)

        # NPS 계산 (Promoters - Detractors)
        if nps_scores:
            promoters = sum(1 for s in nps_scores if s >= 9) / len(nps_scores)
            detractors = sum(1 for s in nps_scores if s <= 6) / len(nps_scores)
            metrics.nps_score = (promoters - detractors) * 100

        # 해결률 (수정 제안이 없는 비율)
        corrections = sum(1 for fb in feedback_list if fb.feedback_type == FeedbackType.CORRECTION)
        if metrics.total_feedback > 0:
            metrics.resolution_rate = 1 - (corrections / metrics.total_feedback)

        return metrics

    def get_top_issues(self, limit: int = 10) -> List[Dict[str, Any]]:
        """상위 이슈 (부정 피드백이 많은 쿼리)"""
        query = FeedbackQuery(
            sentiments=[FeedbackSentiment.NEGATIVE],
            limit=10000,
        )
        negative_feedback = self.storage.query(query)

        # 쿼리별 그룹화
        query_issues: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "queries": [], "reasons": []}
        )

        for fb in negative_feedback:
            # 쿼리 정규화 (소문자, 공백 제거)
            normalized = fb.query.lower().strip()
            query_issues[normalized]["count"] += 1
            query_issues[normalized]["queries"].append(fb.query)

            if fb.metadata.get("reason"):
                query_issues[normalized]["reasons"].append(fb.metadata["reason"])

        # 상위 이슈 정렬
        issues = [
            {
                "query": data["queries"][0],
                "negative_count": data["count"],
                "reasons": list(set(data["reasons"]))[:5],
            }
            for query, data in sorted(
                query_issues.items(),
                key=lambda x: x[1]["count"],
                reverse=True,
            )[:limit]
        ]

        return issues

    def get_improvement_suggestions(self) -> List[Dict[str, Any]]:
        """개선 제안"""
        suggestions = []

        metrics = self.calculate_metrics()

        # 낮은 만족도
        if metrics.satisfaction_rate < 0.7:
            suggestions.append({
                "type": "satisfaction",
                "priority": "high" if metrics.satisfaction_rate < 0.5 else "medium",
                "message": f"만족도가 {metrics.satisfaction_rate:.1%}로 낮습니다.",
                "recommendation": "부정 피드백이 많은 쿼리 유형을 분석하고 답변 품질을 개선하세요.",
            })

        # 낮은 평점
        if metrics.avg_rating > 0 and metrics.avg_rating < 3.5:
            suggestions.append({
                "type": "rating",
                "priority": "high" if metrics.avg_rating < 2.5 else "medium",
                "message": f"평균 평점이 {metrics.avg_rating:.1f}점으로 낮습니다.",
                "recommendation": "낮은 평점의 피드백을 분석하여 개선 포인트를 파악하세요.",
            })

        # 수정 제안이 많음
        if metrics.resolution_rate < 0.9:
            suggestions.append({
                "type": "corrections",
                "priority": "medium",
                "message": f"답변 수정률이 {1 - metrics.resolution_rate:.1%}입니다.",
                "recommendation": "수정 제안을 학습 데이터로 활용하여 모델을 개선하세요.",
            })

        # 상위 이슈 기반 제안
        top_issues = self.get_top_issues(5)
        if top_issues:
            suggestions.append({
                "type": "top_issues",
                "priority": "high",
                "message": f"{len(top_issues)}개의 반복적인 문제 쿼리가 발견되었습니다.",
                "recommendation": "해당 쿼리에 대한 특화 답변 또는 문서를 추가하세요.",
                "issues": top_issues[:3],
            })

        return suggestions

    def compare_periods(
        self,
        current_start: float,
        current_end: float,
        previous_start: float,
        previous_end: float,
    ) -> Dict[str, Any]:
        """기간 비교"""
        current = self.calculate_metrics(current_start, current_end)
        previous = self.calculate_metrics(previous_start, previous_end)

        def calc_change(curr: float, prev: float) -> float:
            if prev == 0:
                return 0.0
            return (curr - prev) / prev * 100

        return {
            "current": current.to_dict(),
            "previous": previous.to_dict(),
            "changes": {
                "satisfaction_rate": calc_change(
                    current.satisfaction_rate, previous.satisfaction_rate
                ),
                "avg_rating": calc_change(current.avg_rating, previous.avg_rating),
                "total_feedback": calc_change(
                    current.total_feedback, previous.total_feedback
                ),
            },
        }


class TrendAnalyzer:
    """
    트렌드 분석기

    시간에 따른 피드백 변화 분석
    """

    def __init__(self, storage: FeedbackStorage):
        self.storage = storage
        self.analyzer = FeedbackAnalyzer(storage)

    def get_daily_trend(self, days: int = 30) -> List[TrendData]:
        """일별 트렌드"""
        trends = []
        now = time.time()
        day_seconds = 86400  # 24시간

        for i in range(days, 0, -1):
            end_time = now - (i - 1) * day_seconds
            start_time = end_time - day_seconds

            metrics = self.analyzer.calculate_metrics(start_time, end_time)

            trend = TrendData(
                period="daily",
                start_time=start_time,
                end_time=end_time,
                metrics=metrics,
            )
            trends.append(trend)

        # 변화율 계산
        for i in range(1, len(trends)):
            prev = trends[i - 1].metrics
            curr = trends[i].metrics

            if prev.total_feedback > 0:
                trends[i].change_from_previous = {
                    "satisfaction_rate": curr.satisfaction_rate - prev.satisfaction_rate,
                    "avg_rating": curr.avg_rating - prev.avg_rating,
                    "total_feedback": curr.total_feedback - prev.total_feedback,
                }

        return trends

    def get_weekly_trend(self, weeks: int = 12) -> List[TrendData]:
        """주별 트렌드"""
        trends = []
        now = time.time()
        week_seconds = 604800  # 7일

        for i in range(weeks, 0, -1):
            end_time = now - (i - 1) * week_seconds
            start_time = end_time - week_seconds

            metrics = self.analyzer.calculate_metrics(start_time, end_time)

            trend = TrendData(
                period="weekly",
                start_time=start_time,
                end_time=end_time,
                metrics=metrics,
            )
            trends.append(trend)

        return trends

    def detect_anomalies(self, threshold: float = 2.0) -> List[Dict[str, Any]]:
        """이상 탐지 (표준편차 기반)"""
        daily_trends = self.get_daily_trend(30)

        if len(daily_trends) < 7:
            return []

        # 만족도의 평균과 표준편차
        satisfaction_rates = [t.metrics.satisfaction_rate for t in daily_trends if t.metrics.total_feedback > 0]

        if len(satisfaction_rates) < 7:
            return []

        import statistics
        mean = statistics.mean(satisfaction_rates)
        stdev = statistics.stdev(satisfaction_rates)

        anomalies = []
        for trend in daily_trends:
            if trend.metrics.total_feedback == 0:
                continue

            z_score = (trend.metrics.satisfaction_rate - mean) / stdev if stdev > 0 else 0

            if abs(z_score) > threshold:
                anomalies.append({
                    "date": datetime.fromtimestamp(trend.start_time).strftime("%Y-%m-%d"),
                    "satisfaction_rate": trend.metrics.satisfaction_rate,
                    "z_score": round(z_score, 2),
                    "type": "positive_spike" if z_score > 0 else "negative_spike",
                })

        return anomalies


class SentimentAnalyzer:
    """
    감정 분석기

    텍스트 기반 감정 분석
    """

    def __init__(self):
        # 긍정/부정 키워드 (한국어)
        self.positive_keywords = [
            "좋", "훌륭", "만족", "감사", "도움", "정확", "빠른", "유용",
            "최고", "완벽", "추천", "편리", "쉬운", "잘", "굿", "좋아",
        ]
        self.negative_keywords = [
            "나쁜", "불만", "오류", "틀린", "느린", "불편", "어려운",
            "이상", "실패", "안됨", "별로", "최악", "싫", "못", "잘못",
        ]

    def analyze(self, text: str) -> Dict[str, Any]:
        """텍스트 감정 분석"""
        text_lower = text.lower()

        positive_count = sum(1 for kw in self.positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in self.negative_keywords if kw in text_lower)

        total = positive_count + negative_count

        if total == 0:
            sentiment = FeedbackSentiment.NEUTRAL
            confidence = 0.5
        else:
            positive_ratio = positive_count / total
            if positive_ratio > 0.6:
                sentiment = FeedbackSentiment.POSITIVE
                confidence = min(0.5 + positive_ratio * 0.5, 1.0)
            elif positive_ratio < 0.4:
                sentiment = FeedbackSentiment.NEGATIVE
                confidence = min(0.5 + (1 - positive_ratio) * 0.5, 1.0)
            else:
                sentiment = FeedbackSentiment.NEUTRAL
                confidence = 0.5

        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "positive_keywords_found": positive_count,
            "negative_keywords_found": negative_count,
        }

    def extract_topics(self, feedbacks: List[FeedbackData]) -> Dict[str, int]:
        """피드백에서 토픽 추출"""
        topic_counts: Dict[str, int] = defaultdict(int)

        for fb in feedbacks:
            if fb.feedback_type == FeedbackType.COMMENT and fb.value:
                # 간단한 키워드 추출
                words = re.findall(r'\b[가-힣a-zA-Z]{2,}\b', str(fb.value))
                for word in words:
                    topic_counts[word.lower()] += 1

        # 상위 토픽 반환
        return dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:20])

    def get_sentiment_distribution(
        self,
        feedbacks: List[FeedbackData],
    ) -> Dict[str, Any]:
        """감정 분포"""
        distribution = {
            "positive": 0,
            "negative": 0,
            "neutral": 0,
        }

        for fb in feedbacks:
            if fb.sentiment == FeedbackSentiment.POSITIVE:
                distribution["positive"] += 1
            elif fb.sentiment == FeedbackSentiment.NEGATIVE:
                distribution["negative"] += 1
            else:
                distribution["neutral"] += 1

        total = sum(distribution.values())
        if total > 0:
            distribution["positive_ratio"] = distribution["positive"] / total
            distribution["negative_ratio"] = distribution["negative"] / total
            distribution["neutral_ratio"] = distribution["neutral"] / total
        else:
            distribution["positive_ratio"] = 0
            distribution["negative_ratio"] = 0
            distribution["neutral_ratio"] = 0

        return distribution
