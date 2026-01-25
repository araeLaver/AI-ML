# -*- coding: utf-8 -*-
"""
피드백 루프 모듈

[기능]
- 사용자 피드백 수집 및 저장
- 피드백 분석 및 품질 모니터링
- 자동 학습 개선 파이프라인
- A/B 테스팅 프레임워크
- 능동 학습 (Active Learning)
"""

from .collector import (
    FeedbackCollector,
    FeedbackType,
    FeedbackData,
    ImplicitFeedbackTracker,
    ExplicitFeedbackHandler,
)
from .storage import (
    FeedbackStorage,
    InMemoryFeedbackStorage,
    SQLFeedbackStorage,
    FeedbackQuery,
)
from .analyzer import (
    FeedbackAnalyzer,
    QualityMetrics,
    TrendAnalyzer,
    SentimentAnalyzer,
)
from .learning import (
    LearningPipeline,
    RetrainTrigger,
    DataSelector,
    ModelUpdater,
    IncrementalLearner,
)
from .ab_testing import (
    ABTestManager,
    Experiment,
    Variant,
    StatisticalAnalyzer,
)
from .active_learning import (
    ActiveLearner,
    UncertaintySampler,
    DiversitySampler,
    QueryStrategy,
)

__all__ = [
    # Collector
    "FeedbackCollector",
    "FeedbackType",
    "FeedbackData",
    "ImplicitFeedbackTracker",
    "ExplicitFeedbackHandler",
    # Storage
    "FeedbackStorage",
    "InMemoryFeedbackStorage",
    "SQLFeedbackStorage",
    "FeedbackQuery",
    # Analyzer
    "FeedbackAnalyzer",
    "QualityMetrics",
    "TrendAnalyzer",
    "SentimentAnalyzer",
    # Learning
    "LearningPipeline",
    "RetrainTrigger",
    "DataSelector",
    "ModelUpdater",
    "IncrementalLearner",
    # A/B Testing
    "ABTestManager",
    "Experiment",
    "Variant",
    "StatisticalAnalyzer",
    # Active Learning
    "ActiveLearner",
    "UncertaintySampler",
    "DiversitySampler",
    "QueryStrategy",
]
