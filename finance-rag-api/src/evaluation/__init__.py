# -*- coding: utf-8 -*-
"""
고급 평가 모듈

[기능]
- RAGAS 기반 RAG 품질 평가
- Human evaluation 관리
- 평가 파이프라인
"""

from .ragas import (
    RAGASEvaluator,
    RAGASMetrics,
    FaithfulnessEvaluator,
    AnswerRelevancyEvaluator,
    ContextPrecisionEvaluator,
    ContextRecallEvaluator,
)
from .human_eval import (
    HumanEvaluator,
    AnnotationTask,
    AnnotationResult,
    EvaluationCriteria,
    InterRaterReliability,
    BlindEvaluator,
)
from .pipeline import (
    EvaluationPipeline,
    EvaluationRun,
    EvaluationReport,
    BatchEvaluator,
    ComparisonAnalyzer,
)
from .metrics import (
    MetricAggregator,
    ConfidenceInterval,
    StatisticalTest,
    EffectSize,
)
from .sampling import (
    EvaluationSampler,
    StratifiedSampler,
    DiversitySampler as EvalDiversitySampler,
    TemporalSampler,
)

__all__ = [
    # RAGAS
    "RAGASEvaluator",
    "RAGASMetrics",
    "FaithfulnessEvaluator",
    "AnswerRelevancyEvaluator",
    "ContextPrecisionEvaluator",
    "ContextRecallEvaluator",
    # Human Evaluation
    "HumanEvaluator",
    "AnnotationTask",
    "AnnotationResult",
    "EvaluationCriteria",
    "InterRaterReliability",
    "BlindEvaluator",
    # Pipeline
    "EvaluationPipeline",
    "EvaluationRun",
    "EvaluationReport",
    "BatchEvaluator",
    "ComparisonAnalyzer",
    # Metrics
    "MetricAggregator",
    "ConfidenceInterval",
    "StatisticalTest",
    "EffectSize",
    # Sampling
    "EvaluationSampler",
    "StratifiedSampler",
    "EvalDiversitySampler",
    "TemporalSampler",
]
