# -*- coding: utf-8 -*-
"""
평가 파이프라인 모듈

[기능]
- 자동화된 평가 실행
- 배치 평가
- 결과 비교 분석
- 리포트 생성
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .ragas import RAGASEvaluator, RAGASMetrics, EvaluationSample
from .human_eval import HumanEvaluator, AnnotationResult

logger = logging.getLogger(__name__)


class EvaluationStatus(Enum):
    """평가 실행 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class EvaluationRun:
    """평가 실행 정보"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # 실행 정보
    status: EvaluationStatus = EvaluationStatus.PENDING
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # 설정
    config: Dict[str, Any] = field(default_factory=dict)

    # 결과
    total_samples: int = 0
    evaluated_samples: int = 0
    ragas_results: List[RAGASMetrics] = field(default_factory=list)
    human_results: List[AnnotationResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # 메타데이터
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    dataset_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return 0.0

    @property
    def progress(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.evaluated_samples / self.total_samples

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "progress": round(self.progress, 2),
            "total_samples": self.total_samples,
            "evaluated_samples": self.evaluated_samples,
            "duration_seconds": round(self.duration_seconds, 2),
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "error_count": len(self.errors),
        }


@dataclass
class EvaluationReport:
    """평가 리포트"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""
    generated_at: float = field(default_factory=time.time)

    # RAGAS 메트릭 요약
    ragas_summary: Dict[str, Any] = field(default_factory=dict)

    # Human 평가 요약
    human_summary: Dict[str, Any] = field(default_factory=dict)

    # 비교 분석 (있는 경우)
    comparison: Optional[Dict[str, Any]] = None

    # 권장사항
    recommendations: List[str] = field(default_factory=list)

    # 상세 결과
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "generated_at": datetime.fromtimestamp(self.generated_at).isoformat(),
            "ragas_summary": self.ragas_summary,
            "human_summary": self.human_summary,
            "comparison": self.comparison,
            "recommendations": self.recommendations,
        }

    def to_markdown(self) -> str:
        """마크다운 형식 리포트"""
        md = []
        md.append(f"# Evaluation Report")
        md.append(f"\n**Run ID:** {self.run_id}")
        md.append(f"\n**Generated:** {datetime.fromtimestamp(self.generated_at).isoformat()}")

        md.append("\n\n## RAGAS Metrics")
        if self.ragas_summary:
            for metric, values in self.ragas_summary.items():
                if isinstance(values, dict) and "mean" in values:
                    mean_val = values['mean']
                    min_val = values.get('min')
                    max_val = values.get('max')
                    min_str = f"{min_val:.4f}" if isinstance(min_val, (int, float)) else "N/A"
                    max_str = f"{max_val:.4f}" if isinstance(max_val, (int, float)) else "N/A"
                    md.append(f"- **{metric}:** {mean_val:.4f} (min: {min_str}, max: {max_str})")
                else:
                    md.append(f"- **{metric}:** {values}")

        if self.human_summary:
            md.append("\n\n## Human Evaluation")
            for key, value in self.human_summary.items():
                md.append(f"- **{key}:** {value}")

        if self.comparison:
            md.append("\n\n## Comparison Analysis")
            for key, value in self.comparison.items():
                md.append(f"- **{key}:** {value}")

        if self.recommendations:
            md.append("\n\n## Recommendations")
            for i, rec in enumerate(self.recommendations, 1):
                md.append(f"{i}. {rec}")

        return "\n".join(md)


class BatchEvaluator:
    """
    배치 평가기

    대량의 샘플을 효율적으로 평가
    """

    def __init__(
        self,
        ragas_evaluator: Optional[RAGASEvaluator] = None,
        batch_size: int = 10,
        parallel: bool = False,
    ):
        self.ragas_evaluator = ragas_evaluator or RAGASEvaluator()
        self.batch_size = batch_size
        self.parallel = parallel

    def evaluate(
        self,
        samples: List[EvaluationSample],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[RAGASMetrics]:
        """배치 평가 실행"""
        results = []
        total = len(samples)

        for i in range(0, total, self.batch_size):
            batch = samples[i:i + self.batch_size]
            batch_results = self._evaluate_batch(batch)
            results.extend(batch_results)

            if progress_callback:
                progress_callback(len(results), total)

        return results

    def _evaluate_batch(
        self,
        batch: List[EvaluationSample],
    ) -> List[RAGASMetrics]:
        """단일 배치 평가"""
        # 실제로는 병렬 처리 가능
        return [self.ragas_evaluator.evaluate(sample) for sample in batch]


class ComparisonAnalyzer:
    """
    비교 분석기

    여러 모델/설정 간 성능 비교
    """

    def compare_runs(
        self,
        run_a: EvaluationRun,
        run_b: EvaluationRun,
    ) -> Dict[str, Any]:
        """두 실행 결과 비교"""
        comparison = {
            "run_a": run_a.id,
            "run_b": run_b.id,
            "metrics_comparison": {},
            "winner": None,
            "significant_differences": [],
        }

        # RAGAS 메트릭 비교
        if run_a.ragas_results and run_b.ragas_results:
            a_metrics = self._aggregate_metrics(run_a.ragas_results)
            b_metrics = self._aggregate_metrics(run_b.ragas_results)

            for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "overall_score"]:
                a_val = a_metrics.get(metric, 0)
                b_val = b_metrics.get(metric, 0)
                diff = b_val - a_val
                pct_change = (diff / a_val * 100) if a_val != 0 else 0

                comparison["metrics_comparison"][metric] = {
                    "run_a": round(a_val, 4),
                    "run_b": round(b_val, 4),
                    "difference": round(diff, 4),
                    "pct_change": round(pct_change, 2),
                    "better": "B" if diff > 0 else ("A" if diff < 0 else "tie"),
                }

                if abs(diff) > 0.05:  # 5% 이상 차이
                    comparison["significant_differences"].append(metric)

            # 승자 결정
            a_wins = sum(1 for m in comparison["metrics_comparison"].values() if m["better"] == "A")
            b_wins = sum(1 for m in comparison["metrics_comparison"].values() if m["better"] == "B")

            if a_wins > b_wins:
                comparison["winner"] = run_a.id
            elif b_wins > a_wins:
                comparison["winner"] = run_b.id
            else:
                comparison["winner"] = "tie"

        return comparison

    def _aggregate_metrics(
        self,
        metrics_list: List[RAGASMetrics],
    ) -> Dict[str, float]:
        """메트릭 집계"""
        if not metrics_list:
            return {}

        n = len(metrics_list)
        return {
            "faithfulness": sum(m.faithfulness for m in metrics_list) / n,
            "answer_relevancy": sum(m.answer_relevancy for m in metrics_list) / n,
            "context_precision": sum(m.context_precision for m in metrics_list) / n,
            "context_recall": sum(m.context_recall for m in metrics_list) / n,
            "overall_score": sum(m.overall_score for m in metrics_list) / n,
        }

    def rank_runs(
        self,
        runs: List[EvaluationRun],
        metric: str = "overall_score",
    ) -> List[Tuple[EvaluationRun, float]]:
        """여러 실행 순위 매기기"""
        scored_runs = []
        for run in runs:
            if run.ragas_results:
                metrics = self._aggregate_metrics(run.ragas_results)
                score = metrics.get(metric, 0)
                scored_runs.append((run, score))

        scored_runs.sort(key=lambda x: x[1], reverse=True)
        return scored_runs


class EvaluationPipeline:
    """
    평가 파이프라인

    전체 평가 워크플로우 관리
    """

    def __init__(
        self,
        ragas_evaluator: Optional[RAGASEvaluator] = None,
        human_evaluator: Optional[HumanEvaluator] = None,
    ):
        self.ragas_evaluator = ragas_evaluator or RAGASEvaluator()
        self.human_evaluator = human_evaluator
        self.batch_evaluator = BatchEvaluator(ragas_evaluator=self.ragas_evaluator)
        self.comparison_analyzer = ComparisonAnalyzer()

        self._runs: Dict[str, EvaluationRun] = {}
        self._reports: Dict[str, EvaluationReport] = {}

    def create_run(
        self,
        name: str,
        description: str = "",
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        dataset_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> EvaluationRun:
        """평가 실행 생성"""
        run = EvaluationRun(
            name=name,
            description=description,
            model_name=model_name,
            model_version=model_version,
            dataset_name=dataset_name,
            config=config or {},
        )
        self._runs[run.id] = run
        logger.info(f"Created evaluation run: {run.id}")
        return run

    def execute_run(
        self,
        run: EvaluationRun,
        samples: List[EvaluationSample],
        include_human_eval: bool = False,
    ) -> EvaluationRun:
        """평가 실행"""
        run.status = EvaluationStatus.RUNNING
        run.started_at = time.time()
        run.total_samples = len(samples)

        try:
            # RAGAS 자동 평가
            def progress_callback(current: int, total: int):
                run.evaluated_samples = current

            run.ragas_results = self.batch_evaluator.evaluate(
                samples,
                progress_callback=progress_callback,
            )

            # Human 평가 설정 (있는 경우)
            if include_human_eval and self.human_evaluator:
                for sample in samples:
                    self.human_evaluator.create_task(
                        question=sample.question,
                        answer=sample.answer,
                        contexts=sample.contexts,
                    )

            run.status = EvaluationStatus.COMPLETED

        except Exception as e:
            run.status = EvaluationStatus.FAILED
            run.errors.append(str(e))
            logger.error(f"Evaluation run failed: {e}")

        finally:
            run.completed_at = time.time()

        return run

    def generate_report(
        self,
        run: EvaluationRun,
        include_detailed_results: bool = False,
    ) -> EvaluationReport:
        """평가 리포트 생성"""
        report = EvaluationReport(run_id=run.id)

        # RAGAS 요약
        if run.ragas_results:
            report.ragas_summary = self.ragas_evaluator.aggregate_metrics(
                run.ragas_results
            )

        # Human 평가 요약
        if self.human_evaluator:
            report.human_summary = self.human_evaluator.get_evaluation_summary()

        # 권장사항 생성
        report.recommendations = self._generate_recommendations(report)

        # 상세 결과 (옵션)
        if include_detailed_results:
            report.detailed_results = [
                m.to_dict() for m in run.ragas_results
            ]

        self._reports[report.id] = report
        logger.info(f"Generated report: {report.id}")
        return report

    def _generate_recommendations(
        self,
        report: EvaluationReport,
    ) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []

        if not report.ragas_summary:
            return recommendations

        # 충실도가 낮은 경우
        faithfulness = report.ragas_summary.get("faithfulness", {})
        if isinstance(faithfulness, dict) and faithfulness.get("mean", 1) < 0.7:
            recommendations.append(
                "Faithfulness가 낮습니다. 검색된 컨텍스트만을 사용하여 답변을 생성하도록 "
                "프롬프트를 개선하거나, 더 관련성 높은 문서를 검색하세요."
            )

        # 답변 관련성이 낮은 경우
        relevancy = report.ragas_summary.get("answer_relevancy", {})
        if isinstance(relevancy, dict) and relevancy.get("mean", 1) < 0.7:
            recommendations.append(
                "Answer Relevancy가 낮습니다. 질문의 의도를 더 정확하게 파악하고 "
                "직접적인 답변을 생성하도록 개선하세요."
            )

        # 컨텍스트 정밀도가 낮은 경우
        precision = report.ragas_summary.get("context_precision", {})
        if isinstance(precision, dict) and precision.get("mean", 1) < 0.6:
            recommendations.append(
                "Context Precision이 낮습니다. 검색 알고리즘을 개선하거나 "
                "Re-ranking 적용을 고려하세요."
            )

        # 컨텍스트 재현율이 낮은 경우
        recall = report.ragas_summary.get("context_recall", {})
        if isinstance(recall, dict) and recall.get("mean", 1) < 0.6:
            recommendations.append(
                "Context Recall이 낮습니다. 검색할 문서 수(top-k)를 늘리거나 "
                "쿼리 확장 기법을 적용하세요."
            )

        # 전체적으로 좋은 경우
        overall = report.ragas_summary.get("overall_score", {})
        if isinstance(overall, dict) and overall.get("mean", 0) >= 0.8:
            recommendations.append(
                "전반적인 성능이 우수합니다. "
                "현재 설정을 프로덕션 환경에 배포하는 것을 권장합니다."
            )

        return recommendations

    def compare_runs(
        self,
        run_id_a: str,
        run_id_b: str,
    ) -> Dict[str, Any]:
        """두 실행 비교"""
        run_a = self._runs.get(run_id_a)
        run_b = self._runs.get(run_id_b)

        if not run_a or not run_b:
            return {"error": "Run not found"}

        return self.comparison_analyzer.compare_runs(run_a, run_b)

    def get_run(self, run_id: str) -> Optional[EvaluationRun]:
        """실행 조회"""
        return self._runs.get(run_id)

    def list_runs(
        self,
        status: Optional[EvaluationStatus] = None,
    ) -> List[EvaluationRun]:
        """실행 목록"""
        runs = list(self._runs.values())
        if status:
            runs = [r for r in runs if r.status == status]
        return sorted(runs, key=lambda r: r.started_at or 0, reverse=True)

    def get_report(self, report_id: str) -> Optional[EvaluationReport]:
        """리포트 조회"""
        return self._reports.get(report_id)

    def export_results(
        self,
        run_id: str,
        format: str = "json",
    ) -> str:
        """결과 내보내기"""
        run = self._runs.get(run_id)
        if not run:
            return ""

        if format == "json":
            data = {
                "run": run.to_dict(),
                "ragas_results": [m.to_dict() for m in run.ragas_results],
            }
            return json.dumps(data, indent=2, ensure_ascii=False)

        elif format == "csv":
            lines = ["id,faithfulness,answer_relevancy,context_precision,context_recall,overall_score"]
            for i, m in enumerate(run.ragas_results):
                lines.append(
                    f"{i},{m.faithfulness},{m.answer_relevancy},"
                    f"{m.context_precision},{m.context_recall},{m.overall_score}"
                )
            return "\n".join(lines)

        return ""
