# -*- coding: utf-8 -*-
"""
A/B 테스트 프레임워크

RAG 구성 비교 실험을 위한 프레임워크입니다.
"""

import hashlib
import json
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


class VariantType(Enum):
    """변형 유형"""
    CONTROL = "control"  # 대조군
    TREATMENT = "treatment"  # 실험군


@dataclass
class ExperimentConfig:
    """실험 설정

    Attributes:
        experiment_id: 실험 ID
        name: 실험 이름
        description: 실험 설명
        traffic_split: 트래픽 분배 (treatment 비율, 0.0~1.0)
        start_time: 시작 시간
        end_time: 종료 시간 (None이면 무기한)
        metrics: 측정할 메트릭 목록
    """
    experiment_id: str
    name: str
    description: str = ""
    traffic_split: float = 0.5  # 50% treatment
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metrics: list[str] = field(default_factory=lambda: [
        "latency_ms",
        "relevance_score",
        "user_satisfaction",
    ])

    def is_active(self) -> bool:
        """실험 활성 상태 확인"""
        now = datetime.now()
        if now < self.start_time:
            return False
        if self.end_time and now >= self.end_time:
            return False
        return True


@dataclass
class Variant:
    """실험 변형

    Attributes:
        variant_id: 변형 ID
        variant_type: 변형 유형 (control/treatment)
        config: 변형 설정 (RAG 파라미터)
        description: 설명
    """
    variant_id: str
    variant_type: VariantType
    config: dict[str, Any]
    description: str = ""


@dataclass
class ExperimentResult:
    """실험 결과

    Attributes:
        experiment_id: 실험 ID
        variant_id: 변형 ID
        user_id: 사용자 ID
        query: 쿼리
        response: 응답
        metrics: 측정된 메트릭
        timestamp: 타임스탬프
    """
    experiment_id: str
    variant_id: str
    user_id: str
    query: str
    response: Any
    metrics: dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "experiment_id": self.experiment_id,
            "variant_id": self.variant_id,
            "user_id": self.user_id,
            "query": self.query,
            "response": str(self.response)[:500],  # 응답 길이 제한
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ExperimentSummary:
    """실험 요약 통계

    Attributes:
        experiment_id: 실험 ID
        total_requests: 총 요청 수
        control_count: 대조군 요청 수
        treatment_count: 실험군 요청 수
        control_metrics: 대조군 메트릭 평균
        treatment_metrics: 실험군 메트릭 평균
        improvement: 개선율 (treatment vs control)
    """
    experiment_id: str
    total_requests: int
    control_count: int
    treatment_count: int
    control_metrics: dict[str, float]
    treatment_metrics: dict[str, float]
    improvement: dict[str, float] = field(default_factory=dict)

    def calculate_improvement(self) -> None:
        """개선율 계산"""
        for metric in self.control_metrics:
            control_val = self.control_metrics.get(metric, 0)
            treatment_val = self.treatment_metrics.get(metric, 0)

            if control_val > 0:
                self.improvement[metric] = (
                    (treatment_val - control_val) / control_val * 100
                )
            else:
                self.improvement[metric] = 0.0


class ABTestManager:
    """A/B 테스트 관리자

    실험 생성, 변형 할당, 결과 수집을 관리합니다.
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
    ):
        """
        Args:
            storage_path: 결과 저장 경로
        """
        self.storage_path = Path(storage_path) if storage_path else None

        self._experiments: dict[str, ExperimentConfig] = {}
        self._variants: dict[str, list[Variant]] = {}  # experiment_id -> variants
        self._results: dict[str, list[ExperimentResult]] = {}  # experiment_id -> results
        self._user_assignments: dict[str, dict[str, str]] = {}  # user_id -> {exp_id -> variant_id}

    def create_experiment(
        self,
        name: str,
        control_config: dict[str, Any],
        treatment_config: dict[str, Any],
        traffic_split: float = 0.5,
        description: str = "",
        metrics: Optional[list[str]] = None,
    ) -> ExperimentConfig:
        """실험 생성

        Args:
            name: 실험 이름
            control_config: 대조군 설정
            treatment_config: 실험군 설정
            traffic_split: 트래픽 분배 비율
            description: 설명
            metrics: 측정할 메트릭 목록

        Returns:
            실험 설정
        """
        experiment_id = str(uuid.uuid4())[:8]

        config = ExperimentConfig(
            experiment_id=experiment_id,
            name=name,
            description=description,
            traffic_split=traffic_split,
            metrics=metrics or ["latency_ms", "relevance_score"],
        )

        # 변형 생성
        control = Variant(
            variant_id=f"{experiment_id}_control",
            variant_type=VariantType.CONTROL,
            config=control_config,
            description="Control group (baseline)",
        )

        treatment = Variant(
            variant_id=f"{experiment_id}_treatment",
            variant_type=VariantType.TREATMENT,
            config=treatment_config,
            description="Treatment group (experimental)",
        )

        self._experiments[experiment_id] = config
        self._variants[experiment_id] = [control, treatment]
        self._results[experiment_id] = []

        return config

    def get_variant(
        self,
        experiment_id: str,
        user_id: str,
    ) -> Optional[Variant]:
        """사용자에게 변형 할당

        동일 사용자는 항상 같은 변형을 받습니다 (일관성 보장).

        Args:
            experiment_id: 실험 ID
            user_id: 사용자 ID

        Returns:
            할당된 변형 (None이면 실험 없음)
        """
        if experiment_id not in self._experiments:
            return None

        experiment = self._experiments[experiment_id]
        if not experiment.is_active():
            return None

        variants = self._variants[experiment_id]

        # 이미 할당된 경우 기존 할당 반환
        if user_id in self._user_assignments:
            if experiment_id in self._user_assignments[user_id]:
                variant_id = self._user_assignments[user_id][experiment_id]
                return next(
                    (v for v in variants if v.variant_id == variant_id),
                    None
                )

        # 새로운 할당 (해시 기반으로 일관성 보장)
        hash_input = f"{experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        normalized = (hash_value % 1000) / 1000.0

        if normalized < experiment.traffic_split:
            variant = next(v for v in variants if v.variant_type == VariantType.TREATMENT)
        else:
            variant = next(v for v in variants if v.variant_type == VariantType.CONTROL)

        # 할당 저장
        if user_id not in self._user_assignments:
            self._user_assignments[user_id] = {}
        self._user_assignments[user_id][experiment_id] = variant.variant_id

        return variant

    def record_result(
        self,
        experiment_id: str,
        variant_id: str,
        user_id: str,
        query: str,
        response: Any,
        metrics: dict[str, float],
    ) -> ExperimentResult:
        """실험 결과 기록

        Args:
            experiment_id: 실험 ID
            variant_id: 변형 ID
            user_id: 사용자 ID
            query: 쿼리
            response: 응답
            metrics: 측정된 메트릭

        Returns:
            실험 결과
        """
        result = ExperimentResult(
            experiment_id=experiment_id,
            variant_id=variant_id,
            user_id=user_id,
            query=query,
            response=response,
            metrics=metrics,
        )

        if experiment_id in self._results:
            self._results[experiment_id].append(result)

        # 파일 저장 (선택)
        if self.storage_path:
            self._save_result(result)

        return result

    def get_summary(self, experiment_id: str) -> Optional[ExperimentSummary]:
        """실험 요약 통계 조회

        Args:
            experiment_id: 실험 ID

        Returns:
            실험 요약 (None이면 실험 없음)
        """
        if experiment_id not in self._results:
            return None

        results = self._results[experiment_id]
        if not results:
            return ExperimentSummary(
                experiment_id=experiment_id,
                total_requests=0,
                control_count=0,
                treatment_count=0,
                control_metrics={},
                treatment_metrics={},
            )

        # 변형별 결과 분리
        control_results = [
            r for r in results
            if r.variant_id.endswith("_control")
        ]
        treatment_results = [
            r for r in results
            if r.variant_id.endswith("_treatment")
        ]

        # 메트릭 평균 계산
        control_metrics = self._calculate_avg_metrics(control_results)
        treatment_metrics = self._calculate_avg_metrics(treatment_results)

        summary = ExperimentSummary(
            experiment_id=experiment_id,
            total_requests=len(results),
            control_count=len(control_results),
            treatment_count=len(treatment_results),
            control_metrics=control_metrics,
            treatment_metrics=treatment_metrics,
        )
        summary.calculate_improvement()

        return summary

    def _calculate_avg_metrics(
        self,
        results: list[ExperimentResult],
    ) -> dict[str, float]:
        """메트릭 평균 계산"""
        if not results:
            return {}

        # 모든 메트릭 키 수집
        all_metrics: dict[str, list[float]] = {}
        for result in results:
            for key, value in result.metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)

        # 평균 계산
        return {
            key: sum(values) / len(values)
            for key, values in all_metrics.items()
        }

    def _save_result(self, result: ExperimentResult) -> None:
        """결과를 파일에 저장"""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)

        file_path = self.storage_path / f"{result.experiment_id}_results.jsonl"
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    def list_experiments(self) -> list[dict[str, Any]]:
        """모든 실험 목록 조회"""
        return [
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "description": exp.description,
                "is_active": exp.is_active(),
                "traffic_split": exp.traffic_split,
                "result_count": len(self._results.get(exp.experiment_id, [])),
            }
            for exp in self._experiments.values()
        ]

    def stop_experiment(self, experiment_id: str) -> bool:
        """실험 중지

        Args:
            experiment_id: 실험 ID

        Returns:
            성공 여부
        """
        if experiment_id not in self._experiments:
            return False

        self._experiments[experiment_id].end_time = datetime.now()
        return True


class RAGExperiment:
    """RAG A/B 테스트 헬퍼

    RAG 시스템의 다양한 구성을 비교 실험합니다.
    """

    # 사전 정의된 실험 템플릿
    EXPERIMENT_TEMPLATES = {
        "reranker_comparison": {
            "name": "Re-ranker 비교",
            "control": {"reranker_type": "keyword"},
            "treatment": {"reranker_type": "cross-encoder"},
            "metrics": ["latency_ms", "relevance_score", "mrr"],
        },
        "chunk_size_comparison": {
            "name": "청크 크기 비교",
            "control": {"chunk_size": 256},
            "treatment": {"chunk_size": 512},
            "metrics": ["latency_ms", "relevance_score", "context_coverage"],
        },
        "hybrid_vs_vector": {
            "name": "Hybrid vs Vector Only",
            "control": {"search_type": "vector"},
            "treatment": {"search_type": "hybrid"},
            "metrics": ["latency_ms", "precision_at_5", "recall_at_5"],
        },
        "query_expansion": {
            "name": "Query Expansion 효과",
            "control": {"query_expansion": False},
            "treatment": {"query_expansion": True},
            "metrics": ["latency_ms", "recall_at_5", "relevance_score"],
        },
    }

    def __init__(
        self,
        ab_manager: Optional[ABTestManager] = None,
    ):
        """
        Args:
            ab_manager: A/B 테스트 관리자
        """
        self.ab_manager = ab_manager or ABTestManager()

    def create_from_template(
        self,
        template_name: str,
        traffic_split: float = 0.5,
    ) -> Optional[ExperimentConfig]:
        """템플릿에서 실험 생성

        Args:
            template_name: 템플릿 이름
            traffic_split: 트래픽 분배 비율

        Returns:
            실험 설정 (None이면 템플릿 없음)
        """
        if template_name not in self.EXPERIMENT_TEMPLATES:
            return None

        template = self.EXPERIMENT_TEMPLATES[template_name]

        return self.ab_manager.create_experiment(
            name=template["name"],
            control_config=template["control"],
            treatment_config=template["treatment"],
            traffic_split=traffic_split,
            metrics=template["metrics"],
        )

    def run_with_experiment(
        self,
        experiment_id: str,
        user_id: str,
        query: str,
        control_fn: Callable[[str], Any],
        treatment_fn: Callable[[str], Any],
    ) -> tuple[Any, dict[str, Any]]:
        """실험 적용하여 실행

        Args:
            experiment_id: 실험 ID
            user_id: 사용자 ID
            query: 쿼리
            control_fn: 대조군 실행 함수
            treatment_fn: 실험군 실행 함수

        Returns:
            (응답, 메타데이터)
        """
        variant = self.ab_manager.get_variant(experiment_id, user_id)

        if variant is None:
            # 실험 없음, 기본 실행
            response = control_fn(query)
            return response, {"variant": None}

        start_time = time.time()

        # 변형에 따라 실행
        if variant.variant_type == VariantType.CONTROL:
            response = control_fn(query)
        else:
            response = treatment_fn(query)

        latency_ms = (time.time() - start_time) * 1000

        # 결과 기록
        self.ab_manager.record_result(
            experiment_id=experiment_id,
            variant_id=variant.variant_id,
            user_id=user_id,
            query=query,
            response=response,
            metrics={"latency_ms": latency_ms},
        )

        return response, {
            "variant": variant.variant_type.value,
            "variant_id": variant.variant_id,
            "latency_ms": latency_ms,
        }


# 전역 A/B 테스트 관리자
_manager: Optional[ABTestManager] = None


def get_ab_manager() -> ABTestManager:
    """전역 A/B 테스트 관리자 반환"""
    global _manager
    if _manager is None:
        _manager = ABTestManager()
    return _manager


def create_experiment(
    name: str,
    control_config: dict[str, Any],
    treatment_config: dict[str, Any],
    traffic_split: float = 0.5,
) -> ExperimentConfig:
    """실험 생성 (편의 함수)"""
    manager = get_ab_manager()
    return manager.create_experiment(
        name=name,
        control_config=control_config,
        treatment_config=treatment_config,
        traffic_split=traffic_split,
    )


def get_variant(experiment_id: str, user_id: str) -> Optional[Variant]:
    """변형 할당 (편의 함수)"""
    manager = get_ab_manager()
    return manager.get_variant(experiment_id, user_id)
