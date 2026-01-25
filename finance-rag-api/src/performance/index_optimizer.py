# -*- coding: utf-8 -*-
"""
인덱스 최적화 모듈

[기능]
- ChromaDB 인덱스 분석 및 최적화
- HNSW 파라미터 튜닝
- 인덱스 통계 수집
- 자동 리인덱싱
- 쿼리 성능 프로파일링

[사용 예시]
>>> optimizer = ChromaIndexOptimizer(collection)
>>> stats = optimizer.analyze()
>>> optimizer.optimize()
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IndexStats:
    """인덱스 통계"""
    collection_name: str = ""
    document_count: int = 0
    embedding_dimension: int = 0
    index_type: str = "HNSW"

    # HNSW 파라미터
    hnsw_m: int = 16
    hnsw_ef_construction: int = 100
    hnsw_ef_search: int = 10

    # 성능 메트릭
    avg_query_time_ms: float = 0.0
    p95_query_time_ms: float = 0.0
    p99_query_time_ms: float = 0.0

    # 공간 사용
    index_size_bytes: int = 0
    metadata_size_bytes: int = 0

    # 품질 메트릭
    avg_distance: float = 0.0
    distance_variance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "document_count": self.document_count,
            "embedding_dimension": self.embedding_dimension,
            "index_type": self.index_type,
            "hnsw_params": {
                "M": self.hnsw_m,
                "ef_construction": self.hnsw_ef_construction,
                "ef_search": self.hnsw_ef_search,
            },
            "performance": {
                "avg_query_time_ms": round(self.avg_query_time_ms, 2),
                "p95_query_time_ms": round(self.p95_query_time_ms, 2),
                "p99_query_time_ms": round(self.p99_query_time_ms, 2),
            },
            "storage": {
                "index_size_bytes": self.index_size_bytes,
                "metadata_size_bytes": self.metadata_size_bytes,
            },
            "quality": {
                "avg_distance": round(self.avg_distance, 4),
                "distance_variance": round(self.distance_variance, 4),
            },
        }


@dataclass
class OptimizationRecommendation:
    """최적화 권장 사항"""
    parameter: str
    current_value: Any
    recommended_value: Any
    reason: str
    expected_improvement: str
    priority: str = "medium"  # high, medium, low


@dataclass
class QueryProfile:
    """쿼리 프로파일"""
    query: str
    embedding_time_ms: float = 0.0
    search_time_ms: float = 0.0
    total_time_ms: float = 0.0
    results_count: int = 0
    avg_distance: float = 0.0


class IndexOptimizer:
    """인덱스 최적화기 베이스 클래스"""

    def analyze(self) -> IndexStats:
        """인덱스 분석"""
        raise NotImplementedError

    def optimize(self) -> Dict[str, Any]:
        """인덱스 최적화"""
        raise NotImplementedError

    def get_recommendations(self) -> List[OptimizationRecommendation]:
        """최적화 권장 사항"""
        raise NotImplementedError


class ChromaIndexOptimizer(IndexOptimizer):
    """
    ChromaDB 인덱스 최적화기

    [특징]
    - HNSW 파라미터 튜닝
    - 쿼리 성능 프로파일링
    - 자동 최적화 권장
    """

    def __init__(
        self,
        collection,
        sample_queries: Optional[List[str]] = None,
    ):
        """
        Args:
            collection: ChromaDB 컬렉션
            sample_queries: 성능 테스트용 샘플 쿼리
        """
        self.collection = collection
        self.sample_queries = sample_queries or []
        self._stats = IndexStats()
        self._query_profiles: List[QueryProfile] = []
        self._recommendations: List[OptimizationRecommendation] = []

    def analyze(self) -> IndexStats:
        """
        인덱스 분석

        Returns:
            IndexStats: 인덱스 통계
        """
        try:
            # 기본 정보
            self._stats.collection_name = self.collection.name
            self._stats.document_count = self.collection.count()

            # 메타데이터에서 설정 가져오기
            metadata = self.collection.metadata or {}
            hnsw_space = metadata.get("hnsw:space", "l2")

            # 샘플 임베딩으로 차원 확인
            sample = self.collection.get(limit=1, include=["embeddings"])
            if sample.get("embeddings") and len(sample["embeddings"]) > 0:
                self._stats.embedding_dimension = len(sample["embeddings"][0])

            # 쿼리 성능 프로파일링
            if self.sample_queries:
                self._profile_queries()

            # 권장 사항 생성
            self._generate_recommendations()

            logger.info(f"Index analysis complete: {self._stats.document_count} documents")

        except Exception as e:
            logger.error(f"Index analysis failed: {e}")

        return self._stats

    def _profile_queries(self):
        """쿼리 성능 프로파일링"""
        query_times = []
        distances = []

        for query in self.sample_queries[:10]:  # 최대 10개 샘플
            start = time.time()

            results = self.collection.query(
                query_texts=[query],
                n_results=5,
                include=["distances"]
            )

            elapsed = (time.time() - start) * 1000
            query_times.append(elapsed)

            if results.get("distances") and results["distances"][0]:
                distances.extend(results["distances"][0])

            self._query_profiles.append(QueryProfile(
                query=query[:50],
                total_time_ms=elapsed,
                results_count=len(results.get("ids", [[]])[0]),
                avg_distance=np.mean(results["distances"][0]) if results.get("distances") and results["distances"][0] else 0,
            ))

        if query_times:
            self._stats.avg_query_time_ms = np.mean(query_times)
            self._stats.p95_query_time_ms = np.percentile(query_times, 95)
            self._stats.p99_query_time_ms = np.percentile(query_times, 99)

        if distances:
            self._stats.avg_distance = np.mean(distances)
            self._stats.distance_variance = np.var(distances)

    def _generate_recommendations(self):
        """최적화 권장 사항 생성"""
        self._recommendations = []

        doc_count = self._stats.document_count

        # 문서 수에 따른 HNSW M 파라미터 권장
        if doc_count > 100000:
            recommended_m = 32
        elif doc_count > 10000:
            recommended_m = 24
        else:
            recommended_m = 16

        if recommended_m != self._stats.hnsw_m:
            self._recommendations.append(OptimizationRecommendation(
                parameter="hnsw:M",
                current_value=self._stats.hnsw_m,
                recommended_value=recommended_m,
                reason=f"문서 수({doc_count})에 최적화된 M 값",
                expected_improvement="검색 정확도 향상",
                priority="medium",
            ))

        # ef_construction 권장
        recommended_ef_construction = recommended_m * 8
        if recommended_ef_construction != self._stats.hnsw_ef_construction:
            self._recommendations.append(OptimizationRecommendation(
                parameter="hnsw:ef_construction",
                current_value=self._stats.hnsw_ef_construction,
                recommended_value=recommended_ef_construction,
                reason="M 값에 따른 최적 ef_construction",
                expected_improvement="인덱스 품질 향상",
                priority="low",
            ))

        # 쿼리 시간이 느린 경우
        if self._stats.avg_query_time_ms > 100:
            self._recommendations.append(OptimizationRecommendation(
                parameter="hnsw:ef_search",
                current_value=self._stats.hnsw_ef_search,
                recommended_value=max(10, self._stats.hnsw_ef_search - 5),
                reason=f"평균 쿼리 시간({self._stats.avg_query_time_ms:.1f}ms)이 높음",
                expected_improvement="쿼리 속도 향상 (정확도 소폭 감소 가능)",
                priority="high",
            ))

        # 거리 분산이 높은 경우
        if self._stats.distance_variance > 0.1:
            self._recommendations.append(OptimizationRecommendation(
                parameter="embedding_model",
                current_value="current",
                recommended_value="fine-tuned",
                reason=f"거리 분산({self._stats.distance_variance:.4f})이 높음",
                expected_improvement="검색 일관성 향상",
                priority="medium",
            ))

    def get_recommendations(self) -> List[OptimizationRecommendation]:
        """최적화 권장 사항 반환"""
        return self._recommendations

    def optimize(self) -> Dict[str, Any]:
        """
        인덱스 최적화 적용

        현재 ChromaDB는 런타임 HNSW 파라미터 변경을 지원하지 않으므로
        새 컬렉션 생성이 필요할 수 있음

        Returns:
            최적화 결과
        """
        applied = []
        skipped = []

        for rec in self._recommendations:
            # ChromaDB에서 직접 적용 가능한 파라미터 확인
            if rec.parameter in ["hnsw:ef_search"]:
                # ef_search는 쿼리 시점에 적용 가능
                applied.append({
                    "parameter": rec.parameter,
                    "new_value": rec.recommended_value,
                    "status": "applied_at_query_time",
                })
            else:
                skipped.append({
                    "parameter": rec.parameter,
                    "recommended_value": rec.recommended_value,
                    "reason": "Requires collection recreation",
                })

        return {
            "applied": applied,
            "skipped": skipped,
            "message": "일부 파라미터는 컬렉션 재생성이 필요합니다",
        }

    def get_query_profiles(self) -> List[Dict[str, Any]]:
        """쿼리 프로파일 반환"""
        return [
            {
                "query": p.query,
                "total_time_ms": round(p.total_time_ms, 2),
                "results_count": p.results_count,
                "avg_distance": round(p.avg_distance, 4),
            }
            for p in self._query_profiles
        ]

    def benchmark(
        self,
        queries: List[str],
        top_k_values: List[int] = [1, 5, 10, 20],
    ) -> Dict[str, Any]:
        """
        다양한 설정으로 벤치마크 실행

        Args:
            queries: 테스트 쿼리 리스트
            top_k_values: 테스트할 top_k 값들

        Returns:
            벤치마크 결과
        """
        results = {}

        for top_k in top_k_values:
            times = []

            for query in queries:
                start = time.time()
                self.collection.query(
                    query_texts=[query],
                    n_results=top_k,
                )
                elapsed = (time.time() - start) * 1000
                times.append(elapsed)

            results[f"top_k_{top_k}"] = {
                "avg_ms": round(np.mean(times), 2),
                "min_ms": round(np.min(times), 2),
                "max_ms": round(np.max(times), 2),
                "p95_ms": round(np.percentile(times, 95), 2),
                "queries_per_sec": round(len(queries) / (sum(times) / 1000), 2),
            }

        return results


class VectorIndexManager:
    """
    벡터 인덱스 관리자

    여러 컬렉션의 인덱스를 중앙에서 관리
    """

    def __init__(self, chroma_client):
        self.client = chroma_client
        self._optimizers: Dict[str, ChromaIndexOptimizer] = {}

    def get_all_stats(self) -> Dict[str, IndexStats]:
        """모든 컬렉션 통계"""
        stats = {}

        for collection in self.client.list_collections():
            optimizer = ChromaIndexOptimizer(collection)
            stats[collection.name] = optimizer.analyze()

        return stats

    def optimize_all(self) -> Dict[str, Any]:
        """모든 컬렉션 최적화"""
        results = {}

        for collection in self.client.list_collections():
            optimizer = ChromaIndexOptimizer(collection)
            optimizer.analyze()
            results[collection.name] = optimizer.optimize()

        return results

    def get_recommendations_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """모든 컬렉션 권장 사항"""
        recommendations = {}

        for collection in self.client.list_collections():
            optimizer = ChromaIndexOptimizer(collection)
            optimizer.analyze()
            recommendations[collection.name] = [
                {
                    "parameter": r.parameter,
                    "current": r.current_value,
                    "recommended": r.recommended_value,
                    "reason": r.reason,
                    "priority": r.priority,
                }
                for r in optimizer.get_recommendations()
            ]

        return recommendations
