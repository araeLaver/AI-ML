# -*- coding: utf-8 -*-
"""
과금/사용량 추적 모듈

[기능]
- 사용량 측정
- 쿼터 관리
- 과금 계산
- 사용 리포트
"""

import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QuotaExceededError(Exception):
    """쿼터 초과 오류"""

    def __init__(
        self,
        message: str,
        resource: str,
        limit: int,
        used: int,
        reset_at: Optional[datetime] = None,
    ):
        super().__init__(message)
        self.resource = resource
        self.limit = limit
        self.used = used
        self.reset_at = reset_at


class ResourceType(Enum):
    """리소스 유형"""
    API_CALL = "api_call"
    RAG_QUERY = "rag_query"
    DOCUMENT_INDEX = "document_index"
    STORAGE_MB = "storage_mb"
    TOKEN_INPUT = "token_input"
    TOKEN_OUTPUT = "token_output"


class BillingPlan(Enum):
    """과금 플랜"""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass
class PlanLimits:
    """플랜별 제한"""
    api_calls_per_month: int
    rag_queries_per_month: int
    documents_per_month: int
    storage_mb: int
    tokens_per_month: int
    price_per_month: float  # USD
    overage_rate: Dict[ResourceType, float] = field(default_factory=dict)


# 플랜별 제한 정의
PLAN_LIMITS = {
    BillingPlan.FREE: PlanLimits(
        api_calls_per_month=1000,
        rag_queries_per_month=100,
        documents_per_month=50,
        storage_mb=100,
        tokens_per_month=100000,
        price_per_month=0.0,
        overage_rate={},  # 초과 불가
    ),
    BillingPlan.STARTER: PlanLimits(
        api_calls_per_month=10000,
        rag_queries_per_month=1000,
        documents_per_month=500,
        storage_mb=1000,
        tokens_per_month=1000000,
        price_per_month=29.0,
        overage_rate={
            ResourceType.API_CALL: 0.001,
            ResourceType.RAG_QUERY: 0.01,
            ResourceType.TOKEN_INPUT: 0.00001,
            ResourceType.TOKEN_OUTPUT: 0.00003,
        },
    ),
    BillingPlan.PROFESSIONAL: PlanLimits(
        api_calls_per_month=100000,
        rag_queries_per_month=10000,
        documents_per_month=5000,
        storage_mb=10000,
        tokens_per_month=10000000,
        price_per_month=99.0,
        overage_rate={
            ResourceType.API_CALL: 0.0008,
            ResourceType.RAG_QUERY: 0.008,
            ResourceType.TOKEN_INPUT: 0.000008,
            ResourceType.TOKEN_OUTPUT: 0.000024,
        },
    ),
    BillingPlan.ENTERPRISE: PlanLimits(
        api_calls_per_month=1000000,
        rag_queries_per_month=100000,
        documents_per_month=50000,
        storage_mb=100000,
        tokens_per_month=100000000,
        price_per_month=499.0,
        overage_rate={
            ResourceType.API_CALL: 0.0005,
            ResourceType.RAG_QUERY: 0.005,
            ResourceType.TOKEN_INPUT: 0.000005,
            ResourceType.TOKEN_OUTPUT: 0.000015,
        },
    ),
}


@dataclass
class UsageRecord:
    """사용량 레코드"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    resource_type: ResourceType = ResourceType.API_CALL
    quantity: int = 1
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "resource_type": self.resource_type.value,
            "quantity": self.quantity,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class UsageSummary:
    """사용량 요약"""
    user_id: str
    period_start: datetime
    period_end: datetime
    usage: Dict[ResourceType, int] = field(default_factory=dict)
    limits: Dict[ResourceType, int] = field(default_factory=dict)

    def get_remaining(self, resource: ResourceType) -> int:
        """남은 사용량"""
        used = self.usage.get(resource, 0)
        limit = self.limits.get(resource, 0)
        return max(0, limit - used)

    def get_usage_percent(self, resource: ResourceType) -> float:
        """사용률"""
        used = self.usage.get(resource, 0)
        limit = self.limits.get(resource, 1)
        return (used / limit) * 100

    def is_exceeded(self, resource: ResourceType) -> bool:
        """초과 여부"""
        return self.get_remaining(resource) <= 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "usage": {k.value: v for k, v in self.usage.items()},
            "limits": {k.value: v for k, v in self.limits.items()},
            "remaining": {
                k.value: self.get_remaining(k) for k in self.usage.keys()
            },
        }


class UsageTracker:
    """사용량 추적기"""

    def __init__(self):
        # user_id -> [UsageRecord]
        self._records: Dict[str, List[UsageRecord]] = defaultdict(list)
        # user_id -> BillingPlan
        self._plans: Dict[str, BillingPlan] = {}

    def set_plan(self, user_id: str, plan: BillingPlan) -> None:
        """사용자 플랜 설정"""
        self._plans[user_id] = plan

    def get_plan(self, user_id: str) -> BillingPlan:
        """사용자 플랜 조회"""
        return self._plans.get(user_id, BillingPlan.FREE)

    def record(
        self,
        user_id: str,
        resource_type: ResourceType,
        quantity: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UsageRecord:
        """사용량 기록"""
        record = UsageRecord(
            user_id=user_id,
            resource_type=resource_type,
            quantity=quantity,
            metadata=metadata or {},
        )
        self._records[user_id].append(record)

        logger.debug(
            f"Usage recorded: {user_id} - {resource_type.value} x {quantity}"
        )
        return record

    def get_usage(
        self,
        user_id: str,
        resource_type: Optional[ResourceType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """기간별 사용량 조회"""
        records = self._records.get(user_id, [])

        total = 0
        for record in records:
            if resource_type and record.resource_type != resource_type:
                continue
            if start_date and record.timestamp < start_date:
                continue
            if end_date and record.timestamp > end_date:
                continue
            total += record.quantity

        return total

    def get_monthly_summary(
        self,
        user_id: str,
        year: Optional[int] = None,
        month: Optional[int] = None,
    ) -> UsageSummary:
        """월간 사용량 요약"""
        now = datetime.utcnow()
        year = year or now.year
        month = month or now.month

        period_start = datetime(year, month, 1)
        if month == 12:
            period_end = datetime(year + 1, 1, 1)
        else:
            period_end = datetime(year, month + 1, 1)

        plan = self.get_plan(user_id)
        plan_limits = PLAN_LIMITS[plan]

        usage = {}
        limits = {
            ResourceType.API_CALL: plan_limits.api_calls_per_month,
            ResourceType.RAG_QUERY: plan_limits.rag_queries_per_month,
            ResourceType.DOCUMENT_INDEX: plan_limits.documents_per_month,
            ResourceType.STORAGE_MB: plan_limits.storage_mb,
            ResourceType.TOKEN_INPUT: plan_limits.tokens_per_month,
            ResourceType.TOKEN_OUTPUT: plan_limits.tokens_per_month,
        }

        for resource_type in ResourceType:
            usage[resource_type] = self.get_usage(
                user_id,
                resource_type,
                period_start,
                period_end,
            )

        return UsageSummary(
            user_id=user_id,
            period_start=period_start,
            period_end=period_end,
            usage=usage,
            limits=limits,
        )

    def get_records(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[UsageRecord]:
        """사용량 레코드 조회"""
        records = self._records.get(user_id, [])
        # 최신순 정렬
        sorted_records = sorted(
            records,
            key=lambda r: r.timestamp,
            reverse=True,
        )
        return sorted_records[offset:offset + limit]


class QuotaManager:
    """쿼터 관리자"""

    def __init__(
        self,
        usage_tracker: UsageTracker,
        allow_overage: bool = False,
    ):
        self.usage_tracker = usage_tracker
        self.allow_overage = allow_overage

    def check_quota(
        self,
        user_id: str,
        resource_type: ResourceType,
        quantity: int = 1,
    ) -> Tuple[bool, int]:
        """쿼터 확인

        Returns:
            (allowed, remaining)
        """
        summary = self.usage_tracker.get_monthly_summary(user_id)

        remaining = summary.get_remaining(resource_type)
        allowed = remaining >= quantity

        if not allowed and self.allow_overage:
            plan = self.usage_tracker.get_plan(user_id)
            plan_limits = PLAN_LIMITS[plan]
            # 초과 요금이 설정된 경우 허용
            allowed = resource_type in plan_limits.overage_rate

        return allowed, remaining

    def consume(
        self,
        user_id: str,
        resource_type: ResourceType,
        quantity: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UsageRecord:
        """쿼터 소비"""
        allowed, remaining = self.check_quota(user_id, resource_type, quantity)

        if not allowed:
            summary = self.usage_tracker.get_monthly_summary(user_id)
            raise QuotaExceededError(
                f"Quota exceeded for {resource_type.value}",
                resource=resource_type.value,
                limit=summary.limits.get(resource_type, 0),
                used=summary.usage.get(resource_type, 0),
                reset_at=summary.period_end,
            )

        return self.usage_tracker.record(
            user_id=user_id,
            resource_type=resource_type,
            quantity=quantity,
            metadata=metadata,
        )

    def get_quota_status(
        self,
        user_id: str,
    ) -> Dict[str, Any]:
        """쿼터 상태 조회"""
        summary = self.usage_tracker.get_monthly_summary(user_id)
        plan = self.usage_tracker.get_plan(user_id)

        return {
            "plan": plan.value,
            "period": {
                "start": summary.period_start.isoformat(),
                "end": summary.period_end.isoformat(),
            },
            "quotas": {
                resource.value: {
                    "used": summary.usage.get(resource, 0),
                    "limit": summary.limits.get(resource, 0),
                    "remaining": summary.get_remaining(resource),
                    "usage_percent": round(summary.get_usage_percent(resource), 2),
                }
                for resource in ResourceType
            },
        }


@dataclass
class Invoice:
    """청구서"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    period_start: datetime = field(default_factory=datetime.utcnow)
    period_end: datetime = field(default_factory=datetime.utcnow)
    plan: BillingPlan = BillingPlan.FREE
    base_amount: float = 0.0
    overage_amount: float = 0.0
    total_amount: float = 0.0
    line_items: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
            "plan": self.plan.value,
            "base_amount": self.base_amount,
            "overage_amount": self.overage_amount,
            "total_amount": self.total_amount,
            "line_items": self.line_items,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
        }


class BillingCalculator:
    """과금 계산기"""

    def __init__(self, usage_tracker: UsageTracker):
        self.usage_tracker = usage_tracker

    def calculate_overage(
        self,
        user_id: str,
        year: int,
        month: int,
    ) -> Dict[ResourceType, Tuple[int, float]]:
        """초과 사용량 계산

        Returns:
            {resource_type: (overage_quantity, overage_cost)}
        """
        summary = self.usage_tracker.get_monthly_summary(user_id, year, month)
        plan = self.usage_tracker.get_plan(user_id)
        plan_limits = PLAN_LIMITS[plan]

        overages = {}

        for resource_type, used in summary.usage.items():
            limit = summary.limits.get(resource_type, 0)
            overage_qty = max(0, used - limit)

            if overage_qty > 0:
                rate = plan_limits.overage_rate.get(resource_type, 0)
                overage_cost = overage_qty * rate
                overages[resource_type] = (overage_qty, overage_cost)

        return overages

    def generate_invoice(
        self,
        user_id: str,
        year: int,
        month: int,
    ) -> Invoice:
        """청구서 생성"""
        plan = self.usage_tracker.get_plan(user_id)
        plan_limits = PLAN_LIMITS[plan]

        period_start = datetime(year, month, 1)
        if month == 12:
            period_end = datetime(year + 1, 1, 1)
        else:
            period_end = datetime(year, month + 1, 1)

        # 기본 요금
        base_amount = plan_limits.price_per_month
        line_items = [
            {
                "description": f"{plan.value.title()} Plan - Monthly",
                "quantity": 1,
                "unit_price": base_amount,
                "amount": base_amount,
            }
        ]

        # 초과 사용량
        overage_amount = 0.0
        overages = self.calculate_overage(user_id, year, month)

        for resource_type, (qty, cost) in overages.items():
            if cost > 0:
                overage_amount += cost
                line_items.append({
                    "description": f"Overage - {resource_type.value}",
                    "quantity": qty,
                    "unit_price": plan_limits.overage_rate.get(resource_type, 0),
                    "amount": cost,
                })

        total_amount = base_amount + overage_amount

        return Invoice(
            user_id=user_id,
            period_start=period_start,
            period_end=period_end,
            plan=plan,
            base_amount=base_amount,
            overage_amount=overage_amount,
            total_amount=total_amount,
            line_items=line_items,
        )


class BillingService:
    """과금 서비스"""

    def __init__(
        self,
        allow_overage: bool = True,
    ):
        self.usage_tracker = UsageTracker()
        self.quota_manager = QuotaManager(
            self.usage_tracker,
            allow_overage=allow_overage,
        )
        self.calculator = BillingCalculator(self.usage_tracker)

        # 청구서 저장
        self._invoices: Dict[str, Invoice] = {}

    def set_user_plan(self, user_id: str, plan: BillingPlan) -> None:
        """사용자 플랜 설정"""
        self.usage_tracker.set_plan(user_id, plan)
        logger.info(f"User {user_id} plan set to {plan.value}")

    def track_usage(
        self,
        user_id: str,
        resource_type: ResourceType,
        quantity: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UsageRecord:
        """사용량 추적"""
        return self.quota_manager.consume(
            user_id=user_id,
            resource_type=resource_type,
            quantity=quantity,
            metadata=metadata,
        )

    def check_quota(
        self,
        user_id: str,
        resource_type: ResourceType,
        quantity: int = 1,
    ) -> bool:
        """쿼터 확인"""
        allowed, _ = self.quota_manager.check_quota(
            user_id, resource_type, quantity
        )
        return allowed

    def get_usage_summary(
        self,
        user_id: str,
        year: Optional[int] = None,
        month: Optional[int] = None,
    ) -> UsageSummary:
        """사용량 요약"""
        return self.usage_tracker.get_monthly_summary(user_id, year, month)

    def get_quota_status(self, user_id: str) -> Dict[str, Any]:
        """쿼터 상태"""
        return self.quota_manager.get_quota_status(user_id)

    def generate_invoice(
        self,
        user_id: str,
        year: Optional[int] = None,
        month: Optional[int] = None,
    ) -> Invoice:
        """청구서 생성"""
        now = datetime.utcnow()
        year = year or now.year
        month = month or now.month

        invoice = self.calculator.generate_invoice(user_id, year, month)
        self._invoices[invoice.id] = invoice

        return invoice

    def get_invoice(self, invoice_id: str) -> Optional[Invoice]:
        """청구서 조회"""
        return self._invoices.get(invoice_id)

    def list_invoices(
        self,
        user_id: str,
        limit: int = 10,
    ) -> List[Invoice]:
        """청구서 목록"""
        invoices = [
            inv for inv in self._invoices.values()
            if inv.user_id == user_id
        ]
        return sorted(
            invoices,
            key=lambda i: i.created_at,
            reverse=True,
        )[:limit]


# 사용량 추적 컨텍스트 매니저
class UsageContext:
    """사용량 추적 컨텍스트"""

    def __init__(
        self,
        billing_service: BillingService,
        user_id: str,
        resource_type: ResourceType,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.billing_service = billing_service
        self.user_id = user_id
        self.resource_type = resource_type
        self.metadata = metadata or {}
        self._start_time: Optional[float] = None

    def __enter__(self) -> "UsageContext":
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:  # 성공적으로 완료된 경우만 기록
            duration_ms = (time.time() - self._start_time) * 1000
            self.metadata["duration_ms"] = duration_ms

            self.billing_service.track_usage(
                user_id=self.user_id,
                resource_type=self.resource_type,
                quantity=1,
                metadata=self.metadata,
            )
        return False


def track_usage(
    billing_service: BillingService,
    resource_type: ResourceType,
):
    """사용량 추적 데코레이터"""

    def decorator(func: Callable):
        def wrapper(user_id: str, *args, **kwargs):
            with UsageContext(billing_service, user_id, resource_type):
                return func(user_id, *args, **kwargs)
        return wrapper
    return decorator
