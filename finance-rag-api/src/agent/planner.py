# -*- coding: utf-8 -*-
"""
작업 계획 모듈

[기능]
- 복잡한 쿼리 분해
- 실행 계획 수립
- 의존성 관리
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class PlanStatus(Enum):
    """계획 상태"""
    DRAFT = "draft"
    READY = "ready"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepDependency(Enum):
    """단계 의존성 유형"""
    NONE = "none"  # 의존성 없음
    SEQUENTIAL = "sequential"  # 순차적
    PARALLEL = "parallel"  # 병렬 가능
    CONDITIONAL = "conditional"  # 조건부


@dataclass
class PlanStep:
    """계획 단계"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    action_type: str = "tool_call"  # tool_call, llm_call, decision
    action_params: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # 선행 단계 IDs
    status: str = "pending"  # pending, running, completed, failed, skipped
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    assigned_agent: Optional[str] = None
    priority: int = 0

    @property
    def duration_ms(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "action_type": self.action_type,
            "dependencies": self.dependencies,
            "status": self.status,
            "result": str(self.result)[:200] if self.result else None,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 2),
            "assigned_agent": self.assigned_agent,
        }


@dataclass
class Plan:
    """실행 계획"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    goal: str = ""
    steps: List[PlanStep] = field(default_factory=list)
    status: PlanStatus = PlanStatus.DRAFT
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def completed_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == "completed")

    @property
    def progress(self) -> float:
        return self.completed_steps / self.total_steps if self.total_steps > 0 else 0.0

    def add_step(
        self,
        name: str,
        description: str = "",
        action_type: str = "tool_call",
        action_params: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        priority: int = 0,
    ) -> PlanStep:
        """단계 추가"""
        step = PlanStep(
            name=name,
            description=description,
            action_type=action_type,
            action_params=action_params or {},
            dependencies=dependencies or [],
            priority=priority,
        )
        self.steps.append(step)
        return step

    def get_step(self, step_id: str) -> Optional[PlanStep]:
        """단계 조회"""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def get_ready_steps(self) -> List[PlanStep]:
        """실행 가능한 단계들 (의존성 충족)"""
        completed_ids = {s.id for s in self.steps if s.status == "completed"}
        ready = []

        for step in self.steps:
            if step.status != "pending":
                continue

            # 모든 의존성이 완료되었는지 확인
            if all(dep in completed_ids for dep in step.dependencies):
                ready.append(step)

        # 우선순위로 정렬
        ready.sort(key=lambda x: x.priority, reverse=True)
        return ready

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status.value,
            "progress": round(self.progress, 2),
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
        }


class TaskPlanner:
    """
    작업 계획자

    복잡한 쿼리를 실행 가능한 계획으로 변환
    """

    def __init__(
        self,
        llm_func: Optional[Callable] = None,
        available_tools: Optional[List[str]] = None,
    ):
        self.llm_func = llm_func or (lambda x: f"Plan: {x[:50]}")
        self.available_tools = available_tools or [
            "search_documents",
            "calculator",
            "data_lookup",
            "compare",
        ]
        self._plans: Dict[str, Plan] = {}
        self._templates: Dict[str, List[Dict[str, Any]]] = self._init_templates()

    def _init_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """계획 템플릿 초기화"""
        return {
            "comparison": [
                {"name": "검색1", "action": "search_documents", "priority": 2},
                {"name": "검색2", "action": "search_documents", "priority": 2},
                {"name": "비교분석", "action": "compare", "depends_on": ["검색1", "검색2"]},
                {"name": "답변생성", "action": "llm_call", "depends_on": ["비교분석"]},
            ],
            "calculation": [
                {"name": "데이터조회", "action": "search_documents", "priority": 2},
                {"name": "계산", "action": "calculator", "depends_on": ["데이터조회"]},
                {"name": "결과해석", "action": "llm_call", "depends_on": ["계산"]},
            ],
            "research": [
                {"name": "검색", "action": "search_documents", "priority": 2},
                {"name": "분석", "action": "llm_call", "depends_on": ["검색"]},
                {"name": "요약", "action": "llm_call", "depends_on": ["분석"]},
            ],
            "simple_qa": [
                {"name": "검색", "action": "search_documents"},
                {"name": "답변", "action": "llm_call", "depends_on": ["검색"]},
            ],
        }

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """쿼리 분석"""
        query_lower = query.lower()

        # 쿼리 유형 판별
        query_type = "simple_qa"
        keywords = {
            "comparison": ["비교", "vs", "차이", "대비"],
            "calculation": ["계산", "얼마", "몇", "수익률", "증감률"],
            "research": ["분석", "조사", "상세", "설명"],
        }

        for qtype, kws in keywords.items():
            if any(kw in query_lower for kw in kws):
                query_type = qtype
                break

        # 엔티티 추출 (간단한 규칙 기반)
        entities = []
        company_keywords = ["삼성", "애플", "테슬라", "네이버", "카카오"]
        for company in company_keywords:
            if company in query:
                entities.append(company)

        return {
            "query_type": query_type,
            "entities": entities,
            "complexity": "complex" if len(entities) > 1 else "simple",
        }

    def create_plan(
        self,
        query: str,
        name: Optional[str] = None,
    ) -> Plan:
        """쿼리에 대한 계획 생성"""
        analysis = self.analyze_query(query)
        query_type = analysis["query_type"]

        plan = Plan(
            name=name or f"Plan for: {query[:30]}",
            description=f"Query type: {query_type}",
            goal=query,
        )

        # 템플릿 기반 계획 생성
        template = self._templates.get(query_type, self._templates["simple_qa"])
        step_map = {}  # name -> step_id

        for tmpl in template:
            # 의존성 해결
            deps = []
            for dep_name in tmpl.get("depends_on", []):
                if dep_name in step_map:
                    deps.append(step_map[dep_name])

            # 파라미터 설정
            params = {"query": query}
            if analysis["entities"]:
                params["entities"] = analysis["entities"]

            step = plan.add_step(
                name=tmpl["name"],
                description=f"{tmpl['name']} 단계",
                action_type=tmpl["action"],
                action_params=params,
                dependencies=deps,
                priority=tmpl.get("priority", 0),
            )
            step_map[tmpl["name"]] = step.id

        plan.status = PlanStatus.READY
        self._plans[plan.id] = plan

        logger.info(f"Created plan: {plan.id} with {plan.total_steps} steps")
        return plan

    def validate_plan(self, plan: Plan) -> tuple[bool, List[str]]:
        """계획 검증"""
        errors = []

        # 빈 계획 확인
        if not plan.steps:
            errors.append("Plan has no steps")

        # 순환 의존성 확인
        if self._has_cycle(plan):
            errors.append("Circular dependency detected")

        # 존재하지 않는 의존성 확인
        step_ids = {s.id for s in plan.steps}
        for step in plan.steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    errors.append(f"Step {step.name} has invalid dependency: {dep}")

        return len(errors) == 0, errors

    def _has_cycle(self, plan: Plan) -> bool:
        """순환 의존성 확인"""
        visited = set()
        rec_stack = set()

        def dfs(step_id: str) -> bool:
            visited.add(step_id)
            rec_stack.add(step_id)

            step = plan.get_step(step_id)
            if step:
                for dep in step.dependencies:
                    if dep not in visited:
                        if dfs(dep):
                            return True
                    elif dep in rec_stack:
                        return True

            rec_stack.remove(step_id)
            return False

        for step in plan.steps:
            if step.id not in visited:
                if dfs(step.id):
                    return True

        return False

    def execute_plan(
        self,
        plan: Plan,
        executor=None,
    ) -> Plan:
        """계획 실행"""
        valid, errors = self.validate_plan(plan)
        if not valid:
            plan.status = PlanStatus.FAILED
            logger.error(f"Plan validation failed: {errors}")
            return plan

        plan.status = PlanStatus.EXECUTING
        plan.started_at = time.time()

        while True:
            ready_steps = plan.get_ready_steps()
            if not ready_steps:
                break

            for step in ready_steps:
                self._execute_step(step, plan, executor)

                if step.status == "failed":
                    plan.status = PlanStatus.FAILED
                    return plan

        plan.status = PlanStatus.COMPLETED
        plan.completed_at = time.time()
        return plan

    def _execute_step(
        self,
        step: PlanStep,
        plan: Plan,
        executor=None,
    ) -> None:
        """단계 실행"""
        step.status = "running"
        step.start_time = time.time()

        try:
            if step.action_type == "tool_call" and executor:
                result = executor.execute(
                    step.action_params.get("tool", "search_documents"),
                    step.action_params,
                )
                step.result = result.result.data if result.result else None
                step.status = "completed" if result.result and result.result.success else "failed"

            elif step.action_type == "llm_call":
                # 이전 단계 결과 수집
                context = self._gather_context(step, plan)
                prompt = f"Goal: {plan.goal}\nContext: {context}"
                step.result = self.llm_func(prompt)
                step.status = "completed"

            elif step.action_type == "compare":
                # 비교 단계
                context = self._gather_context(step, plan)
                step.result = {"comparison": context, "note": "비교 결과"}
                step.status = "completed"

            else:
                step.result = {"note": f"Unknown action: {step.action_type}"}
                step.status = "completed"

        except Exception as e:
            step.error = str(e)
            step.status = "failed"
            logger.error(f"Step execution error: {e}")

        finally:
            step.end_time = time.time()

    def _gather_context(self, step: PlanStep, plan: Plan) -> Dict[str, Any]:
        """선행 단계 결과 수집"""
        context = {}
        for dep_id in step.dependencies:
            dep_step = plan.get_step(dep_id)
            if dep_step and dep_step.result:
                context[dep_step.name] = dep_step.result
        return context

    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """계획 조회"""
        return self._plans.get(plan_id)

    def list_plans(self, status: Optional[PlanStatus] = None) -> List[Plan]:
        """계획 목록"""
        if status:
            return [p for p in self._plans.values() if p.status == status]
        return list(self._plans.values())


class HierarchicalPlanner(TaskPlanner):
    """
    계층적 계획자

    큰 목표를 하위 목표로 분해
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def decompose_goal(
        self,
        goal: str,
        max_depth: int = 3,
    ) -> Dict[str, Any]:
        """목표 분해"""
        def _decompose(current_goal: str, depth: int) -> Dict[str, Any]:
            if depth >= max_depth:
                return {"goal": current_goal, "is_atomic": True}

            # LLM으로 하위 목표 생성 (시뮬레이션)
            sub_goals = self._generate_subgoals(current_goal)

            return {
                "goal": current_goal,
                "is_atomic": len(sub_goals) == 0,
                "sub_goals": [
                    _decompose(sg, depth + 1) for sg in sub_goals
                ],
            }

        return _decompose(goal, 0)

    def _generate_subgoals(self, goal: str) -> List[str]:
        """하위 목표 생성 (규칙 기반 시뮬레이션)"""
        goal_lower = goal.lower()

        if "비교" in goal_lower:
            return [
                f"첫 번째 항목 정보 수집",
                f"두 번째 항목 정보 수집",
                f"비교 분석 수행",
            ]
        elif "분석" in goal_lower:
            return [
                f"관련 데이터 수집",
                f"데이터 분석",
                f"인사이트 도출",
            ]
        else:
            return []  # 원자적 목표

    def create_hierarchical_plan(
        self,
        goal: str,
    ) -> Plan:
        """계층적 계획 생성"""
        decomposition = self.decompose_goal(goal)
        plan = Plan(
            name=f"Hierarchical plan: {goal[:30]}",
            goal=goal,
        )

        def _add_steps(node: Dict[str, Any], parent_id: Optional[str] = None) -> str:
            step = plan.add_step(
                name=node["goal"][:30],
                description=node["goal"],
                action_type="llm_call" if node.get("is_atomic") else "composite",
                dependencies=[parent_id] if parent_id else [],
            )

            if not node.get("is_atomic") and "sub_goals" in node:
                prev_id = None
                for sub in node["sub_goals"]:
                    sub_id = _add_steps(sub, prev_id)
                    prev_id = sub_id

            return step.id

        _add_steps(decomposition)
        plan.status = PlanStatus.READY
        self._plans[plan.id] = plan

        return plan
