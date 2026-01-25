# -*- coding: utf-8 -*-
"""
에이전트 시스템 테스트
"""

import time

import pytest

from src.agent.tools import (
    Tool,
    ToolRegistry,
    ToolResult,
    ToolParameter,
    ParameterType,
    FunctionTool,
    create_tool,
    CalculatorTool,
    SearchTool,
)
from src.agent.executor import (
    ToolExecutor,
    ExecutionContext,
    ExecutionResult,
    ExecutionStatus,
    ChainedExecutor,
)
from src.agent.reasoning import (
    ReasoningEngine,
    ReasoningStep,
    ReasoningChain,
    StepType,
    StepStatus,
    ReActReasoner,
)
from src.agent.orchestrator import (
    AgentOrchestrator,
    Agent,
    AgentRole,
    AgentState,
    ResearcherAgent,
    AnalystAgent,
    ResponderAgent,
    AgentTask,
)
from src.agent.memory import (
    AgentMemory,
    ConversationMemory,
    WorkingMemory,
    LongTermMemory,
    MemoryType,
)
from src.agent.planner import (
    TaskPlanner,
    Plan,
    PlanStep,
    PlanStatus,
    HierarchicalPlanner,
)


# =============================================================================
# Tools Tests
# =============================================================================

class TestToolParameter:
    """도구 파라미터 테스트"""

    def test_to_schema(self):
        """스키마 변환 테스트"""
        param = ToolParameter(
            name="query",
            param_type=ParameterType.STRING,
            description="검색 쿼리",
            required=True,
        )

        schema = param.to_schema()
        assert schema["type"] == "string"
        assert schema["description"] == "검색 쿼리"


class TestFunctionTool:
    """함수 도구 테스트"""

    def test_create_from_function(self):
        """함수로부터 도구 생성"""
        def sample_func(query: str, limit: int = 10) -> dict:
            """샘플 함수"""
            return {"query": query, "limit": limit}

        tool = FunctionTool(sample_func)

        assert tool.name == "sample_func"
        assert len(tool.parameters) == 2

    def test_execute(self):
        """실행 테스트"""
        def add(a: int, b: int) -> int:
            return a + b

        tool = FunctionTool(add)
        result = tool.execute(a=1, b=2)

        assert result.success is True
        assert result.data == 3


class TestToolDecorator:
    """도구 데코레이터 테스트"""

    def test_create_tool_decorator(self):
        """데코레이터 테스트"""
        @create_tool(name="my_tool", description="My custom tool")
        def my_func(x: str) -> str:
            return x.upper()

        assert my_func.name == "my_tool"
        assert my_func.description == "My custom tool"

        result = my_func.execute(x="hello")
        assert result.data == "HELLO"


class TestToolRegistry:
    """도구 레지스트리 테스트"""

    def test_register_and_get(self):
        """등록 및 조회 테스트"""
        registry = ToolRegistry()

        tool = CalculatorTool()
        registry.register(tool)

        retrieved = registry.get("calculator")
        assert retrieved is not None
        assert retrieved.name == "calculator"

    def test_register_function(self):
        """함수 등록 테스트"""
        registry = ToolRegistry()

        def double(n: int) -> int:
            return n * 2

        tool = registry.register_function(double, category="math")

        assert registry.get("double") is not None
        assert len(registry.get_by_category("math")) == 1


class TestCalculatorTool:
    """계산기 도구 테스트"""

    def test_basic_calculation(self):
        """기본 계산 테스트"""
        calc = CalculatorTool()

        result = calc.execute(expression="100 * 1.15")
        assert result.success is True
        assert abs(result.data["result"] - 115.0) < 0.0001  # 부동소수점 비교

    def test_invalid_expression(self):
        """잘못된 표현식 테스트"""
        calc = CalculatorTool()

        result = calc.execute(expression="import os")
        assert result.success is False


# =============================================================================
# Executor Tests
# =============================================================================

class TestToolExecutor:
    """도구 실행기 테스트"""

    def test_execute(self):
        """실행 테스트"""
        registry = ToolRegistry()
        registry.register(CalculatorTool())

        executor = ToolExecutor(registry)
        result = executor.execute("calculator", {"expression": "2 + 2"})

        assert result.status == ExecutionStatus.COMPLETED
        assert result.result.data["result"] == 4

    def test_execute_not_found(self):
        """없는 도구 테스트"""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)

        result = executor.execute("unknown_tool", {})
        assert result.status == ExecutionStatus.FAILED
        assert "not found" in result.error.lower()

    def test_execute_many(self):
        """다중 실행 테스트"""
        registry = ToolRegistry()
        registry.register(CalculatorTool())

        executor = ToolExecutor(registry)
        results = executor.execute_many([
            ("calculator", {"expression": "1 + 1"}),
            ("calculator", {"expression": "2 * 2"}),
        ])

        assert len(results) == 2
        assert all(r.status == ExecutionStatus.COMPLETED for r in results)


class TestChainedExecutor:
    """체인 실행기 테스트"""

    def test_chain_execution(self):
        """체인 실행 테스트"""
        registry = ToolRegistry()
        registry.register(CalculatorTool())

        executor = ToolExecutor(registry)
        chained = ChainedExecutor(executor)

        chained.add("calculator", {"expression": "10 + 5"}, "step1")
        chained.add("calculator", {"expression": "20 + 10"}, "step2")

        context = ExecutionContext()
        results = chained.execute(context)

        assert len(results) == 2
        assert all(r.status == ExecutionStatus.COMPLETED for r in results)


# =============================================================================
# Reasoning Tests
# =============================================================================

class TestReasoningStep:
    """추론 단계 테스트"""

    def test_create_step(self):
        """단계 생성 테스트"""
        step = ReasoningStep(
            step_type=StepType.SEARCH,
            description="문서 검색",
        )

        assert step.step_type == StepType.SEARCH
        assert step.status == StepStatus.PENDING


class TestReasoningChain:
    """추론 체인 테스트"""

    def test_add_steps(self):
        """단계 추가 테스트"""
        chain = ReasoningChain(query="테스트 쿼리")

        chain.add_step(ReasoningStep(step_type=StepType.SEARCH))
        chain.add_step(ReasoningStep(step_type=StepType.THINK))

        assert chain.total_steps == 2

    def test_get_ready_steps(self):
        """실행 가능 단계 테스트"""
        chain = ReasoningChain(query="테스트")

        step1 = ReasoningStep(step_type=StepType.SEARCH)
        step2 = ReasoningStep(step_type=StepType.THINK)

        chain.add_step(step1)
        chain.add_step(step2)

        assert chain.get_current_step() == step1


class TestReasoningEngine:
    """추론 엔진 테스트"""

    def test_decompose_query(self):
        """쿼리 분해 테스트"""
        engine = ReasoningEngine()
        steps = engine.decompose_query("삼성전자 실적 분석")

        assert len(steps) > 0
        assert any(s.step_type == StepType.SEARCH for s in steps)

    def test_plan(self):
        """계획 수립 테스트"""
        engine = ReasoningEngine()
        chain = engine.plan("삼성전자와 애플 비교")

        assert chain.total_steps > 0
        assert chain.status == StepStatus.PENDING

    def test_reason(self):
        """추론 실행 테스트"""
        engine = ReasoningEngine()
        chain = engine.reason("테스트 쿼리")

        assert chain.status == StepStatus.COMPLETED
        assert chain.completed_steps > 0


# =============================================================================
# Orchestrator Tests
# =============================================================================

class TestAgent:
    """에이전트 테스트"""

    def test_researcher_agent(self):
        """연구 에이전트 테스트"""
        agent = ResearcherAgent()

        task = AgentTask(
            description="정보 검색",
            input_data={"query": "테스트"},
        )

        result = agent.process(task)
        assert result.status == "completed"

    def test_analyst_agent(self):
        """분석 에이전트 테스트"""
        agent = AnalystAgent()

        task = AgentTask(
            description="데이터 분석",
            input_data={"data": "테스트 데이터"},
        )

        result = agent.process(task)
        assert result.status == "completed"


class TestAgentOrchestrator:
    """에이전트 오케스트레이터 테스트"""

    def test_register_agents(self):
        """에이전트 등록 테스트"""
        orchestrator = AgentOrchestrator()

        researcher = ResearcherAgent()
        orchestrator.register_agent(researcher)

        assert orchestrator.get_agent("Researcher") is not None

    def test_setup_default_agents(self):
        """기본 에이전트 설정 테스트"""
        orchestrator = AgentOrchestrator()
        orchestrator.setup_default_agents()

        status = orchestrator.get_status()
        assert len(status["agents"]) >= 3

    def test_process_query(self):
        """쿼리 처리 테스트"""
        orchestrator = AgentOrchestrator()
        orchestrator.setup_default_agents()

        result = orchestrator.process_query("삼성전자 실적")

        assert result["status"] == "completed"
        assert len(result["tasks"]) > 0


# =============================================================================
# Memory Tests
# =============================================================================

class TestConversationMemory:
    """대화 메모리 테스트"""

    def test_add_and_get(self):
        """추가 및 조회 테스트"""
        memory = ConversationMemory()

        entry = memory.add_user_message("안녕하세요")
        retrieved = memory.get(entry.id)

        assert retrieved is not None
        assert retrieved.content == "안녕하세요"

    def test_get_recent(self):
        """최근 대화 테스트"""
        memory = ConversationMemory()

        memory.add_user_message("메시지 1")
        memory.add_assistant_message("응답 1")
        memory.add_user_message("메시지 2")

        recent = memory.get_recent(2)
        assert len(recent) == 2

    def test_get_context_window(self):
        """컨텍스트 윈도우 테스트"""
        memory = ConversationMemory()

        memory.add_user_message("질문")
        memory.add_assistant_message("답변")

        context = memory.get_context_window()
        assert len(context) == 2
        assert context[0]["role"] == "user"


class TestWorkingMemory:
    """작업 메모리 테스트"""

    def test_variables(self):
        """변수 테스트"""
        memory = WorkingMemory()

        memory.set_variable("result", 42)
        assert memory.get_variable("result") == 42

    def test_add_and_search(self):
        """추가 및 검색 테스트"""
        memory = WorkingMemory()

        memory.add("검색 결과: 삼성전자", importance=0.8, key="search")
        results = memory.search("삼성")

        assert len(results) >= 1


class TestLongTermMemory:
    """장기 메모리 테스트"""

    def test_add_with_tags(self):
        """태그 추가 테스트"""
        memory = LongTermMemory()

        entry = memory.add(
            "삼성전자 2024년 실적",
            tags=["삼성", "실적"],
        )

        results = memory.search_by_tag("삼성")
        assert len(results) == 1

    def test_search(self):
        """검색 테스트"""
        memory = LongTermMemory()

        memory.add("애플 아이폰 매출")
        memory.add("삼성 갤럭시 판매량")

        results = memory.search("아이폰")
        assert len(results) >= 1


# =============================================================================
# Planner Tests
# =============================================================================

class TestPlanStep:
    """계획 단계 테스트"""

    def test_create_step(self):
        """단계 생성 테스트"""
        step = PlanStep(
            name="검색",
            action_type="tool_call",
            action_params={"query": "테스트"},
        )

        assert step.name == "검색"
        assert step.status == "pending"


class TestPlan:
    """계획 테스트"""

    def test_add_step(self):
        """단계 추가 테스트"""
        plan = Plan(goal="테스트 목표")

        step = plan.add_step(
            name="단계1",
            action_type="search",
        )

        assert plan.total_steps == 1
        assert step in plan.steps

    def test_get_ready_steps(self):
        """실행 가능 단계 테스트"""
        plan = Plan(goal="테스트")

        step1 = plan.add_step(name="1단계")
        step2 = plan.add_step(name="2단계", dependencies=[step1.id])

        ready = plan.get_ready_steps()
        assert len(ready) == 1
        assert ready[0].name == "1단계"


class TestTaskPlanner:
    """작업 계획자 테스트"""

    def test_analyze_query(self):
        """쿼리 분석 테스트"""
        planner = TaskPlanner()

        analysis = planner.analyze_query("삼성전자와 애플 비교")
        assert analysis["query_type"] == "comparison"

    def test_create_plan(self):
        """계획 생성 테스트"""
        planner = TaskPlanner()

        plan = planner.create_plan("삼성전자 실적 분석")

        assert plan.status == PlanStatus.READY
        assert plan.total_steps > 0

    def test_validate_plan(self):
        """계획 검증 테스트"""
        planner = TaskPlanner()
        plan = planner.create_plan("테스트 쿼리")

        valid, errors = planner.validate_plan(plan)
        assert valid is True


class TestHierarchicalPlanner:
    """계층적 계획자 테스트"""

    def test_decompose_goal(self):
        """목표 분해 테스트"""
        planner = HierarchicalPlanner()

        decomposition = planner.decompose_goal("삼성과 애플 비교 분석")

        assert "goal" in decomposition
        assert "sub_goals" in decomposition or decomposition.get("is_atomic")

    def test_create_hierarchical_plan(self):
        """계층적 계획 생성 테스트"""
        planner = HierarchicalPlanner()

        plan = planner.create_hierarchical_plan("삼성과 애플 비교")

        assert plan.status == PlanStatus.READY
