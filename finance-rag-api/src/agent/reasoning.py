# -*- coding: utf-8 -*-
"""
추론 엔진 모듈

[기능]
- 멀티스텝 추론
- 추론 체인 관리
- 중간 결과 저장
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class StepType(Enum):
    """추론 단계 유형"""
    THINK = "think"  # 생각/분석
    SEARCH = "search"  # 검색
    CALCULATE = "calculate"  # 계산
    COMPARE = "compare"  # 비교
    SUMMARIZE = "summarize"  # 요약
    VERIFY = "verify"  # 검증
    ANSWER = "answer"  # 최종 답변
    TOOL_CALL = "tool_call"  # 도구 호출


class StepStatus(Enum):
    """단계 상태"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ReasoningStep:
    """추론 단계"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_type: StepType = StepType.THINK
    description: str = ""
    input_data: Any = None
    output_data: Any = None
    status: StepStatus = StepStatus.PENDING
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 도구 호출 관련
    tool_name: Optional[str] = None
    tool_params: Optional[Dict[str, Any]] = None

    @property
    def duration_ms(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "step_type": self.step_type.value,
            "description": self.description,
            "input_data": str(self.input_data)[:200] if self.input_data else None,
            "output_data": str(self.output_data)[:500] if self.output_data else None,
            "status": self.status.value,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 2),
            "tool_name": self.tool_name,
        }


@dataclass
class ReasoningChain:
    """추론 체인"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    steps: List[ReasoningStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    status: StepStatus = StepStatus.PENDING
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def completed_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)

    @property
    def total_duration_ms(self) -> float:
        return sum(s.duration_ms for s in self.steps)

    def add_step(self, step: ReasoningStep) -> None:
        """단계 추가"""
        self.steps.append(step)

    def get_current_step(self) -> Optional[ReasoningStep]:
        """현재 단계"""
        for step in self.steps:
            if step.status in [StepStatus.PENDING, StepStatus.IN_PROGRESS]:
                return step
        return None

    def get_last_output(self) -> Any:
        """마지막 출력"""
        for step in reversed(self.steps):
            if step.status == StepStatus.COMPLETED and step.output_data is not None:
                return step.output_data
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer[:500] if self.final_answer else None,
            "status": self.status.value,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "total_duration_ms": round(self.total_duration_ms, 2),
        }


class ReasoningEngine:
    """
    추론 엔진

    복잡한 쿼리를 단계별로 분해하고 추론
    """

    def __init__(
        self,
        tool_executor=None,
        llm_func: Optional[Callable] = None,
        max_steps: int = 10,
    ):
        self.tool_executor = tool_executor
        self.llm_func = llm_func or self._default_llm
        self.max_steps = max_steps
        self._chains: Dict[str, ReasoningChain] = {}

    def _default_llm(self, prompt: str) -> str:
        """기본 LLM (더미)"""
        return f"[LLM Response for: {prompt[:50]}...]"

    def create_chain(self, query: str) -> ReasoningChain:
        """추론 체인 생성"""
        chain = ReasoningChain(query=query)
        self._chains[chain.id] = chain
        logger.info(f"Created reasoning chain: {chain.id}")
        return chain

    def decompose_query(self, query: str) -> List[ReasoningStep]:
        """쿼리 분해"""
        steps = []

        # 간단한 규칙 기반 분해 (실제로는 LLM 사용)
        keywords = {
            "비교": StepType.COMPARE,
            "계산": StepType.CALCULATE,
            "검색": StepType.SEARCH,
            "분석": StepType.THINK,
            "요약": StepType.SUMMARIZE,
        }

        # 기본 단계: 검색
        steps.append(ReasoningStep(
            step_type=StepType.SEARCH,
            description=f"'{query}'에 대한 관련 문서 검색",
            tool_name="search_documents",
            tool_params={"query": query},
        ))

        # 키워드 기반 추가 단계
        for keyword, step_type in keywords.items():
            if keyword in query:
                steps.append(ReasoningStep(
                    step_type=step_type,
                    description=f"{keyword} 수행",
                ))
                break

        # 분석 단계
        steps.append(ReasoningStep(
            step_type=StepType.THINK,
            description="검색 결과 분석",
        ))

        # 최종 답변
        steps.append(ReasoningStep(
            step_type=StepType.ANSWER,
            description="최종 답변 생성",
        ))

        return steps

    def plan(self, query: str) -> ReasoningChain:
        """쿼리에 대한 추론 계획 수립"""
        chain = self.create_chain(query)

        # 쿼리 분해
        steps = self.decompose_query(query)
        for step in steps:
            chain.add_step(step)

        chain.status = StepStatus.PENDING
        return chain

    def execute_step(
        self,
        chain: ReasoningChain,
        step: ReasoningStep,
    ) -> ReasoningStep:
        """단계 실행"""
        step.status = StepStatus.IN_PROGRESS
        step.start_time = time.time()

        try:
            if step.step_type == StepType.TOOL_CALL and step.tool_name:
                # 도구 호출
                if self.tool_executor:
                    result = self.tool_executor.execute(
                        step.tool_name,
                        step.tool_params or {},
                    )
                    step.output_data = result.result.data if result.result else None
                    if not result.result or not result.result.success:
                        step.error = result.error
                        step.status = StepStatus.FAILED
                    else:
                        step.status = StepStatus.COMPLETED
                else:
                    step.error = "Tool executor not available"
                    step.status = StepStatus.FAILED

            elif step.step_type == StepType.SEARCH:
                # 검색 단계
                if self.tool_executor:
                    result = self.tool_executor.execute(
                        "search_documents",
                        step.tool_params or {"query": chain.query},
                    )
                    step.output_data = result.result.data if result.result else None
                    step.status = StepStatus.COMPLETED
                else:
                    step.output_data = {"note": "검색 결과 (시뮬레이션)"}
                    step.status = StepStatus.COMPLETED

            elif step.step_type == StepType.THINK:
                # 분석 단계
                prev_output = chain.get_last_output()
                prompt = f"다음 정보를 분석해주세요: {prev_output}"
                step.output_data = self.llm_func(prompt)
                step.status = StepStatus.COMPLETED

            elif step.step_type == StepType.ANSWER:
                # 최종 답변 생성
                all_outputs = [s.output_data for s in chain.steps if s.output_data]
                prompt = f"다음 정보를 바탕으로 '{chain.query}'에 답변해주세요: {all_outputs}"
                step.output_data = self.llm_func(prompt)
                chain.final_answer = str(step.output_data)
                step.status = StepStatus.COMPLETED

            elif step.step_type == StepType.CALCULATE:
                # 계산 단계
                if self.tool_executor:
                    result = self.tool_executor.execute(
                        "calculator",
                        step.tool_params or {"expression": "0"},
                    )
                    step.output_data = result.result.data if result.result else None
                    step.status = StepStatus.COMPLETED
                else:
                    step.output_data = {"result": 0}
                    step.status = StepStatus.COMPLETED

            elif step.step_type == StepType.COMPARE:
                # 비교 단계
                prev_output = chain.get_last_output()
                prompt = f"다음 항목들을 비교해주세요: {prev_output}"
                step.output_data = self.llm_func(prompt)
                step.status = StepStatus.COMPLETED

            elif step.step_type == StepType.SUMMARIZE:
                # 요약 단계
                prev_output = chain.get_last_output()
                prompt = f"다음 내용을 요약해주세요: {prev_output}"
                step.output_data = self.llm_func(prompt)
                step.status = StepStatus.COMPLETED

            elif step.step_type == StepType.VERIFY:
                # 검증 단계
                prev_output = chain.get_last_output()
                step.output_data = {"verified": True, "content": prev_output}
                step.status = StepStatus.COMPLETED

            else:
                step.output_data = "Unknown step type"
                step.status = StepStatus.COMPLETED

        except Exception as e:
            step.error = str(e)
            step.status = StepStatus.FAILED
            logger.error(f"Step execution error: {e}")

        finally:
            step.end_time = time.time()

        return step

    def execute_chain(self, chain: ReasoningChain) -> ReasoningChain:
        """전체 체인 실행"""
        chain.status = StepStatus.IN_PROGRESS

        for i, step in enumerate(chain.steps):
            if i >= self.max_steps:
                logger.warning(f"Max steps ({self.max_steps}) reached")
                break

            self.execute_step(chain, step)

            if step.status == StepStatus.FAILED:
                chain.status = StepStatus.FAILED
                return chain

        chain.status = StepStatus.COMPLETED
        chain.completed_at = time.time()
        return chain

    def reason(self, query: str) -> ReasoningChain:
        """쿼리에 대한 추론 실행"""
        chain = self.plan(query)
        return self.execute_chain(chain)

    def get_chain(self, chain_id: str) -> Optional[ReasoningChain]:
        """체인 조회"""
        return self._chains.get(chain_id)

    def get_recent_chains(self, limit: int = 10) -> List[ReasoningChain]:
        """최근 체인들"""
        chains = sorted(
            self._chains.values(),
            key=lambda c: c.created_at,
            reverse=True,
        )
        return chains[:limit]


class ReActReasoner:
    """
    ReAct (Reasoning + Acting) 추론기

    Thought -> Action -> Observation 패턴
    """

    def __init__(
        self,
        tool_executor=None,
        llm_func: Optional[Callable] = None,
        max_iterations: int = 5,
    ):
        self.tool_executor = tool_executor
        self.llm_func = llm_func or (lambda x: f"Response: {x[:50]}")
        self.max_iterations = max_iterations

    def reason(self, query: str) -> Dict[str, Any]:
        """ReAct 패턴으로 추론"""
        thoughts = []
        actions = []
        observations = []

        context = f"Question: {query}\n"

        for i in range(self.max_iterations):
            # Thought
            thought_prompt = context + "\nThought: "
            thought = self.llm_func(thought_prompt)
            thoughts.append(thought)
            context += f"Thought: {thought}\n"

            # 최종 답변 확인
            if "Final Answer:" in thought:
                final_answer = thought.split("Final Answer:")[-1].strip()
                return {
                    "answer": final_answer,
                    "thoughts": thoughts,
                    "actions": actions,
                    "observations": observations,
                    "iterations": i + 1,
                }

            # Action 결정
            action_prompt = context + "\nAction: "
            action = self.llm_func(action_prompt)
            actions.append(action)
            context += f"Action: {action}\n"

            # Action 실행 (Observation)
            observation = self._execute_action(action)
            observations.append(observation)
            context += f"Observation: {observation}\n"

        return {
            "answer": "Maximum iterations reached",
            "thoughts": thoughts,
            "actions": actions,
            "observations": observations,
            "iterations": self.max_iterations,
        }

    def _execute_action(self, action: str) -> str:
        """액션 실행"""
        # 간단한 액션 파싱
        if "search" in action.lower():
            query = action.split(":")[-1].strip() if ":" in action else action
            if self.tool_executor:
                result = self.tool_executor.execute(
                    "search_documents",
                    {"query": query},
                )
                return str(result.result.data) if result.result else "No results"
            return f"Search results for: {query}"

        elif "calculate" in action.lower():
            expression = action.split(":")[-1].strip() if ":" in action else "0"
            if self.tool_executor:
                result = self.tool_executor.execute(
                    "calculator",
                    {"expression": expression},
                )
                return str(result.result.data) if result.result else "Calculation error"
            return f"Calculation result: {expression}"

        return f"Unknown action: {action}"
