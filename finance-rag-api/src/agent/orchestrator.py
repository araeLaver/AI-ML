# -*- coding: utf-8 -*-
"""
에이전트 오케스트레이터 모듈

[기능]
- 멀티 에이전트 관리
- 에이전트 간 협업
- 작업 분배 및 조율
"""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .tools import Tool, ToolRegistry, ToolResult
from .executor import ToolExecutor, ExecutionContext
from .memory import AgentMemory, ConversationMemory

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """에이전트 역할"""
    COORDINATOR = "coordinator"  # 조율자
    RESEARCHER = "researcher"  # 연구자
    ANALYST = "analyst"  # 분석가
    CALCULATOR = "calculator"  # 계산기
    VERIFIER = "verifier"  # 검증자
    RESPONDER = "responder"  # 응답자
    SPECIALIST = "specialist"  # 전문가


class AgentState(Enum):
    """에이전트 상태"""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentMessage:
    """에이전트 간 메시지"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    receiver: str = ""
    content: Any = None
    message_type: str = "request"  # request, response, broadcast
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "content": str(self.content)[:200] if self.content else None,
            "message_type": self.message_type,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentTask:
    """에이전트 작업"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    input_data: Any = None
    output_data: Any = None
    assigned_to: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "assigned_to": self.assigned_to,
            "status": self.status,
            "priority": self.priority,
        }


class Agent(ABC):
    """
    에이전트 기본 클래스

    특정 역할을 수행하는 에이전트
    """

    def __init__(
        self,
        name: str,
        role: AgentRole,
        tools: Optional[List[Tool]] = None,
        llm_func: Optional[Callable] = None,
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.role = role
        self.tools = tools or []
        self.llm_func = llm_func or (lambda x: f"Response: {x[:50]}")
        self.state = AgentState.IDLE
        self.memory = ConversationMemory()
        self._message_queue: List[AgentMessage] = []

    @abstractmethod
    def process(self, task: AgentTask) -> AgentTask:
        """작업 처리"""
        pass

    def receive_message(self, message: AgentMessage) -> None:
        """메시지 수신"""
        self._message_queue.append(message)
        logger.debug(f"Agent {self.name} received message from {message.sender}")

    def send_message(
        self,
        receiver: str,
        content: Any,
        message_type: str = "request",
    ) -> AgentMessage:
        """메시지 발송"""
        message = AgentMessage(
            sender=self.name,
            receiver=receiver,
            content=content,
            message_type=message_type,
        )
        return message

    def get_pending_messages(self) -> List[AgentMessage]:
        """대기 중인 메시지"""
        messages = self._message_queue
        self._message_queue = []
        return messages

    def use_tool(self, tool_name: str, **params) -> Optional[ToolResult]:
        """도구 사용"""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool.execute(**params)
        return None

    def get_info(self) -> Dict[str, Any]:
        """에이전트 정보"""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.value,
            "state": self.state.value,
            "tools": [t.name for t in self.tools],
        }


class ResearcherAgent(Agent):
    """연구 에이전트 - 정보 검색"""

    def __init__(
        self,
        name: str = "Researcher",
        search_func: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(name, AgentRole.RESEARCHER, **kwargs)
        self.search_func = search_func

    def process(self, task: AgentTask) -> AgentTask:
        """검색 작업 처리"""
        self.state = AgentState.ACTING
        task.status = "in_progress"

        try:
            query = task.input_data.get("query", str(task.input_data))

            if self.search_func:
                results = self.search_func(query)
            else:
                # 도구 사용
                result = self.use_tool("search_documents", query=query)
                results = result.data if result and result.success else None

            task.output_data = {
                "query": query,
                "results": results,
                "source": "search",
            }
            task.status = "completed"

        except Exception as e:
            task.status = "failed"
            task.output_data = {"error": str(e)}
            logger.error(f"Researcher error: {e}")

        finally:
            self.state = AgentState.IDLE
            task.completed_at = time.time()

        return task


class AnalystAgent(Agent):
    """분석 에이전트 - 데이터 분석"""

    def __init__(self, name: str = "Analyst", **kwargs):
        super().__init__(name, AgentRole.ANALYST, **kwargs)

    def process(self, task: AgentTask) -> AgentTask:
        """분석 작업 처리"""
        self.state = AgentState.THINKING
        task.status = "in_progress"

        try:
            data = task.input_data

            # LLM을 사용한 분석
            prompt = f"다음 데이터를 분석해주세요:\n{data}"
            analysis = self.llm_func(prompt)

            task.output_data = {
                "analysis": analysis,
                "input_summary": str(data)[:200],
            }
            task.status = "completed"

        except Exception as e:
            task.status = "failed"
            task.output_data = {"error": str(e)}

        finally:
            self.state = AgentState.IDLE
            task.completed_at = time.time()

        return task


class CalculatorAgent(Agent):
    """계산 에이전트 - 수치 계산"""

    def __init__(self, name: str = "Calculator", **kwargs):
        super().__init__(name, AgentRole.CALCULATOR, **kwargs)

    def process(self, task: AgentTask) -> AgentTask:
        """계산 작업 처리"""
        self.state = AgentState.ACTING
        task.status = "in_progress"

        try:
            expression = task.input_data.get("expression", "")

            result = self.use_tool("calculator", expression=expression)

            if result and result.success:
                task.output_data = result.data
                task.status = "completed"
            else:
                task.status = "failed"
                task.output_data = {"error": result.error if result else "Unknown error"}

        except Exception as e:
            task.status = "failed"
            task.output_data = {"error": str(e)}

        finally:
            self.state = AgentState.IDLE
            task.completed_at = time.time()

        return task


class ResponderAgent(Agent):
    """응답 에이전트 - 최종 응답 생성"""

    def __init__(self, name: str = "Responder", **kwargs):
        super().__init__(name, AgentRole.RESPONDER, **kwargs)

    def process(self, task: AgentTask) -> AgentTask:
        """응답 생성"""
        self.state = AgentState.THINKING
        task.status = "in_progress"

        try:
            context = task.input_data.get("context", {})
            query = task.input_data.get("query", "")

            prompt = f"""
다음 정보를 바탕으로 질문에 답변해주세요.

질문: {query}

수집된 정보:
{context}

답변:
"""
            response = self.llm_func(prompt)

            task.output_data = {
                "response": response,
                "query": query,
            }
            task.status = "completed"

        except Exception as e:
            task.status = "failed"
            task.output_data = {"error": str(e)}

        finally:
            self.state = AgentState.IDLE
            task.completed_at = time.time()

        return task


class AgentOrchestrator:
    """
    에이전트 오케스트레이터

    멀티 에이전트 시스템 조율
    """

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        llm_func: Optional[Callable] = None,
    ):
        self.tool_registry = tool_registry or ToolRegistry()
        self.llm_func = llm_func
        self._agents: Dict[str, Agent] = {}
        self._task_queue: List[AgentTask] = []
        self._completed_tasks: List[AgentTask] = []
        self._message_bus: List[AgentMessage] = []

    def register_agent(self, agent: Agent) -> None:
        """에이전트 등록"""
        self._agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name} ({agent.role.value})")

    def unregister_agent(self, name: str) -> bool:
        """에이전트 등록 해제"""
        if name in self._agents:
            del self._agents[name]
            return True
        return False

    def get_agent(self, name: str) -> Optional[Agent]:
        """에이전트 조회"""
        return self._agents.get(name)

    def get_agents_by_role(self, role: AgentRole) -> List[Agent]:
        """역할별 에이전트"""
        return [a for a in self._agents.values() if a.role == role]

    def create_task(
        self,
        description: str,
        input_data: Any,
        priority: int = 0,
    ) -> AgentTask:
        """작업 생성"""
        task = AgentTask(
            description=description,
            input_data=input_data,
            priority=priority,
        )
        self._task_queue.append(task)
        return task

    def assign_task(
        self,
        task: AgentTask,
        agent_name: str,
    ) -> bool:
        """작업 할당"""
        agent = self._agents.get(agent_name)
        if not agent:
            return False

        task.assigned_to = agent_name
        return True

    def execute_task(self, task: AgentTask) -> AgentTask:
        """작업 실행"""
        if not task.assigned_to:
            # 자동 할당
            agent = self._select_agent_for_task(task)
            if agent:
                task.assigned_to = agent.name
            else:
                task.status = "failed"
                task.output_data = {"error": "No suitable agent found"}
                return task

        agent = self._agents.get(task.assigned_to)
        if not agent:
            task.status = "failed"
            task.output_data = {"error": f"Agent not found: {task.assigned_to}"}
            return task

        return agent.process(task)

    def _select_agent_for_task(self, task: AgentTask) -> Optional[Agent]:
        """작업에 적합한 에이전트 선택"""
        description = task.description.lower()

        # 키워드 기반 역할 매칭
        role_keywords = {
            AgentRole.RESEARCHER: ["검색", "찾", "search", "find"],
            AgentRole.ANALYST: ["분석", "해석", "analyze", "interpret"],
            AgentRole.CALCULATOR: ["계산", "수치", "calculate", "compute"],
            AgentRole.RESPONDER: ["답변", "응답", "respond", "answer"],
        }

        for role, keywords in role_keywords.items():
            if any(kw in description for kw in keywords):
                agents = self.get_agents_by_role(role)
                if agents:
                    # IDLE 상태인 에이전트 우선
                    for agent in agents:
                        if agent.state == AgentState.IDLE:
                            return agent
                    return agents[0]

        # 기본: 첫 번째 에이전트
        return next(iter(self._agents.values()), None)

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        복합 쿼리 처리

        쿼리를 분석하여 여러 에이전트에 분배하고 결과 취합
        """
        results = {
            "query": query,
            "tasks": [],
            "final_response": None,
            "status": "processing",
        }

        # 1. 검색 단계
        researcher_agents = self.get_agents_by_role(AgentRole.RESEARCHER)
        if researcher_agents:
            search_task = self.create_task(
                description="관련 정보 검색",
                input_data={"query": query},
            )
            search_task.assigned_to = researcher_agents[0].name
            search_task = self.execute_task(search_task)
            results["tasks"].append(search_task.to_dict())

            search_results = search_task.output_data

        else:
            search_results = None

        # 2. 분석 단계
        analyst_agents = self.get_agents_by_role(AgentRole.ANALYST)
        if analyst_agents and search_results:
            analysis_task = self.create_task(
                description="검색 결과 분석",
                input_data=search_results,
            )
            analysis_task.assigned_to = analyst_agents[0].name
            analysis_task = self.execute_task(analysis_task)
            results["tasks"].append(analysis_task.to_dict())

            analysis_results = analysis_task.output_data
        else:
            analysis_results = search_results

        # 3. 응답 생성
        responder_agents = self.get_agents_by_role(AgentRole.RESPONDER)
        if responder_agents:
            response_task = self.create_task(
                description="최종 응답 생성",
                input_data={
                    "query": query,
                    "context": analysis_results,
                },
            )
            response_task.assigned_to = responder_agents[0].name
            response_task = self.execute_task(response_task)
            results["tasks"].append(response_task.to_dict())

            if response_task.output_data:
                results["final_response"] = response_task.output_data.get("response")

        results["status"] = "completed"
        return results

    def broadcast_message(self, sender: str, content: Any) -> None:
        """전체 에이전트에 메시지 브로드캐스트"""
        for agent_name, agent in self._agents.items():
            if agent_name != sender:
                message = AgentMessage(
                    sender=sender,
                    receiver=agent_name,
                    content=content,
                    message_type="broadcast",
                )
                agent.receive_message(message)
                self._message_bus.append(message)

    def send_message(
        self,
        sender: str,
        receiver: str,
        content: Any,
    ) -> bool:
        """에이전트 간 메시지 전송"""
        agent = self._agents.get(receiver)
        if not agent:
            return False

        message = AgentMessage(
            sender=sender,
            receiver=receiver,
            content=content,
            message_type="request",
        )
        agent.receive_message(message)
        self._message_bus.append(message)
        return True

    def get_status(self) -> Dict[str, Any]:
        """오케스트레이터 상태"""
        return {
            "agents": {
                name: agent.get_info()
                for name, agent in self._agents.items()
            },
            "pending_tasks": len(self._task_queue),
            "completed_tasks": len(self._completed_tasks),
            "messages_processed": len(self._message_bus),
        }

    def setup_default_agents(self) -> None:
        """기본 에이전트 설정"""
        # 기본 에이전트들 생성
        researcher = ResearcherAgent(llm_func=self.llm_func)
        analyst = AnalystAgent(llm_func=self.llm_func)
        responder = ResponderAgent(llm_func=self.llm_func)

        self.register_agent(researcher)
        self.register_agent(analyst)
        self.register_agent(responder)

        logger.info("Default agents set up")
