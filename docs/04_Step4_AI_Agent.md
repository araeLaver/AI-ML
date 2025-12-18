# Step 4: AI Agent 개발 (2-3개월)

## 목표
> 자율적으로 작업을 수행하는 AI 에이전트 구축

## AI Agent란?

```
┌─────────────────────────────────────────────────────────────────────┐
│                      AI Agent vs 일반 LLM                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  일반 LLM:  질문 → 답변 (1회성)                                     │
│                                                                     │
│  AI Agent:  목표 → 계획 → 실행 → 평가 → 반복 (자율적)               │
│                                                                     │
│             ┌────────────────────────────────────┐                 │
│             │         Agent 핵심 구성요소         │                 │
│             ├────────────────────────────────────┤                 │
│             │  1. LLM (두뇌)                     │                 │
│             │  2. Tools (손과 발)                │                 │
│             │  3. Memory (기억)                  │                 │
│             │  4. Planning (계획 능력)           │                 │
│             └────────────────────────────────────┘                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 학습 순서

### Week 1-2: Agent 기본 개념

#### ReAct 패턴 (Reasoning + Acting)

```python
"""
ReAct 패턴: 생각 → 행동 → 관찰 반복

Thought: 사용자가 삼성전자 주가를 물었다. 주가 API를 호출해야 한다.
Action: get_stock_price("005930")
Observation: 현재가 72,000원, 전일 대비 +1.5%
Thought: 주가 정보를 얻었다. 사용자에게 친절하게 설명해야 한다.
Action: respond_to_user
Final Answer: 삼성전자(005930)의 현재 주가는 72,000원입니다.
             전일 대비 1.5% 상승했습니다.
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

# 도구 정의
tools = [
    Tool(
        name="get_stock_price",
        func=lambda symbol: f"주가: 72,000원, 변동: +1.5%",
        description="주식 심볼로 현재가를 조회합니다"
    ),
    Tool(
        name="get_company_info",
        func=lambda symbol: f"삼성전자: IT/전자 대기업",
        description="회사 정보를 조회합니다"
    )
]

# Agent 생성
llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 실행
result = agent_executor.invoke({"input": "삼성전자 주가와 회사 정보를 알려줘"})
```

---

### Week 3-4: LangGraph (권장 프레임워크)

#### LangGraph 기본 구조

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# 상태 정의
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_action: str

# 노드 함수들
def analyze_request(state: AgentState) -> AgentState:
    """사용자 요청 분석"""
    messages = state["messages"]
    # LLM으로 요청 분석
    analysis = llm.invoke(messages)
    return {"messages": [analysis], "next_action": "search"}

def search_data(state: AgentState) -> AgentState:
    """데이터 검색"""
    # 검색 로직
    results = search_tool.invoke(state["messages"][-1])
    return {"messages": [results], "next_action": "respond"}

def generate_response(state: AgentState) -> AgentState:
    """응답 생성"""
    response = llm.invoke(state["messages"])
    return {"messages": [response], "next_action": "end"}

# 그래프 구성
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("analyze", analyze_request)
workflow.add_node("search", search_data)
workflow.add_node("respond", generate_response)

# 엣지 (흐름) 정의
workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "search")
workflow.add_edge("search", "respond")
workflow.add_edge("respond", END)

# 컴파일
app = workflow.compile()

# 실행
result = app.invoke({"messages": ["금융 이상거래 탐지 방법 알려줘"], "next_action": ""})
```

#### 조건부 라우팅

```python
from langgraph.graph import StateGraph, END

def route_by_intent(state: AgentState) -> str:
    """의도에 따라 다른 노드로 라우팅"""
    last_message = state["messages"][-1]

    if "주가" in last_message or "stock" in last_message.lower():
        return "stock_agent"
    elif "이상거래" in last_message or "fraud" in last_message.lower():
        return "fraud_agent"
    else:
        return "general_agent"

workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("classifier", classify_intent)
workflow.add_node("stock_agent", handle_stock_query)
workflow.add_node("fraud_agent", handle_fraud_query)
workflow.add_node("general_agent", handle_general_query)

# 조건부 엣지
workflow.add_conditional_edges(
    "classifier",
    route_by_intent,
    {
        "stock_agent": "stock_agent",
        "fraud_agent": "fraud_agent",
        "general_agent": "general_agent"
    }
)
```

---

### Week 5-6: 멀티 에이전트 시스템

#### 역할 분담 패턴

```python
from langgraph.graph import StateGraph, END

class MultiAgentState(TypedDict):
    task: str
    security_analysis: str
    performance_analysis: str
    style_analysis: str
    final_report: str

# 전문 에이전트들
def security_analyst(state: MultiAgentState) -> MultiAgentState:
    """보안 분석 에이전트"""
    prompt = f"""
    당신은 보안 전문가입니다. 다음 코드의 보안 취약점을 분석하세요.
    코드: {state['task']}
    """
    analysis = llm.invoke(prompt)
    return {"security_analysis": analysis}

def performance_analyst(state: MultiAgentState) -> MultiAgentState:
    """성능 분석 에이전트"""
    prompt = f"""
    당신은 성능 최적화 전문가입니다. 다음 코드의 성능을 분석하세요.
    코드: {state['task']}
    """
    analysis = llm.invoke(prompt)
    return {"performance_analysis": analysis}

def style_analyst(state: MultiAgentState) -> MultiAgentState:
    """코드 스타일 분석 에이전트"""
    prompt = f"""
    당신은 코드 리뷰어입니다. 다음 코드의 스타일과 가독성을 분석하세요.
    코드: {state['task']}
    """
    analysis = llm.invoke(prompt)
    return {"style_analysis": analysis}

def synthesizer(state: MultiAgentState) -> MultiAgentState:
    """분석 결과 종합 에이전트"""
    prompt = f"""
    다음 분석 결과들을 종합하여 최종 코드 리뷰 보고서를 작성하세요.

    보안 분석: {state['security_analysis']}
    성능 분석: {state['performance_analysis']}
    스타일 분석: {state['style_analysis']}
    """
    report = llm.invoke(prompt)
    return {"final_report": report}

# 멀티 에이전트 그래프
workflow = StateGraph(MultiAgentState)

workflow.add_node("security", security_analyst)
workflow.add_node("performance", performance_analyst)
workflow.add_node("style", style_analyst)
workflow.add_node("synthesize", synthesizer)

# 병렬 실행 후 종합
workflow.set_entry_point("security")
workflow.add_edge("security", "performance")
workflow.add_edge("performance", "style")
workflow.add_edge("style", "synthesize")
workflow.add_edge("synthesize", END)
```

---

### Week 7-8: Tool Use / MCP (Model Context Protocol)

#### 커스텀 도구 만들기

```python
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class StockPriceInput(BaseModel):
    symbol: str = Field(description="주식 종목 코드")
    market: str = Field(default="KRX", description="거래소 (KRX, NYSE, NASDAQ)")

class StockPriceTool(BaseTool):
    name = "stock_price"
    description = "주식의 현재가와 변동률을 조회합니다"
    args_schema: Type[BaseModel] = StockPriceInput

    def _run(self, symbol: str, market: str = "KRX") -> str:
        # 실제 API 호출 로직
        price_data = fetch_stock_price(symbol, market)
        return f"종목: {symbol}, 현재가: {price_data['price']}원, 변동: {price_data['change']}%"

class TransactionAnalyzerInput(BaseModel):
    transaction_id: str = Field(description="거래 ID")

class TransactionAnalyzerTool(BaseTool):
    name = "analyze_transaction"
    description = "거래의 이상 여부를 분석합니다"
    args_schema: Type[BaseModel] = TransactionAnalyzerInput

    def _run(self, transaction_id: str) -> str:
        # 이상 거래 분석 로직
        result = analyze_fraud(transaction_id)
        return f"거래 ID: {transaction_id}, 이상 확률: {result['score']}%, 사유: {result['reason']}"
```

#### MCP (Model Context Protocol) 연동

```python
# MCP 서버 예시 (금융 데이터 제공)
from mcp import Server, Tool

server = Server("financial-data-server")

@server.tool()
async def get_market_data(symbol: str, period: str = "1d") -> dict:
    """시장 데이터 조회"""
    data = await fetch_market_data(symbol, period)
    return {
        "symbol": symbol,
        "period": period,
        "data": data
    }

@server.tool()
async def calculate_risk_score(portfolio: list) -> dict:
    """포트폴리오 리스크 점수 계산"""
    score = await compute_risk(portfolio)
    return {
        "risk_score": score,
        "risk_level": "HIGH" if score > 70 else "MEDIUM" if score > 40 else "LOW"
    }
```

---

### Week 9-10: 워크플로우 자동화

#### n8n + AI 연동

```javascript
// n8n 워크플로우 예시
{
  "nodes": [
    {
      "name": "Webhook Trigger",
      "type": "n8n-nodes-base.webhook",
      "parameters": {
        "path": "financial-alert"
      }
    },
    {
      "name": "OpenAI",
      "type": "@n8n/n8n-nodes-langchain.openAi",
      "parameters": {
        "model": "gpt-4o",
        "prompt": "다음 거래 데이터를 분석하고 이상 여부를 판단하세요: {{$json.transaction}}"
      }
    },
    {
      "name": "Slack Notification",
      "type": "n8n-nodes-base.slack",
      "parameters": {
        "channel": "#alerts",
        "message": "이상 거래 탐지: {{$json.analysis}}"
      }
    }
  ]
}
```

---

## 실습 프로젝트

### 프로젝트: 자동 코드 리뷰 AI Agent

**김다운님의 두 번째 포트폴리오 프로젝트**

```
code_review_agent/
├── agents/
│   ├── __init__.py
│   ├── security_agent.py    # 보안 취약점 분석
│   ├── performance_agent.py # 성능 분석
│   ├── style_agent.py       # 코드 스타일 분석
│   └── orchestrator.py      # 에이전트 조율
├── tools/
│   ├── github_tools.py      # GitHub API 연동
│   ├── code_analyzer.py     # 코드 분석 도구
│   └── report_generator.py  # 리포트 생성
├── workflows/
│   └── review_workflow.py   # LangGraph 워크플로우
├── api/
│   └── webhook.py           # GitHub Webhook 처리
├── tests/
├── docker-compose.yml
└── main.py
```

**구현 기능**:
1. GitHub PR Webhook 수신
2. 코드 변경사항 분석
3. 멀티 에이전트 리뷰 (보안/성능/스타일)
4. 자동 코멘트 작성
5. Slack 알림

---

## 체크리스트

### Agent 기초
- [ ] ReAct 패턴 이해
- [ ] Tool 정의 및 사용
- [ ] Agent 실행 및 디버깅

### LangGraph
- [ ] StateGraph 구성
- [ ] 노드와 엣지 정의
- [ ] 조건부 라우팅
- [ ] 상태 관리

### 멀티 에이전트
- [ ] 역할 분담 설계
- [ ] 에이전트 간 통신
- [ ] 결과 종합

### Tool/MCP
- [ ] 커스텀 Tool 개발
- [ ] MCP 프로토콜 이해
- [ ] 외부 API 연동

### 워크플로우 자동화
- [ ] n8n 또는 Zapier 사용
- [ ] Webhook 처리
- [ ] 알림 시스템 구축

---

## 다음 단계
Step 4 완료 후 → **Step 5: MLOps + 모델 서빙** 으로 진행 (핵심 차별화 영역)
