"""공용 state + 전문/단일 Agent 공통 조립: create_agent + JEV 가이드라인/동적 도구 미들웨어."""
import os
from functools import cache
from typing import NotRequired, TypedDict

from langchain.agents.middleware import AgentMiddleware, AgentState, ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState

from ..middleware.dynamic_tools import DynamicToolMiddleware
from ..middleware.jev_guidelines import JevGuidelineMiddleware


# 상위 그래프와 create_agent 공용 state. 인증/세션/저장 필드는 두지 않는다.
class Decision(TypedDict):
    allowed: bool  # JEV guard PASS 여부
    complexity: str  # "SIMPLE" | "COMPLEX"
    capabilities: list[str]  # Simple 도구 노출용 capability 이름


class ChainState(MessagesState):
    decision: NotRequired[Decision]
    context: NotRequired[dict | None]  # 선택: {"stadium", "intent", "origin": {"lat", "lng"}}


class V2AgentState(AgentState):
    decision: NotRequired[Decision]
    context: NotRequired[dict | None]


RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "12"))
# Orchestrator 는 전문 Agent 3개 + 이동 시간 조회를 한 턴에 부르므로 전문 Agent 보다 넉넉하게 둔다.
ORCHESTRATOR_RECURSION_LIMIT = int(os.getenv("ORCHESTRATOR_RECURSION_LIMIT", "25"))


@cache
def llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL") or "gpt-5.6-luna", temperature=0, timeout=25,
        max_retries=0, reasoning_effort="medium", use_responses_api=True,
    )


class FinalAnswerMiddleware(AgentMiddleware):
    """recursion 한도 전에 마지막 model 호출을 도구 없이 해서, 도구 실패·반복이 있어도 모은 결과로 답하고 끝나게 한다.

    한 도구 왕복 = model + ToolCallLimit.after_model + tools = 3 step, 마지막 답 = 2 step 이라
    limit 안의 model 호출 수는 (limit - 2) // 3 + 1 이다. 이번 요청의 호출 수 = 마지막 Human 뒤 AIMessage 수."""

    def __init__(self, limit):
        super().__init__()
        self.max_calls = max(1, (limit - 2) // 3 + 1)

    def wrap_model_call(self, request, handler):
        messages = request.state["messages"]
        start = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
        if sum(isinstance(m, AIMessage) for m in messages[start + 1:]) >= self.max_calls - 1:
            request = request.override(tools=[])
        return handler(request)


def build_agent(model, tools, rules, capability_tools=None, limit=RECURSION_LIMIT):
    """create_agent 를 JEV 가이드라인 + 동적 도구 노출 + 종료 보장 미들웨어와 함께 조립한다.

    capability_tools 가 None 이면 role_tools(주어진 tools) 그대로 노출한다(구성 시점에 고정된
    전문 Agent). capability_tools 를 주면 decision.capabilities 와의 교집합만 노출한다(Simple).
    limit: invoke_agent 에 넘길 recursion_limit. 그 안에서 도구 없는 최종 답으로 끝난다(FinalAnswerMiddleware).
    get_directions 는 요청당 2회까지만 실행하고(외부 429 반복 방지), 넘으면 오류 ToolMessage 로 모델이 다음으로 간다."""
    from langchain.agents import create_agent
    return create_agent(
        model=model, tools=tools, state_schema=V2AgentState,
        middleware=[
            JevGuidelineMiddleware(rules),
            DynamicToolMiddleware([t.name for t in tools], capability_tools),
            FinalAnswerMiddleware(limit),
            ToolCallLimitMiddleware(tool_name="get_directions", run_limit=2),
        ],
    )


def invoke_agent(agent, state, limit=RECURSION_LIMIT):
    return agent.invoke(state, {"recursion_limit": limit})


def final_text(result) -> str:
    """에이전트 결과에서 도구 호출이 없는 마지막 AI 답변만 꺼낸다."""
    for msg in reversed(result.get("messages") or []):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            return msg.text
    return ""
