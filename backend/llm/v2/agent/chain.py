"""상위 StateGraph 조립: jev_router 로 승인/복잡도/capability 를 판정하고 simple/orchestrator 로 분기한다."""
from functools import cache

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from ..middleware.jev_guidelines import SCOPE_MESSAGE
from . import classifier, orchestrator_chain, simple_chain
from .common import ORCHESTRATOR_RECURSION_LIMIT, RECURSION_LIMIT, ChainState, invoke_agent


def _last_question_and_history(messages):
    """마지막 HumanMessage 를 이번 질문으로, 그 앞을 history 로 나눈다."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i].content, messages[:i]
    return "", messages


def jev_router(state):
    """요청당 JEV 한 번. 거절이면 안내 메시지를 여기서 붙이고 바로 END 로 간다 (별도 노드 없음).
    분류기 예외는 그대로 올려 모델·도구 실행 없이 실패한다 (fail closed)."""
    question, history = _last_question_and_history(state["messages"])
    decision = classifier.classify(question, history, state.get("context"))
    if decision["allowed"] is not True:
        return {"decision": decision, "messages": [AIMessage(SCOPE_MESSAGE)]}
    return {"decision": decision}


def _route(state) -> str:
    decision = state["decision"]
    if decision["allowed"] is not True:
        return END
    return "orchestrator" if decision["complexity"] == "COMPLEX" else "simple_agent"


def build_graph(model, tools_by_name):
    simple_agent = simple_chain.build(model, tools_by_name)
    orchestrator = orchestrator_chain.build(model, tools_by_name)

    def runner(agent, limit=RECURSION_LIMIT):
        def run(state):  # 내부 tool/AI 메시지는 부모에 복제하지 않고 최종 공개 답변 하나만 돌려준다
            return {"messages": [invoke_agent(agent, state, limit)["messages"][-1]]}
        return run

    graph = StateGraph(ChainState)
    graph.add_node("jev_router", jev_router)
    graph.add_node("simple_agent", runner(simple_agent))
    graph.add_node("orchestrator", runner(orchestrator, ORCHESTRATOR_RECURSION_LIMIT))

    graph.add_edge(START, "jev_router")
    graph.add_conditional_edges("jev_router", _route, ["simple_agent", "orchestrator", END])
    graph.add_edge("simple_agent", END)
    graph.add_edge("orchestrator", END)

    return graph.compile()


@cache
def get_graph():
    from llm.tools import create_default_tools
    from llm.tools.knowledge import create_knowledge_tools
    from .common import llm
    tools_by_name = {t.name: t for t in (*create_default_tools(), *create_knowledge_tools())}
    return build_graph(llm(), tools_by_name)
