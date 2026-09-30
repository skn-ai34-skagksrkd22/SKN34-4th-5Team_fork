"""Orchestrator: 전문 Agent 를 도구로 실제 호출하고 결과로 코스·일정을 조율한다."""
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage

from . import baseball_chain, place_chain, travel_chain
from .common import ORCHESTRATOR_RECURSION_LIMIT, build_agent, final_text, invoke_agent

ORCHESTRATOR_RULES = """역할: 경기 전후 코스·하루 일정 조율 에이전트.
직접 조회하지 말고 전문 에이전트에게 하위 작업을 구체적으로 맡긴다.
- ask_baseball: 경기 시각·구장·구장 안 정보
- ask_travel_research: 조건에 맞는 맛집·카페·관광·실내활동 후보
- ask_place_data: 기존 공개 코스, 특정 장소 확인
구장이 정해지지 않았으면 ask_baseball 결과로 구장을 확인한 뒤 장소를 조사한다. 부족한 정보가 있으면 필요한 전문
에이전트만 다시 부른다. 후보 사이 이동 시간은 get_directions 로 확인한다.
get_directions 가 실패하면 한 번까지만 다시 부르고, 그래도 실패하면 이동 시간을 미확인으로 밝히고 그대로 답한다.
받은 결과만으로 시간 순서의 계획을 만들고 경기 시작 전에 구장에 도착하게 짠다. 확인 안 된 시각·영업시간은 단정하지
않고, 조회 실패나 결과 충돌은 그대로 밝힌다."""

SPECIALISTS = {
    "ask_baseball": (baseball_chain, "야구 전문 에이전트: 경기 일정·시각, 구장 확인, 구장 안 정보, 야구 커뮤니티를 조회해 돌려준다."),
    "ask_travel_research": (travel_chain, "주변 후보 조사 에이전트: 구장 주변 맛집·카페·관광·실내활동 후보를 찾아 돌려준다."),
    "ask_place_data": (place_chain, "장소 확인 에이전트: 기존 공개 코스와 특정 장소가 실제 검색되는지 확인해 돌려준다."),
}


def _delegate(name, description, agent):
    @tool(name, description=description)
    def ask(task: str, runtime: ToolRuntime) -> str:
        # 전문 Agent 에는 작업 문자열과 선택 context 만 넘긴다. 승인(decision)은 작업 문자열이 아닌 내부 state 로 전달.
        try:
            result = invoke_agent(agent, {
                "messages": [HumanMessage(task)],
                "context": runtime.state.get("context"),
                "decision": runtime.state.get("decision"),
            })
        except Exception as exc:  # 한도 초과/조회 실패를 결과 없음과 구분해 Orchestrator 에 알린다
            return f"[조회 실패] {name}: {type(exc).__name__}"
        return final_text(result) or f"[조회 실패] {name}: 답변 없음"
    return ask


def build(model, tools_by_name):
    delegates = [
        _delegate(name, desc, module.build(model, tools_by_name)) for name, (module, desc) in SPECIALISTS.items()
    ]
    return build_agent(model, [*delegates, tools_by_name["get_directions"]], ORCHESTRATOR_RULES, limit=ORCHESTRATOR_RECURSION_LIMIT)
