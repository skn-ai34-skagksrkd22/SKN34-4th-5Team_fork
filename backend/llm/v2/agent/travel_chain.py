"""주변 후보 조사 전문 Agent: 구장 주변 맛집·카페·관광·실내활동 후보 조사."""
from .common import build_agent

TRAVEL_RESEARCH_RULES = """역할: 구장 주변 후보 조사 전문 에이전트.
get_stadium 으로 좌표를 확인한 뒤 search_places·search_tourism·search_documents_tool 로 조건에 맞는 맛집·카페·관광·
실내활동 후보를 찾는다. 날씨 조건이 있으면 get_weather 로 확인한다. 후보 목록과 근거만 돌려주고 최종 하루 일정은
확정하지 않는다."""

TOOLS = ("get_stadium", "search_places", "search_tourism", "search_documents_tool", "get_weather")


def build(model, tools_by_name):
    return build_agent(model, [tools_by_name[n] for n in TOOLS], TRAVEL_RESEARCH_RULES)
