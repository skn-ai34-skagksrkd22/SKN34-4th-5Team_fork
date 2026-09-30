"""장소 확인 전문 Agent: 기존 공개 코스와 특정 장소가 실제 검색되는지 확인한다 (구 course_chain 확인 역할)."""
from .common import build_agent

PLACE_DATA_RULES = """역할: 특정 장소·공개 코스 확인 전문 에이전트.
search_courses·get_course 로 기존 공개 코스를, get_stadium·search_places 로 특정 장소가 실제 검색되는지 확인한다.
장소 상세 페이지·URL 추출·저장 기능은 없으니 지원한다고 말하지 않는다. 확인된 대상과 확인 못 한 대상을 나눠 돌려준다."""

TOOLS = ("search_courses", "get_course", "get_stadium", "search_places")


def build(model, tools_by_name):
    return build_agent(model, [tools_by_name[n] for n in TOOLS], PLACE_DATA_RULES)
