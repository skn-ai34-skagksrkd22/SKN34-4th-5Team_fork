"""Simple: 단일 목적 질문. JEV capability 로 노출 도구를 고르는 create_agent (전문 Agent 호출 없음)."""
from ..middleware.dynamic_tools import CAPABILITY_TOOLS
from .common import build_agent

SIMPLE_RULES = """역할: 단일 목적 안내 에이전트.
허용된 도구만 필요한 만큼 호출해 한 가지 목적의 질문에 답한다. 구장 ID가 필요한 도구는 get_stadium 으로 먼저 확인한다.
질문의 구장이 모호하면 어느 구장인지 되묻는다. 인사·감사·잡담에는 도구 없이 짧게 답한다.
고정 도구로 안 되는 집계만 get_baseball_schema → execute_baseball_select 순서로 조회한다."""

TOOLS = tuple(dict.fromkeys(n for names in CAPABILITY_TOOLS.values() for n in names))


def build(model, tools_by_name):
    return build_agent(model, [tools_by_name[n] for n in TOOLS], SIMPLE_RULES, CAPABILITY_TOOLS)
