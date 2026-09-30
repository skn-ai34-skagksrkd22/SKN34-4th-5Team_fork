"""미들웨어 2: 동적 도구 할당 + 실행 직전 allowlist. 요청 state 만 읽고 캐시된 agent/도구 배열은 바꾸지 않는다."""
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage


# JEV capability → 기존 llm/tools 실제 도구 이름 (선행 도구 포함). 키는 agent.classifier.CAPABILITIES 와 같다.
CAPABILITY_TOOLS = {
    "schedule": ("get_games",),
    "standings": ("get_standings",),
    "players": ("search_players",),
    "baseball_stats": ("get_baseball_schema", "execute_baseball_select"),
    "rules": ("search_kbo_documents",),
    "stadium_info": (
        "get_stadium", "get_seat_zones", "get_seat_views", "get_seat_maps", "get_ticket_prices",
        "get_ticket_policies", "get_food_stores", "get_facilities", "get_stadium_contents", "get_transport", "search_kbo_documents",
    ),
    "parking_transport": ("get_stadium", "get_transport", "search_kbo_documents"),
    "community": ("search_community_posts", "get_prediction_games", "get_games"),
    "nearby_places": ("get_stadium", "search_places", "search_documents_tool"),
    "tourism": ("get_stadium", "search_tourism", "search_documents_tool"),
    "directions": ("get_stadium", "get_directions"),
    "courses": ("search_courses", "get_course"),
    "weather": ("get_games", "get_stadium", "get_weather"),
}


class DynamicToolMiddleware(AgentMiddleware):
    def __init__(self, role_tools, capability_tools=None):
        super().__init__()
        self.role_tools = frozenset(role_tools)
        self.capability_tools = capability_tools  # None 이면 역할 고정 도구 묶음 그대로

    def allowed(self, state) -> frozenset:
        if self.capability_tools is None:
            return self.role_tools
        capabilities = (state.get("decision") or {}).get("capabilities") or ()
        return self.role_tools & {n for c in capabilities for n in self.capability_tools.get(c, ())}

    def wrap_model_call(self, request, handler):
        allowed = self.allowed(request.state)
        return handler(request.override(tools=[t for t in request.tools if getattr(t, "name", None) in allowed]))

    def wrap_tool_call(self, request, handler):
        name = request.tool_call["name"]
        if name not in self.allowed(request.state):
            return ToolMessage(
                content=f"허용되지 않은 도구입니다: {name}", tool_call_id=request.tool_call["id"], name=name, status="error",
            )
        return handler(request)
