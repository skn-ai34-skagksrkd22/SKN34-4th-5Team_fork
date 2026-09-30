"""야구 전문 Agent: 경기·순위·선수·규칙 + 구장 안 정보 + 야구 커뮤니티 (구 stadium/community 흡수)."""
from .common import build_agent

BASEBALL_RULES = """역할: 야구 전문 에이전트 (경기·순위·선수·규칙, 구장 안 정보, 야구 커뮤니티).
일정·결과·순위·선수는 get_games·get_standings·search_players, 구장 안 가격·좌석·교통·시설·매점은 구장 도구,
규칙·반입·재입장은 search_kbo_documents 결과만 근거로 쓴다. 고정 도구로 안 되는 집계만 get_baseball_schema →
execute_baseball_select 순서로 조회한다. 게시글은 개인 의견으로 전하고 팬 투표는 실제 승리 확률이 아니라고 밝힌다.
받은 하위 작업만 처리하고 결과를 사실 위주로 정리해 돌려준다. 하루 일정을 확정하지 않는다."""

TOOLS = (
    "get_games", "get_standings", "search_players", "get_baseball_schema", "execute_baseball_select",
    "get_stadium", "get_seat_zones", "get_seat_views", "get_seat_maps", "get_ticket_prices",
    "get_ticket_policies", "get_transport", "get_food_stores", "get_facilities", "get_stadium_contents",
    "search_community_posts", "get_prediction_games",
    "search_kbo_documents",
)


def build(model, tools_by_name):
    return build_agent(model, [tools_by_name[n] for n in TOOLS], BASEBALL_RULES)
