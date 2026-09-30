from typing import Any

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, Field, StrictInt

from baseball.query_repository import BaseballQueryError
from baseball.query_service import BaseballQueryService, BaseballQueryValidationError


class ExecuteBaseballSelectInput(BaseModel):
    sql: str = Field(
        description=(
            "get_baseball_schema로 확인한 public 야구 테이블의 quoted_name만 사용하는 "
            "단일 PostgreSQL SELECT. JOIN, 집계, 서브쿼리, 비재귀 CTE를 포함할 수 있다."
        )
    )
    params: dict[str, Any] | None = Field(
        default=None,
        description="SQL의 %(name)s named parameter 값. 문자열 보간 대신 사용한다.",
    )
    max_rows: StrictInt = Field(
        default=100,
        description="반환할 최대 행 수. 서비스 설정 범위 안의 정수여야 한다.",
    )


def create_baseball_tools(service: BaseballQueryService | None = None):
    """기존 BaseballQueryService를 호출하는 LangChain 도구 두 개를 만든다."""
    service = service or BaseballQueryService()

    def schema() -> dict:
        """SQL 작성 전에 public 야구 테이블, quoted_name, 컬럼, FK 관계를 조회한다."""
        try:
            return service.get_baseball_schema()
        except Exception as exc:
            raise ToolException(f"[조회 실패] 스키마 조회 오류: {type(exc).__name__}") from None

    def select(sql: str, params: dict[str, Any] | None = None, max_rows: int = 100) -> dict:
        """스키마에 있는 quoted 테이블만 대상으로 단일 읽기 SELECT를 실행한다."""
        try:
            return service.execute_baseball_select(sql, params, max_rows)
        except (BaseballQueryValidationError, BaseballQueryError) as exc:
            raise ToolException(str(exc)) from None

    invalid_input = "도구 입력 형식이 올바르지 않습니다. 스키마와 인자 설명을 확인하세요."
    return (
        StructuredTool.from_function(
            schema,
            name="get_baseball_schema",
            description=(
                "야구 SQL 작성의 첫 단계로 호출한다. public 스키마의 허용 테이블 quoted_name, "
                "컬럼 타입, PK/null 여부, FK 관계를 반환한다."
            ),
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            select,
            name="execute_baseball_select",
            description=(
                "get_baseball_schema 결과를 바탕으로 PostgreSQL 단일 SELECT를 실행한다. "
                "quoted 테이블 JOIN, 집계, 서브쿼리, 비재귀 CTE와 %(name)s named params를 "
                "지원하며 결과 행 수는 max_rows로 제한한다."
            ),
            args_schema=ExecuteBaseballSelectInput,
            handle_tool_error=True,
            handle_validation_error=invalid_input,
        ),
    )


get_baseball_schema, execute_baseball_select = create_baseball_tools()


"""baseball domain tools."""
from datetime import date
from pydantic import Field, StrictInt, model_validator

from .common import LimitInput, _json, _result, _rows, _tool

class StandingsInput(LimitInput):
    snapshot_date: date | None = None

class GamesInput(LimitInput):
    start_date: date
    end_date: date
    team_code: str | None = Field(default=None, pattern="^[A-Z]{2}$")
    stadium_id: StrictInt | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_range(self):
        if self.start_date > self.end_date or (self.end_date - self.start_date).days > 366:
            raise ValueError("날짜 범위는 순서대로 최대 366일이어야 합니다.")
        from community.models import TEAM_CODES
        if self.team_code and self.team_code not in TEAM_CODES:
            raise ValueError("올바른 팀 코드가 아닙니다.")
        return self

class PlayerInput(LimitInput):
    team_code: str | None = Field(default=None, pattern="^(SS|KT|LG|HT|OB|NC|HH|LT|SK|WO)$")
    player_code: str | None = Field(default=None, min_length=1, max_length=40)
    name: str | None = Field(default=None, min_length=1, max_length=80)

    @model_validator(mode="after")
    def any_filter(self):
        if not any((self.team_code, self.player_code, self.name)):
            raise ValueError("구단, 선수 코드, 이름 중 하나가 필요합니다.")
        return self

def create_baseball_domain_tools():
    from django.db.models import Q
    from baseball.models import Game, StandingHistory
    from tving import service as tving_service

    def get_standings(snapshot_date=None, limit=20):
        """정확한 날짜 또는 저장된 최신 날짜의 KBO 순위를 조회한다."""
        freshness = tving_service.ensure_standings_fresh(snapshot_date)
        actual = snapshot_date or StandingHistory.objects.order_by("-snapshot_date").values_list("snapshot_date", flat=True).first()
        rows = [] if actual is None else _rows(
            StandingHistory.objects.filter(snapshot_date=actual).order_by("rank", "team__team_code"),
            ("team__team_code", "team__team_name_ko", "snapshot_date", "rank", "wins", "losses", "draws", "games_behind"), limit,
        )
        return _result(rows, requested_date=_json(snapshot_date), actual_date=_json(actual) if rows else None, **freshness)

    def get_games(start_date, end_date, team_code=None, stadium_id=None, limit=20):
        """날짜 범위의 일정과 결과를 팀/구장으로 필터링한다."""
        freshness = tving_service.ensure_game_range_fresh(start_date, end_date)
        query = Game.objects.filter(game_date__range=(start_date, end_date))
        if team_code:
            query = query.filter(Q(home_team__team_code=team_code) | Q(away_team__team_code=team_code))
        if stadium_id is not None:
            query = query.filter(stadium_id=stadium_id)
        return _result(_rows(query.order_by("game_date", "game_time", "game_code"), (
            "id", "game_code", "game_date", "game_time", "home_team__team_code", "home_team__team_name_ko",
            "away_team__team_code", "away_team__team_name_ko", "stadium_id", "stadium__stadium_name_ko",
            "home_score", "away_score", "status_code", "game_type",
        ), limit), **freshness)

    def search_players(team_code=None, player_code=None, name=None, limit=20):
        """TVING 공통 DB-first 경로로 선수 명단/상세를 갱신한 뒤 공개 선수 정보를 찾는다."""
        stale, warning = False, None
        try:
            teams = [team_code] if team_code else []
            if name and not teams:
                saved, _ = tving_service.search_entities(kind="player", page_size=100)
                needle = name.casefold()
                teams = sorted({row["teamCode"] for row in saved if needle in row["name"].casefold()})
            for code in teams:
                refreshed = tving_service.refresh_team(code)
                stale, warning = stale or refreshed["stale"], warning or refreshed["warning"]
            if player_code:
                refreshed = tving_service.refresh_athlete(player_code)
                stale, warning = stale or refreshed["stale"], warning or refreshed["warning"]
        except tving_service.TvingError:
            stale, warning = True, "최신 선수 정보를 확인하지 못해 저장된 자료만 조회합니다."
        rows, _ = tving_service.search_entities(kind="player", team=team_code, player=player_code, page_size=100)
        if name:
            needle = name.casefold()
            rows = [row for row in rows if needle in row["name"].casefold()]
        return _result(rows[:limit], stale=stale, warning=warning)

    specs = (
        (get_standings, 'get_standings', '정확한 날짜 또는 최신 저장 스냅샷의 순위와 실제 날짜를 반환한다.', StandingsInput),
        (get_games, 'get_games', '날짜 범위의 실제 일정/결과를 팀 또는 구장으로 좁힌다.', GamesInput),
        (search_players, 'search_players', 'TVING DB-first 최신성 경로로 구단/코드/이름에 맞는 선수를 조회한다.', PlayerInput),
    )
    return tuple(_tool(*spec) for spec in specs)
