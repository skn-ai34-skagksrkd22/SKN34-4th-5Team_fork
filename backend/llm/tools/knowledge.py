"""Document retrieval shared by answer agents."""
import contextvars
import functools
import json
import os
import re
from django.db import connection, transaction
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool, ToolException, tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

NO_DOCUMENTS_MESSAGE = "관련 문서를 찾을 수 없습니다."

TOP_K = int(os.getenv("VENUE_TOP_K", "10"))
MAX_DISTANCE = float(os.getenv("VENUE_MAX_DISTANCE", "0.5"))
VENUE_CATEGORIES = ["FOOD_IN", "FOOD_OUT", "CAFE", "SPOT", "FACILITY", "CONTENT", "TRANSPORT", "STADIUM", "OPERATION"]

def grade(meta: dict) -> str:
    ev = str(meta.get("evidence_type") or "").upper()
    st = str(meta.get("status") or "").upper()
    if ev == "UNOFFICIAL":
        return "UNOFFICIAL"
    if ev == "THIRD_PARTY_API":
        return "THIRD_PARTY"
    if st in {"CONFIRMED", "CONFIRMED_OFFICIAL", "CONFIRMED_BASELINE"} or (not st and ev in {"", "OFFICIAL", "DERIVED"}):
        return "OFFICIAL"
    return "UNCERTAIN"

def _metadata_dict(metadata) -> dict:
    """DB driver가 반환한 metadata를 항상 dict로 정규화한다."""
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            value = json.loads(metadata)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}
    return {}

def vector_search(query: str, stadium: str | None, categories: list[str] | None,
                  in_stadium_flag: str | None = None) -> list[dict]:
    """구장·카테고리·안팎 조건을 적용한 벡터 검색."""
    from ..v1.rag.club.retrieval import embed
    qvec = embed(query)
    conds = ["metadata->>'category' = ANY(%(cats)s)"]
    params: dict = {"cats": categories or VENUE_CATEGORIES, "v": "[" + ",".join(map(str, qvec)) + "]", "k": TOP_K}
    if stadium:
        conds.append("(metadata->>'stadium_code' = %(st)s OR metadata->>'stadium_code' IS NULL)")
        params["st"] = stadium
    if in_stadium_flag:
        conds.append("metadata->>'in_stadium_flag' = %(flag)s")
        params["flag"] = in_stadium_flag
    sql = f"""
        SELECT content, metadata, embedding <=> %(v)s::vector AS dist
        FROM llm_documentchunk WHERE {' AND '.join(conds)}
        ORDER BY embedding <=> %(v)s::vector LIMIT %(k)s"""
    with transaction.atomic(), connection.cursor() as cur:
        cur.execute("SET LOCAL hnsw.ef_search = 200")
        cur.execute(sql, params)
        rows = cur.fetchall()
    out = []
    for content, meta, dist in rows:
        if float(dist) > MAX_DISTANCE:
            continue
        meta = _metadata_dict(meta)
        out.append({"content": content, "metadata": {**meta, "distance": float(dist), "match_type": "vector"}})
    return out

def _keyword_patterns(query: str) -> list[str]:
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", query)
    aliases = {"포토존": "PHOTOZONE", "사진": "PHOTOZONE", "교통": "TRANSPORT", "대중교통": "TRANSPORT"}
    tokens.extend(aliases[t] for t in tokens if t in aliases)
    return [f"%{t}%" for t in dict.fromkeys(tokens) if len(t) >= 2]

def keyword_fallback_search(query: str, stadium: str | None, categories: list[str] | None,
                            in_stadium_flag: str | None, top_k: int) -> list[dict]:
    """벡터가 부족할 때 content/metadata 를 ILIKE 로 보조 검색 (test.py keyword_fallback_search)"""
    patterns = _keyword_patterns(query)
    if not patterns:
        return []
    conds, params = ["(content ILIKE ANY(%s) OR metadata::text ILIKE ANY(%s))"], [patterns, patterns]
    conds.append("metadata->>'category' = ANY(%s)")
    params.append(categories or VENUE_CATEGORIES)
    if in_stadium_flag:
        conds.append("metadata->>'in_stadium_flag' = %s")
        params.append(in_stadium_flag)
    if stadium:
        conds.append("(metadata->>'stadium_code' = %s OR metadata->>'stadium_code' IS NULL)")
        params.append(stadium)
    params.append(max(top_k * 100, 500))
    with connection.cursor() as cur:
        cur.execute(f"SELECT content, metadata FROM llm_documentchunk WHERE {' AND '.join(conds)} ORDER BY id LIMIT %s", params)
        rows = cur.fetchall()

    def score(row):
        text = f"{row[0]} {json.dumps(row[1] or {}, ensure_ascii=False)}".lower()
        return sum(p.strip("%").lower() in text for p in patterns)

    rows = sorted(rows, key=score, reverse=True)[:top_k]
    return [{"content": c, "metadata": {**_metadata_dict(m), "distance": None, "match_type": "keyword_fallback"}}
            for c, m in rows]

def search_documents(query: str, slots: dict) -> dict:
    """검색어 변환 → 벡터 검색 → 소프트 필터 → 키워드 fallback (test.py search_documents)"""
    transformed = transform_query(query)
    slots = {**infer_slots(f"{query} {transformed}"), **{k: v for k, v in slots.items() if v}}
    stadium, cats, flag = slots.get("stadium_code"), slots.get("categories"), slots.get("in_stadium_flag")

    docs = vector_search(transformed, stadium, cats, flag)
    method = "vector" if docs else "none"

    fetch_k = max(TOP_K * 2, 10)
    if not docs or len(docs) < 3:
        # 내부/외부가 명시된 경우 fallback에서도 그 조건을 풀지 않는다.
        fb = keyword_fallback_search(transformed, stadium, cats, flag, fetch_k)
        if not fb and transformed != query:
            fb = keyword_fallback_search(query, stadium, cats, flag, fetch_k)
        if fb:
            docs, method = fb, "keyword_fallback"
    docs = docs[:fetch_k]
    return {"original_query": query, "transformed_query": transformed, "documents": docs, "search_method": method}

def format_documents(docs: list[dict]) -> str:
    if not docs:
        return NO_DOCUMENTS_MESSAGE
    return "\n\n".join(
        f"[문서 {i}] (등급={grade(d['metadata'])} · 구장={d['metadata'].get('stadium_code')} · 카테고리={d['metadata'].get('category')})\n{d['content']}"
        for i, d in enumerate(docs, 1)
    )

CATEGORIES = {'OPERATION', 'CAFE', 'FOOD_IN', 'FOOD_OUT', 'TRANSPORT', 'STADIUM', 'PRICE', 'SEAT', 'TICKET_POLICY', 'FACILITY', 'CONTENT', 'CARRY_IN', 'REENTRY', 'RULE', 'SPOT'}
DOC_K = 5

def search_kbo_rows(query, stadium=None, categories=None, k=DOC_K, _search=None, _embed=None) -> list[dict]:
    """pgvector 검색 + 키워드 재정렬. 카테고리로 0건이면 카테고리를 풀고 한 번 더."""
    cats = [c.upper() for c in (categories or []) if c and c.upper() in CATEGORIES] or None
    if _search is None:
        from ..v1.rag.club.retrieval import embed, keyword_rerank, search
        _embed = embed
        _search = lambda v, kk, st, ct: keyword_rerank(query, search(v, k=kk, stadium=st, categories=ct)[0], k=k)  # noqa: E731
    vec = _embed(query)
    rows = _search(vec, k * 3, stadium, cats)
    if not rows and cats:
        rows = _search(vec, k * 3, stadium, None)
    return list(rows)[:k]

TEAM_ALIASES: dict[str, tuple[str, str]] = {
    "SSG랜더스": ("SSG", "MUNHAK"), "인천SSG랜더스필드": ("SSG", "MUNHAK"), "SSG랜더스필드": ("SSG", "MUNHAK"),
    "랜더스필드": ("SSG", "MUNHAK"), "쓱랜더스": ("SSG", "MUNHAK"), "문학경기장": ("SSG", "MUNHAK"),
    "문학구장": ("SSG", "MUNHAK"), "인천야구장": ("SSG", "MUNHAK"), "인천구장": ("SSG", "MUNHAK"),
    "랜더스": ("SSG", "MUNHAK"), "SSG": ("SSG", "MUNHAK"), "문학": ("SSG", "MUNHAK"), "쓱": ("SSG", "MUNHAK"),
    "LG트윈스": ("LG", "JAMSIL"), "엘지트윈스": ("LG", "JAMSIL"), "트윈스": ("LG", "JAMSIL"),
    "LG": ("LG", "JAMSIL"), "엘지": ("LG", "JAMSIL"),
    "두산베어스": ("DOOSAN", "JAMSIL"), "베어스": ("DOOSAN", "JAMSIL"), "두산": ("DOOSAN", "JAMSIL"),
    "잠실야구장": ("LG", "JAMSIL"), "잠실구장": ("LG", "JAMSIL"), "잠실": ("LG", "JAMSIL"),
    "KIA타이거즈": ("KIA", "GWANGJU"), "기아타이거즈": ("KIA", "GWANGJU"), "챔피언스필드": ("KIA", "GWANGJU"),
    "광주야구장": ("KIA", "GWANGJU"), "광주구장": ("KIA", "GWANGJU"), "무등구장": ("KIA", "GWANGJU"),
    "타이거즈": ("KIA", "GWANGJU"), "챔필": ("KIA", "GWANGJU"), "KIA": ("KIA", "GWANGJU"), "기아": ("KIA", "GWANGJU"),
    "삼성라이온즈": ("SAMSUNG", "DAEGU"), "라이온즈파크": ("SAMSUNG", "DAEGU"), "대구야구장": ("SAMSUNG", "DAEGU"),
    "대구구장": ("SAMSUNG", "DAEGU"), "라이온즈": ("SAMSUNG", "DAEGU"), "삼성": ("SAMSUNG", "DAEGU"), "라팍": ("SAMSUNG", "DAEGU"),
    "롯데자이언츠": ("LOTTE", "SAJIK"), "사직야구장": ("LOTTE", "SAJIK"), "사직구장": ("LOTTE", "SAJIK"),
    "부산야구장": ("LOTTE", "SAJIK"), "부산구장": ("LOTTE", "SAJIK"), "자이언츠": ("LOTTE", "SAJIK"),
    "롯데": ("LOTTE", "SAJIK"), "사직": ("LOTTE", "SAJIK"),
    "한화이글스": ("HANWHA", "DAEJEON"), "한화생명볼파크": ("HANWHA", "DAEJEON"), "이글스파크": ("HANWHA", "DAEJEON"),
    "대전야구장": ("HANWHA", "DAEJEON"), "대전구장": ("HANWHA", "DAEJEON"), "이글스": ("HANWHA", "DAEJEON"), "한화": ("HANWHA", "DAEJEON"),
    "케이티위즈": ("KT", "SUWON"), "KT위즈": ("KT", "SUWON"), "수원야구장": ("KT", "SUWON"),
    "수원구장": ("KT", "SUWON"), "위즈파크": ("KT", "SUWON"), "케이티": ("KT", "SUWON"), "위즈": ("KT", "SUWON"), "KT": ("KT", "SUWON"),
    "NC다이노스": ("NC", "CHANGWON"), "엔씨다이노스": ("NC", "CHANGWON"), "창원야구장": ("NC", "CHANGWON"),
    "창원구장": ("NC", "CHANGWON"), "마산구장": ("NC", "CHANGWON"), "다이노스": ("NC", "CHANGWON"),
    "엔씨파크": ("NC", "CHANGWON"), "NC파크": ("NC", "CHANGWON"), "엔씨": ("NC", "CHANGWON"), "엔팍": ("NC", "CHANGWON"), "NC": ("NC", "CHANGWON"),
    "키움히어로즈": ("KIWOOM", "GOCHEOK"), "고척스카이돔": ("KIWOOM", "GOCHEOK"), "히어로즈": ("KIWOOM", "GOCHEOK"),
    "고척돔": ("KIWOOM", "GOCHEOK"), "서울돔": ("KIWOOM", "GOCHEOK"), "키움": ("KIWOOM", "GOCHEOK"), "고척": ("KIWOOM", "GOCHEOK"),
}

_ALIASES_LONGEST_FIRST = sorted(TEAM_ALIASES, key=len, reverse=True)   # "잠실야구장" 이 "잠실" 보다 먼저

CATEGORY_RULES = [
    (("교통", "대중교통", "버스", "지하철", "주차", "주차장", "오는길", "가는 법"), ["TRANSPORT"]),
    (("포토존", "포토", "사진", "포토부스", "최정존", "기념존", "전시"), ["CONTENT", "FACILITY", "SPOT"]),
    (("수유실", "수유", "유모차", "의무실", "의무", "물품보관소", "보관소", "짐보관", "화장실", "휠체어", "충전소", "편의시설", "흡연"), ["FACILITY", "STADIUM", "CONTENT"]),
    (("카페", "커피", "음료", "디저트"), ["CAFE", "FOOD_IN", "FOOD_OUT"]),
]

OUTSIDE_WORDS = ("근처", "주변", "외부", "밖")

INSIDE_WORDS = ("내부", "구장 안", "구장안", "안에서", "안에", "1루", "3루", "내야", "외야")

FOOD_WORDS = ("먹거리", "음식점", "식당", "매장", "치킨", "피자", "버거", "스낵", "맛집", "밥", "떡볶이", "만두")

def infer_slots(question: str) -> dict:
    """구장·팀·카테고리·안/밖 추론"""
    slots: dict = {}
    compact = question.replace(" ", "")
    for alias in _ALIASES_LONGEST_FIRST:
        if alias.upper() in compact.upper():
            slots["team_code"], slots["stadium_code"] = TEAM_ALIASES[alias]
            break
    for terms, cats in CATEGORY_RULES:
        if any(t in question for t in terms):
            slots["categories"] = cats
            break
    else:
        if any(t in question for t in OUTSIDE_WORDS) and any(t in question for t in FOOD_WORDS):
            slots["categories"] = ["FOOD_OUT", "CAFE"]
        elif any(t in question for t in FOOD_WORDS):
            # 내부 먹거리 질문에 외부 매장과 카페가 섞이지 않도록 원본 규칙을 유지한다.
            slots["categories"] = ["FOOD_IN", "CONTENT", "FACILITY"]
    if any(t in question for t in OUTSIDE_WORDS):
        slots["in_stadium_flag"] = "N"
    elif any(t in question for t in INSIDE_WORDS):
        slots["in_stadium_flag"] = "Y"
    return slots

_transformer = None

def transform_query(query: str) -> str:
    """검색어만 변환하며 답변 도구 루프를 시작하지 않는다."""
    global _transformer
    if _transformer is None:
        from ..v1.rag.venue.prompts import QUERY_TRANSFORM
        prompt = ChatPromptTemplate.from_messages([("system", QUERY_TRANSFORM), ("human", "{query}")])
        model = ChatOpenAI(model=os.getenv("LLM_MODEL") or "gpt-5.6-luna", temperature=0, timeout=25, max_retries=0, reasoning_effort="medium", use_responses_api=True)
        _transformer = prompt | model | StrOutputParser()
    return _transformer.invoke({"query": query}).strip() or query

search_context: contextvars.ContextVar[dict] = contextvars.ContextVar("venue_search_context")

def _tool_error(function):
    """예외가 SIMPLE 턴 전체를 깨지 않도록 status=error ToolMessage로 바꾼다."""
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ToolException:
            raise
        except Exception as exc:
            raise ToolException(f"[조회 실패] 문서 검색 중 오류가 발생했습니다: {type(exc).__name__}") from None
    return wrapped

@tool
@_tool_error
def search_documents_tool(query: str) -> str:
    """구장 안팎 문서(먹거리·편의시설·교통·포토존·주변 맛집)에서 질문과 관련된 근거를 검색한다."""
    ctx = search_context.get({})
    result = search_documents(query, ctx.get("slots", {}))
    ctx["last"] = result
    return format_documents(result["documents"])

search_documents_tool.handle_tool_error = True

class SearchInput(BaseModel):
    query: str = Field(description="검색할 내용 (예: '고척 주차 요금', '보조배터리 반입')")
    stadium_code: str | None = Field(default=None, description="구장 코드나 이름. 모르면 비운다")
    categories: list[str] | None = Field(default=None, description="TRANSPORT, FOOD_IN, FOOD_OUT, CAFE, SPOT, FACILITY, SEAT, CARRY_IN, REENTRY, RULE 중에서. 모르면 비운다")

assistant_context: contextvars.ContextVar[dict] = contextvars.ContextVar("knowledge_assistant_context")
STADIUMS = {"JAMSIL", "GOCHEOK", "MUNHAK", "SUWON", "DAEJEON", "DAEGU", "GWANGJU", "SAJIK", "CHANGWON"}

def _stadium_code(value):
    code = str(value or "").strip().upper()
    if code in STADIUMS:
        return code
    from ..v1.rag.club.router import detect_stadium
    result = detect_stadium(str(value or ""))
    return result if result in STADIUMS else None

def _kbo_grade(row):
    ev, status = str(row.get("evidence_type") or "").upper(), str(row.get("status") or "").upper()
    if ev == "UNOFFICIAL":
        return "UNOFFICIAL"
    if ev == "THIRD_PARTY_API":
        return "THIRD_PARTY"
    return "OFFICIAL" if status in {"CONFIRMED", "CONFIRMED_OFFICIAL", "CONFIRMED_BASELINE", ""} else "UNCERTAIN"

def _format_kbo(rows):
    if not rows:
        return "검색 결과 없음"
    return "\n\n".join(f"[{i}] 등급={_kbo_grade(row)} · 구장={row.get('stadium') or '공통'} · 분류={row.get('category')}\n{row.get('content')}"
                       for i, row in enumerate(rows, 1))

def search_kbo_documents(query, stadium_code=None, categories=None, _search=None, _embed=None) -> str:
    """구장 안내 문서를 더 찾는다 (이미 받은 참고 문서에 없을 때만)."""
    state = assistant_context.get(None)
    if state is None:
        state = {"hint": None, "question": "", "history": [], "schema_seen": False,
                 "tools": [], "sources": [], "course": None}
        assistant_context.set(state)
    state["tools"].append("rag")
    code = _stadium_code(stadium_code) if stadium_code else state.get("hint")
    rows = search_kbo_rows(query, code, categories, _search=_search, _embed=_embed)
    for row in rows:
        state["sources"].append({"doc_id": row.get("doc_id"), "grade": _kbo_grade(row),
                                 "category": row.get("category"), "stadium": row.get("stadium")})
    return _format_kbo(rows)

def create_knowledge_tools():
    return (
        search_documents_tool,
        StructuredTool.from_function(
            _tool_error(search_kbo_documents), name="search_kbo_documents", args_schema=SearchInput,
            description=search_kbo_documents.__doc__,
            handle_validation_error="도구 인자 형식이 올바르지 않습니다. 설명을 보고 다시 부르세요.",
            handle_tool_error=True,
        ),
    )
