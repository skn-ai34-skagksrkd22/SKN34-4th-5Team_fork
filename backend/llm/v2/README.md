# llm/v2 체인

LangGraph `StateGraph` 하나에 작업 노드 세 개를 둡니다.

```
START → jev_router ─ NON_PASS → END (jev_router 가 범위 안내 AIMessage 를 붙임)
                   ├ SIMPLE   → simple_agent → END
                   └ COMPLEX  → orchestrator → END
```

## 사용

```python
from langchain_core.messages import AIMessage, HumanMessage
from llm.v2.agent.chain import get_graph

result = get_graph().invoke({
    "messages": [HumanMessage("잠실 가요"), AIMessage("네"), HumanMessage("주차는?")],
    "context": {"stadium": "잠실야구장", "intent": "route", "origin": {"lat": 37.5, "lng": 127.0}},  # 선택
})
answer = result["messages"][-1].content
```

마지막 메시지가 이번 질문입니다. 결과 `messages`는 입력 대화 뒤에 최종 답변 `AIMessage` 하나만 붙습니다. `context`는 신뢰하지 않는 참고 데이터로만 쓰입니다. `get_graph()`는 첫 호출 때 실제 모델(`LLM_MODEL`)과 `llm/tools` 레지스트리로 한 번 조립됩니다. 테스트나 다른 모델은 `build_graph(model, tools_by_name)`를 씁니다.

## 구성

| 파일 | 역할 |
|---|---|
| `agent/chain.py` | 그래프 조립, `jev_router` 진입 노드와 분기 |
| `agent/common.py` | `ChainState`/`Decision`/create_agent state, 모델, `build_agent` |
| `agent/classifier.py` | JEV 한 번 호출로 guard, complexity, capability(Noul) 판정. `jev_router`(chain.py)가 요청마다 한 번 부르고 입력 decision은 덮어씀 |
| `agent/simple_chain.py` | 단일 목적 create_agent. capability에 매핑된 도구만 노출 |
| `agent/orchestrator_chain.py` | `ask_baseball`, `ask_travel_research`, `ask_place_data` 도구로 전문 Agent를 실제 호출하고 `get_directions`로 일정 조율 |
| `agent/baseball_chain.py` | 경기·순위·선수·규칙 + 구장 안 정보 + 야구 커뮤니티 (구 stadium/community 흡수) |
| `agent/travel_chain.py` | 구장 주변 맛집·카페·관광 후보 조사 |
| `agent/place_chain.py` | 공개 코스, 특정 장소 확인 |
| `middleware/jev_guidelines.py` | 승인 확인 + 공통 페르소나·내용·근거·말투 규칙(v1에서 옮겨 v2가 소유) + 선택 context 시스템 프롬프트. 역할 규칙은 각 chain 파일에 있음 |
| `middleware/dynamic_tools.py` | capability → 도구 매핑, 역할 도구와의 교집합만 노출, 실행 직전 allowlist 차단 |

## 제한

- 스트리밍, SSE, 채팅 저장, 세션, checkpointer/saver는 이 모듈에 없습니다. `compile()`만 합니다.
- `llm/service/chat.py`는 `llm.v2.agent.chain.chain` (`.stream({"question","chat_history"}) -> Iterator[str]`)을
  기대하지만 이 모듈은 그 이름을 내보내지 않습니다(`get_graph()`가 받는 입력도 `messages` 기반으로 다릅니다).
  이 연동은 v2 범위 밖의 별도 통합 작업입니다.
- 장소 URL 추출·저장 도구는 없어서 PlaceData Agent는 검색·공개 코스 확인만 합니다.
- JEV 판정은 1차 필터이며 도구의 인증·권한 검사를 대신하지 않습니다. JEV 호출에는 `OPENROUTER_API_KEY`가 필요합니다(첫 판정 때 생성).
- 전문/Simple Agent 는 `AGENT_RECURSION_LIMIT`(기본 12), Orchestrator 는 `ORCHESTRATOR_RECURSION_LIMIT`(기본 25)로 멈춥니다. 전문 Agent 실패는 `[조회 실패] ...`로 Orchestrator에 돌아갑니다.

## 테스트

```
cd backend && python -m unittest llm.v2.tests.test_chain -v
```

fake model, fake tools, mocked JEV로 실제 StateGraph/create_agent 루프를 돌립니다. provider·DB·JEV 네트워크 호출은 없습니다.
