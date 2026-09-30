"""llm/serializer/message.py 공개 채팅 계약 테스트. 실제 LangChain Message/Chunk·astream_events 사용.

외부 LLM/DB 호출 없음 (GenericFakeChatModel, 로컬 @tool, SimpleTestCase).
"""
import asyncio
import json
import uuid

from django.test import SimpleTestCase
from langchain_core.callbacks import adispatch_custom_event
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from llm.serializer.message import (
    ANSWER_TEXT_EVENT, GENERIC_ERROR_MESSAGE, done_payload, error_payload, project_event, project_history,
)
from llm.views.sse import _frame

SENTINELS = ("SYS_SENTINEL", "RAG_SENTINEL", "TOOL_SENTINEL", "ARG_SENTINEL", "READY", "REASON_SENTINEL")


@tool
def search_places(query: str) -> str:
    """장소 검색"""
    return "TOOL_SENTINEL rows"


@tool
def broken_tool(query: str) -> str:
    """항상 실패"""
    raise ValueError("TOOL_SENTINEL internal error")


def _collect(runnable, value, **kwargs):
    async def run():
        out = []
        async for event in runnable.astream_events(value, version="v2"):
            projected = project_event(event, **kwargs)
            if projected is not None:
                out.append(projected)
        return out
    return asyncio.run(run())


def _assert_no_leak(test, payload):
    body = json.dumps(payload, ensure_ascii=False)
    for sentinel in SENTINELS:
        test.assertNotIn(sentinel, body)


def _call(name, call_id):
    return {"name": name, "args": {"query": "ARG_SENTINEL"}, "id": call_id, "type": "tool_call"}


class PublicEventProjectionTest(SimpleTestCase):
    def test_only_final_answer_run_stream_becomes_delta(self):
        # planner 는 도구 호출 없이 READY 만 낸다 -> "tool_calls 없음" 은 최종 답변 근거가 아니다.
        planner = GenericFakeChatModel(messages=iter([AIMessage("READY")]))
        answer = GenericFakeChatModel(messages=iter([AIMessage("잠실 야구장 🙂\n\"안내\"")]))
        answer_run_id = uuid.uuid4()

        async def pipeline(question):
            async for _ in planner.astream([SystemMessage("SYS_SENTINEL RAG_SENTINEL"), HumanMessage(question)]):
                pass
            async for _ in answer.astream([HumanMessage(question)], config={"run_id": answer_run_id}):
                pass
            return "SYS_SENTINEL chain output"

        events = _collect(RunnableLambda(pipeline), "질문", answer_run_id=answer_run_id)

        self.assertTrue(events)
        self.assertEqual({name for name, _ in events}, {"delta"})
        self.assertEqual("".join(data["text"] for _, data in events), "잠실 야구장 🙂\n\"안내\"")
        for _, data in events:
            self.assertEqual(set(data), {"text"})
        _assert_no_leak(self, events)

    def test_without_answer_run_id_no_model_stream_is_public(self):
        model = GenericFakeChatModel(messages=iter([AIMessage("READY")]))
        self.assertEqual(_collect(model, "q"), [])

    def test_fallback_without_model_uses_explicit_answer_event_not_chain_stream(self):
        async def pipeline(_):
            await adispatch_custom_event("internal_debug", "SYS_SENTINEL")
            await adispatch_custom_event(ANSWER_TEXT_EVENT, "야구 관련 질문만 도와드릴 수 있어요.")
            return "RAG_SENTINEL chain output"

        self.assertEqual(_collect(RunnableLambda(pipeline), "파이썬 코딩 알려줘"), [
            ("delta", {"text": "야구 관련 질문만 도와드릴 수 있어요."}),
        ])

    def test_content_blocks_expose_only_text_blocks(self):
        chunk = AIMessageChunk(content=[
            {"type": "reasoning", "reasoning": "REASON_SENTINEL"},
            {"type": "text", "text": "답변"},
        ])
        event = {"event": "on_chat_model_stream", "run_id": "r1", "data": {"chunk": chunk},
                 "metadata": {"thread_id": "SYS_SENTINEL"}, "parent_ids": []}
        self.assertEqual(project_event(event, answer_run_id="r1"), ("delta", {"text": "답변"}))

        reasoning_only = dict(event, data={"chunk": AIMessageChunk(content=[
            {"type": "reasoning", "reasoning": "REASON_SENTINEL"}])})
        self.assertIsNone(project_event(reasoning_only, answer_run_id="r1"))

    def test_non_message_chunk_on_answer_path_fails_instead_of_passing_through(self):
        event = {"event": "on_chat_model_stream", "run_id": "r1", "data": {"chunk": {"raw": "SYS_SENTINEL"}}}
        with self.assertRaises(ValueError):
            project_event(event, answer_run_id="r1")
        with self.assertRaises(ValueError):
            project_event({"event": "on_custom_event", "name": ANSWER_TEXT_EVENT, "data": {"x": 1}})

    def test_repeated_tool_calls_stay_distinct_by_tool_call_id(self):
        # 실행 경계가 run_id 를 지정하고 run_id→tool_call_id 대응표를 명시적으로 넘긴다.
        runs = {"call_a": uuid.uuid4(), "call_b": uuid.uuid4()}
        mapping = {str(run_id): call_id for call_id, run_id in runs.items()}

        async def pipeline(_):
            await search_places.ainvoke(_call("search_places", "call_a"), config={"run_id": runs["call_a"]})
            search_places.invoke(_call("search_places", "call_b"), config={"run_id": runs["call_b"]})
            return "ok"

        events = _collect(RunnableLambda(pipeline), "q", tool_call_ids=mapping)

        self.assertEqual(events, [
            ("tool", {"id": "call_a", "tool_name": "search_places", "status": "running"}),
            ("tool", {"id": "call_a", "tool_name": "search_places", "status": "completed"}),
            ("tool", {"id": "call_b", "tool_name": "search_places", "status": "running"}),
            ("tool", {"id": "call_b", "tool_name": "search_places", "status": "completed"}),
        ])
        _assert_no_leak(self, events)

    def test_nested_tool_start_without_mapping_is_not_guessed(self):
        # 중첩 on_tool_start 에는 인자만 오고 tool_call_id 가 없다(설치 버전 실측).
        # 대응표가 없으면 running 을 추정하지 않고, 종료는 ToolMessage 의 tool_call_id 로만 낸다.
        async def pipeline(_):
            await search_places.ainvoke(_call("search_places", "call_a"))
            return "ok"

        self.assertEqual(_collect(RunnableLambda(pipeline), "q"), [
            ("tool", {"id": "call_a", "tool_name": "search_places", "status": "completed"}),
        ])

    def test_mapping_conflicting_with_tool_message_id_fails(self):
        output = ToolMessage("x", tool_call_id="call_b")
        event = {"event": "on_tool_end", "name": "search_places", "run_id": "r1", "data": {"output": output}}
        with self.assertRaises(ValueError):
            project_event(event, tool_call_ids={"r1": "call_a"})

    def test_tool_exception_is_failed_without_error_detail(self):
        run_id = uuid.uuid4()

        async def pipeline(_):
            try:
                await broken_tool.ainvoke(_call("broken_tool", "call_x"), config={"run_id": run_id})
            except ValueError:
                pass
            return "ok"

        events = _collect(RunnableLambda(pipeline), "q", tool_call_ids={str(run_id): "call_x"})

        self.assertEqual(events, [
            ("tool", {"id": "call_x", "tool_name": "broken_tool", "status": "running"}),
            ("tool", {"id": "call_x", "tool_name": "broken_tool", "status": "failed"}),
        ])
        _assert_no_leak(self, events)

    def test_tool_error_result_message_is_failed(self):
        output = ToolMessage("TOOL_SENTINEL", tool_call_id="call_e", status="error")
        event = {"event": "on_tool_end", "name": "search_places", "data": {"output": output, "input": {}}}
        self.assertEqual(project_event(event),
                         ("tool", {"id": "call_e", "tool_name": "search_places", "status": "failed"}))

    def test_tool_without_tool_call_id_is_not_exposed_or_guessed(self):
        # 일반 dict 입력(tool_call 아님)이면 LangChain 이 tool_call_id 를 주지 않는다. run_id 로 대체하지 않는다.
        self.assertEqual(_collect(search_places, {"query": "ARG_SENTINEL"}), [])

    def test_unknown_or_internal_events_are_ignored(self):
        for kind in ("on_chat_model_start", "on_chat_model_end", "on_chain_stream", "on_chain_end",
                     "on_retriever_end", "on_custom_event", "on_prompt_end", "whatever"):
            event = {"event": kind, "name": "x", "run_id": "r1",
                     "data": {"input": "SYS_SENTINEL", "output": "RAG_SENTINEL", "chunk": "READY"}}
            self.assertIsNone(project_event(event, answer_run_id="r1"), kind)

    def test_error_payload_is_generic(self):
        self.assertEqual(error_payload(), {"detail": GENERIC_ERROR_MESSAGE})

    def test_sse_frame_round_trips_unicode_text(self):
        text = "한글 🙂\n\"quote\""
        frame = _frame("delta", {"text": text})
        self.assertTrue(frame.startswith("event: delta\ndata: "))
        self.assertEqual(json.loads(frame.split("data: ", 1)[1]), {"text": text})


def _turn():
    return [
        SystemMessage("SYS_SENTINEL", id="s1"),
        HumanMessage("잠실 근처 맛집", id="h1"),
        AIMessage("", id="p1", tool_calls=[
            {"name": "search_places", "args": {"query": "ARG_SENTINEL"}, "id": "call_a"},
            {"name": "search_places", "args": {"query": "ARG_SENTINEL"}, "id": "call_b"},
        ], response_metadata={"model_name": "SYS_SENTINEL"}),
        ToolMessage("TOOL_SENTINEL", tool_call_id="call_a", name="search_places"),
        ToolMessage("TOOL_SENTINEL", tool_call_id="call_b", name="search_places", status="error"),
        AIMessage(
            [{"type": "reasoning", "reasoning": "REASON_SENTINEL"}, {"type": "text", "text": "추천 드립니다"}],
            id="a1", response_metadata={"model_name": "SYS_SENTINEL"},
            usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        ),
    ]


DONE = {"h1": {"status": "completed", "answer_id": "a1"}}


class PublicHistoryProjectionTest(SimpleTestCase):
    def test_turn_projects_to_user_and_assistant_with_ordered_tools(self):
        items = project_history(_turn(), DONE)

        self.assertEqual(items, [
            {"id": "h1", "role": "user", "content": "잠실 근처 맛집", "status": "completed", "tools": []},
            {"id": "a1", "role": "assistant", "content": "추천 드립니다", "status": "completed", "tools": [
                {"id": "call_a", "tool_name": "search_places", "status": "completed"},
                {"id": "call_b", "tool_name": "search_places", "status": "failed"},
            ]},
        ])
        _assert_no_leak(self, items)

    def test_tools_are_scoped_per_turn(self):
        messages = _turn() + [HumanMessage("다음 질문", id="h2"), AIMessage("바로 답변", id="a2")]
        items = project_history(messages, {**DONE, "h2": {"status": "completed", "answer_id": "a2"}})
        self.assertEqual(items[-1],
                         {"id": "a2", "role": "assistant", "content": "바로 답변", "status": "completed", "tools": []})

    def test_reused_tool_call_id_in_later_turn_does_not_overwrite_earlier_status(self):
        messages = [
            HumanMessage("q1", id="h1"),
            AIMessage("", id="p1", tool_calls=[{"name": "search_places", "args": {}, "id": "call_1"}]),
            ToolMessage("ok", tool_call_id="call_1"),
            AIMessage("a1", id="a1"),
            HumanMessage("q2", id="h2"),
            AIMessage("", id="p2", tool_calls=[{"name": "search_places", "args": {}, "id": "call_1"}]),
            ToolMessage("bad", tool_call_id="call_1", status="error"),
            AIMessage("a2", id="a2"),
        ]
        turns = {"h1": {"status": "completed", "answer_id": "a1"}, "h2": {"status": "completed", "answer_id": "a2"}}
        items = project_history(messages, turns)
        self.assertEqual(items[1]["tools"], [{"id": "call_1", "tool_name": "search_places", "status": "completed"}])
        self.assertEqual(items[3]["tools"], [{"id": "call_1", "tool_name": "search_places", "status": "failed"}])

    def test_final_answer_comes_only_from_explicit_answer_id(self):
        # 도구 호출 없는 planner READY 는 answer_id 가 아니면 말풍선이 되지 않는다.
        messages = [HumanMessage("q", id="h1"), AIMessage("READY", id="p1"), AIMessage("진짜 답", id="a1")]
        items = project_history(messages, DONE)
        self.assertEqual([item["id"] for item in items], ["h1", "a1"])
        _assert_no_leak(self, items)

    def test_pending_turn_keeps_unresolved_tool_as_running_on_user_item(self):
        messages = [
            HumanMessage("q", id="h1"),
            AIMessage("", id="p1", tool_calls=[{"name": "search_places", "args": {}, "id": "call_a"}]),
        ]
        items = project_history(messages, {"h1": {"status": "pending", "answer_id": None}})
        self.assertEqual(items, [{"id": "h1", "role": "user", "content": "q", "status": "pending", "tools": [
            {"id": "call_a", "tool_name": "search_places", "status": "running"}]}])

    def test_failed_turn_without_ai_message_keeps_tools_without_fake_assistant(self):
        messages = [
            HumanMessage("q", id="h1"),
            AIMessage("", id="p1", tool_calls=[
                {"name": "search_places", "args": {}, "id": "call_a"},
                {"name": "search_places", "args": {}, "id": "call_b"},
            ]),
            ToolMessage("TOOL_SENTINEL", tool_call_id="call_a"),
        ]
        items = project_history(messages, {"h1": {"status": "failed", "answer_id": None}})
        self.assertEqual(items, [{"id": "h1", "role": "user", "content": "q", "status": "failed", "tools": [
            {"id": "call_a", "tool_name": "search_places", "status": "completed"},
            {"id": "call_b", "tool_name": "search_places", "status": "running"},
        ]}])
        _assert_no_leak(self, items)

    def test_missing_tool_result_is_not_proof_of_cancellation(self):
        # 결과 ToolMessage 가 없으면 턴이 실패/취소여도 도구는 running(미해결). 턴 상태는 유지한다.
        messages = [
            HumanMessage("q", id="h1"),
            AIMessage("", id="p1", tool_calls=[{"name": "search_places", "args": {}, "id": "call_a"}]),
        ]
        for status in ("failed", "cancelled"):
            items = project_history(messages, {"h1": {"status": status, "answer_id": None}})
            self.assertEqual(items, [{"id": "h1", "role": "user", "content": "q", "status": status, "tools": [
                {"id": "call_a", "tool_name": "search_places", "status": "running"}]}])

    def test_completed_turn_with_unresolved_tool_call_fails_closed(self):
        messages = [
            HumanMessage("q", id="h1"),
            AIMessage("", id="p1", tool_calls=[{"name": "search_places", "args": {}, "id": "call_a"}]),
            AIMessage("답", id="a1"),
        ]
        with self.assertRaises(ValueError):
            project_history(messages, DONE)

    def test_missing_or_inconsistent_turn_info_fails_closed(self):
        messages = [HumanMessage("q", id="h1"), AIMessage("a", id="a1")]
        for turns in ({}, {"h1": {"status": "done"}}, {"h1": {"status": "completed", "answer_id": None}},
                      {"h1": {"status": "completed", "answer_id": "nope"}}):
            with self.assertRaises(ValueError, msg=turns):
                project_history(messages, turns)

    def test_message_without_id_fails(self):
        with self.assertRaises(ValueError):
            project_history([HumanMessage("q")], {})


class DonePayloadTest(SimpleTestCase):
    def test_done_matches_history_projection(self):
        messages = _turn()
        payload = done_payload(messages, DONE)

        self.assertEqual(payload, {
            "message_id": "a1",
            "assistant_message": "추천 드립니다",
            "tools": project_history(messages, DONE)[-1]["tools"],
        })
        _assert_no_leak(self, payload)

    def test_done_does_not_reuse_earlier_answer_when_latest_turn_is_not_completed(self):
        earlier = [HumanMessage("q1", id="h1"), AIMessage("이전 답", id="a1")]
        cases = (
            ([HumanMessage("q2", id="h2")], {"status": "failed", "answer_id": None}),
            ([HumanMessage("q2", id="h2")], {"status": "pending", "answer_id": None}),
            ([HumanMessage("q2", id="h2"), AIMessage("부분", id="a2")], {"status": "failed", "answer_id": "a2"}),
            ([HumanMessage("q2", id="h2"), AIMessage("부분", id="a2")], {"status": "cancelled", "answer_id": "a2"}),
        )
        for latest, turn in cases:
            turns = {"h1": {"status": "completed", "answer_id": "a1"}, "h2": turn}
            with self.assertRaises(ValueError, msg=turn):
                done_payload(earlier + latest, turns)

    def test_done_requires_saved_completed_assistant_message(self):
        with self.assertRaises(ValueError):
            done_payload([HumanMessage("q", id="h1")], {"h1": {"status": "failed", "answer_id": None}})


class WireStringTypeTest(SimpleTestCase):
    """Enum 도입 뒤에도 공개 dict/튜플에는 Enum 이 아닌 순수 str 이 나간다."""

    def test_history_and_event_values_are_plain_str(self):
        messages = [
            HumanMessage("q", id="h1"),
            AIMessage("", id="t1", tool_calls=[_call("search_places", "c1")]),
            ToolMessage("TOOL_SENTINEL", tool_call_id="c1", name="search_places", status="success", id="r1"),
            AIMessage("a", id="a1"),
        ]
        items = project_history(messages, {"h1": {"status": "completed", "answer_id": "a1"}})
        values = [items[0]["role"], items[1]["role"], items[1]["status"], items[1]["tools"][0]["status"]]
        self.assertEqual(values, ["user", "assistant", "completed", "completed"])
        self.assertTrue(all(type(v) is str for v in values))
        frame = project_event({"event": "on_tool_start", "name": "search_places", "run_id": "r",
                               "data": {"input": _call("search_places", "c1")}})
        self.assertEqual(frame, ("tool", {"id": "c1", "tool_name": "search_places", "status": "running"}))
        self.assertIs(type(frame[0]), str)
        self.assertIs(type(frame[1]["status"]), str)
