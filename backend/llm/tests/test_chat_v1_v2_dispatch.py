"""views.message.ChatMessageView 의 버전별 직접 분기(chat_v1/chat_v2) + 각 경로의 작은 회귀.

llm.service.chat(파사드)는 안 건드린다 -- 기존 64개 호출부는 test_v2_chat.py 등에서 그대로 검증된다.
여기는 View 가 실제로 chat_v1.send_message / chat_v2.send_message 를 직접 부르는지, 그리고 각각의
실제 구현(v1 astream_events 계약, v2 get_graph()+messages)이 살아있는지만 본다.
"""
import uuid
from unittest.mock import patch as mock_patch

from rest_framework.test import APIClient

from llm.models import ChatSession
from llm.service import chat_v1, chat_v2
from llm.service.chat import ChatThread
from llm.tests.test_v2_chat import CheckpointTestCase, FakeChain, v1_pipeline, v2_chain


class ViewDispatchTest(CheckpointTestCase):
    """message.py 가 version 값으로 chat_v1/chat_v2 의 send_message 를 직접 고르는지."""

    def setUp(self):
        self.client = APIClient()
        self.session = ChatSession.objects.create(guest="99999999-9999-9999-9999-999999999999")
        self.client.cookies["guest_id"] = str(self.session.guest)

    def test_v1_url_calls_chat_v1_send_message_directly(self):
        with mock_patch.object(chat_v1, "send_message") as v1_send, \
             mock_patch.object(chat_v2, "send_message") as v2_send:
            v1_send.return_value = iter([("done", {"message_id": None, "assistant_message": None, "tools": []})])
            self.client.post(
                f"/api/v1/chat/sessions/{self.session.id}/messages/",
                {"content": "안녕"}, format="json",
            )
        v1_send.assert_called_once()
        v2_send.assert_not_called()

    def test_v2_url_calls_chat_v2_send_message_directly(self):
        with mock_patch.object(chat_v1, "send_message") as v1_send, \
             mock_patch.object(chat_v2, "send_message") as v2_send:
            v2_send.return_value = iter([("done", {"message_id": None, "assistant_message": None, "tools": []})])
            self.client.post(
                f"/api/v2/chat/sessions/{self.session.id}/messages/",
                {"content": "안녕"}, format="json",
            )
        v2_send.assert_called_once()
        v1_send.assert_not_called()


class V1RegressionTest(CheckpointTestCase):
    """V1: astream_events 계약 그대로 -- SSE done 프레임까지 실제 chat_v1.py 경로로 저장되는지."""

    def setUp(self):
        self.client = APIClient()
        self.session = ChatSession.objects.create(guest="88888888-8888-8888-8888-888888888888")
        self.client.cookies["guest_id"] = str(self.session.guest)

    def test_v1_send_message_streams_done_and_saves_turn(self):
        with mock_patch.object(v1_pipeline, "chat_chain", return_value=FakeChain(chunks=("안", "녕"))):
            response = self.client.post(
                f"/api/v1/chat/sessions/{self.session.id}/messages/",
                {"content": "테스트 질문"}, format="json",
            )
            body = b"".join(response.streaming_content).decode()
        self.assertIn("event: done", body)
        messages, turns = ChatThread(self.session.id).state()
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1].content, "안녕")
        self.assertEqual(list(turns.values())[0]["status"], "completed")


class FakeCompiledGraph:
    """get_graph() 가 돌려주는 실제 Pregel 대신 .stream(messages+updates, subgraphs) 만 흉내낸다.
    분류기·전문 Agent 내부 토큰도 섞어 흘려 공개 답변만 나가는지 본다."""

    def __init__(self, final_text):
        self.final_text = final_text
        self.received_input = None
        self.received_config = None


    def stream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
        self.received_input = graph_input
        self.received_config = config
        self.received_stream_mode = stream_mode
        self.received_subgraphs = subgraphs
        from langchain_core.messages import AIMessage, AIMessageChunk
        yield ("jev_router:1",), "messages", (AIMessageChunk("INTERNAL_CLASSIFIER"), {"langgraph_node": "model"})
        yield ("simple_agent:1", "tools:2"), "messages", (AIMessageChunk("INTERNAL_SPECIALIST"), {"langgraph_node": "model"})
        yield ("simple_agent:1",), "messages", (AIMessageChunk(self.final_text), {"langgraph_node": "model"})
        yield (), "updates", {"simple_agent": {"messages": [AIMessage(self.final_text)]}}


class V2RegressionTest(CheckpointTestCase):
    """V2: get_graph() 를 실제로 호출하고 messages/context/thread_id 로 그래프를 돌리는지."""

    def setUp(self):
        self.client = APIClient()
        self.session = ChatSession.objects.create(guest="77777777-7777-7777-7777-777777777777")
        self.client.cookies["guest_id"] = str(self.session.guest)

    def test_v2_send_message_uses_get_graph_messages_and_thread_id(self):
        fake_graph = FakeCompiledGraph("v2 답변")
        with mock_patch.object(v2_chain, "get_graph", return_value=fake_graph):
            response = self.client.post(
                f"/api/v2/chat/sessions/{self.session.id}/messages/",
                {"content": "v2 질문", "context": {"stadium": "잠실야구장"}}, format="json",
            )
            body = b"".join(response.streaming_content).decode()
        self.assertIn("event: done", body)
        self.assertIn("v2 답변", body)

        self.assertIsNone(fake_graph.received_config)  # checkpointer 없는 그래프: thread_id 는 CRUD(ChatThread) 에만
        self.assertEqual(fake_graph.received_stream_mode, ["messages", "updates"])
        self.assertTrue(fake_graph.received_subgraphs)
        self.assertNotIn("INTERNAL_", body)
        self.assertEqual(body.count("event: delta"), 1)
        human_messages = [m for m in fake_graph.received_input["messages"] if m.type == "human"]
        self.assertEqual(human_messages[-1].content, "v2 질문")
        self.assertEqual(fake_graph.received_input["context"], {"stadium": "잠실야구장"})

        messages, turns = ChatThread(self.session.id).state()
        self.assertEqual(messages[-1].content, "v2 답변")
        self.assertEqual(list(turns.values())[0]["status"], "completed")
