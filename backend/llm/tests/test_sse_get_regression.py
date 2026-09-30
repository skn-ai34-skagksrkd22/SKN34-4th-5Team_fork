"""backend/llm/views/message.py 회귀 테스트 (이 슬라이스 담당 파일만, test_v2_chat.py는 건드리지 않음).

EventStreamRenderer 는 Accept: text/event-stream 협상을 허용만 하고 실제로는 일반
Response(list/dict) 를 항상 JSONRenderer 로 위임해야 한다. 이전에는 response.exception
이 없는 성공 GET(목록 조회) 경로에서 data 를 그대로 반환해 Django 가 리스트를 바이트로
이어붙인 깨진 본문을 text/event-stream Content-Type 으로 내려보냈다. POST/PUT 성공
스트림은 StreamingHttpResponse 라 이 렌더러 자체를 거치지 않으므로 별도로 확인한다.

실제 LLM/OpenAI 호출 없음: GET 목록 조회 경로만 다루므로 send_message()/체인에 도달하지 않는다.
"""
from rest_framework.test import APIClient

from llm.models import ChatSession
from llm.tests.test_v2_chat import CheckpointTestCase, seed


class SSEAcceptGetListRegressionTest(CheckpointTestCase):
    """Accept: text/event-stream + 성공 GET 목록 조회 -> 유효한 JSON 바디."""

    def setUp(self):
        self.client_a = APIClient()
        self.client_a.cookies["guest_id"] = "e0e0e0e0-e0e0-e0e0-e0e0-e0e0e0e0e0e0"
        self.session = ChatSession.objects.create(guest="e0e0e0e0-e0e0-e0e0-e0e0-e0e0e0e0e0e0")
        seed(self.session, ("hello", "hi"))

    def test_get_list_with_sse_accept_returns_valid_json_not_broken_body(self):
        response = self.client_a.get(
            f"/api/v2/chat/sessions/{self.session.id}/messages/",
            HTTP_ACCEPT="text/event-stream",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/json")
        body = response.json()
        self.assertIsInstance(body, list)
        self.assertEqual([item["content"] for item in body], ["hello", "hi"])

    def test_get_list_prestream_404_with_sse_accept_returns_valid_json(self):
        response = self.client_a.get(
            "/api/v2/chat/sessions/00000000-0000-0000-0000-000000000000/messages/",
            HTTP_ACCEPT="text/event-stream",
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertIn("detail", response.json())

    def test_get_list_with_json_accept_is_unaffected(self):
        response = self.client_a.get(
            f"/api/v2/chat/sessions/{self.session.id}/messages/",
            HTTP_ACCEPT="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/json")

    def test_post_success_stream_is_still_a_raw_streaminghttpresponse(self):
        """POST 성공 응답은 StreamingHttpResponse 라 EventStreamRenderer 를 거치지 않는다."""
        from django.http import StreamingHttpResponse
        from unittest.mock import patch

        with patch("llm.service.chat_v2.send_message", return_value=iter([("delta", {"text": "hi"})])):
            response = self.client_a.post(
                f"/api/v2/chat/sessions/{self.session.id}/messages/",
                data={"content": "hello"},
                format="json",
                HTTP_ACCEPT="text/event-stream",
            )
        self.assertIsInstance(response, StreamingHttpResponse)
        self.assertEqual(response["Content-Type"], "text/event-stream")
