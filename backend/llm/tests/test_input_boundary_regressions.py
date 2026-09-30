"""backend/llm/serializer/message.py + views/message.py 입력 경계 회귀 테스트.

이 슬라이스가 담당하는 두 파일만 검증한다 (기존 tests/test_v2_chat.py는 건드리지 않음):
- Accept: text/event-stream 로 프리스트림 검증 에러(400/404)가 나면 EventStreamRenderer가
  원본 에러 dict를 그대로 돌려줘 Content-Type은 text/event-stream인데 본문은 dict 키만
  바이트로 이어붙인 깨진 값이 됐다 (예: b"content"). 성공 스트림은 StreamingHttpResponse를
  그대로 돌려주고 이 렌더러를 거치지 않으므로 영향 없다.
- content/message_id 경계값(공백만, NUL 문자, 초과 큰 정수)이 이미 DRF 기본 동작으로
  막히는지도 같이 못박아둔다 (회귀 방지용 - 실제로 커스텀 코드를 추가하지는 않았다).

실제 LLM/OpenAI 호출 없음: 이 파일의 모든 테스트는 스트림 시작 전에 막히는 경로만
다루므로 send_message()/체인까지 도달하지 않는다.
"""
from rest_framework.test import APIClient

from llm.models import ChatSession
from llm.tests.test_v2_chat import CheckpointTestCase


class SSEAcceptPrestreamErrorRendererTest(CheckpointTestCase):
    """Accept: text/event-stream + 프리스트림 검증 실패 -> 유효한 JSON 에러 바디."""

    def setUp(self):
        self.client_a = APIClient()
        self.client_a.cookies["guest_id"] = "f0f0f0f0-f0f0-f0f0-f0f0-f0f0f0f0f0f0"
        self.session = ChatSession.objects.create(guest="f0f0f0f0-f0f0-f0f0-f0f0-f0f0f0f0f0f0")

    def test_blank_content_validation_error_returns_valid_json_not_broken_sse(self):
        response = self.client_a.post(
            f"/api/v2/chat/sessions/{self.session.id}/messages/",
            data={"content": ""},
            format="json",
            HTTP_ACCEPT="text/event-stream",
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertIn("content", response.json())

    def test_missing_message_id_on_put_returns_valid_json_not_broken_sse(self):
        response = self.client_a.put(
            f"/api/v2/chat/sessions/{self.session.id}/messages/",
            data={"content": "hello"},
            format="json",
            HTTP_ACCEPT="text/event-stream",
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertIn("message_id", response.json())

    def test_unowned_session_404_returns_valid_json_not_broken_sse(self):
        response = self.client_a.post(
            "/api/v2/chat/sessions/00000000-0000-0000-0000-000000000000/messages/",
            data={"content": "hello"},
            format="json",
            HTTP_ACCEPT="text/event-stream",
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertIn("detail", response.json())

    def test_success_json_accept_path_is_unaffected(self):
        response = self.client_a.get(
            f"/api/v2/chat/sessions/{self.session.id}/messages/",
            HTTP_ACCEPT="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/json")


class ContentBoundaryValidationTest(CheckpointTestCase):
    """content/message_id 경계값이 (이미 있는 DRF 동작으로) 막히는지 회귀 고정."""

    def setUp(self):
        self.client_a = APIClient()
        self.client_a.cookies["guest_id"] = "f1f1f1f1-f1f1-f1f1-f1f1-f1f1f1f1f1f1"
        self.session = ChatSession.objects.create(guest="f1f1f1f1-f1f1-f1f1-f1f1-f1f1f1f1f1f1")

    def test_whitespace_only_content_rejected(self):
        response = self.client_a.post(
            f"/api/v2/chat/sessions/{self.session.id}/messages/",
            data={"content": "   "},
            format="json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("content", response.json())

    def test_null_byte_content_rejected(self):
        response = self.client_a.post(
            f"/api/v2/chat/sessions/{self.session.id}/messages/",
            data={"content": "a\x00b"},
            format="json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("content", response.json())

    def test_huge_message_id_on_delete_resolves_to_404_not_500(self):
        # message_id 는 UUIDField 라 정수는 UUID(int=...) 로 바뀐다 (2**128 미만이면 유효).
        # 그런 id 의 사용자 메시지는 없으므로 service 계층의 Http404 로 404 가 나와야 한다.
        response = self.client_a.delete(
            f"/api/v2/chat/sessions/{self.session.id}/messages/",
            data={"message_id": 10 ** 30},
            format="json",
        )
        self.assertEqual(response.status_code, 404)

