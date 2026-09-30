"""backend/llm/serializer/message.py 의 origin lat/lng OverflowError 회귀 테스트.

거대한 JSON 정수 리터럴(예: 10**400)을 origin.lat/lng 로 보내면 float() 변환이
OverflowError 를 던진다. 기존 _validate_context 는 (TypeError, ValueError) 만
잡아서 이 경우 500 으로 터졌었다. POST/PUT 모두 서비스 호출(체인/메시지 수정) 전에
400 으로 막혀야 하고, 유효한 경계값(-90/90, -180/180)은 계속 통과해야 한다.
"""
import json

from rest_framework.test import APIClient

from llm.models import ChatSession
from llm.tests.test_v2_chat import CheckpointTestCase, FakeChain, patch_chain, seed, snapshot

HUGE_POSITIVE = "1" + "0" * 400
HUGE_NEGATIVE = "-" + HUGE_POSITIVE


class CoordinateOverflowRegressionTest(CheckpointTestCase):
    def setUp(self):
        self.client_a = APIClient()
        self.client_a.cookies["guest_id"] = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"
        self.session = ChatSession.objects.create(guest="eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee")

    def _post_raw(self, body_text):
        return self.client_a.post(
            f"/api/v2/chat/sessions/{self.session.id}/messages/",
            data=body_text, content_type="application/json", HTTP_ACCEPT="text/event-stream",
        )

    def test_huge_positive_lat_json_literal_rejected_not_500(self):
        body = '{"content": "질문", "context": {"origin": {"lat": %s, "lng": 127.0}}}' % HUGE_POSITIVE
        response = self._post_raw(body)
        self.assertEqual(response.status_code, 400)

    def test_huge_negative_lat_json_literal_rejected_not_500(self):
        body = '{"content": "질문", "context": {"origin": {"lat": %s, "lng": 127.0}}}' % HUGE_NEGATIVE
        response = self._post_raw(body)
        self.assertEqual(response.status_code, 400)

    def test_huge_positive_lng_json_literal_rejected_not_500(self):
        body = '{"content": "질문", "context": {"origin": {"lat": 37.5, "lng": %s}}}' % HUGE_POSITIVE
        response = self._post_raw(body)
        self.assertEqual(response.status_code, 400)

    def test_huge_negative_lng_json_literal_rejected_not_500(self):
        body = '{"content": "질문", "context": {"origin": {"lat": 37.5, "lng": %s}}}' % HUGE_NEGATIVE
        response = self._post_raw(body)
        self.assertEqual(response.status_code, 400)

    def test_valid_boundary_coordinates_still_accepted(self):
        fake = FakeChain(chunks=("안",))
        context = {"origin": {"lat": 90, "lng": -180}}
        with patch_chain(return_value=fake):
            response = self.client_a.post(
                f"/api/v2/chat/sessions/{self.session.id}/messages/",
                {"content": "질문", "context": context}, format="json", HTTP_ACCEPT="text/event-stream",
            )
            self.assertEqual(response.status_code, 200)
            b"".join(response.streaming_content)
        self.assertEqual(fake.received_inputs["context"]["origin"]["lat"], 90.0)
        self.assertEqual(fake.received_inputs["context"]["origin"]["lng"], -180.0)

    def test_put_huge_lat_rejected_before_message_mutation(self):
        """PUT 도 message_update() 호출(메시지 삭제/수정) 전에 400 으로 막혀야 한다."""
        target = seed(self.session, ("수정 대상 질문", "답변"))[0]
        before = snapshot(self.session)
        fake = FakeChain(chunks=("안",))
        body = json.dumps({
            "content": "수정된 질문",
            "message_id": target.id,
            "context": {"origin": {"lat": int(HUGE_POSITIVE), "lng": 127.0}},
        })
        with patch_chain(return_value=fake):
            response = self.client_a.put(
                f"/api/v2/chat/sessions/{self.session.id}/messages/",
                data=body, content_type="application/json", HTTP_ACCEPT="text/event-stream",
            )
        self.assertEqual(response.status_code, 400)
        self.assertIsNone(fake.received_inputs)  # 체인까지 안 갔다
        self.assertEqual(snapshot(self.session), before)  # 절단 안 됨
