"""배포 회귀 4건: 일반 질문 선형화, 첫 순회 전 close 정리, 공개 wire 호환, legacy 대화 이관/삭제."""
from unittest.mock import patch

from django.core.signals import request_finished
from django.db import close_old_connections

from llm.models import ChatSession
from llm.service import chat as chat_service
from llm.tests.test_v2_chat import CheckpointTestCase, PausingChain, history, patch_chain, snapshot
from llm.views.sse import event_stream_response


class OrdinarySendLinearizationTest(CheckpointTestCase):
    def setUp(self):
        self.session = ChatSession.objects.create(guest="12121212-1212-1212-1212-121212121212")

    def test_concurrent_sends_from_same_base_both_stay_in_root_history(self):
        with patch_chain(return_value=PausingChain(chunks=("답",))):
            a = chat_service.send_message(self.session, "A")
            b = chat_service.send_message(self.session, "B")  # A pending 을 읽은 같은 base
            a_frames, b_frames = list(a), list(b)
        self.assertEqual([e for e, _ in a_frames], ["delta", "done"])
        self.assertEqual([e for e, _ in b_frames], ["delta", "done"])
        self.assertEqual([(i["role"], i["content"], i["status"]) for i in history(self.session)], [
            ("user", "A", "completed"), ("assistant", "답", "completed"),
            ("user", "B", "completed"), ("assistant", "답", "completed"),
        ])


class CloseBeforeFirstIterationTest(CheckpointTestCase):
    def setUp(self):
        self.session = ChatSession.objects.create(guest="13131313-1313-1313-1313-131313131313")

    def test_response_close_before_iteration_cancels_primed_turn(self):
        fake = PausingChain()
        with patch_chain(return_value=fake):
            response = event_stream_response(chat_service.send_message(self.session, "질문", version="v2"))
            request_finished.disconnect(close_old_connections)  # 테스트 트랜잭션 연결은 살려 둔다
            try:
                response.close()
            finally:
                request_finished.connect(close_old_connections)
        self.assertIsNone(fake.received_inputs)
        messages, turns = snapshot(self.session)
        self.assertEqual(turns[messages[0].id], {"status": "cancelled", "answer_id": None})


class PublicWireCompatTest(CheckpointTestCase):
    """v1/v2 공개 wire 는 옛 ChatMessageSerializer 필드(정수 id·sequence_no·created_at·updated_at·stopped)를 유지한다."""

    def setUp(self):
        from rest_framework.test import APIClient
        self.client_a = APIClient()
        self.client_a.cookies["guest_id"] = "14141414-1414-1414-1414-141414141414"
        self.session = ChatSession.objects.create(guest="14141414-1414-1414-1414-141414141414")

    def _url(self, version):
        return f"/api/{version}/chat/sessions/{self.session.id}/messages/"

    def _post(self, version, content):
        from llm.tests.test_v2_chat import FakeChain, _read_sse_body
        with patch_chain(return_value=FakeChain(chunks=("답",))):
            response = self.client_a.post(self._url(version), {"content": content}, format="json",
                                          HTTP_ACCEPT="text/event-stream")
            return _read_sse_body(b"".join(response.streaming_content).decode("utf-8"))

    def test_history_items_keep_legacy_fields_and_done_id_matches(self):
        for version in ("v1", "v2"):
            with self.subTest(version=version):
                done = self._post(version, f"{version} 질문")[-1]
                self.assertEqual(done[0], "done")
                items = self.client_a.get(self._url(version)).json()
                for item in items:
                    self.assertTrue({"id", "sequence_no", "role", "content", "status", "tools",
                                     "created_at", "updated_at"} <= set(item))
                    self.assertIs(type(item["id"]), int)
                    self.assertIs(type(item["sequence_no"]), int)
                self.assertEqual([i["sequence_no"] for i in items], list(range(1, len(items) + 1)))
                self.assertEqual(done[1]["message_id"], str(items[-1]["id"]))  # 옛 done: 숫자 문자열 id
                self.assertRegex(done[1]["message_id"], r"^[1-9]\d*$")
        self.assertEqual(len({i["id"] for i in items}), len(items))

    def test_cancelled_turn_is_public_stopped_and_integer_id_edits_and_deletes(self):
        with patch_chain(return_value=PausingChain()):
            chat_service.send_message(self.session, "끊긴 질문").close()
        self._post("v2", "둘째 질문")
        items = self.client_a.get(self._url("v2")).json()
        self.assertEqual([(i["content"], i["status"]) for i in items],
                         [("끊긴 질문", "stopped"), ("둘째 질문", "completed"), ("답", "completed")])
        first, second = items[0]["id"], items[1]["id"]

        from llm.tests.test_v2_chat import FakeChain
        with patch_chain(return_value=FakeChain(chunks=("새 답",))):
            response = self.client_a.put(self._url("v2"), {"content": "고친 질문", "message_id": second},
                                         format="json", HTTP_ACCEPT="text/event-stream")
            self.assertEqual(response.status_code, 200)
            b"".join(response.streaming_content)
        items = self.client_a.get(self._url("v2")).json()
        self.assertEqual([(i["content"], i["status"]) for i in items],
                         [("끊긴 질문", "stopped"), ("고친 질문", "completed"), ("새 답", "completed")])
        self.assertEqual(items[1]["id"], second)  # 수정 대상은 id 유지

        response = self.client_a.delete(self._url("v2"), {"message_id": first}, format="json")
        self.assertEqual(response.status_code, 204)
        self.assertEqual(self.client_a.get(self._url("v2")).json(), [])


def _legacy(session_id, *rows):
    """옛 llm_chatmessage/llm_chattoolcall 행을 raw SQL 로 넣는다(모델은 state 에서 빠졌다). 행 id 목록을 돌려준다."""
    from django.db import connection
    ids = []
    with connection.cursor() as cursor:
        for sequence_no, (role, message, status, tools) in enumerate(rows, start=1):
            cursor.execute(
                'INSERT INTO "llm_chatmessage" (session_id, sequence_no, role, message, status, created_at, updated_at)'
                " VALUES (%s, %s, %s, %s, %s, '2025-01-02T03:04:05Z', '2025-01-02T03:04:06Z') RETURNING id",
                [session_id, sequence_no, role, message, status])
            ids.append(cursor.fetchone()[0])
            for tool_name, tool_status, result in tools:
                cursor.execute(
                    'INSERT INTO "llm_chattoolcall" (message_id, tool_name, status, arguments, result, truncated, created_at)'
                    " VALUES (%s, %s, %s, '{}', %s, false, now())", [ids[-1], tool_name, tool_status, result])
    return ids


def _legacy_count(session_id):
    from django.db import connection
    with connection.cursor() as cursor:
        cursor.execute('SELECT count(*) FROM "llm_chatmessage" WHERE session_id = %s', [session_id])
        messages = cursor.fetchone()[0]
        cursor.execute('SELECT count(*) FROM "llm_chattoolcall" t JOIN "llm_chatmessage" m ON m.id = t.message_id'
                       " WHERE m.session_id = %s", [session_id])
        return messages, cursor.fetchone()[0]


class LegacyChatHistoryTest(CheckpointTestCase):
    """배포 전 ChatMessage/ChatToolCall 대화는 setup_chat_checkpoints 가 checkpoint 로 옮겨 GET 에 보이고,
    세션/회원 삭제 때 옛 행도 지워진다(고아 개인정보 없음)."""

    GUEST = "15151515-1515-1515-1515-151515151515"

    def setUp(self):
        from rest_framework.test import APIClient
        self.session = ChatSession.objects.create(guest=self.GUEST)
        self.addCleanup(chat_service.ChatThread(self.session.id).delete)
        self.client_a = APIClient()
        self.client_a.cookies["guest_id"] = self.GUEST

    def _items(self):
        return self.client_a.get(f"/api/v2/chat/sessions/{self.session.id}/messages/").json()

    def test_setup_converts_legacy_rows_and_get_shows_them_with_legacy_ids(self):
        from django.core.management import call_command
        ids = _legacy(self.session.id,
                      ("user", "옛 질문", "completed", []),
                      ("assistant", "옛 답", "completed", [("search_players", "completed", '{"rows": 1}')]),
                      ("user", "끊긴 질문", "stopped", []))
        for _ in range(2):  # 반복 실행해도 한 번만 옮긴다
            call_command("setup_chat_checkpoints", stdout=open("/dev/null", "w"))
        items = self._items()
        self.assertEqual([(i["id"], i["role"], i["content"], i["status"]) for i in items], [
            (ids[0], "user", "옛 질문", "completed"), (ids[1], "assistant", "옛 답", "completed"),
            (ids[2], "user", "끊긴 질문", "stopped")])
        self.assertEqual([t["tool_name"] for t in items[1]["tools"]], ["search_players"])
        self.assertEqual(items[1]["tools"][0]["status"], "completed")
        self.assertEqual((items[0]["created_at"], items[0]["updated_at"]), ("2025-01-02T03:04:05Z", "2025-01-02T03:04:06Z"))
        self.assertEqual(_legacy_count(self.session.id), (3, 1))  # 롤백 대비: 옛 행은 세션 삭제 전까지 보존

        from llm.tests.test_v2_chat import FakeChain
        with patch_chain(return_value=FakeChain(chunks=("새 답",))):
            list(chat_service.send_message(self.session, "새 질문"))
        items = self._items()
        self.assertEqual([i["content"] for i in items], ["옛 질문", "옛 답", "끊긴 질문", "새 질문", "새 답"])
        self.assertGreater(items[3]["id"], max(ids))  # 새 번호는 옛 id 와 겹치지 않는다

    def test_session_and_account_delete_erase_legacy_rows(self):
        from django.contrib.auth import get_user_model
        other = ChatSession.objects.create(guest="16161616-1616-1616-1616-161616161616")
        member = get_user_model().objects.create_user(username="legacy-erase", password="pw12345!")
        owned = ChatSession.objects.create(user=member)
        for session in (self.session, other, owned):
            _legacy(session.id, ("user", "개인정보", "completed", [("search_places", "completed", "{}")]))
        with self.captureOnCommitCallbacks(execute=True):
            self.session.delete()
            member.delete()  # 회원 삭제 연쇄
        self.assertEqual(_legacy_count(self.session.id), (0, 0))
        self.assertEqual(_legacy_count(owned.id), (0, 0))
        self.assertEqual(_legacy_count(other.id), (1, 1))

    def test_setup_erases_legacy_rows_of_already_deleted_sessions(self):
        from django.core.management import call_command
        gone = "17171717-1717-1717-1717-171717171717"  # 이 수정 전에 지워져 고아가 된 세션
        _legacy(gone, ("user", "고아", "completed", [("search_places", "completed", "{}")]))
        call_command("setup_chat_checkpoints", stdout=open("/dev/null", "w"))
        self.assertEqual(_legacy_count(gone), (0, 0))
