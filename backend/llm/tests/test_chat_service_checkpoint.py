"""ChatService ↔ checkpoint 통합: 실제 v1 RagChatChain(도구 루프) + 실제 PostgresSaver + HTTP SSE.

모델·도구·검색만 가짜다. 확인하는 것:
- 첫 delta 가 모델 생성 완료 전에 HTTP 소비자에게 도착한다 (WSGI 동기 이터레이터 경로).
- 시스템 프롬프트·RAG 문서·도구 인자/결과·planner 텍스트·reasoning 이 SSE/GET 어디에도 안 나온다.
- 공개 도구 프레임(running/completed|failed, tool_call_id)과 저장된 Human/AI(tool_calls)/Tool/AI 가 맞는다.
- 도구 실패 폴백 답은 정확히 한 번 나간다. done 은 최종 저장 뒤에만, 저장 실패면 error 만.
- 클라이언트 종료 → cancelled. 세션 삭제(직접·연쇄)는 thread 를 지우고 되살리지 않는다.
"""
import json
import logging
import os
import subprocess
import sys
import threading
import time
from io import StringIO
from contextlib import ExitStack
from unittest.mock import patch

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.signals import request_finished
from django.db import close_old_connections, connection, transaction
from django.test import TransactionTestCase
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import tool
from rest_framework.test import APIClient

from llm.models import ChatSession, ChatThreadDeletion
from llm.service import chat as chat_service
from llm.service.chat import ChatThread
from llm.service.chat_thread import purge_deleted_threads, reserve_thread_deletion
from llm.tests.test_v2_chat import (
    CheckpointTestCase, PausingChain, _read_sse_body, guest_request, history, patch_chain, seed, snapshot,
)
from llm.v1.rag import dispatcher, persona
from llm.v1.rag.assistant import pipeline as assistant
from llm.v1.rag.pipeline import rag_chain
from llm.views.sse import event_stream_response

GUEST = "c0c0c0c0-c0c0-c0c0-c0c0-c0c0c0c0c0c0"
RAG_SECRET = "RAG_DOC_SECRET"
ARG_SECRET = "TOOL_ARG_SECRET"
RESULT_SECRET = "TOOL_RESULT_SECRET"
PLANNER_SECRET = "READY_PLANNER_SECRET"
REASONING_SECRET = "REASONING_SECRET"
SECRETS = (RAG_SECRET, ARG_SECRET, RESULT_SECRET, PLANNER_SECRET, REASONING_SECRET,
           "KBO 직관 도우미", assistant.PLANNER_RULE)


def _chunk(**kw):
    return ChatGenerationChunk(message=AIMessageChunk(**kw))


def scripted_model(steps):
    """호출마다 steps 의 다음 생성기로 청크를 낸다 (planner invoke 도 스트리밍 콜백 아래에선 _stream 을 탄다)."""

    class _Model(BaseChatModel):
        @property
        def _llm_type(self):
            return "scripted-fake"

        def bind_tools(self, tools, **_kw):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            return generate_from_stream(self._stream(messages, stop, run_manager, **kw))

        def _stream(self, messages, stop=None, run_manager=None, **_kw):
            yield from steps.pop(0)()

    return _Model()


def plan_lookup():
    yield _chunk(content="", tool_call_chunks=[{
        "name": "lookup", "args": json.dumps({"q": ARG_SECRET}), "id": "call_1", "index": 0,
    }])


def plan_ready():
    yield _chunk(content=PLANNER_SECRET)


@tool
def lookup(q: str) -> str:
    """테스트용 조회 도구."""
    return f"{RESULT_SECRET}:{q}"


@tool
def broken(q: str) -> str:
    """항상 실패하는 도구."""
    raise RuntimeError(f"db down {q}")


def plan_broken():
    yield _chunk(content="", tool_call_chunks=[{
        "name": "broken", "args": json.dumps({"q": ARG_SECRET}), "id": "call_9", "index": 0,
    }])


def fake_pipeline(steps, tools=(lookup,)):
    stack = ExitStack()
    stack.enter_context(patch.object(assistant, "llm", lambda: scripted_model(steps)))
    stack.enter_context(patch.object(assistant.tools, "build_tools", lambda: list(tools)))
    stack.enter_context(patch.object(assistant, "retrieve", lambda inputs: {
        **inputs, "context": f"[1] {RAG_SECRET}", "stadium": None, "doc_count": 1,
    }))
    stack.enter_context(patch_chain(return_value=rag_chain))
    return stack


class V1CheckpointStreamingTest(CheckpointTestCase):
    def setUp(self):
        self.client_a = APIClient()
        self.client_a.cookies["guest_id"] = GUEST
        self.session = ChatSession.objects.create(guest=GUEST)
        self.url = f"/api/v1/chat/sessions/{self.session.id}/messages/"

    def _post(self, content):
        return self.client_a.post(self.url, {"content": content}, format="json", HTTP_ACCEPT="text/event-stream")

    def _assert_no_secrets(self, *payloads):
        body = json.dumps(payloads, ensure_ascii=False)
        for secret in SECRETS:
            self.assertNotIn(secret, body)

    def test_live_first_delta_tool_frames_saved_messages_and_no_leak(self):
        gate, events = threading.Event(), []

        def answer():
            yield _chunk(content=[{"type": "reasoning", "summary": [{"type": "summary_text", "text": REASONING_SECRET}]}])
            yield _chunk(content=[{"type": "text", "text": "첫 답변"}])
            if not gate.wait(5):
                raise TimeoutError("consumer never received first delta before the model finished")
            yield _chunk(content="은 이어서")
            events.append("producer_done")

        pieces = []
        with fake_pipeline([plan_lookup, plan_ready, answer]):
            response = self._post("LG 몇 위야?")
            self.assertEqual(response.status_code, 200)
            for piece in response.streaming_content:
                piece = piece.decode("utf-8")
                if piece.startswith("event: delta") and not gate.is_set():
                    events.append(("first_delta", list(events)))
                    gate.set()
                pieces.append(piece)

        self.assertEqual(events, [("first_delta", []), "producer_done"])  # 생성 완료 전에 받았다
        frames = _read_sse_body("".join(pieces))
        tool = {"id": "call_1", "tool_name": "lookup"}
        self.assertEqual(frames[:2], [("tool", {**tool, "status": "running"}), ("tool", {**tool, "status": "completed"})])
        self.assertEqual([e for e, _ in frames[2:]], ["delta", "delta", "done"])
        self.assertEqual("".join(d["text"] for e, d in frames if e == "delta"), "첫 답변은 이어서")
        done = frames[-1][1]
        self.assertEqual(done["assistant_message"], "첫 답변은 이어서")
        self.assertEqual(done["tools"], [{**tool, "status": "completed"}])

        # 저장: 표준 Human / AI(tool_calls) / ToolMessage / AI. 원본 인자·결과는 saver 안에만 있다.
        messages, turns = snapshot(self.session)
        self.assertEqual([type(m) for m in messages], [HumanMessage, AIMessage, ToolMessage, AIMessage])
        self.assertEqual(messages[1].tool_calls[0]["args"], {"q": ARG_SECRET})
        self.assertEqual((messages[2].tool_call_id, messages[2].content), ("call_1", f"{RESULT_SECRET}:{ARG_SECRET}"))
        self.assertEqual(turns[messages[0].id], {"status": "completed", "answer_id": messages[3].id})
        listed = self.client_a.get(self.url).json()
        self.assertEqual(done["message_id"], str(listed[-1]["id"]))  # 공개 wire 는 옛 정수 id(숫자 문자열) 호환
        self.assertEqual([{k: i[k] for k in ("role", "content", "tools")} for i in listed],
                         [{k: i[k] for k in ("role", "content", "tools")} for i in history(self.session)])
        self.assertEqual(listed[1]["tools"], done["tools"])
        self._assert_no_secrets(frames, listed)

    def test_tool_failure_is_recorded_and_fallback_answer_emitted_once(self):
        fallback = {"answer": "폴백 답변입니다", "sources": [], "route": "club", "places": []}
        with fake_pipeline([plan_broken], tools=(broken,)), \
                patch.object(dispatcher, "_domain_answer", return_value=fallback), \
                self.assertLogs(level="DEBUG") as logs:
            response = self._post("LG 몇 위야?")
            frames = _read_sse_body(b"".join(response.streaming_content).decode("utf-8"))
        # 로그(traceback 포함)에도 도구 인자·예외 원문이 없다
        logged = "\n".join(logging.Formatter().format(record) for record in logs.records)
        self.assertIn("assistant tool failed: broken", logged)
        for secret in (ARG_SECRET, "db down"):
            self.assertNotIn(secret, logged)

        expected = persona.finalize("폴백 답변입니다")
        tool = {"id": "call_9", "tool_name": "broken"}
        self.assertEqual(frames, [
            ("tool", {**tool, "status": "running"}),
            ("tool", {**tool, "status": "failed"}),
            ("delta", {"text": expected}),
            ("done", {"message_id": frames[-1][1]["message_id"], "assistant_message": expected,
                      "tools": [{**tool, "status": "failed"}]}),
        ])
        messages, _ = snapshot(self.session)
        self.assertEqual([type(m) for m in messages], [HumanMessage, AIMessage, ToolMessage, AIMessage])
        self.assertEqual((messages[2].status, messages[2].content), ("error", "tool error"))  # 오류 내용 저장 안 함
        self._assert_no_secrets(frames, history(self.session))
        self.assertNotIn("db down", json.dumps([frames, [m.content for m in messages]], ensure_ascii=False))


class CancelAndDeleteTest(CheckpointTestCase):
    def setUp(self):
        self.client_a = APIClient()
        self.client_a.cookies["guest_id"] = GUEST
        self.session = ChatSession.objects.create(guest=GUEST)
        self.thread = ChatThread(self.session.id)

    def test_final_save_failure_emits_error_without_done(self):
        real_update = ChatThread.update
        calls = []

        def flaky(self, messages, turns, *args):
            calls.append(next(iter(turns.values()))["status"])
            if len(calls) > 1:
                raise RuntimeError("saver down")
            return real_update(self, messages, turns, *args)

        with patch_chain(return_value=PausingChain(chunks=("답",))), \
                patch.object(ChatThread, "update", flaky):
            frames = list(chat_service.send_message(self.session, "LG 몇 위야?"))

        self.assertEqual(calls, ["pending", "completed"])
        self.assertEqual(frames, [("delta", {"text": "답"}), ("error", {"detail": "답변 생성에 실패했습니다. 다시 시도해 주세요."})])
        self.assertEqual([i["status"] for i in history(self.session)], ["pending"])  # done 없이 질문만 남음

    def test_client_disconnect_marks_cancelled(self):
        from llm.service import chat as chat_service
        with patch_chain(return_value=PausingChain(chunks=("안", "녕"))):
            # 테스트 클라이언트의 closing_iterator_wrapper 는 close 중에 close_old_connections 를 다시
            # 연결해 테스트 트랜잭션 연결을 닫아 버린다. 그래서 view 와 같은 SSE 응답을 직접 만든다.
            response = event_stream_response(chat_service.send_message(self.session, "질문", version="v2"))
            first = next(iter(response.streaming_content))
            # WSGI 서버가 연결 종료 때 부르는 close(). 테스트 트랜잭션 연결은 살려 둔다.
            request_finished.disconnect(close_old_connections)
            try:
                response.close()
            finally:
                request_finished.connect(close_old_connections)
        self.assertTrue(first.startswith(b"event: delta"))
        messages, turns = snapshot(self.session)
        self.assertEqual([m.content for m in messages], ["질문"])  # 부분 답변 저장 안 함
        self.assertEqual(turns[messages[0].id], {"status": "cancelled", "answer_id": None})
        self.assertEqual(history(self.session)[0]["status"], "cancelled")

    def test_session_delete_api_removes_thread(self):
        seed(self.session, ("질문", "답변"))
        with self.captureOnCommitCallbacks(execute=True):
            response = self.client_a.delete(f"/api/v2/chat/sessions/{self.session.id}/")
        self.assertEqual(response.status_code, 204)
        self.assertEqual(self.thread.state(), ([], {}))
        self.assertEqual(self.thread.history(), [])

    def test_stream_finishing_after_session_delete_does_not_resurrect(self):
        with patch_chain(return_value=PausingChain(chunks=("안", "녕"))):
            from llm.service import chat as chat_service
            events = chat_service.send_message(self.session, "질문")
            next(events)
            with self.captureOnCommitCallbacks(execute=True):
                ChatSession.objects.filter(id=self.session.id).delete()
            self.assertEqual(self.thread.history(), [])
            remaining = list(events)
        self.assertEqual([e for e, _ in remaining], ["delta", "error"])  # 남은 청크 뒤 done 없음
        self.assertEqual(self.thread.history(), [])

    def test_delete_during_stream_is_not_revived_by_stale_completion(self):
        with patch_chain(return_value=PausingChain(chunks=("안", "녕"))):
            events = chat_service.send_message(self.session, "질문")
            next(events)
            human = snapshot(self.session)[0][0]
            self.assertEqual(chat_service.message_delete(guest_request(self.session), self.session.id, human.id), 1)
            remaining = list(events)
        self.assertEqual([e for e, _ in remaining], ["delta", "error"])  # 남은 청크 뒤 done 없음  # done 없음
        self.assertEqual(snapshot(self.session), ([], {}))
        self.assertEqual(self.thread.history()[0].values["revision"], 1)  # 최신은 편집 branch

    def test_put_during_stream_keeps_edit_and_only_new_stream_completes(self):
        with patch_chain(return_value=PausingChain(chunks=("안", "녕"))):
            old = chat_service.send_message(self.session, "옛 질문")
            next(old)
            human = snapshot(self.session)[0][0]
            new = chat_service.message_update(guest_request(self.session), self.session.id, human.id, "새 질문")
            self.assertEqual([e for e, _ in old], ["delta", "error"])  # 편집 전 스트림은 stale
            self.assertEqual([(i["content"], i["status"]) for i in history(self.session)], [("새 질문", "pending")])
            frames = list(new)
        self.assertEqual([e for e, _ in frames], ["delta", "delta", "done"])
        messages, turns = snapshot(self.session)
        self.assertEqual([(m.id, m.content) for m in messages], [(human.id, "새 질문"), (messages[1].id, "안녕")])
        self.assertEqual(turns, {human.id: {"status": "completed", "answer_id": messages[1].id}})


class SessionDeleteOutboxTest(CheckpointTestCase):
    """세션 삭제: 같은 트랜잭션의 outbox 행 → commit 뒤 checkpoint 삭제. 실패 행은 남아 재시도된다."""

    def setUp(self):
        self.member = get_user_model().objects.create_user(username="ckpt-outbox", password="pw12345!")
        self.sessions = [ChatSession.objects.create(user=self.member) for _ in range(3)]
        for session in self.sessions:
            seed(session, ("질문", "답변"))

    def _outbox(self):
        from llm.models import ChatThreadDeletion
        return set(ChatThreadDeletion.objects.values_list("thread_id", flat=True))

    def test_orm_rollback_keeps_session_checkpoint_and_outbox_unchanged(self):
        session = self.sessions[0]
        with self.captureOnCommitCallbacks(execute=True):
            with self.assertRaisesMessage(RuntimeError, "orm failed"), transaction.atomic():
                self.member.delete()
                raise RuntimeError("orm failed")
        self.assertEqual(len(snapshot(session)[0]), 2)  # rollback 전에 checkpoint 를 지우지 않는다
        self.assertEqual(ChatSession.objects.filter(id__in=[s.id for s in self.sessions]).count(), 3)
        self.assertEqual(self._outbox(), set())

    def test_cascade_partial_checkpoint_failure_keeps_retryable_outbox(self):
        """여러 세션 연쇄 삭제: 첫 checkpoint 삭제 성공 뒤 다음이 실패해도 ORM 삭제는 확정되고 실패분만 outbox 에 남는다."""
        first, failing, last = self.sessions
        real = ChatThread.delete

        def flaky(self_):
            if self_.thread_id == str(failing.id):
                raise RuntimeError("saver down")
            return real(self_)

        with patch.object(ChatThread, "delete", flaky), self.captureOnCommitCallbacks(execute=True):
            self.member.delete()
        self.assertFalse(ChatSession.objects.filter(id__in=[s.id for s in self.sessions]).exists())
        self.assertEqual(ChatThread(first.id).history(), [])
        self.assertEqual(ChatThread(last.id).history(), [])
        self.assertEqual(len(snapshot(failing)[0]), 2)
        self.assertEqual(self._outbox(), {failing.id})

        call_command("setup_chat_checkpoints", stdout=StringIO())  # 재시도 성공
        self.assertEqual(ChatThread(failing.id).history(), [])
        self.assertEqual(self._outbox(), set())

    def test_missed_on_commit_is_drained_by_setup_command(self):
        """commit 뒤 on_commit 이 실행되지 못해도(process crash) outbox 행이 남아 다음 drain 이 처리한다."""
        session = self.sessions[0]
        session_id = session.id  # delete() 뒤 instance.pk 는 None
        with self.captureOnCommitCallbacks(execute=False):
            session.delete()
        self.assertEqual(len(ChatThread(session_id).state()[0]), 2)
        self.assertEqual(self._outbox(), {session_id})
        call_command("setup_chat_checkpoints", stdout=StringIO())
        self.assertEqual(ChatThread(session_id).history(), [])
        self.assertEqual(self._outbox(), set())

    def test_late_writer_reservation_survives_paused_drainer(self):
        """drainer 가 checkpoint 를 지우고 멈춘 사이 늦은 writer 는 세션 행이 없어 쓰지 못한다. 그 사이 재예약(token 교체)이
        생겨도 재개한 drainer 는 옛 token 으로 그 예약을 지우지 못하고, 다음 purge 가 치워 고아가 남지 않는다."""
        session = self.sessions[0]
        session_id = session.id
        writer = ChatThread(session_id)
        writer.state()  # 세션 삭제 전에 읽은 요청
        with self.captureOnCommitCallbacks(execute=False):
            session.delete()
        real_delete, paused = ChatThread.delete, []

        def drainer_delete(self_):
            real_delete(self_)
            if paused:
                return
            paused.append(None)  # 한 번만: checkpoint 삭제 직후 멈춘 drainer 사이에 끼어든다
            self.assertFalse(writer.update([HumanMessage("늦은 질문", id="h-late")], {}))
            self.assertEqual(ChatThread(session_id).history(), [])  # 삭제된 세션에는 쓰지 않았다
            reserve_thread_deletion(session_id)

        with patch.object(ChatThread, "delete", drainer_delete):
            purge_deleted_threads([session_id])  # drainer 재개 → 자기 행 정리 시도
        self.assertEqual(self._outbox(), {session_id})  # 새 token 예약은 남는다
        purge_deleted_threads([session_id])
        self.assertEqual(ChatThread(session_id).history(), [])
        self.assertEqual(self._outbox(), set())


# 별도 프로세스의 writer: saver.put 전후로 멈출 수 있어, commit 직후 강제 종료를 흉내낸다.


_WRITER = """
import sys
import django
django.setup()
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
from llm.service.chat import ChatThread

crash, real_put = sys.argv[2] == "crash", PostgresSaver.put

def put(self, *args, **kwargs):
    print("put", flush=True)
    if crash:
        sys.stdin.readline()  # 부모가 세션 삭제를 시작할 때까지
    config = real_put(self, *args, **kwargs)
    if crash:
        print("committed", flush=True)
        sys.stdin.readline()  # 부모가 SIGKILL 할 때까지
    return config

PostgresSaver.put = put
print("result", ChatThread(sys.argv[1]).update([HumanMessage("늦은 질문", id="h-late")], {}), flush=True)
"""


class WriteFenceTest(TransactionTestCase):
    """세션 삭제 ↔ checkpoint 쓰기 순서: 서로 다른 DB 연결·프로세스에서 실제 commit 으로 검증한다."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        ChatThread.setup()

    def setUp(self):
        self.session = ChatSession.objects.create(guest=GUEST)
        self.addCleanup(ChatThread(self.session.id).delete)

    def _writer(self, mode):
        env = {**os.environ, "DJANGO_SETTINGS_MODULE": "config.settings", "DB_NAME": connection.settings_dict["NAME"]}
        proc = subprocess.Popen([sys.executable, "-c", _WRITER, str(self.session.id), mode], cwd=settings.BASE_DIR,
                                env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.addCleanup(lambda: (proc.kill(), proc.wait(), proc.stdin.close(), proc.stdout.close()))
        return proc

    def _lock_waiter(self, finished):
        """다른 연결이 행 잠금을 기다리면 True. finished() 가 먼저 참이면(기다리지 않고 끝남) False."""
        deadline = time.monotonic() + 10
        with connection.cursor() as cursor:
            while time.monotonic() < deadline and not finished():
                cursor.execute("SELECT count(*) FROM pg_locks WHERE NOT granted")
                if cursor.fetchone()[0]:
                    return True
                time.sleep(0.05)
        return False

    def _assert_gone(self):
        self.assertFalse(ChatSession.objects.filter(id=self.session.id).exists())
        self.assertEqual(ChatThread(self.session.id).history(), [])
        self.assertFalse(ChatThreadDeletion.objects.exists())

    def test_writer_killed_after_checkpoint_commit_leaves_no_orphan(self):
        """write-before-delete: writer 가 saver commit 직후 죽어도, 기다리던 세션 삭제가 이어서 purge 한다."""
        writer = self._writer("crash")
        self.assertEqual(writer.stdout.readline(), "put\n")
        errors = []

        def delete_session():
            try:
                ChatSession.objects.filter(id=self.session.id).delete()  # autocommit: on_commit purge 도 바로 돈다
            except Exception as exc:  # pragma: no cover - 실패 원인을 본 스레드로 넘긴다
                errors.append(exc)
            finally:
                connection.close()

        deleter = threading.Thread(target=delete_session)
        deleter.start()
        waited = self._lock_waiter(lambda: not deleter.is_alive())
        writer.stdin.write("\n")
        writer.stdin.flush()
        self.assertEqual(writer.stdout.readline(), "committed\n")
        writer.kill()
        writer.wait()
        deleter.join(10)
        self.assertFalse(deleter.is_alive())
        self.assertEqual(errors, [])
        self._assert_gone()
        self.assertTrue(waited)  # 삭제는 writer 의 쓰기 구간이 끝날 때까지 기다렸다

    def test_writer_waiting_on_uncommitted_delete_does_not_write(self):
        """delete-before-write: 세션 삭제가 먼저 행을 잡으면 writer 는 그 commit 뒤 checkpoint 를 쓰지 않는다."""
        with transaction.atomic():
            ChatSession.objects.filter(id=self.session.id).delete()
            writer = self._writer("once")
            waited = self._lock_waiter(lambda: writer.poll() is not None)
        output = writer.communicate(timeout=30)[0]
        self.assertEqual(output, "result False\n")  # saver.put 을 부르지 않았다
        self._assert_gone()
        self.assertTrue(waited)
