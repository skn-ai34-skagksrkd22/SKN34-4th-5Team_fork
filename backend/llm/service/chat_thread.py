"""대화 저장 계층 — 대화 원본은 LangGraph checkpoint(PostgresSaver)에만 저장한다.

- thread_id = str(ChatSession.id). checkpoint_ns 는 쓰지 않는다. 최신 state 의 messages 가 현재 대화다.
- messages 는 add_messages reducer, turns 는 키 단위 병합 reducer. 쓰기는 모두 graph.update_state 다.
- 메시지 DELETE/PUT 은 RemoveMessage 로 최신 state 에서 지운다. 과거 checkpoint 는 세션 삭제 때 지워진다:
  llm.apps 의 post_delete 가 같은 트랜잭션에 ChatThreadDeletion(outbox) 행을 남기고(reserve_thread_deletion),
  commit 뒤 purge_deleted_threads 가 checkpoint 를 지운 다음 행을 치운다. 실패 행은 남아 다음 drain 이 재시도한다.
- 같은 대화의 동시 요청은 직렬화하지 않는다. revision 판정·쓰기 구간만 세션 행 잠금으로 짧게 줄 세운다(update).
  일반 쓰기는 그 잠금 안의 최신 checkpoint 에 이어 써서 선형화하고, 편집만 읽은 checkpoint 에서 갈라진다.

의존 방향: llm.service.chat(facade·스트리밍) → 이 모듈. 이 모듈은 facade 를 import 하지 않는다.
"""
import json
import logging
import uuid
from contextlib import contextmanager
from typing import Annotated, TypedDict
from urllib.parse import quote

from django.db import connection, connections, transaction
from django.http import Http404
from django.utils import timezone
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import START, StateGraph, add_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from llm.enum import TurnStatus
from llm.models import ChatSession, ChatThreadDeletion

log = logging.getLogger(__name__)


def _merge_turns(current, update):
    """turns reducer: {HumanMessage.id: {"status", "answer_id"}} 를 키 단위로 덮어쓰고, None 이면 지운다."""
    merged = {**(current or {}), **update}
    return {key: turn for key, turn in merged.items() if turn is not None}


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    turns: Annotated[dict, _merge_turns]
    revision: int  # 편집 세대. DELETE/PUT 에서만 +1, 일반 질문·답변은 그대로 물려받는다
    wire: Annotated[dict, _merge_turns]  # {message.id: {"id": int, "created_at", "updated_at"}} 옛 공개 wire 호환 번호


_builder = StateGraph(ChatState)
_builder.add_node("record", lambda state: {})  # 실행하지 않는다. update_state 가 기록할 노드일 뿐
_builder.add_edge(START, "record")


def _conn_string():
    # connections 의 settings_dict 를 써서 테스트 실행 시 test DB 를 그대로 따라간다.
    db = connections["default"].settings_dict
    port = f":{db['PORT']}" if db.get("PORT") else ""
    return "postgresql://{}:{}@{}{}/{}".format(
        *(quote(str(db.get(key) or ""), safe="") for key in ("USER", "PASSWORD", "HOST")), port,
        quote(str(db["NAME"]), safe=""),
    )


def _pending_turn():
    # checkpoint 에는 Enum 이 아니라 원래 문자열을 넣는다(serde 가 Enum 을 타입째 저장하지 않게).
    return {"status": TurnStatus.PENDING.value, "answer_id": None}


def _numbered(messages, wire):
    """새 Human/AI 메시지에 thread 안 정수 id(옛 ChatMessage.id 호환)를 붙인다. 번호는 지워져도 재사용하지 않는다.
    같은 id 로 바뀐 질문(PUT)은 id·created_at 을 유지하고 updated_at 만 바꾼다."""
    now = timezone.now().isoformat().replace("+00:00", "Z")  # DRF DateTimeField 출력 형식
    next_no = max((entry["id"] for entry in wire.values()), default=0) + 1
    numbered = {}
    for message in messages:
        if not isinstance(message, (HumanMessage, AIMessage)):
            continue
        if message.id in wire:
            numbered[message.id] = {**wire[message.id], "updated_at": now}
        else:
            numbered[message.id] = {"id": next_no, "created_at": now, "updated_at": now}
            next_no += 1
    return numbered


def _into_turn(current, added, turns):
    """이미 저장된 질문 턴의 기록(답변·도구)을 그 턴 끝(다음 질문 앞)에 끼운다. 턴 = Human 부터 다음 Human 전까지라,
    뒤에 끼어든 동시 질문 뒤에 붙이면 답변이 남의 턴으로 넘어간다. 새 질문이거나 마지막 턴이면 그대로 덧붙인다."""
    ids = [m.id for m in current]
    owner = next((key for key in turns if key in ids), None)
    if owner is None:
        return added
    start = ids.index(owner) + 1
    end = next((i for i in range(start, len(current)) if isinstance(current[i], HumanMessage)), None)
    if end is None:
        return added
    return [RemoveMessage(id=REMOVE_ALL_MESSAGES), *current[:end], *added, *current[end:]]


class ChatThread:
    """대화 하나 = LangGraph thread. 도구 요청/결과 원본까지 messages 에 저장하고 공개는 project_history 가 고른다.

    인스턴스 하나 = 요청 하나. 처음 읽은 checkpoint 와 이후 자기가 쓴 checkpoint 를 self.config 에, 그 요청이
    읽은(또는 자기가 올린) revision 을 self.revision 에 붙잡는다.

    모든 쓰기는 세션 행 잠금 안에서 최신 revision 이 self.revision 과 같을 때만 저장한다. 다르면 그 사이 다른
    편집(DELETE/PUT)이 먼저 저장된 것이라 stale 이다: 쓰지 않고 False(done 없음). 같은 base 의 동시 편집 둘은
    잠금 순서대로 먼저 온 쪽만 revision 을 올리고, 진 쪽과 그 branch 의 pending/완료/취소 쓰기는 모두 막힌다.

    일반 질문·답변 쓰기는 행 잠금 안에서 최신 checkpoint 에 이어 써 선형화한다(같은 base 의 동시 질문도 둘 다 남는다).

    ponytail: 같은 thread 동시 요청의 생성·스트리밍은 직렬화하지 않는다(쓰기 구간만 update 의 행 잠금). 동시 질문의
    턴은 쓰기 순서대로 섞여 쌓이고, 각 요청의 모델 입력은 자기가 읽은 시점의 완료 턴뿐이다.
    """

    def __init__(self, thread_id):
        self.thread_id = str(thread_id)
        self.root = {"configurable": {"thread_id": self.thread_id}}
        self.config = self.root  # 아직 읽기 전: thread 의 최신 checkpoint
        self.revision = 0
        self.wire = {}  # 읽거나 쓴 state 의 공개 번호표 (serializer.message.wire_history)

    def _anchor(self, config):
        """이 요청이 이어 쓸 checkpoint. checkpoint_ns 는 넘기지 않고 thread_id/checkpoint_id 만 남긴다."""
        checkpoint_id = config["configurable"].get("checkpoint_id")
        self.config = {"configurable": {**self.root["configurable"], **({"checkpoint_id": checkpoint_id} if checkpoint_id else {})}}

    @staticmethod
    @contextmanager
    def _graph():
        # ponytail: 호출마다 짧은 연결 + compile. 트래픽이 늘면 psycopg_pool 기반 saver 하나를 공유한다.
        with PostgresSaver.from_conn_string(_conn_string()) as saver:
            yield _builder.compile(checkpointer=saver)

    @classmethod
    def setup(cls):
        """checkpoint 테이블 생성/마이그레이션 (manage.py setup_chat_checkpoints)."""
        with cls._graph() as graph:
            graph.checkpointer.setup()

    def state(self):
        """(messages, turns). 저장된 적 없으면 ([], {})."""
        with self._graph() as graph:
            snapshot = graph.get_state(self.config)
        self._anchor(snapshot.config)  # 빈 thread 면 checkpoint_id 가 없어 root 그대로
        values = snapshot.values
        self.revision = values.get("revision", 0)
        self.wire = dict(values.get("wire", {}))
        return list(values.get("messages", [])), dict(values.get("turns", {}))

    def history(self):
        """과거 checkpoint 포함 StateSnapshot 목록 (최신순)."""
        with self._graph() as graph:
            return list(graph.get_state_history(self.root))

    def delete(self):
        """thread 의 모든 checkpoint 삭제. 멱등이라 실패 시 재시도 가능."""
        with self._graph() as graph:
            graph.checkpointer.delete_thread(self.thread_id)

    def update(self, messages, turns, revision=None):
        """messages/turns 변경분을 기록한다. 세션이 없거나 읽은 뒤 다른 편집이 저장됐으면(stale) False.

        revision: 편집(DELETE/PUT)만 새 세대(읽은 revision + 1)를 넘긴다.

        ponytail: 판정과 쓰기는 같은 잠금 안이라 사이에 끼어들 수 없다. 다만 True 를 돌려준 뒤 SSE done 이 전송되기
        전에 커밋된 편집은 이 요청이 볼 수 없어 done 이 나갈 수 있다(최신 state 는 그 편집이다).
        """
        # 세션 행 잠금 = checkpoint 쓰기 fence. PostgresSaver 는 ORM 과 다른 연결로 바로 commit 하므로, 이 짧은 잠금이
        # 세션 삭제와 쓰기의 순서를 정한다: 삭제가 먼저면 행이 없어 쓰지 않고, 쓰기가 먼저면 삭제는 이 transaction 이
        # 끝날 때(프로세스가 죽어도 연결이 끊기며) 풀려 post_delete purge 가 방금 쓴 checkpoint 까지 지운다.
        # 같은 세션의 다른 writer 도 여기서 줄 서므로 revision 판정과 쓰기 사이에 최신이 바뀌지 않는다. LLM 생성·SSE 는 잠그지 않는다.
        with transaction.atomic():
            if not ChatSession.objects.select_for_update().filter(id=self.thread_id).exists():
                return False
            with self._graph() as graph:
                latest = graph.get_state(self.root).values
                if latest.get("revision", 0) != self.revision:
                    return False
                values = {"messages": messages, "turns": turns}
                if revision is None:
                    # 일반 질문·답변은 잠금 안의 최신에 이어 쓴다(같은 base 의 동시 질문이 branch 로 갈라져 사라지지 않게).
                    base, self.config = latest, self.root
                    values["messages"] = _into_turn(latest.get("messages", []), messages, turns)
                else:
                    # 편집만 읽은 checkpoint 에서 갈라진다: 사용자가 본 대화 기준으로 뒤를 지운다.
                    base = graph.get_state(self.config).values
                    values["revision"] = self.revision = revision
                values["wire"] = _numbered(messages, {**latest.get("wire", {}), **base.get("wire", {})})
                self.wire = {**base.get("wire", {}), **values["wire"]}
                self._anchor(graph.update_state(self.config, values))
                return True

    def _find(self, message_id):
        """(messages, turns, index): message_id 인 HumanMessage 위치. 없으면 404.

        message_id: 메시지 ID(str) 또는 공개 wire 의 정수 id (옛 클라이언트 PUT/DELETE 호환)."""
        messages, turns = self.state()
        if isinstance(message_id, int):
            message_id = next((key for key, entry in self.wire.items() if entry["id"] == message_id), None)
        for index, message in enumerate(messages):
            if isinstance(message, HumanMessage) and message.id == message_id:
                return messages, turns, index
        # 존재하지 않는/다른 세션의/user 가 아닌 message_id -- 404 로 통일한다
        raise Http404("message not found")

    def _write(self, messages, turns, revision=None):
        # ponytail: 질문/편집 저장이 동시 편집에 밀려 stale 이어도 404 로 합친다(아주 짧은 경쟁 구간).
        if not self.update(messages, turns, revision):
            raise Http404("session not found")

    def delete_from(self, message_id):
        """대상 HumanMessage 부터 끝까지 RemoveMessage 로 지우고 지운 메시지 수를 돌려준다."""
        messages, turns, index = self._find(message_id)
        removed = messages[index:]
        self._write([RemoveMessage(id=m.id) for m in removed], {m.id: None for m in removed if m.id in turns},
                    self.revision + 1)
        return len(removed)

    def edit(self, message_id, content):
        """대상 뒤 메시지를 지우고 대상을 같은 ID 의 새 질문(pending)으로 바꾼다.

        반환 (앞부분 messages, 앞부분 turns, 새 HumanMessage) -- 다시 답할 입력이다.
        """
        messages, turns, index = self._find(message_id)
        later = messages[index + 1:]
        human = HumanMessage(content=content, id=messages[index].id)
        self._write(
            [*(RemoveMessage(id=m.id) for m in later), human],
            {**{m.id: None for m in later if m.id in turns}, human.id: _pending_turn()},
            self.revision + 1,
        )
        prefix = messages[:index]
        kept = {m.id for m in prefix if isinstance(m, HumanMessage)}
        return prefix, {key: turn for key, turn in turns.items() if key in kept}, human

    def ask(self, content):
        """새 질문(pending)을 덧붙인다. 반환은 edit 와 같다."""
        messages, turns = self.state()
        human = HumanMessage(content, id=str(uuid.uuid4()))
        self._write([human], {human.id: _pending_turn()})
        return messages, turns, human


# ── 세션 삭제 → checkpoint 삭제 outbox ──────────────────────────────────────────────
def reserve_thread_deletion(thread_id):
    """thread 의 checkpoint 삭제를 outbox 에 예약한다. 이미 있으면 token 을 새로 바꿔, 읽어 둔 옛 token 으로
    drain 중인 쪽이 이 예약을 지우지 못하게 한다(아래 purge 의 조건부 삭제)."""
    if not ChatThreadDeletion.objects.filter(thread_id=thread_id).update(token=uuid.uuid4()):
        ChatThreadDeletion.objects.get_or_create(thread_id=thread_id)


def purge_deleted_threads(thread_ids=None):
    """outbox(ChatThreadDeletion) 의 checkpoint 를 지우고 성공한 행만 치운다. None 이면 남은 행 전부.

    실패한 행은 로그만 남기고 그대로 둔다: 다음 drain(setup_chat_checkpoints 시작 경로 등)이 재시도한다.
    행은 checkpoint 삭제 전에 읽은 token 이 그대로일 때만 지운다: 그 사이 재예약(token 교체)이 있었으면 행이 남아
    다음 purge/drain 이 다시 지운다. (늦은 writer 는 ChatThread.update 의 세션 행 잠금 때문에 삭제 뒤 쓰지 못한다.)
    delete_thread 는 멱등이라 여러 프로세스가 같은 행을 동시에 drain 해도 안전하다.
    """
    rows = ChatThreadDeletion.objects.all()
    if thread_ids is not None:
        rows = rows.filter(thread_id__in=thread_ids)
    for thread_id, token in list(rows.values_list("thread_id", "token")):
        try:
            ChatThread(thread_id).delete()
        except Exception:
            log.exception("chat checkpoint delete failed, kept for retry: %s", thread_id)
            continue
        ChatThreadDeletion.objects.filter(thread_id=thread_id, token=token).delete()


# ── 배포 전 대화(llm_chatmessage/llm_chattoolcall) 이관·삭제 ─────────────────────────────
# 0007 이 state 에서만 뺀 옛 테이블. 배포 절차: migrate → setup_chat_checkpoints(아래 둘을 부른다) → 서버 시작.
# - 이관은 checkpoint 가 아직 없는 세션만 한 번 옮긴다(멱등). 옛 행은 지우지 않고 세션 삭제 때까지 보존한다:
#   롤백 = 이전 코드로 되돌리고 `migrate llm 0006` (0007 역방향이 FK 를 다시 건다. 고아 행은 여기서 지워져 막히지 않는다).
# - ponytail: 롤백 기간에 옛 코드가 쓴 행은, 이미 이관된 세션이면 다시 앞으로 배포할 때 checkpoint 에 합치지 않는다.
# 테이블을 수동 DROP 한 뒤에도 돌도록 존재를 먼저 확인한다.
_LEGACY_TURN_STATUS = {"completed": TurnStatus.COMPLETED.value, "stopped": TurnStatus.CANCELLED.value}


def _legacy_tables():
    with connection.cursor() as cursor:
        cursor.execute("SELECT to_regclass('llm_chatmessage') IS NOT NULL AND to_regclass('llm_chattoolcall') IS NOT NULL")
        return cursor.fetchone()[0]


def _wire_time(value):
    return value.isoformat().replace("+00:00", "Z")  # _numbered 와 같은 DRF 형식


def _legacy_state(rows, tools):
    """옛 행(sequence_no 순) → (messages, turns, wire). 턴 = user 행부터 다음 user 행 전. 답변은 턴의 마지막 assistant 행.
    완료 턴에 답이 없거나 pending(배포로 끊김)·failed 는 failed, stopped 는 cancelled. 옛 정수 id·시각을 wire 에 둔다."""
    messages, turns, wire = [], {}, {}
    grouped = []
    for row in rows:
        role = {"human": "user", "ai": "assistant"}.get(row["role"], row["role"])
        if role == "user":
            grouped.append((row, []))
        elif grouped:  # 첫 질문 앞의 답변 행은 어느 턴에도 속하지 않아 버린다(옛 GET 도 턴 밖이었다)
            grouped[-1][1].append(row)
    for user, answers in grouped:
        answer = answers[-1] if answers else None
        status = _LEGACY_TURN_STATUS.get(user["status"], TurnStatus.FAILED.value)
        if status == TurnStatus.COMPLETED and answer is None:
            status = TurnStatus.FAILED.value
        human = HumanMessage(user["message"], id=f"legacy-{user['id']}")
        messages.append(human)
        calls = [t for r in (user, *answers) for t in tools.get(r["id"], [])]
        if calls:
            messages.append(AIMessage("", id=f"legacy-{user['id']}-tools", tool_calls=[
                {"name": t["tool_name"], "args": t["arguments"] if isinstance(t["arguments"], dict) else {},
                 "id": f"legacy-tool-{t['id']}"} for t in calls]))
            for t in calls:
                # started 는 결과가 없다: 완료 턴이면 미해결로 둘 수 없어 failed, 아니면 running 그대로.
                if t["status"] != "started" or status == TurnStatus.COMPLETED:
                    messages.append(ToolMessage(json.dumps(t["result"], ensure_ascii=False), id=f"legacy-tool-{t['id']}-result",
                                                tool_call_id=f"legacy-tool-{t['id']}", name=t["tool_name"],
                                                status="success" if t["status"] == "completed" else "error"))
        ai = AIMessage(answer["message"], id=f"legacy-{answer['id']}") if answer else None
        if ai:
            messages.append(ai)
        turns[human.id] = {"status": status, "answer_id": ai.id if ai else None}
        for row, message in ((user, human), (answer, ai)):
            if message:
                wire[message.id] = {"id": row["id"], "created_at": _wire_time(row["created_at"]),
                                    "updated_at": _wire_time(row["updated_at"])}
    return messages, turns, wire


def convert_legacy_threads():
    """checkpoint 가 없는 세션의 옛 대화를 첫 checkpoint 로 옮긴다. 옮긴 세션 수를 돌려준다."""
    if not _legacy_tables():
        return 0
    with connection.cursor() as cursor:
        cursor.execute('SELECT DISTINCT m.session_id FROM "llm_chatmessage" m JOIN "llm_chatsession" s ON s.id = m.session_id')
        session_ids = [row[0] for row in cursor.fetchall()]
    converted = 0
    for session_id in session_ids:
        with transaction.atomic():  # update 와 같은 세션 행 잠금 fence
            if not ChatSession.objects.select_for_update().filter(id=session_id).exists():
                continue
            with connection.cursor() as cursor:
                cursor.execute('SELECT id, role, message, status, created_at, updated_at FROM "llm_chatmessage"'
                               " WHERE session_id = %s ORDER BY sequence_no, id", [session_id])
                columns = [c[0] for c in cursor.description]
                rows = [dict(zip(columns, r)) for r in cursor.fetchall()]
                cursor.execute('SELECT t.id, t.message_id, t.tool_name, t.status, t.arguments, t.result FROM "llm_chattoolcall" t'
                               ' JOIN "llm_chatmessage" m ON m.id = t.message_id WHERE m.session_id = %s ORDER BY t.id', [session_id])
                columns = [c[0] for c in cursor.description]
                tools = {}
                for r in cursor.fetchall():
                    tool = dict(zip(columns, r))
                    tools.setdefault(tool["message_id"], []).append(tool)
            messages, turns, wire = _legacy_state(rows, tools)
            thread = ChatThread(session_id)
            with thread._graph() as graph:
                # get_tuple 대신 공개 get_state 로 판정: 빈 thread 는 config 에 checkpoint_id 가 없다(ChatThread.state 와 같은 신호).
                if graph.get_state(thread.root).config["configurable"].get("checkpoint_id") is not None:
                    continue  # 이미 이관됐거나 배포 뒤 새로 쓴 대화
                graph.update_state(thread.root, {"messages": messages, "turns": turns, "wire": wire})
            converted += 1
    return converted


def erase_legacy_rows(session_id=None):
    """세션의 옛 행을 지운다(세션 삭제 transaction 안에서). None 이면 세션이 이미 없는 고아 행 전부."""
    if not _legacy_tables():
        return
    if session_id is None:
        where, params = 'NOT EXISTS (SELECT 1 FROM "llm_chatsession" s WHERE s.id = m.session_id)', []
    else:
        where, params = "m.session_id = %s", [session_id]
    with connection.cursor() as cursor:
        cursor.execute(f'DELETE FROM "llm_chattoolcall" WHERE message_id IN (SELECT m.id FROM "llm_chatmessage" m WHERE {where})', params)
        cursor.execute(f'DELETE FROM "llm_chatmessage" m WHERE {where}', params)
