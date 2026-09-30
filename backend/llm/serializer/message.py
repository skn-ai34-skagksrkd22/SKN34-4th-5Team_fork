import math

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from rest_framework import serializers

from llm.enum import ChatRole, ContextIntent, MessageStatus, PublicChatEvent, PublicToolStatus, TurnStatus

CONTEXT_INTENT_CHOICES = tuple(ContextIntent.values)  # ("route", "baseball", "stadium")


def _validate_context(value):
    """POST/PUT content 와 함께 오는 선택 사항 context 검증. 없으면 그대로 통과 (하위 호환).

    형식: {"stadium"?: str(<=100), "intent"?: "route"|"baseball"|"stadium",
           "origin"?: {"lat": float(-90~90), "lng": float(-180~180)}}
    origin 이 있으면 lat/lng 둘 다 있어야 하고, 둘 다 유한한 실수(math.isfinite)여야 한다.
    DRF FloatField 의 min_value/max_value 비교는 NaN 과 항상 False 라 안 걸리고
    "NaN"/"Infinity" 문자열도 float() 변환만으로 통과하므로, 범위 검사 전에 isfinite 로 막는다.
    JSON 정수 리터럴이 float 로 표현 불가능할 만큼 크면 float() 가 OverflowError 를 던진다
    (TypeError/ValueError 와 별도 예외라 같이 잡아야 한다).
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise serializers.ValidationError("context 는 객체여야 합니다.")

    errors = {}
    cleaned = {}

    if "stadium" in value:
        stadium = value["stadium"]
        if not isinstance(stadium, str) or not stadium or len(stadium) > 100:
            errors["stadium"] = "stadium 은 1~100자 문자열이어야 합니다."
        else:
            cleaned["stadium"] = stadium

    if "intent" in value:
        intent = value["intent"]
        if intent not in CONTEXT_INTENT_CHOICES:
            errors["intent"] = f"intent 는 {CONTEXT_INTENT_CHOICES} 중 하나여야 합니다."
        else:
            cleaned["intent"] = intent

    if "origin" in value:
        origin = value["origin"]
        if not isinstance(origin, dict) or "lat" not in origin or "lng" not in origin:
            errors["origin"] = "origin 은 lat/lng 이 모두 있는 객체여야 합니다."
        else:
            origin_errors = {}
            origin_cleaned = {}
            for key, lo, hi in (("lat", -90, 90), ("lng", -180, 180)):
                raw = origin[key]
                try:
                    num = float(raw)
                except (TypeError, ValueError, OverflowError):
                    origin_errors[key] = "유한한 실수여야 합니다."
                    continue
                if not math.isfinite(num):
                    origin_errors[key] = "좌표는 유한한 실수여야 합니다."
                elif not (lo <= num <= hi):
                    origin_errors[key] = f"{lo}~{hi} 범위여야 합니다."
                else:
                    origin_cleaned[key] = num
            if origin_errors:
                errors["origin"] = origin_errors
            else:
                cleaned["origin"] = origin_cleaned

    if errors:
        raise serializers.ValidationError(errors)
    return cleaned


class ChatMessageInputSerializer(serializers.Serializer):
    """POST /messages/ 요청 바디 검증 (content + 선택 사항 context)."""
    content = serializers.CharField(max_length=2200, allow_blank=False)
    context = serializers.DictField(required=False, allow_null=True)

    def validate_content(self, value):
        # CharField 는 기본적으로 int/float 등도 str() 로 강제 변환해 통과시킨다.
        # content 는 실제 문자열 입력만 허용한다.
        if not isinstance(self.initial_data.get("content"), str):
            raise serializers.ValidationError("content 는 문자열이어야 합니다.")
        return value

    def validate_context(self, value):
        return _validate_context(value)


class MessageIdField(serializers.Field):
    """공개 wire 의 정수 id(옛 ChatMessage.id 호환, 숫자 문자열 포함) → int, 저장된 HumanMessage.id(UUID) → str.
    최신 스냅샷에 없는 id 는 서비스 계층이 404 로 처리한다."""
    default_error_messages = {"invalid": "메시지 번호 또는 UUID 여야 합니다."}

    def to_internal_value(self, data):
        if isinstance(data, str) and data.isascii() and data.isdigit():
            data = int(data)
        if isinstance(data, int) and not isinstance(data, bool):
            if data < 1:
                self.fail("invalid")
            return data
        return str(serializers.UUIDField(format="hex_verbose").run_validation(data))


class ChatMessageUpdateSerializer(ChatMessageInputSerializer):
    """PUT /messages/ 요청 바디 검증 (content + message_id + 선택 사항 context)."""
    message_id = MessageIdField()


class ChatMessageDeleteSerializer(serializers.Serializer):
    """DELETE /messages/ 요청 바디 검증 (message_id만 필요)."""
    message_id = MessageIdField()


# ── LangChain 메시지·이벤트 → 공개 채팅 계약 (ORM 무관 순수 함수) ─────────────────────
# 허용 필드만 새 dict 로 복사한다. prompt/RAG/추론 블록/도구 인자·결과/metadata/usage/
# run·trace ID 는 나가지 않는다. SSE 는 sse.py 의 (event, data) 튜플 형식을 따른다.
#   delta {text} / tool {id, tool_name, status} / done {message_id, assistant_message, tools}
#   / error {detail}

GENERIC_ERROR_MESSAGE = "답변 생성에 실패했습니다. 다시 시도해 주세요."
# 모델 없는 scope/fallback 답변: 실행 경계가 dispatch_custom_event(ANSWER_TEXT_EVENT, text) 로 보낸다.
ANSWER_TEXT_EVENT = "public_answer_text"
# 여기서 만드는 dict 는 JSON·SSE 로 그대로 나가므로 Enum 이 아니라 .value(순수 str)를 담는다.
TURN_STATUSES = tuple(TurnStatus.values)  # ("pending", "completed", "failed", "cancelled")
# ToolMessage.status (LangChain 표준) → 공개 상태.
_TOOL_RESULT_STATUS = {"success": PublicToolStatus.COMPLETED.value, "error": PublicToolStatus.FAILED.value}


def _public_tool(tool_call_id, name, status):
    return {"id": tool_call_id, "tool_name": name, "status": status}


def _nonempty_str(value):
    return isinstance(value, str) and bool(value)


def project_event(event, *, answer_run_id=None, tool_call_ids=None):
    """astream_events(v2) 이벤트 하나 → (event, data) 또는 None(비공개/무시).

    answer_run_id: 실행 경계가 최종 답변 모델 호출에 지정한 run_id
        (model.stream(conv, config={"run_id": answer_run_id})). 다른 chat_model(planner 등)
        스트림은 내보내지 않는다.
    tool_call_ids: 실행 경계가 만든 {도구 run_id(str): tool_call_id} 대응표
        (tool.invoke(call, config={"run_id": run_id})). 중첩 호출의 on_tool_start 에는 인자만
        오고 tool_call_id 가 없으므로 이 표 없이는 running 을 내보내지 않는다. run_id 를
        tool_call_id 로 쓰지 않는다.
    답변 경로의 계약 위반·대응표 불일치는 ValueError (호출자가 done 없이 error 로 끝낸다).
    """
    kind = event.get("event")
    data = event.get("data")
    run_id = event.get("run_id")

    if kind == "on_chat_model_stream":
        if answer_run_id is None or run_id != str(answer_run_id):
            return None
        chunk = data.get("chunk") if isinstance(data, dict) else None
        if not isinstance(chunk, BaseMessage):
            raise ValueError("unexpected answer chunk")
        text = str(chunk.text)  # type == "text" 블록만. reasoning 등은 제외된다.
        return (PublicChatEvent.DELTA.value, {"text": text}) if text else None

    if kind == "on_custom_event" and event.get("name") == ANSWER_TEXT_EVENT:
        if not isinstance(data, str):
            raise ValueError("unexpected answer text")
        return (PublicChatEvent.DELTA.value, {"text": data}) if data else None

    name = event.get("name")
    if kind not in ("on_tool_start", "on_tool_end", "on_tool_error") or not _nonempty_str(name):
        return None
    mapped = (tool_call_ids or {}).get(run_id)

    if kind == "on_tool_start":
        call = data.get("input") if isinstance(data, dict) else None
        # 루트 호출이면 입력 자체가 ToolCall 이라 id 가 명시돼 있다.
        if not mapped and isinstance(call, dict) and call.get("type") == "tool_call":
            mapped = call.get("id")
        return (PublicChatEvent.TOOL.value, _public_tool(mapped, name, PublicToolStatus.RUNNING.value)) if _nonempty_str(mapped) else None

    if kind == "on_tool_end":
        output = data.get("output") if isinstance(data, dict) else None
        if not isinstance(output, ToolMessage) or not _nonempty_str(output.tool_call_id):
            return None
        tool_call_id = output.tool_call_id
        status = _TOOL_RESULT_STATUS.get(output.status)
    else:  # on_tool_error: langchain-core 1.6.5 는 data.tool_call_id 를 싣는다.
        tool_call_id = (data.get("tool_call_id") if isinstance(data, dict) else None) or mapped
        status = PublicToolStatus.FAILED.value
    if mapped and mapped != tool_call_id:
        raise ValueError("tool_call_id mismatch")
    return (PublicChatEvent.TOOL.value, _public_tool(tool_call_id, name, status)) if _nonempty_str(tool_call_id) and status else None


def project_history(messages, turns):
    """저장된 LangChain 메시지 목록 → 공개 대화 항목 [{id, role, content, status, tools}].

    turns: 저장 계층이 준 {HumanMessage.id: {"status": TURN_STATUSES, "answer_id": AIMessage.id|None}}.
        최종 답변과 턴 상태는 여기서만 정한다(도구 호출 없는 AIMessage 라고 최종으로 보지 않음).
    - 턴 = Human 부터 다음 Human 전까지. 도구 결과는 같은 턴 안에서만 대응시킨다.
    - answer_id 가 있으면 assistant 항목에 그 턴의 도구가 호출 순서대로 붙는다. 없으면(실패/진행
      중 턴) 저장된 유일한 기록인 user 항목에 붙는다. 가짜 AI 항목·ID 는 만들지 않는다.
    - planner AIMessage, ToolMessage, SystemMessage 는 말풍선이 아니다.
    - 결과 ToolMessage 없는 호출은 완료되지 않은 턴(pending/failed/cancelled)에서 running(미해결)으로
      둔다. 턴 상태는 그대로 두며 도구 취소·완료를 추정하지 않는다.
    - 턴 정보 누락, 모르는 상태, 완료 턴의 answer 누락·미해결 도구, id 없는 메시지는 ValueError.
    """
    items = []
    for human, turn_messages in _split_turns(messages):
        turn = turns.get(human.id) if _nonempty_str(human.id) else None
        status = turn.get("status") if isinstance(turn, dict) else None
        if status not in TURN_STATUSES:
            raise ValueError("missing turn status")
        answer_id = turn.get("answer_id")
        if status == TurnStatus.COMPLETED and not answer_id:
            raise ValueError("completed turn without answer")

        results = {m.tool_call_id: m for m in turn_messages if isinstance(m, ToolMessage)}
        tools, answer = [], None
        for message in turn_messages:
            if not isinstance(message, AIMessage):
                continue
            if answer_id and message.id == answer_id:
                answer = message
            for call in message.tool_calls:
                result = results.get(call.get("id"))
                if result:
                    tool_status = _TOOL_RESULT_STATUS.get(result.status)
                elif status != TurnStatus.COMPLETED:
                    # 결과가 없다는 것은 취소의 증거가 아니다. 턴이 실패/취소돼도 도구는 미해결(running).
                    tool_status = PublicToolStatus.RUNNING.value
                else:
                    tool_status = None
                if not tool_status:
                    raise ValueError("unresolved tool call")
                tools.append(_public_tool(call["id"], call["name"], tool_status))
        if answer_id and answer is None:
            raise ValueError("answer message not found")

        items.append(_history_item(human, ChatRole.USER.value, status, [] if answer else tools))
        if answer:
            items.append(_history_item(answer, ChatRole.ASSISTANT.value, status, tools))
    return items


def _split_turns(messages):
    turns = []
    for message in messages:
        if isinstance(message, HumanMessage):
            turns.append((message, []))
        elif turns:
            turns[-1][1].append(message)
    return turns


def _history_item(message, role, status, tools):
    if not _nonempty_str(message.id):
        raise ValueError("message without id")
    return {"id": message.id, "role": role, "content": str(message.text), "status": status, "tools": tools}


def wire_history(messages, turns, wire):
    """HTTP 공개 목록(v1/v2 공통). 옛 ChatMessageSerializer 필드를 유지한다: 정수 id·sequence_no·created_at·updated_at,
    턴 취소는 옛 이름 stopped. 항목 순서·내용·tools 는 project_history 그대로다.

    wire: ChatThread.wire ({message.id: {"id", "created_at", "updated_at"}}). sequence_no 는 현재 대화 안의 위치(1부터)라
    동시 질문이 턴 안에 끼어들면 뒤 항목의 값이 밀릴 수 있다. 번호표 없는 메시지는 ValueError.
    """
    return [_wire_item(item, wire, number) for number, item in enumerate(project_history(messages, turns), 1)]


def _wire_item(item, wire, sequence_no):
    entry = wire.get(item["id"])
    if not entry:
        raise ValueError("message without public id")
    status = MessageStatus.STOPPED.value if item["status"] == TurnStatus.CANCELLED else item["status"]
    return {**item, "id": entry["id"], "sequence_no": sequence_no, "status": status,
            "created_at": entry["created_at"], "updated_at": entry["updated_at"]}


def wire_done(messages, turns, wire):
    """HTTP 공개 done: done_payload 의 message_id 를 옛 형식(정수 id 의 숫자 문자열)으로 바꾼다."""
    payload = done_payload(messages, turns)
    entry = wire.get(payload["message_id"])
    if not entry:
        raise ValueError("message without public id")
    return {**payload, "message_id": str(entry["id"])}


def done_payload(messages, turns):
    """저장이 확인된 스냅샷 → 마지막 턴의 done 데이터. 저장 성공을 확인한 뒤에만 부른다.

    GET 과 같은 project_history 를 거치므로 message_id·tools 가 조회 결과와 일치한다.
    마지막 항목은 항상 마지막 턴의 것이므로, 마지막 턴이 완료가 아니면 이전 턴의 완료 답변을
    쓰지 않고 ValueError.
    """
    items = project_history(messages, turns)
    last = items[-1] if items else None
    if not last or last["role"] != ChatRole.ASSISTANT or last["status"] != TurnStatus.COMPLETED:
        raise ValueError("no completed assistant message")
    return {"message_id": last["id"], "assistant_message": last["content"], "tools": last["tools"]}


def error_payload():
    return {"detail": GENERIC_ERROR_MESSAGE}
