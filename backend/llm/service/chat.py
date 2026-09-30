"""채팅 유스케이스 facade — 체인 버전 선택 + 공개 API(send/list/update/delete)의 호환 경로.

실제 실행 계약은 파일별로 하나만 존재한다: v1 astream_events 는 llm.service.chat_v1, v2
get_graph() LangGraph 실행은 llm.service.chat_v2. 이 모듈은 기존 llm.service.chat.* 호출부
(테스트 포함)를 그대로 유지하기 위해 version 에 따라 그 둘로 위임만 한다.
views.message.ChatMessageView 도 같은 두 모듈을 직접 부른다 (이 파사드를 거치지 않는다).

대화 저장(LangGraph checkpoint, 동시 쓰기 순서, 세션 삭제 outbox)은 llm.service.chat_thread 가 맡는다.
"""
import logging
import os

from llm.enum import ChainVersion
from llm.serializer.message import wire_history
from llm.service import chat_v1, chat_v2
from llm.service.chat_thread import ChatThread
from llm.service.ownership import get_owned_session

log = logging.getLogger(__name__)

SUPPORTED_CHAIN_VERSIONS = tuple(ChainVersion.values)  # ("v1", "v2")


def chain_version():
    """설정된 LLM 체인 버전. env LLM_CHAIN_VERSION (기본 "v2")."""
    return os.getenv("LLM_CHAIN_VERSION", ChainVersion.V2.value)


def resolve_version(version=None):
    """호출자가 넘긴 version(URL 등)이 있으면 그걸 쓰고, 없으면 env 로 내려간다.

    HTTP 가 아닌 기존 호출자(예: 관리 커맨드, 배치)는 그대로 env 기본값을 쓴다.
    URL 로 들어온 값은 v1/v2 가 아니면 조용히 무시하지 않고 여기서 바로 막는다.
    """
    resolved = version if version is not None else chain_version()
    if resolved not in SUPPORTED_CHAIN_VERSIONS:
        raise ValueError(f"지원하지 않는 채팅 버전입니다: {resolved!r}")
    return resolved


def _dispatch(version):
    version = resolve_version(version)
    if version == ChainVersion.V1:
        return chat_v1.send_message, chat_v1.message_update
    return chat_v2.send_message, chat_v2.message_update


# ── 공개 유스케이스 ────────────────────────────────────────────────────────────────
def send_message(session, content, context=None, version=None):
    """사용자 메시지를 저장하고 (event, data) 튜플을 흘려보내는 제너레이터를 돌려준다.

    context: {"stadium"?, "intent"?, "origin"?} 선택 사항 (v2 만 쓴다).
    version: "v1"/"v2". 없으면 env(LLM_CHAIN_VERSION). 그 밖의 값은 ValueError (저장 전).
    HTTP/SSE 로 감싸는 일은 views 의 책임이다.
    """
    send, _update = _dispatch(version)
    return send(session, content, context)


def list_messages(request, session_id):
    """소유한 세션의 최신 state → 공개 대화 항목 목록 (옛 wire 필드 포함)."""
    session = get_owned_session(request, session_id)
    thread = ChatThread(session.id)
    return wire_history(*thread.state(), thread.wire)


def message_update(request, session_id, message_id, content, context=None, version=None):
    """해당 사용자 메시지 뒤를 지우고 같은 ID 로 질문을 바꾼 뒤 다시 답한다."""
    _send, update = _dispatch(version)  # 검증 먼저: 잘못된 버전이면 세션 조회도 하지 않는다
    session = get_owned_session(request, session_id)
    return update(session, message_id, content, context)


def message_delete(request, session_id, message_id):
    """해당 사용자 메시지부터 이후 메시지를 최신 대화에서 제거한다. 제거한 메시지 수를 돌려준다."""
    session = get_owned_session(request, session_id)
    return ChatThread(session.id).delete_from(message_id)
