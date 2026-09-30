from rest_framework.permissions import AllowAny
from rest_framework.generics import GenericAPIView
from rest_framework.renderers import BaseRenderer, JSONRenderer
from rest_framework.response import Response
from rest_framework import status as http_status

from llm.serializer.message import (
    ChatMessageDeleteSerializer,
    ChatMessageInputSerializer,
    ChatMessageUpdateSerializer,
)
from llm.enum import ChainVersion
from llm.service import chat as chat_service
from llm.service import chat_v1, chat_v2
from llm.service.chat import list_messages, message_delete
from llm.service.ownership import get_owned_session
from llm.views.sse import event_stream_response


def _dispatch(version):
    """version -> chat_v1/chat_v2 의 (send_message, message_update)."""
    if chat_service.resolve_version(version) == ChainVersion.V1:
        return chat_v1.send_message, chat_v1.message_update
    return chat_v2.send_message, chat_v2.message_update


class EventStreamRenderer(BaseRenderer):
    """POST/PUT 성공 응답은 StreamingHttpResponse 를 그대로 돌려주므로 이 렌더러를 거치지
    않는다 - DRF 콘텐츠 협상이 Accept: text/event-stream 요청을 거부하지 않게만 해준다.

    하지만 GET(목록 조회)은 항상 일반 Response 를 통해 여기로 오고, is_valid(raise_exception=
    True)/404 같은 스트림 시작 *전* 에러도 DRF 예외 핸들러가 만든 일반 Response 를 이 렌더러로
    다시 협상한다 (Accept 가 text/event-stream 이라서). 이때 원래 data(list/dict) 를 그대로
    돌려주면 Django 가 그 키/값만 바이트로 이어붙여 응답 바디로 써버려 Content-Type 은
    text/event-stream 인데 본문은 깨진 상태가 된다. 그래서 이 렌더러로 오는 모든 일반 Response
    는 JSONRenderer 로 위임하고 Content-Type 도 application/json 으로 맞춘다 (실제 SSE 성공
    스트림은 StreamingHttpResponse 라 이 렌더러 자체를 거치지 않으므로 영향 없음).
    """

    media_type = "text/event-stream"
    format = "sse"

    def render(self, data, accepted_media_type=None, renderer_context=None):
        renderer_context = renderer_context or {}
        response = renderer_context.get("response")
        if response is not None:
            response["Content-Type"] = "application/json"
        return JSONRenderer().render(data, "application/json", renderer_context)


class ChatMessageView(GenericAPIView):
    """채팅을 비회원 / 회원 둘다 동시에 할수있도록 처리한다.
    URL:
        - /api/v1/chat/sessions/<session_id>/messages/
        - /api/v2/chat/sessions/<session_id>/messages/

    (config/urls.py 는 plain path("api/", include("llm.urls")) 로 llm/urls.py 를 include 하고,
    그 안의 re_path(r"^(?P<version>v1|v2)/chat/", ...) 가 URL 을 kwargs["version"] 으로 넘긴다.
    정규식이 이미 v1/v2 만 허용하므로 여기서는 그 값을 그대로 service 에 전달한다 -- 중복 검증 없음.)

    GET: 해당 대화방의 저장된 메시지를 조회합니다 (공개 항목 {id, role, content, status, tools}).
    POST: 사용자 메시지를 보내고 AI 답변을 생성합니다 (SSE 스트리밍).
    PUT: 메시지 하나를 수정하고 그 이후 대화를 다시 생성합니다.
    DELETE: 메시지 하나부터 이후 대화를 모두 삭제합니다.
    """
    permission_classes = [AllowAny]
    renderer_classes = [JSONRenderer, EventStreamRenderer]

    # GET: /api/v2/chat/sessions/<session_id>/messages/
    def get(self, request, *args, **kwargs):
        """해당 세션의 채팅목록을 가져온다. 소유하지 않은/존재하지 않는 세션이면 404."""
        return Response(list_messages(request, kwargs["session_id"]))

    # POST: /api/v2/chat/sessions/<session_id>/messages/
    def post(self, request, *args, **kwargs):
        """LLM 호출 (SSE 스트리밍). version 으로 chat_v1/chat_v2 의 실제 구현을 직접 고른다."""
        session = get_owned_session(request, kwargs["session_id"])
        serializer = ChatMessageInputSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        send, _update = _dispatch(kwargs.get("version"))
        events = send(session, data["content"], data.get("context"))
        return event_stream_response(events)

    def put(self, request, *args, **kwargs):
        """이후 채팅목록을 수정 + LLM 호출. version 으로 chat_v1/chat_v2 의 실제 구현을 직접 고른다."""
        serializer = ChatMessageUpdateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        _send, update = _dispatch(kwargs.get("version"))
        session = get_owned_session(request, kwargs["session_id"])
        events = update(session, data["message_id"], data["content"], data.get("context"))
        return event_stream_response(events)

    def delete(self, request, *args, **kwargs):
        """이후 채팅목록을 삭제한다."""
        serializer = ChatMessageDeleteSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        message_id = serializer.validated_data["message_id"]
        deleted_count = message_delete(request, kwargs["session_id"], message_id)
        return Response({"deleted_count": deleted_count}, status=http_status.HTTP_204_NO_CONTENT)
