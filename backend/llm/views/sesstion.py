import uuid

from rest_framework.permissions import AllowAny
from rest_framework.request import Request
from rest_framework.pagination import PageNumberPagination
from rest_framework.generics import GenericAPIView
from rest_framework import mixins
from llm.serializer.sesstion import ChatSessionSerializer
from llm.service.ownership import (
    GUEST_COOKIE_NAME,
    get_owned_session,
    owned_session_queryset,
    parse_guest_id,
)


class ChatRoomListView(
    mixins.ListModelMixin,      # GET       요청 : 채팅방 리스트전달
    mixins.CreateModelMixin,    # POST      요청 : 채팅방 생성
    GenericAPIView
):
    """ 채팅방 리스트 조회 View
        URL: /api/v2/chat/sessions/

        GET: 회원/비회원 모두 자신의 대화방 목록을 조회합니다.
                return list_대화방
        POST: 회원/비회원 모두 새 대화방을 생성합니다. 비회원은 guest_id 쿠키로 식별합니다.
    """
    permission_classes = [AllowAny]
    serializer_class = ChatSessionSerializer
    pagination_class = PageNumberPagination

    def get_queryset(self):
        return owned_session_queryset(self.request).order_by('-updated_at')

    # GET /api/v2/chat/sessions/ : 회원/비회원 자신의 대화방 목록을 조회 합니다.
    def get(self, request : Request, *args, **kwargs):
        return self.list(request, *args, **kwargs)

    def perform_create(self, serializer):
        # 회원시
        if self.request.user.is_authenticated:
            serializer.save(
                user=self.request.user,
                guest=None
            )
        # 비회원시
        else:
            serializer.save(
                user=None,
                guest=self.guest_id
            )

    # POST /api/v2/chat/sessions/
    def post(self, request : Request, *args, **kwargs):
        # 1. 만약 회원 사용자일시 
        if request.user.is_authenticated:
            return self.create(request, *args, **kwargs)
        # 2. 만약 비회왼이라면
        # 3. 쿠키에서 Guest_id 를 찾는다 (UUID 형식 검증은 ownership.parse_guest_id 한 곳에서).
        self.guest_id = parse_guest_id(request)
        # 4. 쿠키가 없거나 깨진 UUID 면 새로 발급하고, 아래 set_cookie 가 기존 쿠키를 덮어쓴다.
        if self.guest_id is None:
            self.guest_id = uuid.uuid4()
        # 5. 비회원 전용 채팅방 생성
        response = self.create(request, *args, **kwargs)

        # 브라우저가 다음 요청에도 동일 guest를 식별하도록 저장
        response.set_cookie(
            GUEST_COOKIE_NAME,
            str(self.guest_id),
            httponly=True,
            samesite="Lax",
        )

        return response
    
class ChatRoomDetailView(
    mixins.UpdateModelMixin,    # PATCH     요청 : 채팅방 수정(예: 체팅 제목)
    mixins.DestroyModelMixin,   # DELETE    요청 : 채팅방 삭제
    GenericAPIView
):
    """
    URL: /api/v2/chat/sessions/<session_id>/
    
    PATCH: 대화방 정보를 수정합니다.
    DELETE: 대화방을 삭제합니다. 대화 checkpoint 는 commit 뒤 지운다(llm.apps 의 post_delete outbox, 실패 시 재시도).
    """

    permission_classes = [AllowAny]
    serializer_class = ChatSessionSerializer

    def get_object(self):
        return get_owned_session(self.request, self.kwargs["session_id"])

    def put(self, request, *args, **kwargs):
        return self.update(request, *args, **kwargs)
    
    def patch(self, request, *args, **kwargs):
        return self.partial_update(request, *args, **kwargs)

    def delete(self, request, *args, **kwargs):
        return self.destroy(request, *args, **kwargs)
