from django.db import DataError, IntegrityError, transaction
from django.db.models import Count, F, Q
from django.contrib.auth import get_user_model
from drf_spectacular.utils import OpenApiParameter, OpenApiResponse, OpenApiTypes, PolymorphicProxySerializer, extend_schema, extend_schema_view
from rest_framework import generics, status
from rest_framework.exceptions import NotAuthenticated, PermissionDenied, ValidationError
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from .models import CommunityPost, TEAM_CODES
from .pagination import CommunityPostPageSerializer, CommunityPostPagination, CommunityPostQuery
from .serializers import CommunityPostPatchSerializer, CommunityPostSerializer, CommunityPostWriteSerializer


POST_INPUT_FIELDS = ("board", "team_code", "category", "title", "content", "content_doc")

User = get_user_model()

def post_queryset():
    return CommunityPost.objects.prefetch_related("images").annotate(
        upvote_count=Count("votes", filter=Q(votes__value="up"), distinct=True),
        downvote_count=Count("votes", filter=Q(votes__value="down"), distinct=True),
        actual_comment_count=Count("comments", distinct=True),
    ).order_by("-post_number")


def same_submission(post, validated_data):
    return all(getattr(post, field) == validated_data.get(field, None) for field in POST_INPUT_FIELDS)


@extend_schema_view(
    get=extend_schema(
        parameters=[
            OpenApiParameter("board", OpenApiTypes.STR, enum=("free", "teams")),
            OpenApiParameter("team", OpenApiTypes.STR),
            OpenApiParameter("mine", OpenApiTypes.STR, enum=("1",)),

            # 특정 회원이 작성한 게시글 조회
            OpenApiParameter("author_id", OpenApiTypes.INT, description="작성자 회원 ID", required=False,),
            OpenApiParameter("page", {"type": "integer", "minimum": 1, "maximum": 2_147_483_647},),
            OpenApiParameter("page_size", {"type": "integer", "minimum": 1, "maximum": 100},),
            OpenApiParameter("q", {"type": "string", "maxLength": 200}),
            OpenApiParameter("search_field", OpenApiTypes.STR, enum=("all", "title", "author"),),
        ],
        responses={
            200: PolymorphicProxySerializer(
                component_name="CommunityPostListResponse",
                serializers=[CommunityPostSerializer(many=True), CommunityPostPageSerializer],
                resource_type_field_name=None,
                many=False,
            ),
            400: OpenApiTypes.OBJECT,
            401: OpenApiTypes.OBJECT,
            404: OpenApiTypes.OBJECT,
        },
    ),
    post=extend_schema(
        request=CommunityPostWriteSerializer,
        parameters=[OpenApiParameter("Idempotency-Key", OpenApiTypes.STR, OpenApiParameter.HEADER, required=True)],
        responses={200: CommunityPostSerializer, 201: CommunityPostSerializer, 400: OpenApiTypes.OBJECT, 401: OpenApiTypes.OBJECT, 409: OpenApiTypes.OBJECT},
    ),
)
class CommunityPostListCreateView(generics.ListCreateAPIView):
    serializer_class = CommunityPostSerializer
    pagination_class = CommunityPostPagination
    permission_classes = (AllowAny,)
    http_method_names = ("get", "post", "head", "options")

    def get_permissions(self):
        return (IsAuthenticated(),) if self.request.method == "POST" else (AllowAny(),)

    def get_queryset(self):
        # 신고 처리로 숨긴 글은 공개 목록에 나오지 않는다
        queryset = post_queryset().filter(is_hidden=False)
        query = CommunityPostQuery.from_params(self.request.query_params)
        board = self.request.query_params.get("board")
        team = self.request.query_params.get("team")
        mine = self.request.query_params.get("mine")
        if board is not None and board not in {"free", "teams"}:
            raise ValidationError({"board": "free 또는 teams를 입력해 주세요."})
        if team is not None:
            team = team.upper()
            if team not in TEAM_CODES:
                raise ValidationError({"team": "올바른 팀 코드를 입력해 주세요."})
            if board == "free":
                raise ValidationError({"team": "팀 필터는 teams 게시판에서만 사용할 수 있습니다."})
            queryset = queryset.filter(team_code=team)
            
        author_id = self.request.query_params.get("author_id")

        if mine is not None:
            if mine != "1":
                raise ValidationError({"mine": "mine은 1만 사용할 수 있습니다."})

            if not self.request.user.is_authenticated:
                raise NotAuthenticated()

            queryset = queryset.filter(owner=self.request.user)

        # 특정 회원이 작성한 게시글 조회
        if author_id is not None:
            if (
                not author_id.isascii()
                or not author_id.isdigit()
                or author_id.startswith("0")
            ):
                raise ValidationError(
                    {"author_id": "author_id는 양의 정수여야 합니다."}
                )

            if not self.request.user.is_authenticated:
                raise NotAuthenticated()

            target_user = User.objects.filter(
                pk=int(author_id),
                is_active=True,
            ).first()

            if not target_user:
                from rest_framework.exceptions import NotFound
                raise NotFound("사용자를 찾을 수 없습니다.")

            if (
                target_user.id != self.request.user.id
                and not bool((target_user.visibility or {}).get("posts", False))
            ):
                raise PermissionDenied("공개하지 않은 활동입니다.")

            queryset = queryset.filter(owner=target_user)
            
        if board is not None:
            queryset = queryset.filter(board=board)
        if query.q:
            fields = {
                "all": Q(title__icontains=query.q) | Q(content__icontains=query.q),
                "title": Q(title__icontains=query.q),
                "author": Q(author__icontains=query.q),
            }
            queryset = queryset.filter(fields[query.search_field])
        return queryset

    def create(self, request, *args, **kwargs):
        key = request.headers.get("Idempotency-Key", "").strip()
        if not key or len(key) > 128:
            raise ValidationError({"idempotencyKey": "1~128자의 Idempotency-Key가 필요합니다."})

        existing = CommunityPost.objects.filter(owner=request.user, idempotency_key=key).first()
        serializer = self.get_serializer(existing, data=request.data) if existing else self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        values = serializer.validated_data
        if existing:
            if not same_submission(existing, values):
                return Response({"idempotencyKey": "같은 키로 다른 게시글을 만들 수 없습니다."}, status=status.HTTP_409_CONFLICT)
            return Response(self.get_serializer(post_queryset().get(pk=existing.pk)).data)

        try:
            with transaction.atomic():
                post = serializer.save(
                    owner=request.user,
                    author=request.user.nickname or request.user.username,
                    idempotency_key=key,
                    is_sample=False,
                )
        except IntegrityError:
            existing = CommunityPost.objects.filter(owner=request.user, idempotency_key=key).first()
            if existing and same_submission(existing, values):
                return Response(self.get_serializer(post_queryset().get(pk=existing.pk)).data)
            return Response({"idempotencyKey": "같은 키로 다른 게시글을 만들 수 없습니다."}, status=status.HTTP_409_CONFLICT)
        except DataError as exc:
            raise ValidationError({"postNumber": "게시글 번호를 더 발급할 수 없습니다."}) from exc

        post = post_queryset().get(pk=post.pk)
        return Response(self.get_serializer(post).data, status=status.HTTP_201_CREATED)


@extend_schema_view(
    get=extend_schema(responses={200: CommunityPostSerializer, 404: OpenApiTypes.OBJECT}),
    patch=extend_schema(
        request=CommunityPostPatchSerializer,
        responses={200: CommunityPostSerializer, 400: OpenApiTypes.OBJECT, 401: OpenApiTypes.OBJECT, 403: OpenApiTypes.OBJECT, 404: OpenApiTypes.OBJECT},
    ),
    delete=extend_schema(responses={204: OpenApiResponse(description="본문 없음"), 401: OpenApiTypes.OBJECT, 403: OpenApiTypes.OBJECT, 404: OpenApiTypes.OBJECT}),
)
class CommunityPostDetailView(generics.RetrieveUpdateDestroyAPIView):
    serializer_class = CommunityPostSerializer
    lookup_field = "source_id"
    lookup_url_kwarg = "source_id"
    http_method_names = ("get", "patch", "delete", "head", "options")

    def get_permissions(self):
        return (AllowAny(),) if self.request.method in {"GET", "HEAD", "OPTIONS"} else (IsAuthenticated(),)

    def get_queryset(self):
        # 숨긴 글은 관리자만 열람할 수 있다 (작성자의 수정·삭제는 그대로 허용)
        if self.request.method == "GET" and not self.request.user.is_staff:
            return post_queryset().filter(is_hidden=False)
        return post_queryset()

    def get_object(self):
        post = super().get_object()
        if self.request.method not in {"GET", "HEAD", "OPTIONS"} and post.owner_id != self.request.user.id:
            raise PermissionDenied("작성자만 수정하거나 삭제할 수 있습니다.")
        if self.request.method == "GET":
            CommunityPost.objects.filter(pk=post.pk).update(views=F("views") + 1)
            post = post_queryset().get(pk=post.pk)
        return post
