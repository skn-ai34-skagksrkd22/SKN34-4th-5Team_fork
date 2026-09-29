from django.db import transaction
from django.db.models import Count, Q
from django.shortcuts import get_object_or_404
from django.contrib.auth import get_user_model
from drf_spectacular.utils import OpenApiParameter, OpenApiResponse, OpenApiTypes, extend_schema
from rest_framework import serializers, status
from rest_framework.exceptions import NotFound, PermissionDenied
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import CommunityComment, CommunityPost, CommunityReport, CommunityVote
from .serializers import (
    CommunityCommentSerializer,
    CommunityCommentWriteSerializer,
    CommunityMemberCommentSerializer,  # 추가
    CommunityReportResultSerializer,
    CommunityReportWriteSerializer,
    CommunityVoteStateSerializer,
    CommunityVoteWriteSerializer,
)

from .pagination import CommunityCommentPagination

User = get_user_model()

def _vote_counts(post):
    return post.votes.aggregate(
        recommendations=Count("id", filter=Q(value="up")),
        downvotes=Count("id", filter=Q(value="down")),
    )


def _vote_response(post, user):
    counts = _vote_counts(post)
    return {
        "vote": post.votes.filter(user=user).values_list("value", flat=True).first(),
        **counts,
    }


class CommentListCreateView(APIView):
    def get_permissions(self):
        return [AllowAny()] if self.request.method == "GET" else [IsAuthenticated()]

    @extend_schema(
        parameters=[OpenApiParameter("order", OpenApiTypes.STR, enum=("oldest", "newest"))],
        responses={200: CommunityCommentSerializer(many=True), 400: OpenApiTypes.OBJECT, 404: OpenApiTypes.OBJECT}, auth=[],
    )
    def get(self, request, source_id):
        post = get_object_or_404(CommunityPost, source_id=source_id)
        order = request.query_params.get("order", "oldest")
        if order not in {"oldest", "newest"}:
            raise serializers.ValidationError({"order": "oldest 또는 newest를 입력해 주세요."})
        ordering = ("created_at", "id") if order == "oldest" else ("-created_at", "-id")
        comments = post.comments.select_related("author").order_by(*ordering)
        return Response(CommunityCommentSerializer(comments, many=True).data)

    @extend_schema(
        request=CommunityCommentWriteSerializer,
        responses={201: CommunityCommentSerializer, 400: OpenApiTypes.OBJECT, 401: OpenApiTypes.OBJECT, 404: OpenApiTypes.OBJECT},
    )
    @transaction.atomic
    def post(self, request, source_id):
        post = get_object_or_404(CommunityPost.objects.select_for_update(), source_id=source_id)
        serializer = CommunityCommentWriteSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        comment = CommunityComment.objects.create(
            post=post, author=request.user, **serializer.validated_data
        )
        CommunityPost.objects.filter(pk=post.pk).update(comment_count=post.comments.count())
        return Response(CommunityCommentSerializer(comment).data, status=status.HTTP_201_CREATED)

class MemberCommentListView(APIView):
    """
    특정 회원이 작성한 댓글을 조회합니다.

    GET /api/v1/community/comments/?author_id=21&page=1&page_size=20

    - 로그인 사용자만 조회 가능
    - 본인은 공개 설정과 관계없이 조회 가능
    - 다른 회원은 visibility.posts=True인 경우만 조회 가능
    - 비활성/존재하지 않는 회원은 404
    - 숨겨진 게시글의 댓글은 조회하지 않음
    """

    permission_classes = (IsAuthenticated,)

    @extend_schema(
        parameters=[
            OpenApiParameter(
                "author_id",
                OpenApiTypes.INT,
                required=True,
                description="댓글 작성자 회원 ID",
            ),
            OpenApiParameter(
                "page",
                OpenApiTypes.INT,
                required=False,
            ),
            OpenApiParameter(
                "page_size",
                OpenApiTypes.INT,
                required=False,
            ),
        ],
        responses={
            200: CommunityMemberCommentSerializer(many=True),
            400: OpenApiTypes.OBJECT,
            401: OpenApiTypes.OBJECT,
            403: OpenApiTypes.OBJECT,
            404: OpenApiTypes.OBJECT,
        },
    )
    def get(self, request):
        author_id = request.query_params.get("author_id")

        # author_id가 없거나 양의 정수가 아니면 잘못된 요청
        if (
            not author_id
            or not author_id.isascii()
            or not author_id.isdigit()
            or author_id.startswith("0")
        ):
            raise serializers.ValidationError(
                {"author_id": "author_id는 양의 정수여야 합니다."}
            )

        # 비활성 회원은 존재하지 않는 회원과 동일하게 처리
        target_user = User.objects.filter(
            pk=int(author_id),
            is_active=True,
        ).first()

        if not target_user:
            raise NotFound("사용자를 찾을 수 없습니다.")

        # 본인은 공개 여부와 관계없이 자신의 댓글을 조회할 수 있음
        # 다른 회원은 활동 공개 설정이 켜져 있어야 조회 가능
        if (
            target_user.id != request.user.id
            and not bool((target_user.visibility or {}).get("posts", False))
        ):
            raise PermissionDenied("공개하지 않은 활동입니다.")

        # 숨겨진 게시글의 댓글은 회원 활동 목록에서도 제외
        comments = (
            CommunityComment.objects
            .select_related("author", "post")
            .filter(
                author=target_user,
                post__is_hidden=False,
            )
            .order_by("-created_at", "-id")
        )

        # 기존 공개 pagination 형식 사용
        paginator = CommunityCommentPagination()
        page = paginator.paginate_queryset(
            comments,
            request,
            view=self,
        )

        serializer = CommunityMemberCommentSerializer(
            page if page is not None else comments,
            many=True,
        )

        # page/page_size가 전달된 경우:
        # {count, next, previous, results} 형태로 반환
        if page is not None:
            return paginator.get_paginated_response(serializer.data)

        return Response(serializer.data)

class CommentDetailView(APIView):
    permission_classes = (IsAuthenticated,)

    @staticmethod
    def _locked_owned_comment(comment_id, user):
        post_id = get_object_or_404(
            CommunityComment.objects.only("post_id"), pk=comment_id
        ).post_id
        post = get_object_or_404(CommunityPost.objects.select_for_update(), pk=post_id)
        comment = get_object_or_404(
            CommunityComment.objects.select_for_update().select_related("author"),
            pk=comment_id,
            post=post,
        )
        if comment.author_id != user.id:
            raise PermissionDenied("본인 댓글만 수정하거나 삭제할 수 있습니다.")
        return post, comment

    @extend_schema(
        request=CommunityCommentWriteSerializer,
        responses={200: CommunityCommentSerializer, 400: OpenApiTypes.OBJECT, 401: OpenApiTypes.OBJECT, 403: OpenApiTypes.OBJECT, 404: OpenApiTypes.OBJECT},
    )
    @transaction.atomic
    def patch(self, request, comment_id):
        _, comment = self._locked_owned_comment(comment_id, request.user)
        serializer = CommunityCommentWriteSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        comment.content = serializer.validated_data["content"]
        comment.save(update_fields=("content", "updated_at"))
        return Response(CommunityCommentSerializer(comment).data)

    @extend_schema(responses={204: OpenApiResponse(description="본문 없음"), 401: OpenApiTypes.OBJECT, 403: OpenApiTypes.OBJECT, 404: OpenApiTypes.OBJECT})
    @transaction.atomic
    def delete(self, request, comment_id):
        post, comment = self._locked_owned_comment(comment_id, request.user)
        comment.delete()
        CommunityPost.objects.filter(pk=post.pk).update(comment_count=post.comments.count())
        return Response(status=status.HTTP_204_NO_CONTENT)


class VoteView(APIView):
    permission_classes = (IsAuthenticated,)

    @extend_schema(responses={200: CommunityVoteStateSerializer, 401: OpenApiTypes.OBJECT, 404: OpenApiTypes.OBJECT})
    def get(self, request, source_id):
        post = get_object_or_404(CommunityPost, source_id=source_id)
        return Response(CommunityVoteStateSerializer(_vote_response(post, request.user)).data)

    @extend_schema(
        request=CommunityVoteWriteSerializer,
        responses={200: CommunityVoteStateSerializer, 400: OpenApiTypes.OBJECT, 401: OpenApiTypes.OBJECT, 404: OpenApiTypes.OBJECT},
    )
    @transaction.atomic
    def post(self, request, source_id):
        post = get_object_or_404(CommunityPost.objects.select_for_update(), source_id=source_id)
        serializer = CommunityVoteWriteSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        desired = serializer.validated_data["vote"]
        current = CommunityVote.objects.filter(post=post, user=request.user).first()

        if desired is None:
            if current:
                current.delete()
        elif current:
            if current.value != desired:
                current.value = desired
                current.save(update_fields=("value",))
        else:
            CommunityVote.objects.create(post=post, user=request.user, value=desired)

        counts = _vote_counts(post)
        CommunityPost.objects.filter(pk=post.pk).update(recommendations=counts["recommendations"])
        return Response(CommunityVoteStateSerializer({"vote": desired, **counts}).data)


class ReportCreateView(APIView):
    permission_classes = (IsAuthenticated,)

    @extend_schema(
        request=CommunityReportWriteSerializer,
        responses={200: CommunityReportResultSerializer, 201: CommunityReportResultSerializer, 400: OpenApiTypes.OBJECT, 401: OpenApiTypes.OBJECT, 404: OpenApiTypes.OBJECT},
    )
    def post(self, request, source_id):
        post = get_object_or_404(CommunityPost, source_id=source_id)
        serializer = CommunityReportWriteSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        report, created = CommunityReport.objects.get_or_create(
            post=post,
            reporter=request.user,
            defaults=serializer.validated_data,
        )
        response_status = status.HTTP_201_CREATED if created else status.HTTP_200_OK
        data = CommunityReportResultSerializer({"id": report.id, "created": created}).data
        return Response(data, status=response_status)
