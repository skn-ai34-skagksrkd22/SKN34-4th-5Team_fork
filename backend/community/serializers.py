import uuid

from django.db import models, transaction
from drf_spectacular.utils import extend_schema_serializer
from rest_framework import serializers

from .models import CommunityComment, CommunityImage, CommunityPost, FREE_CATEGORIES, TEAM_CATEGORIES, TEAM_CODES


FONT_NAMES = {"sans", "serif", "mono"}
FONT_SIZES = {12, 14, 16, 18, 20, 24, 28, 32}
FONT_COLORS = {"#26354b", "#e1131b", "#246bf3", "#18825c", "#7550ae"}


def validate_content_doc(document, user, post=None, course=None, max_chars=20000):
    if not isinstance(document, dict) or set(document) != {"version", "blocks"} or type(document["version"]) is not int or document["version"] != 1:
        raise serializers.ValidationError("본문 서식이 올바르지 않아요.")
    blocks = document["blocks"]
    if not isinstance(blocks, list) or not 1 <= len(blocks) <= 500:
        raise serializers.ValidationError("본문 서식이 올바르지 않아요.")
    lines, image_ids, chars = [], [], 0
    for block in blocks:
        if not isinstance(block, dict):
            raise serializers.ValidationError("본문 서식이 올바르지 않아요.")
        if block.get("type") == "image":
            if set(block) != {"type", "id"}:
                raise serializers.ValidationError("이미지 정보가 올바르지 않아요.")
            try:
                image_id = uuid.UUID(block["id"])
                if not isinstance(block["id"], str) or str(image_id) != block["id"].lower():
                    raise ValueError("noncanonical image id")
                image_ids.append(image_id)
            except (TypeError, ValueError, AttributeError) as exc:
                raise serializers.ValidationError("이미지 정보가 올바르지 않아요.") from exc
            lines.append("[이미지]")
            continue
        if set(block) != {"type", "align", "runs"} or block["type"] != "paragraph" or not isinstance(block["align"], str) or block["align"] not in {"left", "center", "right"}:
            raise serializers.ValidationError("문단 서식이 올바르지 않아요.")
        runs = block["runs"]
        if not isinstance(runs, list) or len(runs) > 500:
            raise serializers.ValidationError("글자 서식이 올바르지 않아요.")
        pieces = []
        for run in runs:
            if not isinstance(run, dict) or set(run) != {"text", "font", "size", "color", "bold", "italic", "underline"}:
                raise serializers.ValidationError("글자 서식이 올바르지 않아요.")
            if not isinstance(run["text"], str) or len(run["text"]) > 20000 or "\n" in run["text"] or "\r" in run["text"]:
                raise serializers.ValidationError("본문 글자가 올바르지 않아요.")
            if not isinstance(run["font"], str) or run["font"] not in FONT_NAMES or type(run["size"]) is not int or run["size"] not in FONT_SIZES or not isinstance(run["color"], str) or run["color"] not in FONT_COLORS:
                raise serializers.ValidationError("글꼴 서식이 올바르지 않아요.")
            if any(type(run[field]) is not bool for field in ("bold", "italic", "underline")):
                raise serializers.ValidationError("글자 서식이 올바르지 않아요.")
            chars += len(run["text"])
            pieces.append(run["text"])
        lines.append("".join(pieces))
    if chars > max_chars or len(image_ids) > 10 or len(set(image_ids)) != len(image_ids):
        raise serializers.ValidationError(f"본문은 {max_chars:,}자, 이미지는 10장까지 등록할 수 있어요.")
    if image_ids and not user.is_authenticated:
        raise serializers.ValidationError("이미지를 첨부하려면 로그인해 주세요.")
    allowed = CommunityImage.objects.filter(id__in=image_ids, owner=user) if image_ids else CommunityImage.objects.none()
    if course is not None:
        allowed = allowed.filter(draft__isnull=True, post__isnull=True).filter(models.Q(course__isnull=True) | models.Q(course=course))
    elif post is None:
        allowed = allowed.filter(draft__isnull=True, post__isnull=True, course__isnull=True)
    else:
        allowed = allowed.filter(draft__isnull=True, course__isnull=True).filter(models.Q(post__isnull=True) | models.Q(post=post))
    if allowed.count() != len(image_ids):
        raise serializers.ValidationError("본인이 올린 이미지만 첨부할 수 있어요.")
    return "\n".join(lines).strip(), image_ids


@extend_schema_serializer(component_name="CommunityImageMetadata")
class CommunityImageMetadataSerializer(serializers.ModelSerializer):
    contentType = serializers.CharField(source="content_type", read_only=True)

    class Meta:
        model = CommunityImage
        fields = ("id", "contentType", "size", "width", "height")
        read_only_fields = fields


@extend_schema_serializer(component_name="CommunityPost")
class CommunityPostSerializer(serializers.ModelSerializer):
    id = serializers.CharField(source="source_id", read_only=True)
    sourceId = serializers.CharField(source="source_id", read_only=True)
    postNumber = serializers.CharField(source="post_number", read_only=True)
    teamCode = serializers.CharField(source="team_code", allow_blank=True)
    authorId = serializers.IntegerField(source="owner_id", read_only=True, allow_null=True)
    createdAt = serializers.DateTimeField(source="created_at", read_only=True, allow_null=True)
    views = serializers.IntegerField(read_only=True)
    recommendations = serializers.SerializerMethodField()
    downvotes = serializers.SerializerMethodField()
    commentCount = serializers.SerializerMethodField()
    isSample = serializers.BooleanField(source="is_sample", read_only=True)
    images = CommunityImageMetadataSerializer(many=True, read_only=True)
    category = serializers.ChoiceField(choices=TEAM_CATEGORIES)
    content = serializers.CharField(max_length=20000, allow_blank=False, trim_whitespace=True)
    contentDoc = serializers.JSONField(source="content_doc", allow_null=True, required=False)

    class Meta:
        model = CommunityPost
        fields = (
            "id", "sourceId", "postNumber", "board", "teamCode", "authorId", "author", "title", "content", "contentDoc",
            "category", "createdAt", "views", "recommendations", "downvotes", "commentCount", "isSample",
            "images",
        )
        read_only_fields = ("author",)

    @staticmethod
    def _count(obj, annotation, relation, value=None):
        if hasattr(obj, annotation):
            return getattr(obj, annotation)
        queryset = getattr(obj, relation)
        return queryset.filter(value=value).count() if value else queryset.count()

    def get_recommendations(self, obj) -> int:
        return self._count(obj, "upvote_count", "votes", "up")

    def get_downvotes(self, obj) -> int:
        return self._count(obj, "downvote_count", "votes", "down")

    def get_commentCount(self, obj) -> int:
        return self._count(obj, "actual_comment_count", "comments")

    def validate(self, attrs):
        board = attrs.get("board", getattr(self.instance, "board", None))
        team_code = attrs.get("team_code", getattr(self.instance, "team_code", "")).upper()
        category = attrs.get("category", getattr(self.instance, "category", None))

        if board == "free" and team_code:
            raise serializers.ValidationError({"teamCode": "자유게시판은 팀 코드를 사용할 수 없습니다."})
        if board == "teams" and team_code not in TEAM_CODES:
            raise serializers.ValidationError({"teamCode": "올바른 팀 코드를 입력해 주세요."})
        allowed_categories = FREE_CATEGORIES if board == "free" else TEAM_CATEGORIES
        if category not in allowed_categories:
            raise serializers.ValidationError({"category": "올바른 카테고리를 입력해 주세요."})
        if "content_doc" in attrs and attrs["content_doc"] is not None:
            text, image_ids = validate_content_doc(attrs["content_doc"], self.context["request"].user, self.instance)
            if not text or attrs.get("content", getattr(self.instance, "content", "")) != text:
                raise serializers.ValidationError({"contentDoc": "본문 내용과 서식이 일치하지 않아요."})
            self._image_ids = image_ids
        elif "content" in attrs and self.instance and self.instance.content_doc is not None:
            attrs["content_doc"] = None
        attrs["team_code"] = team_code
        return attrs

    def create(self, validated_data):
        post = super().create(validated_data)
        if hasattr(self, "_image_ids"):
            CommunityImage.objects.filter(id__in=self._image_ids, owner=post.owner).update(post=post)
        return post

    @transaction.atomic
    def update(self, instance, validated_data):
        post = super().update(instance, validated_data)
        if hasattr(self, "_image_ids"):
            CommunityImage.objects.filter(post=post).exclude(id__in=self._image_ids).update(post=None)
            CommunityImage.objects.filter(id__in=self._image_ids, owner=post.owner).update(post=post)
        elif validated_data.get("content_doc", "not-updated") is None:
            CommunityImage.objects.filter(post=post).update(post=None)
        return post


@extend_schema_serializer(component_name="CommunityPostWrite")
class CommunityPostWriteSerializer(serializers.Serializer):
    board = serializers.ChoiceField(choices=("free", "teams"))
    teamCode = serializers.CharField(max_length=2, allow_blank=True)
    category = serializers.ChoiceField(choices=TEAM_CATEGORIES)
    title = serializers.CharField(max_length=200)
    content = serializers.CharField(max_length=20000)
    contentDoc = serializers.JSONField(required=False, allow_null=True)


@extend_schema_serializer(component_name="CommunityPostPatch")
class CommunityPostPatchSerializer(CommunityPostWriteSerializer):
    board = serializers.ChoiceField(choices=("free", "teams"), required=False)
    teamCode = serializers.CharField(max_length=2, allow_blank=True, required=False)
    category = serializers.ChoiceField(choices=TEAM_CATEGORIES, required=False)
    title = serializers.CharField(max_length=200, required=False)
    content = serializers.CharField(max_length=20000, required=False)
    contentDoc = serializers.JSONField(required=False, allow_null=True)


@extend_schema_serializer(component_name="CommunityCommentWrite")
class CommunityCommentWriteSerializer(serializers.Serializer):
    content = serializers.CharField(max_length=2000, allow_blank=False, trim_whitespace=True)


@extend_schema_serializer(component_name="CommunityComment")
class CommunityCommentSerializer(serializers.ModelSerializer):
    postId = serializers.CharField(source="post_id", read_only=True)
    authorId = serializers.IntegerField(source="author_id", read_only=True)
    author = serializers.SerializerMethodField()
    createdAt = serializers.DateTimeField(source="created_at", read_only=True)
    updatedAt = serializers.DateTimeField(source="updated_at", read_only=True)

    class Meta:
        model = CommunityComment
        fields = ("id", "postId", "authorId", "author", "content", "createdAt", "updatedAt")
        read_only_fields = ("id",)

    def get_author(self, comment) -> str:
        return comment.author.nickname or comment.author.username

@extend_schema_serializer(component_name="CommunityMemberComment")
class CommunityMemberCommentSerializer(serializers.ModelSerializer):
    """
    회원 활동 페이지에서 사용하는 댓글 응답입니다.

    일반 댓글 API보다 원문 게시글 정보를 추가로 제공합니다.
    """

    createdAt = serializers.DateTimeField(source="created_at",read_only=True,)
    postId = serializers.CharField(source="post.source_id",read_only=True,)
    postTitle = serializers.CharField(source="post.title",read_only=True,)
    board = serializers.CharField(source="post.board",read_only=True,)
    teamCode = serializers.CharField(source="post.team_code",read_only=True,)
    class Meta:
        model = CommunityComment
        fields = (
            "id",
            "content",
            "createdAt",
            "postId",
            "postTitle",
            "board",
            "teamCode",
        )
        read_only_fields = fields
        
@extend_schema_serializer(component_name="CommunityVoteWrite")
class CommunityVoteWriteSerializer(serializers.Serializer):
    vote = serializers.ChoiceField(choices=("up", "down"), allow_null=True)


@extend_schema_serializer(component_name="CommunityVoteState")
class CommunityVoteStateSerializer(serializers.Serializer):
    vote = serializers.ChoiceField(choices=("up", "down"), allow_null=True)
    recommendations = serializers.IntegerField(min_value=0)
    downvotes = serializers.IntegerField(min_value=0)


@extend_schema_serializer(component_name="CommunityReportWrite")
class CommunityReportWriteSerializer(serializers.Serializer):
    reason = serializers.ChoiceField(choices=("spam", "abuse", "inappropriate", "privacy", "other"))
    detail = serializers.CharField(max_length=50, allow_blank=True, trim_whitespace=True)


@extend_schema_serializer(component_name="CommunityReportResult")
class CommunityReportResultSerializer(serializers.Serializer):
    id = serializers.IntegerField(min_value=1)
    created = serializers.BooleanField()


@extend_schema_serializer(component_name="PredictionChoiceWrite")
class PredictionChoiceWriteSerializer(serializers.Serializer):
    choice = serializers.RegexField(r"^(home|away)$", allow_null=True)


@extend_schema_serializer(component_name="PredictionTeam")
class PredictionTeamSerializer(serializers.Serializer):
    code = serializers.ChoiceField(choices=TEAM_CODES)
    name = serializers.CharField()
    score = serializers.IntegerField(min_value=0, allow_null=True)


@extend_schema_serializer(component_name="PredictionVotes")
class PredictionVotesSerializer(serializers.Serializer):
    home = serializers.IntegerField(min_value=0)
    away = serializers.IntegerField(min_value=0)
    total = serializers.IntegerField(min_value=0)
    homePercent = serializers.IntegerField(min_value=0, max_value=100)
    awayPercent = serializers.IntegerField(min_value=0, max_value=100)


@extend_schema_serializer(component_name="PredictionGame")
class PredictionGameSerializer(serializers.Serializer):
    gameId = serializers.CharField()
    date = serializers.DateField()
    startsAt = serializers.DateTimeField(allow_null=True)
    stadium = serializers.CharField(allow_blank=True)
    away = PredictionTeamSerializer()
    home = PredictionTeamSerializer()
    status = serializers.ChoiceField(choices=("scheduled", "live", "final", "cancelled", "postponed", "suspended", "unknown"))
    result = serializers.ChoiceField(choices=("home", "away", "draw"), allow_null=True)
    locked = serializers.BooleanField()
    voided = serializers.BooleanField()
    stale = serializers.BooleanField()
    sourceFetchedAt = serializers.DateTimeField()
    votes = PredictionVotesSerializer()
    myChoice = serializers.RegexField(r"^(home|away)$", allow_null=True)
