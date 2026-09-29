import base64
import calendar
import re
from io import BytesIO
from zoneinfo import ZoneInfo

from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError as DjangoValidationError
from rest_framework import serializers
from drf_spectacular.utils import extend_schema_field, extend_schema_serializer
from rest_framework.validators import UniqueValidator
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.serializers import TokenRefreshSerializer
from PIL import Image, UnidentifiedImageError

User = get_user_model()


class PasswordAwareTokenRefreshSerializer(TokenRefreshSerializer):
    def validate(self, attrs):
        JWTAuthentication().get_user(self.token_class(attrs["refresh"]))
        return super().validate(attrs)


class SignInRequestSerializer(serializers.Serializer):
    username = serializers.CharField(write_only=True)
    password = serializers.CharField(write_only=True, trim_whitespace=False)


class TokenPairSerializer(serializers.Serializer):
    access = serializers.CharField(read_only=True)
    refresh = serializers.CharField(read_only=True)


class TokenRefreshRequestSerializer(serializers.Serializer):
    refresh = serializers.CharField(write_only=True)


class TokenRefreshResponseSerializer(serializers.Serializer):
    access = serializers.CharField(read_only=True)


class LogoutRequestSerializer(serializers.Serializer):
    refresh = serializers.CharField(write_only=True)


class PasswordValidationMixin:
    """공통 비밀번호 검증 로직"""

    # 비밀번호 검증
    def validate_password(self, value):
        if len(value) < 8:
            raise serializers.ValidationError("비밀번호는 8자 이상이어야 합니다.")

        if not any(char.isdigit() for char in value) or not any(
            char.isalpha() for char in value
        ):
            raise serializers.ValidationError(
                "비밀번호는 영문자와 숫자를 모두 포함해야 합니다."
            )

        if any(char.isspace() for char in value):
            raise serializers.ValidationError("비밀번호에는 공백을 포함할 수 없습니다.")
        if len(value) > 128:
            raise serializers.ValidationError("비밀번호는 128자 이하여야 합니다.")

        return value


class SignupSerializer(PasswordValidationMixin, serializers.ModelSerializer):
    username = serializers.RegexField(r"^[A-Za-z0-9]{4,20}$", validators=[UniqueValidator(queryset=User.objects.all(), message="이미 사용 중인 아이디입니다.")])
    email = serializers.EmailField(required=True, max_length=254)
    first_name = serializers.CharField(required=True, allow_blank=False, max_length=150)
    birth_date = serializers.DateField(required=True)
    gender = serializers.ChoiceField(choices=User._meta.get_field("gender").choices, required=True)
    re_password = serializers.CharField(write_only=True, trim_whitespace=False)

    class Meta:
        model = User
        fields = ["username", "email", "password", "re_password", "first_name", "birth_date", "gender"]
        extra_kwargs = {"password": {"write_only": True, "trim_whitespace": False}}

    # 비밀번호 일치 검증
    def validate(self, attrs):
        if attrs.get("password") != attrs.get("re_password"):
            raise serializers.ValidationError({"re_password": "비밀번호가 서로 다릅니다."})

        try:
            validate_password(attrs["password"], user=User(username=attrs["username"]))
        except DjangoValidationError as error:
            raise serializers.ValidationError({"password": error.messages})

        return attrs

    def validate_birth_date(self, value):
        from django.utils import timezone
        if value > timezone.localdate():
            raise serializers.ValidationError("미래 생년월일은 사용할 수 없습니다.")
        return value

    def create(self, validated_data):
        validated_data.pop("re_password")
        return User.objects.create_user(**validated_data)


TEAM_CODES = {"LG", "HH", "SK", "SS", "NC", "KT", "LT", "HT", "OB", "WO"}
DEFAULT_NOTIFICATIONS = {"comments": True, "courses": True, "announcements": True}
DEFAULT_VISIBILITY = {"courses": False, "posts": False, "likes": False}


def _next_nickname_change(changed_at):
    local = changed_at.astimezone(ZoneInfo("Asia/Seoul"))
    month = local.month - 1 + 6
    year, month = local.year + month // 12, month % 12 + 1
    return local.replace(year=year, month=month, day=min(local.day, calendar.monthrange(year, month)[1]))


def _boolean_settings(value, defaults):
    if not isinstance(value, dict) or set(value) != set(defaults) or any(type(item) is not bool for item in value.values()):
        raise serializers.ValidationError("설정 값을 확인해 주세요.")
    return value


class NotificationSettingsSerializer(serializers.Serializer):
    comments = serializers.BooleanField()
    courses = serializers.BooleanField()
    announcements = serializers.BooleanField()


class VisibilitySettingsSerializer(serializers.Serializer):
    courses = serializers.BooleanField()
    posts = serializers.BooleanField()
    likes = serializers.BooleanField()


@extend_schema_field(NotificationSettingsSerializer)
class NotificationSettingsField(serializers.JSONField):
    pass


@extend_schema_field(VisibilitySettingsSerializer)
class VisibilitySettingsField(serializers.JSONField):
    pass


class MemberUserUpdateSerializer(serializers.ModelSerializer):
    notifications = NotificationSettingsField(required=False)
    visibility = VisibilitySettingsField(required=False)

    class Meta:
        model = User
        fields = ("id", "username", "email", "first_name", "birth_date", "gender", "is_staff", "is_superuser", "is_active", "nickname", "team_code", "avatar", "nickname_changed_at", "notifications", "visibility")
        read_only_fields = ("id", "username", "email", "is_staff", "is_superuser", "is_active", "nickname_changed_at")
        extra_kwargs = {"first_name": {"required": False}, "birth_date": {"required": False}, "gender": {"required": False}}

    def validate_nickname(self, value):
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z가-힣]{1,12}", value):
            raise serializers.ValidationError("닉네임은 한글·영문만 1~12자로 입력해 주세요.")
        if value != self.instance.nickname and self.instance.nickname_changed_at:
            from django.utils import timezone
            if timezone.now() < _next_nickname_change(self.instance.nickname_changed_at):
                raise serializers.ValidationError("닉네임은 변경 후 6개월이 지나야 다시 바꿀 수 있어요.")
        return value

    def validate_team_code(self, value):
        if value and value not in TEAM_CODES:
            raise serializers.ValidationError("응원팀을 확인해 주세요.")
        return value

    def validate_avatar(self, value):
        if not value:
            return ""
        if len(value) >= 600000 or not value.startswith("data:image/jpeg;base64,"):
            raise serializers.ValidationError("JPEG 프로필 사진을 다시 선택해 주세요.")
        try:
            raw = base64.b64decode(value.partition(",")[2], validate=True)
            image = Image.open(BytesIO(raw))
            if image.format != "JPEG" or image.width < 64 or image.height < 64 or image.width > 10000 or image.height > 10000 or image.width * image.height > 40000000:
                raise ValueError
            image.load()
        except (ValueError, UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise serializers.ValidationError("읽을 수 없는 JPEG 프로필 사진이에요.")
        return value

    def validate_notifications(self, value):
        return _boolean_settings(value, DEFAULT_NOTIFICATIONS)

    def validate_visibility(self, value):
        return _boolean_settings(value, DEFAULT_VISIBILITY)

    def update(self, instance, validated_data):
        if "nickname" in validated_data and validated_data["nickname"] != instance.nickname:
            from django.utils import timezone
            instance.nickname_changed_at = timezone.now()
        return super().update(instance, validated_data)

    def to_representation(self, instance):
        data = super().to_representation(instance)
        data["notifications"] = {**DEFAULT_NOTIFICATIONS, **(instance.notifications or {})}
        data["visibility"] = {**DEFAULT_VISIBILITY, **(instance.visibility or {})}
        return data


class MemberUserSerializer(MemberUserUpdateSerializer):
    notifications = NotificationSettingsField(read_only=True)
    visibility = VisibilitySettingsField(read_only=True)

    class Meta(MemberUserUpdateSerializer.Meta):
        read_only_fields = MemberUserUpdateSerializer.Meta.fields

@extend_schema_serializer(component_name="PublicMember")
class PublicMemberSerializer(serializers.ModelSerializer):
    """
    다른 회원의 공개 활동 페이지에서 사용할 최소 회원 정보입니다.

    - 개인정보 노출을 막기 위해 필요한 값만 반환합니다.
    - visibility 전체 객체는 반환하지 않습니다.
    """
    nickname = serializers.SerializerMethodField()
    activityVisible = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = ("id", "nickname", "activityVisible")
        read_only_fields = fields

    def get_nickname(self, obj):
        # 기존 커뮤니티 작성자 표시 규칙과 동일하게 처리
        return obj.nickname or obj.username

    def get_activityVisible(self, obj):
        # visibility.posts가 없으면 비공개로 처리
        return bool((obj.visibility or {}).get("posts", False))
    
class PasswordUpdateRequestSerializer(serializers.Serializer):
    current_password = serializers.CharField(write_only=True, required=False, max_length=128, trim_whitespace=False)
    new_password = serializers.CharField(write_only=True, required=False, max_length=128, trim_whitespace=False)
    new_password_confirm = serializers.CharField(write_only=True, required=False, max_length=128, trim_whitespace=False)
    uid = serializers.CharField(write_only=True, required=False)
    token = serializers.CharField(write_only=True, required=False)
    old_password = serializers.CharField(write_only=True, required=False, max_length=128, trim_whitespace=False)
    password = serializers.CharField(write_only=True, required=False, max_length=128, trim_whitespace=False)
    re_password = serializers.CharField(write_only=True, required=False, max_length=128, trim_whitespace=False)


class SendEmailSerializer(serializers.ModelSerializer):
    """
        이메일을 검증합니다.
    """
    email = serializers.EmailField()

    class Meta:
        model = User
        fields = ["email"]

    def is_valid(self, *, raise_exception=False):
        return super().is_valid(raise_exception=raise_exception)


class UsernameRequestResponseSerializer(serializers.Serializer):
    ok = serializers.BooleanField(read_only=True)


class EmailChangeRequestResponseSerializer(serializers.Serializer):
    request_id = serializers.UUIDField(read_only=True)


class EmailVerificationRequestSerializer(serializers.Serializer):
    request_id = serializers.UUIDField(write_only=True)
    code = serializers.RegexField(r"^\d{6}$", write_only=True)


class EmailVerificationResponseSerializer(serializers.Serializer):
    verified = serializers.BooleanField(read_only=True)
    email = serializers.EmailField(read_only=True)


class ResetPasswordSerializer(PasswordValidationMixin, serializers.ModelSerializer):
    """
        비밀번호 검증 로직
        1. 엑세스 토큰이 있으면 기존 비밀번호와 바꿀 비밀번호 검증 비밀번호를 받아서 검증한다.
        2. 엑세스 토큰이 없으면 parm 에서 토큰을 찾는다. 즉 이메일로 전송된 url을 통해서 비밀번호를 바꾸는 것이다.
    """
    old_password = serializers.CharField(write_only=True, required=False)
    password = serializers.CharField(write_only=True)
    re_password = serializers.CharField(write_only=True)
    token = serializers.CharField(write_only=True, required=False)

    class Meta:
        model = User
        fields = ["old_password", "password", "re_password", "token"]
        extra_kwargs = {"password": {"write_only": True}}

    def validate(self, attrs):
        # 새 비밀번호 일치 검증
        if attrs.get("password") != attrs.get("re_password"):
            raise serializers.ValidationError({"re_password": "비밀번호가 서로 다릅니다."})

        request = self.context.get("request")
        user = getattr(request, "user", None)

        if user and user.is_authenticated:
            old_password = attrs.get("old_password")
            if not old_password:
                raise serializers.ValidationError(
                    {"old_password": "기존 비밀번호를 입력해 주세요."}
                )

            if not user.check_password(old_password):
                raise serializers.ValidationError(
                    {"old_password": "기존 비밀번호가 일치하지 않습니다."}
                )

            attrs.pop("token", None)
            return attrs

        reset_token = self._get_reset_token()
        if not reset_token:
            raise serializers.ValidationError({"token": "비밀번호 재설정 토큰이 필요합니다."})

        attrs["token"] = reset_token
        return attrs

    def _get_reset_token(self):
        request = self.context.get("request")
        if not request:
            return None

        token = request.query_params.get("token") or request.query_params.get("parm")
        if token:
            return token

        view_kwargs = getattr(self.context.get("view"), "kwargs", {})
        return view_kwargs.get("token") or view_kwargs.get("parm")
