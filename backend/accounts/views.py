from collections.abc import Mapping
from datetime import timedelta
import hashlib
import secrets
from types import SimpleNamespace

from django.contrib.auth import get_user_model
from django.contrib.auth.hashers import check_password, make_password
from django.core.cache import cache
from django.core.exceptions import ValidationError as DjangoValidationError
from django.core.mail import send_mail
from rest_framework.decorators import api_view, permission_classes
from rest_framework.exceptions import NotAuthenticated, Throttled, ValidationError
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from drf_spectacular.utils import extend_schema
from rest_framework_simplejwt.views import TokenBlacklistView, TokenObtainPairView, TokenRefreshView
from django.db import IntegrityError, transaction
from django.utils import timezone
from .auth_service import AuthService
from .models import EmailChangeChallenge
from .serializers import (
    EmailChangeRequestResponseSerializer,
    EmailVerificationRequestSerializer,
    EmailVerificationResponseSerializer,
    LogoutRequestSerializer,
    MemberUserSerializer,
    MemberUserUpdateSerializer,
    PasswordUpdateRequestSerializer,
    PublicMemberSerializer,  
    ResetPasswordSerializer,
    SendEmailSerializer,
    SignInRequestSerializer,
    SignupSerializer,
    TokenPairSerializer,
    TokenRefreshRequestSerializer,
    TokenRefreshResponseSerializer,
    UsernameRequestResponseSerializer,
)

User = get_user_model()


@extend_schema(request=SignInRequestSerializer, responses=TokenPairSerializer, auth=[])
class SignInView(TokenObtainPairView):
    pass


@extend_schema(request=TokenRefreshRequestSerializer, responses=TokenRefreshResponseSerializer, auth=[])
class RefreshView(TokenRefreshView):
    pass


@extend_schema(request=LogoutRequestSerializer, responses={200: {"type": "object", "properties": {}, "additionalProperties": False}})
class LogoutView(TokenBlacklistView):
    pass


@extend_schema(request=SignupSerializer, responses={201: None}, auth=[])
@api_view(["POST"])
def signup(request):
    """
        회원가입하는 함수입니다.
        Url : /api/v1/auth/signup/
        Args:
            - username
            - email, first_name, birth_date, gender
            - password
            - re_password
        Return:
            - HTTP_201_CREATED
    """
    serializer = SignupSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    try:
        with transaction.atomic():
            serializer.save()
    except IntegrityError:
        raise ValidationError({'username': '이미 사용 중인 아이디입니다.'})

    return Response(
        status=status.HTTP_201_CREATED,
    )

@extend_schema(request=SendEmailSerializer, responses={200: None}, auth=[])
@api_view(['POST'])
def change_password(request):
    """
        사용자 비밀번호 재설정 url 을 이메일로 전송합니다.
        Url: /api/v1/auth/password/request
        Args:
            - email
        Return:
            - HTTP_200_OK
        1. 검증
        2. 비밀번호 재설정용 토큰 발급
        3. 이메일 전송
    """
    # 1. 검증
    serializer = SendEmailSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)

    # 2. 재설정 토큰 발급 및 이메일 전송
    email = serializer.validated_data['email']
    _throttle('password', email)
    AuthService.send_reset_email(email)
    return Response(
        status=status.HTTP_200_OK
    )

@extend_schema(request=PasswordUpdateRequestSerializer, responses={200: None})
@api_view(['POST'])
@transaction.atomic
def set_password(request):
    """
        사용자 비밀번호를 변경합니다.
        Url: POST /auth/password
        Query (이메일 재설정): uid, token
        Headers (로그인 상태): Authorization: Bearer <access_token>
        Args:
            - current_password: 기존 비밀번호 (로그인 상태에서 필수)
            - new_password: 새 비밀번호
            - new_password_confirm: 새 비밀번호 확인
        기존 old_password, password, re_password 필드명도 지원합니다.
        Return: HTTP_200_OK
    """
    # 1. 이메일 재설정 토큰 또는 로그인 사용자 확인
    if not isinstance(request.data, Mapping):
        raise ValidationError({'detail': '객체 형태의 요청 본문이 필요합니다.'})
    uid = request.data.get('uid')
    token = request.data.get('token')
    is_reset = uid is not None or token is not None
    user = AuthService.get_password_user(request.user, uid=uid, token=token)

    # 2. 요청 필드명을 기존 serializer에 맞추고 비밀번호 검증
    data = request.data.copy()
    data.pop('uid', None)
    for source, target in (
        ('current_password', 'old_password'),
        ('new_password', 'password'),
        ('new_password_confirm', 're_password'),
    ):
        if source in data:
            data[target] = data[source]

    # 이메일 토큰 방식은 로그인 여부와 관계없이 기존 비밀번호를 요구하지 않습니다.
    serializer = ResetPasswordSerializer(
        data=data,
        context={'request': SimpleNamespace(
            user=None if is_reset else user,
            query_params={'token': token} if token else {},
        )},
    )
    for field in ('old_password', 'password', 're_password'):
        serializer.fields[field].trim_whitespace = False
    serializer.is_valid(raise_exception=True)
    # 3. 비밀번호 정책 검증 및 저장
    AuthService.set_password(user, serializer.validated_data['password'])
    return Response(status=status.HTTP_200_OK)


@extend_schema(methods=['GET'], responses=MemberUserSerializer, auth=[{"jwtAuth": []}])
@extend_schema(methods=['PATCH'], request=MemberUserUpdateSerializer, responses=MemberUserSerializer, auth=[{"jwtAuth": []}])
@api_view(['GET', 'PATCH'])
@transaction.atomic
def get_user(request):
    """
        로그인한 사용자 정보를 조회합니다.
        Url: GET /api/v1/auth/user
        Headers: Authorization: Bearer <access_token>
        Return:
            - HTTP_200_OK
            - HTTP_401_UNAUTHORIZED (토큰 미전달 또는 JWT 인증 실패)
    """
    user = request.user
    if not user.is_authenticated:
        return Response(
            {'detail': '인증이 필요합니다.'},
            status=status.HTTP_401_UNAUTHORIZED,
        )

    if request.method == 'PATCH':
        allowed = {'first_name', 'birth_date', 'gender', 'nickname', 'team_code', 'avatar', 'notifications', 'visibility'}
        if not isinstance(request.data, Mapping) or not set(request.data).issubset(allowed):
            raise ValidationError({'detail': '변경할 수 없는 회원 정보가 포함됐습니다.'})
        user = User.objects.select_for_update().get(pk=user.pk, is_active=True)
        serializer = MemberUserUpdateSerializer(user, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
    return Response(MemberUserSerializer(user).data, status=status.HTTP_200_OK)

@extend_schema(
    responses={
        200: PublicMemberSerializer,
        404: {"type": "object"},
    },
    auth=[{"jwtAuth": []}],
)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_public_user(request, member_id):
    """
    다른 회원의 공개 활동 페이지에 필요한 최소 회원 정보를 조회합니다.

    Url:
        GET /api/v1/auth/users/{member_id}/public/

    Return:
        - id
        - nickname
        - activityVisible
    """

    # 탈퇴/비활성 회원은 존재하지 않는 회원과 동일하게 처리
    user = User.objects.filter(
        pk=member_id,
        is_active=True,
    ).first()

    if not user:
        return Response(
            {"detail": "사용자를 찾을 수 없습니다."},
            status=status.HTTP_404_NOT_FOUND,
        )

    return Response(PublicMemberSerializer(user).data)

def _throttle(kind, value, seconds=60):
    key = f"auth:{kind}:{hashlib.sha256(value.casefold().encode()).hexdigest()}"
    if not cache.add(key, True, timeout=seconds):
        raise Throttled(wait=seconds)


@extend_schema(request=SendEmailSerializer, responses=UsernameRequestResponseSerializer, auth=[])
@api_view(['POST'])
def request_username(request):
    serializer = SendEmailSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    email = serializer.validated_data['email']
    _throttle('username', email)
    usernames = list(User.objects.filter(email__iexact=email, is_active=True).order_by('pk').values_list('username', flat=True))
    if usernames:
        send_mail(
            subject='[KBO ROUTE] 아이디 안내',
            message='가입한 아이디입니다.\n\n' + '\n'.join(usernames),
            from_email=None,
            recipient_list=[email],
        )
    return Response({'ok': True})


@extend_schema(request=SendEmailSerializer, responses=EmailChangeRequestResponseSerializer, auth=[{"jwtAuth": []}])
@api_view(['POST'])
@transaction.atomic
def request_email_change(request):
    if not request.user.is_authenticated:
        raise NotAuthenticated('인증이 필요합니다.')
    serializer = SendEmailSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    email = serializer.validated_data['email']
    if User.objects.filter(email__iexact=email).exclude(pk=request.user.pk).exists():
        raise ValidationError({'email': '이미 사용 중인 이메일입니다.'})
    user = User.objects.select_for_update().get(pk=request.user.pk)
    latest = EmailChangeChallenge.objects.filter(user=user).order_by('-created_at').first()
    if latest and latest.created_at > timezone.now() - timedelta(seconds=60):
        raise Throttled(wait=60)
    EmailChangeChallenge.objects.filter(user=user, used_at__isnull=True).update(used_at=timezone.now())
    code = f"{secrets.randbelow(1_000_000):06d}"
    challenge = EmailChangeChallenge.objects.create(
        user=user,
        email=email,
        code_hash=make_password(code),
        expires_at=timezone.now() + timedelta(minutes=10),
    )
    send_mail(
        subject='[KBO ROUTE] 이메일 변경 인증 코드',
        message=f'인증 코드는 {code} 입니다. 10분 안에 입력해 주세요.',
        from_email=None,
        recipient_list=[email],
    )
    return Response({'request_id': str(challenge.pk)})


@extend_schema(request=EmailVerificationRequestSerializer, responses=EmailVerificationResponseSerializer, auth=[{"jwtAuth": []}])
@api_view(['POST'])
def verify_email_change(request):
    if not request.user.is_authenticated:
        raise NotAuthenticated('인증이 필요합니다.')
    if not isinstance(request.data, Mapping):
        raise ValidationError({'detail': '객체 형태의 요청 본문이 필요합니다.'})
    request_id, code = request.data.get('request_id'), request.data.get('code')
    if not isinstance(request_id, str) or not isinstance(code, str) or len(code) != 6 or not code.isdigit():
        raise ValidationError({'code': '6자리 인증 코드를 입력해 주세요.'})
    with transaction.atomic():
        user = User.objects.select_for_update().get(pk=request.user.pk)
        try:
            challenge = EmailChangeChallenge.objects.select_for_update().get(pk=request_id, user=user)
        except (ValueError, DjangoValidationError, EmailChangeChallenge.DoesNotExist):
            return Response({'code': ['인증 요청을 확인할 수 없습니다.']}, status=status.HTTP_400_BAD_REQUEST)
        now = timezone.now()
        if challenge.used_at or challenge.expires_at <= now or challenge.attempts >= 5:
            return Response({'code': ['만료되었거나 사용할 수 없는 인증 코드입니다.']}, status=status.HTTP_400_BAD_REQUEST)
        if not check_password(code, challenge.code_hash):
            challenge.attempts += 1
            challenge.save(update_fields=['attempts'])
            return Response({'code': ['인증 코드가 일치하지 않습니다.']}, status=status.HTTP_400_BAD_REQUEST)
        if User.objects.filter(email__iexact=challenge.email).exclude(pk=user.pk).exists():
            return Response({'email': ['이미 사용 중인 이메일입니다.']}, status=status.HTTP_400_BAD_REQUEST)
        user.email = challenge.email
        user.save(update_fields=['email'])
        challenge.used_at = now
        challenge.save(update_fields=['used_at'])
    return Response({'verified': True, 'email': user.email})
