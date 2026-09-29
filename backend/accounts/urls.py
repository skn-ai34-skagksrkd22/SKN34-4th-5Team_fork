from .admin_views import MemberList, MemberRole
from django.urls import path
from .views import LogoutView, RefreshView, SignInView, change_password, get_public_user, get_user, request_email_change, request_username, set_password, signup, verify_email_change

"""
    Django 직접 호출과 Nginx 경유 모두 /api/v1/auth/입니다.
    
    POST /api/v1/auth/signin
    POST /api/v1/auth/token/refresh/
    POST /api/v1/auth/signup/
    POST /api/v1/auth/password/request
    POST /api/v1/auth/password
    GET  /api/v1/auth/user
    POST /api/v1/auth/logout
"""
urlpatterns = [
    path("admin/members/", MemberList.as_view(), name="admin_members"),
    path("admin/members/<int:pk>/role/", MemberRole.as_view(), name="admin_member_role"),
    path("signin", SignInView.as_view(), name="login"),
    path("token/refresh/", RefreshView.as_view(), name="token_refresh"),
    path("signup/", signup, name="signup"),
    path("password/request", change_password, name="password_request"),
    path("password", set_password, name="password_reset"),
    path("user", get_user, name="auth_user"),
    path("users/<int:member_id>/public/", get_public_user, name="auth_public_user",),  # 다른 회원의 공개 활동 정보 조회
    path("username/request", request_username, name="username_request"),
    path("email/request", request_email_change, name="email_request"),
    path("email/verify", verify_email_change, name="email_verify"),
    path("logout", LogoutView.as_view(), name="auth_logout"),
]
