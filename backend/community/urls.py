from django.urls import path

from .admin_views import AdminPostDetail, AdminPostList, AdminReportAction, AdminReportList
from .drafts import CommunityDraftDetailView, CommunityDraftListCreateView
from .images import CommunityImageDetailView, CommunityImageUploadView
from .interactions import CommentDetailView, CommentListCreateView, MemberCommentListView, ReportCreateView, VoteView
from .predictions import prediction_game_detail, prediction_game_list, prediction_game_vote
from .publishing import CommunityDraftPublishView
from .views import CommunityPostDetailView, CommunityPostListCreateView

urlpatterns = [
    path("admin/posts/", AdminPostList.as_view(), name="community-admin-posts"),
    path("admin/posts/<str:source_id>/", AdminPostDetail.as_view(), name="community-admin-post"),
    path("admin/reports/", AdminReportList.as_view(), name="community-admin-reports"),
    path("admin/reports/<int:report_id>/action/", AdminReportAction.as_view(), name="community-admin-report-action"),
    path("drafts/", CommunityDraftListCreateView.as_view(), name="community-draft-list"),
    path("drafts/<uuid:draft_id>/", CommunityDraftDetailView.as_view(), name="community-draft-detail"),
    path("images/", CommunityImageUploadView.as_view(), name="community-image-upload"),
    path("images/<uuid:image_id>/", CommunityImageDetailView.as_view(), name="community-image-detail"),
    path("posts/", CommunityPostListCreateView.as_view(), name="community-post-list"),
    path("posts/<str:source_id>/", CommunityPostDetailView.as_view(), name="community-post-detail"),
    path("posts/<str:source_id>/comments/", CommentListCreateView.as_view()),
    path("comments/", MemberCommentListView.as_view(), name="community-member-comments",), # 특정 회원이 작성한 댓글 목록
    path("comments/<int:comment_id>/", CommentDetailView.as_view()),
    path("posts/<str:source_id>/vote/", VoteView.as_view()),
    path("posts/<str:source_id>/reports/", ReportCreateView.as_view()),
    path("drafts/<uuid:draft_id>/publish/", CommunityDraftPublishView.as_view(), name="community-draft-publish"),
    path("predictions/games/", prediction_game_list),
    path("predictions/games/<str:game_id>/", prediction_game_detail),
    path("predictions/games/<str:game_id>/vote/", prediction_game_vote),
]
