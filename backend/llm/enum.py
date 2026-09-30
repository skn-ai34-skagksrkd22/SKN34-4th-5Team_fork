from django.db import models
from enum import Enum

class ChatRole(models.TextChoices):
    USER = "user", "사용자"
    ASSISTANT = "assistant", "AI"


# MessageStatus/ToolStatus 는 옛 메시지 테이블 시절 값(stopped/started)이다. 외부 import 호환용으로 그대로 두고,
# checkpoint 대화 흐름은 아래 TurnStatus/PublicToolStatus 를 쓴다.
class MessageStatus(models.TextChoices):
    PENDING = "pending", "진행 중"
    COMPLETED = "completed", "완료"
    FAILED = "failed", "실패"
    STOPPED = "stopped", "중단"


class ToolStatus(models.TextChoices):
    STARTED = "started", "시작"
    COMPLETED = "completed", "완료"
    FAILED = "failed", "실패"


# ── checkpoint 대화·공개 채팅 계약 ─────────────────────────────────────────────────
# 값이 곧 저장(checkpoint)·응답·SSE 문자열이다. 경계에서는 .value(순수 str)를 내보낸다.
class TurnStatus(models.TextChoices):
    """checkpoint turns[*].status 와 공개 대화 항목 status."""
    PENDING = "pending", "진행 중"
    COMPLETED = "completed", "완료"
    FAILED = "failed", "실패"
    CANCELLED = "cancelled", "취소"


class PublicToolStatus(models.TextChoices):
    """공개 tool 이벤트·대화 항목의 도구 status."""
    RUNNING = "running", "실행 중"
    COMPLETED = "completed", "완료"
    FAILED = "failed", "실패"


class PublicChatEvent(models.TextChoices):
    """SSE event 이름."""
    DELTA = "delta", "답변 조각"
    TOOL = "tool", "도구"
    DONE = "done", "완료"
    ERROR = "error", "오류"


class ChainVersion(models.TextChoices):
    """LLM 체인 버전 (URL·env LLM_CHAIN_VERSION)."""
    V1 = "v1", "v1"
    V2 = "v2", "v2"


class ContextIntent(models.TextChoices):
    """요청 context.intent."""
    ROUTE = "route", "길찾기"
    BASEBALL = "baseball", "야구"
    STADIUM = "stadium", "구장"


class AgentType(str, Enum):
    BASEBALL = "baseball"
    STADIUM = "stadium"
    TRAVEL = "travel"
    COURSE = "course"
    COMMUNITY = "community"
