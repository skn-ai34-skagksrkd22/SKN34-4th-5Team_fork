import uuid

from django.conf import settings
from django.db import models
from django.db.models import Q
from pgvector.django import VectorField, HnswIndex

class Document(models.Model):
    title = models.CharField(max_length=255)
    source = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)


class DocumentChunk(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")
    content = models.TextField()
    chunk_index = models.IntegerField()
    metadata = models.JSONField(default=dict)
    embedding = VectorField(dimensions=1536)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            HnswIndex(
                name="chunk_embedding_hnsw",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            ),
        ]


# 채팅방 테이블
class ChatSession(models.Model):
    """
    NOTE: 9월 25일 비회원 로직추가 
    1. User에 Null, Black을 추가함
    2. Guest을 추가
    3. 제약조건: 둘중하나만 반드시 존재해야함.  
    """
    id = models.UUIDField(
        primary_key=True,default=uuid.uuid4,editable=False
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="chat_sessions",
        null=True,
        blank=True
    )
    guest = models.UUIDField(
        null= True,
        blank=True,
        db_index=True
    )
    title = models.CharField(max_length=255, blank=True, default="메세지 제목")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    class Meta:
        constraints = [
            models.CheckConstraint(
                # 제약조건 회원(User) or 비회원(guest) 중 하나는 반드시 존재야한다. 라는 제약조건
                condition=( 
                    Q(user__isnull=False, guest__isnull=True)
                    | Q(user__isnull=True, guest__isnull=False)
                ),
                name="chat_session_has_one_owner",
            )
        ]


class ChatThreadDeletion(models.Model):
    """삭제된 ChatSession 의 checkpoint 삭제 outbox. 세션 삭제와 같은 트랜잭션에 쓰고, checkpoint 삭제 성공 뒤 지운다.

    FK 가 아니다: 세션 행은 이미 없다. 남아 있는 행 = 아직 지우지 못한 thread (llm.service.chat_thread.purge_deleted_threads).
    """
    thread_id = models.UUIDField(primary_key=True)
    token = models.UUIDField(default=uuid.uuid4)  # 예약마다 새 값. drain 은 읽은 token 일 때만 행을 지운다
    created_at = models.DateTimeField(auto_now_add=True)
