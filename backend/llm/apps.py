from django.apps import AppConfig


def _enqueue_thread_deletion(sender, instance, **kwargs):
    """ChatSession 이 어떤 경로로 지워지든(세션 API, 회원 삭제 연쇄, admin) 대화 checkpoint 삭제를 예약한다.

    1단계: 세션 행 삭제와 같은 트랜잭션에 outbox 행을 쓴다. ORM 삭제가 롤백되면 outbox 행도 사라지고
    checkpoint 는 건드리지 않는다. 배포 전 옛 대화 행(llm_chatmessage/llm_chattoolcall)도 같은 트랜잭션에서 지운다.
    2단계: commit 뒤 그 thread 의 checkpoint 를 지우고 outbox 행을 치운다.
    checkpoint 삭제가 실패하거나 on_commit 이 실행되지 못해도(프로세스 종료) 행이 남아
    setup_chat_checkpoints(시작 경로)의 drain 이 재시도한다.
    """
    from django.db import transaction

    from llm.service.chat_thread import erase_legacy_rows, purge_deleted_threads, reserve_thread_deletion

    reserve_thread_deletion(instance.pk)
    erase_legacy_rows(instance.pk)
    thread_id = instance.pk
    transaction.on_commit(lambda: purge_deleted_threads([thread_id]), robust=True)  # 이미 commit 된 삭제를 500 으로 만들지 않는다


class LlmConfig(AppConfig):
    name = 'llm'

    def ready(self):
        from django.db.models.signals import post_delete
        post_delete.connect(_enqueue_thread_deletion, sender="llm.ChatSession",
                            dispatch_uid="llm.chat_session_checkpoint_cleanup")
