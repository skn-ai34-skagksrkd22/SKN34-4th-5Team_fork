from django.core.management.base import BaseCommand

from llm.service.chat_thread import ChatThread, convert_legacy_threads, erase_legacy_rows, purge_deleted_threads


class Command(BaseCommand):
    help = ("채팅 checkpoint 테이블(PostgresSaver.setup)을 생성/마이그레이션하고, 삭제된 세션의 남은 checkpoint "
            "(ChatThreadDeletion outbox)를 지운다. 배포 전 대화(llm_chatmessage)를 checkpoint 로 옮기고, 이미 지워진 세션의 "
            "옛 행을 지운다. 반복 실행해도 안전하다.")

    def handle(self, *args, **options):
        ChatThread.setup()
        purge_deleted_threads()  # 실패·누락된 세션 삭제 후속 처리 재시도
        erase_legacy_rows()  # 이 수정 전에 지워진 세션이 남긴 고아 옛 행
        converted = convert_legacy_threads()  # 옛 행은 롤백 대비로 남긴다(세션 삭제 때 지운다)
        self.stdout.write(f"legacy chat sessions converted: {converted}")
        self.stdout.write(self.style.SUCCESS("chat checkpoint tables ready"))
