from django.db import migrations


class Migration(migrations.Migration):
    """대화 원본은 LangGraph checkpoint(llm.service.chat.ChatThread)로 이동.

    Django state 에서만 모델을 뺀다. llm_chatmessage/llm_chattoolcall 테이블과 행은 이관·삭제 없이
    그대로 남는다(데이터 손실 방지). 다만 llm_chatmessage.session_id → llm_chatsession FK 는 지운다:
    CASCADE 는 Django 가 하던 일이라 모델이 빠지면 옛 행이 있는 세션을 지울 수 없게 되기 때문이다.
    그래서 세션을 지우면 옛 행은 고아로 남는다. 필요 없어지면 백업 뒤 두 테이블을 직접 DROP 한다.
    """

    dependencies = [
        ("llm", "0006_chattoolcall_and_more"),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.DeleteModel(name="ChatToolCall"),
                migrations.DeleteModel(name="ChatMessage"),
            ],
            database_operations=[
                migrations.RunSQL(
                    'ALTER TABLE "llm_chatmessage" DROP CONSTRAINT IF EXISTS '
                    '"llm_chatmessage_session_id_63b50114_fk_llm_chatsession_id"',
                    # 되돌릴 때 고아 행이 있으면 실패한다 -- 그 행을 정리한 뒤 다시 시도한다.
                    'ALTER TABLE "llm_chatmessage" ADD CONSTRAINT "llm_chatmessage_session_id_63b50114_fk_llm_chatsession_id"'
                    ' FOREIGN KEY ("session_id") REFERENCES "llm_chatsession" ("id") DEFERRABLE INITIALLY DEFERRED',
                ),
            ],
        ),
    ]
