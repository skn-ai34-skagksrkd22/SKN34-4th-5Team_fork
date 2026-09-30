"""0007 migration: Django state 에서만 모델을 빼고 legacy 테이블/행과 다른 데이터는 그대로 보존."""

from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import TransactionTestCase

BEFORE = [("llm", "0006_chattoolcall_and_more")]
AFTER = [("llm", "0007_remove_chat_message_tables")]


def _tables():
    with connection.cursor() as cursor:
        return set(connection.introspection.table_names(cursor))


def _count(table):
    with connection.cursor() as cursor:
        cursor.execute(f'SELECT count(*) FROM "{table}"')
        return cursor.fetchone()[0]


class ChatTableRemovalMigrationTest(TransactionTestCase):
    def _migrate(self, target):
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(target)
        return executor.loader.project_state(target).apps

    def tearDown(self):
        self._migrate(executor_leaf())

    def test_legacy_tables_and_rows_preserved_and_session_still_deletable(self):
        apps = self._migrate(BEFORE)
        ChatSession = apps.get_model("llm", "ChatSession")
        ChatMessage = apps.get_model("llm", "ChatMessage")
        ChatToolCall = apps.get_model("llm", "ChatToolCall")
        Document = apps.get_model("llm", "Document")

        guest_session = ChatSession.objects.create(guest="11111111-1111-1111-1111-111111111111", title="보존될 방")
        message = ChatMessage.objects.create(session=guest_session, sequence_no=1, role="user", message="옛 대화")
        ChatToolCall.objects.create(message=message, tool_name="search_players", arguments={"q": 1})
        Document.objects.create(title="doc", source="src")

        before = _tables()
        counts = {t: _count(t) for t in before if t != "django_migrations"}

        apps = self._migrate(AFTER)
        self.assertEqual(_tables(), before)  # 물리 테이블은 하나도 안 지운다
        for table, count in counts.items():
            self.assertEqual(_count(table), count, table)
        with connection.cursor() as cursor:
            cursor.execute('SELECT message FROM "llm_chatmessage"')
            self.assertEqual(cursor.fetchall(), [("옛 대화",)])
        with self.assertRaises(LookupError):  # Django state 에서는 빠졌다
            apps.get_model("llm", "ChatMessage")

        # 옛 행이 남아 있어도 세션은 지워진다(FK 제거). 옛 행은 고아로 보존된다.
        apps.get_model("llm", "ChatSession").objects.filter(id=guest_session.id).delete()
        self.assertEqual(_count("llm_chatsession"), 0)
        self.assertEqual((_count("llm_chatmessage"), _count("llm_chattoolcall")), (1, 1))
        with connection.cursor() as cursor:  # flush 가 모르는 legacy 행은 직접 치운다
            cursor.execute('DELETE FROM "llm_chattoolcall"; DELETE FROM "llm_chatmessage"')


def executor_leaf():
    executor = MigrationExecutor(connection)
    return executor.loader.graph.leaf_nodes()
