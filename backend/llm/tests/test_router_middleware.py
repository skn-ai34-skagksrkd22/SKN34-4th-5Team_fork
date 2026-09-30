"""chain.jev_router / _route: 거절은 SCOPE_MESSAGE 후 END, 승인은 SIMPLE/COMPLEX 분기, 분류기 예외는 fail closed."""
from unittest import TestCase
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from llm.v2.agent import chain
from llm.v2.middleware.jev_guidelines import SCOPE_MESSAGE


def decision(allowed=True, complexity="SIMPLE"):
    return {"allowed": allowed, "complexity": complexity, "capabilities": []}


class JevRouterTest(TestCase):
    def test_classifies_latest_question_with_history_and_context(self):
        history = [HumanMessage("이전 질문"), AIMessage("이전 답변")]
        state = {"messages": [*history, HumanMessage("잠실 주차")], "context": {"stadium": "잠실"}}
        with patch.object(chain.classifier, "classify", return_value=decision()) as classify:
            self.assertEqual(chain.jev_router(state), {"decision": decision()})
        classify.assert_called_once_with("잠실 주차", history, {"stadium": "잠실"})

    def test_refusal_appends_scope_message_and_ends(self):
        with patch.object(chain.classifier, "classify", return_value=decision(allowed=False)):
            out = chain.jev_router({"messages": [HumanMessage("오늘 주식 뭐 사?")]})
        self.assertEqual(out["messages"][0].content, SCOPE_MESSAGE)
        self.assertEqual(chain._route(out), END)

    def test_routes_by_complexity(self):
        self.assertEqual(chain._route({"decision": decision()}), "simple_agent")
        self.assertEqual(chain._route({"decision": decision(complexity="COMPLEX")}), "orchestrator")

    def test_classifier_exception_fails_closed(self):
        with patch.object(chain.classifier, "classify", side_effect=RuntimeError("down")):
            with self.assertRaises(RuntimeError):
                chain.jev_router({"messages": [HumanMessage("q")]})
