import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.message_output import visible_assistant_text


class MessageOutputTests(unittest.TestCase):
    def test_hides_user_and_tool_messages(self):
        self.assertIsNone(visible_assistant_text(HumanMessage(content="用户问题")))
        self.assertIsNone(
            visible_assistant_text(
                ToolMessage(content='{"found": false}', tool_call_id="call-1")
            )
        )

    def test_hides_intermediate_assistant_tool_call(self):
        message = AIMessage(
            content="我先查询记录",
            tool_calls=[
                {
                    "name": "fetch_external_data",
                    "args": {"user_id": "1001", "month": "2026-08"},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        )
        self.assertIsNone(visible_assistant_text(message))

    def test_returns_final_assistant_text(self):
        self.assertEqual(
            visible_assistant_text(AIMessage(content="本月暂无使用记录。")),
            "本月暂无使用记录。",
        )

    def test_supports_text_content_blocks(self):
        message = AIMessage(
            content=[
                {"type": "text", "text": "第一段"},
                {"type": "text", "text": "第二段"},
            ]
        )
        self.assertEqual(visible_assistant_text(message), "第一段第二段")


if __name__ == "__main__":
    unittest.main()
