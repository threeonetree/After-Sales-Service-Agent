import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.report_prompt_state import report_mode_active


class ReportPromptStateTests(unittest.TestCase):
    def test_report_context_in_current_turn_enables_report_prompt(self):
        messages = [
            HumanMessage(content="生成2025年12月使用报告"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "fill_context_for_report",
                        "args": {},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="Report context is ready.",
                tool_call_id="call-1",
                name="fill_context_for_report",
            ),
        ]

        self.assertTrue(report_mode_active(messages))

    def test_new_user_turn_clears_previous_report_mode(self):
        messages = [
            HumanMessage(content="生成2025年12月使用报告"),
            ToolMessage(
                content="Report context is ready.",
                tool_call_id="call-1",
                name="fill_context_for_report",
            ),
            AIMessage(content="报告内容"),
            HumanMessage(content="扫地机器人漏扫怎么办？"),
        ]

        self.assertFalse(report_mode_active(messages))

    def test_non_report_turn_uses_main_prompt(self):
        self.assertFalse(
            report_mode_active(
                [
                    HumanMessage(content="查询天气"),
                    ToolMessage(
                        content="{}",
                        tool_call_id="call-2",
                        name="get_weather",
                    ),
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()
