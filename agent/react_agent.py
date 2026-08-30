from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.messages import SystemMessage
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts, load_report_prompts
from utils.logger_handler import logger
from agent.tools.agent_tools import (rag_summarize,get_weather,get_user_id,get_user_location,
                                get_current_month,fetch_external_data,fill_context_for_report)

# 报告模式标记（替代原 middleware 的 dynamic_prompt 切换）
_report_mode = False


@dataclass
class AgentExecution:
    """A complete agent run retained for local testing and later Ragas evaluation."""

    response: str
    messages: list[Any]
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, str]]

def _state_modifier(state):
    """动态切换系统提示词：检测 fill_context_for_report 调用后切换为报告提示词"""
    global _report_mode
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, "name") and last_msg.name == "fill_context_for_report":
            _report_mode = True
        elif hasattr(last_msg, "type") and last_msg.type == "ai":
            _report_mode = False
    system_text = load_report_prompts() if _report_mode else load_system_prompts()
    return [SystemMessage(content=system_text)] + state["messages"]


class ReactAgent:
    def __init__(self):
        self.agent = create_react_agent(
            model=chat_model,
            tools=[rag_summarize,get_weather,get_user_id,get_user_location,
                   get_current_month,fetch_external_data,fill_context_for_report],
            prompt=_state_modifier,
            checkpointer=MemorySaver(),
        )

    def execute_stream(self,query:str,thread_id:str="default",context:dict=None):
        global _report_mode
        _report_mode = False
        config = {"configurable":{"thread_id":thread_id}}
        input_dict = {
            "messages":[
                {"role":"user","content":query},
            ]
        }

        for chunk in self.agent.stream(input_dict,config=config,stream_mode="values"):
            latest_message = chunk["messages"][-1]
            if hasattr(latest_message, "content") and latest_message.content:
                yield latest_message.content.strip() + '\n'

    def execute_with_trace(
        self, query: str, thread_id: str = "evaluation", context: Optional[dict] = None
    ) -> AgentExecution:
        """Run once and preserve final output plus raw tool-call history.

        This method does not change the Streamlit streaming behavior. It is the
        stable integration point for regression tests and Ragas agent metrics.
        """
        global _report_mode
        _report_mode = False
        config = {"configurable": {"thread_id": thread_id}}
        result = self.agent.invoke({"messages": [{"role": "user", "content": query}]}, config=config)
        messages = result["messages"]
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, str]] = []

        for message in messages:
            for tool_call in getattr(message, "tool_calls", None) or []:
                tool_calls.append(
                    {"name": tool_call.get("name", ""), "args": tool_call.get("args", {})}
                )
            if getattr(message, "type", None) == "tool":
                tool_results.append(
                    {"name": getattr(message, "name", ""), "content": str(message.content)}
                )

        response = ""
        for message in reversed(messages):
            if getattr(message, "type", None) == "ai" and getattr(message, "content", None):
                response = str(message.content)
                break
        return AgentExecution(response, messages, tool_calls, tool_results)

if __name__ == '__main__':
    agent = ReactAgent()
    for chunk in agent.execute_stream("我想查看自己的使用记录"):
        print(chunk,end="",flush=True)
