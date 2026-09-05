"""Agent entry points with deterministic personal-data routing."""

from dataclasses import dataclass, field
import json
from typing import Any
from uuid import uuid4

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.message_output import visible_assistant_text
from agent.personal_data_route import route_personal_data_request
from agent.report_prompt_state import report_mode_active
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts, load_report_prompts
from services.image_input import PreparedImage, prepare_question
from services.visual_support import KnowledgeSource, VisualContext, VisualSupportService
from agent.tools.agent_tools import (
    current_user_id,
    execution_context,
    fetch_external_data,
    fill_context_for_report,
    get_user_id,
    get_user_location,
    get_weather,
    rag_summarize,
    rag,
    requested_usage_month,
    user_data,
)


@dataclass
class AgentExecution:
    """Current-run output and actual model tool calls for evaluation."""

    response: str
    messages: list[Any]
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, str]]
    sources: list[KnowledgeSource] = field(default_factory=list)
    observation: dict | None = None


def _state_modifier(state):
    messages = state.get("messages", [])
    month = requested_usage_month.get()
    system_text = (
        load_report_prompts()
        if month and report_mode_active(messages)
        else load_system_prompts()
    )
    if month:
        system_text += (
            f"\n本轮已由程序验证：用户ID={current_user_id.get()}，报告月份={month}。"
            "将用户的相对月份或简短月份回复按此月份理解。只查询并报告该月。"
        )
    else:
        system_text += (
            "\n本轮未授权生成使用报告。不得调用报告工具或生成个人使用报告；"
            "需要使用记录时，请用户明确提出查询或报告请求并指定月份。"
        )
    return [SystemMessage(content=system_text)] + messages


class ReactAgent:
    def __init__(self, model=None, visual_service=None):
        selected_model = model if model is not None else chat_model
        self.agent = create_react_agent(
            model=selected_model,
            tools=[
                rag_summarize, get_weather, get_user_id, get_user_location,
                fetch_external_data, fill_context_for_report,
            ],
            prompt=_state_modifier,
            checkpointer=MemorySaver(),
        )
        self._pending_requests: dict[str, str] = {}
        self._visual_context: dict[str, VisualContext] = {}
        self.visual_service = visual_service or VisualSupportService(
            selected_model, rag.retriever_docs
        )

    def execute_stream(
        self, query: str, thread_id: str = "default", context: dict | None = None,
        images: list[PreparedImage] | None = None,
    ):
        """Expose only the completed answer; never stream tool or user messages."""
        execution = self.execute_with_trace(query, thread_id, context, images)
        yield execution.response + "\n"

    def execute_with_trace(
        self, query: str, thread_id: str = "evaluation", context: dict | None = None,
        images: list[PreparedImage] | None = None,
    ) -> AgentExecution:
        """Use the same routing as the UI and trace only the current run."""
        selected_user = str((context or {}).get("user_id") or current_user_id.get())
        # User identity is part of the checkpoint key, even outside Streamlit.
        conversation_key = json.dumps([selected_user, thread_id], ensure_ascii=False)
        config = {"configurable": {"thread_id": conversation_key}}
        query = prepare_question(query, bool(images))
        if images:
            # Never fall back to the previous picture if this new upload fails.
            # OCR never goes through the personal-data router or account tools.
            self._visual_context.pop(conversation_key, None)
            self._pending_requests.pop(conversation_key, None)
            result = self.visual_service.analyze(query, images)
            return self._remember_visual(config, conversation_key, query, result)
        route = route_personal_data_request(
            query,
            selected_user,
            user_data,
            pending_intent=self._pending_requests.get(conversation_key),
        )
        if route.pending_intent:
            self._pending_requests[conversation_key] = route.pending_intent
        else:
            self._pending_requests.pop(conversation_key, None)

        human_message = HumanMessage(content=query, id=str(uuid4()))
        if route.handled:
            self._visual_context.pop(conversation_key, None)
            response = route.response or ""
            messages = [human_message, AIMessage(content=response, id=str(uuid4()))]
            self._remember(config, messages)
            return AgentExecution(response, messages, [], [])

        if route.report_month:
            self._visual_context.pop(conversation_key, None)
        elif conversation_key in self._visual_context:
            result = self.visual_service.follow_up(query, self._visual_context[conversation_key])
            return self._remember_visual(config, conversation_key, query, result)

        # A report must not read another month's records from conversation memory.
        run_config = (
            {"configurable": {"thread_id": f"{conversation_key}:report:{uuid4()}"}}
            if route.report_month else config
        )
        with execution_context(selected_user, route.report_month):
            result = self.agent.invoke({"messages": [human_message]}, config=run_config)
        messages = result["messages"]
        start = next(i for i, message in enumerate(messages) if message.id == human_message.id)
        messages = messages[start:]
        response = next(
            (text for message in reversed(messages)
             if (text := visible_assistant_text(message))),
            "",
        )
        if route.report_month:
            self._remember(
                config,
                [human_message, AIMessage(content=response, id=str(uuid4()))],
            )
        tool_calls = [
            {"name": call.get("name", ""), "args": call.get("args", {})}
            for message in messages
            for call in (getattr(message, "tool_calls", None) or [])
        ]
        tool_results = [
            {"name": getattr(message, "name", ""), "content": str(message.content)}
            for message in messages if getattr(message, "type", None) == "tool"
        ]
        return AgentExecution(response, messages, tool_calls, tool_results)

    def _remember_visual(self, config, conversation_key, query, result):
        observation = result.context.observation.model_dump()
        # Only the final dialogue enters LangGraph; the observation is retained
        # separately as bounded text for the dedicated visual follow-up route.
        human = HumanMessage(content=query, id=str(uuid4()))
        assistant = AIMessage(content=result.response, id=str(uuid4()))
        self._remember(config, [human, assistant])
        self._visual_context[conversation_key] = result.context
        return AgentExecution(
            result.response, [human, assistant], [], [], result.sources, observation
        )

    def reset_conversation(self, thread_id: str, user_id: str) -> None:
        """Drop a closed conversation's state when starting fresh or switching user."""
        key = json.dumps([str(user_id), thread_id], ensure_ascii=False)
        self._pending_requests.pop(key, None)
        self._visual_context.pop(key, None)
        self.agent.checkpointer.delete_thread(key)

    def _remember(self, config: dict, messages: list[Any]) -> None:
        """Save direct answers for follow-ups without invoking the model."""
        self.agent.update_state(config, {"messages": messages}, as_node="agent")


if __name__ == "__main__":
    agent = ReactAgent()
    for chunk in agent.execute_stream("我想查看自己的使用记录"):
        print(chunk, end="", flush=True)
