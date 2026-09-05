"""Exercise the real LangGraph entry points with a scripted, offline model."""

import importlib
import json
import sys
from types import ModuleType
from unittest.mock import Mock, patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class ScriptedChatModel(FakeMessagesListChatModel):
    seen: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, **kwargs)


@pytest.fixture
def runtime():
    # Mock only external services; use the real Agent, graph, tools and memory.
    factory = ModuleType("model.factory")
    factory.chat_model = None
    rag = ModuleType("rag.rag_service")
    rag.RagSummerizeService = Mock()
    with patch.dict(sys.modules, {"model.factory": factory, "rag.rag_service": rag}):
        sys.modules.pop("agent.react_agent", None)
        sys.modules.pop("agent.tools.agent_tools", None)
        module = importlib.import_module("agent.react_agent")
        tools = importlib.import_module("agent.tools.agent_tools")
        data = Mock()
        data.get_usage_record.side_effect = lambda user_id, month: (
            {"用户ID": user_id, "时间": month, "特征": f"{user_id}已清扫3次"}
            if month == "2025-12" else None
        )
        data.available_months.return_value = ["2025-11", "2025-12"]
        module.user_data = tools.user_data = data
        yield module, tools, data


def make_agent(runtime, responses):
    module, _, _ = runtime
    model = ScriptedChatModel(responses=responses)
    return module.ReactAgent(model=model), model


def tool_request(name, args=None, call_id="call-1"):
    return AIMessage(
        content="内部工具计划不应显示",
        tool_calls=[{"name": name, "args": args or {}, "id": call_id, "type": "tool_call"}],
    )


@pytest.mark.parametrize("entry", ["stream", "trace"])
def test_missing_record_does_not_invoke_model(runtime, entry):
    agent, model = make_agent(runtime, [AIMessage(content="不应执行")])
    query = "查询2026年8月使用记录"
    if entry == "stream":
        text = "".join(agent.execute_stream(query, context={"user_id": "1001"}))
    else:
        execution = agent.execute_with_trace(query, context={"user_id": "1001"})
        text = execution.response
        assert execution.tool_calls == []
    assert "2026年8月暂无使用记录" in text
    assert model.seen == []


def test_direct_answer_is_available_to_later_conversation(runtime):
    agent, model = make_agent(runtime, [AIMessage(content="说明")])
    agent.execute_with_trace("查询2025-12使用记录", thread_id="same")
    agent.execute_with_trace("解释一下刚才的记录", thread_id="same")
    history = model.seen[-1]
    assert any("已清扫3次" in str(message.content) for message in history)


def test_month_reply_resumes_record_lookup_without_model(runtime):
    agent, model = make_agent(runtime, [AIMessage(content="不应执行")])
    agent.execute_with_trace("查询2026-08使用记录", thread_id="same")
    execution = agent.execute_with_trace("2025-12", thread_id="same")
    assert "2025年12月使用记录" in execution.response
    assert model.seen == []


def test_report_is_scoped_and_tool_output_stays_internal(runtime):
    responses = [
        tool_request("fill_context_for_report"),
        tool_request("fetch_external_data", {"user_id": "1001", "month": "2025-11"}, "wrong"),
        tool_request("fetch_external_data", {"user_id": "1001", "month": "2025-12"}, "right"),
        AIMessage(content="### 2025年12月使用报告\n清扫3次。"),
    ]
    agent, model = make_agent(runtime, responses)
    # Seed earlier text and verify it cannot contaminate the scoped report.
    agent.execute_with_trace("查询2026-08使用记录", thread_id="same")
    text = "".join(agent.execute_stream("生成2025-12使用报告", thread_id="same"))
    assert text == "### 2025年12月使用报告\n清扫3次。\n"
    assert "2026年8月暂无使用记录" not in str(model.seen)
    tool_results = [message for message in model.seen[-1] if isinstance(message, ToolMessage)]
    wrong = json.loads(next(message.content for message in tool_results if message.tool_call_id == "wrong"))
    right = json.loads(next(message.content for message in tool_results if message.tool_call_id == "right"))
    assert wrong["found"] is False
    assert wrong["allowed_month"] == "2025-12"
    assert right["found"] is True
    assert runtime[1].requested_usage_month.get() is None
    # Only the report month was actually read, even when the model tried another.
    assert all(call.args[1] != "2025-11" for call in runtime[2].get_usage_record.call_args_list)


def test_month_reply_can_start_pending_report(runtime):
    agent, model = make_agent(runtime, [AIMessage(content="报告")])
    first = agent.execute_with_trace("生成使用报告", thread_id="same")
    assert "指定一个月份" in first.response
    assert not model.seen
    agent.execute_with_trace("那就2025-12吧", thread_id="same")
    assert "报告月份=2025-12" in model.seen[-1][0].content


def test_context_binds_selected_user_and_is_reset(runtime):
    agent, model = make_agent(runtime, [
        tool_request("get_user_id"),
        AIMessage(content="查询完成"),
    ])
    runtime[1].set_current_user_id("1001")
    execution = agent.execute_with_trace(
        "生成2025-12使用报告", context={"user_id": "1002"},
    )
    assert execution.tool_results[0]["content"] == "1002"
    assert runtime[1].current_user_id.get() == "1001"


def test_identity_and_pending_state_do_not_leak_between_users(runtime):
    agent, model = make_agent(runtime, [AIMessage(content="请说明问题")])
    agent.execute_with_trace("生成使用报告", thread_id="same", context={"user_id": "1001"})
    agent.execute_with_trace("2025-12", thread_id="same", context={"user_id": "1002"})
    assert "报告月份=" not in model.seen[-1][0].content
    humans = [m.content for m in model.seen[-1] if isinstance(m, HumanMessage)]
    assert humans == ["2025-12"]


def test_unapproved_report_tools_cannot_fetch_data(runtime):
    agent, _ = make_agent(runtime, [
        tool_request("fetch_external_data", {"user_id": "1001", "month": "2025-12"}),
        AIMessage(content="请指定月份"),
    ])
    execution = agent.execute_with_trace("你好")
    assert json.loads(execution.tool_results[0]["content"])["found"] is False
    runtime[2].get_usage_record.assert_not_called()


def test_model_cannot_change_selected_user(runtime):
    _, tools, data = runtime
    with tools.execution_context("1001", "2025-12"):
        result = tools.fetch_external_data.invoke({"user_id": "1002", "month": "2025-12"})
    assert json.loads(result)["found"] is False
    data.get_usage_record.assert_not_called()


def test_scope_is_reset_after_model_failure(runtime):
    agent, model = make_agent(runtime, [AIMessage(content="未执行")])
    with patch.object(type(model), "_generate", side_effect=RuntimeError("模型调用失败")):
        with pytest.raises(RuntimeError, match="模型调用失败"):
            agent.execute_with_trace("生成2025-12使用报告", context={"user_id": "1002"})
    assert runtime[1].requested_usage_month.get() is None
    assert runtime[1].current_user_id.get() == "1001"


def test_trace_contains_only_current_turn_tools(runtime):
    agent, _ = make_agent(runtime, [
        tool_request("get_user_id"), AIMessage(content="1001"), AIMessage(content="你好"),
    ])
    first = agent.execute_with_trace("你好，确认当前用户", thread_id="same")
    second = agent.execute_with_trace("你好", thread_id="same")
    assert len(first.tool_calls) == 1
    assert second.tool_calls == []
