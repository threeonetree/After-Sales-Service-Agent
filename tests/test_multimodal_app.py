"""Run the real Streamlit page, replacing only the network-facing Agent."""
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import sys

from PIL import Image
import pytest
from streamlit.testing.v1 import AppTest

from services.visual_support import KnowledgeSource


@pytest.fixture
def app_runtime(monkeypatch):
    module = ModuleType("agent.react_agent")
    agent = Mock()
    agent.execute_with_trace.return_value = SimpleNamespace(
        response="先检查滚刷。",
        sources=[KnowledgeSource(1, "手册.pdf · 第 1 页", "清理滚刷。")],
        observation={"findings": ["图1：滚刷缠绕"], "visible_text": [], "uncertainties": []},
    )
    module.ReactAgent = Mock(return_value=agent)
    monkeypatch.setitem(sys.modules, "agent.react_agent", module)
    path = Path(__file__).resolve().parents[1] / "app.py"
    return AppTest.from_file(str(path), default_timeout=15), agent


def test_text_chat_sources_rerun_and_new_conversation(app_runtime):
    app, agent = app_runtime
    app.run()
    assert not app.exception
    assert app.chat_input[0].proto.accept_file != 0
    app.chat_input[0].set_value("你好").run()
    assert not app.exception
    assert [message["content"] for message in app.session_state["message"]] == ["你好", "先检查滚刷。"]
    assert len(app.expander) == 0
    assert not app.sidebar.selectbox
    assert agent.execute_with_trace.call_args.kwargs["images"] == []
    app.run()
    assert agent.execute_with_trace.call_count == 1  # UI reruns do not spend quota.
    app.button(key="new_chat").click().run()
    assert not app.session_state["message"]
    agent.reset_conversation.assert_called_once()


def test_image_only_submission_shows_preview_without_internal_evidence(app_runtime, monkeypatch):
    import streamlit as st
    from streamlit.elements.widgets.chat import ChatInputValue
    image = BytesIO()
    Image.new("RGB", (100, 80), "red").save(image, "PNG")
    monkeypatch.setattr(st, "chat_input", lambda *args, **kwargs: ChatInputValue(
        text="", files=[image], _include_files=True,
    ))
    app, agent = app_runtime
    app.run()
    assert not app.exception
    assert len(agent.execute_with_trace.call_args.kwargs["images"]) == 1
    assert app.session_state["message"][0]["images"]
    assert len(app.image) == 1
    assert app.image[0].captions == ["图 1"]
    assert "sources" not in app.session_state["message"][1]
    assert "observation" not in app.session_state["message"][1]
    assert not app.expander


def test_bad_image_never_calls_agent(app_runtime, monkeypatch):
    import streamlit as st
    from streamlit.elements.widgets.chat import ChatInputValue
    monkeypatch.setattr(st, "chat_input", lambda *args, **kwargs: ChatInputValue(
        text="看图", files=[BytesIO(b"broken")], _include_files=True,
    ))
    app, agent = app_runtime
    app.run()
    assert not app.exception
    assert "损坏" in app.error[0].value
    agent.execute_with_trace.assert_not_called()
    assert not app.session_state["message"]


def test_quota_failure_is_displayed_once_without_automatic_retry(app_runtime):
    app, agent = app_runtime
    agent.execute_with_trace.side_effect = RuntimeError("403 AllocationQuota.FreeTierOnly")
    app.run().chat_input[0].set_value("你好").run()
    assert "免费额度已用完" in app.error[0].value
    app.run()
    assert agent.execute_with_trace.call_count == 1
    assert len(app.error) == 1


def test_user_switch_clears_history(app_runtime):
    app, agent = app_runtime
    app.run().chat_input[0].set_value("你好").run()
    app.selectbox[0].select("1002").run()
    assert not app.exception
    assert not app.session_state["message"]
    assert agent.reset_conversation.call_args.args[1] == "1001"


def test_examples_prefill_without_sending_and_welcome_disappears(app_runtime):
    app, agent = app_runtime
    app.run().button(key="example_0").click().run()
    assert not app.exception
    agent.execute_with_trace.assert_not_called()
    assert app.chat_input[0].value == "扫地机器人清扫时经常漏扫怎么办？"
    app.chat_input[0].set_value("怎么清理滚刷？").run()
    assert not app.exception
    assert not any("有什么可以帮你" in text.value for text in app.markdown)


def test_internal_citations_and_raw_errors_never_reach_customer(app_runtime):
    app, agent = app_runtime
    agent.execute_with_trace.return_value.response = "先清理滚刷（参考知识片段1-12，3-25）。"
    app.run().chat_input[0].set_value("滚刷缠绕").run()
    assert app.session_state["message"][-1]["content"] == "先清理滚刷。"
    agent.execute_with_trace.side_effect = RuntimeError("private-provider-detail sk-secret")
    app.chat_input[0].set_value("再看一下").run()
    assert "问题编号" in app.error[0].value
    assert "private-provider-detail" not in app.error[0].value
    assert "sk-secret" not in app.error[0].value


def test_new_user_request_uses_new_identity_and_thread(app_runtime):
    app, agent = app_runtime
    app.run().chat_input[0].set_value("你好").run()
    first_thread = agent.execute_with_trace.call_args.kwargs["thread_id"]
    app.selectbox[0].select("1002").run()
    app.chat_input[0].set_value("本月记录").run()
    assert agent.execute_with_trace.call_args.kwargs["context"] == {"user_id": "1002"}
    assert agent.execute_with_trace.call_args.kwargs["thread_id"] != first_thread
