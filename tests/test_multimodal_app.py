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
    assert len(app.expander) == 2
    assert agent.execute_with_trace.call_args.kwargs["images"] == []
    app.run()
    assert agent.execute_with_trace.call_count == 1  # UI reruns do not spend quota.
    app.button[0].click().run()
    assert not app.session_state["message"]
    agent.reset_conversation.assert_called_once()


def test_image_only_submission_renders_preview_and_sources(app_runtime, monkeypatch):
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
    assert app.session_state["message"][1]["sources"][0]["title"].startswith("手册")


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
