"""Shared pytest setup: automated regression tests must stay offline."""

import socket

import pytest


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def fail_connect(*args, **kwargs):
        raise AssertionError("Tests must not access the network; use a fixture or mock.")

    monkeypatch.setattr(socket.socket, "connect", fail_connect)


@pytest.fixture(autouse=True)
def isolate_model_environment(monkeypatch):
    """Replace only service settings, preserving Windows home and system variables."""
    import os
    import dotenv

    for name in tuple(os.environ):
        upper = name.upper()
        if upper.startswith(("DASHSCOPE_", "OPENAI_", "LANGSMITH_", "LANGCHAIN_")) or upper.endswith("_PROXY"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-test-key")
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)


SUITES = {
    "test_image_input": "图片校验与预处理",
    "test_visual_support": "图片理解与知识检索",
    "test_multimodal_app": "客服页面交互",
    "test_personal_data_route": "用户记录与月份校验",
    "test_react_agent_routing": "Agent 路由与用户隔离",
    "test_report_prompt_state": "报告生成约束",
    "test_rag_sources": "知识检索来源",
    "test_vector_store_batching": "知识库入库",
    "test_model_factory": "模型配置",
    "test_model_errors": "模型错误诊断",
    "test_customer_output": "客户答复与错误日志",
    "test_run_tests": "测试报告生成",
}


def pytest_runtest_setup(item):
    try:
        import allure
    except ImportError:
        return  # Plain pytest remains usable without the reporting plugin.
    allure.dynamic.parent_suite("售后助手离线回归")
    allure.dynamic.suite(SUITES.get(item.path.stem, item.path.stem.removeprefix("test_")))
    allure.dynamic.title(item.name)
