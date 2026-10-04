import re
from unittest.mock import Mock

import pytest

from utils.app_errors import record_app_error
from utils.customer_text import customer_text


@pytest.mark.parametrize("label", [
    "（参考知识片段1-12，3-25）", "【知识片段2】", "[参考资料1]", "(参见知识库片段 2、3)",
])
def test_internal_labels_removed_without_changing_steps(label):
    assert customer_text(f"先清理滚刷{label}。") == "先清理滚刷。"


def test_device_codes_dates_and_normal_numbers_preserved():
    text = "1. 型号X1（E42），2025-12记录。参阅手册[1]。\n2. 请勿拆电池。"
    assert customer_text(text) == text


def test_app_error_has_matching_id_and_redacted_traceback(monkeypatch):
    logger = Mock()
    monkeypatch.setattr("utils.app_errors.get_logger", lambda name: logger)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "private-configured-key")
    try:
        raise ValueError("private-configured-key sk-abcdef data:image/png;base64,QUJDRA==")
    except ValueError as error:
        message = record_app_error(error)
    reference = re.search(r"问题编号：(\w+)", message).group(1)
    _, logged_id, trace = logger.error.call_args.args
    assert logged_id == reference
    assert "test_customer_output.py" in trace
    assert "ValueError" in trace and "Traceback" in trace
    for secret in ("private-configured-key", "sk-abcdef", "QUJDRA=="):
        assert secret not in message and secret not in trace
    assert "ValueError" not in message
