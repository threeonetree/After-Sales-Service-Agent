"""Short customer messages with redacted developer tracebacks and matching IDs."""
import traceback
import uuid

from utils.logger_handler import get_logger
from utils.model_errors import _exception_text, redact_diagnostics


def record_app_error(error: Exception) -> str:
    reference = uuid.uuid4().hex[:12]
    # No locals, request payloads, or session state are collected here.
    trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    get_logger("support").error("error_id=%s\n%s", reference, redact_diagnostics(trace))
    detail = _exception_text(error).lower()
    if "allocationquota.freetieronly" in detail or "free tier only" in detail:
        message = "服务免费额度已用完，暂时无法处理新问题。"
    elif any(word in detail for word in ("timeout", "timed out", "connection", "network")):
        message = "连接暂时中断，请稍后重试。"
    else:
        message = "暂时无法完成处理，请稍后重试或联系维护人员。"
    return f"{message}（问题编号：{reference}）"
