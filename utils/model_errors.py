"""Safe, user-facing diagnostics for Bailian/DashScope failures."""

from __future__ import annotations

import os


class ModelConfigurationError(RuntimeError):
    """Raised when the local model client is not configured."""


def require_dashscope_api_key() -> str:
    """Return the configured key without ever logging or displaying it."""
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise ModelConfigurationError(
            "DASHSCOPE_API_KEY is not set. Configure it in the Windows environment "
            "or in a local .env file."
        )
    return api_key


def _exception_text(error: BaseException) -> str:
    """Collect useful public error fields from a chained exception."""
    parts: list[str] = []
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        parts.append(str(current))
        for attribute in ("code", "status_code", "message"):
            value = getattr(current, attribute, None)
            if value is not None:
                parts.append(str(value))
        current = current.__cause__ or current.__context__
    return " ".join(parts)


def user_facing_model_error(error: BaseException) -> str:
    """Translate common failures without leaking credentials or stack traces."""
    details = _exception_text(error)
    lowered = details.lower()

    if "allocationquota.freetieronly" in lowered or "free tier only" in lowered:
        return (
            "百炼免费额度已用完，平台已按“额度用完即停”拒绝本次请求。"
            "应用没有切换到付费模型，也不会继续产生模型费用。"
        )
    if "dashscope_api_key" in lowered or "noapikey" in lowered or "no api key" in lowered:
        return (
            "未读取到 DASHSCOPE_API_KEY。请在 Windows 环境变量或项目本地 .env "
            "文件中配置后，重新打开终端。"
        )
    if "invalidapikey" in lowered or "invalid api key" in lowered or "unauthorized" in lowered:
        return "百炼 API Key 无效或无权访问当前业务空间，请检查 Key 与模型所在空间。"
    if isinstance(error, ModuleNotFoundError):
        return "项目依赖尚未安装完整，请先运行：python -m pip install -r requirements.txt"
    if any(token in lowered for token in ("connection", "timeout", "timed out", "network")):
        return "暂时无法连接百炼服务，请检查网络后重试。"

    return f"模型服务调用失败：{details or type(error).__name__}"
