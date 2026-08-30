"""Select user-visible output from LangGraph message states."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
    return "".join(text_parts).strip()


def visible_assistant_text(message: Any) -> str | None:
    """Return only a final assistant answer, never user or tool messages."""
    if not isinstance(message, AIMessage):
        return None
    if getattr(message, "tool_calls", None):
        return None
    return _content_text(message.content) or None
