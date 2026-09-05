"""Pure helpers for selecting the report prompt in one conversation turn."""

from __future__ import annotations

from typing import Any, Iterable


def report_mode_active(messages: Iterable[Any]) -> bool:
    """Return whether the current turn has entered report-generation mode.

    Only messages after the latest user message count, so one user's report
    state cannot leak into a later turn or another conversation.
    """

    for message in reversed(list(messages)):
        if getattr(message, "type", None) == "human":
            return False
        if (
            getattr(message, "type", None) == "tool"
            and getattr(message, "name", None) == "fill_context_for_report"
        ):
            return True
    return False
