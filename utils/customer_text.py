"""Remove internal citation labels only; preserve instructions and device codes."""
import re


_INTERNAL_CITATION = re.compile(
    r"[（(\[【]\s*(?:参考|参见|来源[：:]?\s*)?"
    r"(?:知识片段|知识库片段|资料片段|参考资料|知识库依据)\s*"
    r"[：:#]?\s*\d+(?:\s*[-—–]\s*\d+)?"
    r"(?:\s*[,，、;；]\s*\d+(?:\s*[-—–]\s*\d+)?)*\s*[）)\]】]"
)


def customer_text(text: str) -> str:
    return _INTERNAL_CITATION.sub("", text).strip()
