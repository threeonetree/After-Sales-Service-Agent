"""Deterministic routing for personal usage records and reports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from services.user_data_service import MONTH_FIELD, USER_ID_FIELD


class UserDataReader(Protocol):
    def get_usage_record(self, user_id: str, month: str) -> dict | None: ...

    def available_months(self, user_id: str) -> list[str]: ...


@dataclass(frozen=True)
class PersonalDataRoute:
    """The pre-model decision for a personal-data request."""

    handled: bool = False
    response: str | None = None
    report_month: str | None = None
    pending_intent: str | None = None


_FULL_MONTH_PATTERN = re.compile(
    r"(?<!\d)(\d{4})\s*(?:[-/.年])\s*(\d+)(?:\s*月)?"
)
_ABBREVIATED_RANGE_PATTERN = re.compile(
    r"(?:至|到|和|、)\s*(?:20\d{2}\s*年\s*)?\d{1,2}\s*月"
)
_CURRENT_MONTH_TERMS = ("本月", "这个月", "当前月", "当月")
_PREVIOUS_MONTH_TERMS = ("上月", "上个月")
_RECORD_TERMS = ("使用记录", "清扫记录", "清洁记录", "运行记录", "使用情况")
_QUERY_TERMS = ("查询", "查看", "看看", "查一下", "显示", "调取", "有没有", "有无")
_REPORT_ACTION_TERMS = ("生成", "制作", "写", "做", "出一份", "查看", "查询", "我的")
_REPORT_DATA_TERMS = ("使用报告", "清扫报告", "清洁报告", "运行报告", "数据报告")
_MUTATION_TERMS = ("删除", "清空", "修改", "更改", "添加", "导入", "上传")
_GENERAL_QUESTION_TERMS = ("是什么", "什么意思", "怎么", "如何", "为什么", "在哪")
_NEGATED_REPORT = re.compile(r"(?:不要|不用|无需|不需要)[^，。；,;]*报告")
_MONTH_REPLY = re.compile(
    r"^(?:那就|那|就|查|选|选择)?\s*(?:\d{4}\s*[-/.年]\s*\d+\s*月?|"
    r"本月|这个月|上月|上个月)\s*(?:的|吧|呢)?[。！？!?]?$"
)
_MULTI_PERIOD_TERMS = ("半年", "季度", "全年", "每月", "所有月份", "全部记录", "历史记录", "最近几个月")


def route_personal_data_request(
    query: str,
    user_id: str,
    user_data: UserDataReader,
    *,
    today: date | None = None,
    pending_intent: str | None = None,
) -> PersonalDataRoute:
    """Resolve structured personal-data requests before the model runs.

    Record lookups are rendered directly. A report reaches the Agent only when
    it names exactly one month and that month has data for the selected user.
    """

    normalized_query = query.strip()
    if not normalized_query or not user_id:
        return PersonalDataRoute()
    if pending_intent in {"record", "report"} and _MONTH_REPLY.fullmatch(normalized_query):
        prefix, suffix = (
            ("生成", "使用报告") if pending_intent == "report" else ("查询", "使用记录")
        )
        normalized_query = prefix + normalized_query + suffix

    is_report_request = _is_report_request(normalized_query)
    is_record_request = _is_record_request(normalized_query)
    if not is_report_request and not is_record_request:
        return PersonalDataRoute()
    intent = "report" if is_report_request else "record"

    month, month_error = _extract_month(normalized_query, today or date.today())
    if month_error:
        return PersonalDataRoute(handled=True, response=month_error, pending_intent=intent)

    try:
        if is_report_request:
            if month is None:
                return PersonalDataRoute(
                    handled=True,
                    response=_month_required_response(user_data.available_months(user_id)),
                    pending_intent=intent,
                )
            if user_data.get_usage_record(user_id, month) is None:
                return PersonalDataRoute(
                    handled=True,
                    response=_no_record_response(
                        month,
                        user_data.available_months(user_id),
                        report_request=True,
                    ),
                    pending_intent=intent,
                )
            return PersonalDataRoute(report_month=month)

        requested_month = month or (today or date.today()).strftime("%Y-%m")
        record = user_data.get_usage_record(user_id, requested_month)
        if record is None:
            return PersonalDataRoute(
                handled=True,
                response=_no_record_response(
                    requested_month,
                    user_data.available_months(user_id),
                    report_request=False,
                ),
                pending_intent=intent,
            )
        return PersonalDataRoute(
            handled=True,
            response=_record_response(user_id, requested_month, record),
        )
    except (FileNotFoundError, ValueError):
        return PersonalDataRoute(
            handled=True,
            response="使用记录数据暂时不可用，请检查本地数据文件后重试。",
        )


def _is_report_request(query: str) -> bool:
    if (
        "报告" not in query
        or _NEGATED_REPORT.search(query)
        or any(term in query for term in _GENERAL_QUESTION_TERMS + _MUTATION_TERMS)
    ):
        return False
    return any(term in query for term in _REPORT_ACTION_TERMS + _REPORT_DATA_TERMS)


def _is_record_request(query: str) -> bool:
    if any(term in query for term in _MUTATION_TERMS + _GENERAL_QUESTION_TERMS):
        return False
    if not any(term in query for term in _RECORD_TERMS):
        return False
    return (
        any(term in query for term in _QUERY_TERMS + ("想看", "帮我查", "给我看"))
        or query.strip(" 。！？!?") in _RECORD_TERMS
        or bool(_FULL_MONTH_PATTERN.search(query))
        or any(term in query for term in _CURRENT_MONTH_TERMS + _PREVIOUS_MONTH_TERMS)
    )


def _extract_month(query: str, current_date: date) -> tuple[str | None, str | None]:
    matches = list(_FULL_MONTH_PATTERN.finditer(query))
    if any(
        not 1 <= int(match.group(2)) <= 12 or not 1 <= int(match.group(1)) <= 9999
        for match in matches
    ):
        return None, "月份格式无效，请使用 YYYY-MM，例如 2025-12。"

    months = {
        f"{match.group(1)}-{int(match.group(2)):02d}"
        for match in matches
    }
    if any(term in query for term in _CURRENT_MONTH_TERMS):
        months.add(current_date.strftime("%Y-%m"))
    if any(term in query for term in _PREVIOUS_MONTH_TERMS):
        year, month = current_date.year, current_date.month - 1
        if month == 0:
            year, month = year - 1, 12
        months.add(f"{year}-{month:02d}")
    remaining = _FULL_MONTH_PATTERN.sub("", query)
    remaining = re.sub(r"用户\s*(?:ID|id)?\s*[:：]?\s*\d+", "", remaining)
    if (
        len(months) > 1
        or _ABBREVIATED_RANGE_PATTERN.search(query)
        or any(term in query for term in _MULTI_PERIOD_TERMS)
        or re.search(r"(?:近|前)\s*[\d一二两三四五六七八九十]+\s*个?月", query)
        or re.search(r"\d+\s*[-~～至到]\s*\d+\s*月", remaining)
        or re.search(r"[-~～]\s*\d+\s*月", remaining)
    ):
        return None, "当前仅支持按单个月份查询或生成报告，请指定一个 YYYY-MM 月份。"
    # Never silently replace an unrecognised/partial date with the current month.
    if re.search(r"\d|[一二三四五六七八九十]+月|去年|今年|前年|下月|下个月", remaining):
        return None, "请明确完整的年份和月份，例如 2025-12；也可以使用“本月”或“上月”。"
    if months:
        return next(iter(months)), None
    return None, None


def _record_response(user_id: str, month: str, record: dict) -> str:
    lines = [
        f"### {_display_month(month)}使用记录",
        "",
        f"- **用户ID**：{user_id}",
    ]
    for field, raw_value in record.items():
        if field in {USER_ID_FIELD, MONTH_FIELD} or raw_value is None:
            continue
        value_lines = [part.strip() for part in str(raw_value).splitlines() if part.strip()]
        if not value_lines:
            continue
        if len(value_lines) == 1:
            lines.append(f"- **{field}**：{value_lines[0]}")
        else:
            lines.append(f"- **{field}**：")
            lines.extend(f"  - {part}" for part in value_lines)
    return "\n".join(lines)


def _month_required_response(available_months: list[str]) -> str:
    available = _display_available_months(available_months)
    if available is None:
        return "当前用户暂无可用于生成报告的使用记录。"
    example_month = sorted(available_months)[-1]
    return (
        "生成使用报告前需要指定一个月份。"
        f"当前可用月份：{available}。"
        f"例如：\"生成{_display_month(example_month)}使用报告\"。"
    )


def _no_record_response(
    month: str,
    available_months: list[str],
    *,
    report_request: bool,
) -> str:
    if report_request:
        message = f"{_display_month(month)}没有使用记录，无法生成该月报告。"
    else:
        message = f"{_display_month(month)}暂无使用记录。"
    available = _display_available_months(available_months)
    if available is None:
        return f"{message} 当前用户也没有其他可查询的历史记录。"
    return (
        f"{message}\n\n当前可查询月份：{available}。"
        "请指定其中一个月份后再试。"
    )


def _display_month(month: str) -> str:
    year, month_number = month.split("-", maxsplit=1)
    return f"{year}年{int(month_number)}月"


def _display_available_months(months: list[str]) -> str | None:
    normalized = sorted(set(months))
    if not normalized:
        return None
    if _is_contiguous(normalized):
        if len(normalized) == 1:
            return _display_month(normalized[0])
        return f"{_display_month(normalized[0])}至{_display_month(normalized[-1])}"
    return "、".join(_display_month(month) for month in normalized)


def _is_contiguous(months: list[str]) -> bool:
    indexes = []
    for month in months:
        year, month_number = month.split("-", maxsplit=1)
        indexes.append(int(year) * 12 + int(month_number))
    return all(right - left == 1 for left, right in zip(indexes, indexes[1:]))
