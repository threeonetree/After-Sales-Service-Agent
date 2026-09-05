"""Pure routing tests: fixtures provide real CSV data and a fixed date."""

import csv
from datetime import date
from unittest.mock import Mock

import pytest

from agent.personal_data_route import route_personal_data_request
from services.user_data_service import UserDataService


@pytest.fixture
def user_data(tmp_path):
    """Each test gets its own tiny data file; project data is never modified."""
    records_path = tmp_path / "records.csv"
    with records_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["用户ID", "时间", "特征", "清洁效率"])
        writer.writeheader()
        for month in range(1, 13):
            writer.writerow({
                "用户ID": "1001", "时间": f"2025-{month:02d}",
                "特征": "清扫3次", "清洁效率": "覆盖率：90%\n遗漏区域：厨房",
            })
        writer.writerow({
            "用户ID": "1002", "时间": "2026-08",
            "特征": "清扫8次", "清洁效率": "覆盖率：80%",
        })
    return UserDataService(tmp_path / "profiles.json", records_path)


def test_current_month_without_record_stops_before_model(user_data):
    # Arrange: fix the date so this test still works next month/year.
    today = date(2026, 8, 30)
    # Act: reproduce the exact question that previously generated a false report.
    route = route_personal_data_request(
        "帮我查询本月的使用记录", "1001", user_data, today=today,
    )
    # Assert: require an explicit no-data answer, not a historical report.
    assert route.handled
    assert "2026年8月暂无使用记录" in route.response
    assert "2025年1月至2025年12月" in route.response
    assert "报告" not in route.response
    assert route.pending_intent == "record"


@pytest.mark.parametrize("query", [
    "查看2025年12月的使用记录", "查询2025-12使用记录",
    "2025/12使用记录", "查询2025.12使用情况",
    "查询用户1001在2025年12月的使用记录",
    "查询2025年12月的使用记录，不要生成报告",
])
def test_explicit_month_record_is_rendered_directly(user_data, query):
    route = route_personal_data_request(query, "1001", user_data, today=date(2026, 8, 30))
    assert route.handled
    assert "2025年12月使用记录" in route.response
    assert "清扫3次" in route.response
    assert "覆盖率：90%" in route.response
    assert "厨房" in route.response
    assert route.report_month is None


@pytest.mark.parametrize(("query", "today", "expected_month"), [
    ("查询上个月的使用记录", date(2026, 1, 5), "2025年12月"),
    ("查询本月的使用记录", date(2026, 9, 5), "2026年9月"),
    ("我想看自己的使用记录", date(2026, 8, 30), "2026年8月"),
])
def test_relative_or_default_month_is_resolved(user_data, query, today, expected_month):
    route = route_personal_data_request(query, "1001", user_data, today=today)
    assert route.handled
    assert expected_month in route.response


@pytest.mark.parametrize(("query", "expected"), [
    ("帮我生成使用报告", "需要指定一个月份"),
    ("生成2026年8月的使用报告", "无法生成该月报告"),
    ("生成本月使用报告", "无法生成该月报告"),
])
def test_unavailable_or_unspecified_report_stops(user_data, query, expected):
    route = route_personal_data_request(query, "1001", user_data, today=date(2026, 8, 30))
    assert route.handled
    assert expected in route.response
    assert route.report_month is None


def test_existing_report_has_one_approved_month(user_data):
    route = route_personal_data_request("生成2025年12月使用报告", "1001", user_data)
    assert not route.handled
    assert route.report_month == "2025-12"


@pytest.mark.parametrize("query", [
    "生成2025年1月至3月的使用报告", "查询2025-01和2025-12使用记录",
    "查询本月和上月使用记录", "查询2025年1-3月使用记录",
    "生成最近3个月使用报告", "生成半年使用报告",
])
def test_multiple_months_are_not_silently_truncated(query):
    service = Mock()
    route = route_personal_data_request(query, "1001", service, today=date(2026, 8, 30))
    assert route.handled
    assert "单个月份" in route.response
    service.get_usage_record.assert_not_called()


@pytest.mark.parametrize("query", [
    "查看2025年13月使用记录", "查看2025-123使用记录",
    "查看12月使用记录", "查看去年十二月使用记录", "查询2025年使用记录",
])
def test_invalid_or_ambiguous_date_does_not_default_to_current(query):
    service = Mock()
    route = route_personal_data_request(query, "1001", service, today=date(2026, 8, 30))
    assert route.handled
    assert "2025-12" in route.response
    service.get_usage_record.assert_not_called()


@pytest.mark.parametrize("query", [
    "如何删除使用记录？", "怎么查看使用报告？", "使用报告是什么？",
    "扫地机器人经常漏扫怎么办？", "不要生成报告",
])
def test_general_questions_continue_to_agent(user_data, query):
    route = route_personal_data_request(query, "1001", user_data)
    assert not route.handled
    assert route.report_month is None


def test_selected_user_is_respected(user_data):
    route = route_personal_data_request(
        "查询本月的使用记录", "1002", user_data, today=date(2026, 8, 30),
    )
    assert "清扫8次" in route.response
    assert "清扫3次" not in route.response


def test_gap_in_available_months_is_not_shown_as_continuous():
    service = Mock()
    service.get_usage_record.return_value = None
    service.available_months.return_value = ["2025-01", "2025-03"]
    route = route_personal_data_request("查询本月使用记录", "1001", service)
    assert "2025年1月、2025年3月" in route.response


def test_missing_data_file_returns_data_error_not_no_records(tmp_path):
    service = UserDataService(tmp_path / "profiles.json", tmp_path / "missing.csv")
    route = route_personal_data_request("查询本月使用记录", "1001", service)
    assert route.handled
    assert "数据暂时不可用" in route.response
    assert "暂无使用记录" not in route.response


@pytest.mark.parametrize(("intent", "handled"), [("record", True), ("report", False)])
def test_month_only_reply_resumes_pending_request(user_data, intent, handled):
    route = route_personal_data_request(
        "那就2025-12吧", "1001", user_data, pending_intent=intent,
    )
    assert route.handled is handled
    if intent == "report":
        assert route.report_month == "2025-12"
    else:
        assert "2025年12月使用记录" in route.response
