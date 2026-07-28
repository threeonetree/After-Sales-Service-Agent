"""LangGraph tools for the robot after-sales agent."""

from __future__ import annotations

import json
from contextvars import ContextVar
from datetime import datetime

from langchain_core.tools import tool

from rag.rag_service import RagSummerizeService
from services.user_data_service import UserDataService
from services.weather_service import WeatherService, WeatherServiceError
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path


rag = RagSummerizeService()
current_user_id: ContextVar[str] = ContextVar("current_user_id", default="1001")
user_data = UserDataService(
    profiles_path=get_abs_path(agent_conf["user_profiles_path"]),
    records_path=get_abs_path(agent_conf["external_data_path"]),
)
weather = WeatherService(timeout_seconds=float(agent_conf.get("weather_timeout_seconds", 8)))


def set_current_user_id(user_id: str) -> None:
    """Bind the Streamlit-selected user to the current execution context."""
    current_user_id.set(str(user_id))


def _json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


@tool(description="Search the robot knowledge base for troubleshooting, maintenance, and product guidance.")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


@tool(description="Get live weather for a city from the free Open-Meteo public API. Use the city name as input.")
def get_weather(city: str) -> str:
    try:
        return _json({"available": True, **weather.get_current_weather(city).as_dict()})
    except WeatherServiceError as error:
        return _json({"available": False, "city": city, "error": str(error)})


@tool(description="Get the current user's city from the configured user profile.")
def get_user_location() -> str:
    user_id = current_user_id.get()
    city = user_data.get_city(user_id)
    if city:
        return _json({"available": True, "user_id": user_id, "city": city})
    return _json({"available": False, "user_id": user_id, "error": "No city is configured for this user"})


@tool(description="Get the current user's ID for account-specific requests.")
def get_user_id() -> str:
    return current_user_id.get()


@tool(description="Get the current calendar month in YYYY-MM format.")
def get_current_month() -> str:
    return datetime.now().strftime("%Y-%m")


@tool(description="Fetch a user's robot usage record for a YYYY-MM month. Returns structured data or an explicit no-record result.")
def fetch_external_data(user_id: str, month: str) -> str:
    try:
        record = user_data.get_usage_record(user_id, month)
        if record:
            return _json({"found": True, "user_id": user_id, "month": month, "record": record})
        return _json(
            {
                "found": False,
                "user_id": user_id,
                "month": month,
                "available_months": user_data.available_months(user_id),
            }
        )
    except (FileNotFoundError, ValueError) as error:
        return _json({"found": False, "user_id": user_id, "month": month, "error": str(error)})


@tool(description="Prepare report-generation context before fetching user usage data. Use only for a personal usage report.")
def fill_context_for_report() -> str:
    return "Report context is ready."
