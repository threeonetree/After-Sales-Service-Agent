"""Free, keyless weather adapter based on the Open-Meteo public API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


class WeatherServiceError(RuntimeError):
    """Raised when a weather lookup cannot produce a reliable result."""


@dataclass(frozen=True)
class WeatherSnapshot:
    city: str
    temperature_c: float
    humidity_percent: float
    precipitation_mm: float
    wind_speed_kmh: float
    condition: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "city": self.city,
            "temperature_c": self.temperature_c,
            "humidity_percent": self.humidity_percent,
            "precipitation_mm": self.precipitation_mm,
            "wind_speed_kmh": self.wind_speed_kmh,
            "condition": self.condition,
            "source": "Open-Meteo",
        }


class WeatherService:
    """Look up current weather using public endpoints that require no API key."""

    geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
    forecast_url = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, timeout_seconds: float = 8.0, session: requests.Session | None = None):
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    def get_current_weather(self, city: str) -> WeatherSnapshot:
        location = self._geocode(city)
        weather = self._get_forecast(location["latitude"], location["longitude"])
        current = weather.get("current")
        if not isinstance(current, dict):
            raise WeatherServiceError("Weather service returned no current conditions")

        try:
            return WeatherSnapshot(
                city=location["name"],
                temperature_c=float(current["temperature_2m"]),
                humidity_percent=float(current["relative_humidity_2m"]),
                precipitation_mm=float(current["precipitation"]),
                wind_speed_kmh=float(current["wind_speed_10m"]),
                condition=self._weather_code_label(int(current["weather_code"])),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise WeatherServiceError("Weather service returned an invalid response") from error

    def _geocode(self, city: str) -> dict[str, Any]:
        response = self.session.get(
            self.geocoding_url,
            params={"name": city, "count": 1, "language": "zh", "format": "json"},
            timeout=self.timeout_seconds,
        )
        self._raise_for_status(response, "city lookup")
        results = response.json().get("results", [])
        if not results:
            raise WeatherServiceError(f"City not found: {city}")
        return results[0]

    def _get_forecast(self, latitude: float, longitude: float) -> dict[str, Any]:
        response = self.session.get(
            self.forecast_url,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
            },
            timeout=self.timeout_seconds,
        )
        self._raise_for_status(response, "weather lookup")
        return response.json()

    @staticmethod
    def _raise_for_status(response: requests.Response, operation: str) -> None:
        try:
            response.raise_for_status()
        except requests.RequestException as error:
            raise WeatherServiceError(f"Unable to complete {operation}") from error

    @staticmethod
    def _weather_code_label(code: int) -> str:
        labels = {
            0: "clear",
            1: "mainly clear",
            2: "partly cloudy",
            3: "overcast",
            45: "fog",
            48: "rime fog",
            51: "light drizzle",
            53: "moderate drizzle",
            55: "heavy drizzle",
            61: "light rain",
            63: "moderate rain",
            65: "heavy rain",
            71: "light snow",
            73: "moderate snow",
            75: "heavy snow",
            80: "rain showers",
            81: "moderate rain showers",
            82: "violent rain showers",
            95: "thunderstorm",
        }
        return labels.get(code, f"weather code {code}")
