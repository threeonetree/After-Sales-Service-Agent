"""Local data adapters for the demo user and robot usage data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


USER_ID_FIELD = "\u7528\u6237ID"
MONTH_FIELD = "\u65f6\u95f4"


class UserDataService:
    """Read user profiles and usage records without embedding demo data in tools."""

    def __init__(self, profiles_path: str | Path, records_path: str | Path):
        self.profiles_path = Path(profiles_path)
        self.records_path = Path(records_path)

    def get_profile(self, user_id: str) -> dict[str, Any] | None:
        if not self.profiles_path.exists():
            return None

        with self.profiles_path.open("r", encoding="utf-8") as file:
            profiles = json.load(file)
        profile = profiles.get(str(user_id))
        return profile if isinstance(profile, dict) else None

    def get_city(self, user_id: str) -> str | None:
        profile = self.get_profile(user_id)
        city = profile.get("city") if profile else None
        return city if isinstance(city, str) and city.strip() else None

    def get_usage_record(self, user_id: str, month: str) -> dict[str, Any] | None:
        for record in self._usage_records():
            if record.get(USER_ID_FIELD) == str(user_id) and record.get(MONTH_FIELD) == month:
                return record
        return None

    def available_months(self, user_id: str) -> list[str]:
        return sorted(
            {
                record[MONTH_FIELD]
                for record in self._usage_records()
                if record.get(USER_ID_FIELD) == str(user_id) and record.get(MONTH_FIELD)
            }
        )

    def _usage_records(self) -> list[dict[str, str]]:
        if not self.records_path.exists():
            raise FileNotFoundError(f"Usage record file does not exist: {self.records_path}")

        with self.records_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or USER_ID_FIELD not in reader.fieldnames or MONTH_FIELD not in reader.fieldnames:
                raise ValueError("Usage record CSV must contain user ID and month columns")
            return list(reader)
