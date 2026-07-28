import csv
import json
import tempfile
import unittest
from pathlib import Path

from services.user_data_service import MONTH_FIELD, USER_ID_FIELD, UserDataService


class UserDataServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.profiles_path = root / "users.json"
        self.records_path = root / "records.csv"
        self.profiles_path.write_text(json.dumps({"u-1": {"city": "Beijing"}}), encoding="utf-8")
        with self.records_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=[USER_ID_FIELD, "status", MONTH_FIELD])
            writer.writeheader()
            writer.writerow({USER_ID_FIELD: "u-1", "status": "ok", MONTH_FIELD: "2025-01"})
            writer.writerow({USER_ID_FIELD: "u-1", "status": "due", MONTH_FIELD: "2025-02"})
        self.service = UserDataService(self.profiles_path, self.records_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_loads_profile_city(self):
        self.assertEqual(self.service.get_city("u-1"), "Beijing")
        self.assertIsNone(self.service.get_city("missing"))

    def test_finds_usage_record_and_available_months(self):
        record = self.service.get_usage_record("u-1", "2025-02")
        self.assertEqual(record["status"], "due")
        self.assertEqual(self.service.available_months("u-1"), ["2025-01", "2025-02"])
        self.assertIsNone(self.service.get_usage_record("u-1", "2026-07"))


if __name__ == "__main__":
    unittest.main()
