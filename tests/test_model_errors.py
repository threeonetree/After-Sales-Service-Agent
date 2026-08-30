import os
import unittest
from unittest.mock import patch

from utils.model_errors import (
    ModelConfigurationError,
    require_dashscope_api_key,
    user_facing_model_error,
)


class ModelErrorTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_key_is_rejected(self):
        with self.assertRaises(ModelConfigurationError):
            require_dashscope_api_key()

    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-only"}, clear=True)
    def test_key_is_read_without_transformation(self):
        self.assertEqual(require_dashscope_api_key(), "test-only")

    def test_free_tier_error_has_stop_message(self):
        message = user_facing_model_error(
            RuntimeError("403 AllocationQuota.FreeTierOnly")
        )
        self.assertIn("免费额度已用完", message)
        self.assertIn("不会继续产生模型费用", message)

    def test_missing_dependency_has_install_command(self):
        message = user_facing_model_error(ModuleNotFoundError("langgraph"))
        self.assertIn("requirements.txt", message)


if __name__ == "__main__":
    unittest.main()
