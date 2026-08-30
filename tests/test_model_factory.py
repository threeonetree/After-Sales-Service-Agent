import importlib
import os
import sys
import unittest
from unittest.mock import patch


class ModelFactoryTests(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("model.factory", None)

    @patch.dict(
        os.environ,
        {
            "DASHSCOPE_API_KEY": "test-only",
            "DASHSCOPE_BASE_URL": "https://example.invalid/compatible-mode/v1",
        },
        clear=True,
    )
    def test_chat_uses_openai_compatible_endpoint(self):
        factory = importlib.import_module("model.factory")

        self.assertEqual(
            str(factory.chat_model.openai_api_base),
            "https://example.invalid/compatible-mode/v1",
        )
        self.assertEqual(
            factory.chat_model.extra_body,
            {"enable_thinking": False},
        )
        self.assertEqual(
            factory.embed_model.model,
            "qwen3.7-text-embedding",
        )

    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-only"}, clear=True)
    def test_mainland_endpoint_is_the_default(self):
        factory = importlib.import_module("model.factory")

        self.assertEqual(
            factory.get_dashscope_base_url(),
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )


if __name__ == "__main__":
    unittest.main()
