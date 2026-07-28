import unittest

from evals.run_contract_evals import compare_tool_calls, tool_call_scores, validate_cases


class ContractEvaluationTests(unittest.TestCase):
    def test_matching_calls_pass(self):
        passed, failures = compare_tool_calls(
            [
                {"name": "get_user_id"},
                {"name": "fetch_external_data", "args": {"user_id": "1001"}},
            ],
            [
                {"name": "get_user_id", "args": {}},
                {"name": "fetch_external_data", "args": {"user_id": "1001", "month": "2025-01"}},
            ],
        )
        self.assertTrue(passed)
        self.assertEqual(failures, [])

    def test_extra_or_wrong_calls_fail(self):
        passed, failures = compare_tool_calls(
            [{"name": "get_user_id"}],
            [{"name": "get_weather", "args": {"city": "Beijing"}}, {"name": "get_user_id", "args": {}}],
        )
        self.assertFalse(passed)
        self.assertTrue(failures)

    def test_case_ids_must_be_unique(self):
        with self.assertRaises(ValueError):
            validate_cases(
                [
                    {"id": "same", "query": "one", "expected_tool_calls": []},
                    {"id": "same", "query": "two", "expected_tool_calls": []},
                ]
            )

    def test_tool_scores_reflect_extra_call(self):
        scores = tool_call_scores(
            [{"name": "get_user_id"}],
            [{"name": "get_user_id", "args": {}}, {"name": "rag_summarize", "args": {}}],
        )
        self.assertEqual(scores["strict_accuracy"], 0.0)
        self.assertEqual(scores["tool_precision"], 0.5)
        self.assertEqual(scores["tool_recall"], 1.0)
        self.assertAlmostEqual(scores["tool_f1"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
