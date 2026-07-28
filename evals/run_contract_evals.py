"""Run the agent against explicit business-tool contracts without Ragas."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from agent.react_agent import ReactAgent
from agent.tools.agent_tools import set_current_user_id


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH = PROJECT_ROOT / "evals" / "tool_contract_cases.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "evals" / "results"


def load_cases(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        cases = json.load(file)
    if not isinstance(cases, list):
        raise ValueError("Evaluation cases must be a JSON array")
    return cases


def validate_cases(cases: Iterable[Dict[str, Any]]) -> None:
    seen_ids = set()
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Every evaluation case needs a non-empty id")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate evaluation case id: {case_id}")
        seen_ids.add(case_id)
        if not isinstance(case.get("query"), str) or not case["query"].strip():
            raise ValueError(f"Case {case_id} needs a non-empty query")
        if not isinstance(case.get("expected_tool_calls"), list):
            raise ValueError(f"Case {case_id} needs expected_tool_calls")


def compare_tool_calls(
    expected_calls: List[Dict[str, Any]], actual_calls: List[Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    """Compare exact sequence plus every explicitly specified argument."""
    failures: List[str] = []
    if len(actual_calls) != len(expected_calls):
        failures.append(
            f"Expected {len(expected_calls)} tool calls, got {len(actual_calls)}"
        )

    for index, expected in enumerate(expected_calls):
        if index >= len(actual_calls):
            failures.append(f"Missing tool call at position {index + 1}: {expected.get('name')}")
            continue

        actual = actual_calls[index]
        if actual.get("name") != expected.get("name"):
            failures.append(
                f"Position {index + 1}: expected {expected.get('name')}, got {actual.get('name')}"
            )
            continue

        expected_args = expected.get("args", {})
        actual_args = actual.get("args", {})
        for key, value in expected_args.items():
            if actual_args.get(key) != value:
                failures.append(
                    f"Position {index + 1} ({expected['name']}): argument {key!r} "
                    f"expected {value!r}, got {actual_args.get(key)!r}"
                )

    for index in range(len(expected_calls), len(actual_calls)):
        failures.append(f"Unexpected tool call at position {index + 1}: {actual_calls[index].get('name')}")

    return not failures, failures


def tool_call_scores(
    expected_calls: List[Dict[str, Any]], actual_calls: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Return deterministic tool-call quality metrics in the 0.0-1.0 range."""
    matched = 0
    for expected, actual in zip(expected_calls, actual_calls):
        if actual.get("name") != expected.get("name"):
            continue
        expected_args = expected.get("args", {})
        actual_args = actual.get("args", {})
        if all(actual_args.get(key) == value for key, value in expected_args.items()):
            matched += 1

    expected_count = len(expected_calls)
    actual_count = len(actual_calls)
    precision = matched / actual_count if actual_count else float(expected_count == 0)
    recall = matched / expected_count if expected_count else float(actual_count == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    strict_passed, _ = compare_tool_calls(expected_calls, actual_calls)
    return {
        "matched_tool_calls": matched,
        "expected_tool_calls": expected_count,
        "actual_tool_calls": actual_count,
        "strict_accuracy": float(strict_passed),
        "tool_precision": precision,
        "tool_recall": recall,
        "tool_f1": f1,
    }


def run_case(agent: ReactAgent, case: Dict[str, Any]) -> Dict[str, Any]:
    set_current_user_id(str(case["user_id"]))
    execution = agent.execute_with_trace(
        case["query"], thread_id=f"eval-{case['id']}-{datetime.now().timestamp()}"
    )
    passed, failures = compare_tool_calls(
        case["expected_tool_calls"], execution.tool_calls
    )
    scores = tool_call_scores(case["expected_tool_calls"], execution.tool_calls)
    return {
        "id": case["id"],
        "query": case["query"],
        "expected_tool_calls": case["expected_tool_calls"],
        "actual_tool_calls": execution.tool_calls,
        "tool_results": execution.tool_results,
        "response": execution.response,
        "expected_outcome": case.get("expected_outcome", ""),
        "tool_contract_passed": passed,
        "failures": failures,
        "scores": scores,
    }


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute strict accuracy and micro-averaged tool metrics for all cases."""
    total_cases = len(results)
    strict_passed = sum(result["tool_contract_passed"] for result in results)
    matched = sum(result["scores"]["matched_tool_calls"] for result in results)
    actual = sum(result["scores"]["actual_tool_calls"] for result in results)
    expected = sum(result["scores"]["expected_tool_calls"] for result in results)
    precision = matched / actual if actual else float(expected == 0)
    recall = matched / expected if expected else float(actual == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "total_cases": total_cases,
        "strict_passed_cases": strict_passed,
        "strict_accuracy": strict_passed / total_cases if total_cases else 0.0,
        "tool_precision": precision,
        "tool_recall": recall,
        "tool_f1": f1,
    }


def result_path(output: str | None) -> Path:
    if output:
        return Path(output)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_RESULTS_DIR / f"contract_eval_{timestamp}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run after-sales agent tool contracts")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--case", dest="case_id", help="Run one case by id")
    parser.add_argument("--output", help="Path for the JSON result file")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate cases without calling the model"
    )
    args = parser.parse_args()

    cases = load_cases(args.cases)
    validate_cases(cases)
    if args.case_id:
        cases = [case for case in cases if case["id"] == args.case_id]
        if not cases:
            raise ValueError(f"No case found with id: {args.case_id}")

    if args.dry_run:
        print(f"Validated {len(cases)} evaluation cases. No model call was made.")
        return 0

    agent = ReactAgent()
    results = [run_case(agent, case) for case in cases]
    summary = summarize_results(results)
    output_path = result_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump({"summary": summary, "results": results}, file, ensure_ascii=False, indent=2)

    print(
        "Strict accuracy: "
        f"{summary['strict_accuracy']:.1%} ({summary['strict_passed_cases']}/{summary['total_cases']})"
    )
    print(
        "Tool metrics: "
        f"precision={summary['tool_precision']:.3f}, "
        f"recall={summary['tool_recall']:.3f}, f1={summary['tool_f1']:.3f}"
    )
    print(f"Results: {output_path}")
    return 0 if summary["strict_passed_cases"] == summary["total_cases"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        sys.exit(2)
