import json
import subprocess
import sys
from pathlib import Path

import pytest

from evals.run_agent_eval import (
    _check_permission_boundary,
    _run_permission_probe,
    check_case,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = REPO_ROOT / "evals" / "agent_cases.jsonl"


def _load_cases():
    return [
        json.loads(line)
        for line in CASE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_agent_case_dataset_is_complete_and_has_unique_ids():
    cases = _load_cases()
    ids = [case["id"] for case in cases]

    assert len(cases) == 20
    assert len(ids) == len(set(ids))
    assert all(
        isinstance(case.get("id"), str)
        and case["id"].strip()
        and isinstance(case.get("message"), str)
        and isinstance(case.get("expected"), str)
        for case in cases
    )


def test_every_case_has_a_real_runner_mapping_and_only_known_cases_fail():
    cases = _load_cases()
    results = {case["id"]: check_case(case) for case in cases}

    assert set(results) == {case["id"] for case in cases}
    assert {case_id for case_id, result in results.items() if not result["passed"]} == {
        "correct_city",
        "correct_days",
    }
    for case in cases:
        result = results[case["id"]]
        assert result["actual"] == case["expected"], case["id"]
        if case["id"] in {"correct_city", "correct_days"}:
            assert result["passed"] is False
            assert "correction parser" in result["reason"]
            assert "multi-turn evaluator" in result["reason"]
        else:
            assert result["passed"] is True, (case["id"], result)


def test_cli_reports_tool_contract_scope_and_keeps_known_failures_red():
    try:
        completed = subprocess.run(
            [sys.executable, "evals/run_agent_eval.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            "tool contract evaluation CLI exceeded 30 seconds; "
            f"stdout={exc.stdout!r}; stderr={exc.stderr!r}",
            pytrace=False,
        )

    assert completed.returncode == 1, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    assert report["evaluation_type"] == "travel_tool_contract"
    assert report["natural_language_understanding_evaluated"] is False
    assert report["cases"] == 20
    assert report["passed"] == 18
    assert report["failed"] == 2
    failed = {result["id"]: result for result in report["results"] if not result["passed"]}
    assert set(failed) == {"correct_city", "correct_days"}
    assert all(result["actual"] == "not_implemented" for result in failed.values())
    assert all(
        "correction parser" in result["reason"]
        and "multi-turn evaluator" in result["reason"]
        for result in failed.values()
    )


def test_permission_boundary_observes_denial_without_mutating_input_history():
    result = _run_permission_probe("guest")

    assert result["denied"] is True
    assert result["authorized"] is False
    assert result["tool_execution_count"] == 0
    assert result["client_calls"] == 2
    assert result["caller_history_unchanged"] is True
    assert result["tool_reply"]["role"] == "tool"
    assert result["tool_reply"]["tool_call_id"] == "eval_permission_call"
    assert "Permission denied" in result["tool_reply"]["content"]
    assert result["received_tools"][0] == []


def test_authorized_permission_probe_executes_tool_and_observes_result():
    result = _run_permission_probe("employee")

    assert result["authorized"] is True
    assert result["denied"] is False
    assert result["tool_execution_count"] == 1
    assert result["client_calls"] == 2
    assert result["caller_history_unchanged"] is True
    assert result["tool_reply"]["role"] == "tool"
    assert result["tool_reply"]["tool_call_id"] == "eval_permission_call"
    assert '"actual_total": 600' in result["tool_reply"]["content"]
    assert any(
        schema["function"]["name"] == "calculate_travel_budget"
        for schema in result["received_tools"][0]
    )


def test_permission_evaluation_case_still_reports_expected_denial():
    result = _check_permission_boundary()

    assert result["actual"] == "deny"
    assert result["passed"] is True
    assert "did not execute the tool" in result["reason"]
