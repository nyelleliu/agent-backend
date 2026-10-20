import copy
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.run_agent_loop_eval import (
    ScriptedFakeClient,
    _response,
    _call,
    BudgetedLiveClient,
    TRUSTED_LIVE_ENDPOINT,
    MAX_REQUEST_TIMEOUT_SECONDS,
    MAX_TOTAL_TIMEOUT_SECONDS,
    _live_case_definitions,
    _multiturn_scenario_definitions,
    _scenario_definitions,
    _run_agent_with_temporary_log_filters,
    run_case,
    run_evaluation,
    run_live_evaluation,
    main as eval_main,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_offline_agent_loop_evaluation_cases_pass_real_execution_checks():
    report = run_evaluation()

    assert report["cases"] == 11
    assert report["passed"] == 11
    assert report["failed"] == 0
    assert all(result["task_success"] for result in report["results"])


def test_metrics_capture_tool_calls_permissions_budget_and_fake_usage():
    results = {result["case_id"]: result for result in run_evaluation()["results"]}

    assert results["travel_success"]["tool_execution_count"] == 1
    assert results["travel_success"]["llm_call_attempt_count"] == 2
    assert results["travel_success"]["token_usage_status"] == "available_from_fake_response"
    assert results["travel_success"]["total_tokens"] == 356

    assert results["missing_city_no_execution"]["tool_execution_count"] == 0
    assert results["missing_city_no_execution"]["token_usage_status"] == "unavailable"

    assert results["unauthorized_role_denied"]["tool_execution_count"] == 0
    assert results["tool_call_budget_exhausted"]["tool_execution_count"] == 1
    assert results["tool_call_budget_exhausted"]["llm_call_attempt_count"] == 2


def test_each_case_gets_independent_metrics_and_message_state():
    case = next(case for case in _scenario_definitions() if case["case_id"] == "travel_success")

    first = run_case(case)
    second = run_case(case)

    assert first["tool_calls"] == second["tool_calls"]
    assert first is not second
    assert first["case_id"] == second["case_id"] == "travel_success"


def _wrong_expected_total_case():
    case = copy.deepcopy(
        next(case for case in _scenario_definitions() if case["case_id"] == "travel_success")
    )
    case["expected"]["tool_result_json"][0]["actual_total"] = 1499
    return case


def test_wrong_expected_result_fails_in_normal_python():
    result = run_case(_wrong_expected_total_case())

    assert result["task_success"] is False
    assert "expected 1499" in result["failure_reason"]


def test_wrong_expected_result_fails_under_python_optimization():
    code = (
        "import copy, json; "
        "from evals.run_agent_loop_eval import _scenario_definitions, run_case; "
        "case = copy.deepcopy(next(c for c in _scenario_definitions() "
        "if c['case_id'] == 'travel_success')); "
        "case['expected']['tool_result_json'][0]['actual_total'] = 1499; "
        "result = run_case(case); "
        "print(json.dumps({'task_success': result['task_success'], "
        "'failure_reason': result['failure_reason']})); "
        "raise SystemExit(0 if (not result['task_success'] and "
        "'expected 1499' in result['failure_reason']) else 1)"
    )
    completed = subprocess.run(
        [sys.executable, "-O", "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert '"task_success": false' in completed.stdout
    assert "expected 1499" in completed.stdout


def test_create_attempt_count_includes_exhaustion_and_response_exception():
    exhausted = ScriptedFakeClient([])
    with pytest.raises(RuntimeError, match="ran out of scripted responses"):
        exhausted.create()
    assert exhausted.create_call_attempt_count == 1

    raising = ScriptedFakeClient([ValueError("scripted response failure")])
    with pytest.raises(ValueError, match="scripted response failure"):
        raising.create()
    assert raising.create_call_attempt_count == 1


def test_run_case_reports_create_attempt_when_fake_client_raises():
    result = run_case({
        "case_id": "client_exception",
        "role": "employee",
        "responses": [RuntimeError("scripted failure")],
        "expected": {},
    })

    assert result["task_success"] is False
    assert result["llm_call_attempt_count"] == 1
    assert "scripted failure" in result["failure_reason"]


def _usage_probe(usage):
    return run_case({
        "case_id": "usage_probe",
        "role": "employee",
        "responses": [_response(content="done", usage=usage)],
        "expected": {
            "tool_execution_count": 0,
            "llm_call_attempt_count": 1,
            "issued_tool_calls": [],
            "executed_tool_calls": [],
            "tool_message_count": 0,
            "reply_contains": ["done"],
        },
    })


def test_missing_usage_fields_are_unavailable():
    result = _usage_probe({"prompt_tokens": 2, "completion_tokens": 3})

    assert result["task_success"] is True
    assert result["token_usage_status"] == "unavailable"
    assert result["prompt_tokens"] is None
    assert result["total_tokens"] is None


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": -1, "completion_tokens": 3, "total_tokens": 2},
        {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 99},
        {"prompt_tokens": True, "completion_tokens": 3, "total_tokens": 4},
    ],
)
def test_invalid_or_inconsistent_usage_is_not_counted(usage):
    result = _usage_probe(usage)

    assert result["token_usage_status"] == "invalid"
    assert result["prompt_tokens"] is None
    assert result["completion_tokens"] is None
    assert result["total_tokens"] is None
    assert result["token_usage_error"]


def test_all_multiturn_scenarios_validate_each_turn_and_case_metrics():
    results = {
        case["case_id"]: run_case(case)
        for case in _multiturn_scenario_definitions()
    }

    assert set(results) == {
        "multiturn_add_city",
        "multiturn_add_days",
        "multiturn_add_job_level",
        "multiturn_correct_days",
        "multiturn_correct_unsupported_city",
    }
    for case_id, result in results.items():
        assert result["task_success"] is True, (case_id, result["failure_reason"])
        assert all(turn["task_success"] for turn in result["turns"]), case_id
        assert result["tool_execution_count"] == sum(
            turn["tool_execution_count"] for turn in result["turns"]
        )
        assert result["llm_call_attempt_count"] == sum(
            turn["llm_call_attempt_count"] for turn in result["turns"]
        )


@pytest.mark.parametrize(
    ("case_id", "first_expected_calls", "second_expected_calls"),
    [
        ("multiturn_add_city", 0, 1),
        ("multiturn_add_days", 0, 1),
        ("multiturn_add_job_level", 0, 1),
    ],
)
def test_missing_fields_trigger_followup_then_execute_completed_parameters(
    case_id, first_expected_calls, second_expected_calls
):
    case = next(c for c in _multiturn_scenario_definitions() if c["case_id"] == case_id)
    result = run_case(case)

    assert result["turns"][0]["tool_execution_count"] == first_expected_calls
    assert result["turns"][0]["tool_calls"] == []
    assert result["turns"][1]["tool_execution_count"] == second_expected_calls
    prior_history = result["turns"][1]["history_before"]
    assert any(
        message["role"] == "user" and message["content"] == case["turns"][0]["user_message"]
        for message in prior_history
    )
    assert any(
        message["role"] == "assistant" and message["content"] == result["turns"][0]["reply"]
        for message in prior_history
    )
    assert result["turns"][1]["issued_tool_calls"] == case["turns"][1]["expected"]["issued_tool_calls"]


def test_corrected_days_use_latest_value_in_real_registry_call_and_result():
    case = next(c for c in _multiturn_scenario_definitions() if c["case_id"] == "multiturn_correct_days")
    result = run_case(case)

    corrected = result["turns"][1]
    assert corrected["user_message"] == "更正为 5 天。"
    assert corrected["history_before"][-1]["content"] == "更正为 5 天。"
    assert corrected["issued_tool_calls"][0]["arguments"]["days"] == 5
    assert corrected["tool_calls"][0]["arguments"]["days"] == 5
    assert corrected["tool_calls"][0]["result"]
    assert '"days": 5' in corrected["tool_calls"][0]["result"]
    assert '"actual_total": 2400' in corrected["tool_calls"][0]["result"]


def test_corrected_unsupported_city_is_rejected_without_reusing_old_budget():
    case = next(c for c in _multiturn_scenario_definitions() if c["case_id"] == "multiturn_correct_unsupported_city")
    result = run_case(case)

    corrected = result["turns"][1]
    assert corrected["issued_tool_calls"][0]["arguments"]["city"] == "杭州"
    assert corrected["tool_calls"][0]["arguments"]["city"] == "杭州"
    assert "unsupported city" in corrected["tool_calls"][0]["result"]
    assert '"ok": true' not in corrected["tool_calls"][0]["result"]
    assert "1050" not in corrected["reply"]


def test_cases_do_not_share_messages_metrics_clients_or_tool_records():
    case = next(c for c in _multiturn_scenario_definitions() if c["case_id"] == "multiturn_correct_days")
    first = run_case(case)
    second = run_case(case)

    assert first is not second
    assert first["final_messages"] == second["final_messages"]
    assert first["final_messages"] is not second["final_messages"]
    assert first["turns"][0]["tool_calls"] is not second["turns"][0]["tool_calls"]


class CountingRawClient:
    def __init__(self, responses):
        self.calls = 0
        self.responses = list(responses)
        self.last_kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if not self.responses:
            raise RuntimeError("unexpected additional request")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _live_test_case(turns):
    return {"case_id": "live_test", "role": "employee", "turns": turns}


def test_default_entry_remains_offline_even_when_api_key_is_present(monkeypatch):
    monkeypatch.setenv("AGENT_EVAL_API_KEY", "must-not-be-used")
    monkeypatch.setattr(sys, "argv", ["run_agent_loop_eval.py"])
    monkeypatch.setattr(
        "evals.run_agent_loop_eval._create_live_client",
        lambda *args, **kwargs: pytest.fail("live client constructed in default mode"),
    )

    assert eval_main() == 0


def test_without_live_switch_or_missing_key_never_constructs_live_client(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_agent_loop_eval.py", "--live"])
    monkeypatch.delenv("AGENT_EVAL_API_KEY", raising=False)
    monkeypatch.setattr(
        "evals.run_agent_loop_eval._create_live_client",
        lambda *args, **kwargs: pytest.fail("client constructed without key"),
    )

    assert eval_main() == 2


def test_cli_live_is_explicit_and_reads_only_configured_key_env(monkeypatch, capsys):
    monkeypatch.setenv("EVAL_KEY_FOR_TEST", "secret-test-key")
    monkeypatch.setattr(sys, "argv", ["run_agent_loop_eval.py", "--live", "--api-key-env", "EVAL_KEY_FOR_TEST"])
    observed = {}

    def fake_live(**kwargs):
        observed.update(kwargs)
        return {"mode": "live", "cases_run": 1, "results": [{"task_success": True}]}

    monkeypatch.setattr("evals.run_agent_loop_eval.run_live_evaluation", fake_live)

    assert eval_main() == 0
    assert observed["api_key"] == "secret-test-key"
    assert "secret-test-key" not in capsys.readouterr().out


def test_case_budget_prevents_actual_excess_requests():
    raw = CountingRawClient([_response(content="done"), _response(content="never")])
    case = _live_test_case([
        {"user_message": "first"},
        {"user_message": "second"},
    ])

    report = run_live_evaluation(
        api_key="test-key",
        case_limit=1,
        max_calls_per_case=1,
        max_calls_per_turn=3,
        max_total_calls=5,
        client_factory=lambda *args, **kwargs: raw,
        cases=[case],
    )

    assert raw.calls == 1
    assert report["global_llm_call_attempt_count"] == 2
    assert report["global_llm_forwarded_request_attempt_count"] == 1
    assert report["results"][0]["task_success"] is False
    assert report["results"][0]["llm_call_attempt_count"] == 2
    assert report["results"][0]["stop_reason"] == "per_case_llm_call_budget_exhausted"


def test_global_budget_prevents_actual_excess_requests_inside_agent_loop():
    raw = CountingRawClient([
        _response(tool_calls=[_call({"city": "上海", "days": 1, "job_level": "employee"})]),
        _response(content="must not be requested"),
    ])
    case = _live_test_case([{"user_message": "calculate"}])

    report = run_live_evaluation(
        api_key="test-key",
        max_calls_per_case=3,
        max_calls_per_turn=3,
        max_total_calls=1,
        client_factory=lambda *args, **kwargs: raw,
        cases=[case],
    )

    assert raw.calls == 1
    assert report["global_llm_call_attempt_count"] == 2
    assert report["global_llm_forwarded_request_attempt_count"] == 1
    assert report["results"][0]["task_success"] is False
    assert report["results"][0]["stop_reason"] == "global_llm_call_budget_exhausted"
    assert report["results"][0]["tool_execution_count"] == 1


def test_turn_budget_prevents_actual_excess_requests():
    raw = CountingRawClient([
        _response(tool_calls=[_call({"city": "上海", "days": 1, "job_level": "employee"})]),
        _response(content="must not be requested"),
    ])
    case = _live_test_case([{"user_message": "calculate"}])

    report = run_live_evaluation(
        api_key="test-key",
        max_calls_per_case=3,
        max_calls_per_turn=1,
        max_total_calls=3,
        client_factory=lambda *args, **kwargs: raw,
        cases=[case],
    )

    assert raw.calls == 1
    assert report["results"][0]["task_success"] is False
    assert report["results"][0]["turns"][0]["stop_reason"] == "per_turn_llm_call_budget_exhausted"


def test_request_timeout_is_forwarded_and_total_deadline_blocks_before_request():
    raw = CountingRawClient([_response(content="done")])
    case = _live_test_case([{"user_message": "hello"}])
    report = run_live_evaluation(
        api_key="test-key",
        request_timeout=7.5,
        client_factory=lambda *args, **kwargs: raw,
        cases=[case],
    )
    assert report["results"][0]["task_success"] is True
    assert raw.calls == 1
    assert raw.last_kwargs["timeout"] <= 7.5

    untouched = CountingRawClient([_response(content="must not be requested")])
    expired = run_live_evaluation(
        api_key="test-key",
        total_timeout=1e-9,
        client_factory=lambda *args, **kwargs: untouched,
        cases=[case],
    )
    assert untouched.calls == 0
    assert expired["results"][0]["task_success"] is False
    assert expired["results"][0]["stop_reason"] == "total_runtime_budget_exhausted"


def test_live_request_failure_and_timeout_are_not_success_and_have_no_fake_usage():
    raw = CountingRawClient([TimeoutError("request timed out")])
    case = _live_test_case([{"user_message": "hello"}])

    report = run_live_evaluation(
        api_key="test-key",
        client_factory=lambda *args, **kwargs: raw,
        cases=[case],
    )

    turn = report["results"][0]["turns"][0]
    assert turn["task_success"] is False
    assert turn["failure_type"] == "TimeoutError"
    assert turn["stop_reason"] == "request_timeout"
    assert turn["llm_call_attempt_count"] == 1
    assert turn["token_usage_status"] == "unavailable"


def test_live_client_initialization_failure_is_structured_and_does_not_leak_details():
    report = run_live_evaluation(
        api_key="test-key",
        client_factory=lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("failure detail that should not be logged")
        ),
    )

    assert report["cases_run"] == 0
    assert report["results"] == []
    assert report["stop_reason"] == "client_initialization_failed"
    assert report["initialization_failure"]["failure_type"] == "RuntimeError"
    assert "failure detail" not in str(report)


@pytest.mark.parametrize(
    ("usage", "status"),
    [
        (None, "unavailable"),
        ({"prompt_tokens": 2, "completion_tokens": 3}, "unavailable"),
        ({"prompt_tokens": -1, "completion_tokens": 1, "total_tokens": 0}, "invalid"),
        ({"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 99}, "invalid"),
        ({"prompt_tokens": True, "completion_tokens": 3, "total_tokens": 4}, "invalid"),
        ({"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}, "available"),
    ],
)
def test_live_usage_is_reported_without_estimation(usage, status):
    raw = CountingRawClient([_response(content="done", usage=usage)])
    case = _live_test_case([{"user_message": "hello"}])

    report = run_live_evaluation(
        api_key="test-key",
        client_factory=lambda *args, **kwargs: raw,
        cases=[case],
    )

    turn = report["results"][0]["turns"][0]
    record = turn["token_usage"]
    assert turn["token_usage_status"] == status
    if status != "available":
        assert record is None
    else:
        assert record == [{
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
        }]


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://attacker.invalid/v1",
        "https://user:pass@attacker.invalid/v1",
        "https://attacker.invalid/v1?token=secret",
        "https://attacker.invalid/v1#fragment",
        "not-a-url",
    ],
)
def test_unsafe_custom_endpoint_is_rejected_before_client_construction(endpoint):
    def forbidden_factory(*args, **kwargs):
        pytest.fail("client factory must not run for an unapproved endpoint")

    with pytest.raises(ValueError):
        run_live_evaluation(
            api_key="test-key",
            base_url=endpoint,
            client_factory=forbidden_factory,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_timeout", float("nan")),
        ("request_timeout", float("inf")),
        ("request_timeout", float("-inf")),
        ("request_timeout", 0),
        ("request_timeout", -1),
        ("request_timeout", MAX_REQUEST_TIMEOUT_SECONDS + 1),
        ("total_timeout", float("nan")),
        ("total_timeout", float("inf")),
        ("total_timeout", float("-inf")),
        ("total_timeout", 0),
        ("total_timeout", -1),
        ("total_timeout", MAX_TOTAL_TIMEOUT_SECONDS + 1),
    ],
)
def test_non_finite_or_out_of_range_timeout_is_rejected_before_client(field, value):
    def forbidden_factory(*args, **kwargs):
        pytest.fail("client factory must not run for invalid timeout")

    with pytest.raises(ValueError):
        run_live_evaluation(
            api_key="test-key",
            client_factory=forbidden_factory,
            **{field: value},
        )


def test_explicit_custom_https_endpoint_is_forwarded_only_after_opt_in():
    raw = CountingRawClient([
        _response(tool_calls=[_call({"city": "上海", "days": 3, "job_level": "employee"})]),
        _response(content="上海 3 天预算 1500"),
    ])
    report = run_live_evaluation(
        api_key="test-key",
        base_url="https://custom.example/v1",
        allow_custom_endpoint=True,
        client_factory=lambda *args, **kwargs: raw,
        cases=_live_case_definitions()[:1],
    )
    assert report["cases_run"] == 1
    assert raw.calls == 2  # This built-in case requires a tool turn and final answer.


def test_default_endpoint_ignores_environment_override(monkeypatch):
    monkeypatch.setenv("AGENT_EVAL_BASE_URL", "https://unapproved.example/v1")
    raw = CountingRawClient([
        _response(tool_calls=[_call({"city": "上海", "days": 3, "job_level": "employee"})]),
        _response(content="上海 3 天预算 1500"),
    ])
    observed = {}

    def factory(api_key, **kwargs):
        observed.update(kwargs)
        return raw

    report = run_live_evaluation(
        api_key="test-key",
        client_factory=factory,
        cases=_live_case_definitions()[:1],
    )

    assert report["cases_run"] == 1
    assert observed["base_url"] == TRUSTED_LIVE_ENDPOINT
    assert raw.calls == 2


def test_custom_endpoint_with_non_https_scheme_is_rejected_even_after_opt_in():
    with pytest.raises(ValueError, match="HTTPS"):
        run_live_evaluation(
            api_key="test-key",
            base_url="http://custom.example/v1",
            allow_custom_endpoint=True,
            client_factory=lambda *args, **kwargs: pytest.fail("unsafe URL reached client construction"),
        )


def test_builtin_live_case_definitions_run_through_fake_client_and_report_latest_tool_values():
    raw = CountingRawClient([
        _response(tool_calls=[_call({"city": "上海", "days": 3, "job_level": "employee"})]),
        _response(content="上海 3 天预算 1500"),
        _response(content="请补充目的城市。"),
        _response(tool_calls=[_call({"city": "上海", "days": 3, "job_level": "employee"})]),
        _response(content="上海 3 天预算 1500"),
    ])
    report = run_live_evaluation(
        api_key="test-key",
        case_limit=2,
        max_turns=2,
        max_calls_per_case=3,
        max_total_calls=6,
        client_factory=lambda *args, **kwargs: raw,
    )

    assert report["cases_run"] == 2
    assert all(case["task_success"] for case in report["results"])
    assert report["results"][0]["turns"][0]["tool_calls"][0]["arguments"] == {
        "city": "上海", "days": 3, "job_level": "employee"
    }
    assert report["results"][1]["turns"][0]["tool_execution_count"] == 0
    assert report["results"][1]["turns"][1]["tool_calls"][0]["arguments"] == {
        "city": "上海", "days": 3, "job_level": "employee"
    }


def test_max_turns_and_case_limit_are_enforced_before_or_during_requests():
    cases = _live_case_definitions()
    untouched = CountingRawClient([_response(content="unused")])
    with pytest.raises(ValueError, match="max_turns"):
        run_live_evaluation(
            api_key="test-key",
            case_limit=2,
            max_turns=1,
            cases=cases,
            client_factory=lambda *args, **kwargs: pytest.fail("client created before max_turns validation"),
        )

    raw = CountingRawClient([_response(tool_calls=[_call({"city": "上海", "days": 3, "job_level": "employee"})]), _response(content="上海 3 天预算 1500")])
    report = run_live_evaluation(
        api_key="test-key",
        case_limit=1,
        client_factory=lambda *args, **kwargs: raw,
        cases=cases,
    )
    assert report["cases_requested"] == 1
    assert report["cases_run"] == 1
    assert len(report["results"]) == 1
    assert untouched.calls == 0


def test_sensitive_transcript_is_absent_by_default_and_requires_opt_in():
    marker = "SYNTHETIC_PRIVATE_SENTINEL"
    case = {"case_id": "privacy", "turns": [{"user_message": marker}]}
    raw = CountingRawClient([_response(content=marker)])
    report = run_live_evaluation(
        api_key="test-key", client_factory=lambda *args, **kwargs: raw, cases=[case]
    )
    assert marker not in json.dumps(report)

    raw_with_transcript = CountingRawClient([_response(content=marker)])
    expanded = run_live_evaluation(
        api_key="test-key",
        client_factory=lambda *args, **kwargs: raw_with_transcript,
        cases=[case],
        include_sensitive_transcript=True,
    )
    assert marker in json.dumps(expanded)


def test_agentloop_exception_log_and_live_report_redact_synthetic_secret(caplog):
    marker = "SYNTHETIC_API_KEY_AND_AUTHORIZATION_SENTINEL"
    raw = CountingRawClient([RuntimeError(marker)])
    case = {"case_id": "exception_redaction", "turns": [{"user_message": "hello"}]}
    report = run_live_evaluation(
        api_key="test-key", client_factory=lambda *args, **kwargs: raw, cases=[case]
    )

    assert report["results"][0]["task_success"] is False
    assert marker not in json.dumps(report)
    assert marker not in caplog.text
    assert "Traceback" not in caplog.text


def test_registry_tool_exception_log_suppresses_sensitive_details_but_keeps_diagnostic_type(
    monkeypatch, caplog
):
    from app.tools.travel import TravelBudgetTool

    marker = "SYNTHETIC_RAW_TOOL_RESPONSE_AND_SECRET"

    def fail_with_sensitive_detail(self, arguments):
        raise RuntimeError(f"provider response={marker}; api_key=synthetic")

    monkeypatch.setattr(TravelBudgetTool, "execute", fail_with_sensitive_detail)
    raw = CountingRawClient([
        _response(tool_calls=[_call({"city": "上海", "days": 1, "job_level": "employee"})]),
        _response(content="工具执行失败"),
    ])
    report = run_live_evaluation(
        api_key="test-key",
        client_factory=lambda *args, **kwargs: raw,
        cases=[_live_test_case([{"user_message": "计算差旅预算"}])],
    )

    assert raw.calls == 2
    assert report["results"][0]["tool_execution_count"] == 1
    assert marker not in caplog.text
    assert "synthetic" not in caplog.text
    assert "tool=calculate_travel_budget" in caplog.text
    assert "type=RuntimeError" in caplog.text
    assert "details_suppressed=true" in caplog.text


@pytest.mark.parametrize("raises", [False, True])
def test_temporary_log_filters_are_removed_immediately_after_agent_run(raises):
    agent_logger = logging.getLogger("app.agent.loop")
    registry_logger = logging.getLogger("app.tools.registry")

    class AgentStub:
        def run(self, messages, role):
            assert any(type(item).__name__ == "_SafeAgentLoopExceptionFilter" for item in agent_logger.filters)
            assert any(type(item).__name__ == "_SafeAgentLoopExceptionFilter" for item in registry_logger.filters)
            if raises:
                raise RuntimeError("synthetic failure")
            return "ok", None

    if raises:
        with pytest.raises(RuntimeError, match="synthetic failure"):
            _run_agent_with_temporary_log_filters(AgentStub(), [], "employee")
    else:
        assert _run_agent_with_temporary_log_filters(AgentStub(), [], "employee") == ("ok", None)

    assert not any(
        type(item).__name__ == "_SafeAgentLoopExceptionFilter"
        for item in agent_logger.filters
    )
    assert not any(
        type(item).__name__ == "_SafeAgentLoopExceptionFilter"
        for item in registry_logger.filters
    )


def test_client_initialization_failure_has_no_phantom_case_result():
    report = run_live_evaluation(
        api_key="test-key",
        client_factory=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("private detail")),
    )
    assert report["cases_run"] == 0
    assert report["results"] == []
    assert report["initialization_failure"]["failure_type"] == "RuntimeError"
