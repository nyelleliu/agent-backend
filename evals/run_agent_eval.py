"""Run deterministic business-contract checks without a live model."""
from __future__ import annotations

import json
import sys
import copy
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.loop import AgentLoop
from app.permissions.checker import PermissionChecker
from app.tools.travel import TravelBudgetTool


def _arguments(city, days, job_level, **overrides):
    return {"city": city, "days": days, "job_level": job_level, **overrides}


def _check_tool(case):
    """Execute the tool or its actual validation path for one fixture."""
    case_id = case["id"]
    if case_id == "missing_city":
        return _assert_invalid({"days": 3, "job_level": "employee"}, "missing required arguments")
    if case_id == "missing_days":
        return _assert_invalid({"city": "上海", "job_level": "employee"}, "missing required arguments")
    if case_id == "missing_level":
        return _assert_invalid({"city": "上海", "days": 3}, "missing required arguments")
    if case_id == "unknown_policy":
        return _assert_invalid(_arguments("火星", 3, "employee"), "unsupported city")
    if case_id == "negative_days":
        return _assert_invalid(_arguments("上海", -2, "employee"), "days must be")
    if case_id == "invalid_level":
        return _assert_invalid(_arguments("上海", 3, "intern"), "job_level must be")
    if case_id in {"empty", "policy_question"}:
        return _assert_invalid({}, "missing required arguments")

    args_by_id = {
        "complete_default": _arguments("上海", 3, "employee"),
        "complete_senior": _arguments("北京", 2, "senior"),
        "complete_manager": _arguments("深圳", 5, "manager"),
        "over_lodging": _arguments("上海", 3, "employee", lodging_per_day=400),
        "over_meal": _arguments("北京", 2, "senior", meal_per_day=200),
        "over_transport": _arguments("广州", 4, "manager", transport_total=1000),
        "explicit_actuals": _arguments("上海", 3, "employee", lodging_per_day=300, meal_per_day=80, transport_total=120),
        "zero_transport": _arguments("北京", 1, "employee", transport_total=0),
        "long_trip": _arguments("深圳", 30, "manager"),
    }
    if case_id in {"correct_city", "correct_days"}:
        return {
            "actual": "not_implemented",
            "passed": False,
            "reason": "MVP has no deterministic natural-language correction parser or multi-turn evaluator",
        }
    if case_id not in args_by_id:
        return {"actual": "unimplemented", "passed": False, "reason": "no executable check is defined for this case id"}

    try:
        result = json.loads(TravelBudgetTool().execute(args_by_id[case_id]))
    except Exception as exc:
        return {"actual": "error", "passed": False, "reason": f"tool raised {type(exc).__name__}: {exc}"}

    expected_behaviors = {
        "complete_default": ("complete", result["city"] == "上海" and result["actual_total"] == 1500),
        "complete_senior": ("complete", result["city"] == "北京" and result["policy_budget"] == 1550),
        "complete_manager": ("complete", result["city"] == "深圳" and result["policy_budget"] == 4900),
        "over_lodging": ("over_limit", result["over_limit_items"] == ["lodging_per_day"]),
        "over_meal": ("over_limit", result["over_limit_items"] == ["meal_per_day"]),
        "over_transport": ("over_limit", result["over_limit_items"] == ["transport_total"]),
        "explicit_actuals": ("complete", result["actual_total"] == 1260),
        "zero_transport": ("complete", result["actual_expenses"]["transport_total"] == 0),
        "long_trip": ("complete", result["days"] == 30),
    }
    actual, assertion = expected_behaviors[case_id]
    policy_is_marked_simulated = (
        result["policy_data_type"] == "simulated_mvp"
        and "模拟数据" in result["citation"]
        and "非真实企业制度" in result["citation"]
    )
    passed = assertion and policy_is_marked_simulated
    return {
        "actual": actual if passed else "assertion_failed",
        "passed": passed,
        "reason": "all expected output assertions passed" if passed else "calculation, output, or simulated-policy assertion failed",
    }


def _assert_invalid(arguments, expected_fragment):
    try:
        TravelBudgetTool().execute(arguments)
    except (TypeError, ValueError) as exc:
        passed = expected_fragment in str(exc)
        return {
            "actual": (
                "clarify"
                if "missing required arguments" in str(exc)
                or "unsupported city" in str(exc)
                else "invalid"
            ),
            "passed": passed,
            "reason": str(exc) if passed else f"expected error containing {expected_fragment!r}; got {exc!s}",
        }
    return {"actual": "accepted_invalid_input", "passed": False, "reason": "tool accepted input that should be invalid or incomplete"}


class _ToolCall:
    def __init__(self, name, arguments):
        self.id = "eval_permission_call"
        self.function = SimpleNamespace(name=name, arguments=json.dumps(arguments))

    def model_dump(self):
        return {"id": self.id, "type": "function", "function": {"name": self.function.name, "arguments": self.function.arguments}}


class _PermissionProbeClient:
    def __init__(self, arguments):
        self.calls = 0
        self.received_messages = []
        self.received_tools = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.arguments = arguments

    def create(self, **kwargs):
        self.calls += 1
        self.received_messages.append(copy.deepcopy(kwargs.get("messages", [])))
        self.received_tools.append(copy.deepcopy(kwargs.get("tools", [])))
        if self.calls == 1:
            message = SimpleNamespace(
                content=None,
                tool_calls=[_ToolCall("calculate_travel_budget", self.arguments)],
            )
        else:
            message = SimpleNamespace(content="probe complete", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _RecordingTravelBudgetTool(TravelBudgetTool):
    def __init__(self):
        self.execution_count = 0

    def execute(self, arguments):
        self.execution_count += 1
        return super().execute(arguments)


def _run_permission_probe(role):
    from app.tools.registry import ToolRegistry

    registry = ToolRegistry()
    tool = _RecordingTravelBudgetTool()
    registry.register(tool)
    arguments = {"city": "上海", "days": 1, "job_level": "employee"}
    client = _PermissionProbeClient(arguments)
    agent = AgentLoop(
        client=client,
        tool_registry=registry,
        permission_checker=PermissionChecker(),
        max_steps=2,
    )
    messages = [{"role": "user", "content": "调用差旅工具"}]
    original_messages = copy.deepcopy(messages)
    reply, _task_state = agent.run(messages, role=role)
    trace = client.received_messages[-1] if client.received_messages else []
    tool_replies = [item for item in trace if item.get("role") == "tool"]
    _validate_tool_protocol(client.received_messages)
    caller_history_unchanged = messages == original_messages
    tool_reply = tool_replies[0] if tool_replies else None
    denied = (
        tool is not None
        and tool.execution_count == 0
        and tool_reply is not None
        and "Permission denied" in tool_reply.get("content", "")
        and client.calls == 2
        and caller_history_unchanged
        and not client.received_tools[0]
    )
    authorized = (
        tool.execution_count == 1
        and tool_reply is not None
        and "\"actual_total\": 600" in tool_reply.get("content", "")
        and client.calls == 2
        and caller_history_unchanged
        and any(
            schema.get("function", {}).get("name") == tool.name
            for schema in client.received_tools[0]
        )
    )
    return {
        "role": role,
        "reply": reply,
        "tool_execution_count": tool.execution_count,
        "tool_reply": tool_reply,
        "client_calls": client.calls,
        "caller_history_unchanged": caller_history_unchanged,
        "received_tools": client.received_tools,
        "denied": denied,
        "authorized": authorized,
    }


def _validate_tool_protocol(requests):
    for request_index, messages in enumerate(requests):
        index = 0
        while index < len(messages):
            message = messages[index]
            if message.get("role") == "tool":
                raise AssertionError(f"request {request_index + 1}: orphan tool reply")
            calls = message.get("tool_calls") or []
            if not calls:
                index += 1
                continue
            if message.get("role") != "assistant" or not isinstance(calls, list):
                raise AssertionError("tool calls must be in an assistant message")
            replies = messages[index + 1:index + 1 + len(calls)]
            if len(replies) != len(calls):
                raise AssertionError("assistant tool calls are missing tool replies")
            for call, reply in zip(calls, replies):
                if (
                    reply.get("role") != "tool"
                    or reply.get("tool_call_id") != call.get("id")
                ):
                    raise AssertionError("tool reply does not match assistant call")
            index += 1 + len(calls)


def _check_permission_boundary():
    observation = _run_permission_probe("guest")
    denied = observation["denied"]
    return {
        "actual": "deny" if denied else "executed_or_not_denied",
        "passed": denied,
        "reason": (
            "AgentLoop rejected the forged calculate_travel_budget call, "
            "returned a matching denial tool reply, did not execute the tool, "
            "and preserved caller history"
            if denied
            else "AgentLoop did not enforce or expose the permission denial safely"
        ),
    }
def check_case(case):
    if case["id"] == "permission_boundary":
        return _check_permission_boundary()
    return _check_tool(case)


def main():
    path = Path(__file__).with_name("agent_cases.jsonl")
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [case.get("id") for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("agent_cases.jsonl contains duplicate case ids")

    results = []
    for case in cases:
        checked = check_case(case)
        expected = case["expected"]
        if checked["actual"] != expected:
            if checked["passed"]:
                checked["reason"] = f"actual behavior {checked['actual']!r} does not match expected {expected!r}"
            checked["passed"] = False
        results.append({"id": case["id"], "expected": expected, **checked})

    passed = sum(result["passed"] for result in results)
    report = {
        "evaluation_type": "travel_tool_contract",
        "natural_language_understanding_evaluated": False,
        "cases": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "success_rate": passed / len(results) if results else 0,
        "results": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
