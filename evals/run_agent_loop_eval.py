"""Offline AgentLoop evaluation with scripted, usage-aware fake responses."""
from __future__ import annotations

import copy
import json
import argparse
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.loop import AgentLoop
from app.permissions.checker import PermissionChecker
from app.tools import create_default_registry
from app.tools.registry import ToolRegistry


TRUSTED_LIVE_ENDPOINT = "https://api.deepseek.com"
MAX_REQUEST_TIMEOUT_SECONDS = 120.0
MAX_TOTAL_TIMEOUT_SECONDS = 600.0


class _SafeAgentLoopExceptionFilter(logging.Filter):
    """Suppress exception details from AgentLoop and registry logs during evaluation."""

    def filter(self, record):
        if record.name == "app.tools.registry":
            # Tool errors are already passed through the shared best-effort
            # secret redactor, but arbitrary exception text may contain raw
            # tool input or provider responses. Keep only useful identifiers.
            args = record.args if isinstance(record.args, tuple) else ()
            tool_name = args[0] if len(args) > 0 else "-"
            exception_type = args[1] if len(args) > 1 else "Exception"
            record.msg = "evaluation_tool_error tool=%s type=%s details_suppressed=true"
            record.args = (tool_name, exception_type)
        if record.exc_info:
            record.msg = "AgentLoop failed during isolated live evaluation"
            record.args = ()
            record.exc_info = None
            record.exc_text = None
        return True


def _run_agent_with_temporary_log_filters(agent, messages, role):
    agent_loop_logger = logging.getLogger("app.agent.loop")
    registry_logger = logging.getLogger("app.tools.registry")
    safe_log_filter = _SafeAgentLoopExceptionFilter()
    agent_loop_logger.addFilter(safe_log_filter)
    registry_logger.addFilter(safe_log_filter)
    try:
        return agent.run(messages, role=role)
    finally:
        agent_loop_logger.removeFilter(safe_log_filter)
        registry_logger.removeFilter(safe_log_filter)


def _validate_live_endpoint(endpoint: str, allow_custom_endpoint: bool) -> str:
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("live endpoint must be a non-empty HTTPS URL")
    if endpoint != TRUSTED_LIVE_ENDPOINT and not allow_custom_endpoint:
        raise ValueError("custom endpoint requires --allow-custom-endpoint")
    try:
        parsed = urlsplit(endpoint)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        raise ValueError("invalid live endpoint URL") from None
    if (
        parsed.scheme != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ValueError("live endpoint must be a valid HTTPS URL without credentials, query, or fragment")
    return endpoint.rstrip("/")


@dataclass
class CaseMetrics:
    case_id: str
    task_success: bool = False
    result_category: str | None = None
    tool_execution_count: int = 0
    llm_call_attempt_count: int = 0
    duration_ms: float = 0.0
    token_usage_status: str = "unavailable"
    token_usage_error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    failure_reason: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TurnMetrics:
    turn_id: str
    user_message: str
    task_success: bool = False
    tool_execution_count: int = 0
    llm_call_attempt_count: int = 0
    duration_ms: float = 0.0
    token_usage_status: str = "unavailable"
    token_usage_error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    failure_reason: str | None = None
    history_before: list[dict[str, Any]] = field(default_factory=list)
    issued_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_messages: list[dict[str, Any]] = field(default_factory=list)
    reply: str | None = None


class FakeToolCall:
    def __init__(self, name: str, arguments: dict[str, Any], call_id: str):
        self.id = call_id
        self.function = SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        )

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


def _response(content=None, tool_calls=None, usage=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    response_fields = {"choices": [SimpleNamespace(message=message)]}
    if usage is not None:
        response_fields["usage"] = SimpleNamespace(**usage)
    return SimpleNamespace(**response_fields)


class ScriptedFakeClient:
    """Return scripted responses and count every create() invocation attempt."""

    def __init__(self, responses: list[Any]):
        self._responses = list(responses)
        self.create_call_attempt_count = 0
        self.usage_values: list[tuple[int, int, int] | None] = []
        self.usage_errors: list[str | None] = []
        self.issued_tool_calls: list[dict[str, Any]] = []
        self.received_messages: list[list[dict[str, Any]]] = []
        self.message_traces: list[list[dict[str, Any]]] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        self.create_call_attempt_count += 1
        request_messages = kwargs.get("messages", [])
        self.received_messages.append(copy.deepcopy(request_messages))
        # Retain the AgentLoop's working list as a trace. The caller's input
        # history is isolated; AgentLoop appends assistant/tool exchanges to
        # this internal list after the Fake Client returns its response.
        self.message_traces.append(request_messages)
        response_index = self.create_call_attempt_count - 1
        if response_index >= len(self._responses):
            raise RuntimeError("fake client ran out of scripted responses")

        response = self._responses[response_index]
        if isinstance(response, BaseException):
            raise response

        message = response.choices[0].message
        for tool_call in message.tool_calls or []:
            try:
                arguments = json.loads(tool_call.function.arguments)
            except (TypeError, json.JSONDecodeError):
                arguments = tool_call.function.arguments
            self.issued_tool_calls.append({
                "name": tool_call.function.name,
                "arguments": arguments,
            })

        usage = getattr(response, "usage", None)
        values, error = _validated_usage(usage)
        self.usage_values.append(values)
        self.usage_errors.append(error)
        return response


def _validated_usage(usage):
    if usage is None:
        return None, None

    names = ("prompt_tokens", "completion_tokens", "total_tokens")
    values = tuple(getattr(usage, name, None) for name in names)
    if any(value is None for value in values):
        return None, None
    if any(type(value) is not int or value < 0 for value in values):
        return None, "usage fields must be non-negative integers"

    prompt_tokens, completion_tokens, total_tokens = values
    if total_tokens != prompt_tokens + completion_tokens:
        return None, (
            "usage total_tokens does not equal prompt_tokens + completion_tokens"
        )
    return values, None


class RecordingToolRegistry:
    """Record AgentLoop dispatches while delegating to the real registry."""

    def __init__(self, delegate: ToolRegistry, metrics: CaseMetrics):
        self.delegate = delegate
        self.metrics = metrics

    def schemas(self):
        return self.delegate.schemas()

    def get(self, name):
        return self.delegate.get(name)

    def execute(self, name, raw_arguments=None):
        if isinstance(raw_arguments, str):
            try:
                parsed_arguments = json.loads(raw_arguments) if raw_arguments else {}
            except json.JSONDecodeError:
                parsed_arguments = {"_raw_arguments": raw_arguments}
        else:
            parsed_arguments = raw_arguments

        self.metrics.tool_execution_count += 1
        record = {"name": name, "arguments": parsed_arguments}
        result = self.delegate.execute(name, raw_arguments)
        record["result"] = result
        self.metrics.tool_calls.append(record)
        return result


def _call(arguments: dict[str, Any], call_id="travel-1"):
    return FakeToolCall("calculate_travel_budget", arguments, call_id)


def _scenario_definitions():
    travel_name = "calculate_travel_budget"
    return [
        {
            "case_id": "travel_success",
            "role": "employee",
            "responses": [
                _response(
                    tool_calls=[_call({"city": "上海", "days": 3, "job_level": "employee"})],
                    usage={"prompt_tokens": 120, "completion_tokens": 24, "total_tokens": 144},
                ),
                _response(
                    content="上海 employee 三天模拟预算总额为 1500 元。",
                    usage={"prompt_tokens": 180, "completion_tokens": 32, "total_tokens": 212},
                ),
            ],
            "expected": {
                "tool_execution_count": 1,
                "llm_call_attempt_count": 2,
                "issued_tool_calls": [{"name": travel_name, "arguments": {"city": "上海", "days": 3, "job_level": "employee"}}],
                "executed_tool_calls": [{"name": travel_name, "arguments": {"city": "上海", "days": 3, "job_level": "employee"}}],
                "tool_message_count": 1,
                "tool_result_json": [{"actual_total": 1500}],
                "reply_contains": ["1500"],
            },
        },
        {
            "case_id": "missing_city_no_execution",
            "role": "employee",
            "responses": [_response(content="请补充目的城市后再计算预算。")],
            "expected": {
                "tool_execution_count": 0,
                "llm_call_attempt_count": 1,
                "issued_tool_calls": [],
                "executed_tool_calls": [],
                "tool_message_count": 0,
                "reply_contains": ["城市"],
            },
        },
        {
            "case_id": "unknown_city_rejected",
            "role": "employee",
            "responses": [
                _response(tool_calls=[_call({"city": "杭州", "days": 2, "job_level": "employee"})]),
                _response(content="当前模拟政策未覆盖该城市，暂不能计算预算。"),
            ],
            "expected": {
                "tool_execution_count": 1,
                "llm_call_attempt_count": 2,
                "issued_tool_calls": [{"name": travel_name, "arguments": {"city": "杭州", "days": 2, "job_level": "employee"}}],
                "executed_tool_calls": [{"name": travel_name, "arguments": {"city": "杭州", "days": 2, "job_level": "employee"}}],
                "tool_message_count": 1,
                "tool_result_contains": [{"index": 0, "text": "unsupported city"}],
                "tool_message_contains": [{"index": 0, "text": "unsupported city"}],
                "reply_contains": ["不能计算"],
            },
        },
        {
            "case_id": "invalid_days_rejected",
            "role": "employee",
            "responses": [
                _response(tool_calls=[_call({"city": "上海", "days": "2", "job_level": "employee"})]),
                _response(content="天数参数无效，无法计算预算。"),
            ],
            "expected": {
                "tool_execution_count": 1,
                "llm_call_attempt_count": 2,
                "issued_tool_calls": [{"name": travel_name, "arguments": {"city": "上海", "days": "2", "job_level": "employee"}}],
                "executed_tool_calls": [{"name": travel_name, "arguments": {"city": "上海", "days": "2", "job_level": "employee"}}],
                "tool_message_count": 1,
                "tool_result_contains": [{"index": 0, "text": "days must be an integer"}],
                "tool_message_contains": [{"index": 0, "text": "days must be an integer"}],
                "reply_contains": ["参数无效"],
            },
        },
        {
            "case_id": "unauthorized_role_denied",
            "role": "guest",
            "responses": [
                _response(tool_calls=[_call({"city": "上海", "days": 1, "job_level": "employee"})]),
                _response(content="当前角色没有权限执行差旅计算。"),
            ],
            "expected": {
                "tool_execution_count": 0,
                "llm_call_attempt_count": 2,
                "issued_tool_calls": [{"name": travel_name, "arguments": {"city": "上海", "days": 1, "job_level": "employee"}}],
                "executed_tool_calls": [],
                "tool_message_count": 1,
                "tool_message_contains": [{"index": 0, "text": "Permission denied"}],
                "reply_contains": ["没有权限"],
            },
        },
        {
            "case_id": "tool_call_budget_exhausted",
            "role": "employee",
            "max_tool_calls": 1,
            "responses": [
                _response(tool_calls=[_call({"city": "上海", "days": 1, "job_level": "employee"}, "travel-1")]),
                _response(tool_calls=[_call({"city": "北京", "days": 1, "job_level": "employee"}, "travel-2")]),
            ],
            "expected": {
                "tool_execution_count": 1,
                "llm_call_attempt_count": 2,
                "issued_tool_calls": [
                    {"name": travel_name, "arguments": {"city": "上海", "days": 1, "job_level": "employee"}},
                    {"name": travel_name, "arguments": {"city": "北京", "days": 1, "job_level": "employee"}},
                ],
                "executed_tool_calls": [{"name": travel_name, "arguments": {"city": "上海", "days": 1, "job_level": "employee"}}],
                "tool_message_count": 1,
                "reply_contains": ["tool-call budget exceeded"],
            },
        },
        {
            "case_id": "invalid_tool_call_rejected",
            "role": "employee",
            "responses": [
                _response(tool_calls=[_call(
                    {"city": "上海", "days": 1, "job_level": "employee"},
                    call_id=" ",
                )]),
            ],
            "expected": {
                "tool_execution_count": 0,
                "llm_call_attempt_count": 1,
                "issued_tool_calls": [{
                    "name": travel_name,
                    "arguments": {"city": "上海", "days": 1, "job_level": "employee"},
                }],
                "executed_tool_calls": [],
                "tool_message_count": 0,
                "reply_equals": "Error: agent tool calls are invalid; unable to continue safely.",
            },
            "result_category": "invalid_tool_call",
        },
    ]


def _multiturn_scenario_definitions():
    """Multi-turn conversations scripted turn-by-turn through AgentLoop."""
    tool = "calculate_travel_budget"

    def call(city, days, level, call_id):
        return _call({"city": city, "days": days, "job_level": level}, call_id)

    return [
        {
            "case_id": "multiturn_add_city",
            "role": "employee",
            "turns": [
                {"turn_id": "ask_city", "user_message": "出差 3 天，职级 employee。", "responses": [_response(content="请补充出差城市。")], "expected": {"tool_execution_count": 0, "issued_tool_calls": [], "executed_tool_calls": [], "tool_message_count": 0, "reply_contains": ["城市"]}},
                {"turn_id": "provide_city", "user_message": "去上海。", "history_contains": ["出差 3 天，职级 employee。", "请补充出差城市。"], "responses": [_response(tool_calls=[call("上海", 3, "employee", "city-1")]), _response(content="上海 3 天预算为 1500 元。")], "expected": {"tool_execution_count": 1, "issued_tool_calls": [{"name": tool, "arguments": {"city": "上海", "days": 3, "job_level": "employee"}}], "executed_tool_calls": [{"name": tool, "arguments": {"city": "上海", "days": 3, "job_level": "employee"}}], "tool_message_count": 1, "tool_result_json": [{"city": "上海", "days": 3, "job_level": "employee", "actual_total": 1500}], "reply_contains": ["1500"]}},
            ],
        },
        {
            "case_id": "multiturn_add_days",
            "role": "employee",
            "turns": [
                {"turn_id": "ask_days", "user_message": "去北京，职级 senior。", "responses": [_response(content="请补充出差天数。")], "expected": {"tool_execution_count": 0, "issued_tool_calls": [], "executed_tool_calls": [], "tool_message_count": 0, "reply_contains": ["天数"]}},
                {"turn_id": "provide_days", "user_message": "需要 2 天。", "history_contains": ["去北京，职级 senior。", "请补充出差天数。"], "responses": [_response(tool_calls=[call("北京", 2, "senior", "days-1")]), _response(content="北京 2 天预算为 1550 元。")], "expected": {"tool_execution_count": 1, "issued_tool_calls": [{"name": tool, "arguments": {"city": "北京", "days": 2, "job_level": "senior"}}], "executed_tool_calls": [{"name": tool, "arguments": {"city": "北京", "days": 2, "job_level": "senior"}}], "tool_message_count": 1, "tool_result_json": [{"city": "北京", "days": 2, "job_level": "senior", "actual_total": 1550}], "reply_contains": ["1550"]}},
            ],
        },
        {
            "case_id": "multiturn_add_job_level",
            "role": "employee",
            "turns": [
                {"turn_id": "ask_level", "user_message": "去深圳 2 天。", "responses": [_response(content="请补充职级。")], "expected": {"tool_execution_count": 0, "issued_tool_calls": [], "executed_tool_calls": [], "tool_message_count": 0, "reply_contains": ["职级"]}},
                {"turn_id": "provide_level", "user_message": "我的职级是 manager。", "history_contains": ["去深圳 2 天。", "请补充职级。"], "responses": [_response(tool_calls=[call("深圳", 2, "manager", "level-1")]), _response(content="深圳 2 天预算为 2200 元。")], "expected": {"tool_execution_count": 1, "issued_tool_calls": [{"name": tool, "arguments": {"city": "深圳", "days": 2, "job_level": "manager"}}], "executed_tool_calls": [{"name": tool, "arguments": {"city": "深圳", "days": 2, "job_level": "manager"}}], "tool_message_count": 1, "tool_result_json": [{"city": "深圳", "days": 2, "job_level": "manager", "actual_total": 2200}], "reply_contains": ["2200"]}},
            ],
        },
        {
            "case_id": "multiturn_correct_days",
            "role": "employee",
            "turns": [
                {"turn_id": "initial_days", "user_message": "去上海 3 天，employee。", "responses": [_response(tool_calls=[call("上海", 3, "employee", "initial-1")]), _response(content="上海 3 天预算为 1500 元。")], "expected": {"tool_execution_count": 1, "issued_tool_calls": [{"name": tool, "arguments": {"city": "上海", "days": 3, "job_level": "employee"}}], "executed_tool_calls": [{"name": tool, "arguments": {"city": "上海", "days": 3, "job_level": "employee"}}], "tool_message_count": 1, "tool_result_json": [{"days": 3, "actual_total": 1500}]}},
                {"turn_id": "correct_days", "user_message": "更正为 5 天。", "history_contains": ["去上海 3 天，employee。", "上海 3 天预算为 1500 元。"], "responses": [_response(tool_calls=[call("上海", 5, "employee", "corrected-1")]), _response(content="按更正后的 5 天，上海预算为 2400 元。")], "expected": {"tool_execution_count": 1, "issued_tool_calls": [{"name": tool, "arguments": {"city": "上海", "days": 5, "job_level": "employee"}}], "executed_tool_calls": [{"name": tool, "arguments": {"city": "上海", "days": 5, "job_level": "employee"}}], "tool_message_count": 1, "tool_result_json": [{"city": "上海", "days": 5, "job_level": "employee", "actual_total": 2400}], "reply_contains": ["5 天", "2400"]}},
            ],
        },
        {
            "case_id": "multiturn_correct_unsupported_city",
            "role": "employee",
            "turns": [
                {"turn_id": "initial_city", "user_message": "去上海 2 天，employee。", "responses": [_response(tool_calls=[call("上海", 2, "employee", "initial-1")]), _response(content="上海 2 天预算为 1050 元。")], "expected": {"tool_execution_count": 1, "issued_tool_calls": [{"name": tool, "arguments": {"city": "上海", "days": 2, "job_level": "employee"}}], "executed_tool_calls": [{"name": tool, "arguments": {"city": "上海", "days": 2, "job_level": "employee"}}], "tool_message_count": 1, "tool_result_json": [{"city": "上海", "days": 2, "actual_total": 1050}]}},
                {"turn_id": "correct_city", "user_message": "更正目的地为杭州。", "history_contains": ["去上海 2 天，employee。", "上海 2 天预算为 1050 元。"], "responses": [_response(tool_calls=[call("杭州", 2, "employee", "corrected-1")]), _response(content="杭州不在当前模拟政策覆盖范围，不能沿用上海预算。")], "expected": {"tool_execution_count": 1, "issued_tool_calls": [{"name": tool, "arguments": {"city": "杭州", "days": 2, "job_level": "employee"}}], "executed_tool_calls": [{"name": tool, "arguments": {"city": "杭州", "days": 2, "job_level": "employee"}}], "tool_message_count": 1, "tool_result_contains": [{"index": 0, "text": "unsupported city"}], "tool_message_contains": [{"index": 0, "text": "unsupported city"}], "reply_contains": ["不能沿用上海预算"]}},
            ],
        },
    ]


def _tool_messages(messages):
    return [message for message in messages if message.get("role") == "tool"]


def _require(condition: bool, message: str) -> None:
    """A validation primitive that remains active under ``python -O``."""
    if not condition:
        raise AssertionError(message)


def _validate_tool_message_protocol(
    messages: list[dict[str, Any]],
    *,
    allow_incomplete_terminal_group: bool = False,
    context: str = "request",
) -> None:
    """Validate assistant tool calls and their ordered tool replies."""
    index = 0
    while index < len(messages):
        message = messages[index]
        _require(isinstance(message, dict), f"{context}: message must be an object")

        if message.get("role") == "tool":
            raise AssertionError(f"{context}: orphan tool reply at message {index}")

        calls = message.get("tool_calls")
        if not calls:
            index += 1
            continue

        _require(
            message.get("role") == "assistant" and isinstance(calls, list),
            f"{context}: tool_calls must belong to an assistant message",
        )
        call_ids = [
            call.get("id") if isinstance(call, dict) else None
            for call in calls
        ]
        _require(
            all(isinstance(call_id, str) and call_id for call_id in call_ids),
            f"{context}: tool call id is missing",
        )
        _require(
            len(set(call_ids)) == len(call_ids),
            f"{context}: duplicate tool call id",
        )

        response_start = index + 1
        response_end = response_start + len(call_ids)
        responses = messages[response_start:response_end]
        if len(responses) < len(call_ids):
            terminal = response_end >= len(messages)
            _require(
                allow_incomplete_terminal_group and terminal,
                f"{context}: tool call is missing a reply",
            )

        for call_id, response in zip(call_ids, responses):
            _require(
                isinstance(response, dict)
                and response.get("role") == "tool"
                and response.get("tool_call_id") == call_id,
                f"{context}: tool reply does not match call id {call_id!r}",
            )

        if len(responses) < len(call_ids):
            return
        index = response_end


def _validate_client_request_protocol(client: ScriptedFakeClient, context: str) -> None:
    for request_index, messages in enumerate(client.received_messages):
        _validate_tool_message_protocol(
            messages,
            context=f"{context} request {request_index + 1}",
        )


def _working_message_trace(client: ScriptedFakeClient) -> list[dict[str, Any]]:
    return client.message_traces[-1] if client.message_traces else []


def _assert_case(case, reply, messages, metrics, client):
    expected = case["expected"]
    tool_messages = _tool_messages(messages)
    _validate_client_request_protocol(client, case["case_id"])
    _validate_tool_message_protocol(
        messages,
        allow_incomplete_terminal_group=True,
        context=f"{case['case_id']} execution trace",
    )
    executed_calls = [
        {"name": call["name"], "arguments": call["arguments"]}
        for call in metrics.tool_calls
    ]

    _require(
        metrics.tool_execution_count == expected["tool_execution_count"],
        f"expected tool_execution_count={expected['tool_execution_count']}, got {metrics.tool_execution_count}",
    )
    _require(
        client.create_call_attempt_count == expected["llm_call_attempt_count"],
        f"expected llm_call_attempt_count={expected['llm_call_attempt_count']}, got {client.create_call_attempt_count}",
    )
    _require(
        client.issued_tool_calls == expected["issued_tool_calls"],
        f"issued tool calls differ from expected: {client.issued_tool_calls!r}",
    )
    _require(
        executed_calls == expected["executed_tool_calls"],
        f"executed tool calls differ from expected: {executed_calls!r}",
    )
    _require(
        len(tool_messages) == expected["tool_message_count"],
        f"expected {expected['tool_message_count']} tool messages, got {len(tool_messages)}",
    )

    for index, expected_fields in enumerate(expected.get("tool_result_json", [])):
        _require(index < len(metrics.tool_calls), f"missing tool result at index {index}")
        try:
            actual_result = json.loads(metrics.tool_calls[index]["result"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise AssertionError(f"tool result {index} is not valid JSON: {exc}") from exc
        for field_name, expected_value in expected_fields.items():
            _require(
                actual_result.get(field_name) == expected_value,
                f"tool result field {field_name!r}: expected {expected_value!r}, got {actual_result.get(field_name)!r}",
            )

    for check_name, values in (
        ("tool_result_contains", [call["result"] for call in metrics.tool_calls]),
        ("tool_message_contains", [message["content"] for message in tool_messages]),
    ):
        for check in expected.get(check_name, []):
            index = check["index"]
            _require(index < len(values), f"missing {check_name} item at index {index}")
            _require(
                check["text"] in values[index],
                f"expected {check_name}[{index}] to contain {check['text']!r}",
            )

    for text in expected.get("reply_contains", []):
        _require(text in reply, f"final reply did not contain {text!r}: {reply!r}")
    if "reply_equals" in expected:
        _require(reply == expected["reply_equals"],
                 f"final reply expected {expected['reply_equals']!r}, got {reply!r}")


def _validate_turn(turn, reply, messages, metrics, client, input_messages, history_before):
    _require(
        bool(input_messages) and input_messages[-1].get("role") == "user"
        and input_messages[-1].get("content") == turn["user_message"],
        f"turn {turn['turn_id']}: current user message missing or mismatched",
    )
    for expected_text in turn.get("history_contains", []):
        _require(
            any(expected_text in str(message.get("content", "")) for message in input_messages[:-1]),
            f"turn {turn['turn_id']}: AgentLoop input history missing {expected_text!r}",
        )

    expected = turn["expected"]
    _validate_client_request_protocol(client, turn["turn_id"])
    execution_messages = _working_message_trace(client)
    _validate_tool_message_protocol(
        execution_messages,
        allow_incomplete_terminal_group=True,
        context=f"{turn['turn_id']} execution trace",
    )
    tool_messages = _tool_messages(execution_messages)
    executed_calls = [
        {"name": call["name"], "arguments": call["arguments"]}
        for call in metrics.tool_calls
    ]
    _require(metrics.tool_execution_count == expected["tool_execution_count"],
             f"turn {turn['turn_id']}: expected tool executions {expected['tool_execution_count']}, got {metrics.tool_execution_count}")
    _require(client.issued_tool_calls == expected["issued_tool_calls"],
             f"turn {turn['turn_id']}: issued tool parameters differ: {client.issued_tool_calls!r}")
    _require(executed_calls == expected["executed_tool_calls"],
             f"turn {turn['turn_id']}: registry execution parameters differ: {executed_calls!r}")
    _require(len(tool_messages) == expected["tool_message_count"],
             f"turn {turn['turn_id']}: expected {expected['tool_message_count']} new tool messages, got {len(tool_messages)}")

    for index, expected_fields in enumerate(expected.get("tool_result_json", [])):
        _require(index < len(metrics.tool_calls), f"turn {turn['turn_id']}: missing tool result {index}")
        try:
            result = json.loads(metrics.tool_calls[index]["result"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise AssertionError(f"turn {turn['turn_id']}: tool result {index} is not JSON: {exc}") from exc
        for field_name, expected_value in expected_fields.items():
            _require(result.get(field_name) == expected_value,
                     f"turn {turn['turn_id']}: result {field_name} expected {expected_value!r}, got {result.get(field_name)!r}")

    for check_name, values in (
        ("tool_result_contains", [call["result"] for call in metrics.tool_calls]),
        ("tool_message_contains", [message["content"] for message in tool_messages]),
    ):
        for check in expected.get(check_name, []):
            _require(check["index"] < len(values), f"turn {turn['turn_id']}: missing {check_name}[{check['index']}]")
            _require(check["text"] in values[check["index"]],
                     f"turn {turn['turn_id']}: {check_name}[{check['index']}] missing {check['text']!r}")

    for text in expected.get("reply_contains", []):
        _require(text in (reply or ""), f"turn {turn['turn_id']}: reply did not contain {text!r}: {reply!r}")


def _finalize_token_metrics(metrics: CaseMetrics, client: ScriptedFakeClient) -> None:
    errors = [error for error in client.usage_errors if error]
    if errors:
        metrics.token_usage_status = "invalid"
        metrics.token_usage_error = "; ".join(errors)
        return

    if client.usage_values and all(value is not None for value in client.usage_values):
        metrics.token_usage_status = "available_from_fake_response"
        metrics.prompt_tokens = sum(value[0] for value in client.usage_values)
        metrics.completion_tokens = sum(value[1] for value in client.usage_values)
        metrics.total_tokens = sum(value[2] for value in client.usage_values)
        return

    metrics.token_usage_status = "unavailable"
    metrics.token_usage_error = None


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    """Run one isolated scripted scenario through the real AgentLoop."""
    if "turns" in case:
        return _run_multiturn_case(case)

    metrics = CaseMetrics(case_id=case["case_id"])
    client = ScriptedFakeClient(case["responses"])
    real_registry = create_default_registry()
    registry = RecordingToolRegistry(real_registry, metrics)
    agent = AgentLoop(
        client=client,
        tool_registry=registry,
        permission_checker=PermissionChecker(),
        max_steps=max(1, len(case["responses"])),
        max_tool_calls=case.get("max_tool_calls", 12),
    )
    messages = [{"role": "user", "content": case["case_id"]}]
    input_messages = copy.deepcopy(messages)
    started = time.perf_counter()
    try:
        reply, _task_state = agent.run(messages, role=case["role"])
        metrics.duration_ms = round((time.perf_counter() - started) * 1000, 3)
        _require(
            messages == input_messages,
            f"{case['case_id']}: AgentLoop modified caller history",
        )
        _assert_case(case, reply, _working_message_trace(client), metrics, client)
        metrics.result_category = case.get("result_category")
        metrics.task_success = True
    except Exception as exc:
        metrics.failure_reason = f"{type(exc).__name__}: {exc}"
    finally:
        metrics.llm_call_attempt_count = client.create_call_attempt_count
        if metrics.duration_ms == 0.0:
            metrics.duration_ms = round((time.perf_counter() - started) * 1000, 3)
        _finalize_token_metrics(metrics, client)

    return asdict(metrics)


def _run_multiturn_case(case: dict[str, Any]) -> dict[str, Any]:
    case_metrics = CaseMetrics(case_id=case["case_id"])
    messages: list[dict[str, Any]] = []
    turn_results = []
    case_started = time.perf_counter()

    for turn in case["turns"]:
        turn_metrics = CaseMetrics(case_id=f"{case['case_id']}:{turn['turn_id']}")
        client = ScriptedFakeClient(turn["responses"])
        registry = RecordingToolRegistry(create_default_registry(), turn_metrics)
        agent = AgentLoop(
            client=client,
            tool_registry=registry,
            permission_checker=PermissionChecker(),
            max_steps=max(1, len(turn["responses"])),
            max_tool_calls=case.get("max_tool_calls", 12),
        )
        messages.append({"role": "user", "content": turn["user_message"]})
        input_messages = copy.deepcopy(messages)
        history_before = copy.deepcopy(messages[:-1])
        started = time.perf_counter()
        reply = None
        try:
            reply, _task_state = agent.run(messages, role=case["role"])
            turn_metrics.duration_ms = round((time.perf_counter() - started) * 1000, 3)
            _require(
                messages == input_messages,
                f"turn {turn['turn_id']}: AgentLoop modified caller history",
            )
            _validate_turn(turn, reply, messages, turn_metrics, client, input_messages, history_before)
            turn_metrics.task_success = True
        except Exception as exc:
            turn_metrics.failure_reason = f"{type(exc).__name__}: {exc}"
        finally:
            turn_metrics.llm_call_attempt_count = client.create_call_attempt_count
            if turn_metrics.duration_ms == 0.0:
                turn_metrics.duration_ms = round((time.perf_counter() - started) * 1000, 3)
            _finalize_token_metrics(turn_metrics, client)
            turn_result = asdict(turn_metrics)
            turn_result["issued_tool_calls"] = list(client.issued_tool_calls)
            turn_result["history_before"] = input_messages
            turn_result["tool_messages"] = [
                dict(message)
                for message in _tool_messages(_working_message_trace(client))
            ]
            turn_result["reply"] = reply
            turn_result["user_message"] = turn["user_message"]
            turn_results.append(turn_result)

        if turn_metrics.task_success and not any(
            message.get("role") == "assistant"
            and message.get("content") == reply
            for message in messages[len(input_messages):]
        ):
            # /chat persists the returned assistant reply after AgentLoop.run();
            # reproduce that persistence boundary for the next fake turn.
            messages.append({"role": "assistant", "content": reply})

        case_metrics.tool_execution_count += turn_metrics.tool_execution_count
        case_metrics.llm_call_attempt_count += turn_metrics.llm_call_attempt_count
        case_metrics.tool_calls.extend(turn_metrics.tool_calls)
        case_metrics.duration_ms += turn_metrics.duration_ms
        if turn_metrics.token_usage_status == "invalid":
            case_metrics.token_usage_status = "invalid"
            case_metrics.token_usage_error = turn_metrics.token_usage_error
        elif turn_metrics.token_usage_status != "available_from_fake_response" and case_metrics.token_usage_status != "invalid":
            case_metrics.token_usage_status = "unavailable"
        elif case_metrics.token_usage_status != "invalid":
            case_metrics.token_usage_status = "available_from_fake_response"
            case_metrics.prompt_tokens = (case_metrics.prompt_tokens or 0) + (turn_metrics.prompt_tokens or 0)
            case_metrics.completion_tokens = (case_metrics.completion_tokens or 0) + (turn_metrics.completion_tokens or 0)
            case_metrics.total_tokens = (case_metrics.total_tokens or 0) + (turn_metrics.total_tokens or 0)
        if not turn_metrics.task_success and case_metrics.failure_reason is None:
            case_metrics.failure_reason = f"turn {turn['turn_id']}: {turn_metrics.failure_reason}"

    case_metrics.duration_ms = round((time.perf_counter() - case_started) * 1000, 3)
    case_metrics.task_success = all(turn["task_success"] for turn in turn_results)
    result = asdict(case_metrics)
    result["turns"] = turn_results
    result["final_messages"] = messages
    return result


def run_evaluation() -> dict[str, Any]:
    results = [run_case(case) for case in (_scenario_definitions() + _multiturn_scenario_definitions())]
    passed = sum(result["task_success"] for result in results)
    return {
        "cases": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "success_rate": passed / len(results) if results else 0.0,
        "results": results,
    }


class LiveBudgetStop(RuntimeError):
    """Raised before an LLM request when a configured budget is exhausted."""


class BudgetedLiveClient:
    """Count and gate every client create attempt before forwarding it."""

    def __init__(self, client, *, case_budget, turn_budget, global_budget, deadline, request_timeout):
        self.client = client
        self.case_budget = case_budget
        self.turn_budget = turn_budget
        self.global_budget = global_budget
        self.deadline = deadline
        self.request_timeout = request_timeout
        self.case_attempts = 0
        self.turn_attempts = 0
        self.case_budget_used = 0
        self.turn_budget_used = 0
        self.usage_records: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        self.case_attempts += 1
        self.turn_attempts += 1
        self.global_budget["attempts"] += 1
        if time.monotonic() >= self.deadline:
            raise LiveBudgetStop("total_runtime_budget_exhausted")
        if self.case_budget_used >= self.case_budget:
            raise LiveBudgetStop("per_case_llm_call_budget_exhausted")
        if self.turn_budget_used >= self.turn_budget:
            raise LiveBudgetStop("per_turn_llm_call_budget_exhausted")
        if self.global_budget["used"] >= self.global_budget["limit"]:
            raise LiveBudgetStop("global_llm_call_budget_exhausted")

        self.case_budget_used += 1
        self.turn_budget_used += 1
        self.global_budget["used"] += 1
        kwargs["timeout"] = min(self.request_timeout, max(0.001, self.deadline - time.monotonic()))
        started = time.monotonic()
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            self.usage_records.append({
                "status": "unavailable",
                "raw_usage": None,
                "error_type": type(exc).__name__,
                "duration_ms": round((time.monotonic() - started) * 1000, 3),
            })
            raise

        usage = getattr(response, "usage", None)
        raw = None
        if usage is not None:
            if hasattr(usage, "model_dump"):
                raw = usage.model_dump()
            elif isinstance(usage, dict):
                raw = dict(usage)
            else:
                raw = {
                    name: getattr(usage, name, None)
                    for name in ("prompt_tokens", "completion_tokens", "total_tokens")
                }
        values, error = _validated_usage(
            SimpleNamespace(**raw) if raw is not None else None
        )
        status = "invalid" if error else (
            "available" if values is not None else "unavailable"
        )
        self.usage_records.append({
            "status": status,
            "raw_usage": raw,
            "validated_tokens": (
                {"prompt_tokens": values[0], "completion_tokens": values[1], "total_tokens": values[2]}
                if values is not None else None
            ),
            "error": error,
            "duration_ms": round((time.monotonic() - started) * 1000, 3),
        })
        return response


def _live_case_definitions():
    """Small, manually reviewable real-model smoke cases (not Fake responses)."""
    return [
        {
            "case_id": "live_travel_complete",
            "turns": [{
                "user_message": "请按模拟标准计算上海 3 天、employee 职级的差旅预算。",
                "required_tool_arguments": {"city": "上海", "days": 3, "job_level": "employee"},
                "required_tool_result": {"ok": True, "actual_total": 1500},
                "required_reply_terms": ["1500"],
                "require_tool": True,
                "forbid_tool": False,
            }],
        },
        {
            "case_id": "live_missing_city_then_provide",
            "turns": [
                {
                    "user_message": "请计算 3 天、employee 职级的差旅预算。",
                    "require_tool": False,
                    "forbid_tool": True,
                    "require_clarification": True,
                },
                {
                    "user_message": "目的地是上海。",
                    "required_tool_arguments": {"city": "上海", "days": 3, "job_level": "employee"},
                    "required_tool_result": {"ok": True, "actual_total": 1500},
                    "required_reply_terms": ["1500"],
                    "require_tool": True,
                    "forbid_tool": False,
                },
            ],
        },
    ]


def _create_live_client(api_key: str, *, request_timeout: float, base_url: str = TRUSTED_LIVE_ENDPOINT):
    # Lazy import ensures the default offline mode never constructs an SDK client.
    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=request_timeout,
        max_retries=0,
    )


def run_live_evaluation(
    *, api_key: str, case_limit: int = 1, max_turns: int = 2,
    max_calls_per_case: int = 3, max_calls_per_turn: int = 3,
    max_total_calls: int = 3,
    request_timeout: float = 20.0, total_timeout: float = 90.0,
    client_factory=None, cases=None,
    base_url: str = TRUSTED_LIVE_ENDPOINT,
    allow_custom_endpoint: bool = False,
    include_sensitive_transcript: bool = False,
):
    if not api_key:
        raise ValueError("missing API key; set the explicitly configured key environment variable")
    if case_limit < 1 or case_limit > 2 or max_turns < 1 or max_turns > 2:
        raise ValueError("case_limit must be between 1 and 2 and max_turns between 1 and 2")
    if max_calls_per_case < 1 or max_calls_per_case > 6:
        raise ValueError("max_calls_per_case must be between 1 and 6")
    if max_calls_per_turn < 1 or max_calls_per_turn > 3:
        raise ValueError("max_calls_per_turn must be between 1 and 3")
    if max_total_calls < 1 or max_total_calls > 12:
        raise ValueError("max_total_calls must be between 1 and 12")
    if (
        not isinstance(request_timeout, (int, float))
        or not math.isfinite(request_timeout)
        or not 0 < request_timeout <= MAX_REQUEST_TIMEOUT_SECONDS
    ):
        raise ValueError(f"request_timeout must be finite and between 0 and {MAX_REQUEST_TIMEOUT_SECONDS}s")
    if (
        not isinstance(total_timeout, (int, float))
        or not math.isfinite(total_timeout)
        or not 0 < total_timeout <= MAX_TOTAL_TIMEOUT_SECONDS
    ):
        raise ValueError(f"total_timeout must be finite and between 0 and {MAX_TOTAL_TIMEOUT_SECONDS}s")
    base_url = _validate_live_endpoint(base_url, allow_custom_endpoint)

    selected = list(_live_case_definitions() if cases is None else cases)[:case_limit]
    if any(len(case["turns"]) > max_turns for case in selected):
        raise ValueError("selected case exceeds max_turns")

    run_started = time.monotonic()
    deadline = time.monotonic() + total_timeout
    global_budget = {"used": 0, "attempts": 0, "limit": max_total_calls}
    factory = client_factory or _create_live_client
    initialization_started = time.monotonic()
    # Constructing the client is gated behind this function, which the CLI calls only for --live.
    try:
        raw_client = factory(api_key, request_timeout=request_timeout, base_url=base_url)
    except Exception as exc:
        return {
            "mode": "live",
            "cases_requested": case_limit,
            "cases_run": 0,
            "global_llm_call_attempt_count": global_budget["attempts"],
            "global_llm_forwarded_request_attempt_count": global_budget["used"],
            "global_llm_call_budget": max_total_calls,
            "client_initialization_duration_ms": round((time.monotonic() - initialization_started) * 1000, 3),
            "duration_ms": round((time.monotonic() - run_started) * 1000, 3),
            "stop_reason": "client_initialization_failed",
            "initialization_failure": {"failure_type": type(exc).__name__},
            "results": [],
        }
    live_client = BudgetedLiveClient(
        raw_client,
        case_budget=max_calls_per_case,
        turn_budget=max_calls_per_turn,
        global_budget=global_budget,
        deadline=deadline,
        request_timeout=request_timeout,
    )
    results = []
    run_stop_reason = None

    for case in selected:
        messages = []
        tools = []
        turns = []
        stop_reason = None
        case_started = time.monotonic()
        case_success = True
        for turn_index, turn in enumerate(case["turns"], start=1):
            if time.monotonic() >= deadline:
                stop_reason = "total_runtime_budget_exhausted"
                case_success = False
                break
            messages.append({"role": "user", "content": turn["user_message"]})
            input_messages = [dict(message) for message in messages]
            recorder_metrics = CaseMetrics(case_id=f"{case['case_id']}:{turn_index}")
            registry = RecordingToolRegistry(create_default_registry(), recorder_metrics)
            # Three AgentLoop steps yield at most three create() attempts; the client wrapper
            # additionally applies the hard case/global call gates before forwarding.
            agent = AgentLoop(
                client=live_client,
                tool_registry=registry,
                permission_checker=PermissionChecker(),
                max_steps=3,
                max_tool_calls=3,
            )
            turn_started = time.monotonic()
            usage_before = len(live_client.usage_records)
            live_client.turn_attempts = 0
            live_client.turn_budget_used = 0
            reply = None
            failure_type = None
            turn_stop = None
            try:
                reply, _ = _run_agent_with_temporary_log_filters(
                    agent, messages, role=case.get("role", "employee")
                )
                if time.monotonic() >= deadline:
                    raise LiveBudgetStop("total_runtime_budget_exhausted")
                new_calls = recorder_metrics.tool_calls
                if turn.get("forbid_tool") and new_calls:
                    raise AssertionError("tool executed while case requires clarification")
                if turn.get("require_tool") and not new_calls:
                    raise AssertionError("required travel tool was not executed")
                required = turn.get("required_tool_arguments")
                if required:
                    matching = [
                        call for call in new_calls
                        if call["name"] == "calculate_travel_budget"
                        and all(call["arguments"].get(key) == value for key, value in required.items())
                    ]
                    if not matching:
                        raise AssertionError(f"required tool arguments not observed: {required!r}")
                expected_result = turn.get("required_tool_result")
                if expected_result:
                    successful_results = []
                    for call in new_calls:
                        if call["name"] != "calculate_travel_budget":
                            continue
                        try:
                            successful_results.append(json.loads(call["result"]))
                        except (TypeError, json.JSONDecodeError):
                            continue
                    if not any(
                        all(result.get(key) == value for key, value in expected_result.items())
                        for result in successful_results
                    ):
                        raise AssertionError(f"expected successful tool result fields not observed: {expected_result!r}")
                for term in turn.get("required_reply_terms", []):
                    if term not in (reply or ""):
                        raise AssertionError(f"final reply missing expected term: {term!r}")
                if turn.get("require_clarification") and not any(
                    marker in (reply or "") for marker in ("补充", "请提供", "需要")
                ):
                    raise AssertionError("clarification response not detected")
                task_success = True
            except LiveBudgetStop as exc:
                task_success = False
                turn_stop = str(exc)
                failure_type = type(exc).__name__
            except Exception as exc:
                task_success = False
                failure_type = type(exc).__name__
                # Never serialize exception text, which could contain provider request details.
                turn_stop = (
                    "request_timeout"
                    if isinstance(exc, TimeoutError) or "Timeout" in failure_type
                    else "request_or_agent_failure"
                )
            turns.append({
                "turn": turn_index,
                "user_message": turn["user_message"],
                "history_before": input_messages,
                "task_success": task_success,
                "tool_execution_count": recorder_metrics.tool_execution_count,
                "tool_calls": recorder_metrics.tool_calls,
                "llm_call_attempt_count": live_client.turn_attempts,
                "duration_ms": round((time.monotonic() - turn_started) * 1000, 3),
                "token_usage_status": "invalid" if any(r["status"] == "invalid" for r in live_client.usage_records[usage_before:]) else ("available" if live_client.usage_records[usage_before:] and all(r["status"] == "available" for r in live_client.usage_records[usage_before:]) else "unavailable"),
                "usage": live_client.usage_records[usage_before:],
                "failure_type": failure_type,
                "stop_reason": turn_stop,
                "reply": reply,
            })
            tools.extend(recorder_metrics.tool_calls)
            if reply is not None:
                if not any(m.get("role") == "assistant" and m.get("content") == reply for m in messages[len(input_messages):]):
                    messages.append({"role": "assistant", "content": reply})
            if not task_success:
                case_success = False
                stop_reason = turn_stop
                break

        results.append({
            "case_id": case["case_id"],
            "task_success": case_success and len(turns) == len(case["turns"]),
            "turns": turns,
            "tool_execution_count": len(tools),
            "tool_calls": tools,
            "llm_call_attempt_count": live_client.case_attempts,
            "duration_ms": round((time.monotonic() - case_started) * 1000, 3),
            "failure_type": turns[-1]["failure_type"] if turns else None,
            "stop_reason": stop_reason,
        })
        # Per-case budget is enforced by resetting only after case completion.
        live_client.case_attempts = 0
        live_client.case_budget_used = 0
        if stop_reason:
            run_stop_reason = stop_reason
            break
        if global_budget["used"] >= global_budget["limit"] and case is not selected[-1]:
            run_stop_reason = "global_llm_call_budget_exhausted_before_next_case"
            break

    return {
        "mode": "live",
        "cases_requested": case_limit,
        "cases_run": len(results),
        "global_llm_call_attempt_count": global_budget["attempts"],
        "global_llm_forwarded_request_attempt_count": global_budget["used"],
        "global_llm_call_budget": global_budget["limit"],
        "client_initialization_duration_ms": round((time.monotonic() - initialization_started) * 1000, 3),
        "duration_ms": round((time.monotonic() - run_started) * 1000, 3),
        "stop_reason": run_stop_reason,
        "results": [
            _minimize_live_case_result(result, include_sensitive_transcript)
            for result in results
        ],
    }


def _minimize_live_case_result(result, include_sensitive_transcript):
    turns = []
    for turn in result["turns"]:
        usage_records = turn["usage"]
        validated = [record.get("validated_tokens") for record in usage_records]
        usage_state = turn["token_usage_status"]
        turn_summary = {
            "turn": turn["turn"],
            "task_success": turn["task_success"],
            "tool_execution_count": turn["tool_execution_count"],
            "tool_calls": [
                {"name": call["name"], "arguments": call["arguments"]}
                for call in turn["tool_calls"]
            ],
            "llm_call_attempt_count": turn["llm_call_attempt_count"],
            "duration_ms": turn["duration_ms"],
            "token_usage_status": usage_state,
            "token_usage": validated if usage_state == "available" else None,
            "failure_type": turn["failure_type"],
            "stop_reason": turn["stop_reason"],
        }
        if include_sensitive_transcript:
            turn_summary["transcript"] = {
                "user_message": turn["user_message"],
                "history": turn["history_before"],
                "tool_results": [call["result"] for call in turn["tool_calls"]],
                "reply": turn["reply"],
                "raw_usage": [record.get("raw_usage") for record in usage_records],
            }
        turns.append(turn_summary)

    all_turns = result["turns"]
    usage_states = [turn["token_usage_status"] for turn in all_turns]
    if "invalid" in usage_states:
        case_usage_state = "invalid"
    elif usage_states and all(state == "available" for state in usage_states):
        case_usage_state = "available"
    else:
        case_usage_state = "unavailable"
    valid_token_entries = [
        record["validated_tokens"]
        for turn in all_turns
        for record in turn["usage"]
        if record.get("validated_tokens") is not None
    ]
    case_usage = None
    if case_usage_state == "available":
        case_usage = {
            key: sum(entry[key] for entry in valid_token_entries)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }

    return {
        "case_id": result["case_id"],
        "task_success": result["task_success"],
        "tool_execution_count": result["tool_execution_count"],
        "tool_calls": [
            {"name": call["name"], "arguments": call["arguments"]}
            for call in result["tool_calls"]
        ],
        "llm_call_attempt_count": result["llm_call_attempt_count"],
        "duration_ms": result["duration_ms"],
        "token_usage_status": case_usage_state,
        "token_usage": case_usage,
        "failure_type": result["failure_type"],
        "stop_reason": result["stop_reason"],
        "turns": turns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline or explicitly opted-in live AgentLoop evaluation")
    parser.add_argument("--live", action="store_true", help="explicitly enable real provider requests")
    parser.add_argument("--api-key-env", default="AGENT_EVAL_API_KEY", help="environment variable containing the live API key")
    parser.add_argument("--case-limit", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=2)
    parser.add_argument("--max-calls-per-case", type=int, default=3)
    parser.add_argument("--max-calls-per-turn", type=int, default=3)
    parser.add_argument("--max-total-calls", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=20.0)
    parser.add_argument("--total-timeout", type=float, default=90.0)
    parser.add_argument("--base-url", default=TRUSTED_LIVE_ENDPOINT, help="live endpoint; custom HTTPS hosts require --allow-custom-endpoint")
    parser.add_argument("--allow-custom-endpoint", action="store_true", help="explicitly authorize sending the configured API key to a custom HTTPS host")
    parser.add_argument("--include-sensitive-transcript", action="store_true", help="WARNING: include full messages, tool results, replies, and raw usage in output; may contain sensitive data")
    args = parser.parse_args()
    if args.live:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            print(json.dumps({"mode": "live", "error": f"missing API key in environment variable {args.api_key_env!r}"}))
            return 2
        report = run_live_evaluation(
            api_key=api_key,
            case_limit=args.case_limit,
            max_turns=args.max_turns,
            max_calls_per_case=args.max_calls_per_case,
            max_calls_per_turn=args.max_calls_per_turn,
            max_total_calls=args.max_total_calls,
            request_timeout=args.request_timeout,
            total_timeout=args.total_timeout,
            base_url=args.base_url,
            allow_custom_endpoint=args.allow_custom_endpoint,
            include_sensitive_transcript=args.include_sensitive_transcript,
        )
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 0 if report["cases_run"] > 0 and all(result["task_success"] for result in report["results"]) else 1

    report = run_evaluation()
    compact_results = []
    for result in report["results"]:
        item = {
            "case_id": result["case_id"],
            "task_success": result["task_success"],
            "tool_execution_count": result["tool_execution_count"],
            "llm_call_attempt_count": result["llm_call_attempt_count"],
            "failure_reason": result["failure_reason"],
            "result_category": result["result_category"],
        }
        if "turns" in result:
            item["turns"] = [
                {
                    "turn_id": turn["case_id"].split(":", 1)[1],
                    "task_success": turn["task_success"],
                    "tool_execution_count": turn["tool_execution_count"],
                    "llm_call_attempt_count": turn["llm_call_attempt_count"],
                    "user_message": turn["user_message"],
                    "issued_tool_calls": turn["issued_tool_calls"],
                    "executed_tool_calls": [
                        {"name": call["name"], "arguments": call["arguments"]}
                        for call in turn["tool_calls"]
                    ],
                    "tool_results": [
                        _compact_tool_result(call["result"])
                        for call in turn["tool_calls"]
                    ],
                    "reply": turn["reply"],
                    "failure_reason": turn["failure_reason"],
                }
                for turn in result["turns"]
            ]
        compact_results.append(item)
    output = {
        key: report[key]
        for key in ("cases", "passed", "failed", "success_rate")
    }
    output["results"] = compact_results
    print(json.dumps(output, ensure_ascii=True, indent=2))
    return 0 if report["failed"] == 0 else 1


def _compact_tool_result(value):
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value
    fields = (
        "ok", "city", "days", "job_level", "policy_budget",
        "actual_total", "over_limit_items", "citation",
    )
    return {key: parsed[key] for key in fields if key in parsed}


if __name__ == "__main__":
    raise SystemExit(main())
