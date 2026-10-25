from copy import deepcopy
from types import SimpleNamespace

import pytest

from app.agent.loop import (
    AgentLoop,
    ContextBudgetExceeded,
    InvalidToolHistory,
)
from app.agent.request_budget import serialized_request_size
from app.permissions.checker import PermissionChecker
from app.planner.planner import Planner
from app.tools.registry import Tool, ToolRegistry


class FakeToolCall:
    id = "call_1"

    function = SimpleNamespace(
        name="calculate",
        arguments='{"expression": "100 + 20"}',
    )

    def model_dump(self):
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


class FakeMessageWithTool:
    content = None
    tool_calls = [FakeToolCall()]


class FakeMessageFinal:
    content = "计算结果是 120"
    tool_calls = None


class FakeResponse:
    def __init__(self, message):
        self.choices = [
            SimpleNamespace(message=message)
        ]


class FakeClient:
    def __init__(self):
        self.call_count = 0
        self.received_messages = []
        self.received_requests = []

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.create
            )
        )

    def create(self, **kwargs):
        self.call_count += 1
        self.received_requests.append(deepcopy(kwargs))
        self.received_messages.append(
            [dict(message) for message in kwargs["messages"]]
        )

        if self.call_count == 1:
            return FakeResponse(FakeMessageWithTool())

        return FakeResponse(FakeMessageFinal())


def create_agent(client, max_context_chars=24000):
    from app.tools.calculator import CalculatorTool

    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())

    return AgentLoop(
        client=client,
        tool_registry=tool_registry,
        permission_checker=PermissionChecker(),
        max_steps=5,
        max_context_chars=max_context_chars,
    )


class BatchFakeClient(FakeClient):
    def __init__(self, tool_calls):
        super().__init__()
        self.tool_calls = tool_calls

    def create(self, **kwargs):
        self.call_count += 1
        self.received_requests.append(deepcopy(kwargs))
        self.received_messages.append(
            [dict(message) for message in kwargs["messages"]]
        )
        if self.call_count == 1:
            return FakeResponse(
                SimpleNamespace(content=None, tool_calls=self.tool_calls)
            )
        return FakeResponse(FakeMessageFinal())


class RecordingArgumentTool(Tool):
    name = "calculate"
    description = "Record a payload for AgentLoop tests"
    parameters = {
        "type": "object",
        "properties": {"payload": {}},
        "required": ["payload"],
    }

    def __init__(self):
        self.executions = []

    def execute(self, arguments):
        self.executions.append(deepcopy(arguments))
        return "recorded"


def make_tool_call(call_id, arguments):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name="calculate",
            arguments=arguments,
        ),
        model_dump=lambda: {
            "id": call_id,
            "type": "function",
            "function": {
                "name": "calculate",
                "arguments": arguments,
            },
        },
    )


def make_payload_tool_call(payload):
    return SimpleNamespace(model_dump=lambda: deepcopy(payload))


def create_batch_agent(client, tool, max_tool_calls=12):
    registry = ToolRegistry()
    registry.register(tool)
    return AgentLoop(
        client=client,
        tool_registry=registry,
        permission_checker=PermissionChecker(),
        max_tool_calls=max_tool_calls,
    )


class SequencedBatchFakeClient(FakeClient):
    def __init__(self, batches):
        super().__init__()
        self.batches = batches

    def create(self, **kwargs):
        self.call_count += 1
        self.received_requests.append(deepcopy(kwargs))
        self.received_messages.append(
            [dict(message) for message in kwargs["messages"]]
        )
        batch_index = self.call_count - 1
        if batch_index < len(self.batches):
            return FakeResponse(SimpleNamespace(
                content=None,
                tool_calls=self.batches[batch_index],
            ))
        return FakeResponse(FakeMessageFinal())


def test_agent_can_execute_multiple_steps():
    client = FakeClient()
    agent = create_agent(client, max_context_chars=1000)

    messages = [
        {"role": "system", "content": "Keep the security rules."},
        {"role": "user", "content": "old context " * 250},
        {
            "role": "user",
            "content": "计算 100 + 20",
        }
    ]

    reply, _ = agent.run(messages)

    assert reply == "计算结果是 120"
    assert client.call_count == 2

    assert messages == [
        {"role": "system", "content": "Keep the security rules."},
        {"role": "user", "content": "old context " * 250},
        {"role": "user", "content": "计算 100 + 20"},
    ]

    # The large old request was trimmed, while the active request and
    # mandatory system message survived. The second LLM request contains
    # a complete assistant tool-call/result group.
    assert len(client.received_messages) == 2
    assert all(
        "old context" not in str(message)
        for message in client.received_messages[0]
    )
    assert client.received_messages[0][0]["content"] == "Keep the security rules."
    assert client.received_messages[0][-1]["content"] == "计算 100 + 20"
    groups = AgentLoop._group_messages(client.received_messages[1])
    tool_group = next(group for group in groups if len(group) > 1)
    assert tool_group[0]["tool_calls"][0]["id"] == tool_group[1]["tool_call_id"]
    assert all(
        serialized_request_size(
            request["model"], request["messages"], tools=request["tools"]
        ) <= 1000
        for request in client.received_requests
    )


def test_agent_stops_after_max_steps():
    class AlwaysToolClient(FakeClient):
        def create(self, **kwargs):
            self.call_count += 1
            return FakeResponse(FakeMessageWithTool())

    client = AlwaysToolClient()

    agent = create_agent(client)

    messages = [
        {
            "role": "user",
            "content": "不断计算",
        }
    ]

    reply, _ = agent.run(messages)

    assert reply == "Sorry, I couldn't complete this after several tool calls."
    assert client.call_count == 5


def test_context_budget_trims_old_ordinary_messages_as_complete_units():
    agent = create_agent(FakeClient(), max_context_chars=1000)
    messages = [
        {"role": "system", "content": "security rule"},
        {"role": "user", "content": "old " * 400},
        {"role": "user", "content": "current request"},
    ]

    bounded = agent._bounded_messages(messages)

    assert bounded == [messages[0], messages[2]]
    assert agent._serialized_request_size(bounded, []) <= 1000


def test_context_trimming_preserves_system_and_developer_instruction_order():
    client = FakeClient()
    agent = create_agent(client, max_context_chars=1000)
    messages = [
        {"role": "system", "content": "system security instruction"},
        {"role": "developer", "content": "developer workflow instruction"},
        {"role": "user", "content": "obsolete history " * 100},
        {"role": "user", "content": "current request"},
    ]
    original = deepcopy(messages)

    reply, _ = agent.run(messages)

    sent = client.received_messages[0]
    assert [(message["role"], message["content"]) for message in sent[:2]] == [
        ("system", "system security instruction"),
        ("developer", "developer workflow instruction"),
    ]
    assert sent[-1] == {"role": "user", "content": "current request"}
    assert all("obsolete history" not in str(message) for message in sent)
    assert reply == "计算结果是 120"
    assert messages == original


def test_required_developer_instruction_over_budget_stops_before_llm():
    client = FakeClient()
    agent = create_agent(client, max_context_chars=1000)
    messages = [
        {"role": "system", "content": "security"},
        {"role": "developer", "content": "d" * 1200},
        {"role": "user", "content": "current request"},
    ]
    original = deepcopy(messages)

    reply, _ = agent.run(messages)

    assert reply.startswith("Error: agent context budget exceeded")
    assert client.call_count == 0
    assert messages == original


def test_context_budget_counts_tool_arguments_and_tool_schemas():
    agent = create_agent(FakeClient(), max_context_chars=1000)
    assistant_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call-large",
            "type": "function",
            "function": {
                "name": "calculate",
                "arguments": '{"value":"' + "x" * 1400 + '"}',
            },
        }],
    }
    messages = [
        {"role": "system", "content": "security rule"},
        {"role": "user", "content": "old request"},
        assistant_call,
        {"role": "tool", "tool_call_id": "call-large", "content": "done"},
        {"role": "user", "content": "current request"},
    ]
    large_schema = [{
        "type": "function",
        "function": {"name": "large", "description": "x" * 1400},
    }]

    assert agent._serialized_request_size(messages, []) > 1000
    assert agent._serialized_request_size(messages[:1] + messages[-1:], large_schema) > 1000
    bounded = agent._bounded_messages(messages)
    assert assistant_call not in bounded
    assert agent._serialized_request_size(bounded, []) <= 1000


def test_multiple_tool_calls_and_results_are_kept_as_one_ordered_group():
    agent = create_agent(FakeClient(), max_context_chars=1000)
    assistant_calls = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call-a", "type": "function", "function": {"name": "a", "arguments": "{}"}},
            {"id": "call-b", "type": "function", "function": {"name": "b", "arguments": "{}"}},
        ],
    }
    result_b = {"role": "tool", "tool_call_id": "call-b", "content": "result-b"}
    result_a = {"role": "tool", "tool_call_id": "call-a", "content": "result-a"}
    messages = [
        {"role": "system", "content": "security rule"},
        {"role": "user", "content": "old context " * 250},
        assistant_calls,
        result_b,
        result_a,
        {"role": "user", "content": "current request"},
    ]

    bounded = agent._bounded_messages(messages)
    groups = AgentLoop._group_messages(bounded)
    tool_group = next(group for group in groups if len(group) == 3)

    assert tool_group == [assistant_calls, result_b, result_a]
    assert [message["tool_call_id"] for message in tool_group[1:]] == [
        call["id"] for call in assistant_calls["tool_calls"][::-1]
    ]
    assert bounded[0] == messages[0]
    assert bounded[-1] == messages[-1]
    assert agent._serialized_request_size(bounded, []) <= 1000


@pytest.mark.parametrize(
    "messages",
    [
        [
            {"role": "system", "content": "s" * 1200},
            {"role": "user", "content": "current request"},
        ],
        [
            {"role": "system", "content": "security rule"},
            {"role": "user", "content": "u" * 1200},
        ],
    ],
)
def test_oversized_required_system_or_current_message_stops_before_llm(messages):
    client = FakeClient()
    agent = create_agent(client, max_context_chars=1000)
    original = [dict(message) for message in messages]

    reply, _ = agent.run(messages)

    assert reply.startswith("Error: agent context budget exceeded")
    assert client.call_count == 0
    assert messages == original


def test_oversized_active_tool_group_is_rejected_as_a_whole():
    agent = create_agent(FakeClient(), max_context_chars=1000)
    messages = [
        {"role": "system", "content": "security rule"},
        {"role": "user", "content": "current request"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-large",
                "type": "function",
                "function": {
                    "name": "calculate",
                    "arguments": '{"value":"' + "x" * 1400 + '"}',
                },
            }],
        },
        {"role": "tool", "tool_call_id": "call-large", "content": "result"},
    ]

    with pytest.raises(ContextBudgetExceeded):
        agent._bounded_messages(messages)

    assert [message["role"] for message in messages] == [
        "system", "user", "assistant", "tool",
    ]


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "tool", "tool_call_id": "missing", "content": "orphan"}],
        [{"role": "assistant", "tool_calls": [{"id": "missing"}]}],
        [
            {"role": "assistant", "tool_calls": [{"id": "call-a"}]},
            {"role": "tool", "tool_call_id": "call-b", "content": "mismatch"},
        ],
        [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call-a",
                    "type": "function",
                    "function": {"name": "calculate", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": " ", "content": "result"},
        ],
    ],
)
def test_incomplete_or_mismatched_tool_history_is_rejected(messages):
    agent = create_agent(FakeClient())

    with pytest.raises(InvalidToolHistory):
        agent._bounded_messages(messages)


def test_invalid_tool_history_through_run_does_not_call_fake_client():
    client = FakeClient()
    agent = create_agent(client)
    messages = [
        {"role": "user", "content": "continue"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-orphan",
                "type": "function",
                "function": {"name": "calculate", "arguments": "{}"},
            }],
        },
    ]

    reply, _ = agent.run(messages)

    assert reply.startswith("Error: agent message history is invalid")
    assert client.call_count == 0


@pytest.mark.parametrize(
    "call",
    [
        {"type": "function", "function": {"name": "calculate", "arguments": "{}"}},
        {"id": "call-a", "function": {"name": "calculate", "arguments": "{}"}},
        {"id": "call-a", "type": "custom", "function": {"name": "calculate", "arguments": "{}"}},
        {"id": "call-a", "type": "function", "function": "calculate"},
        {"id": "call-a", "type": "function", "function": {"name": " ", "arguments": "{}"}},
        {"id": "call-a", "type": "function", "function": {"name": "calculate", "arguments": {}}},
    ],
)
def test_malformed_tool_call_history_is_rejected_before_model_request(call):
    client = FakeClient()
    agent = create_agent(client)
    call_id = call.get("id", "call-a")
    messages = [
        {"role": "user", "content": "continue"},
        {"role": "assistant", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call_id, "content": "result"},
    ]
    original = deepcopy(messages)

    reply, _ = agent.run(messages)

    assert reply.startswith("Error: agent message history is invalid")
    assert client.call_count == 0
    assert messages == original


def test_duplicate_tool_call_ids_in_history_are_rejected_before_model_request():
    client = FakeClient()
    agent = create_agent(client)
    call = {
        "id": "duplicate",
        "type": "function",
        "function": {"name": "calculate", "arguments": "{}"},
    }
    messages = [
        {"role": "user", "content": "continue"},
        {"role": "assistant", "tool_calls": [call, deepcopy(call)]},
        {"role": "tool", "tool_call_id": "duplicate", "content": "result"},
        {"role": "tool", "tool_call_id": "duplicate", "content": "result"},
    ]

    reply, _ = agent.run(messages)

    assert reply.startswith("Error: agent message history is invalid")
    assert client.call_count == 0


def test_tool_schema_over_budget_is_rejected_before_fake_client_call():
    class LargeSchemaRegistry:
        def schemas(self):
            return [{
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "x" * 1400,
                    "parameters": {"type": "object", "properties": {}},
                },
            }]

    client = FakeClient()
    agent = AgentLoop(
        client=client,
        tool_registry=LargeSchemaRegistry(),
        permission_checker=PermissionChecker(),
        max_context_chars=1000,
    )

    reply, _ = agent.run([{"role": "user", "content": "calculate"}])

    assert reply.startswith("Error: agent context budget exceeded")
    assert client.call_count == 0


def test_planner_budget_rejection_stops_agent_before_any_fake_client_call():
    class LargeSchemaRegistry:
        def schemas(self):
            return [{
                "type": "function",
                "function": {
                    "name": "data_analysis",
                    "description": "x" * 1400,
                    "parameters": {"type": "object", "properties": {}},
                },
            }]

    client = FakeClient()
    registry = LargeSchemaRegistry()
    checker = PermissionChecker()
    planner = Planner(
        client,
        tool_registry=registry,
        permission_checker=checker,
        max_context_chars=1000,
    )
    agent = AgentLoop(
        client=client,
        tool_registry=registry,
        permission_checker=checker,
        planner=planner,
        max_context_chars=1000,
    )

    reply, _ = agent.run([{"role": "user", "content": "analyze data"}], role="admin")

    assert reply.startswith("Error: planner context budget exceeded")
    assert client.call_count == 0


def test_individually_configured_agent_budget_is_effective():
    agent = create_agent(FakeClient(), max_context_chars=1200)

    assert agent.max_context_chars == 1200

    client = FakeClient()
    constrained = create_agent(client, max_context_chars=1000)
    reply, _ = constrained.run([{"role": "user", "content": "x" * 1200}])

    assert reply.startswith("Error: agent context budget exceeded")
    assert client.call_count == 0


def test_multiple_tool_calls_are_preserved_in_the_final_fake_request():
    client = FakeClient()
    agent = create_agent(client)
    calls = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call-a", "type": "function", "function": {"name": "a", "arguments": "{}"}},
            {"id": "call-b", "type": "function", "function": {"name": "b", "arguments": "{}"}},
        ],
    }
    result_b = {"role": "tool", "tool_call_id": "call-b", "content": "B"}
    result_a = {"role": "tool", "tool_call_id": "call-a", "content": "A"}
    messages = [
        {"role": "system", "content": "security"},
        {"role": "user", "content": "prior request"},
        calls,
        result_b,
        result_a,
        {"role": "user", "content": "current request"},
    ]

    agent.run(messages)

    sent = client.received_messages[0]
    groups = AgentLoop._group_messages(sent)
    tool_group = next(group for group in groups if len(group) == 3)
    assert tool_group == [calls, result_b, result_a]
    assert sent.index(calls) < sent.index(result_b) < sent.index(result_a)


def test_consecutive_history_trimming_keeps_complete_tool_group_in_fake_request():
    client = FakeClient()
    agent = create_agent(client, max_context_chars=1000)
    calls = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "old-call-a", "type": "function", "function": {"name": "a", "arguments": "{}"}},
            {"id": "old-call-b", "type": "function", "function": {"name": "b", "arguments": "{}"}},
        ],
    }
    result_a = {"role": "tool", "tool_call_id": "old-call-a", "content": "A"}
    result_b = {"role": "tool", "tool_call_id": "old-call-b", "content": "B"}
    messages = [
        {"role": "system", "content": "security"},
        {"role": "user", "content": "old context one " * 50},
        {"role": "user", "content": "old context two " * 50},
        calls,
        result_a,
        result_b,
        {"role": "user", "content": "current request"},
    ]
    original_messages = deepcopy(messages)

    agent.run(messages)

    sent = client.received_messages[0]
    groups = AgentLoop._group_messages(sent)
    tool_group = next(group for group in groups if len(group) == 3)
    assert tool_group == [calls, result_a, result_b]
    assert all("old context" not in str(message) for message in sent)
    assert sent[-1] == {"role": "user", "content": "current request"}
    assert messages == original_messages


def test_oversized_tool_result_returns_error_without_internal_retry():
    from app.tools.calculator import CalculatorTool

    client = FakeClient()
    registry = ToolRegistry()
    tool = CalculatorTool()
    executions = []

    def execute_with_large_result(arguments):
        executions.append(arguments)
        return "x" * 1400

    tool.execute = execute_with_large_result
    registry.register(tool)
    agent = AgentLoop(
        client=client,
        tool_registry=registry,
        permission_checker=PermissionChecker(),
        max_context_chars=1000,
    )

    reply, _ = agent.run([{"role": "user", "content": "calculate"}])

    assert reply.startswith("Error: agent context budget exceeded")
    assert client.call_count == 1
    assert len(executions) == 1


def test_normal_sized_tool_arguments_execute_and_keep_call_result_pairing():
    tool = RecordingArgumentTool()
    client = BatchFakeClient([
        make_tool_call("normal-call", '{"payload":{"value":"small"}}')
    ])
    agent = create_batch_agent(client, tool)

    reply, _ = agent.run([{"role": "user", "content": "run the tool"}])

    assert reply == "计算结果是 120"
    assert tool.executions == [{"payload": {"value": "small"}}]
    assert client.call_count == 2
    groups = AgentLoop._group_messages(client.received_messages[1])
    tool_group = next(group for group in groups if len(group) == 2)
    assert tool_group[0]["tool_calls"][0]["id"] == "normal-call"
    assert tool_group[1]["tool_call_id"] == "normal-call"


def test_single_oversized_tool_argument_is_rejected_before_execution():
    tool = RecordingArgumentTool()
    raw_arguments = '{"payload":{"value":"' + "x" * 8_100 + '"}}'
    client = BatchFakeClient([make_tool_call("large-call", raw_arguments)])
    agent = create_batch_agent(client, tool)
    messages = [
        {"role": "system", "content": "Keep the security rules."},
        {
            "role": "user",
            "content": "run the tool",
            "metadata": {"nested": ["caller-owned"]},
        },
    ]
    original = deepcopy(messages)

    reply, _ = agent.run(messages)

    assert reply.startswith("Error: tool arguments exceed")
    assert tool.executions == []
    assert client.call_count == 1
    assert len(client.received_requests) == 1
    assert messages == original


def test_later_oversized_argument_prevents_earlier_tool_in_same_batch():
    tool = RecordingArgumentTool()
    client = BatchFakeClient([
        make_tool_call("small-first", '{"payload":{"value":"ok"}}'),
        make_tool_call(
            "large-second",
            '{"payload":{"value":"' + "x" * 8_100 + '"}}',
        ),
    ])
    agent = create_batch_agent(client, tool)

    reply, _ = agent.run([{"role": "user", "content": "run both tools"}])

    assert reply.startswith("Error: tool arguments exceed")
    assert tool.executions == []
    assert client.call_count == 1


def test_individually_bounded_arguments_are_rejected_when_batch_total_is_too_large():
    tool = RecordingArgumentTool()
    first = '{"payload":{"value":"' + "a" * 4_100 + '"}}'
    second = '{"payload":{"value":"' + "b" * 4_100 + '"}}'
    assert len(first) < 8_000
    assert len(second) < 8_000
    client = BatchFakeClient([
        make_tool_call("batch-a", first),
        make_tool_call("batch-b", second),
    ])
    agent = create_batch_agent(client, tool)

    reply, _ = agent.run([{"role": "user", "content": "run both tools"}])

    assert reply.startswith("Error: tool arguments exceed")
    assert tool.executions == []
    assert client.call_count == 1


def test_oversized_nested_argument_is_rejected_before_tool_execution():
    tool = RecordingArgumentTool()
    nested_json = (
        '{"payload":{"items":['
        + ",".join('{"value":"' + "x" * 2_100 + '"}' for _ in range(4))
        + ']}}'
    )
    client = BatchFakeClient([
        make_tool_call("nested-large", nested_json)
    ])
    agent = create_batch_agent(client, tool)

    reply, _ = agent.run([{"role": "user", "content": "run the tool"}])

    assert reply.startswith("Error: tool arguments exceed")
    assert tool.executions == []
    assert client.call_count == 1


def test_unicode_expansion_is_counted_in_compact_json_size():
    tool = RecordingArgumentTool()
    raw_arguments = '{"payload":{"value":"' + "中" * 1_400 + '"}}'
    assert len(raw_arguments) < 8_000
    client = BatchFakeClient([
        make_tool_call("unicode-large", raw_arguments)
    ])
    agent = create_batch_agent(client, tool)

    reply, _ = agent.run([{"role": "user", "content": "run the tool"}])

    assert reply.startswith("Error: tool arguments exceed")
    assert tool.executions == []
    assert client.call_count == 1


@pytest.mark.parametrize(
    ("arguments", "expected_error"),
    [
        ("{invalid-json", "invalid JSON arguments"),
        ("", "missing required arguments: payload"),
    ],
)
def test_invalid_or_missing_arguments_keep_existing_tool_validation(
    arguments,
    expected_error,
):
    tool = RecordingArgumentTool()
    client = BatchFakeClient([make_tool_call("invalid-call", arguments)])
    agent = create_batch_agent(client, tool)

    reply, _ = agent.run([{"role": "user", "content": "run the tool"}])

    assert reply == "计算结果是 120"
    assert tool.executions == []
    assert client.call_count == 2
    groups = AgentLoop._group_messages(client.received_messages[1])
    tool_group = next(group for group in groups if len(group) == 2)
    assert tool_group[0]["tool_calls"][0]["id"] == "invalid-call"
    assert tool_group[1]["tool_call_id"] == "invalid-call"
    assert expected_error in tool_group[1]["content"]


def test_oversized_batch_is_not_sent_in_a_follow_up_model_request():
    tool = RecordingArgumentTool()
    client = BatchFakeClient([
        make_tool_call(
            "large-call",
            '{"payload":{"value":"' + "x" * 8_100 + '"}}',
        )
    ])
    agent = create_batch_agent(client, tool)
    messages = [{"role": "user", "content": "run the tool"}]
    original = deepcopy(messages)

    reply, _ = agent.run(messages)

    assert reply.startswith("Error: tool arguments exceed")
    assert client.call_count == 1
    assert len(client.received_requests) == 1
    assert all(
        "tool_calls" not in message
        for request in client.received_requests[1:]
        for message in request["messages"]
    )
    assert tool.executions == []
    assert messages == original


def test_tool_permission_denial_still_precedes_execution_and_pairs_result():
    tool = RecordingArgumentTool()
    client = BatchFakeClient([
        make_tool_call("denied-call", '{"payload":{"value":"small"}}')
    ])
    agent = create_batch_agent(client, tool)

    agent.run(
        [{"role": "user", "content": "run the tool"}],
        role="guest",
    )

    assert tool.executions == []
    assert client.call_count == 2
    groups = AgentLoop._group_messages(client.received_messages[1])
    tool_group = next(group for group in groups if len(group) == 2)
    assert tool_group[0]["tool_calls"][0]["id"] == "denied-call"
    assert tool_group[1]["tool_call_id"] == "denied-call"
    assert "Permission denied" in tool_group[1]["content"]


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "function", "function": {"name": "calculate", "arguments": "{}"}},
        {"id": "bad", "function": {"name": "calculate", "arguments": "{}"}},
        {"id": "bad", "type": "custom", "function": {"name": "calculate", "arguments": "{}"}},
        {"id": "bad", "type": "function", "function": None},
        {"id": "bad", "type": "function", "function": {"name": "", "arguments": "{}"}},
        {"id": "bad", "type": "function", "function": {"name": "calculate", "arguments": []}},
    ],
)
def test_malformed_model_tool_call_is_rejected_without_execution(payload):
    tool = RecordingArgumentTool()
    client = BatchFakeClient([make_payload_tool_call(payload)])
    agent = create_batch_agent(client, tool)
    messages = [{"role": "user", "content": "run the tool"}]
    original = deepcopy(messages)

    reply, _ = agent.run(messages)

    assert reply.startswith("Error: agent tool calls are invalid")
    assert client.call_count == 1
    assert len(client.received_requests) == 1
    assert tool.executions == []
    assert messages == original


def test_duplicate_model_tool_call_ids_reject_whole_batch_before_execution():
    tool = RecordingArgumentTool()
    first = make_tool_call("duplicate", '{"payload":{"value":"first"}}')
    second = make_tool_call("duplicate", '{"payload":{"value":"second"}}')
    client = BatchFakeClient([first, second])
    agent = create_batch_agent(client, tool)

    reply, _ = agent.run([{"role": "user", "content": "run both"}])

    assert reply.startswith("Error: agent tool calls are invalid")
    assert client.call_count == 1
    assert tool.executions == []


def test_later_malformed_model_call_prevents_earlier_tool_execution():
    tool = RecordingArgumentTool()
    first = make_tool_call("good-call", '{"payload":{"value":"first"}}')
    second = make_payload_tool_call({
        "id": "bad-call",
        "type": "function",
        "function": {"name": "calculate", "arguments": []},
    })
    client = BatchFakeClient([first, second])
    agent = create_batch_agent(client, tool)

    reply, _ = agent.run([{"role": "user", "content": "run both"}])

    assert reply.startswith("Error: agent tool calls are invalid")
    assert client.call_count == 1
    assert tool.executions == []


def test_normal_multi_tool_call_batch_executes_in_order_and_pairs_ids():
    tool = RecordingArgumentTool()
    client = BatchFakeClient([
        make_tool_call("ordered-a", '{"payload":{"value":"first"}}'),
        make_tool_call("ordered-b", '{"payload":{"value":"second"}}'),
    ])
    agent = create_batch_agent(client, tool)
    messages = [{"role": "user", "content": "run both"}]
    original = deepcopy(messages)

    reply, _ = agent.run(messages)

    assert reply == "计算结果是 120"
    assert tool.executions == [
        {"payload": {"value": "first"}},
        {"payload": {"value": "second"}},
    ]
    assert client.call_count == 2
    sent = client.received_messages[1]
    group = next(group for group in AgentLoop._group_messages(sent) if len(group) == 3)
    assert [call["id"] for call in group[0]["tool_calls"]] == [
        "ordered-a", "ordered-b",
    ]
    assert [message["tool_call_id"] for message in group[1:]] == [
        "ordered-a", "ordered-b",
    ]
    assert messages == original


def test_tool_batch_over_remaining_budget_is_rejected_without_partial_execution():
    tool = RecordingArgumentTool()
    first = make_tool_call("first", '{"payload":{"value":"first"}}')
    second = make_tool_call("second", '{"payload":{"value":"second"}}')
    third = make_tool_call("third", '{"payload":{"value":"third"}}')
    client = SequencedBatchFakeClient([[first], [second, third]])
    agent = create_batch_agent(client, tool, max_tool_calls=2)

    reply, task_state = agent.run([{"role": "user", "content": "run tools"}])

    assert reply == "Error: agent tool-call budget exceeded."
    assert task_state is None
    assert tool.executions == [{"payload": {"value": "first"}}]
    assert client.call_count == 2


def test_tool_batch_exactly_fitting_remaining_budget_executes():
    tool = RecordingArgumentTool()
    client = BatchFakeClient([
        make_tool_call("exact-a", '{"payload":{"value":"a"}}'),
        make_tool_call("exact-b", '{"payload":{"value":"b"}}'),
    ])
    agent = create_batch_agent(client, tool, max_tool_calls=2)

    reply, _ = agent.run([{"role": "user", "content": "run both"}])

    assert reply == "计算结果是 120"
    assert tool.executions == [
        {"payload": {"value": "a"}},
        {"payload": {"value": "b"}},
    ]
    assert client.call_count == 2


def test_multi_tool_batch_still_checks_each_tool_permission():
    allowed_tool = RecordingArgumentTool()
    denied_tool = RecordingArgumentTool()
    denied_tool.name = "restricted_tool"
    registry = ToolRegistry()
    registry.register(allowed_tool)
    registry.register(denied_tool)
    client = BatchFakeClient([
        make_tool_call("allowed-call", '{"payload":{"value":"allowed"}}'),
        make_payload_tool_call({
            "id": "denied-call",
            "type": "function",
            "function": {
                "name": "restricted_tool",
                "arguments": '{"payload":{"value":"denied"}}',
            },
        }),
    ])
    agent = AgentLoop(
        client=client,
        tool_registry=registry,
        permission_checker=PermissionChecker(),
    )

    agent.run([{"role": "user", "content": "run both"}], role="employee")

    assert allowed_tool.executions == [{"payload": {"value": "allowed"}}]
    assert denied_tool.executions == []
    sent = client.received_messages[1]
    group = next(group for group in AgentLoop._group_messages(sent) if len(group) == 3)
    assert [call["id"] for call in group[0]["tool_calls"]] == [
        "allowed-call", "denied-call",
    ]
    assert [message["tool_call_id"] for message in group[1:]] == [
        "allowed-call", "denied-call",
    ]
    assert "Permission denied" in group[2]["content"]
