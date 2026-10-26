import copy
import json
import math
from types import SimpleNamespace

import pytest

from app.tools import create_default_registry
from app.tools.registry import ToolRegistry
from app.tools.travel import POLICY_CITIES, TravelBudgetTool
from app.agent.loop import AgentLoop
from app.permissions.checker import PermissionChecker


def execute(arguments):
    return json.loads(TravelBudgetTool().execute(arguments))


def test_travel_budget_returns_breakdown_and_citation():
    result = execute({
        "city": "上海", "days": 3, "job_level": "employee",
        "lodging_per_day": 400, "meal_per_day": 80, "transport_total": 200,
    })
    assert result["policy_budget"] == 1500
    assert result["actual_total"] == 1640
    assert result["over_limit_items"] == ["lodging_per_day", "transport_total"]
    assert result["citation"] == "差旅预算 MVP 模拟数据（非真实企业制度）"
    assert result["policy_data_type"] == "simulated_mvp"


@pytest.mark.parametrize("city", sorted(POLICY_CITIES))
def test_supported_city_uses_simulated_policy(city):
    result = execute({"city": city, "days": 1, "job_level": "employee"})

    assert result["city"] == city
    assert result["policy_budget"] == 600


@pytest.mark.parametrize("city", ["杭州", "火星", "上海市"])
def test_unsupported_city_is_rejected_without_defaulting(city):
    with pytest.raises(ValueError, match="unsupported city"):
        TravelBudgetTool().execute({"city": city, "days": 1, "job_level": "employee"})


@pytest.mark.parametrize("arguments", [
    {},
    {"city": "上海", "days": 1},
    {"city": "上海", "job_level": "employee"},
    {"days": 1, "job_level": "employee"},
])
def test_direct_tool_call_reports_missing_fields(arguments):
    with pytest.raises(ValueError, match="missing required arguments"):
        TravelBudgetTool().execute(arguments)


@pytest.mark.parametrize("arguments", [None, [], "{}"])
def test_arguments_must_be_an_object(arguments):
    with pytest.raises(ValueError, match="arguments must be a JSON object"):
        TravelBudgetTool().execute(arguments)


def test_non_string_argument_name_is_rejected_predictably():
    with pytest.raises(ValueError, match="argument names must be strings"):
        TravelBudgetTool().execute({"city": "上海", "days": 1, "job_level": "employee", 1: 2})


def test_registry_and_direct_call_have_same_missing_field_semantics():
    registry = ToolRegistry()
    registry.register(TravelBudgetTool())

    direct_error = None
    try:
        TravelBudgetTool().execute({"city": "上海", "days": 1})
    except ValueError as exc:
        direct_error = str(exc)

    registry_error = registry.execute(
        "calculate_travel_budget", {"city": "上海", "days": 1}
    )

    assert direct_error == "missing required arguments: job_level"
    assert registry_error == f"Error: {direct_error}"


def test_default_registry_registers_travel_tool_once_with_matching_schema():
    registry = create_default_registry()
    tool_name = "calculate_travel_budget"
    registered_tool = registry.get(tool_name)
    schema_names = [
        schema["function"]["name"] for schema in registry.schemas()
    ]

    assert isinstance(registered_tool, TravelBudgetTool)
    assert schema_names.count(tool_name) == 1
    assert registered_tool.name == tool_name


@pytest.mark.parametrize("city", [None, 1, [], {}])
def test_city_must_be_string(city):
    with pytest.raises(ValueError, match="city must be a string"):
        TravelBudgetTool().execute({"city": city, "days": 1, "job_level": "employee"})


@pytest.mark.parametrize("level", [None, 1, [], "intern"])
def test_job_level_must_be_a_supported_string(level):
    with pytest.raises(ValueError, match="job_level must be"):
        TravelBudgetTool().execute({"city": "上海", "days": 1, "job_level": level})


@pytest.mark.parametrize("days", [True, False, 1.0, "2", 0, -1, 91])
def test_days_must_be_integer_between_one_and_ninety(days):
    with pytest.raises(ValueError, match="days must be"):
        TravelBudgetTool().execute({"city": "上海", "days": days, "job_level": "employee"})


@pytest.mark.parametrize("amount", [-1, True, "10", None, math.nan, math.inf, -math.inf, 1_000_001, 10**1000])
@pytest.mark.parametrize("field", ["lodging_per_day", "meal_per_day", "transport_total"])
def test_invalid_expense_values_are_rejected(field, amount):
    arguments = {"city": "上海", "days": 1, "job_level": "employee", field: amount}
    with pytest.raises(ValueError):
        TravelBudgetTool().execute(arguments)


def test_explicit_zero_expenses_are_preserved():
    result = execute({
        "city": "上海", "days": 1, "job_level": "employee",
        "lodging_per_day": 0, "meal_per_day": 0, "transport_total": 0,
    })

    assert result["actual_expenses"] == {
        "lodging_per_day": 0, "meal_per_day": 0, "transport_total": 0,
    }
    assert result["actual_total"] == 0
    assert result["over_limit_items"] == []


def test_over_limit_items_and_amounts_use_existing_calculation_rules():
    result = execute({
        "city": "上海", "days": 2, "job_level": "employee",
        "lodging_per_day": 351, "meal_per_day": 100, "transport_total": 151,
    })

    assert result["policy_budget"] == 1_050
    assert result["actual_total"] == 1_053
    assert result["over_limit_items"] == ["lodging_per_day", "transport_total"]


@pytest.mark.parametrize(
    ("role", "authorized"),
    [("employee", True), ("guest", False)],
)
def test_agent_loop_travel_permission_controls_visibility_and_execution(
    role, authorized
):
    class ToolCall:
        id = "travel_permission_test"
        function = SimpleNamespace(
            name="calculate_travel_budget",
            arguments='{"city":"上海","days":1,"job_level":"employee"}',
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

    class Client:
        def __init__(self):
            self.calls = 0
            self.received_messages = []
            self.received_tools = []
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        def create(self, **kwargs):
            self.calls += 1
            self.received_messages.append(copy.deepcopy(kwargs.get("messages", [])))
            self.received_tools.append(copy.deepcopy(kwargs.get("tools", [])))
            if self.calls == 1:
                message = SimpleNamespace(content=None, tool_calls=[ToolCall()])
            else:
                message = SimpleNamespace(content="denied", tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class RecordingTravelBudgetTool(TravelBudgetTool):
        def __init__(self):
            self.execution_count = 0

        def execute(self, arguments):
            self.execution_count += 1
            return super().execute(arguments)

    tool = RecordingTravelBudgetTool()
    registry = ToolRegistry()
    registry.register(tool)
    messages = [{"role": "user", "content": "计算差旅预算"}]
    original_messages = copy.deepcopy(messages)
    client = Client()
    agent = AgentLoop(
        client=client,
        tool_registry=registry,
        permission_checker=PermissionChecker(),
        max_steps=2,
    )

    agent.run(messages, role=role)

    assert messages == original_messages
    offered_tool_names = {
        schema["function"]["name"] for schema in client.received_tools[0]
    }
    assert ("calculate_travel_budget" in offered_tool_names) is authorized
    assert tool.execution_count == (1 if authorized else 0)
    assert client.calls == 2
    request = client.received_messages[1]
    assistant_index = next(
        index
        for index, message in enumerate(request)
        if message.get("role") == "assistant" and message.get("tool_calls")
    )
    assistant_message = request[assistant_index]
    tool_message = request[assistant_index + 1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == assistant_message["tool_calls"][0]["id"]
    if authorized:
        assert json.loads(tool_message["content"])["actual_total"] == 600
    else:
        assert "Permission denied" in tool_message["content"]
        assert "actual_total" not in tool_message["content"]


@pytest.mark.parametrize("arguments", [
    {"city": "北京", "days": 0, "job_level": "employee"},
    {"city": "北京", "days": 3, "job_level": "intern"},
    {"city": "", "days": 3, "job_level": "employee"},
])
def test_travel_budget_rejects_invalid_inputs(arguments):
    with pytest.raises(ValueError):
        TravelBudgetTool().execute(arguments)
