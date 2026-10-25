import json
from types import SimpleNamespace

import pytest

from app.agent.request_budget import ContextBudgetExceeded, serialized_request_size
from app.permissions.checker import PermissionChecker
from app.planner.planner import Planner


class FakeRegistry:
    def __init__(self, names):
        self.names = names

    def schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Fake {name}",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            }
            for name in self.names
        ]


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.create
            )
        )

    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '[\n'
                            '  {"description": "查询销售数据", '
                            '"tools": ["data_analysis"]},\n'
                            '  {"description": "分析销售变化", '
                            '"tools": ["data_analysis"]},\n'
                            '  {"description": "生成分析报告", '
                            '"tools": []}\n'
                            ']'
                        )
                    )
                )
            ]
        )


def test_planner_creates_plan():
    registry = FakeRegistry(["data_analysis"])

    planner = Planner(
        FakeClient(),
        tool_registry=registry,
        permission_checker=PermissionChecker(),
    )

    plan = planner.create_plan(
        "分析今年销售额下降的原因",
        role="admin",
    )

    assert plan == [
        {
            "description": "查询销售数据",
            "tools": ["data_analysis"],
        },
        {
            "description": "分析销售变化",
            "tools": ["data_analysis"],
        },
        {
            "description": "生成分析报告",
            "tools": [],
        },
    ]


class InvalidToolClient:
    def __init__(self):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.create
            )
        )

    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '[{"description": "执行任务", '
                            '"tools": ["fake_tool"]}]'
                        )
                    )
                )
            ]
        )


def test_planner_rejects_unknown_tool():
    registry = FakeRegistry(["data_analysis"])

    planner = Planner(
        InvalidToolClient(),
        tool_registry=registry,
    )

    with pytest.raises(
        ValueError,
        match="Planner returned unknown tools: fake_tool",
    ):
        planner.create_plan("执行一个任务")


class FixedPlanClient:
    def __init__(self, tools):
        self.tools = tools
        self.last_system_prompt = None
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.create
            )
        )

    def create(self, **kwargs):
        self.last_system_prompt = kwargs["messages"][0]["content"]
        plan = [
            {
                "description": "执行任务",
                "tools": list(self.tools),
            }
        ]

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(plan, ensure_ascii=False)
                    )
                )
            ]
        )


def test_planner_hides_capabilities_the_role_cannot_use():
    client = FixedPlanClient(["data_analysis"])

    planner = Planner(
        client,
        tool_registry=FakeRegistry(["data_analysis", "calculate"]),
        permission_checker=PermissionChecker(),
    )

    plan = planner.create_plan("分析销售额", role="employee")

    assert "calculate" in client.last_system_prompt
    assert "data_analysis" not in client.last_system_prompt
    assert plan == [{"description": "执行任务", "tools": []}]


def test_planner_keeps_tools_the_role_may_use():
    client = FixedPlanClient(["data_analysis"])

    planner = Planner(
        client,
        tool_registry=FakeRegistry(["data_analysis"]),
        permission_checker=PermissionChecker(),
    )

    plan = planner.create_plan("分析销售额", role="admin")

    assert plan == [{"description": "执行任务", "tools": ["data_analysis"]}]


def test_planner_without_role_is_fail_closed():
    client = FixedPlanClient(["data_analysis"])

    planner = Planner(
        client,
        tool_registry=FakeRegistry(["data_analysis"]),
        permission_checker=PermissionChecker(),
    )

    plan = planner.create_plan("分析销售额")

    assert plan == [{"description": "执行任务", "tools": []}]
    assert "data_analysis" not in client.last_system_prompt


def test_planner_without_checker_is_fail_closed():
    client = FixedPlanClient(["data_analysis"])

    planner = Planner(
        client,
        tool_registry=FakeRegistry(["data_analysis"]),
    )

    plan = planner.create_plan("分析销售额", role="admin")

    assert plan == [{"description": "执行任务", "tools": []}]
    assert "data_analysis" not in client.last_system_prompt


def test_planner_mcp_capabilities_use_mcp_permissions():
    client = FixedPlanClient(
        ["calculate", "get_company_info", "delete_everything"]
    )

    planner = Planner(
        client,
        tool_registry=FakeRegistry(["calculate"]),
        mcp_tool_registry=FakeRegistry(
            ["get_company_info", "delete_everything"]
        ),
        permission_checker=PermissionChecker(),
    )

    plan = planner.create_plan("查询公司信息", role="employee")

    prompt = client.last_system_prompt
    assert "calculate" in prompt
    assert "get_company_info" in prompt
    assert "delete_everything" not in prompt

    assert plan == [
        {
            "description": "执行任务",
            "tools": ["calculate", "get_company_info"],
        }
    ]


def test_planner_drops_unauthorized_but_registered_tools():
    client = FixedPlanClient(["data_analysis", "calculate"])

    planner = Planner(
        client,
        tool_registry=FakeRegistry(["data_analysis", "calculate"]),
        permission_checker=PermissionChecker(),
    )

    plan = planner.create_plan("分析数据", role="employee")

    assert plan == [
        {
            "description": "执行任务",
            "tools": ["calculate"],
        }
    ]


class RecordingPlanClient:
    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="[]")
                )
            ]
        )


def test_planner_sends_normal_budgeted_request():
    client = RecordingPlanClient()
    planner = Planner(client, max_context_chars=1000)

    assert planner.create_plan("summarize this", role="admin") == []

    assert len(client.calls) == 1
    request = client.calls[0]
    assert set(request) == {"model", "messages"}
    assert request["model"] == "deepseek-chat"
    assert serialized_request_size(
        request["model"],
        request["messages"],
    ) <= 1000


def test_planner_schema_over_budget_rejects_before_fake_client_call():
    class LargeDescriptionRegistry(FakeRegistry):
        def schemas(self):
            schemas = super().schemas()
            schemas[0]["function"]["description"] = "x" * 1400
            return schemas

    client = RecordingPlanClient()
    planner = Planner(
        client,
        tool_registry=LargeDescriptionRegistry(["data_analysis"]),
        permission_checker=PermissionChecker(),
        max_context_chars=1000,
    )

    with pytest.raises(ContextBudgetExceeded, match="planner request"):
        planner.create_plan("summarize this", role="admin")

    assert client.calls == []
