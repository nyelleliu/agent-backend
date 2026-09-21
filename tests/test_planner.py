from types import SimpleNamespace

import pytest

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
    )

    plan = planner.create_plan(
        "分析今年销售额下降的原因"
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
