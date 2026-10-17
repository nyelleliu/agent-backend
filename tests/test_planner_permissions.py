"""Planner output must never grant execution authority.

The plan only guides the model; the AgentLoop re-checks every tool
call against the PermissionChecker. These tests pin that boundary:
even a plan that names an unauthorised tool (including one injected
by a planner that skips filtering) cannot make the execution layer
run it.
"""

import json
from types import SimpleNamespace

from app.agent.loop import AgentLoop
from app.permissions.checker import PermissionChecker
from app.planner.planner import Planner
from app.skills.base import Skill
from app.skills.registry import SkillRegistry
from app.tools.registry import ToolRegistry


class FakeSkill(Skill):
    name = "data_analysis"
    description = "Test data analysis skill"
    permission = "data_analysis"

    def __init__(self):
        self.executed = False

    def run(self, arguments):
        self.executed = True
        return "analysis result"


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


class RogueToolCall:
    id = "call_1"

    function = SimpleNamespace(
        name="data_analysis",
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


class RogueMessage:
    content = None
    tool_calls = [RogueToolCall()]


class RogueResponse:
    choices = [
        SimpleNamespace(message=RogueMessage())
    ]


class RogueClient:
    """Ignores the offered tool list and calls data_analysis anyway."""

    def __init__(self):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.create
            )
        )

    def create(self, **kwargs):
        return RogueResponse()


class FixedPlanClient:
    def __init__(self, tools):
        self.tools = tools
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.create
            )
        )

    def create(self, **kwargs):
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


class UnfilteredPlanner:
    """Planning stub that skips permission filtering on purpose."""

    def create_plan(self, user_request, role=None):
        return [
            {
                "description": "执行任务",
                "tools": ["data_analysis"],
            }
        ]


def build_agent(skill, planner):
    skill_registry = SkillRegistry()
    skill_registry.register(skill)

    return AgentLoop(
        RogueClient(),
        FakeRegistry([]),
        skill_registry,
        PermissionChecker(),
        planner=planner,
        max_steps=1,
    )


def test_unfiltered_plan_cannot_bypass_execution_permissions():
    skill = FakeSkill()
    agent = build_agent(skill, UnfilteredPlanner())

    messages = [{"role": "user", "content": "分析数据"}]
    agent.run(messages, role="employee")

    assert skill.executed is False
    assert "Permission denied" in messages[-1]["content"]


def test_real_planner_plan_cannot_bypass_execution_permissions():
    skill = FakeSkill()

    planner = Planner(
        FixedPlanClient(["data_analysis"]),
        tool_registry=FakeRegistry(["data_analysis"]),
        permission_checker=PermissionChecker(),
    )

    agent = build_agent(skill, planner)

    messages = [{"role": "user", "content": "分析数据"}]
    agent.run(messages, role="employee")

    assert skill.executed is False
    assert "Permission denied" in messages[-1]["content"]


def test_authorised_role_still_executes():
    skill = FakeSkill()

    planner = Planner(
        FixedPlanClient(["data_analysis"]),
        tool_registry=FakeRegistry(["data_analysis"]),
        permission_checker=PermissionChecker(),
    )

    agent = build_agent(skill, planner)

    messages = [{"role": "user", "content": "分析数据"}]
    agent.run(messages, role="finance")

    assert skill.executed is True
    assert messages[-1]["content"] == "analysis result"
