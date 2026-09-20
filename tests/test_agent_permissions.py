from types import SimpleNamespace

from app.agent.loop import AgentLoop
from app.permissions.checker import PermissionChecker
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


class FakeToolCall:
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


class FakeMessage:
    content = None
    tool_calls = [FakeToolCall()]


class FakeResponse:
    choices = [
        SimpleNamespace(message=FakeMessage())
    ]


class TestClient:
    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                return FakeResponse()


def create_agent(skill):
    skill_registry = SkillRegistry()
    skill_registry.register(skill)

    return AgentLoop(
        TestClient(),
        ToolRegistry(),
        skill_registry,
        PermissionChecker(),
        max_steps=1,
    )


def test_employee_cannot_execute_data_analysis():
    skill = FakeSkill()
    agent = create_agent(skill)

    messages = [
        {
            "role": "user",
            "content": "分析数据",
        }
    ]

    agent.run(messages, role="employee")

    assert skill.executed is False
    assert "Permission denied" in messages[-1]["content"]


def test_finance_can_execute_data_analysis():
    skill = FakeSkill()
    agent = create_agent(skill)

    messages = [
        {
            "role": "user",
            "content": "分析数据",
        }
    ]

    agent.run(messages, role="finance")

    assert skill.executed is True
    assert messages[-1]["content"] == "analysis result"
