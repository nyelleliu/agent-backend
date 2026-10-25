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
        self.execution_count = 0

    def run(self, arguments):
        self.executed = True
        self.execution_count += 1
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
    def __init__(self, message):
        self.choices = [SimpleNamespace(message=message)]


class FakeFinalMessage:
    content = "done"
    tool_calls = None


class FakeClient:
    def __init__(self, tool_call=None):
        self.calls = []
        self.tool_call = tool_call or FakeToolCall()
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = (
            SimpleNamespace(content=None, tool_calls=[self.tool_call])
            if len(self.calls) == 1
            else FakeFinalMessage()
        )
        return FakeResponse(message)


def create_agent(skill, client=None):
    skill_registry = SkillRegistry()
    skill_registry.register(skill)
    client = client or FakeClient()

    agent = AgentLoop(
        client,
        ToolRegistry(),
        skill_registry,
        PermissionChecker(),
        max_steps=2,
    )
    return agent, client


class AliasedFakeSkill(FakeSkill):
    name = "analysis_alias"
    permission = "data_analysis"


class AliasedFakeToolCall(FakeToolCall):
    function = SimpleNamespace(
        name="analysis_alias",
        arguments='{"expression": "100 + 20"}',
    )


def test_skill_visibility_uses_permission_identity_and_keeps_public_name():
    skill = AliasedFakeSkill()
    authorized_agent, authorized_client = create_agent(
        skill,
        FakeClient(AliasedFakeToolCall()),
    )

    authorized_agent.run([{"role": "user", "content": "analyze"}], role="finance")

    authorized_names = [
        schema["function"]["name"]
        for schema in authorized_client.calls[0]["tools"]
    ]
    assert authorized_names == ["analysis_alias"]
    assert skill.execution_count == 1

    denied_skill = AliasedFakeSkill()
    denied_agent, denied_client = create_agent(
        denied_skill,
        FakeClient(AliasedFakeToolCall()),
    )
    denied_agent.run([{"role": "user", "content": "analyze"}], role="employee")

    denied_names = [
        schema["function"]["name"]
        for schema in denied_client.calls[0]["tools"]
    ]
    assert denied_names == []
    assert denied_skill.execution_count == 0
    assert "Reason: Permission denied:" in denied_client.calls[1]["messages"][-1]["content"]


def test_employee_cannot_execute_data_analysis():
    skill = FakeSkill()
    agent, client = create_agent(skill)

    messages = [
        {
            "role": "user",
            "content": "分析数据",
        }
    ]

    original_messages = [dict(message) for message in messages]
    agent.run(messages, role="employee")

    assert skill.executed is False
    assert skill.execution_count == 0
    assert client.calls[0]["tools"] == []
    assert "Reason: Permission denied:" in client.calls[1]["messages"][-1]["content"]
    assert messages == original_messages


def test_finance_can_execute_data_analysis():
    skill = FakeSkill()
    agent, client = create_agent(skill)

    messages = [
        {
            "role": "user",
            "content": "分析数据",
        }
    ]

    original_messages = [dict(message) for message in messages]
    agent.run(messages, role="finance")

    assert skill.executed is True
    assert skill.execution_count == 1
    assert len(client.calls) == 2
    assert any(
        schema["function"]["name"] == "data_analysis"
        for schema in client.calls[0]["tools"]
    )
    assert client.calls[1]["messages"][-1]["content"] == "analysis result"
    assert messages == original_messages
