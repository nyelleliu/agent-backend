from types import SimpleNamespace

from app.agent.loop import AgentLoop
from app.tools.registry import ToolRegistry


class FakePlanner:
    def create_plan(self, user_request):
        assert user_request == "计算 100 + 20"

        return [
            {
                "description": "理解计算任务",
                "tools": [],
            },
            {
                "description": "执行计算",
                "tools": [],
            },
            {
                "description": "返回结果",
                "tools": [],
            },
        ]


class FakeMessage:
    content = "计算结果是 120"
    tool_calls = None


class FakeResponse:
    choices = [
        SimpleNamespace(message=FakeMessage())
    ]


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.create
            )
        )

    def create(self, **kwargs):
        messages = kwargs["messages"]

        assert messages[0]["role"] == "system"
        assert "理解计算任务" in messages[0]["content"]
        assert "执行计算" in messages[0]["content"]

        return FakeResponse()


def test_agent_uses_planner():
    client = FakeClient()

    agent = AgentLoop(
        client=client,
        tool_registry=ToolRegistry(),
        planner=FakePlanner(),
    )

    messages = [
        {
            "role": "user",
            "content": "计算 100 + 20",
        }
    ]

    reply = agent.run(messages)

    assert reply == "计算结果是 120"
