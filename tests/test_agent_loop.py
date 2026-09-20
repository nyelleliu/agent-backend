from types import SimpleNamespace

from app.agent.loop import AgentLoop
from app.tools.registry import ToolRegistry


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

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.create
            )
        )

    def create(self, **kwargs):
        self.call_count += 1

        if self.call_count == 1:
            return FakeResponse(FakeMessageWithTool())

        return FakeResponse(FakeMessageFinal())


def create_agent(client):
    from app.tools.calculator import CalculatorTool

    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())

    return AgentLoop(
        client=client,
        tool_registry=tool_registry,
        max_steps=5,
    )


def test_agent_can_execute_multiple_steps():
    client = FakeClient()
    agent = create_agent(client)

    messages = [
        {
            "role": "user",
            "content": "计算 100 + 20",
        }
    ]

    reply = agent.run(messages)

    assert reply == "计算结果是 120"
    assert client.call_count == 2

    assert messages[-1]["role"] == "tool"
    assert messages[-1]["content"] == "120"


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

    reply = agent.run(messages)

    assert reply == "Sorry, I couldn't complete this after several tool calls."
    assert client.call_count == 5
