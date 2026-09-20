from types import SimpleNamespace

from app.agent.loop import AgentLoop
from app.tools.calculator import CalculatorTool
from app.tools.registry import ToolRegistry


class FakeToolCall:
    def __init__(self, expression):
        self.id = "call_1"
        self.function = SimpleNamespace(
            name="calculate",
            arguments=f'{{"expression": "{expression}"}}',
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
    def __init__(self, expression):
        self.content = None
        self.tool_calls = [FakeToolCall(expression)]


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
            return FakeResponse(
                FakeMessage("10 / 0")
            )

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="发现工具执行失败",
                        tool_calls=None,
                    )
                )
            ]
        )


def test_agent_returns_tool_error_to_llm():
    client = FakeClient()

    registry = ToolRegistry()
    registry.register(CalculatorTool())

    agent = AgentLoop(
        client=client,
        tool_registry=registry,
        max_steps=2,
    )

    messages = [
        {
            "role": "user",
            "content": "计算 10 / 0",
        }
    ]

    reply = agent.run(messages)

    assert reply == "发现工具执行失败"

    tool_message = messages[-1]

    assert tool_message["role"] == "tool"
    assert "Tool execution failed." in tool_message["content"]
    assert "division by zero" in tool_message["content"]
