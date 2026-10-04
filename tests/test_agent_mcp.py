from types import SimpleNamespace

from app.agent.loop import AgentLoop
from app.mcp.client import MCPClient
from app.mcp.tool_registry import MCPToolRegistry
from app.tools.registry import ToolRegistry


class FakeToolCall:
    id = "mcp_call_1"

    function = SimpleNamespace(
        name="get_company_info",
        arguments='{"company_name": "OpenAI"}',
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
    content = "OpenAI 是一家 AI 研究和部署公司。"
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
    tool_registry = ToolRegistry()

    mcp_client = MCPClient()
    mcp_tool_registry = MCPToolRegistry(mcp_client)
    mcp_tool_registry.refresh()

    return AgentLoop(
        client=client,
        tool_registry=tool_registry,
        mcp_tool_registry=mcp_tool_registry,
        max_steps=5,
    )


def test_agent_can_execute_mcp_tool():
    client = FakeClient()
    agent = create_agent(client)

    messages = [
        {
            "role": "user",
            "content": "查询 OpenAI 的公司信息",
        }
    ]

    reply = agent.run(messages)

    assert reply == "OpenAI 是一家 AI 研究和部署公司。"
    assert client.call_count == 2

    assert messages[-1]["role"] == "tool"
    assert messages[-1]["content"] == (
        "AI research and deployment company."
    )
