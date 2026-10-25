import json
from copy import deepcopy
from types import SimpleNamespace

from app.agent.loop import AgentLoop
from app.mcp.adapter import MCPToolAdapter
from app.mcp.client import MCPClient
from app.mcp.tool_registry import MCPToolRegistry
from app.permissions.checker import PermissionChecker
from app.tools.registry import ToolRegistry


class FakeToolCall:
    id = "mcp_call_1"

    def __init__(self, name="get_company_info"):
        self.function = SimpleNamespace(
            name=name,
            arguments=json.dumps({"company_name": "OpenAI"}),
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
    def __init__(self, name):
        self.content = None
        self.tool_calls = [FakeToolCall(name)]


class FakeMessageFinal:
    content = "OpenAI 是一家 AI 研究和部署公司。"
    tool_calls = None


class FakeResponse:
    def __init__(self, message):
        self.choices = [SimpleNamespace(message=message)]


class FakeLLMClient:
    def __init__(self, tool_name="get_company_info"):
        self.call_count = 0
        self.tool_name = tool_name
        self.requests = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        self.call_count += 1
        self.requests.append(deepcopy(kwargs))

        if self.call_count == 1:
            return FakeResponse(FakeMessageWithTool(self.tool_name))
        return FakeResponse(FakeMessageFinal())


class FakeMCPClient:
    def __init__(self, tools=None, *, list_error=None, call_error=None):
        self.tools = tools if tools is not None else [
            SimpleNamespace(
                name="get_company_info",
                description="Get basic information about a company.",
                input_schema={
                    "type": "object",
                    "properties": {"company_name": {"type": "string"}},
                    "required": ["company_name"],
                },
            )
        ]
        self.list_error = list_error
        self.call_error = call_error
        self.list_tools_calls = 0
        self.tool_calls = []

    def list_tools(self):
        self.list_tools_calls += 1
        if self.list_error is not None:
            raise self.list_error
        return self.tools

    def call_tool(self, name, arguments):
        self.tool_calls.append((name, arguments))
        if self.call_error is not None:
            raise self.call_error
        return "AI research and deployment company."


def create_agent(llm_client, mcp_client):
    registry = MCPToolRegistry(mcp_client)
    registry.refresh()
    agent = AgentLoop(
        client=llm_client,
        tool_registry=ToolRegistry(),
        mcp_tool_registry=registry,
        permission_checker=PermissionChecker(),
        max_steps=5,
    )
    return agent, registry


def test_agent_can_execute_mcp_tool_without_starting_a_process(monkeypatch):
    process_starts = []

    def reject_real_process_start(_client):
        process_starts.append(True)
        raise AssertionError("MCP subprocess must not start in this test")

    monkeypatch.setattr(MCPClient, "start", reject_real_process_start)
    llm_client = FakeLLMClient()
    mcp_client = FakeMCPClient()
    agent, registry = create_agent(llm_client, mcp_client)
    messages = [{
        "role": "user",
        "content": "查询 OpenAI 的公司信息",
    }]
    original_messages = deepcopy(messages)

    reply, _ = agent.run(messages)

    assert reply == "OpenAI 是一家 AI 研究和部署公司。"
    assert llm_client.call_count == 2
    assert process_starts == []
    assert mcp_client.list_tools_calls == 1
    assert mcp_client.tool_calls == [
        ("get_company_info", {"company_name": "OpenAI"})
    ]
    assert isinstance(registry.get("get_company_info"), MCPToolAdapter)
    assert registry.schemas() == [{
        "type": "function",
        "function": {
            "name": "get_company_info",
            "description": "Get basic information about a company.",
            "parameters": mcp_client.tools[0].input_schema,
        },
    }]
    assert any(
        schema["function"]["name"] == "get_company_info"
        for schema in llm_client.requests[0]["tools"]
    )
    assert llm_client.requests[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "mcp_call_1",
        "content": "AI research and deployment company.",
    }
    assert messages == original_messages


def test_unapproved_discovered_mcp_tool_is_not_exposed_or_executed(monkeypatch):
    process_starts = []
    monkeypatch.setattr(
        MCPClient,
        "start",
        lambda _client: process_starts.append(True),
    )
    mcp_client = FakeMCPClient(tools=[SimpleNamespace(
        name="unapproved_tool",
        description="Not authorized for any role.",
        input_schema={"type": "object", "properties": {}},
    )])
    llm_client = FakeLLMClient(tool_name="unapproved_tool")
    agent, registry = create_agent(llm_client, mcp_client)

    reply, _ = agent.run(
        [{"role": "user", "content": "do it"}],
        role="employee",
    )

    assert reply == "OpenAI 是一家 AI 研究和部署公司。"
    assert registry.get("unapproved_tool") is not None
    assert llm_client.requests[0]["tools"] == []
    assert mcp_client.tool_calls == []
    assert "Permission denied" in llm_client.requests[1]["messages"][-1]["content"]
    assert process_starts == []


def test_registry_refresh_failure_is_observable_offline():
    mcp_client = FakeMCPClient(
        list_error=RuntimeError("offline discovery failure")
    )
    registry = MCPToolRegistry(mcp_client)

    try:
        registry.refresh()
    except RuntimeError as exc:
        assert str(exc) == "offline discovery failure"
    else:
        raise AssertionError("registry.refresh must propagate discovery failure")

    assert mcp_client.list_tools_calls == 1
    assert registry.schemas() == []
