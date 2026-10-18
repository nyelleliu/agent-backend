"""Role-filtered capability lists (P11).

The model is only *offered* tools, skills and MCP tools the current
role may execute. Execution is still checked separately on every call
(tests/test_rag_injection.py), so this is a second line of defence:
without it, every user could read the full tool inventory — including
admin-only skills and MCP tools that were never granted to anyone.
"""

from types import SimpleNamespace

from app.agent.loop import AgentLoop
from app.permissions.checker import PermissionChecker
from app.skills.base import Skill
from app.skills.registry import SkillRegistry
from app.tools.calculator import CalculatorTool
from app.tools.registry import ToolRegistry


class RecordingSkill(Skill):
    def __init__(self, name, permission):
        self.name = name
        self.permission = permission
        self.description = f"{name} skill"
        self.parameters = {"type": "object", "properties": {}}
        self.executed = False

    def run(self, arguments):
        self.executed = True
        return "skill result"


class MultiMCPRegistry:
    """Minimal stand-in for MCPToolRegistry (no subprocess)."""

    def __init__(self, names):
        self.names = list(names)
        self.executed = []

    def get(self, name):
        return object() if name in self.names else None

    def schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": "",
                    "parameters": {},
                },
            }
            for name in self.names
        ]

    def execute(self, name, arguments):
        self.executed.append((name, arguments))
        return "mcp result"


class RecordingClient:
    """LLM that records the tools it was offered, then answers."""

    def __init__(self):
        self.offered = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        self.offered.append(kwargs.get("tools") or [])

        message = SimpleNamespace(content="final answer", tool_calls=None)

        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _registries():
    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())

    skill_registry = SkillRegistry()
    skill_registry.register(
        RecordingSkill("knowledge_search", "knowledge_search")
    )
    skill_registry.register(RecordingSkill("data_analysis", "data_analysis"))

    mcp_registry = MultiMCPRegistry(
        ["get_company_info", "internal_admin_tool"]
    )

    return tool_registry, skill_registry, mcp_registry


def _names(schemas):
    return {schema["function"]["name"] for schema in schemas}


def test_employee_capabilities_exclude_admin_only_and_unlisted_tools():
    tool_registry, skill_registry, mcp_registry = _registries()

    agent = AgentLoop(
        client=RecordingClient(),
        tool_registry=tool_registry,
        skill_registry=skill_registry,
        mcp_tool_registry=mcp_registry,
        permission_checker=PermissionChecker(),
    )

    names = _names(agent.capabilities_for_role("employee"))

    # Employee grants: knowledge_search, calculate, get_current_time.
    assert "calculate" in names
    assert "knowledge_search" in names

    # Not offered to an employee:
    assert "data_analysis" not in names

    # MCP tools follow the explicit MCP allow-list: get_company_info is
    # granted to every role, internal_admin_tool to none.
    assert "get_company_info" in names
    assert "internal_admin_tool" not in names


def test_admin_sees_the_admin_level_skill_but_still_not_unlisted_mcp():
    tool_registry, skill_registry, mcp_registry = _registries()

    agent = AgentLoop(
        client=RecordingClient(),
        tool_registry=tool_registry,
        skill_registry=skill_registry,
        mcp_tool_registry=mcp_registry,
        permission_checker=PermissionChecker(),
    )

    names = _names(agent.capabilities_for_role("admin"))

    assert "data_analysis" in names
    assert "calculate" in names
    assert "get_company_info" in names

    # An MCP tool that nobody granted stays invisible even to admin.
    assert "internal_admin_tool" not in names


def test_missing_permission_checker_exposes_nothing():
    tool_registry, skill_registry, mcp_registry = _registries()

    agent = AgentLoop(
        client=RecordingClient(),
        tool_registry=tool_registry,
        skill_registry=skill_registry,
        mcp_tool_registry=mcp_registry,
        permission_checker=None,
    )

    assert agent.capabilities_for_role("admin") == []
    assert agent.capabilities_for_role("employee") == []


def test_missing_role_exposes_nothing():
    tool_registry, skill_registry, mcp_registry = _registries()

    agent = AgentLoop(
        client=RecordingClient(),
        tool_registry=tool_registry,
        skill_registry=skill_registry,
        mcp_tool_registry=mcp_registry,
        permission_checker=PermissionChecker(),
    )

    assert agent.capabilities_for_role(None) == []
    assert agent.capabilities_for_role("") == []


def test_the_model_is_offered_only_the_filtered_list():
    tool_registry, skill_registry, mcp_registry = _registries()

    client = RecordingClient()
    agent = AgentLoop(
        client=client,
        tool_registry=tool_registry,
        skill_registry=skill_registry,
        mcp_tool_registry=mcp_registry,
        permission_checker=PermissionChecker(),
    )

    agent.run([{"role": "user", "content": "hello"}], role="employee")

    assert client.offered
    offered = _names(client.offered[0])

    assert "data_analysis" not in offered
    assert "internal_admin_tool" not in offered
    assert "calculate" in offered
