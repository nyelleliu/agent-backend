import logging
from typing import Any

from app.core.errors import format_tool_error, log_tool_error
from app.mcp.adapter import MCPToolAdapter
from app.mcp.client import MCPClient


logger = logging.getLogger(__name__)


class MCPToolRegistry:
    def __init__(self, client: MCPClient):
        self.client = client
        self._tools: dict[str, MCPToolAdapter] = {}

    def refresh(self) -> None:
        tools = self.client.list_tools()

        self._tools = {
            tool.name: MCPToolAdapter(
                client=self.client,
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.input_schema,
            )
            for tool in tools
        }

    def get(self, name: str) -> MCPToolAdapter | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [
            tool.openai_schema() for tool in self._tools.values()
        ]

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        tool = self.get(name)

        if tool is None:
            return "unknown MCP tool"

        try:
            return tool.execute(arguments)
        except Exception as exc:
            # The model sees a sanitized category (for example
            # "timeout_error"), never the transport-level detail.
            log_tool_error(logger, exc, tool_name=name)

            return format_tool_error(exc, tool_name=name)
