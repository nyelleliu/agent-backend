from typing import Any

from mcp import ClientSession

from app.mcp.adapter import MCPToolAdapter


class MCPToolRegistry:
    def __init__(self, session: ClientSession):
        self.session = session
        self._tools: dict[str, Any] = {}

    async def refresh(self) -> None:
        result = await self.session.list_tools()

        self._tools = {
            tool.name: tool
            for tool in result.tools
        }

    def get(self, name: str) -> Any | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        schemas = []

        for tool in self._tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                },
            })

        return schemas
