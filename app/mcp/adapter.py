from typing import Any

from app.mcp.client import MCPClient


class MCPToolAdapter:
    def __init__(
        self,
        client: MCPClient,
        name: str,
        description: str,
        input_schema: dict[str, Any],
    ):
        self.client = client
        self.name = name
        self.description = description
        self.parameters = input_schema

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(
        self,
        arguments: dict[str, Any],
    ) -> str:
        return self.client.call_tool(
            self.name,
            arguments,
        )
