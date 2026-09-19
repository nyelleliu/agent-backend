from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]

    @abstractmethod
    def execute(self, arguments: dict[str, Any]) -> Any:
        raise NotImplementedError

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be a JSON object")

        schema = self.parameters or {}
        properties = schema.get("properties") or {}
        required = schema.get("required") or []

        unknown = set(arguments) - set(properties)
        if unknown:
            raise ValueError(f"unexpected arguments: {', '.join(sorted(unknown))}")

        missing = [name for name in required if name not in arguments]
        if missing:
            raise ValueError(f"missing required arguments: {', '.join(missing)}")

        return arguments


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.openai_schema() for tool in self._tools.values()]

    def execute(self, name: str, raw_arguments: str | dict[str, Any] | None = None) -> Any:
        tool = self.get(name)
        if tool is None:
            return "unknown tool"

        try:
            if raw_arguments is None or raw_arguments == "":
                arguments: dict[str, Any] = {}
            elif isinstance(raw_arguments, dict):
                arguments = raw_arguments
            else:
                parsed = json.loads(raw_arguments)
                if parsed is None:
                    arguments = {}
                elif not isinstance(parsed, dict):
                    return "Error: tool arguments must be a JSON object"
                else:
                    arguments = parsed
            arguments = tool.validate_arguments(arguments)
            return tool.execute(arguments)
        except json.JSONDecodeError:
            return "Error: invalid JSON arguments"
        except ZeroDivisionError:
            return "Error: division by zero"
        except Exception as exc:
            return f"Error: {exc}"
