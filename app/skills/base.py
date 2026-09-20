from abc import ABC, abstractmethod
from typing import Any


class Skill(ABC):
    name: str
    description: str
    permission: str
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
    }

    @abstractmethod
    def run(self, arguments: dict[str, Any]) -> Any:
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
