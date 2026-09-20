from typing import Any

from app.skills.base import Skill


class DataAnalysisSkill(Skill):
    name = "data_analysis"
    permission = "data_analysis"
    description = "Analyze numeric data and calculate business metrics"

    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A mathematical expression to calculate",
            }
        },
        "required": ["expression"],
    }

    def __init__(self, tool_registry):
        self.tool_registry = tool_registry

    def run(self, arguments: dict[str, Any]) -> Any:
        expression = arguments.get("expression", "")

        if not expression:
            return "Error: expression is required"

        return self.tool_registry.execute(
            "calculate",
            {"expression": expression},
        )

