import json
from typing import Any


class Planner:
    def __init__(
        self,
        client,
        tool_registry=None,
        skill_registry=None,
    ):
        self.client = client
        self.tool_registry = tool_registry
        self.skill_registry = skill_registry

    def _get_capabilities(self) -> list[dict[str, Any]]:
        capabilities = []

        if self.tool_registry is not None:
            for schema in self.tool_registry.schemas():
                function = schema["function"]

                capabilities.append({
                    "name": function["name"],
                    "description": function["description"],
                    "type": "tool",
                })

        if self.skill_registry is not None:
            for schema in self.skill_registry.schemas():
                function = schema["function"]

                capabilities.append({
                    "name": function["name"],
                    "description": function["description"],
                    "type": "skill",
                })

        return capabilities

    def create_plan(
        self,
        user_request: str,
    ) -> list[dict[str, Any]]:
        capabilities = self._get_capabilities()

        capabilities_text = json.dumps(
            capabilities,
            ensure_ascii=False,
            indent=2,
        )

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a task planner. "
                        "Break the user's request into a small number "
                        "of clear, executable steps.\n\n"
                        "For every step, specify:\n"
                        "1. description: what the step does\n"
                        "2. tools: the names of tools or skills that "
                        "can execute this step\n\n"
                        "Only use tools or skills from the provided "
                        "capability list.\n"
                        "Return valid JSON only.\n\n"
                        "Format:\n"
                        "[\n"
                        '  {"description": "...", '
                        '"tools": ["..."]}\n'
                        "]\n\n"
                        f"Available capabilities:\n{capabilities_text}"
                    ),
                },
                {
                    "role": "user",
                    "content": user_request,
                },
            ],
        )

        content = response.choices[0].message.content or ""

        try:
            plan = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Planner returned invalid JSON"
            ) from exc

        if not isinstance(plan, list):
            raise ValueError("Planner result must be a list")

        valid_capabilities = {
            capability["name"]
            for capability in capabilities
        }

        for step in plan:
            if not isinstance(step, dict):
                raise ValueError("Each plan step must be an object")

            if not isinstance(step.get("description"), str):
                raise ValueError(
                    "Each plan step needs a description"
                )

            if not isinstance(step.get("tools"), list):
                raise ValueError(
                    "Each plan step needs a tools list"
                )

            unknown_tools = [
                tool_name
                for tool_name in step["tools"]
                if tool_name not in valid_capabilities
            ]

            if unknown_tools:
                raise ValueError(
                    "Planner returned unknown tools: "
                    + ", ".join(unknown_tools)
                )

        return plan
