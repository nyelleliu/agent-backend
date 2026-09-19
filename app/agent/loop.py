from typing import Any


class AgentLoop:
    def __init__(
        self,
        client,
        tool_registry,
        skill_registry=None,
        max_steps: int = 5,
    ):
        self.client = client
        self.tool_registry = tool_registry
        self.skill_registry = skill_registry
        self.max_steps = max_steps

    def run(self, messages: list[dict[str, Any]]) -> str:
        reply = "Sorry, I couldn't complete this after several tool calls."

        for _ in range(self.max_steps):
            capabilities = self.tool_registry.schemas()

            if self.skill_registry is not None:
                capabilities += self.skill_registry.schemas()

            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=capabilities,
            )

            msg = response.choices[0].message

            if not msg.tool_calls:
                return msg.content

            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    tc.model_dump()
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                name = tc.function.name
                arguments = tc.function.arguments

                if (
                    self.skill_registry is not None
                    and self.skill_registry.get(name) is not None
                ):
                    result = self.skill_registry.execute(
                        name,
                        self._parse_arguments(arguments),
                    )
                else:
                    result = self.tool_registry.execute(
                        name,
                        arguments,
                    )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })

        return reply

    @staticmethod
    def _parse_arguments(arguments: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments

        import json

        if not arguments:
            return {}

        result = json.loads(arguments)

        if not isinstance(result, dict):
            raise ValueError("tool arguments must be a JSON object")

        return result
