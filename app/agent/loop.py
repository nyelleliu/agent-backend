from typing import Any


class AgentLoop:
    def __init__(
        self,
        client,
        tool_registry,
        skill_registry=None,
        permission_checker=None,
        planner=None,
        task_state_class=None,
        max_steps: int = 5,
        max_retries: int = 2,
    ):
        self.client = client
        self.tool_registry = tool_registry
        self.skill_registry = skill_registry
        self.permission_checker = permission_checker
        self.planner = planner
        self.task_state_class = task_state_class
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.last_task_state = None

    def run(
        self,
        messages: list[dict[str, Any]],
        role: str = "employee",
    ) -> str:
        reply = "Sorry, I couldn't complete this after several tool calls."

        self.last_task_state = None
        task_state = None

        if self.planner is not None:
            user_request = self._get_latest_user_message(messages)

            if user_request:
                plan = self.planner.create_plan(user_request)

                if plan:
                    if self.task_state_class is not None:
                        task_state = self.task_state_class(plan)
                        self.last_task_state = task_state
                        plan_text = task_state.as_text()
                    else:
                        plan_text = "\n".join(
                            f"{i}. [pending] {step}"
                            for i, step in enumerate(plan, start=1)
                        )

                    messages.insert(
                        0,
                        {
                            "role": "system",
                            "content": (
                                "Current task plan:\n"
                                f"{plan_text}\n\n"
                                "Use this plan as guidance, but adjust "
                                "your actions when necessary."
                            ),
                        },
                    )

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
                current_index = None

                if task_state is not None:
                    current_index = self._current_step_index(task_state)

                    if current_index is not None:
                        task_state.start_step(current_index)

                name = tc.function.name
                arguments = tc.function.arguments

                if (
                    self.skill_registry is not None
                    and self.skill_registry.get(name) is not None
                ):
                    skill = self.skill_registry.get(name)

                    if (
                        self.permission_checker is not None
                        and not self.permission_checker.has_permission(
                            role,
                            skill.permission,
                        )
                    ):
                        result = (
                            f"Permission denied: role '{role}' "
                            f"cannot use skill '{name}'"
                        )
                    else:
                        try:
                            result = self.skill_registry.execute(
                                name,
                                self._parse_arguments(arguments),
                            )
                        except Exception as exc:
                            result = f"Error: {exc}"

                else:
                    result = self.tool_registry.execute(
                        name,
                        arguments,
                    )

                is_error = self._is_error_result(result)

                if task_state is not None and current_index is not None:
                    if is_error:
                        task_state.fail_step(current_index)

                        if (
                            task_state.steps[current_index].retry_count
                            < self.max_retries
                        ):
                            task_state.retry_step(current_index)
                    else:
                        task_state.complete_step(current_index)

                if is_error:
                    result = (
                        "Tool execution failed.\n"
                        f"Reason: {result}\n"
                        "Please inspect the error and retry with "
                        "corrected arguments if possible."
                    )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })

        return reply

    @staticmethod
    def _is_error_result(result: Any) -> bool:
        if not isinstance(result, str):
            return False

        error_prefixes = (
            "Error:",
            "Permission denied:",
            "unknown tool",
            "unknown skill",
        )

        return result.startswith(error_prefixes)

    @staticmethod
    def _current_step_index(task_state) -> int | None:
        for index, step in enumerate(task_state.steps):
            if step.status == "pending":
                return index

        return None

    @staticmethod
    def _get_latest_user_message(
        messages: list[dict[str, Any]],
    ) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get("content", "")

        return ""

    @staticmethod
    def _parse_arguments(
        arguments: str | dict[str, Any],
    ) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments

        import json

        if not arguments:
            return {}

        result = json.loads(arguments)

        if not isinstance(result, dict):
            raise ValueError("tool arguments must be a JSON object")

        return result
