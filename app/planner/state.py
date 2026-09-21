from dataclasses import dataclass


@dataclass
class PlanStep:
    description: str
    tools: list[str]
    status: str = "pending"
    retry_count: int = 0


class TaskState:
    def __init__(self, steps: list[dict]):
        self.steps = [
            PlanStep(
                description=step["description"],
                tools=step.get("tools", []),
            )
            for step in steps
        ]

    def start_step(self, index: int):
        self.steps[index].status = "running"

    def complete_step(self, index: int):
        self.steps[index].status = "completed"

    def fail_step(self, index: int):
        self.steps[index].status = "failed"

    def retry_step(self, index: int):
        self.steps[index].retry_count += 1
        self.steps[index].status = "pending"

    def current_step(self) -> PlanStep | None:
        for step in self.steps:
            if step.status == "pending":
                return step

        return None

    def find_step_by_tool(self, tool_name: str) -> int | None:
        for index, step in enumerate(self.steps):
            if (
                step.status == "pending"
                and tool_name in step.tools
            ):
                return index

        return None

    def is_completed(self) -> bool:
        return all(
            step.status == "completed"
            for step in self.steps
        )

    def as_text(self) -> str:
        return "\n".join(
            f"{i}. [{step.status}] {step.description} "
            f"tools={step.tools} "
            f"(retries: {step.retry_count})"
            for i, step in enumerate(self.steps, start=1)
        )
