from types import SimpleNamespace

from app.agent.loop import AgentLoop
from app.planner.state import TaskState
from app.tools.calculator import CalculatorTool
from app.tools.registry import ToolRegistry


class FakePlanner:
    def create_plan(self, user_request):
        return [
            "执行计算",
            "完成任务",
        ]


class FakeToolCall:
    id = "call_1"

    function = SimpleNamespace(
        name="calculate",
        arguments='{"expression": "100 + 20"}',
    )

    def model_dump(self):
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


class FakeMessageWithTool:
    content = None
    tool_calls = [FakeToolCall()]


class FakeMessageFinal:
    content = "计算结果是 120"
    tool_calls = None


class FakeResponse:
    def __init__(self, message):
        self.choices = [
            SimpleNamespace(message=message)
        ]


class FakeClient:
    def __init__(self):
        self.call_count = 0
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.create
            )
        )

    def create(self, **kwargs):
        self.call_count += 1

        if self.call_count == 1:
            return FakeResponse(FakeMessageWithTool())

        return FakeResponse(FakeMessageFinal())


def test_agent_exposes_task_state():
    client = FakeClient()

    tools = ToolRegistry()
    tools.register(CalculatorTool())

    agent = AgentLoop(
        client=client,
        tool_registry=tools,
        planner=FakePlanner(),
        task_state_class=TaskState,
    )

    messages = [
        {
            "role": "user",
            "content": "计算 100 + 20",
        }
    ]

    reply = agent.run(messages)

    assert reply == "计算结果是 120"
    assert agent.last_task_state is not None

    state = agent.last_task_state

    assert state.steps[0].status == "completed"
    assert state.steps[1].status == "pending"
