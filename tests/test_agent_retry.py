from types import SimpleNamespace

from app.agent.loop import AgentLoop
from app.planner.state import TaskState
from app.tools.registry import ToolRegistry


class FakeToolCall:
    id = "call_1"

    function = SimpleNamespace(
        name="calculate",
        arguments='{"expression": "10 / 0"}',
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


class FakeMessage:
    content = None
    tool_calls = [FakeToolCall()]


class FakeResponse:
    choices = [
        SimpleNamespace(message=FakeMessage())
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
        return FakeResponse()


def test_agent_retries_failed_tool():
    client = FakeClient()

    agent = AgentLoop(
        client=client,
        tool_registry=ToolRegistry(),
        planner=None,
        task_state_class=TaskState,
        max_steps=2,
        max_retries=1,
    )

    messages = [
        {
            "role": "user",
            "content": "计算 10 / 0",
        }
    ]

    agent.planner = SimpleNamespace(
        create_plan=lambda request: [
            {
                "description": "执行计算",
                "tools": ["calculate"],
            }
        ]
    )

    agent.run(messages)

    state = agent.last_task_state

    assert state.steps[0].retry_count == 1
    assert state.steps[0].status == "failed"
    assert client.call_count == 2
