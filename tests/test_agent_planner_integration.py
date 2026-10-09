from types import SimpleNamespace

from app.agent.loop import AgentLoop
from app.planner.state import TaskState
from app.tools.registry import ToolRegistry


class FakeToolCall:
    id = "call_1"

    function = SimpleNamespace(
        name="calculate",
        arguments='{"expression": "1 + 1"}',
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


class FakeResponse:
    def __init__(self, message):
        self.choices = [SimpleNamespace(message=message)]


class ScriptedClient:
    def __init__(self, always_tool_call=False):
        self.call_count = 0
        self.always_tool_call = always_tool_call
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        self.call_count += 1

        if self.always_tool_call:
            message = SimpleNamespace(content=None, tool_calls=[FakeToolCall()])
        elif self.call_count == 1:
            message = SimpleNamespace(content=None, tool_calls=[FakeToolCall()])
        else:
            message = SimpleNamespace(content="最终答案", tool_calls=None)

        return FakeResponse(message)


class BrokenPlanner:
    def create_plan(self, user_request, role=None):
        raise ValueError("Planner returned invalid JSON")


class NotAListPlanner:
    def create_plan(self, user_request, role=None):
        return {"description": "not a list"}


class RecordingPlanner:
    def __init__(self, steps=1):
        self.calls = []
        self.steps = steps

    def create_plan(self, user_request, role=None):
        self.calls.append({"request": user_request, "role": role})

        return [
            {"description": f"step {index}", "tools": []}
            for index in range(self.steps)
        ]


def build_agent(planner, client=None, **kwargs):
    return AgentLoop(
        client=client or ScriptedClient(),
        tool_registry=ToolRegistry(),
        planner=planner,
        task_state_class=TaskState,
        **kwargs,
    )


def run(agent, content="随便问点什么"):
    return agent.run([{"role": "user", "content": content}])


def test_agent_survives_planner_failure():
    client = ScriptedClient()
    agent = build_agent(BrokenPlanner(), client=client)

    reply = run(agent, "计算 1 + 1")

    assert reply == "最终答案"
    assert client.call_count == 2
    assert agent.last_task_state is None


def test_agent_survives_invalid_planner_result():
    agent = build_agent(NotAListPlanner())

    reply = run(agent)

    assert reply == "最终答案"
    assert agent.last_task_state is None


def test_agent_passes_role_to_planner():
    planner = RecordingPlanner()
    agent = build_agent(planner)

    run(agent)

    assert planner.calls == [
        {"request": "随便问点什么", "role": "employee"}
    ]


def test_agent_allows_more_steps_than_default_for_long_plan():
    planner = RecordingPlanner(steps=10)
    client = ScriptedClient(always_tool_call=True)
    agent = build_agent(planner, client=client)

    reply = run(agent)

    assert reply == (
        "Sorry, I couldn't complete this after several tool calls."
    )
    assert client.call_count == 11
    assert agent.last_task_state.summary()["total"] == 10


def test_agent_uses_default_budget_for_short_plan():
    planner = RecordingPlanner(steps=2)
    client = ScriptedClient(always_tool_call=True)
    agent = build_agent(planner, client=client)

    run(agent)

    assert client.call_count == 5


def test_agent_does_not_inject_plan_when_planner_fails():
    messages = [{"role": "user", "content": "问题"}]
    agent = build_agent(BrokenPlanner())

    agent.run(messages)

    assert messages[0] == {"role": "user", "content": "问题"}
