from types import SimpleNamespace

from app.agent.loop import AgentLoop
from app.permissions.checker import PermissionChecker
from app.planner.state import TaskState
from app.tools.calculator import CalculatorTool
from app.tools.registry import ToolRegistry


class FakePlanner:
    def create_plan(self, user_request, role=None):
        return [
            {
                "description": "执行计算",
                "tools": ["calculate"],
            },
            {
                "description": "根据计算结果生成最终回答",
                "tools": [],
            },
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
        permission_checker=PermissionChecker(),
        planner=FakePlanner(),
        task_state_class=TaskState,
    )

    messages = [
        {
            "role": "user",
            "content": "计算 100 + 20",
        }
    ]

    reply, state = agent.run(messages)

    assert reply == "计算结果是 120"
    assert state is not None

    # The tool step ran; the tool-less answer step is fulfilled by the
    # final reply itself, so the whole plan settles as completed.
    assert state.steps[0].status == "completed"
    assert state.steps[1].status == "completed"

    summary = state.summary()
    assert summary["status"] == "completed"
    assert summary["total"] == 2
    assert summary["completed"] == 2
    assert summary["pending"] == 0

class OutOfOrderPlanner:
    def create_plan(self, user_request, role=None):
        return [
            {
                "description": "查询销售数据",
                "tools": ["knowledge_search"],
            },
            {
                "description": "执行计算",
                "tools": ["calculate"],
            },
        ]


def test_agent_matches_tool_to_correct_step():
    client = FakeClient()

    tools = ToolRegistry()
    tools.register(CalculatorTool())

    agent = AgentLoop(
        client=client,
        tool_registry=tools,
        permission_checker=PermissionChecker(),
        planner=OutOfOrderPlanner(),
        task_state_class=TaskState,
    )

    messages = [
        {
            "role": "user",
            "content": "查询数据并计算",
        }
    ]

    reply, state = agent.run(messages)

    assert reply == "计算结果是 120"

    # The knowledge_search step was never executed; returning a final
    # answer must NOT mark it completed, and the summary must not
    # report the plan as completed either.
    assert state.steps[0].status == "pending"
    assert state.steps[1].status == "completed"

    summary = state.summary()
    assert summary["status"] == "in_progress"
    assert summary["completed"] == 1
    assert summary["pending"] == 1


class NonAnswerPlanner:
    """Tool-less step whose description is NOT answer-generation work."""

    def create_plan(self, user_request, role=None):
        return [
            {
                "description": "执行计算",
                "tools": ["calculate"],
            },
            {
                "description": "完成任务",
                "tools": [],
            },
        ]


def test_non_answer_toolless_step_stays_pending_after_final_reply():
    client = FakeClient()

    tools = ToolRegistry()
    tools.register(CalculatorTool())

    agent = AgentLoop(
        client=client,
        tool_registry=tools,
        permission_checker=PermissionChecker(),
        planner=NonAnswerPlanner(),
        task_state_class=TaskState,
    )

    messages = [{"role": "user", "content": "计算 100 + 20"}]

    reply, state = agent.run(messages)

    assert reply == "计算结果是 120"
    assert state.steps[0].status == "completed"
    # Ambiguous/vague tool-less steps are left pending rather than
    # being assumed complete.
    assert state.steps[1].status == "pending"
    assert state.summary()["status"] == "in_progress"


def test_task_state_finalize_ignores_failed_and_non_pending_steps():
    state = TaskState(
        [
            {"description": "执行计算", "tools": ["calculate"]},
            {"description": "生成最终回答", "tools": []},
        ]
    )

    state.fail_step(0)

    state.finalize_answer_steps()

    assert state.steps[0].status == "failed"
    assert state.steps[1].status == "completed"
    assert state.summary()["status"] == "failed"


class EmptyReplyClient:
    """Tool call first, then a final message with empty content."""

    def __init__(self):
        self.call_count = 0
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        self.call_count += 1

        if self.call_count == 1:
            return FakeResponse(FakeMessageWithTool())

        return FakeResponse(SimpleNamespace(content="", tool_calls=None))


def test_empty_final_reply_does_not_finalize_plan():
    tools = ToolRegistry()
    tools.register(CalculatorTool())

    agent = AgentLoop(
        client=EmptyReplyClient(),
        tool_registry=tools,
        permission_checker=PermissionChecker(),
        planner=FakePlanner(),
        task_state_class=TaskState,
    )

    reply, state = agent.run(
        [{"role": "user", "content": "计算 100 + 20"}]
    )

    assert reply == ""
    # The executed tool step is settled...
    assert state.steps[0].status == "completed"
    # ...but an empty final reply must not settle answer-type steps.
    assert state.steps[1].status == "pending"
    assert state.summary()["status"] == "in_progress"
