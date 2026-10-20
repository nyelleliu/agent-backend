import threading
import time
from types import SimpleNamespace

from app.agent.loop import AgentLoop
from app.planner.state import TaskState
from app.tools.registry import ToolRegistry


class RequestAwarePlanner:
    """Plan whose content is derived from the current request."""

    def create_plan(self, user_request, role=None):
        return [
            {
                "description": f"handle {user_request}",
                "tools": [],
            }
        ]


class SlowFinalClient:
    """Always answers, but slowly, to force thread interleaving."""

    def __init__(self, delay=0.05):
        self.delay = delay

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        time.sleep(self.delay)

        message = SimpleNamespace(
            content="done",
            tool_calls=None,
        )

        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )


def build_agent():
    return AgentLoop(
        client=SlowFinalClient(),
        tool_registry=ToolRegistry(),
        planner=RequestAwarePlanner(),
        task_state_class=TaskState,
    )


def test_concurrent_runs_return_their_own_task_state():
    agent = build_agent()

    errors = []
    barrier = threading.Barrier(4)

    def worker(index):
        barrier.wait()

        try:
            _, state = agent.run(
                [{"role": "user", "content": f"req-{index}"}]
            )

            assert state is not None
            assert state.steps[0].description == f"handle req-{index}"
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(index,))
        for index in range(4)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert errors == []


def test_sequential_runs_do_not_leak_previous_state():
    agent = build_agent()

    _, first_state = agent.run(
        [{"role": "user", "content": "first"}]
    )

    _, second_state = agent.run(
        [{"role": "user", "content": "second"}]
    )

    assert first_state is not second_state
    assert first_state.steps[0].description == "handle first"
    assert second_state.steps[0].description == "handle second"
