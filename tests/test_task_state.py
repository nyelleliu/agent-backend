from app.planner.state import TaskState


def test_task_state_tracks_progress():
    state = TaskState([
        "查询数据",
        "分析数据",
        "生成报告",
    ])

    assert state.current_step().description == "查询数据"

    state.start_step(0)

    assert state.steps[0].status == "running"

    state.complete_step(0)

    assert state.steps[0].status == "completed"
    assert state.current_step().description == "分析数据"

    state.start_step(1)
    state.complete_step(1)

    state.start_step(2)
    state.complete_step(2)

    assert state.is_completed() is True


def test_task_state_can_mark_failed():
    state = TaskState([
        "查询数据",
        "分析数据",
    ])

    state.start_step(0)
    state.fail_step(0)

    assert state.steps[0].status == "failed"


def test_task_state_can_retry():
    state = TaskState([
        "查询数据",
        "分析数据",
    ])

    state.start_step(0)
    state.fail_step(0)
    state.retry_step(0)

    assert state.steps[0].status == "pending"
    assert state.steps[0].retry_count == 1
