from app.planner.state import TaskState


def test_task_state_tracks_progress():
    state = TaskState([
        {
            "description": "查询数据",
            "tools": ["knowledge_search"],
        },
        {
            "description": "分析数据",
            "tools": ["data_analysis"],
        },
        {
            "description": "生成报告",
            "tools": [],
        },
    ])

    assert state.current_step().description == "查询数据"
    assert state.current_step().tools == ["knowledge_search"]

    state.start_step(0)
    assert state.steps[0].status == "running"

    state.complete_step(0)

    assert state.current_step().description == "分析数据"


def test_task_state_can_mark_failed():
    state = TaskState([
        {
            "description": "查询数据",
            "tools": ["knowledge_search"],
        },
        {
            "description": "分析数据",
            "tools": ["data_analysis"],
        },
    ])

    state.start_step(0)
    state.fail_step(0)

    assert state.steps[0].status == "failed"


def test_task_state_can_retry():
    state = TaskState([
        {
            "description": "查询数据",
            "tools": ["knowledge_search"],
        },
        {
            "description": "分析数据",
            "tools": ["data_analysis"],
        },
    ])

    state.start_step(0)
    state.fail_step(0)
    state.retry_step(0)

    assert state.steps[0].status == "pending"
    assert state.steps[0].retry_count == 1


def build_state():
    return TaskState([
        {
            "description": "查询数据",
            "tools": ["knowledge_search"],
        },
        {
            "description": "生成报告",
            "tools": [],
        },
    ])


def test_task_state_summary_reports_progress():
    state = build_state()

    assert state.summary()["status"] == "in_progress"

    state.start_step(0)
    state.complete_step(0)

    assert state.summary() == {
        "status": "in_progress",
        "total": 2,
        "completed": 1,
        "failed": 0,
        "pending": 1,
        "running": 0,
    }


def test_task_state_summary_reports_completion():
    state = build_state()
    state.start_step(0)
    state.complete_step(0)
    state.complete_step(1)

    assert state.summary()["status"] == "completed"
    assert state.is_completed()


def test_task_state_summary_reports_failure():
    state = build_state()
    state.start_step(0)
    state.fail_step(0)

    summary = state.summary()

    assert summary["status"] == "failed"
    assert summary["failed"] == 1


def test_task_state_as_dict_includes_steps():
    state = build_state()
    state.start_step(0)
    state.fail_step(0)
    state.retry_step(0)

    payload = state.as_dict()

    assert payload["status"] == "in_progress"
    assert payload["steps"] == [
        {
            "description": "查询数据",
            "tools": ["knowledge_search"],
            "status": "pending",
            "retry_count": 1,
        },
        {
            "description": "生成报告",
            "tools": [],
            "status": "pending",
            "retry_count": 0,
        },
    ]


def test_empty_task_state_is_completed():
    state = TaskState([])

    assert state.summary()["status"] == "completed"
    assert state.as_dict()["steps"] == []
