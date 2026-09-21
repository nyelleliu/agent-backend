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
