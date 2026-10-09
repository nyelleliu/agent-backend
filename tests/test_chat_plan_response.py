import main
from app.planner.state import TaskState


def test_chat_response_defaults_to_reply_only():
    assert main.build_chat_response("你好") == {"reply": "你好"}


def test_chat_response_can_include_plan():
    state = TaskState([
        {
            "description": "查询数据",
            "tools": ["knowledge_search"],
        },
    ])

    payload = main.build_chat_response(
        "你好",
        include_plan=True,
        task_state=state,
    )

    assert payload["reply"] == "你好"
    assert payload["plan"]["status"] == "in_progress"
    assert payload["plan"]["total"] == 1
    assert payload["plan"]["steps"] == [
        {
            "description": "查询数据",
            "tools": ["knowledge_search"],
            "status": "pending",
            "retry_count": 0,
        },
    ]


def test_chat_response_plan_is_none_without_planner():
    payload = main.build_chat_response(
        "你好",
        include_plan=True,
        task_state=None,
    )

    assert payload == {"reply": "你好", "plan": None}
