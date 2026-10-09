from app.agent.loop import AgentLoop


def test_tool_execution_failed_is_error():
    result = "Tool execution failed.\nReason: Error: division by zero"
    assert AgentLoop._is_error_result(result) is True


def test_error_prefix_is_error():
    assert AgentLoop._is_error_result("Error: division by zero") is True


def test_permission_denied_is_error():
    assert AgentLoop._is_error_result("Permission denied: employee") is True


def test_unknown_tool_is_error():
    assert AgentLoop._is_error_result("unknown tool") is True


def test_normal_text_is_not_error():
    assert AgentLoop._is_error_result("Here is the answer.") is False


def test_non_string_result_is_not_error():
    assert AgentLoop._is_error_result(42) is False
