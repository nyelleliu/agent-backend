import logging

import pytest

from app.agent.loop import AgentLoop
from app.core.errors import (
    MAX_MESSAGE_CHARS,
    ToolTimeoutError,
    format_tool_error,
    log_tool_error,
    redact_secrets,
)
from app.core.logging_config import request_id_var
from app.tools.calculator import CalculatorTool
from app.tools.registry import Tool, ToolRegistry

DSN_MESSAGE = (
    "could not connect: mysql+pymysql://root:hunter2@"
    "internal-db-01.corp.example.com:3306/agent_db"
)

PATH_MESSAGE = (
    "failed to read E:\\agent-backend\\chroma_data\\index.bin "
    "and /app/chroma_data/index.bin"
)

SECRET_MESSAGE = (
    "request rejected: Authorization=Bearer eyJhbGciOiJIUzI1NiJ9."
    "eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123def456 "
    "password=hunter2 token=sk-live-9f8e7d6c5b"
)


class BrokenTool(Tool):
    name = "broken_tool"
    description = "Always raises"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, exc: Exception):
        self._exc = exc

    def execute(self, arguments):
        raise self._exc


class RecordingLogger:
    def __init__(self):
        self.records = []

    def _make(self, level):
        def log(message, *args):
            self.records.append((level, message % args if args else message))

        return log

    def __getattr__(self, name):
        if name in ("warning", "error", "info", "exception", "debug"):
            return self._make(name)

        raise AttributeError(name)


@pytest.fixture(autouse=True)
def _clean_request_id():
    token = request_id_var.set("req-test-123")
    yield
    request_id_var.reset(token)


def test_safe_input_error_keeps_useful_message():
    message = format_tool_error(
        ValueError("invalid literal for int() with base 10: 'abc'"),
        tool_name="calculate",
    )

    assert message == (
        "Error: invalid literal for int() with base 10: 'abc'"
    )


def test_type_error_and_key_error_are_kept():
    assert "unsupported operand" in format_tool_error(
        TypeError("unsupported operand type(s) for +: 'int' and 'str'")
    )
    assert "'expression'" in format_tool_error(
        KeyError("expression"),
        tool_name="calculate",
    )


@pytest.mark.parametrize(
    "exc",
    [
        ValueError(DSN_MESSAGE),
        RuntimeError(DSN_MESSAGE),
        ConnectionError(DSN_MESSAGE),
    ],
)
def test_database_dsn_never_reaches_the_model(exc):
    message = format_tool_error(exc, tool_name="query_db")

    assert "mysql+pymysql" not in message
    assert "hunter2" not in message
    assert "internal-db-01" not in message
    assert "agent_db" not in message
    assert "root:" not in message


def test_credentials_and_tokens_never_reach_the_model():
    message = format_tool_error(
        RuntimeError(SECRET_MESSAGE),
        tool_name="call_api",
    )

    assert "hunter2" not in message
    assert "sk-live-9f8e7d6c5b" not in message
    assert "eyJhbGciOiJIUzI1NiJ9" not in message
    assert "Bearer" not in message
    assert "Authorization" not in message


def test_absolute_paths_never_reach_the_model():
    message = format_tool_error(
        RuntimeError(PATH_MESSAGE),
        tool_name="read_file",
    )

    assert "E:\\agent-backend" not in message
    assert "/app/chroma_data" not in message
    assert "index.bin" not in message


def test_whitelisted_message_is_still_redacted_and_truncated():
    long_prefix = "x" * (MAX_MESSAGE_CHARS + 500)
    message = format_tool_error(
        ValueError(f"{long_prefix} password=hunter2"),
        tool_name="calculate",
    )

    assert "hunter2" not in message
    assert len(message) <= MAX_MESSAGE_CHARS + 200
    assert "[truncated]" in message


def test_network_error_returns_category_without_original_message():
    message = format_tool_error(
        ConnectionError("[Errno 111] Connection refused"),
        tool_name="fetch",
    )

    assert message.startswith("Error:")
    assert "Connection refused" not in message
    assert "Errno 111" not in message
    assert "io_error" in message


def test_database_error_returns_category_without_original_message():
    message = format_tool_error(
        RuntimeError("deadlock detected on table messages"),
        tool_name="save",
    )

    assert "deadlock" not in message
    assert "internal_error" in message
    assert "messages" not in message


def test_unknown_error_returns_category_without_original_message():
    class WeirdError(Exception):
        pass

    message = format_tool_error(
        WeirdError("segfault in libfoo.so at 0x7ffd"),
        tool_name="mystery",
    )

    assert "segfault" not in message
    assert "libfoo" not in message
    assert "internal_error" in message


def test_timeout_error_uses_timeout_category():
    message = format_tool_error(
        ToolTimeoutError("MCP call timed out after 30s"),
        tool_name="get_company_info",
    )

    assert "timeout_error" in message
    assert "timed out after" not in message


def test_sqlalchemy_error_is_categorized_as_database():
    sqlalchemy = pytest.importorskip("sqlalchemy")

    exc = sqlalchemy.exc.OperationalError(
        "SELECT 1",
        {},
        Exception("Can't connect to MySQL server on 'db-01' (111)"),
    )

    message = format_tool_error(exc, tool_name="save")

    assert "database_error" in message
    assert "db-01" not in message
    assert "Can't connect" not in message


def test_message_includes_request_id_when_available():
    message = format_tool_error(
        RuntimeError("boom"),
        tool_name="mystery",
    )

    assert "request_id=req-test-123" in message


def test_message_is_safe_when_request_id_missing():
    token = request_id_var.set("-")

    try:
        message = format_tool_error(
            RuntimeError("boom"),
            tool_name="mystery",
        )
    finally:
        request_id_var.reset(token)

    assert "request_id" not in message
    assert message.startswith("Error:")
    assert "internal_error" in message


def test_non_exception_input_does_not_crash():
    message = format_tool_error("just a string", tool_name="weird")

    assert message.startswith("Error:")
    assert "just a string" not in message


def test_log_keeps_detail_but_redacts_credentials():
    logger = RecordingLogger()

    log_tool_error(
        logger,
        RuntimeError(f"connect failed {DSN_MESSAGE}"),
        tool_name="save",
    )

    level, logged = logger.records[0]

    assert level == "error"
    assert "database" not in logged or "connect failed" in logged
    assert "connect failed" in logged
    assert "hunter2" not in logged
    assert "mysql+pymysql://root:hunter2" not in logged
    assert "[REDACTED_URL]" in logged


def test_log_separates_input_errors_from_internal_errors():
    logger = RecordingLogger()

    log_tool_error(
        logger,
        ValueError("invalid literal for int() with base 10: 'abc'"),
        tool_name="calculate",
    )
    log_tool_error(
        logger,
        RuntimeError("segfault at 0x7ffd"),
        tool_name="mystery",
    )

    input_level, input_logged = logger.records[0]
    failure_level, failure_logged = logger.records[1]

    assert input_level == "warning"
    assert "invalid literal" in input_logged

    assert failure_level == "error"
    assert "segfault at 0x7ffd" in failure_logged


def test_never_logs_api_key_material():
    logger = RecordingLogger()

    log_tool_error(
        logger,
        RuntimeError("DeepSeek call failed api_key=sk-verysecret123"),
        tool_name="llm",
    )

    assert "sk-verysecret123" not in logger.records[0][1]
    assert "api_key=[REDACTED]" in logger.records[0][1]


def test_tool_registry_sanitizes_unknown_tool_exceptions(caplog):
    registry = ToolRegistry()
    registry.register(BrokenTool(RuntimeError(DSN_MESSAGE)))

    with caplog.at_level(logging.ERROR):
        result = registry.execute("broken_tool", {})

    assert result.startswith("Error:")
    assert "hunter2" not in result
    assert "mysql+pymysql" not in result
    assert "internal_error" in result
    assert "broken_tool" in caplog.text


def test_tool_registry_keeps_documented_input_errors():
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    assert registry.execute("calculate", "{not-json") == (
        "Error: invalid JSON arguments"
    )
    assert registry.execute(
        "calculate",
        {"expression": "1 / 0"},
    ) == "Error: division by zero"
    assert registry.execute("missing") == "unknown tool"


def test_agent_loop_returns_sanitized_tool_error_to_llm():
    from types import SimpleNamespace

    class FakeToolCall:
        id = "call_1"

        function = SimpleNamespace(
            name="broken_tool",
            arguments="{}",
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

    class FakeClient:
        def __init__(self):
            self.calls = 0

            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        def create(self, **kwargs):
            self.calls += 1

            if self.calls == 1:
                message = SimpleNamespace(
                    content=None,
                    tool_calls=[FakeToolCall()],
                )
            else:
                message = SimpleNamespace(content="done", tool_calls=None)

            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)]
            )

    registry = ToolRegistry()
    registry.register(BrokenTool(RuntimeError(DSN_MESSAGE)))

    class AllowAllPermissions:
        """Permission policy is not the subject of this test.

        The broken tool has to actually execute so that its exception
        travels through the loop's sanitization path; authorization is
        covered by tests/test_rag_injection.py and
        tests/test_agent_permissions.py.
        """

        def has_permission(self, role, name):
            return True

        def has_mcp_permission(self, role, name):
            return True

    agent = AgentLoop(
        client=FakeClient(),
        tool_registry=registry,
        permission_checker=AllowAllPermissions(),
    )

    reply, _ = agent.run([{"role": "user", "content": "run tool"}])

    assert reply == "done"
    assert AgentLoop._is_error_result(
        registry.execute("broken_tool", {})
    ) is True


# ---------------------------------------------------------------------------
# Bare API keys (no "key = value" context) and forward-slash drive paths
# ---------------------------------------------------------------------------

FAKE_BARE_KEY = "sk-test-1234567890abcdef"

FORWARD_SLASH_PATH_MESSAGE = (
    "failed to read /E:/agent-backend/chroma_data/index.bin"
)


def test_bare_api_key_without_context_is_redacted():
    redacted = redact_secrets(
        f"upstream rejected {FAKE_BARE_KEY} during call"
    )

    assert FAKE_BARE_KEY not in redacted
    assert "[REDACTED]" in redacted


def test_bare_api_key_is_redacted_in_logs():
    logger = RecordingLogger()

    log_tool_error(
        logger,
        RuntimeError(f"DeepSeek call failed: {FAKE_BARE_KEY}"),
        tool_name="llm",
    )

    joined = " ".join(record[1] for record in logger.records)

    assert FAKE_BARE_KEY not in joined
    assert "[REDACTED]" in joined


def test_bare_api_key_never_reaches_the_model():
    # ValueError is whitelisted, so its message is passed through to
    # the model after redaction — the key must not survive it.
    message = format_tool_error(
        ValueError(f"bad key {FAKE_BARE_KEY} supplied"),
        tool_name="call_api",
    )

    assert FAKE_BARE_KEY not in message
    assert "[REDACTED]" in message


def test_bare_key_pattern_does_not_mangle_normal_text():
    text = "plain sentence with no credentials 你好 world 12345"

    assert redact_secrets(text) == text


def test_forward_slash_drive_path_is_redacted():
    redacted = redact_secrets(FORWARD_SLASH_PATH_MESSAGE)

    assert "/E:/agent-backend" not in redacted
    assert "chroma_data" not in redacted
    assert "[REDACTED_PATH]" in redacted


def test_forward_slash_drive_path_is_redacted_in_logs():
    logger = RecordingLogger()

    log_tool_error(
        logger,
        OSError(FORWARD_SLASH_PATH_MESSAGE),
        tool_name="read_file",
    )

    joined = " ".join(record[1] for record in logger.records)

    assert "/E:/agent-backend" not in joined
    assert "[REDACTED_PATH]" in joined


def test_backslash_drive_path_still_redacted():
    redacted = redact_secrets("failed to read E:\\agent-backend\\x.bin")

    assert "E:\\agent-backend" not in redacted
    assert "[REDACTED_PATH]" in redacted
