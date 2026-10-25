"""Subprocess smoke test proving isolation precedes importing ``main``."""

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_early_bootstrap_isolates_resources_before_main_import():
    script = r'''
import sys
from pathlib import Path

project_root = Path.cwd().resolve()
sys.path.insert(0, str(project_root))

import conftest

state = conftest.bootstrap_isolated_main()
main = state.main_module

assert main.engine.dialect.name == "sqlite"
assert Path(main.engine.url.database).resolve() == state.database_path.resolve()
assert state.database_path.is_file()
assert state.requested_database_name == "agent_db"
assert {"users", "messages", "conversation_summaries"}.issubset(
    set(__import__("sqlalchemy").inspect(main.engine).get_table_names())
)

assert state.requested_chroma_path == (project_root / "chroma_data").resolve()
assert state.chroma_fake_calls == 1
assert main.chroma_client.path == state.chroma_path.resolve()
assert main.chroma_client.path != state.requested_chroma_path
assert not state.chroma_path.exists()

assert main.MCP_AUTO_REFRESH is False
assert state.mcp_refresh_interceptions == 0
assert state.mcp_start_attempts == 0
assert main.mcp_client._thread is None
assert main.mcp_tool_registry._tools == {}

assert isinstance(main.client, conftest._OfflineOpenAI)
assert isinstance(main.redis_client, conftest._OfflineRedis)
assert main.client.request_count == 0
assert main.redis_client.operation_count == 0
assert main.app is not None
assert main.planner is not None
assert main.agent_loop is not None
assert main.memory_manager is not None

try:
    main.client.chat.completions.create(model="must-not-send")
except AssertionError:
    pass
else:
    raise AssertionError("offline OpenAI guard did not reject request")
assert main.client.request_count == 1

try:
    main.redis_client.get("must-not-connect")
except AssertionError:
    pass
else:
    raise AssertionError("offline Redis guard did not reject operation")
assert main.redis_client.operation_count == 1

state.cleanup()
print("ISOLATION_SMOKE_OK")
'''

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ISOLATION_SMOKE_OK" in completed.stdout


def test_explicit_resource_configuration_and_mcp_switch_in_subprocesses():
    script = r'''
import os
import sys
from pathlib import Path

project_root = Path.cwd().resolve()
sys.path.insert(0, str(project_root))

import conftest

overrides = {
    "MYSQL_DATABASE": os.environ["ISOLATION_TEST_MYSQL_DATABASE"],
    "CHROMA_PERSIST_DIRECTORY": os.environ[
        "ISOLATION_TEST_CHROMA_DIRECTORY"
    ],
    "MCP_AUTO_REFRESH": os.environ["ISOLATION_TEST_MCP_AUTO_REFRESH"],
}
state = conftest.bootstrap_isolated_main(environment_overrides=overrides)
main = state.main_module

assert main.engine.dialect.name == "sqlite"
assert Path(main.engine.url.database).resolve() == state.database_path.resolve()
assert state.database_path.is_file()
assert state.requested_database_name == "agent_backend_test"

assert state.requested_chroma_path == Path(
    overrides["CHROMA_PERSIST_DIRECTORY"]
).resolve()
assert state.chroma_fake_calls == 1
assert main.chroma_client.path == state.chroma_path.resolve()
assert main.chroma_client.path != state.requested_chroma_path

expected_refresh_count = (
    1 if overrides["MCP_AUTO_REFRESH"] == "true" else 0
)
assert main.MCP_AUTO_REFRESH is (expected_refresh_count == 1)
assert state.mcp_refresh_interceptions == expected_refresh_count
assert state.mcp_start_attempts == 0
assert main.mcp_client._thread is None

assert isinstance(main.client, conftest._OfflineOpenAI)
assert main.client.request_count == 0
assert isinstance(main.redis_client, conftest._OfflineRedis)
assert main.redis_client.operation_count == 0

state.cleanup()
print("CONFIG_ISOLATION_OK")
'''

    for mcp_auto_refresh in ("false", "true"):
        child_env = os.environ.copy()
        child_env.update(
            {
                "ISOLATION_TEST_MYSQL_DATABASE": "agent_backend_test",
                "ISOLATION_TEST_CHROMA_DIRECTORY": str(
                    PROJECT_ROOT.parent
                    / "agent-backend-chroma-config-test"
                ),
                "ISOLATION_TEST_MCP_AUTO_REFRESH": mcp_auto_refresh,
            }
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        assert "CONFIG_ISOLATION_OK" in completed.stdout


def _run_isolation_child(script, env=None):
    try:
        completed = subprocess.run(
            [sys.executable, "-u", "-c", script],
            cwd=PROJECT_ROOT,
            env=os.environ.copy() if env is None else env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            f"isolation child timed out; stdout={exc.stdout!r}; "
            f"stderr={exc.stderr!r}"
        ) from exc
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_mcp_start_registry_and_stdio_are_blocked_after_main_import():
    script = r'''
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path.cwd().resolve()))

import conftest
state = conftest.bootstrap_isolated_main()
print("STEP_BOOTSTRAPPED", flush=True)
from app.mcp.adapter import MCPToolAdapter
from app.mcp.client import MCPClient
from app.mcp.tool_registry import MCPToolRegistry

client = MCPClient()
try:
    client.start()
except conftest.MCPIsolationError:
    pass
else:
    raise AssertionError("direct MCPClient.start was not rejected")
assert state.mcp_start_attempts == 1
assert client._thread is None
assert state.mcp_stdio_attempts == 0

registry = MCPToolRegistry(client)
try:
    registry.refresh()
except conftest.MCPIsolationError:
    pass
else:
    raise AssertionError("direct real-client registry refresh was not rejected")
assert state.mcp_refresh_interceptions == 1
assert client._thread is None
assert registry._tools == {}
print("STEP_DIRECT_GUARDS", flush=True)

# Deliberately restore only the original Registry method to exercise the
# actual refresh -> list_tools -> _submit -> start path. The session start
# guard must still prevent a thread or stdio launch.
refresh_record = next(
    item for item in state.patch_records
    if item["name"] == "refresh" and item["scope"] == "session"
)
active_refresh = MCPToolRegistry.refresh
MCPToolRegistry.refresh = refresh_record["original"]
try:
    try:
        registry.refresh()
    except conftest.MCPIsolationError:
        pass
    else:
        raise AssertionError("indirect MCP startup was not rejected")
finally:
    MCPToolRegistry.refresh = active_refresh
assert state.mcp_start_attempts == 2
assert client._thread is None
assert state.mcp_stdio_attempts == 0
print("STEP_INDIRECT_GUARD", flush=True)

registry._tools["blocked"] = MCPToolAdapter(
    client=client,
    name="blocked",
    description="test",
    input_schema={},
)
result = registry.execute("blocked", {})
assert result.startswith("Error:")
assert state.mcp_start_attempts == 3
assert client._thread is None
assert state.mcp_stdio_attempts == 0
print("STEP_EXECUTE_GUARD", flush=True)

# Exercise the exact stdio symbol imported by app.mcp.client. It must reject
# before reaching the MCP SDK implementation or starting a process.
stdio_guard_record = next(
    item for item in state.patch_records
    if item["mapping"]
    and item["name"] == "stdio_client"
    and item["scope"] == "session"
)
assert (
    MCPClient._connection_lifecycle.__globals__["stdio_client"]
    is stdio_guard_record["replacement"]
)
coroutine = client._connection_lifecycle()
blocked_before_await = False
try:
    try:
        coroutine.send(None)
    except conftest.MCPIsolationError:
        blocked_before_await = True
    except StopIteration as exc:
        raise AssertionError(
            "MCP lifecycle completed without reaching the stdio guard"
        ) from exc
    except Exception as exc:
        raise AssertionError(
            f"unexpected error before the MCP stdio guard: {type(exc).__name__}"
        ) from exc
    else:
        raise AssertionError(
            "MCP lifecycle yielded before the stdio guard rejected startup"
        )
finally:
    coroutine.close()
assert blocked_before_await
assert state.mcp_stdio_attempts == 1
assert client._thread is None
print("STEP_STDIO_GUARD", flush=True)

class FakeClient:
    def __init__(self):
        self.list_calls = 0
        self.tool_calls = []
    def list_tools(self):
        self.list_calls += 1
        return [SimpleNamespace(
            name="fake_tool",
            description="offline",
            input_schema={"type": "object", "properties": {}},
        )]
    def call_tool(self, name, arguments):
        self.tool_calls.append((name, arguments))
        return "fake result"

fake = FakeClient()
fake_registry = MCPToolRegistry(fake)
fake_registry.refresh()
assert fake.list_calls == 1
assert fake_registry.get("fake_tool") is not None
assert fake_registry.execute("fake_tool", {"ok": True}) == "fake result"
assert fake.tool_calls == [("fake_tool", {"ok": True})]
assert state.mcp_start_attempts == 3
assert state.mcp_stdio_attempts == 1

print("STEP_FAKE_CLIENT", flush=True)
state.cleanup()
state.cleanup()
assert MCPClient.start is next(
    item["original"] for item in state.patch_records
    if item["name"] == "start" and item["scope"] == "session"
)
assert MCPToolRegistry.refresh is refresh_record["original"]
assert state._cleaned
print("MCP_SESSION_GUARDS_OK")
'''
    output = _run_isolation_child(script)
    assert "MCP_SESSION_GUARDS_OK" in output


def test_isolation_cleanup_on_bootstrap_failure_restores_resources():
    script = r'''
import importlib
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().resolve()))
import conftest
from app.mcp.client import MCPClient
from app.mcp.tool_registry import MCPToolRegistry
import app.mcp.client as mcp_client_module
import mcp.client.stdio as mcp_stdio_module

class TrackingState(conftest._IsolationState):
    instances = []
    def __init__(self):
        super().__init__()
        self.__class__.instances.append(self)

conftest._IsolationState = TrackingState
original_import = importlib.import_module
original_start = MCPClient.start
original_refresh = MCPToolRegistry.refresh
original_client_stdio = mcp_client_module.stdio_client
original_sdk_stdio = mcp_stdio_module.stdio_client
controlled = {
    "DEEPSEEK_API_KEY", "JWT_SECRET_KEY", "MYSQL_PASSWORD",
    "AUTO_CREATE_TABLES", "LOG_FILE", "MYSQL_HOST", "MYSQL_PORT",
    "MYSQL_USER", "MYSQL_DATABASE", "CHROMA_PERSIST_DIRECTORY",
    "MCP_AUTO_REFRESH",
}
environment_before = {key: os.environ.get(key) for key in controlled}
root_handlers_before = set(logging.getLogger().handlers)
captured = {}

def fail_after_main_import(name, *args, **kwargs):
    module = original_import(name, *args, **kwargs)
    if name == "main":
        state = TrackingState.instances[-1]
        captured["state"] = state
        captured["engine"] = state.engine
        captured["pool"] = state.engine.pool
        captured["temp_path"] = state.root
        raise RuntimeError("simulated bootstrap failure after main import")
    return module

importlib.import_module = fail_after_main_import
try:
    try:
        conftest.bootstrap_isolated_main()
    except RuntimeError as exc:
        assert "simulated bootstrap failure" in str(exc)
    else:
        raise AssertionError("simulated bootstrap failure did not occur")
finally:
    importlib.import_module = original_import

state = captured["state"]
assert state._cleaned
assert not captured["temp_path"].exists()
assert captured["engine"].pool is not captured["pool"]
assert MCPClient.start is original_start
assert MCPToolRegistry.refresh is original_refresh
assert mcp_client_module.stdio_client is original_client_stdio
assert mcp_stdio_module.stdio_client is original_sdk_stdio
assert {key: os.environ.get(key) for key in controlled} == environment_before
assert set(logging.getLogger().handlers) == root_handlers_before
state.cleanup()
print("BOOTSTRAP_FAILURE_CLEANUP_OK")
'''
    output = _run_isolation_child(script)
    assert "BOOTSTRAP_FAILURE_CLEANUP_OK" in output
