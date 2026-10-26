"""Regression tests for the four degradation paths.

These tests never start a real MCP server or call an external API:
- MCP startup failure is simulated in a subprocess that applies the
  test isolation bootstrap before importing main, proving the app
  still initializes without touching external resources.
- Agent failure is simulated by patching AgentLoop.run.
- MCP call timeout is simulated with a slow fake session on a real
  event loop running in a thread, so the client's own timeout path
  is exercised.

Database isolation: /register, /login and /chat keep running their
real endpoint code, but the ``get_db`` dependency is answered by an
in-memory stub so no test row is ever written to the developer's
agent_db via SessionLocal.
"""

import asyncio
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]

CHAT_FALLBACK_REPLY = "（系统）当前无法处理该请求，请稍后重试。"

SECRET_INTERNALS = (
    "agent crashed while querying "
    "mysql+pymysql://root:hunter2@internal-db-01:3306/agent_db"
)


class _StubQuery:
    """Chainable query over an in-memory bucket.

    Filter conditions are SQLAlchemy expressions; each bucket holds
    only this test's own rows, so no narrowing is needed. ``order_by``
    honours real ``asc()``/``desc()`` clauses (detected from the
    compiled expression text, e.g. ``messages.id DESC``), so callers
    that reverse the result — like MemoryManager._recent_messages —
    still observe correct chronological order. ``limit`` truncates
    after sorting, mirroring SQL semantics.
    """

    def __init__(self, rows):
        self.rows = list(rows)

    def filter(self, *conditions, **kwargs):
        return self

    def order_by(self, *clauses):
        for clause in reversed(clauses):
            element = getattr(clause, "element", clause)
            name = getattr(element, "name", "id")
            descending = "desc" in str(clause).lower()
            self.rows.sort(
                key=lambda row: getattr(row, name, 0) or 0,
                reverse=descending,
            )

        return self

    def limit(self, count):
        self.rows = self.rows[:count]
        return self

    def first(self):
        return self.rows[0] if self.rows else None

    def all(self):
        return list(self.rows)


class _StubDB:
    """In-memory stand-in for the SQLAlchemy session.

    /register, /login and /chat run their real endpoint code, but
    every row is kept here instead of the developer's agent_db.
    ``add`` mirrors what an INSERT would persist (autoincrement id
    and the ``role`` column default), so tokens and stored rows look
    exactly like ones a real database would return. Unknown models
    raise so a surprise query fails loudly instead of passing
    silently.
    """

    _BUCKETS = ("User", "Message", "ConversationSummary")

    def __init__(self):
        self._rows = {name: [] for name in self._BUCKETS}
        self._next_id = {}
        self.rollback_count = 0

    def query(self, model):
        name = getattr(model, "__name__", None)

        if name not in self._rows:
            raise ValueError(f"Unexpected model in this test: {model}")

        return _StubQuery(self._rows[name])

    def add(self, obj):
        name = type(obj).__name__

        if name not in self._rows:
            raise ValueError(f"Unexpected model in this test: {type(obj)}")

        self._next_id[name] = self._next_id.get(name, 0) + 1

        if getattr(obj, "id", None) is None:
            obj.id = self._next_id[name]

        if name == "User" and getattr(obj, "role", None) is None:
            obj.role = "employee"

        self._rows[name].append(obj)

    def commit(self):
        pass

    def rollback(self):
        self.rollback_count += 1

    def close(self):
        pass


MCP_UNAVAILABLE_SCRIPT = textwrap.dedent(
    '''
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    repo_root = Path.cwd().resolve()
    sys.path.insert(0, str(repo_root))

    import conftest

    refresh_failures = []

    def fail_mcp_refresh(_registry):
        refresh_failures.append("MCP server failed to start")
        raise RuntimeError("MCP server failed to start")

    state = conftest.bootstrap_isolated_main(
        mcp_refresh_callback=fail_mcp_refresh
    )
    main = state.main_module
    fake_model_requests = []

    def fake_create(**kwargs):
        fake_model_requests.append(kwargs)
        message = SimpleNamespace(
            content="你好，有什么可以帮你？",
            tool_calls=None,
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    try:
        # The model transport remains the offline rejecting client; only this
        # test's expected completion is supplied by a local scripted response.
        main.client.chat.completions.create = fake_create

        assert main.engine.dialect.name == "sqlite"
        assert Path(main.engine.url.database).resolve() == state.database_path.resolve()
        assert state.database_path.is_file()
        assert state.requested_chroma_path == (
            repo_root / "chroma_data"
        ).resolve()
        assert main.chroma_client.path == state.chroma_path.resolve()
        assert main.chroma_client.path != state.requested_chroma_path
        assert state.mcp_refresh_interceptions == 1
        assert refresh_failures == ["MCP server failed to start"]
        assert state.mcp_start_attempts == 0
        assert main.mcp_client._thread is None

        # MCP is degraded to "no tools", but application initialization and
        # non-MCP capabilities remain available.
        assert main.mcp_tool_registry._tools == {}
        assert main.mcp_tool_registry.get("get_company_info") is None
        assert main.agent_loop.mcp_tool_registry is main.mcp_tool_registry
        assert main.agent_loop.tool_registry.get("calculate") is not None
        assert main.skill_registry.get("knowledge_search") is not None
        assert isinstance(main.client, conftest._OfflineOpenAI)
        assert isinstance(main.redis_client, conftest._OfflineRedis)

        reply, task_state = main.agent_loop.run(
            [{"role": "user", "content": "你好"}]
        )

        assert reply == "你好，有什么可以帮你？"
        assert task_state is None
        assert len(fake_model_requests) == 1
        assert main.client.request_count == 0
        assert main.redis_client.operation_count == 0
    finally:
        state.cleanup()

    print("degraded-ok")
    '''
)


ASYNCIO_LOOP_DIAGNOSTIC_SCRIPT = textwrap.dedent(
    """
    import _socket
    import asyncio
    import sys
    import traceback

    policy_name = type(asyncio.get_event_loop_policy()).__name__
    print(f"probe:platform={sys.platform}", flush=True)
    print(f"probe:executable={sys.executable}", flush=True)
    print(f"probe:policy={policy_name}", flush=True)
    print(
        f"probe:native_socketpair={hasattr(_socket, 'socketpair')}",
        flush=True,
    )
    print("probe:before-loop-create", flush=True)
    try:
        loop = asyncio.new_event_loop()
    except BaseException as exc:
        print("probe:error-stage=loop-create", flush=True)
        print(
            f"probe:error-winerror={getattr(exc, 'winerror', None)}",
            flush=True,
        )
        traceback.print_exc()
        raise

    print(f"probe:loop-created={type(loop).__name__}", flush=True)
    loop.call_soon(loop.stop)
    print("probe:before-loop-run", flush=True)
    try:
        loop.run_forever()
    except BaseException as exc:
        print("probe:error-stage=loop-run", flush=True)
        print(
            f"probe:error-winerror={getattr(exc, 'winerror', None)}",
            flush=True,
        )
        traceback.print_exc()
        raise

    print("probe:loop-running", flush=True)
    print("probe:before-loop-close", flush=True)
    loop.close()
    print("probe:loop-closed", flush=True)
    """
)


HTTP_CHAT_FAILURE_SCRIPT = textwrap.dedent(
    """
    import runpy
    from pathlib import Path

    import conftest

    state = conftest.bootstrap_isolated_main()
    try:
        tests = runpy.run_path(
            str(Path.cwd() / "tests" / "test_degradation.py")
        )
        tests["_exercise_chat_agent_failure_http"](state.main_module)
    finally:
        state.cleanup()

    print("http-chat-failure-ok")
    """
)


MCP_TIMEOUT_SCRIPT = textwrap.dedent(
    """
    import runpy
    from pathlib import Path

    tests = runpy.run_path(
        str(Path.cwd() / "tests" / "test_degradation.py")
    )
    tests["_exercise_mcp_tool_call_timeout"]()
    print("mcp-timeout-ok")
    """
)


_LOOP_START_DIAGNOSTIC = None


def _run_bounded_subprocess(command, *, timeout, cwd, env=None):
    """Run a test helper with a wall-clock bound and kill its process tree."""
    kwargs = {
        "args": command,
        "cwd": cwd,
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }

    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    process = subprocess.Popen(**kwargs)

    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return SimpleNamespace(
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=2)

        return SimpleNamespace(
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
        )


def _classify_windows_loop_diagnostic(diagnostic, *, host_platform):
    """Classify only stage-specific Proactor/socketpair failures as skippable."""
    if host_platform != "win32":
        return None

    stdout_lines = set((diagnostic.stdout or "").splitlines())
    stderr = diagnostic.stderr or ""

    required_environment = {
        "probe:platform=win32",
        "probe:policy=WindowsProactorEventLoopPolicy",
        "probe:native_socketpair=False",
    }
    if not required_environment.issubset(stdout_lines):
        return None

    loop_created = "probe:loop-created=ProactorEventLoop" in stdout_lines
    loop_started = "probe:loop-running" in stdout_lines
    probe_reported_error = any(
        line.startswith("probe:error-stage=") for line in stdout_lines
    )

    if diagnostic.timed_out:
        creation_block = (
            "probe:before-loop-create" in stdout_lines
            and not loop_created
            and "probe:before-loop-run" not in stdout_lines
            and not probe_reported_error
        )
        startup_block = (
            loop_created
            and "probe:before-loop-run" in stdout_lines
            and not loop_started
            and not probe_reported_error
        )

        if creation_block:
            return (
                "Windows Proactor loop creation timed out after the explicit "
                "before-loop-create marker; native socketpair is unavailable. "
                "The child process tree was terminated. WinError 10013 was "
                "not confirmed."
            )
        if startup_block:
            return (
                "Windows Proactor loop startup timed out after the explicit "
                "before-loop-run marker; native socketpair is unavailable. "
                "The child process tree was terminated. WinError 10013 was "
                "not confirmed."
            )
        return None

    confirmed_socketpair_10013 = (
        not diagnostic.timed_out
        and diagnostic.returncode != 0
        and "probe:before-loop-create" in stdout_lines
        and "probe:error-stage=loop-create" in stdout_lines
        and "probe:error-winerror=10013" in stdout_lines
        and "probe:loop-created=ProactorEventLoop" not in stdout_lines
        and "proactor_events.py" in stderr
        and "_make_self_pipe" in stderr
        and "socket.py" in stderr
        and "_fallback_socketpair" in stderr
        and "[WinError 10013]" in stderr
    )
    if confirmed_socketpair_10013:
        return (
            "Windows Proactor loop creation raised WinError 10013 in the "
            "self-pipe socketpair fallback path."
        )

    return None


def _windows_loop_start_block_reason():
    """Return a reason only when a bounded diagnostic proves a loop-stage block."""
    global _LOOP_START_DIAGNOSTIC

    if sys.platform != "win32":
        return None

    if _LOOP_START_DIAGNOSTIC is None:
        _LOOP_START_DIAGNOSTIC = _run_bounded_subprocess(
            [sys.executable, "-u", "-c", ASYNCIO_LOOP_DIAGNOSTIC_SCRIPT],
            timeout=15,
            cwd=REPO_ROOT,
        )

    return _classify_windows_loop_diagnostic(
        _LOOP_START_DIAGNOSTIC,
        host_platform=sys.platform,
    )


def _loop_diagnostic_result(
    *,
    stdout="",
    stderr="",
    returncode=0,
    timed_out=False,
):
    return SimpleNamespace(
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        timed_out=timed_out,
    )


def test_loop_diagnostic_classifier_accepts_only_verified_windows_stages():
    environment = (
        "probe:platform=win32\n"
        "probe:policy=WindowsProactorEventLoopPolicy\n"
        "probe:native_socketpair=False\n"
    )
    creation_timeout = _loop_diagnostic_result(
        stdout=environment + "probe:before-loop-create\n",
        timed_out=True,
    )
    creation_reason = _classify_windows_loop_diagnostic(
        creation_timeout,
        host_platform="win32",
    )
    assert creation_reason is not None
    assert "loop creation timed out" in creation_reason
    assert "WinError 10013 was not confirmed" in creation_reason

    startup_timeout = _loop_diagnostic_result(
        stdout=(
            environment
            + "probe:before-loop-create\n"
            + "probe:loop-created=ProactorEventLoop\n"
            + "probe:before-loop-run\n"
        ),
        timed_out=True,
    )
    startup_reason = _classify_windows_loop_diagnostic(
        startup_timeout,
        host_platform="win32",
    )
    assert startup_reason is not None
    assert "loop startup timed out" in startup_reason


@pytest.mark.parametrize(
    ("diagnostic", "host_platform"),
    [
        (
            _loop_diagnostic_result(
                stdout=(
                    "probe:platform=win32\n"
                    "probe:policy=WindowsProactorEventLoopPolicy\n"
                    "probe:native_socketpair=False\n"
                ),
                timed_out=True,
            ),
            "win32",
        ),
        (
            _loop_diagnostic_result(
                stdout=(
                    "probe:platform=win32\n"
                    "probe:policy=WindowsProactorEventLoopPolicy\n"
                    "probe:native_socketpair=False\n"
                    "probe:before-loop-create\n"
                    "probe:error-stage=loop-create\n"
                    "probe:error-winerror=None\n"
                ),
                stderr="PermissionError while building loop; then timed out",
                timed_out=True,
            ),
            "win32",
        ),
        (
            _loop_diagnostic_result(
                stdout=(
                    "probe:platform=win32\n"
                    "probe:policy=WindowsSelectorEventLoopPolicy\n"
                    "probe:native_socketpair=False\n"
                    "probe:before-loop-create\n"
                ),
                timed_out=True,
            ),
            "win32",
        ),
        (
            _loop_diagnostic_result(
                stdout=(
                    "probe:platform=win32\n"
                    "probe:policy=WindowsProactorEventLoopPolicy\n"
                    "probe:native_socketpair=True\n"
                    "probe:before-loop-create\n"
                ),
                timed_out=True,
            ),
            "win32",
        ),
        (
            _loop_diagnostic_result(
                stdout=(
                    "probe:platform=win32\n"
                    "probe:policy=WindowsProactorEventLoopPolicy\n"
                    "probe:native_socketpair=False\n"
                    "probe:before-loop-create\n"
                    "probe:loop-created=ProactorEventLoop\n"
                    "probe:before-loop-run\n"
                    "probe:loop-running\n"
                    "probe:before-loop-close\n"
                ),
                timed_out=True,
            ),
            "win32",
        ),
        (
            _loop_diagnostic_result(
                stdout="diagnostic output unavailable",
                timed_out=True,
            ),
            "win32",
        ),
        (
            _loop_diagnostic_result(
                stdout=(
                    "probe:platform=win32\n"
                    "probe:policy=WindowsProactorEventLoopPolicy\n"
                    "probe:native_socketpair=False\n"
                    "probe:before-loop-create\n"
                    "probe:error-stage=other\n"
                ),
                stderr="unrelated error mentions WinError 10013",
                returncode=1,
            ),
            "win32",
        ),
        (
            _loop_diagnostic_result(
                stdout=(
                    "probe:platform=win32\n"
                    "probe:policy=WindowsProactorEventLoopPolicy\n"
                    "probe:native_socketpair=False\n"
                    "probe:before-loop-create\n"
                    "probe:loop-created=ProactorEventLoop\n"
                    "probe:before-loop-run\n"
                    "probe:loop-running\n"
                    "probe:before-loop-close\n"
                    "probe:loop-closed\n"
                ),
                returncode=0,
            ),
            "win32",
        ),
        (
            _loop_diagnostic_result(
                stdout=(
                    "probe:platform=win32\n"
                    "probe:policy=WindowsProactorEventLoopPolicy\n"
                    "probe:native_socketpair=False\n"
                    "probe:before-loop-create\n"
                ),
                timed_out=True,
            ),
            "linux",
        ),
    ],
    ids=(
        "missing-stage-marker",
        "timeout-after-error-marker",
        "non-proactor-policy",
        "native-socketpair-present",
        "loop-already-running",
        "insufficient-output",
        "unrelated-winerror-text",
        "normal-exit",
        "non-windows-host",
    ),
)
def test_loop_diagnostic_classifier_rejects_unverified_skip_conditions(
    diagnostic,
    host_platform,
):
    assert (
        _classify_windows_loop_diagnostic(
            diagnostic,
            host_platform=host_platform,
        )
        is None
    )


def test_loop_diagnostic_classifier_requires_socketpair_trace_for_winerror():
    stdout = (
        "probe:platform=win32\n"
        "probe:policy=WindowsProactorEventLoopPolicy\n"
        "probe:native_socketpair=False\n"
        "probe:before-loop-create\n"
        "probe:error-stage=loop-create\n"
        "probe:error-winerror=10013\n"
    )
    unrelated_trace = _loop_diagnostic_result(
        stdout=stdout,
        stderr="RuntimeError: WinError 10013 from an unrelated operation",
        returncode=1,
    )
    assert _classify_windows_loop_diagnostic(
        unrelated_trace,
        host_platform="win32",
    ) is None

    expected_trace = _loop_diagnostic_result(
        stdout=stdout,
        stderr=(
            'File "asyncio\\proactor_events.py", in _make_self_pipe\n'
            'File "socket.py", in _fallback_socketpair\n'
            "OSError: [WinError 10013] access denied"
        ),
        returncode=1,
    )
    reason = _classify_windows_loop_diagnostic(
        expected_trace,
        host_platform="win32",
    )
    assert reason is not None
    assert "self-pipe socketpair fallback path" in reason


def test_main_initializes_when_mcp_server_unavailable():
    result = _run_bounded_subprocess(
        [sys.executable, "-c", MCP_UNAVAILABLE_SCRIPT],
        timeout=30,
        cwd=REPO_ROOT,
    )

    assert not result.timed_out, result.stdout + result.stderr
    assert result.returncode == 0, (
        f"returncode={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "degraded-ok" in result.stdout


def register_and_login(client):
    username = f"degrade_{uuid.uuid4().hex[:8]}"
    password = "degradepass123"

    registered = client.post(
        "/register",
        json={"username": username, "password": password},
    )
    assert registered.status_code == 200

    logged_in = client.post(
        "/login",
        json={"username": username, "password": password},
    )
    assert logged_in.status_code == 200

    return username, logged_in.json()["access_token"]


class DummyRedis:
    """Healthy in-memory cache so this test stays focused on the
    agent-failure path instead of the local Redis deployment."""

    def get(self, key):
        return None

    def setex(self, key, ttl, value):
        return True

    def delete(self, key):
        return 1


def _exercise_chat_agent_failure(main_module):
    store = _StubDB()
    overrides = main_module.app.dependency_overrides
    missing = object()
    previous_get_db = overrides.get(main_module.get_db, missing)
    overrides[main_module.get_db] = lambda: store
    previous_redis = main_module.memory_manager.redis_client
    previous_agent_run = main_module.agent_loop.run

    try:
        main_module.memory_manager.redis_client = DummyRedis()
        user_id = int(uuid.uuid4().hex[:8], 16)
        db_user = SimpleNamespace(id=user_id, role="employee")

        def exploding_run(messages, role="employee"):
            raise RuntimeError(SECRET_INTERNALS)

        main_module.agent_loop.run = exploding_run

        with pytest.raises(main_module.HTTPException) as exc_info:
            main_module.chat(
                data=main_module.ChatMessage(message="帮我查一下差旅标准"),
                current_user={"user_id": user_id},
                db_user=db_user,
                db=store,
            )

        assert exc_info.value.status_code == 503
        detail = exc_info.value.detail
        assert "agent crashed" not in detail
        assert "mysql+pymysql" not in detail
        assert "hunter2" not in detail
        assert "internal-db-01" not in detail
        assert "Traceback" not in detail

        messages = store.query(main_module.Message).all()

        assert [message.role for message in messages] == ["user", "assistant"]
        assert messages[0].content == "帮我查一下差旅标准"
        assert messages[1].content == CHAT_FALLBACK_REPLY

        for message in messages:
            assert "mysql+pymysql" not in message.content
            assert "hunter2" not in message.content
            assert "agent crashed" not in message.content
            assert message.role not in {"tool", "function"}

        assert main_module.client.request_count == 0
        assert main_module.redis_client.operation_count == 0
        assert main_module.mcp_client._thread is None
    finally:
        main_module.memory_manager.redis_client = previous_redis
        main_module.agent_loop.run = previous_agent_run

        if previous_get_db is missing:
            overrides.pop(main_module.get_db, None)
        else:
            overrides[main_module.get_db] = previous_get_db


def test_chat_agent_failure_direct_route_returns_503_and_keeps_history_paired():
    """Test sync route logic directly; this does not cover ASGI/middleware."""
    import main as main_module

    _exercise_chat_agent_failure(main_module)


def test_chat_agent_failure_rolls_back_compression_database_failure(
    monkeypatch,
):
    import main as main_module

    store = _StubDB()
    user_id = int(uuid.uuid4().hex[:8], 16)
    db_user = SimpleNamespace(id=user_id, role="employee")
    monkeypatch.setattr(main_module.memory_manager, "redis_client", DummyRedis())
    monkeypatch.setattr(
        main_module.agent_loop,
        "run",
        lambda messages, role="employee": ("safe reply", None),
    )

    def fail_compression(_user_id, _db):
        raise RuntimeError("simulated summary database failure")

    monkeypatch.setattr(
        main_module.memory_manager,
        "maybe_compress_history",
        fail_compression,
    )

    response = main_module.chat(
        data=main_module.ChatMessage(message="hello"),
        current_user={"user_id": user_id},
        db_user=db_user,
        db=store,
    )

    assert response["reply"] == "safe reply"
    assert store.rollback_count == 1
    messages = store.query(main_module.Message).all()
    assert [message.role for message in messages] == ["user", "assistant"]
    assert [message.content for message in messages] == ["hello", "safe reply"]


def _exercise_chat_agent_failure_http(main_module):
    store = _StubDB()
    overrides = main_module.app.dependency_overrides
    missing = object()
    previous_get_db = overrides.get(main_module.get_db, missing)
    previous_redis = main_module.memory_manager.redis_client
    previous_agent_run = main_module.agent_loop.run
    overrides[main_module.get_db] = lambda: store

    try:
        client = TestClient(
            main_module.app,
            client=("198.51.100.30", 40000),
        )
        main_module.memory_manager.redis_client = DummyRedis()
        _, token = register_and_login(client)
        user_id = main_module.decode_token(token)["user_id"]

        def exploding_run(messages, role="employee"):
            raise RuntimeError(SECRET_INTERNALS)

        main_module.agent_loop.run = exploding_run
        response = client.post(
            "/chat",
            json={"message": "帮我查一下差旅标准"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 503
        body = response.text
        assert "agent crashed" not in body
        assert "mysql+pymysql" not in body
        assert "hunter2" not in body
        assert "internal-db-01" not in body
        assert "Traceback" not in body

        messages = (
            store.query(main_module.Message)
            .filter(main_module.Message.user_id == str(user_id))
            .order_by(main_module.Message.id.asc())
            .all()
        )
        assert [message.role for message in messages] == ["user", "assistant"]
        assert messages[0].content == "帮我查一下差旅标准"
        assert messages[1].content == CHAT_FALLBACK_REPLY

        for message in messages:
            assert "mysql+pymysql" not in message.content
            assert "hunter2" not in message.content
            assert "agent crashed" not in message.content

        assert main_module.client.request_count == 0
        assert main_module.mcp_client._thread is None
    finally:
        main_module.memory_manager.redis_client = previous_redis
        main_module.agent_loop.run = previous_agent_run
        if previous_get_db is missing:
            overrides.pop(main_module.get_db, None)
        else:
            overrides[main_module.get_db] = previous_get_db


def test_chat_agent_failure_returns_503_and_keeps_history_paired():
    loop_block_reason = _windows_loop_start_block_reason()
    if loop_block_reason:
        pytest.skip(
            "HTTP/ASGI TestClient check not run: " + loop_block_reason
        )

    result = _run_bounded_subprocess(
        [sys.executable, "-u", "-c", HTTP_CHAT_FAILURE_SCRIPT],
        timeout=30,
        cwd=REPO_ROOT,
    )
    assert not result.timed_out, (
        "HTTP/ASGI child exceeded 30 seconds.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "http-chat-failure-ok" in result.stdout


def _exercise_mcp_tool_call_timeout():
    from app.mcp.adapter import MCPToolAdapter
    from app.mcp.client import MCPClient
    from app.mcp.tool_registry import MCPToolRegistry

    client = MCPClient(call_timeout=0.2, startup_timeout=1.0)

    # Never spawn a real server for this test.
    previous_start = client.start
    client.start = lambda: None

    class SlowSession:
        async def call_tool(self, name, arguments=None):
            await asyncio.sleep(30)

            return SimpleNamespace(is_error=False, content=[])

    client._loop = asyncio.new_event_loop()
    client._session = SlowSession()
    loop_ready = threading.Event()
    loop_stopped = threading.Event()
    loop_failures = []

    client._loop.call_soon(loop_ready.set)

    def run_loop():
        try:
            client._loop.run_forever()
        except BaseException as exc:
            loop_failures.append(exc)
        finally:
            loop_stopped.set()

    thread = threading.Thread(
        target=run_loop,
        daemon=True,
    )
    thread.start()

    if not loop_ready.wait(timeout=2):
        if loop_failures:
            raise RuntimeError(
                f"event loop failed during startup: {loop_failures[0]!r}"
            )
        if loop_stopped.is_set():
            raise RuntimeError("event loop thread stopped before becoming ready")
        raise RuntimeError("event loop did not become ready within 2 seconds")

    registry = MCPToolRegistry(client)
    registry._tools["get_company_info"] = MCPToolAdapter(
        client=client,
        name="get_company_info",
        description="Get basic information about a company.",
        input_schema={},
    )

    try:
        started_at = time.monotonic()
        result = registry.execute(
            "get_company_info",
            {"company_name": "OpenAI"},
        )
        elapsed = time.monotonic() - started_at
    finally:
        def request_shutdown():
            async def cancel_pending_tasks():
                current = asyncio.current_task()
                pending = [
                    task
                    for task in asyncio.all_tasks(client._loop)
                    if task is not current
                ]
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                client._loop.stop()

            asyncio.create_task(cancel_pending_tasks())

        if thread.is_alive():
            try:
                client._loop.call_soon_threadsafe(request_shutdown)
            except RuntimeError:
                pass
        thread.join(timeout=2)

        if thread.is_alive():
            raise RuntimeError(
                "event loop thread did not exit within 2 seconds; "
                "left the loop open and delegated cleanup to the bounded child"
            )

        client._loop.close()
        client.start = previous_start

    if loop_failures:
        raise RuntimeError(f"event loop thread failed: {loop_failures[0]!r}")

    # The call is bounded by the configured timeout, not 30 seconds.
    assert elapsed < 5

    # The caller receives the timeout category, never transport detail.
    assert result.startswith("Error:")
    assert "timeout_error" in result
    assert "asyncio" not in result
    assert "Traceback" not in result
    assert "sleep" not in result

    # It is not mistaken for a successful tool result.
    assert result != "AI research and deployment company."
    assert result != "No information found for OpenAI."


def test_mcp_tool_call_timeout_is_bounded_and_sanitized():
    loop_block_reason = _windows_loop_start_block_reason()
    if loop_block_reason:
        pytest.skip(
            "MCP asyncio timeout check not run: " + loop_block_reason
        )

    result = _run_bounded_subprocess(
        [sys.executable, "-u", "-c", MCP_TIMEOUT_SCRIPT],
        timeout=15,
        cwd=REPO_ROOT,
    )
    assert not result.timed_out, (
        "MCP timeout child exceeded 15 seconds.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "mcp-timeout-ok" in result.stdout
