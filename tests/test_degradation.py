"""Regression tests for the four degradation paths.

These tests never start a real MCP server or call an external API:
- MCP startup failure is simulated in a subprocess that imports main
  with a failing MCPToolRegistry, proving the app still initializes
  (and with AUTO_CREATE_TABLES=false the subprocess never runs DDL
  against the developer's MySQL either).
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
import subprocess
import sys
import textwrap
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

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
        pass

    def close(self):
        pass


MCP_UNAVAILABLE_SCRIPT = textwrap.dedent(
    '''
    from types import SimpleNamespace

    import os

    import openai


    # Confirmed against main.py's gate: create_all only runs when
    # AUTO_CREATE_TABLES is 1/true/yes, so "false" keeps this
    # subprocess from ever issuing DDL against the developer's MySQL.
    os.environ["AUTO_CREATE_TABLES"] = "false"


    def fake_create(**kwargs):
        message = SimpleNamespace(
            content="你好，有什么可以帮你？",
            tool_calls=None,
        )

        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=fake_create)
            )


    openai.OpenAI = FakeOpenAI

    import app.mcp.tool_registry as mcp_tool_registry_module


    class UnavailableMCPRegistry:
        """Stands in for MCPToolRegistry when the server cannot start."""

        _tools = {}

        def __init__(self, client):
            self.client = client

        def refresh(self):
            raise RuntimeError("MCP server failed to start")

        def get(self, name):
            return None

        def schemas(self):
            return []

        def execute(self, name, arguments):
            return "unknown MCP tool"


    mcp_tool_registry_module.MCPToolRegistry = UnavailableMCPRegistry

    import main

    # MCP is degraded to "no tools", but initialization succeeded.
    assert main.mcp_tool_registry._tools == {}
    assert main.mcp_tool_registry.get("get_company_info") is None
    assert main.agent_loop.mcp_tool_registry is main.mcp_tool_registry

    # Non-MCP capabilities are untouched.
    assert main.agent_loop.tool_registry.get("calculate") is not None
    assert main.skill_registry.get("knowledge_search") is not None

    # A basic chat that does not need MCP still works.
    reply, task_state = main.agent_loop.run(
        [{"role": "user", "content": "你好"}]
    )

    assert reply == "你好，有什么可以帮你？"
    assert task_state is None

    print("degraded-ok")
    '''
)


def test_main_initializes_when_mcp_server_unavailable():
    result = subprocess.run(
        [sys.executable, "-c", MCP_UNAVAILABLE_SCRIPT],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr
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


def test_chat_agent_failure_returns_503_and_keeps_history_paired(
    monkeypatch,
):
    import main as main_module

    store = _StubDB()

    # Precisely scoped override: any pre-existing get_db override is
    # restored (not cleared) when this test finishes.
    overrides = main_module.app.dependency_overrides
    missing = object()
    previous_get_db = overrides.get(main_module.get_db, missing)
    overrides[main_module.get_db] = lambda: store

    try:
        client = TestClient(
            main_module.app,
            client=("198.51.100.30", 40000),
        )

        monkeypatch.setattr(
            main_module.memory_manager,
            "redis_client",
            DummyRedis(),
        )

        _, token = register_and_login(client)
        user_id = main_module.decode_token(token)["user_id"]

        def exploding_run(messages, role="employee"):
            raise RuntimeError(SECRET_INTERNALS)

        monkeypatch.setattr(main_module.agent_loop, "run", exploding_run)

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

        # The conversation stays paired: user message plus exactly one
        # system fallback reply, with no internal detail stored. Rows
        # were written through the stub session above, so reading them
        # back verifies exactly what the endpoint persisted — without
        # touching the developer's agent_db.
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
    finally:
        if previous_get_db is missing:
            overrides.pop(main_module.get_db, None)
        else:
            overrides[main_module.get_db] = previous_get_db


def test_mcp_tool_call_timeout_is_bounded_and_sanitized(monkeypatch):
    from app.mcp.adapter import MCPToolAdapter
    from app.mcp.client import MCPClient
    from app.mcp.tool_registry import MCPToolRegistry

    client = MCPClient(call_timeout=0.2, startup_timeout=1.0)

    # Never spawn a real server for this test.
    monkeypatch.setattr(client, "start", lambda: None)

    class SlowSession:
        async def call_tool(self, name, arguments=None):
            await asyncio.sleep(30)

            return SimpleNamespace(is_error=False, content=[])

    client._loop = asyncio.new_event_loop()
    client._session = SlowSession()

    thread = threading.Thread(
        target=client._loop.run_forever,
        daemon=True,
    )
    thread.start()

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
        client._loop.call_soon_threadsafe(client._loop.stop)
        thread.join(timeout=2)
        client._loop.close()

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
