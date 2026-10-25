"""Test-only isolation for application imports and external service calls."""

from __future__ import annotations

import importlib
import logging
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parent
_STATE = None


class MCPIsolationError(RuntimeError):
    """Raised when application code tries to start MCP during pytest."""


class _FakeCollection:
    def __init__(self, name: str):
        self.name = name
        self.records: dict[str, dict] = {}

    def add(self, documents, ids, metadatas):
        for document, item_id, metadata in zip(documents, ids, metadatas):
            self.records[item_id] = {
                "document": document,
                "metadata": metadata,
            }

    def get(self, where=None, include=None):
        return {"ids": [], "documents": [], "metadatas": []}

    def query(self, **kwargs):
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    def delete(self, ids=None, where=None):
        for item_id in ids or []:
            self.records.pop(item_id, None)


class _FakeChromaClient:
    def __init__(self, isolated_path: Path):
        self.path = isolated_path.resolve()
        self.collections: dict[str, _FakeCollection] = {}

    def get_or_create_collection(self, name: str):
        return self.collections.setdefault(name, _FakeCollection(name))


class _OfflineOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.request_count = 0
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._reject_request)
        )

    def _reject_request(self, **kwargs):
        self.request_count += 1
        raise AssertionError("offline test OpenAI client rejects requests")


class _OfflineRedis:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.operation_count = 0

    def _reject_operation(self, *args, **kwargs):
        self.operation_count += 1
        raise AssertionError("offline test Redis client rejects operations")

    get = _reject_operation
    setex = _reject_operation
    delete = _reject_operation


class _IsolationState:
    def __init__(self):
        self.temp_dir = tempfile.TemporaryDirectory(
            prefix=".agent-backend-pytest-",
            dir=PROJECT_ROOT,
        )
        self.root = Path(self.temp_dir.name).resolve()
        self.database_path = self.root / "agent-backend-test.sqlite3"
        self.chroma_path = self.root / "chroma-test-double"
        self.requested_database_name: str | None = None
        self.requested_chroma_path: Path | None = None
        self.chroma_fake_calls = 0
        self.mcp_refresh_interceptions = 0
        self.mcp_start_attempts = 0
        self.mcp_stdio_attempts = 0
        self.main_module = None
        self.engine = None
        self.openai_client = None
        self.redis_client = None
        self.created_logging_handlers: list[logging.Handler] = []
        self.original_environment: dict[str, str | None] = {}
        self.patch_records = []
        self.handlers_before: set[logging.Handler] = set()
        self._cleaned = False

    def cleanup(self):
        global _STATE

        if self._cleaned:
            return
        self._cleaned = True
        self.restore_patches()
        self.restore_environment()

        engine, self.engine = self.engine, None
        if engine is not None:
            engine.dispose()
        root_logger = logging.getLogger()
        handlers = self.created_logging_handlers or [
            handler
            for handler in root_logger.handlers
            if handler not in self.handlers_before
        ]
        for handler in handlers:
            root_logger.removeHandler(handler)
            handler.close()
        self.created_logging_handlers = []
        self.temp_dir.cleanup()
        if _STATE is self:
            _STATE = None

    def set_patch(self, target, name, replacement, *, scope):
        record = {
            "target": target,
            "name": name,
            "original": getattr(target, name),
            "replacement": replacement,
            "mapping": False,
            "scope": scope,
            "active": True,
        }
        self.patch_records.append(record)
        setattr(target, name, replacement)
        return record["original"]

    def set_mapping_patch(self, mapping, name, replacement, *, scope):
        record = {
            "target": mapping,
            "name": name,
            "original": mapping[name],
            "replacement": replacement,
            "mapping": True,
            "scope": scope,
            "active": True,
        }
        self.patch_records.append(record)
        mapping[name] = replacement
        return record["original"]

    def restore_patches(self, scope=None):
        for record in reversed(self.patch_records):
            if not record["active"]:
                continue
            if scope is not None and record["scope"] != scope:
                continue
            if record["mapping"]:
                record["target"][record["name"]] = record["original"]
            else:
                setattr(record["target"], record["name"], record["original"])
            record["active"] = False

    def restore_environment(self):
        for key, value in self.original_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def bootstrap_isolated_main(
    mcp_refresh_callback=None,
    environment_overrides=None,
):
    """Import ``main`` with only temporary, offline test resources installed."""
    global _STATE

    if _STATE is not None:
        return _STATE
    if "main" in sys.modules:
        raise RuntimeError(
            "main was imported before the test isolation bootstrap"
        )

    state = _IsolationState()

    # Set required application values only for this import, then restore the
    # process environment so unrelated tests can control their own settings.
    test_environment = {
        "DEEPSEEK_API_KEY": "offline-test-key",
        "JWT_SECRET_KEY": "offline-test-secret-key-with-sufficient-length",
        "MYSQL_PASSWORD": "offline-test-password",
        "AUTO_CREATE_TABLES": "true",
        "LOG_FILE": str(state.root / "pytest-application.log"),
    }
    environment_overrides = dict(environment_overrides or {})
    if mcp_refresh_callback is not None:
        # A refresh callback represents a test that explicitly exercises
        # refresh success/failure. Keep that intent compatible with the new
        # opt-in application default unless the test supplies a value itself.
        environment_overrides.setdefault("MCP_AUTO_REFRESH", "true")
    controlled_environment = set(test_environment) | {
        "MYSQL_HOST",
        "MYSQL_PORT",
        "MYSQL_USER",
        "MYSQL_DATABASE",
        "CHROMA_PERSIST_DIRECTORY",
        "MCP_AUTO_REFRESH",
    } | set(environment_overrides)
    state.original_environment = {
        key: os.environ.get(key) for key in controlled_environment
    }

    import chromadb
    import dotenv
    import openai
    import redis
    from sqlalchemy import create_engine
    from sqlalchemy.engine.url import make_url

    import app.core.database as database_module
    import app.mcp.client as mcp_client_module
    import mcp.client.stdio as mcp_stdio_module
    from app.mcp.client import MCPClient
    from app.mcp.tool_registry import MCPToolRegistry

    original_refresh = MCPToolRegistry.refresh
    state.handlers_before = set(logging.getLogger().handlers)

    expected_database_name = environment_overrides.get(
        "MYSQL_DATABASE",
        "agent_db",
    )
    expected_chroma_path = Path(
        environment_overrides.get(
            "CHROMA_PERSIST_DIRECTORY",
            "./chroma_data",
        )
    )
    if not expected_chroma_path.is_absolute():
        expected_chroma_path = PROJECT_ROOT / expected_chroma_path
    expected_chroma_path = expected_chroma_path.resolve()

    state.expected_chroma_path = expected_chroma_path

    def isolated_engine_factory(business_database_url, **_kwargs):
        # Deliberately discard the configured MySQL URL. The application
        # records only its database name and receives an independent
        # file-backed SQLite database in temp storage. Credentials and the
        # full URL are never retained in test state.
        state.requested_database_name = make_url(
            business_database_url
        ).database
        state.engine = create_engine(
            f"sqlite:///{state.database_path.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        return state.engine

    def isolated_persistent_client(*, path, **_kwargs):
        requested_path = Path(path)
        if not requested_path.is_absolute():
            requested_path = PROJECT_ROOT / requested_path
        state.requested_chroma_path = requested_path.resolve()
        if state.requested_chroma_path != state.expected_chroma_path:
            raise AssertionError("unexpected Chroma path in isolated test")

        # Never forward even the requested path to Chroma's real client.
        state.chroma_fake_calls += 1
        return _FakeChromaClient(state.chroma_path)

    def isolated_openai(**kwargs):
        state.openai_client = _OfflineOpenAI(**kwargs)
        return state.openai_client

    def isolated_redis(**kwargs):
        state.redis_client = _OfflineRedis(**kwargs)
        return state.redis_client

    def block_application_refresh(_registry):
        state.mcp_refresh_interceptions += 1
        if mcp_refresh_callback is not None:
            return mcp_refresh_callback(_registry)
        return None

    def block_mcp_start(_client):
        state.mcp_start_attempts += 1
        raise MCPIsolationError(
            "MCP process startup is disabled during pytest"
        )

    def block_real_registry_refresh(registry):
        if isinstance(registry.client, MCPClient):
            state.mcp_refresh_interceptions += 1
            raise MCPIsolationError(
                "MCP registry refresh is disabled during pytest"
            )
        return original_refresh(registry)

    def block_stdio_start(*_args, **_kwargs):
        state.mcp_stdio_attempts += 1
        raise MCPIsolationError(
            "MCP stdio startup is disabled during pytest"
        )

    try:
        for key in controlled_environment:
            os.environ.pop(key, None)
        os.environ.update(test_environment)
        os.environ.update(environment_overrides)
        # The test bootstrap supplies every required value and must not read
        # a developer's .env while importing the application.
        state.set_patch(
            dotenv,
            "load_dotenv",
            lambda *args, **kwargs: False,
            scope="import",
        )
        state.set_patch(
            database_module,
            "create_engine_safely",
            isolated_engine_factory,
            scope="import",
        )
        state.set_patch(
            chromadb,
            "PersistentClient",
            isolated_persistent_client,
            scope="import",
        )
        state.set_patch(openai, "OpenAI", isolated_openai, scope="import")
        state.set_patch(redis, "Redis", isolated_redis, scope="import")
        state.set_patch(
            MCPToolRegistry,
            "refresh",
            block_application_refresh,
            scope="import",
        )
        state.set_patch(MCPClient, "start", block_mcp_start, scope="session")
        state.set_patch(
            mcp_client_module,
            "stdio_client",
            block_stdio_start,
            scope="session",
        )
        state.set_mapping_patch(
            MCPClient._connection_lifecycle.__globals__,
            "stdio_client",
            block_stdio_start,
            scope="session",
        )
        state.set_patch(
            mcp_stdio_module,
            "stdio_client",
            block_stdio_start,
            scope="session",
        )

        state.main_module = importlib.import_module("main")
        state.created_logging_handlers = [
            handler
            for handler in logging.getLogger().handlers
            if handler not in state.handlers_before
        ]
        state.restore_patches(scope="import")
        state.set_patch(
            MCPToolRegistry,
            "refresh",
            block_real_registry_refresh,
            scope="session",
        )
        state.restore_environment()
    except Exception:
        state.created_logging_handlers = [
            handler
            for handler in logging.getLogger().handlers
            if handler not in state.handlers_before
        ]
        state.cleanup()
        raise

    _STATE = state
    return state


def pytest_configure(config):
    """Run before test-module collection/import, not as a late fixture."""
    bootstrap_isolated_main()


def pytest_unconfigure(config):
    global _STATE

    if _STATE is not None:
        _STATE.cleanup()
        _STATE = None
