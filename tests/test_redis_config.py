"""Tests for Redis client configuration and its failure behavior.

Covers REDIS_PASSWORD handling (set / unset / blank), the other
connection parameters, and the fact that an unreachable Redis still
degrades through MemoryManager instead of raising.
"""

from types import SimpleNamespace

import redis

import main
from app.memory.manager import MemoryManager


def connection_kwargs(client):
    return client.connection_pool.connection_kwargs


def test_password_is_read_from_environment(monkeypatch):
    monkeypatch.setenv("REDIS_PASSWORD", "s3cret-redis")

    client = main.build_redis_client()

    assert connection_kwargs(client)["password"] == "s3cret-redis"


def test_no_password_when_variable_is_unset(monkeypatch):
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)

    client = main.build_redis_client()

    assert connection_kwargs(client)["password"] is None


def test_blank_password_is_treated_as_unset(monkeypatch):
    for blank in ("", "   "):
        monkeypatch.setenv("REDIS_PASSWORD", blank)

        client = main.build_redis_client()

        assert connection_kwargs(client)["password"] is None


def test_password_is_trimmed(monkeypatch):
    monkeypatch.setenv("REDIS_PASSWORD", "  s3cret  ")

    client = main.build_redis_client()

    assert connection_kwargs(client)["password"] == "s3cret"


def test_host_and_port_defaults(monkeypatch):
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.delenv("REDIS_PORT", raising=False)

    client = main.build_redis_client()
    kwargs = connection_kwargs(client)

    assert kwargs["host"] == "localhost"
    assert kwargs["port"] == 6379


def test_host_and_port_can_be_overridden(monkeypatch):
    monkeypatch.setenv("REDIS_HOST", "redis-cache")
    monkeypatch.setenv("REDIS_PORT", "6380")

    client = main.build_redis_client()
    kwargs = connection_kwargs(client)

    assert kwargs["host"] == "redis-cache"
    assert kwargs["port"] == 6380


class FakeColumn:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return ("eq", self.name, other)

    def __gt__(self, other):
        return ("gt", self.name, other)

    def asc(self):
        return ("asc", self.name)

    def desc(self):
        return ("desc", self.name)


class FakeMessage:
    id = FakeColumn("id")
    user_id = FakeColumn("user_id")

    def __init__(self, id, role, content):
        self.id = id
        self.role = role
        self.content = content
        self.user_id = "1"


class FakeSummary:
    user_id = FakeColumn("user_id")

    def __init__(self, summary_text=None, covers_up_to_message_id=0, version=0):
        self.summary_text = summary_text
        self.covers_up_to_message_id = covers_up_to_message_id
        self.user_id = "1"
        self.version = version


class FakeQuery:
    def __init__(self, rows):
        self.rows = rows

    def filter(self, *conditions):
        for operator, field, value in conditions:
            if operator == "eq":
                self.rows = [
                    row for row in self.rows
                    if str(getattr(row, field)) == str(value)
                ]
            elif operator == "gt":
                self.rows = [
                    row for row in self.rows
                    if getattr(row, field) > value
                ]

        return self

    def order_by(self, *args):
        for arg in args:
            reverse = arg[0] == "desc"
            field = arg[1]

            self.rows = sorted(
                self.rows,
                key=lambda row: getattr(row, field),
                reverse=reverse,
            )

        return self

    def limit(self, count):
        self.rows = self.rows[:count]
        return self

    def first(self):
        return self.rows[0] if self.rows else None

    def all(self):
        return self.rows


class FakeDB:
    def __init__(self, messages, summary=None):
        self.messages = messages
        self.summary = summary

    def query(self, model):
        if model is FakeSummary:
            return FakeQuery([self.summary] if self.summary else [])

        if model is FakeMessage:
            return FakeQuery(list(self.messages))

        raise ValueError(f"Unknown model: {model}")


class FakeLLM:
    chat = SimpleNamespace(
        completions=SimpleNamespace(create=lambda **kwargs: None)
    )


class FailingRedisClient:
    """Stands in for a real client when Redis is unreachable.

    Raises the real redis exception type. A live connection is not
    used on purpose: redis-py retries a refused connection for about
    26 seconds, which would make this test slow and flaky.
    """

    def get(self, key):
        raise redis.exceptions.ConnectionError(
            "Error 111 connecting to redis:6379. Connection refused."
        )

    def setex(self, key, ttl, value):
        raise redis.exceptions.ConnectionError("redis unreachable")

    def delete(self, key):
        raise redis.exceptions.ConnectionError("redis unreachable")


def build_manager(redis_client):
    return MemoryManager(
        redis_client=redis_client,
        llm_client=FakeLLM(),
        message_model=FakeMessage,
        summary_model=FakeSummary,
    )


def test_connection_failure_still_degrades_to_database():
    manager = build_manager(FailingRedisClient())

    messages = [
        FakeMessage(1, "user", "你好"),
        FakeMessage(2, "assistant", "你好，有什么可以帮你？"),
    ]

    history = manager.get_chat_history(1, FakeDB(messages))

    assert history == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，有什么可以帮你？"},
    ]


def test_connection_failure_still_returns_summary_from_database():
    manager = build_manager(FailingRedisClient())

    messages = [
        FakeMessage(1, "user", "你好"),
        FakeMessage(2, "assistant", "你好"),
        FakeMessage(3, "user", "继续开发 Agent"),
        FakeMessage(4, "assistant", "好的"),
    ]

    summary = FakeSummary(
        summary_text="用户正在开发企业内部 Agent。",
        covers_up_to_message_id=2,
    )

    history = manager.get_chat_history(1, FakeDB(messages, summary))

    assert history[0]["role"] == "system"
    assert "企业内部 Agent" in history[0]["content"]
    assert len(history) == 3
    assert history[-1]["content"] == "好的"
