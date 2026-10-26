from types import SimpleNamespace

from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import ast
import json
import re
import tempfile

import pytest

from app.agent.request_budget import (
    DEFAULT_MAX_CONTEXT_CHARS,
    serialized_request_size,
)
from app.memory.manager import MemoryManager


@pytest.fixture
def memory_sqlite_path():
    repository_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(
        prefix=".memory-sqlite-test-", dir=repository_root
    ) as isolated_dir:
        isolated_dir = Path(isolated_dir).resolve()
        assert isolated_dir.parent == repository_root
        yield isolated_dir / "memory-test.sqlite3"


class FakeColumn:
    __hash__ = object.__hash__

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


class FakeRedis:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def setex(self, key, seconds, value):
        self.data[key] = value

    def delete(self, key):
        self.data.pop(key, None)


class FakeQuery:
    def __init__(self, rows):
        self.rows = rows

    def filter(self, *conditions):
        for condition in conditions:
            operator, field, value = condition

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

    def with_for_update(self):
        return self

    def update(self, values, synchronize_session=False):
        if not self.rows:
            return 0
        for row in self.rows:
            for column, value in values.items():
                setattr(row, column.name, value)
        return len(self.rows)

    def first(self):
        return self.rows[0] if self.rows else None

    def all(self):
        return self.rows


class FakeDB:
    def __init__(self, messages, summary=None, commit_error=None):
        self.messages = messages
        self.summary = summary
        self.commit_error = commit_error
        self.rollback_count = 0
        self.committed_summary = summary
        self.committed_summary_values = self._summary_values(summary)

    def query(self, model):
        if model is FakeSummary:
            rows = [self.summary] if self.summary else []
            return FakeQuery(rows)

        if model is FakeMessage:
            return FakeQuery(self.messages)

        raise ValueError(f"Unknown model: {model}")

    def add(self, obj):
        self.summary = obj

    def commit(self):
        if isinstance(self.commit_error, list) and self.commit_error:
            raise self.commit_error.pop(0)
        if self.commit_error is not None and not isinstance(self.commit_error, list):
            raise self.commit_error
        self.committed_summary = self.summary
        self.committed_summary_values = self._summary_values(self.summary)

    def rollback(self):
        self.rollback_count += 1
        self.summary = self.committed_summary
        if self.summary is not None:
            (
                self.summary.summary_text,
                self.summary.covers_up_to_message_id,
                self.summary.user_id,
                self.summary.pending_message_id,
                self.summary.pending_content_hash,
                self.summary.pending_offset,
                self.summary.pending_summary_text,
                self.summary.failure_count,
                self.summary.next_retry_at,
                self.summary.failure_status,
                self.summary.failure_code,
                self.summary.version,
            ) = self.committed_summary_values

    @staticmethod
    def _summary_values(summary):
        if summary is None:
            return None
        return (
            summary.summary_text,
            summary.covers_up_to_message_id,
            summary.user_id,
            summary.pending_message_id,
            summary.pending_content_hash,
            summary.pending_offset,
            summary.pending_summary_text,
            summary.failure_count,
            summary.next_retry_at,
            summary.failure_status,
            summary.failure_code,
            summary.version,
        )


class MultiUserFakeDB(FakeDB):
    """Share message and summary rows while retaining the existing fake query."""

    def __init__(self, messages, summaries):
        super().__init__(messages)
        self.summaries = {str(summary.user_id): summary for summary in summaries}

    def query(self, model):
        if model is FakeSummary:
            return FakeQuery(list(self.summaries.values()))
        return super().query(model)

    def add(self, obj):
        self.summaries[str(obj.user_id)] = obj


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
    version = FakeColumn("version")
    summary_text = FakeColumn("summary_text")
    covers_up_to_message_id = FakeColumn("covers_up_to_message_id")
    pending_message_id = FakeColumn("pending_message_id")
    pending_content_hash = FakeColumn("pending_content_hash")
    pending_offset = FakeColumn("pending_offset")
    pending_summary_text = FakeColumn("pending_summary_text")
    failure_count = FakeColumn("failure_count")
    next_retry_at = FakeColumn("next_retry_at")
    failure_status = FakeColumn("failure_status")
    failure_code = FakeColumn("failure_code")

    def __init__(
        self,
        summary_text,
        covers_up_to_message_id,
        user_id="1",
        **state,
    ):
        self.user_id = user_id
        self.summary_text = summary_text
        self.covers_up_to_message_id = covers_up_to_message_id
        self.pending_message_id = state.get("pending_message_id")
        self.pending_content_hash = state.get("pending_content_hash")
        self.pending_offset = state.get("pending_offset", 0)
        self.pending_summary_text = state.get("pending_summary_text")
        self.failure_count = state.get("failure_count", 0)
        self.next_retry_at = state.get("next_retry_at")
        self.failure_status = state.get("failure_status")
        self.failure_code = state.get("failure_code")
        self.version = state.get("version", 0)


class FakeLLM:
    def __init__(self, error=None, summary_text="新的对话摘要", scripted=None):
        self.called = False
        self.last_prompt = None
        self.requests = []
        self.error = error
        self.summary_text = summary_text
        self.scripted = list(scripted or [])

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.create
            )
        )

    def create(self, **kwargs):
        self.called = True
        self.requests.append(deepcopy(kwargs))
        self.last_prompt = kwargs["messages"][1]["content"]

        if self.scripted:
            outcome = self.scripted.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            summary_text = outcome
        elif self.error is not None:
            raise self.error
        else:
            summary_text = self.summary_text

        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=summary_text))
            ]
        )


def create_manager(llm, max_context_chars=DEFAULT_MAX_CONTEXT_CHARS):
    return MemoryManager(
        redis_client=FakeRedis(),
        llm_client=llm,
        message_model=FakeMessage,
        summary_model=FakeSummary,
        max_context_chars=max_context_chars,
    )


def _file_sqlite_memory_models(database_path):
    from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
    from sqlalchemy.orm import declarative_base, sessionmaker

    database_path = Path(database_path).resolve()
    Base = declarative_base()

    class MessageRow(Base):
        __tablename__ = "memory_test_messages"

        id = Column(Integer, primary_key=True)
        user_id = Column(String(50), nullable=False, index=True)
        role = Column(String(20), nullable=False)
        content = Column(Text, nullable=False)

    class SummaryRow(Base):
        __tablename__ = "memory_test_summaries"

        user_id = Column(String(50), primary_key=True)
        summary_text = Column(Text)
        covers_up_to_message_id = Column(Integer, nullable=False, default=0)
        pending_message_id = Column(Integer)
        pending_content_hash = Column(String(64))
        pending_offset = Column(Integer, nullable=False, default=0)
        pending_summary_text = Column(Text)
        failure_count = Column(Integer, nullable=False, default=0)
        next_retry_at = Column(DateTime)
        failure_status = Column(String(20))
        failure_code = Column(String(64))
        version = Column(Integer, nullable=False, default=0)

    engine = create_engine(
        f"sqlite:///{database_path.as_posix()}",
        connect_args={"check_same_thread": False, "timeout": 5},
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return engine, Session, MessageRow, SummaryRow


def _cleanup_file_sqlite(engine, database_path):
    engine.dispose()
    database_path = Path(database_path).resolve()
    for suffix in ("", "-journal", "-wal", "-shm"):
        owned_path = Path(f"{database_path}{suffix}")
        if owned_path.exists():
            owned_path.unlink()


class BrokenRedis:
    """Redis stub that fails on every operation."""

    def get(self, key):
        raise ConnectionError(
            "Error 111 connecting to redis:6379. Connection refused."
        )

    def setex(self, key, seconds, value):
        raise ConnectionError("redis is unavailable")

    def delete(self, key):
        raise ConnectionError("redis is unavailable")


def create_manager_with_redis(redis_client, llm):
    return MemoryManager(
        redis_client=redis_client,
        llm_client=llm,
        message_model=FakeMessage,
        summary_model=FakeSummary,
    )


def test_get_history_without_summary():
    messages = [
        FakeMessage(1, "user", "你好"),
        FakeMessage(2, "assistant", "你好，有什么可以帮你？"),
    ]

    db = FakeDB(messages)
    manager = create_manager(FakeLLM())

    history = manager.get_chat_history(1, db)

    assert len(history) == 2
    assert history[0]["content"] == "你好"
    assert history[1]["content"] == "你好，有什么可以帮你？"


def test_get_history_is_capped_at_recent_limit():
    messages = [
        FakeMessage(index, "user", f"message {index}")
        for index in range(1, 31)
    ]

    db = FakeDB(messages)
    manager = create_manager(FakeLLM())
    manager.recent_limit = 20

    history = manager.get_chat_history(1, db)

    assert len(history) == 20
    assert history[0]["content"] == "message 11"
    assert history[-1]["content"] == "message 30"


def test_recent_messages_after_summary_are_capped():
    summary = FakeSummary("早期对话摘要", 2)

    messages = [
        FakeMessage(index, "user", f"message {index}")
        for index in range(1, 31)
    ]

    db = FakeDB(messages, summary)
    manager = create_manager(FakeLLM())
    manager.recent_limit = 5

    history = manager.get_chat_history(1, db)

    assert history[0]["role"] == "system"
    assert "早期对话摘要" in history[0]["content"]
    assert len(history) == 6
    assert history[-1]["content"] == "message 30"


def test_get_history_survives_redis_failure():
    messages = [
        FakeMessage(1, "user", "你好"),
        FakeMessage(2, "assistant", "你好，有什么可以帮你？"),
    ]

    db = FakeDB(messages)
    manager = create_manager_with_redis(BrokenRedis(), FakeLLM())

    history = manager.get_chat_history(1, db)

    assert len(history) == 2
    assert history[0]["content"] == "你好"
    assert history[1]["content"] == "你好，有什么可以帮你？"


def test_summary_history_survives_redis_failure():
    summary = FakeSummary("用户正在开发企业内部 Agent。", 2)

    messages = [
        FakeMessage(1, "user", "你好"),
        FakeMessage(2, "assistant", "你好"),
        FakeMessage(3, "user", "继续开发 Agent"),
        FakeMessage(4, "assistant", "好的"),
    ]

    db = FakeDB(messages, summary)
    manager = create_manager_with_redis(BrokenRedis(), FakeLLM())

    history = manager.get_chat_history(1, db)

    assert history[0]["role"] == "system"
    assert "企业内部 Agent" in history[0]["content"]
    assert len(history) == 3
    assert history[-1]["content"] == "好的"


def test_stale_summary_cache_is_rejected_when_cache_delete_failed():
    class DeleteFailRedis(FakeRedis):
        def delete(self, key):
            raise ConnectionError("offline")

    summary = FakeSummary("new summary", 1, version=2)
    messages = [FakeMessage(2, "user", "new message")]
    redis = DeleteFailRedis()
    redis.data["summary:1"] = '{"summary":"stale","covers_up_to":0,"version":1}'
    manager = MemoryManager(
        redis, FakeLLM(), FakeMessage, FakeSummary
    )
    db = FakeDB(messages, summary)

    history = manager.get_chat_history(1, db)

    assert "new summary" in history[0]["content"]
    assert "stale" not in history[0]["content"]


def test_compression_survives_redis_failure():
    messages = [
        FakeMessage(index, "user", f"message {index}")
        for index in range(1, 21)
    ]

    db = FakeDB(messages)
    manager = create_manager_with_redis(BrokenRedis(), FakeLLM())

    manager.maybe_compress_history(
        user_id=1,
        db=db,
        threshold=20,
    )

    assert db.summary is not None
    assert db.summary.summary_text == "新的对话摘要"


def test_get_history_with_summary():
    summary = FakeSummary(
        "用户正在开发企业内部 Agent。",
        2,
    )

    messages = [
        FakeMessage(1, "user", "你好"),
        FakeMessage(2, "assistant", "你好"),
        FakeMessage(3, "user", "继续开发 Agent"),
        FakeMessage(4, "assistant", "好的"),
    ]

    db = FakeDB(messages, summary)
    manager = create_manager(FakeLLM())

    history = manager.get_chat_history(1, db)

    assert history[0]["role"] == "system"
    assert "企业内部 Agent" in history[0]["content"]

    assert len(history) == 3
    assert history[1]["content"] == "继续开发 Agent"
    assert history[2]["content"] == "好的"


def test_memory_history_cache_cas_recovery_and_pending_state_are_user_scoped():
    messages = [
        FakeMessage(1, "user", "A earlier"),
        FakeMessage(2, "user", "B earlier"),
        FakeMessage(3, "assistant", "A recent"),
        FakeMessage(4, "assistant", "B recent"),
    ]
    for message, user_id in zip(messages, ("A", "B", "A", "B")):
        message.user_id = user_id
    summary_a = FakeSummary("A summary", 1, user_id="A", version=4)
    summary_b = FakeSummary("B summary", 2, user_id="B", version=8)
    db = MultiUserFakeDB(messages, [summary_a, summary_b])
    manager = create_manager(FakeLLM())

    history_a = manager.get_chat_history("A", db)
    history_b = manager.get_chat_history("B", db)
    assert "A summary" in history_a[0]["content"]
    assert [message["content"] for message in history_a[1:]] == ["A recent"]
    assert "B summary" in history_b[0]["content"]
    assert [message["content"] for message in history_b[1:]] == ["B recent"]
    cached_a = json.loads(manager.redis_client.data["summary:A"])
    cached_b = json.loads(manager.redis_client.data["summary:B"])
    assert cached_a["summary"] == "A summary"
    assert cached_a["version"] == 4
    assert cached_b["summary"] == "B summary"
    assert cached_b["version"] == 8

    assert MemoryManager._cas_update(
        db,
        FakeSummary,
        "A",
        summary_a,
        {FakeSummary.summary_text: "A updated"},
    ) is True
    assert summary_a.summary_text == "A updated"
    assert summary_b.summary_text == "B summary"
    summary_a.failure_status = "failed"
    summary_b.failure_status = "failed"
    summary_b.version = 8
    assert manager.recover_failed_compression("A", db) is True
    assert summary_a.failure_status is None
    assert summary_b.failure_status == "failed"
    assert summary_b.version == 8

    pending_a = FakeSummary(
        "A summary", 1, user_id="A", pending_message_id=2,
        pending_content_hash=manager._content_hash("B earlier"),
        pending_offset=3, pending_summary_text="A partial", version=10,
    )
    pending_b = FakeSummary(
        "B summary", 2, user_id="B", pending_message_id=4,
        pending_content_hash=manager._content_hash("B recent"),
        pending_offset=7, pending_summary_text="B partial", version=12,
    )
    pending_db = MultiUserFakeDB(messages, [pending_a, pending_b])
    llm = FakeLLM()
    pending_manager = create_manager(llm)
    pending_manager.maybe_compress_history("A", pending_db, threshold=20)

    assert llm.requests == []
    assert pending_a.failure_status == "failed"
    assert pending_a.failure_code == "pending_message_not_next"
    assert pending_a.pending_message_id == 2
    assert pending_a.pending_content_hash == manager._content_hash("B earlier")
    assert pending_a.pending_offset == 3
    assert pending_a.pending_summary_text == "A partial"
    assert pending_b.failure_status is None
    assert pending_b.pending_message_id == 4
    assert pending_b.pending_offset == 7
    assert pending_b.pending_summary_text == "B partial"


def test_compress_history_when_threshold_reached():
    messages = [
        FakeMessage(i, "user", f"message {i}")
        for i in range(1, 21)
    ]

    llm = FakeLLM()
    db = FakeDB(messages)
    manager = create_manager(llm)

    manager.maybe_compress_history(
        user_id=1,
        db=db,
        threshold=20,
    )

    assert llm.called is True
    assert db.summary is not None
    assert db.summary.summary_text == "新的对话摘要"
    assert db.summary.covers_up_to_message_id == 20
    assert len(llm.requests) == 1
    request = llm.requests[0]
    assert serialized_request_size(
        request["model"],
        request["messages"],
    ) <= manager.max_context_chars


def test_compress_history_includes_previous_summary():
    messages = [
        FakeMessage(i, "user", f"message {i}")
        for i in range(3, 23)
    ]

    old_summary = FakeSummary(
        summary_text="\u7528\u6237\u4e4b\u524d\u5df2\u7ecf\u5b8c\u6210\u4e86 MCP \u63a5\u5165\u3002",
        covers_up_to_message_id=2,
    )

    llm = FakeLLM()
    db = FakeDB(messages, old_summary)
    manager = create_manager(llm)

    manager.maybe_compress_history(
        user_id=1,
        db=db,
        threshold=20,
    )

    assert llm.called is True
    assert "\u7528\u6237\u4e4b\u524d\u5df2\u7ecf\u5b8c\u6210\u4e86 MCP \u63a5\u5165\u3002" in llm.last_prompt
    assert "message 3" in llm.last_prompt
    assert "message 22" in llm.last_prompt


def test_summary_manager_uses_shared_default_context_budget():
    manager = create_manager(FakeLLM())

    assert manager.max_context_chars == DEFAULT_MAX_CONTEXT_CHARS == 24_000


def test_large_history_is_summarized_in_earliest_contiguous_budgeted_batch():
    messages = [
        FakeMessage(index, "user", f"message {index}: " + "x" * 500)
        for index in range(1, 21)
    ]
    llm = FakeLLM()
    db = FakeDB(messages)
    manager = create_manager(llm, max_context_chars=1_000)

    manager.maybe_compress_history(1, db, threshold=20)

    assert len(llm.requests) == 1
    request = llm.requests[0]
    assert serialized_request_size(
        request["model"], request["messages"]
    ) <= 1_000
    assert db.summary is not None
    assert 1 <= db.summary.covers_up_to_message_id < 20
    assert [message.id for message in messages if message.id > db.summary.covers_up_to_message_id]
    assert f"message {db.summary.covers_up_to_message_id}:" in llm.last_prompt
    assert f"message {db.summary.covers_up_to_message_id + 1}:" not in llm.last_prompt

    previous_cursor = db.summary.covers_up_to_message_id
    manager.maybe_compress_history(1, db, threshold=1)

    assert len(llm.requests) == 2
    assert serialized_request_size(
        llm.requests[1]["model"], llm.requests[1]["messages"]
    ) <= 1_000
    assert db.summary.covers_up_to_message_id > previous_cursor
    assert "Previous summary:\n新的对话摘要" in llm.requests[1]["messages"][1]["content"]


def test_summary_budget_counts_system_message_previous_summary_and_new_rows():
    previous_text = "p" * 200
    summary = FakeSummary(previous_text, 0)
    messages = [
        FakeMessage(index, "user", f"new message {index}: " + "r" * 200)
        for index in range(1, 21)
    ]
    llm = FakeLLM(summary_text="compact")
    manager = create_manager(llm, max_context_chars=1_000)
    db = FakeDB(messages, summary)
    manager.redis_client.data["summary:1"] = "cached old summary"

    manager.maybe_compress_history(1, db, threshold=20)

    request = llm.requests[0]
    assert set(request) == {"model", "messages"}
    assert request["messages"][0]["content"]
    assert "Previous summary:\n" + previous_text in request["messages"][1]["content"]
    assert "new message 1:" in request["messages"][1]["content"]
    actual_size = serialized_request_size(request["model"], request["messages"])
    assert actual_size == serialized_request_size(
        "deepseek-chat", request["messages"]
    )
    assert actual_size <= manager.max_context_chars
    assert manager.redis_client.data.get("summary:1") is None


def test_single_oversized_message_starts_chunking_without_advancing_cursor():
    previous_summary = FakeSummary("still valid", 0)
    messages = [
        FakeMessage(index, "user", "x" * 1_500)
        for index in range(1, 21)
    ]
    llm = FakeLLM(summary_text="compact")
    manager = create_manager(llm, max_context_chars=1_000)
    db = FakeDB(messages, previous_summary)

    manager.maybe_compress_history(1, db, threshold=20)

    assert len(llm.requests) == 1
    assert serialized_request_size(
        llm.requests[0]["model"], llm.requests[0]["messages"]
    ) <= manager.max_context_chars
    assert db.summary.summary_text == "still valid"
    assert db.summary.covers_up_to_message_id == 0
    assert db.summary.pending_message_id == 1
    assert db.summary.pending_offset > 0
    assert len(db.messages) == 20


def test_oversized_message_after_prefix_is_resumable_and_later_rows_process():
    messages = [
        FakeMessage(index, "user", f"ordinary {index}")
        for index in range(1, 21)
    ]
    messages[3].content = "段落\n" + ("特殊字符 \" \\ 😀 " * 400)
    original_contents = [message.content for message in messages]
    llm = FakeLLM(summary_text="compact")
    manager = create_manager(llm, max_context_chars=1_000)
    db = FakeDB(messages)

    manager.maybe_compress_history(1, db, threshold=20)

    assert db.summary.covers_up_to_message_id == 3
    assert db.summary.pending_message_id == 4
    assert db.summary.pending_content_hash == manager._content_hash(messages[3].content)
    assert 0 < db.summary.pending_offset < len(messages[3].content)
    assert [message.content for message in messages] == original_contents
    assert len(llm.requests) == 2
    assert all(
        serialized_request_size(request["model"], request["messages"]) <= 1_000
        for request in llm.requests
    )

    previous_offset = db.summary.pending_offset
    resumed_manager = create_manager(llm, max_context_chars=1_000)
    resumed_manager.redis_client.data["summary:1"] = "cached prior state"
    final_request = None
    while db.summary.pending_message_id is not None and len(llm.requests) < 100:
        request_count = len(llm.requests)
        resumed_manager.maybe_compress_history(1, db, threshold=20)
        if len(llm.requests) > request_count:
            final_request = llm.requests[-1]
        if db.summary.pending_message_id is not None:
            assert resumed_manager.redis_client.data["summary:1"] == "cached prior state"
            if db.summary.pending_offset > previous_offset:
                assert db.summary.covers_up_to_message_id == 3
                previous_offset = db.summary.pending_offset
            else:
                pytest.fail(
                    "chunk processing stopped before completing the message: "
                    f"{db.summary.failure_code}"
                )

    assert db.summary.pending_message_id is None
    assert db.summary.pending_offset == 0
    assert db.summary.pending_summary_text is None
    assert db.summary.summary_text == "compact"
    assert db.summary.covers_up_to_message_id == 4
    assert db.summary.failure_status is None
    assert "summary:1" not in resumed_manager.redis_client.data
    assert [message.content for message in messages] == original_contents
    assert all(
        serialized_request_size(request["model"], request["messages"]) <= 1_000
        for request in llm.requests
    )
    assert final_request is not None
    final_user = final_request["messages"][1]["content"]
    match = re.search(
        r"Message 4 \(user\), characters (\d+)-(\d+):\n",
        final_user,
    )
    assert match is not None
    final_start, final_end = map(int, match.groups())
    assert 0 < final_start < final_end == len(messages[3].content)
    assert "Previous summary:\ncompact" in final_user
    assert final_user[match.end():] == messages[3].content[final_start:final_end]
    assert messages[3].content not in final_user


def test_first_and_only_chunk_completes_without_a_previous_partial_summary():
    llm = FakeLLM(scripted=["complete summary"])
    manager, db, message = _pending_chunk(offset=0)
    message.content = "shortfirstandfinalchunk"
    db.summary.summary_text = None
    db.summary.pending_content_hash = manager._content_hash(message.content)
    manager.llm_client = llm

    manager.maybe_compress_history(1, db, threshold=20)

    assert len(llm.requests) == 1
    request = llm.requests[0]
    assert serialized_request_size(request["model"], request["messages"]) <= 1_000
    assert "Previous summary:" not in request["messages"][1]["content"]
    assert "characters 0-" in request["messages"][1]["content"]
    assert db.summary.summary_text == "complete summary"
    assert db.summary.covers_up_to_message_id == 1
    assert db.summary.pending_message_id is None
    assert db.summary.pending_offset == 0
    assert db.summary.pending_summary_text is None
    assert message.content == "shortfirstandfinalchunk"


@pytest.mark.parametrize(
    "outcome, error_match, failure_code",
    [
        (RuntimeError("offline"), "offline", "model_or_summary_error"),
        ("", "empty summary", "invalid_summary_response"),
    ],
)
def test_final_chunk_model_failure_preserves_committed_progress(
    outcome, error_match, failure_code
):
    llm = FakeLLM(scripted=[outcome])
    manager, db, message = _pending_chunk(offset=49_900)
    manager.llm_client = llm
    manager.redis_client.data["summary:1"] = "cached previous value"
    committed_summary = db.summary.pending_summary_text
    original_content = message.content

    with pytest.raises(Exception, match=error_match):
        manager.maybe_compress_history(1, db, threshold=20)

    request = llm.requests[0]
    assert serialized_request_size(request["model"], request["messages"]) <= 1_000
    assert "Previous summary:\nsafe prior" in request["messages"][1]["content"]
    assert original_content not in request["messages"][1]["content"]
    assert db.summary.covers_up_to_message_id == 0
    assert db.summary.pending_message_id == 1
    assert db.summary.pending_offset == 49_900
    assert db.summary.pending_summary_text == committed_summary
    assert db.summary.failure_code == failure_code
    assert message.content == original_content
    assert manager.redis_client.data["summary:1"] == "cached previous value"


def test_final_chunk_commit_failure_keeps_cursor_and_pending_progress():
    llm = FakeLLM(scripted=["final summary"])
    manager, db, message = _pending_chunk(offset=49_900)
    manager.llm_client = llm
    manager.redis_client.data["summary:1"] = "cached previous value"
    db.commit_error = RuntimeError("commit failed")
    original_content = message.content

    with pytest.raises(RuntimeError, match="commit failed"):
        manager.maybe_compress_history(1, db, threshold=20)

    request = llm.requests[0]
    assert serialized_request_size(request["model"], request["messages"]) <= 1_000
    assert db.summary.covers_up_to_message_id == 0
    assert db.summary.pending_message_id == 1
    assert db.summary.pending_offset == 49_900
    assert db.summary.pending_summary_text == "safe prior"
    assert message.content == original_content
    assert manager.redis_client.data["summary:1"] == "cached previous value"


def test_final_chunk_oversized_output_gets_one_bounded_repair():
    llm = FakeLLM(scripted=["z" * 1_400, "compact final"])
    manager, db, message = _pending_chunk(offset=49_900)
    manager.max_context_chars = 1_600
    manager.llm_client = llm
    original_content = message.content

    manager.maybe_compress_history(1, db, threshold=20)

    assert len(llm.requests) == 2
    assert all(
        serialized_request_size(request["model"], request["messages"])
        <= 1_600
        for request in llm.requests
    )
    assert "characters 49900-50000:" in llm.requests[0]["messages"][1]["content"]
    assert original_content not in llm.requests[0]["messages"][1]["content"]
    assert llm.requests[1]["messages"][1]["content"] == "z" * 1_400
    assert db.summary.summary_text == "compact final"
    assert db.summary.covers_up_to_message_id == 1
    assert db.summary.pending_message_id is None
    assert db.summary.pending_summary_text is None
    assert message.content == original_content


def test_failed_chunk_retries_after_backoff_from_persisted_offset(monkeypatch):
    messages = [
        FakeMessage(index, "user", "x" * 5_000 if index == 1 else f"m{index}")
        for index in range(1, 21)
    ]
    llm = FakeLLM(scripted=["first chunk summary", RuntimeError("offline"), "resumed summary"])
    manager = create_manager(llm, max_context_chars=1_000)
    db = FakeDB(messages)
    manager.maybe_compress_history(1, db, threshold=20)

    committed_offset = db.summary.pending_offset
    assert committed_offset > 0
    with pytest.raises(RuntimeError, match="offline"):
        manager.maybe_compress_history(1, db, threshold=20)
    retry_at = db.summary.next_retry_at
    assert db.summary.pending_offset == committed_offset
    assert db.summary.failure_status == "retry_wait"
    assert db.summary.covers_up_to_message_id == 0
    request_count = len(llm.requests)

    manager.maybe_compress_history(1, db, threshold=20)
    assert len(llm.requests) == request_count

    monkeypatch.setattr(
        MemoryManager,
        "_now_utc",
        staticmethod(lambda: retry_at + timedelta(seconds=1)),
    )
    manager.maybe_compress_history(1, db, threshold=20)
    assert len(llm.requests) == request_count + 1
    assert f"characters {committed_offset}-" in llm.last_prompt
    assert "Previous summary:\nfirst chunk summary" in llm.last_prompt
    assert db.summary.pending_offset > committed_offset
    assert db.summary.failure_status is None


def _pending_chunk(summary_text=None, offset=100):
    message = FakeMessage(1, "user", "x" * 50_000)
    manager = create_manager(FakeLLM(), max_context_chars=1_000)
    record = FakeSummary(
        None if offset == 0 else "already committed", 0, pending_message_id=1,
        pending_content_hash=manager._content_hash(message.content),
        pending_offset=offset,
        pending_summary_text=(summary_text if summary_text is not None else "safe prior"),
        version=4,
    )
    return manager, FakeDB([message], record), message


def test_chunk_summary_output_is_budgeted_and_can_continue():
    llm = FakeLLM(scripted=["compact", "compact next"])
    manager = create_manager(llm, max_context_chars=1_000)
    db = FakeDB([FakeMessage(1, "user", "x" * 5_000)])

    manager.maybe_compress_history(1, db, threshold=20)
    first_offset = db.summary.pending_offset
    assert first_offset > 0
    assert db.summary.pending_summary_text == "compact"
    assert "Target summary length" in llm.requests[0]["messages"][0]["content"]
    manager.maybe_compress_history(1, db, threshold=20)
    assert db.summary.pending_offset > first_offset
    assert len(llm.requests) == 2
    assert all(
        serialized_request_size(request["model"], request["messages"]) <= 1_000
        for request in llm.requests
    )


def test_oversized_chunk_output_gets_one_bounded_repair_then_atomic_progress():
    llm = FakeLLM(scripted=["z" * 1_400, "small summary"])
    manager, db, message = _pending_chunk(offset=0)
    db.summary.summary_text = None
    db.summary.pending_summary_text = None
    manager.max_context_chars = 1_600
    manager.llm_client = llm
    manager.maybe_compress_history(1, db, threshold=20)

    assert len(llm.requests) == 2
    assert db.summary.pending_offset > 0
    assert db.summary.pending_summary_text == "small summary"
    assert db.summary.covers_up_to_message_id == 0
    assert message.content == "x" * 50_000
    assert "Compress this summary" in llm.requests[1]["messages"][0]["content"]
    assert "z" * 1_400 == llm.requests[1]["messages"][1]["content"]
    assert all(
        serialized_request_size(request["model"], request["messages"]) <= 1_600
        for request in llm.requests
    )


def test_oversized_repair_result_fails_without_advancing_offset():
    llm = FakeLLM(scripted=["z" * 1_400, "q" * 1_400])
    manager, db, message = _pending_chunk(offset=0)
    db.summary.summary_text = None
    db.summary.pending_summary_text = None
    manager.max_context_chars = 1_600
    manager.llm_client = llm
    old_summary = db.summary.pending_summary_text
    manager.redis_client.data["summary:1"] = "cached prior value"
    manager.maybe_compress_history(1, db, threshold=20)

    assert len(llm.requests) == 2
    assert "Compress this summary" in llm.requests[1]["messages"][0]["content"]
    assert "z" * 1_400 == llm.requests[1]["messages"][1]["content"]
    assert db.summary.pending_offset == 0
    assert db.summary.pending_summary_text == old_summary
    assert db.summary.failure_status == "failed"
    assert db.summary.failure_code == "summary_exceeds_continuation_budget"
    assert db.summary.covers_up_to_message_id == 0
    assert manager.redis_client.data["summary:1"] == "cached prior value"
    assert message.content == "x" * 50_000


def test_legacy_oversized_pending_summary_is_repaired_once_without_offset_move():
    llm = FakeLLM(scripted=["r" * 900, "repaired"])
    manager, db, message = _pending_chunk("s" * 900, offset=100)
    manager.max_context_chars = 1_000
    manager.llm_client = llm
    manager.maybe_compress_history(1, db, threshold=20)

    assert len(llm.requests) <= 1
    assert db.summary.pending_offset == 100
    assert db.summary.failure_status == "failed"
    assert db.summary.failure_code == "partial_summary_unrecoverable"
    assert message.content == "x" * 50_000
    manager.maybe_compress_history(1, db, threshold=20)
    assert len(llm.requests) <= 1


def test_full_continuation_request_budget_includes_prompt_and_previous_text():
    manager, db, _ = _pending_chunk("p" * 700, offset=100)
    llm = FakeLLM(scripted=["compact repair"])
    manager.llm_client = llm
    manager.maybe_compress_history(1, db, threshold=20)

    assert len(llm.requests) == 1
    assert db.summary.pending_offset == 100
    assert db.summary.pending_summary_text == "compact repair"
    assert db.summary.failure_status is None
    assert serialized_request_size(
        llm.requests[0]["model"], llm.requests[0]["messages"]
    ) <= manager.max_context_chars
    too_large = manager._chunk_summary_request(
        "p" * 700, "user", 1, "x", 100, 101
    )
    assert serialized_request_size("deepseek-chat", too_large) > 1_000


def test_failed_repair_retries_with_backoff_then_requires_operator_recovery(
    monkeypatch,
):
    from app.memory.manager import (
        MAX_SUMMARY_FAILURES,
        SUMMARY_RETRY_BASE_SECONDS,
        SUMMARY_RETRY_MAX_SECONDS,
    )

    llm = FakeLLM(error=RuntimeError("offline"))
    manager, db, message = _pending_chunk("s" * 700, offset=100)
    manager.max_context_chars = 1_000
    manager.llm_client = llm
    now = [datetime(2026, 1, 1)]
    monkeypatch.setattr(MemoryManager, "_now_utc", staticmethod(lambda: now[0]))
    committed_partial = db.summary.pending_summary_text
    committed_offset = db.summary.pending_offset
    committed_hash = db.summary.pending_content_hash
    committed_version = db.summary.version
    original_content = message.content

    with pytest.raises(RuntimeError, match="offline"):
        manager.maybe_compress_history(1, db, threshold=20)
    assert db.summary.failure_status == "retry_wait"
    assert db.summary.failure_code == "model_or_summary_error"
    assert db.summary.failure_count == 1
    first_retry_at = db.summary.next_retry_at
    assert first_retry_at == now[0] + timedelta(
        seconds=SUMMARY_RETRY_BASE_SECONDS
    )
    assert db.summary.pending_offset == committed_offset
    assert db.summary.pending_content_hash == committed_hash
    assert db.summary.pending_summary_text == committed_partial
    assert db.summary.covers_up_to_message_id == 0
    assert db.summary.version == committed_version + 1
    assert message.content == original_content

    request_count = len(llm.requests)
    manager.maybe_compress_history(1, db, threshold=20)
    assert len(llm.requests) == request_count

    while db.summary.failure_count < MAX_SUMMARY_FAILURES:
        now[0] = db.summary.next_retry_at + timedelta(seconds=1)
        with pytest.raises(RuntimeError, match="offline"):
            manager.maybe_compress_history(1, db, threshold=20)
        if db.summary.failure_status == "retry_wait":
            expected_delay = min(
                SUMMARY_RETRY_BASE_SECONDS
                * 2 ** (db.summary.failure_count - 1),
                SUMMARY_RETRY_MAX_SECONDS,
            )
            assert db.summary.next_retry_at == now[0] + timedelta(
                seconds=expected_delay
            )
        else:
            assert db.summary.failure_status == "failed"
            assert db.summary.next_retry_at is None
    assert db.summary.failure_status == "failed"
    assert db.summary.failure_count == MAX_SUMMARY_FAILURES
    request_count = len(llm.requests)
    manager.maybe_compress_history(1, db, threshold=20)
    assert len(llm.requests) == request_count

    llm.error = None
    failed_version = db.summary.version
    assert manager.recover_failed_compression(1, db) is True
    assert db.summary.failure_status is None
    assert db.summary.failure_count == 0
    assert db.summary.version == failed_version + 1
    assert db.summary.pending_offset == committed_offset
    assert db.summary.pending_content_hash == committed_hash
    assert db.summary.pending_summary_text == committed_partial

    manager.maybe_compress_history(1, db, threshold=20)
    assert len(llm.requests) == request_count + 1
    assert db.summary.failure_status is None
    assert db.summary.failure_count == 0
    assert db.summary.pending_offset == committed_offset
    assert db.summary.pending_summary_text == "新的对话摘要"
    assert db.summary.covers_up_to_message_id == 0
    assert message.content == original_content


def test_oversized_output_repair_model_failure_is_retryable(monkeypatch):
    llm = FakeLLM(scripted=["z" * 1_400, RuntimeError("offline"), "compact"])
    manager, db, message = _pending_chunk(offset=0)
    manager.max_context_chars = 1_600
    manager.llm_client = llm
    original_content = message.content
    committed_partial = db.summary.pending_summary_text

    now = [datetime(2026, 1, 1)]
    monkeypatch.setattr(MemoryManager, "_now_utc", staticmethod(lambda: now[0]))
    with pytest.raises(RuntimeError, match="offline"):
        manager.maybe_compress_history(1, db, threshold=20)

    assert len(llm.requests) == 2
    assert db.summary.pending_offset == 0
    assert db.summary.pending_summary_text == committed_partial
    assert db.summary.covers_up_to_message_id == 0
    assert db.summary.failure_status == "retry_wait"
    assert db.summary.failure_code == "model_or_summary_error"
    assert db.summary.failure_count == 1
    retry_at = db.summary.next_retry_at
    assert message.content == original_content

    now[0] = retry_at + timedelta(seconds=1)
    manager.maybe_compress_history(1, db, threshold=20)
    assert len(llm.requests) == 3
    assert db.summary.failure_status is None
    assert db.summary.failure_count == 0
    assert db.summary.pending_offset > 0
    assert db.summary.covers_up_to_message_id == 0
    assert message.content == original_content


def test_chunk_output_cas_conflict_does_not_overwrite_new_state(monkeypatch):
    llm = FakeLLM(scripted=["safe"])
    manager, db, _ = _pending_chunk(offset=0)
    manager.llm_client = llm
    original_cas = MemoryManager._cas_update

    def conflict(database, Summary, user_id, record, values):
        db.summary.pending_offset = 321
        db.summary.pending_summary_text = "concurrent value"
        db.summary.version += 1
        return False

    monkeypatch.setattr(MemoryManager, "_cas_update", staticmethod(conflict))
    manager.maybe_compress_history(1, db, threshold=20)
    assert db.summary.pending_offset == 321
    assert db.summary.pending_summary_text == "concurrent value"
    assert db.summary.covers_up_to_message_id == 0
    monkeypatch.setattr(MemoryManager, "_cas_update", staticmethod(original_cas))


def test_failure_limit_stops_automatic_retries_until_operator_recovery(monkeypatch):
    from app.memory.manager import MAX_SUMMARY_FAILURES

    messages = [
        FakeMessage(index, "user", "x" * 5_000 if index == 1 else f"m{index}")
        for index in range(1, 21)
    ]
    llm = FakeLLM(error=RuntimeError("offline"))
    manager = create_manager(llm, max_context_chars=1_000)
    db = FakeDB(messages)
    now = [datetime(2026, 1, 1)]
    monkeypatch.setattr(MemoryManager, "_now_utc", staticmethod(lambda: now[0]))

    with pytest.raises(RuntimeError):
        manager.maybe_compress_history(1, db, threshold=20)
    for _ in range(1, MAX_SUMMARY_FAILURES):
        now[0] = db.summary.next_retry_at + timedelta(seconds=1)
        with pytest.raises(RuntimeError):
            manager.maybe_compress_history(1, db, threshold=20)

    assert db.summary.failure_count == MAX_SUMMARY_FAILURES
    assert db.summary.failure_status == "failed"
    request_count = len(llm.requests)
    manager.maybe_compress_history(1, db, threshold=20)
    assert len(llm.requests) == request_count
    assert len(db.messages) == 20

    llm.error = None
    assert manager.recover_failed_compression(1, db) is True
    assert db.summary.pending_offset == 0
    assert db.summary.pending_message_id == 1
    manager.maybe_compress_history(1, db, threshold=20)
    assert len(llm.requests) == request_count + 1


def test_changed_message_content_invalidates_saved_chunk_offset():
    messages = [
        FakeMessage(index, "user", "x" * 5_000 if index == 1 else f"m{index}")
        for index in range(1, 21)
    ]
    llm = FakeLLM()
    manager = create_manager(llm, max_context_chars=1_000)
    db = FakeDB(messages)
    manager.maybe_compress_history(1, db, threshold=20)
    assert db.summary.pending_offset > 0
    db.messages[0].content = "changed original message"
    request_count = len(llm.requests)

    manager.maybe_compress_history(1, db, threshold=20)

    assert len(llm.requests) == request_count
    assert db.summary.pending_offset == 0
    assert db.summary.failure_status == "failed"
    assert db.summary.failure_code == "content_changed"
    assert db.summary.covers_up_to_message_id == 0
    assert db.messages[0].content == "changed original message"


def test_chunk_commit_failure_keeps_committed_offset_and_cursor():
    messages = [
        FakeMessage(index, "user", "x" * 5_000 if index == 1 else f"m{index}")
        for index in range(1, 21)
    ]
    llm = FakeLLM()
    manager = create_manager(llm, max_context_chars=1_000)
    db = FakeDB(messages)
    manager.maybe_compress_history(1, db, threshold=20)
    old_offset = db.summary.pending_offset
    db.commit_error = RuntimeError("commit failed")

    with pytest.raises(RuntimeError, match="commit failed"):
        manager.maybe_compress_history(1, db, threshold=20)

    assert db.summary.pending_offset == old_offset
    assert db.summary.covers_up_to_message_id == 0
    assert len(db.messages) == 20


def test_stale_concurrent_summary_update_is_rejected_by_database_cas():
    from sqlalchemy import Column, Integer, String, Text, create_engine
    from sqlalchemy.orm import declarative_base, sessionmaker
    from sqlalchemy.pool import StaticPool

    Base = declarative_base()

    class SummaryRow(Base):
        __tablename__ = "summary_cas_test"
        user_id = Column(String(20), primary_key=True)
        summary_text = Column(Text)
        covers_up_to_message_id = Column(Integer, nullable=False)
        version = Column(Integer, nullable=False)

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session.begin() as session:
        session.add(SummaryRow(
            user_id="1", summary_text="initial",
            covers_up_to_message_id=5, version=0,
        ))

    with Session() as session:
        stale_record = session.query(SummaryRow).filter_by(user_id="1").first()
        session.expunge(stale_record)

    with Session() as first:
        first_record = first.query(SummaryRow).filter_by(user_id="1").first()
        assert MemoryManager._cas_update(
            first,
            SummaryRow,
            "1",
            first_record,
            {
                SummaryRow.summary_text: "newer",
                SummaryRow.covers_up_to_message_id: 8,
            },
        ) is True
    with Session() as second:
        assert MemoryManager._cas_update(
            second,
            SummaryRow,
            "1",
            stale_record,
            {
                SummaryRow.summary_text: "stale overwrite",
                SummaryRow.covers_up_to_message_id: 6,
            },
        ) is False
    with Session() as session:
        current = session.query(SummaryRow).filter_by(user_id="1").first()
        assert current.summary_text == "newer"
        assert current.covers_up_to_message_id == 8
        assert current.version == 1
    engine.dispose()


def test_file_sqlite_pending_chunk_resumes_in_new_session_and_manager(
    memory_sqlite_path,
):
    database_path = Path(memory_sqlite_path).resolve()
    assert database_path.parent.parent == Path(__file__).resolve().parents[1]
    assert not database_path.exists()
    engine, Session, MessageRow, SummaryRow = _file_sqlite_memory_models(
        database_path
    )
    content = "persist-this-chunk-" * 260
    first_llm = FakeLLM(summary_text="persisted intermediate summary")

    try:
        with Session() as session:
            session.add(MessageRow(
                id=1, user_id="1", role="user", content=content,
            ))
            session.add(SummaryRow(
                user_id="1", summary_text="legacy summary",
                covers_up_to_message_id=0, pending_offset=0,
                failure_count=0, version=0,
            ))
            session.commit()

        with Session() as first_session:
            first_manager = MemoryManager(
                FakeRedis(), first_llm, MessageRow, SummaryRow,
                max_context_chars=1_000,
            )
            first_manager.maybe_compress_history(
                "1", first_session, threshold=20
            )

        with Session() as read_session:
            committed = read_session.query(SummaryRow).filter_by(
                user_id="1"
            ).one()
            assert committed.pending_message_id == 1
            assert 0 < committed.pending_offset < len(content)
            first_offset = committed.pending_offset
            first_partial = committed.pending_summary_text
            first_hash = committed.pending_content_hash
            first_version = committed.version
            assert first_partial
            assert first_hash == MemoryManager._content_hash(content)
            assert committed.covers_up_to_message_id == 0
            assert first_version > 0
            assert read_session.query(MessageRow).filter_by(id=1).one().content == content

        # End all first-engine sessions and dispose its pool before reopening
        # the same file through a newly-created engine and Session factory.
        engine.dispose()
        engine, Session, MessageRow, SummaryRow = _file_sqlite_memory_models(
            database_path
        )
        resumed_llm = FakeLLM(summary_text="completed persisted summary")
        with Session() as resumed_session:
            restored = resumed_session.query(SummaryRow).filter_by(
                user_id="1"
            ).one()
            assert restored.pending_offset == first_offset
            assert restored.pending_summary_text == first_partial
            assert restored.pending_content_hash == first_hash
            assert restored.version == first_version
            resumed_manager = MemoryManager(
                FakeRedis(), resumed_llm, MessageRow, SummaryRow,
                max_context_chars=1_000,
            )
            resumed_manager.maybe_compress_history(
                "1", resumed_session, threshold=20
            )

        first_resume_request = resumed_llm.requests[0]
        first_resume_user = first_resume_request["messages"][1]["content"]
        assert f"characters {first_offset}-" in first_resume_user
        assert f"Previous summary:\n{first_partial}" in first_resume_user
        assert serialized_request_size(
            first_resume_request["model"], first_resume_request["messages"]
        ) <= 1_000

        for _ in range(100):
            with Session() as session:
                current = session.query(SummaryRow).filter_by(
                    user_id="1"
                ).one()
                if current.pending_message_id is None:
                    break
                old_offset = current.pending_offset
                manager = MemoryManager(
                    FakeRedis(), resumed_llm, MessageRow, SummaryRow,
                    max_context_chars=1_000,
                )
                manager.maybe_compress_history("1", session, threshold=20)
            with Session() as verify_session:
                after = verify_session.query(SummaryRow).filter_by(
                    user_id="1"
                ).one()
                if after.pending_message_id is not None:
                    assert after.pending_offset > old_offset
                    assert after.pending_summary_text
                    assert after.pending_content_hash == first_hash
                    assert after.covers_up_to_message_id == 0
        else:
            pytest.fail("file-backed chunk resume did not complete")

        with Session() as final_session:
            final = final_session.query(SummaryRow).filter_by(
                user_id="1"
            ).one()
            stored_message = final_session.query(MessageRow).filter_by(
                id=1
            ).one()
            assert final.summary_text == "completed persisted summary"
            assert final.covers_up_to_message_id == 1
            assert final.pending_message_id is None
            assert final.pending_content_hash is None
            assert final.pending_offset == 0
            assert final.pending_summary_text is None
            assert final.failure_status is None
            assert final.version > first_version
            assert stored_message.content == content
        assert all(
            serialized_request_size(request["model"], request["messages"])
            <= 1_000
            for request in [*first_llm.requests, *resumed_llm.requests]
        )
    finally:
        _cleanup_file_sqlite(engine, database_path)


def test_chunk_input_slices_are_complete_ordered_and_committed_with_progress(
    monkeypatch,
):
    class CommitRecordingDB(FakeDB):
        def __init__(self, messages):
            super().__init__(messages)
            self.commit_snapshots = []

        def commit(self):
            super().commit()
            summary = self.summary
            self.commit_snapshots.append({
                "pending_message_id": summary.pending_message_id,
                "pending_content_hash": summary.pending_content_hash,
                "pending_offset": summary.pending_offset,
                "pending_summary_text": summary.pending_summary_text,
                "covers_up_to": summary.covers_up_to_message_id,
                "version": summary.version,
                "failure_status": summary.failure_status,
            })

    content = ('汉字😀\n引号" 路径\\分隔符 ' * 260) + "末尾"
    original_content = content
    message = FakeMessage(1, "user", content)
    llm = FakeLLM(
        scripted=["first committed partial", RuntimeError("mid-chunk failure")],
        summary_text="subsequent committed partial",
    )
    db = CommitRecordingDB([message])
    manager = create_manager(llm, max_context_chars=1_000)
    manager.redis_client.data["summary:1"] = "cache survives failed request"
    successful_request_indexes = []
    failed_request = None
    failure_retry_at = None

    for _ in range(100):
        before = db.summary
        before_offset = before.pending_offset if before else 0
        before_partial = (
            before.pending_summary_text
            if before and before.pending_offset > 0
            else (before.summary_text if before else None)
        )
        before_cursor = before.covers_up_to_message_id if before else 0
        request_index = len(llm.requests)
        try:
            manager.maybe_compress_history(1, db, threshold=20)
        except RuntimeError as exc:
            assert str(exc) == "mid-chunk failure"
            failed_request = llm.requests[request_index]
            assert db.summary.pending_offset == before_offset
            assert db.summary.pending_summary_text == before_partial
            assert db.summary.covers_up_to_message_id == before_cursor == 0
            assert db.summary.failure_status == "retry_wait"
            assert manager.redis_client.data["summary:1"] == (
                "cache survives failed request"
            )
            failure_retry_at = db.summary.next_retry_at
            monkeypatch.setattr(
                MemoryManager,
                "_now_utc",
                staticmethod(lambda: failure_retry_at + timedelta(seconds=1)),
            )
        else:
            successful_request_indexes.append(request_index)
            assert db.summary.pending_message_id is None or (
                db.summary.pending_offset >= before_offset
            )
            if db.summary.pending_message_id is not None:
                assert db.summary.pending_offset > before_offset
                assert db.summary.pending_summary_text
                assert db.summary.covers_up_to_message_id == 0
        if db.summary and db.summary.pending_message_id is None:
            break
    else:
        pytest.fail("recorded chunk sequence did not finish")

    assert failed_request is not None
    assert failure_retry_at is not None
    assert db.summary.summary_text == "subsequent committed partial"
    assert db.summary.covers_up_to_message_id == 1
    assert db.summary.pending_message_id is None
    assert message.content == original_content
    assert db.messages[0].content == original_content

    for request in llm.requests:
        assert serialized_request_size(
            request["model"], request["messages"]
        ) <= 1_000

    slice_pattern = re.compile(
        r"Message 1 \(user\), characters (\d+)-(\d+):\n"
    )
    successful_slices = []
    for index in successful_request_indexes:
        request = llm.requests[index]
        user_prompt = request["messages"][1]["content"]
        match = slice_pattern.search(user_prompt)
        assert match is not None
        start, end = map(int, match.groups())
        piece = user_prompt[match.end():]
        assert piece == original_content[start:end]
        successful_slices.append((start, end, piece))

    next_offset = 0
    for start, end, piece in successful_slices:
        assert start == next_offset
        assert end > start
        next_offset = end
    assert next_offset == len(original_content)
    assert "".join(piece for _, _, piece in successful_slices) == original_content

    failed_match = slice_pattern.search(
        failed_request["messages"][1]["content"]
    )
    assert failed_match is not None
    failed_start, failed_end = map(int, failed_match.groups())
    retry_slice = next(
        chunk for chunk in successful_slices if chunk[0] == failed_start
    )
    assert failed_end == retry_slice[1]
    assert failed_start == db.commit_snapshots[1]["pending_offset"]

    committed_progress = [
        snapshot for snapshot in db.commit_snapshots
        if snapshot["pending_message_id"] == 1
        and snapshot["pending_offset"] > 0
    ]
    assert committed_progress
    assert all(
        snapshot["pending_content_hash"] == manager._content_hash(original_content)
        and snapshot["pending_summary_text"]
        and snapshot["covers_up_to"] == 0
        for snapshot in committed_progress
    )
    assert all(
        right["version"] > left["version"]
        for left, right in zip(db.commit_snapshots, db.commit_snapshots[1:])
    )


def test_threaded_sqlite_cas_interleaving_rejects_stale_chunk_result(
    memory_sqlite_path,
):
    from threading import Barrier, Event, Thread

    database_path = Path(memory_sqlite_path).resolve()
    assert database_path.parent.parent == Path(__file__).resolve().parents[1]
    assert not database_path.exists()
    engine, Session, MessageRow, SummaryRow = _file_sqlite_memory_models(
        database_path
    )
    content = "parallel-cas-message-" * 300
    content_hash = MemoryManager._content_hash(content)
    manager_a = MemoryManager(
        FakeRedis(), FakeLLM(scripted=["result from A"]),
        MessageRow, SummaryRow, max_context_chars=1_000,
    )
    manager_b = MemoryManager(
        manager_a.redis_client, FakeLLM(scripted=["result from B"]),
        MessageRow, SummaryRow, max_context_chars=1_000,
    )
    manager_a.redis_client.data["summary:1"] = "cached authoritative summary"
    both_read = Barrier(2)
    both_generated = Barrier(2)
    a_committed = Event()
    outcomes = {}
    errors = []

    try:
        with Session() as session:
            session.add(MessageRow(
                id=1, user_id="1", role="user", content=content,
            ))
            session.add(SummaryRow(
                user_id="1", summary_text="existing summary",
                covers_up_to_message_id=0, pending_message_id=1,
                pending_content_hash=content_hash, pending_offset=100,
                pending_summary_text="base partial", failure_count=0,
                version=10,
            ))
            session.commit()

        def task(label, manager):
            try:
                # Both tasks read through distinct, short-lived Sessions. The
                # detached snapshots intentionally retain the same version.
                with Session() as session:
                    row = session.query(SummaryRow).filter_by(
                        user_id="1"
                    ).one()
                    snapshot = SimpleNamespace(version=row.version)
                    read_state = (
                        row.version, row.pending_offset,
                        row.pending_summary_text,
                        row.covers_up_to_message_id,
                    )
                outcomes[f"{label}_read"] = read_state
                both_read.wait(timeout=10)

                request = manager._chunk_summary_request(
                    "base partial", "user", 1, content[100:101], 100, 101,
                    target_chars=40,
                )
                response = manager.llm_client.chat.completions.create(
                    model="deepseek-chat", messages=request,
                )
                result = response.choices[0].message.content
                assert serialized_request_size(
                    "deepseek-chat", request
                ) <= manager.max_context_chars
                outcomes[f"{label}_generated"] = result
                both_generated.wait(timeout=10)

                if label == "B":
                    assert a_committed.wait(timeout=10)
                with Session() as session:
                    values = {
                        SummaryRow.pending_offset: 200 if label == "A" else 300,
                        SummaryRow.pending_summary_text: result,
                        SummaryRow.failure_count: 0,
                        SummaryRow.next_retry_at: None,
                        SummaryRow.failure_status: None,
                        SummaryRow.failure_code: None,
                    }
                    updated = MemoryManager._cas_update(
                        session, SummaryRow, "1", snapshot, values
                    )
                outcomes[f"{label}_cas"] = updated
                if label == "A":
                    a_committed.set()
                else:
                    # CAS conflict is a safe exit; B re-reads the database's
                    # authoritative state instead of reusing its old snapshot.
                    with Session() as session:
                        authoritative = session.query(SummaryRow).filter_by(
                            user_id="1"
                        ).one()
                        outcomes["B_authoritative"] = (
                            authoritative.version,
                            authoritative.pending_offset,
                            authoritative.pending_summary_text,
                            authoritative.covers_up_to_message_id,
                            authoritative.pending_message_id,
                        )
            except BaseException as exc:
                errors.append(exc)
                a_committed.set()
                try:
                    both_read.abort()
                    both_generated.abort()
                except Exception:
                    pass

        task_a = Thread(target=task, args=("A", manager_a), daemon=True)
        task_b = Thread(target=task, args=("B", manager_b), daemon=True)
        task_a.start()
        task_b.start()
        task_a.join(timeout=20)
        task_b.join(timeout=20)
        assert not task_a.is_alive()
        assert not task_b.is_alive()
        assert not errors

        assert outcomes["A_read"] == outcomes["B_read"] == (
            10, 100, "base partial", 0
        )
        assert outcomes["A_generated"] == "result from A"
        assert outcomes["B_generated"] == "result from B"
        assert outcomes["A_cas"] is True
        assert outcomes["B_cas"] is False
        assert outcomes["B_authoritative"] == (
            11, 200, "result from A", 0, 1
        )
        assert manager_a.redis_client.data["summary:1"] == (
            "cached authoritative summary"
        )

        with Session() as session:
            final = session.query(SummaryRow).filter_by(user_id="1").one()
            stored_message = session.query(MessageRow).filter_by(id=1).one()
            assert final.version == 11
            assert final.pending_offset == 200
            assert final.pending_summary_text == "result from A"
            assert final.covers_up_to_message_id == 0
            assert final.pending_message_id == 1
            assert final.pending_content_hash == content_hash
            assert stored_message.content == content
    finally:
        _cleanup_file_sqlite(engine, database_path)


def test_summary_state_upgrade_is_idempotent_and_preserves_legacy_rows():
    from sqlalchemy import create_engine, inspect
    from scripts.upgrade_memory_summary_state import upgrade

    engine = create_engine("sqlite://")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE conversation_summaries ("
                "id INTEGER PRIMARY KEY, user_id VARCHAR(50) UNIQUE, "
                "summary_text TEXT, covers_up_to_message_id INTEGER)"
            )
            connection.exec_driver_sql(
                "INSERT INTO conversation_summaries "
                "(id, user_id, summary_text, covers_up_to_message_id) "
                "VALUES (1, 'legacy-user', 'existing summary', 42)"
            )

        upgrade(engine)
        upgrade(engine)

        columns = {
            column["name"]: column
            for column in inspect(engine).get_columns("conversation_summaries")
        }
        assert {
            "pending_message_id", "pending_content_hash", "pending_offset",
            "pending_summary_text", "failure_count", "next_retry_at",
            "failure_status", "failure_code", "version",
        } <= set(columns)
        assert columns["pending_content_hash"]["type"].length == 64
        assert columns["failure_status"]["type"].length == 20
        assert columns["failure_code"]["type"].length == 64
        for name in ("pending_offset", "failure_count", "version"):
            assert columns[name]["nullable"] is False
            assert columns[name]["default"] in ("0", "(0)")
        with engine.connect() as connection:
            row = connection.exec_driver_sql(
                "SELECT user_id, summary_text, covers_up_to_message_id, "
                "pending_offset, failure_count, version "
                "FROM conversation_summaries WHERE id = 1"
            ).one()
        assert tuple(row) == (
            "legacy-user", "existing summary", 42, 0, 0, 0
        )
    finally:
        engine.dispose()


def _create_partial_summary_schema(extra_columns):
    from sqlalchemy import create_engine

    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE conversation_summaries ("
            "id INTEGER PRIMARY KEY, user_id VARCHAR(50) UNIQUE, "
            "summary_text TEXT, covers_up_to_message_id INTEGER)"
        )
        connection.exec_driver_sql(
            "INSERT INTO conversation_summaries "
            "(id, user_id, summary_text, covers_up_to_message_id) "
            "VALUES (1, 'legacy-user', 'kept summary', 42)"
        )
        for declaration in extra_columns:
            connection.exec_driver_sql(
                "ALTER TABLE conversation_summaries ADD COLUMN "
                + declaration
            )
    return engine


def test_summary_upgrade_rejects_nullable_existing_version_without_changes():
    from sqlalchemy import inspect
    from scripts.upgrade_memory_summary_state import upgrade

    engine = _create_partial_summary_schema(("version INTEGER NULL",))
    try:
        with pytest.raises(RuntimeError, match="column version.*nullable"):
            upgrade(engine)
        columns = {
            column["name"]
            for column in inspect(engine).get_columns("conversation_summaries")
        }
        assert "pending_message_id" not in columns
        assert "version" in columns
        with engine.connect() as connection:
            row = connection.exec_driver_sql(
                "SELECT summary_text, covers_up_to_message_id "
                "FROM conversation_summaries WHERE id=1"
            ).one()
        assert tuple(row) == ("kept summary", 42)
    finally:
        engine.dispose()


def test_summary_upgrade_rejects_hash_column_too_short():
    from scripts.upgrade_memory_summary_state import upgrade

    engine = _create_partial_summary_schema((
        "pending_content_hash VARCHAR(32) NULL",
    ))
    try:
        with pytest.raises(RuntimeError, match="pending_content_hash.*length>=64"):
            upgrade(engine)
    finally:
        engine.dispose()


def test_summary_upgrade_rejects_incompatible_state_default():
    from scripts.upgrade_memory_summary_state import upgrade

    engine = _create_partial_summary_schema((
        "pending_offset INTEGER NOT NULL DEFAULT 7",
    ))
    try:
        with pytest.raises(RuntimeError, match="pending_offset.*default 0"):
            upgrade(engine)
    finally:
        engine.dispose()


def test_summary_upgrade_accepts_orm_client_default_without_server_default():
    from sqlalchemy import create_engine
    from scripts.upgrade_memory_summary_state import upgrade

    # ConversationSummary.version has SQLAlchemy default=0 (client-side),
    # not server_default=0. Existing compatible rows need not have a DB
    # default; the migration adds DEFAULT 0 only for columns it creates.
    engine = create_engine("sqlite://")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE conversation_summaries ("
                "id INTEGER PRIMARY KEY, user_id VARCHAR(50) UNIQUE, "
                "summary_text TEXT, covers_up_to_message_id INTEGER, "
                "version INTEGER NOT NULL)"
            )
            connection.exec_driver_sql(
                "INSERT INTO conversation_summaries "
                "(id, user_id, summary_text, covers_up_to_message_id, version) "
                "VALUES (1, 'legacy-user', 'kept summary', 42, 0)"
            )
        upgrade(engine)
    finally:
        engine.dispose()


def test_summary_upgrade_continues_compatible_partial_schema():
    from sqlalchemy import inspect
    from scripts.upgrade_memory_summary_state import upgrade

    engine = _create_partial_summary_schema((
        "version INT NOT NULL DEFAULT ((0))",
        "pending_offset INTEGER NOT NULL DEFAULT '0'",
        "pending_content_hash VARCHAR(80) NULL",
    ))
    try:
        upgrade(engine)
        columns = {
            column["name"]: column
            for column in inspect(engine).get_columns("conversation_summaries")
        }
        assert columns["pending_content_hash"]["type"].length == 80
        assert "failure_count" in columns
        with engine.connect() as connection:
            row = connection.exec_driver_sql(
                "SELECT summary_text, covers_up_to_message_id, version, "
                "pending_offset FROM conversation_summaries WHERE id=1"
            ).one()
        assert tuple(row) == ("kept summary", 42, 0, 0)
    finally:
        engine.dispose()


def test_summary_upgrade_rejects_bad_partial_schema_before_adding_columns():
    from sqlalchemy import inspect
    from scripts.upgrade_memory_summary_state import upgrade

    engine = _create_partial_summary_schema((
        "failure_count INTEGER NOT NULL DEFAULT 4",
    ))
    try:
        with pytest.raises(RuntimeError, match="failure_count.*default 0"):
            upgrade(engine)
        columns = {
            column["name"]
            for column in inspect(engine).get_columns("conversation_summaries")
        }
        assert "failure_count" in columns
        assert "pending_message_id" not in columns
        with engine.connect() as connection:
            row = connection.exec_driver_sql(
                "SELECT summary_text, covers_up_to_message_id, failure_count "
                "FROM conversation_summaries WHERE id=1"
            ).one()
        assert tuple(row) == ("kept summary", 42, 4)
    finally:
        engine.dispose()


def test_summary_upgrade_accepts_sqlite_type_and_default_aliases():
    from scripts.upgrade_memory_summary_state import upgrade

    engine = _create_partial_summary_schema((
        "version INT NOT NULL DEFAULT ((0))",
        "pending_offset INT NOT NULL DEFAULT (0)",
        "failure_count INTEGER NOT NULL DEFAULT '0'",
        "pending_content_hash VARCHAR(64) NULL",
    ))
    try:
        upgrade(engine)
        upgrade(engine)
    finally:
        engine.dispose()


def test_summary_state_upgrade_fails_closed_for_missing_table():
    from sqlalchemy import create_engine
    from scripts.upgrade_memory_summary_state import upgrade

    engine = create_engine("sqlite://")
    try:
        with pytest.raises(RuntimeError, match="required table is missing"):
            upgrade(engine)
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    ("database_value", "expected_message"),
    [
        (None, "MYSQL_DATABASE is missing"),
        ("   ", "MYSQL_DATABASE is empty"),
    ],
)
def test_summary_upgrade_requires_explicit_database_before_engine_creation(
    monkeypatch, database_value, expected_message
):
    from scripts import upgrade_memory_summary_state as migration

    monkeypatch.setenv("MYSQL_PASSWORD", "synthetic-test-password")
    monkeypatch.setenv("MYSQL_HOST", "synthetic-test-host")
    if database_value is None:
        monkeypatch.delenv("MYSQL_DATABASE", raising=False)
    else:
        monkeypatch.setenv("MYSQL_DATABASE", database_value)
    engine_calls = []
    monkeypatch.setattr(
        migration,
        "create_engine",
        lambda *args, **kwargs: engine_calls.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match=expected_message):
        migration._engine_from_environment()

    assert engine_calls == []


@pytest.mark.parametrize(
    ("host_value", "expected_message"),
    [
        (None, "MYSQL_HOST is missing"),
        ("  ", "MYSQL_HOST is empty"),
    ],
)
def test_summary_upgrade_requires_explicit_host_before_engine_creation(
    monkeypatch, host_value, expected_message
):
    from scripts import upgrade_memory_summary_state as migration

    monkeypatch.setenv("MYSQL_PASSWORD", "synthetic-test-password")
    monkeypatch.setenv("MYSQL_DATABASE", "synthetic_test_database")
    if host_value is None:
        monkeypatch.delenv("MYSQL_HOST", raising=False)
    else:
        monkeypatch.setenv("MYSQL_HOST", host_value)
    engine_calls = []
    monkeypatch.setattr(
        migration,
        "create_engine",
        lambda *args, **kwargs: engine_calls.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match=expected_message):
        migration._engine_from_environment()

    assert engine_calls == []


def test_summary_upgrade_uses_explicit_target_without_connecting(monkeypatch):
    from scripts import upgrade_memory_summary_state as migration

    monkeypatch.setenv("MYSQL_PASSWORD", "synthetic-test-password")
    monkeypatch.setenv("MYSQL_USER", "synthetic-test-user")
    monkeypatch.setenv("MYSQL_HOST", "  synthetic-test-host  ")
    monkeypatch.setenv("MYSQL_PORT", "3306")
    monkeypatch.setenv("MYSQL_DATABASE", "  synthetic_test_database  ")
    captured = {}
    sentinel = object()

    def fake_create_engine(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(migration, "create_engine", fake_create_engine)

    engine = migration._engine_from_environment()

    assert engine is sentinel
    assert "@synthetic-test-host:3306/synthetic_test_database" in captured["url"]
    assert captured["kwargs"] == {"pool_pre_ping": True}


@pytest.mark.parametrize(
    ("variable", "value", "expected_message"),
    [
        ("MYSQL_HOST", None, "MYSQL_HOST is missing"),
        ("MYSQL_HOST", "   ", "MYSQL_HOST is empty"),
        ("MYSQL_DATABASE", None, "MYSQL_DATABASE is missing"),
        ("MYSQL_DATABASE", "   ", "MYSQL_DATABASE is empty"),
    ],
)
def test_summary_upgrade_cli_reports_target_error_before_engine_or_ddl(
    monkeypatch, capsys, variable, value, expected_message
):
    from scripts import upgrade_memory_summary_state as migration

    monkeypatch.setenv("MYSQL_PASSWORD", "synthetic-test-password")
    monkeypatch.setenv("MYSQL_HOST", "synthetic-test-host")
    monkeypatch.setenv("MYSQL_DATABASE", "synthetic_test_database")
    if value is None:
        monkeypatch.delenv(variable, raising=False)
    else:
        monkeypatch.setenv(variable, value)

    engine_calls = []
    upgrade_calls = []
    monkeypatch.setattr(
        migration,
        "create_engine",
        lambda *args, **kwargs: engine_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        migration,
        "upgrade",
        lambda engine: upgrade_calls.append(engine),
    )

    result = migration.main()
    captured = capsys.readouterr()

    assert result == 1
    assert expected_message in captured.err
    assert "synthetic-test-password" not in captured.out + captured.err
    assert engine_calls == []
    assert upgrade_calls == []


def test_summary_model_failure_preserves_cursor_history_and_cache():
    previous_summary = FakeSummary("committed summary", 5)
    messages = [
        FakeMessage(index, "user", f"message {index}")
        for index in range(6, 26)
    ]
    llm = FakeLLM(error=RuntimeError("offline model failure"))
    manager = create_manager(llm)
    db = FakeDB(messages, previous_summary)
    manager.redis_client.data["summary:1"] = "cached summary"

    with pytest.raises(RuntimeError, match="offline model failure"):
        manager.maybe_compress_history(1, db, threshold=20)

    assert db.summary.summary_text == "committed summary"
    assert db.summary.covers_up_to_message_id == 5
    assert len(db.messages) == 20
    assert manager.redis_client.data["summary:1"] == "cached summary"


def test_database_commit_failure_rolls_back_summary_and_keeps_cache():
    previous_summary = FakeSummary("committed summary", 5)
    messages = [
        FakeMessage(index, "user", f"message {index}")
        for index in range(6, 26)
    ]
    llm = FakeLLM()
    manager = create_manager(llm)
    db = FakeDB(
        messages,
        previous_summary,
        commit_error=RuntimeError("database commit failed"),
    )
    manager.redis_client.data["summary:1"] = "cached summary"

    with pytest.raises(RuntimeError, match="database commit failed"):
        manager.maybe_compress_history(1, db, threshold=20)

    assert db.rollback_count >= 1
    assert db.summary.summary_text == "committed summary"
    assert db.summary.covers_up_to_message_id == 5
    assert len(db.messages) == 20
    assert manager.redis_client.data["summary:1"] == "cached summary"


def test_new_summary_commit_failure_persists_bounded_retry_state():
    messages = [
        FakeMessage(index, "user", f"message {index}")
        for index in range(1, 21)
    ]
    llm = FakeLLM()
    manager = create_manager(llm)
    db = FakeDB(
        messages,
        commit_error=[RuntimeError("transient commit failure")],
    )

    with pytest.raises(RuntimeError, match="transient commit failure"):
        manager.maybe_compress_history(1, db, threshold=20)

    assert len(llm.requests) == 1
    assert db.summary is not None
    assert db.summary.summary_text is None
    assert db.summary.covers_up_to_message_id == 0
    assert db.summary.failure_status == "retry_wait"
    assert db.summary.failure_count == 1
    assert db.summary.failure_code == "summary_commit_error"


def test_chat_route_keeps_summary_compression_best_effort():
    main_path = Path(__file__).resolve().parents[1] / "main.py"
    module = ast.parse(main_path.read_text(encoding="utf-8-sig"))
    chat = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "chat"
    )
    compression_try = next(
        node for node in ast.walk(chat)
        if isinstance(node, ast.Try)
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "maybe_compress_history"
            for statement in node.body
            for child in ast.walk(statement)
        )
    )
    assert any(
        any(
            isinstance(child, ast.Name) and child.id == "Exception"
            for child in ast.walk(handler.type)
        )
        for handler in compression_try.handlers
    )
    assert any(
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "build_chat_response"
        and node.lineno > compression_try.end_lineno
        for node in chat.body
    )
