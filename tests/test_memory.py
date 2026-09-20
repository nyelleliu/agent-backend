from types import SimpleNamespace

from app.memory.manager import MemoryManager


class FakeColumn:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return ("eq", self.name, other)

    def __gt__(self, other):
        return ("gt", self.name, other)

    def asc(self):
        return ("asc", self.name)


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
        self.rows = sorted(
            self.rows,
            key=lambda row: row.id,
        )
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
            rows = [self.summary] if self.summary else []
            return FakeQuery(rows)

        if model is FakeMessage:
            return FakeQuery(self.messages)

        raise ValueError(f"Unknown model: {model}")

    def add(self, obj):
        self.summary = obj

    def commit(self):
        pass


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

    def __init__(
        self,
        summary_text,
        covers_up_to_message_id,
        user_id="1",
    ):
        self.user_id = user_id
        self.summary_text = summary_text
        self.covers_up_to_message_id = covers_up_to_message_id


class FakeLLM:
    def __init__(self):
        self.called = False

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.create
            )
        )

    def create(self, **kwargs):
        self.called = True

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="新的对话摘要"
                    )
                )
            ]
        )


def create_manager(llm):
    return MemoryManager(
        redis_client=FakeRedis(),
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
