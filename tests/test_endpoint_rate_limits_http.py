"""HTTP-level rate limiting on the real routes (P2).

tests/test_endpoint_rate_limits.py drives the endpoint *functions*
with stub dependencies. These tests go through the ASGI stack with
TestClient instead, so routing, dependency injection, the exception
handlers that build the 429 body and the ``Retry-After`` header are
all exercised for real.

Every test installs a fresh :class:`RateLimiter` with a small budget,
so no outcome depends on how much quota earlier tests happened to
leave behind. Nothing here writes a row to agent_db: the database is
replaced with a recording stub, and the agent / storage callables are
replaced with functions that fail the test if they are ever reached
from a throttled request.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main
from app.core.rate_limit import RateLimiter


class RecordingDB:
    """Stands in for the SQLAlchemy session: records work, writes nothing."""

    def __init__(self, users=()):
        self.users = list(users)
        self.queries = 0
        self.added = []
        self.commits = 0

    def query(self, model):
        self.queries += 1
        return self

    def filter(self, *conditions, **kwargs):
        return self

    def first(self):
        return self.users[0] if self.users else None

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass

    def close(self):
        pass


@pytest.fixture
def clean_app():
    yield
    main.app.dependency_overrides.clear()


def _client(ip, db=None, user=None, holder=None):
    """TestClient bound to a client IP, with DB/auth overridden."""
    if db is not None:
        main.app.dependency_overrides[main.get_db] = lambda: db

    if holder is not None:
        main.app.dependency_overrides[main.get_current_user] = lambda: {
            "user_id": holder["user"].id,
            "role": holder["user"].role,
        }
        main.app.dependency_overrides[main.get_current_db_user] = (
            lambda: holder["user"]
        )
    elif user is not None:
        main.app.dependency_overrides[main.get_current_user] = lambda: {
            "user_id": user.id,
            "role": user.role,
        }
        main.app.dependency_overrides[main.get_current_db_user] = lambda: user

    return TestClient(main.app, client=(ip, 40000))


def _login_json(username="nobody", password="wrong-pass-1"):
    return {"username": username, "password": password}


def _forbidden_pipeline(*args, **kwargs):
    raise AssertionError(
        "a throttled request must never reach the agent or storage"
    )


# ---------------------------------------------------------------------------
# /login
# ---------------------------------------------------------------------------


def test_login_returns_429_with_retry_after_and_skips_the_db(
    clean_app,
    monkeypatch,
):
    monkeypatch.setattr(
        main,
        "login_limiter",
        RateLimiter(max_attempts=2, window_seconds=300),
    )

    db = RecordingDB()
    client = _client("203.0.113.60", db=db)

    for _ in range(2):
        response = client.post("/login", json=_login_json())
        assert response.status_code == 401

    queries_before = db.queries

    response = client.post("/login", json=_login_json())

    assert response.status_code == 429
    assert "Too many login attempts" in response.text

    retry_after = int(response.headers["Retry-After"])
    assert retry_after >= 1

    # Rejected before any database lookup happened.
    assert db.queries == queries_before


def test_successful_login_resets_the_ip_budget(clean_app, monkeypatch):
    """Documents the current policy: a success clears the IP budget.

    The reset is deliberate (main.login), and this test pins it down so
    any future change to the login limiter is a conscious one.
    """
    monkeypatch.setattr(
        main,
        "login_limiter",
        RateLimiter(max_attempts=2, window_seconds=300),
    )

    password = "correct-pass-123"
    db = RecordingDB(
        users=[
            SimpleNamespace(
                id=7,
                role="employee",
                password_hash=main.hash_password(password),
            )
        ]
    )
    client = _client("203.0.113.61", db=db)

    assert client.post(
        "/login",
        json=_login_json("alice", "wrong-pass-1"),
    ).status_code == 401

    accepted = client.post(
        "/login",
        json=_login_json("alice", password),
    )
    assert accepted.status_code == 200
    assert "access_token" in accepted.json()

    # The budget was reset by the success, so these are ordinary
    # credential failures again — not a 429.
    assert client.post(
        "/login",
        json=_login_json("alice", "wrong-pass-1"),
    ).status_code == 401
    assert client.post(
        "/login",
        json=_login_json("alice", "wrong-pass-1"),
    ).status_code == 401

    # ... and the refilled budget is exactly max_attempts wide.
    assert client.post(
        "/login",
        json=_login_json("alice", "wrong-pass-1"),
    ).status_code == 429


def test_login_budget_is_isolated_per_client_ip(clean_app, monkeypatch):
    monkeypatch.setattr(
        main,
        "login_limiter",
        RateLimiter(max_attempts=1, window_seconds=300),
    )

    db = RecordingDB()
    first = _client("203.0.113.62", db=db)
    second = _client("203.0.113.63", db=db)

    assert first.post("/login", json=_login_json()).status_code == 401
    assert first.post("/login", json=_login_json()).status_code == 429

    # A different client IP keeps its own budget.
    assert second.post("/login", json=_login_json()).status_code == 401


# ---------------------------------------------------------------------------
# /register
# ---------------------------------------------------------------------------


def test_register_returns_429_and_writes_nothing_when_throttled(
    clean_app,
    monkeypatch,
):
    monkeypatch.setattr(
        main,
        "register_limiter",
        RateLimiter(max_attempts=1, window_seconds=3600),
    )

    db = RecordingDB()
    client = _client("203.0.113.64", db=db)

    accepted = client.post(
        "/register",
        json={"username": "newuser1", "password": "longenough1"},
    )
    assert accepted.status_code == 200
    assert len(db.added) == 1

    response = client.post(
        "/register",
        json={"username": "newuser2", "password": "longenough1"},
    )

    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) >= 1
    # Nothing was even staged for the throttled request.
    assert len(db.added) == 1


# ---------------------------------------------------------------------------
# /chat
# ---------------------------------------------------------------------------


def test_chat_returns_429_before_the_agent_or_the_database(
    clean_app,
    monkeypatch,
):
    monkeypatch.setattr(
        main,
        "chat_limiter",
        RateLimiter(max_attempts=1, window_seconds=60),
    )

    user = SimpleNamespace(id=987650, role="employee")
    db = RecordingDB()
    client = _client("203.0.113.65", db=db, user=user)

    monkeypatch.setattr(main.agent_loop, "run", _forbidden_pipeline)

    # Consume the one allowed slot directly so this test stays on the
    # HTTP contract: no LLM call, no agent turn, no history row.
    assert main.chat_limiter.hit(main.chat_rate_key(user.id))[0] is True

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 429
    assert "Too many chat requests" in response.text
    assert int(response.headers["Retry-After"]) >= 1
    assert db.added == []


def test_chat_budget_is_isolated_per_user(clean_app, monkeypatch):
    monkeypatch.setattr(
        main,
        "chat_limiter",
        RateLimiter(max_attempts=1, window_seconds=60),
    )

    first = SimpleNamespace(id=987652, role="employee")
    second = SimpleNamespace(id=987653, role="employee")
    holder = {"user": first}
    client = _client("203.0.113.66", holder=holder)

    monkeypatch.setattr(main.agent_loop, "run", _forbidden_pipeline)

    assert main.chat_limiter.hit(main.chat_rate_key(first.id))[0] is True

    throttled = client.post("/chat", json={"message": "hello"})
    assert throttled.status_code == 429

    # Switching the authenticated user switches the budget: the second
    # user still has a full allowance even though the first is
    # exhausted (max_attempts=1, so this one hit is the whole budget).
    holder["user"] = second
    assert main.chat_limiter.hit(main.chat_rate_key(second.id))[0] is True


# ---------------------------------------------------------------------------
# /upload_doc
# ---------------------------------------------------------------------------


def test_upload_doc_returns_429_before_storing(clean_app, monkeypatch):
    monkeypatch.setattr(
        main,
        "upload_limiter",
        RateLimiter(max_attempts=1, window_seconds=60),
    )

    user = SimpleNamespace(id=987654, role="employee")
    client = _client("203.0.113.67", user=user)

    monkeypatch.setattr(main, "add_document", _forbidden_pipeline)

    assert main.upload_limiter.hit(main.upload_rate_key(user.id))[0] is True

    response = client.post(
        "/upload_doc",
        json={"text": "document body", "permission": "employee"},
    )

    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) >= 1
