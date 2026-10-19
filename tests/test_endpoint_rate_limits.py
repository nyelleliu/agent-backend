"""Endpoint rate limiting: login bypass fix, /chat and /upload_doc.

Every test here drives the endpoint functions with stub
dependencies. No user row, message row or document is written to
agent_db, and no vector store is touched — a rejected request is
proven to be rejected *before* any database work happens.
"""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import main


class _NoUserQuery:
    """Query stub that always answers "no such user"."""

    def filter(self, *conditions):
        return self

    def first(self):
        return None


class _StubDB:
    """Database stub: performs no round trip and writes nothing."""

    def query(self, model):
        return _NoUserQuery()

    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        # No-op: objects stay on the stub, nothing reaches MySQL.
        pass

    def rollback(self):
        pass


def _request(host: str):
    return SimpleNamespace(client=SimpleNamespace(host=host))


def _login(username: str, host: str, db):
    data = main.UserLogin(username=username, password="not-a-real-pass")
    return main.login(data, _request(host), db)


def _unique_ip(marker: str) -> str:
    return f"203.0.113.{marker}"


# ---------------------------------------------------------------------------
# Login: switching usernames must not reset the budget
# ---------------------------------------------------------------------------


def test_login_budget_is_shared_across_usernames_from_one_ip():
    ip = _unique_ip("42")
    key = main.login_rate_key(ip)
    db = _StubDB()

    main.login_limiter.reset(key)

    try:
        for attempt in range(main.LOGIN_MAX_ATTEMPTS):
            with pytest.raises(HTTPException) as exc_info:
                _login(f"first-victim-{attempt}", ip, db)

            # Unknown user → invalid credentials, no DB write.
            assert exc_info.value.status_code == 401

        # Same IP, brand-new username: still throttled. Before the
        # fix the key was "login:{ip}:{username}", so this attempt
        # would have been allowed and enumeration unlimited.
        with pytest.raises(HTTPException) as exc_info:
            _login("another-username-entirely", ip, db)

        assert exc_info.value.status_code == 429
        assert exc_info.value.headers.get("Retry-After")
    finally:
        main.login_limiter.reset(key)


def test_login_budget_is_isolated_per_ip():
    first_ip = _unique_ip("43")
    second_ip = _unique_ip("44")
    db = _StubDB()

    main.login_limiter.reset(main.login_rate_key(first_ip))
    main.login_limiter.reset(main.login_rate_key(second_ip))

    try:
        for attempt in range(main.LOGIN_MAX_ATTEMPTS):
            with pytest.raises(HTTPException) as exc_info:
                _login(f"user-{attempt}", first_ip, db)
            assert exc_info.value.status_code == 401

        with pytest.raises(HTTPException) as exc_info:
            _login("user-0", first_ip, db)
        assert exc_info.value.status_code == 429

        # A different client IP keeps its own budget.
        with pytest.raises(HTTPException) as exc_info:
            _login("user-0", second_ip, db)
        assert exc_info.value.status_code == 401
    finally:
        main.login_limiter.reset(main.login_rate_key(first_ip))
        main.login_limiter.reset(main.login_rate_key(second_ip))


# ---------------------------------------------------------------------------
# Register: still per IP, and a throttled request writes nothing
# ---------------------------------------------------------------------------


def test_register_is_rate_limited_per_ip_and_writes_nothing_when_throttled():
    ip = _unique_ip("45")
    key = main.register_rate_key(ip)
    db = _StubDB()

    main.register_limiter.reset(key)

    try:
        for attempt in range(main.REGISTER_MAX_ATTEMPTS):
            data = main.UserRegister(
                username=f"reg-user-{attempt}-{ip.replace('.', '')}",
                password="longenough1",
            )
            response = main.register(data, _request(ip), db)

            assert response["message"] == "User registered successfully"

        # Objects were only staged on the stub — nothing committed.
        assert len(db.added) == main.REGISTER_MAX_ATTEMPTS

        with pytest.raises(HTTPException) as exc_info:
            main.register(
                main.UserRegister(
                    username="one-more-user",
                    password="longenough1",
                ),
                _request(ip),
                db,
            )

        assert exc_info.value.status_code == 429
        # Rejected before any object is created.
        assert len(db.added) == main.REGISTER_MAX_ATTEMPTS
    finally:
        main.register_limiter.reset(key)


# ---------------------------------------------------------------------------
# /chat: throttled before the LLM call and before any row is written
# ---------------------------------------------------------------------------


def test_chat_is_rate_limited_per_user_before_writing_history():
    user_id = 987654
    key = main.chat_rate_key(user_id)

    main.chat_limiter.reset(key)

    try:
        for _ in range(main.CHAT_MAX_ATTEMPTS):
            assert main.chat_limiter.hit(key)[0] is True

        with pytest.raises(HTTPException) as exc_info:
            main.chat(
                main.ChatMessage(message="hello"),
                current_user={"user_id": user_id, "role": "employee"},
                db_user=SimpleNamespace(id=user_id, role="employee"),
                # db is None on purpose: hitting it would raise, so a
                # passing test proves the rejection comes first.
                db=None,
            )

        assert exc_info.value.status_code == 429
        assert exc_info.value.headers.get("Retry-After")
    finally:
        main.chat_limiter.reset(key)


def test_chat_budget_is_isolated_per_user():
    first_user = 987656
    second_user = 987657

    main.chat_limiter.reset(main.chat_rate_key(first_user))
    main.chat_limiter.reset(main.chat_rate_key(second_user))

    try:
        for _ in range(main.CHAT_MAX_ATTEMPTS):
            assert main.chat_limiter.hit(main.chat_rate_key(first_user))[0]

        with pytest.raises(HTTPException) as exc_info:
            main.chat(
                main.ChatMessage(message="hello"),
                current_user={"user_id": first_user, "role": "employee"},
                db_user=SimpleNamespace(id=first_user, role="employee"),
                db=None,
            )
        assert exc_info.value.status_code == 429

        # Another user still has budget (hit() returning True means
        # the endpoint would proceed).
        assert main.chat_limiter.hit(main.chat_rate_key(second_user))[0]
    finally:
        main.chat_limiter.reset(main.chat_rate_key(first_user))
        main.chat_limiter.reset(main.chat_rate_key(second_user))


# ---------------------------------------------------------------------------
# /upload_doc: throttled before the permission check and any storage
# ---------------------------------------------------------------------------


def test_upload_doc_is_rate_limited_before_storing():
    user_id = 987658
    key = main.upload_rate_key(user_id)

    main.upload_limiter.reset(key)

    try:
        for _ in range(main.UPLOAD_MAX_ATTEMPTS):
            assert main.upload_limiter.hit(key)[0] is True

        with pytest.raises(HTTPException) as exc_info:
            main.upload_doc(
                main.DocumentUpload(
                    text="document body",
                    permission="employee",
                ),
                SimpleNamespace(id=user_id, role="employee"),
            )

        assert exc_info.value.status_code == 429
        assert exc_info.value.headers.get("Retry-After")
    finally:
        main.upload_limiter.reset(key)
