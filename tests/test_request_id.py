"""X-Request-ID validation (P1).

The client-supplied id ends up in every log line of the request and
in the response header, so it is only accepted when it matches a
strict whitelist; anything else is replaced with a server-generated
id. These tests go through the real middleware via TestClient.
"""

import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main
from app.core.logging_config import RequestIdFilter
from app.core.rate_limit import RateLimiter


class _EmptyDB:
    """Query stub: the username never exists, nothing is written."""

    def query(self, model):
        return self

    def filter(self, *conditions):
        return self

    def first(self):
        return None

    def close(self):
        pass


class _RequestIdRecorder(logging.Handler):
    """Captures the request id each log record was tagged with."""

    def __init__(self):
        super().__init__()
        self.ids = []
        self.addFilter(RequestIdFilter())

    def emit(self, record):
        self.ids.append(getattr(record, "request_id", None))


@pytest.fixture
def client(monkeypatch):
    """Client with a stubbed DB and a fresh, wide login budget.

    These tests are about the request id, not throttling; a generous
    limiter keeps every request on the 401 path regardless of how many
    logins earlier tests in the session performed.
    """
    monkeypatch.setattr(
        main,
        "login_limiter",
        RateLimiter(max_attempts=10_000, window_seconds=300),
    )
    main.app.dependency_overrides[main.get_db] = lambda: _EmptyDB()
    test_client = TestClient(main.app)

    yield test_client

    main.app.dependency_overrides.clear()


def _login(client, headers=None):
    return client.post(
        "/login",
        json={"username": "no-such-user", "password": "whatever-1"},
        headers=headers or {},
    )


def test_valid_request_id_is_echoed_verbatim(client):
    response = _login(client, {"X-Request-ID": "gw-abc_123"})

    assert response.status_code == 401
    assert response.headers["X-Request-ID"] == "gw-abc_123"


def test_request_id_at_the_length_limit_is_accepted(client):
    at_limit = "b" * main.REQUEST_ID_MAX_LENGTH

    response = _login(client, {"X-Request-ID": at_limit})

    assert response.headers["X-Request-ID"] == at_limit


def test_oversized_request_id_is_replaced(client):
    oversized = "c" * (main.REQUEST_ID_MAX_LENGTH + 1)

    response = _login(client, {"X-Request-ID": oversized})
    echoed = response.headers["X-Request-ID"]

    assert echoed != oversized
    assert main.REQUEST_ID_PATTERN.fullmatch(echoed)


@pytest.mark.parametrize(
    "offending",
    [
        "has space",
        "slash/inside",
        "colon:inside",
        "semi;colon",
        "star*inside",
        "paren(inside)",
        "",
    ],
)
def test_request_id_with_disallowed_characters_is_replaced(
    client,
    offending,
):
    response = _login(client, {"X-Request-ID": offending})
    echoed = response.headers["X-Request-ID"]

    assert echoed != offending
    assert main.REQUEST_ID_PATTERN.fullmatch(echoed)


def test_missing_request_id_gets_a_server_generated_one(client):
    response = _login(client)

    echoed = response.headers["X-Request-ID"]

    assert main.REQUEST_ID_PATTERN.fullmatch(echoed)


def test_request_id_for_rejects_values_http_would_not_accept():
    """Values a proxy or a hostile client could smuggle past httpx.

    Newlines and non-latin-1 text never reach the HTTP layer here
    (httpx refuses them), so they are checked on the function itself.
    """
    for offending in [
        "line1\nline2",
        "carriage\rreturn",
        "unicode-身份",
        "a" * 200,
        "\t",
        None,
    ]:
        headers = {} if offending is None else {"X-Request-ID": offending}
        request = SimpleNamespace(headers=headers)

        request_id = main.request_id_for(request)

        assert request_id != offending
        assert main.REQUEST_ID_PATTERN.fullmatch(request_id)


def _request_scoped_ids(recorder):
    """Ids recorded inside a request.

    Records emitted outside the middleware's context (for example the
    httpx client logging its own request) carry the default "-".
    """
    return [
        request_id
        for request_id in recorder.ids
        if request_id not in (None, "-")
    ]


def test_log_records_and_response_header_share_one_id(client):
    """The id a caller sees back is the id that was logged."""
    recorder = _RequestIdRecorder()
    root_logger = logging.getLogger()
    root_logger.addHandler(recorder)

    try:
        response = _login(client, {"X-Request-ID": "shared-id-42"})
    finally:
        root_logger.removeHandler(recorder)

    echoed = response.headers["X-Request-ID"]
    logged = _request_scoped_ids(recorder)

    assert echoed == "shared-id-42"
    assert logged
    assert all(request_id == echoed for request_id in logged)


def test_replaced_id_is_the_one_logged_not_the_offending_one(client):
    recorder = _RequestIdRecorder()
    root_logger = logging.getLogger()
    root_logger.addHandler(recorder)

    try:
        response = _login(client, {"X-Request-ID": "bad id here"})
    finally:
        root_logger.removeHandler(recorder)

    echoed = response.headers["X-Request-ID"]
    logged = _request_scoped_ids(recorder)

    assert "bad id here" not in echoed
    assert logged
    assert all(request_id == echoed for request_id in logged)
