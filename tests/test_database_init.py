"""Secret-safe database bootstrap (P5).

Engine creation must never leak the DSN: a missing driver, an
unparseable URL or a construction failure has to surface as a
classified, credential-free error with the original context
suppressed, so the traceback that reaches the console is clean too.

Everything here is synthetic: no connection is attempted against a
real server, and every credential is a made-up string.
"""

import pytest
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import (
    ArgumentError,
    NoSuchModuleError,
    OperationalError,
)

from app.core.database import (
    DatabaseInitError,
    build_database_url,
    classify_database_error,
    create_engine_safely,
    describe_database_failure,
    verify_url_credentials,
)

FAKE_PASSWORD = "S3cr3t!@:/?#[]&+"
FAKE_DSN = "mysql+pymysql://root:S3cr3t@db.internal:3306/agent_db"


def test_password_with_special_characters_survives_the_url_round_trip():
    url = build_database_url("root", FAKE_PASSWORD, "db.internal", "3306")

    parsed = make_url(url)

    assert parsed.password == FAKE_PASSWORD
    assert verify_url_credentials(url, FAKE_PASSWORD)
    # The raw password never appears unencoded in the userinfo part.
    assert FAKE_PASSWORD not in url.split("@", 1)[0]


def test_database_name_defaults_to_agent_db():
    url = build_database_url("root", FAKE_PASSWORD, "db.internal", "3306")

    assert make_url(url).database == "agent_db"
    assert FAKE_PASSWORD not in url.split("@", 1)[0]


def test_database_name_can_be_configured_without_connecting():
    url = build_database_url(
        "root",
        FAKE_PASSWORD,
        "db.internal",
        "3306",
        database="agent_backend_dev",
    )

    parsed = make_url(url)
    assert parsed.database == "agent_backend_dev"
    assert parsed.password == FAKE_PASSWORD
    assert FAKE_PASSWORD not in url.split("@", 1)[0]


def test_build_database_url_encodes_the_username_too():
    url = build_database_url("ro@ot/x", "pw", "db.internal", "3306")

    assert make_url(url).username == "ro@ot/x"
    assert "ro@ot/x" not in url


def test_missing_driver_is_classified_and_hides_credentials():
    url = "mysql+nosuchdriver://root:S3cr3t@db.internal:3306/agent_db"

    with pytest.raises(DatabaseInitError) as exc_info:
        create_engine_safely(url)

    message = str(exc_info.value)

    assert "S3cr3t" not in message
    assert "db.internal" not in message
    assert "driver_missing" in message
    # No chained cause: the driver's own message and traceback are
    # suppressed, so nothing can print them at import time.
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__ is True


def test_create_engine_safely_hides_a_driver_message_that_embeds_a_dsn(
    monkeypatch,
):
    import app.core.database as database_module

    def exploding_engine(*args, **kwargs):
        raise RuntimeError(f"failed to connect using {FAKE_DSN}")

    monkeypatch.setattr(
        database_module,
        "_sqlalchemy_create_engine",
        exploding_engine,
    )

    with pytest.raises(DatabaseInitError) as exc_info:
        create_engine_safely(FAKE_DSN)

    message = str(exc_info.value)

    assert "S3cr3t" not in message
    assert "mysql+pymysql" not in message
    assert "db.internal" not in message
    assert exc_info.value.__cause__ is None


def test_classify_database_error_uses_types_only():
    assert (
        classify_database_error(NoSuchModuleError("dialect"))
        == "driver_missing"
    )
    assert classify_database_error(ModuleNotFoundError("pymysql")) == (
        "driver_missing"
    )
    assert classify_database_error(ArgumentError("bad url")) == "invalid_url"
    assert classify_database_error(ValueError("invalid literal")) == (
        "invalid_url"
    )
    assert classify_database_error(ConnectionError("refused")) == (
        "connection_failed"
    )
    assert classify_database_error(RuntimeError("boom")) == "unknown"


def test_describe_database_failure_redacts_and_classifies():
    exc = OperationalError(
        "SELECT 1",
        {},
        Exception("connect to mysql+pymysql://root:S3cr3t@db.internal:3306"),
    )

    description = describe_database_failure(exc)

    assert "S3cr3t" not in description
    assert "connection_failed" in description
    # Multi-line driver messages must not break the log line format.
    assert "\n" not in description


def test_describe_database_failure_truncates_long_messages():
    exc = RuntimeError("x" * 10_000)

    description = describe_database_failure(exc)

    assert len(description) < 400
