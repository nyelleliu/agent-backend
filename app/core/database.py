"""Secret-safe bootstrap for the SQLAlchemy engine.

Two rules drive this module:

1. Connection credentials must never reach a log line, an exception
   message or a traceback. The password is percent-encoded before it is
   placed in the URL (so a password containing ``@``, ``/`` or ``:``
   cannot break the parse), and a failure to *create* the engine is
   re-raised as a classified, one-line error with the original context
   suppressed (``from None``), so an import-time crash cannot echo the
   DSN back to the console.
2. Operators still need to diagnose the failure, so every error is
   mapped to a coarse category (driver missing, unparseable URL,
   connection refused, ...) that is safe to log and to report.

This module only covers engine *creation* and the classification of
errors raised while talking to the database. It does not promise that
no credential can ever appear anywhere: redaction of free-form text is
best-effort and lives in :mod:`app.core.errors`.
"""

from urllib.parse import quote

from sqlalchemy import create_engine as _sqlalchemy_create_engine
from sqlalchemy.engine.url import make_url

from app.core.errors import redact_secrets

__all__ = [
    "DatabaseInitError",
    "build_database_url",
    "classify_database_error",
    "create_engine_safely",
    "describe_database_failure",
    "verify_url_credentials",
]


class DatabaseInitError(RuntimeError):
    """The engine could not be created.

    Carries a coarse category only. It never embeds the URL, the
    password, or the original exception message.
    """

    error_category = "database_init_failed"


def build_database_url(
    user: str,
    password: str,
    host: str,
    port: str | int,
    database: str = "agent_db",
) -> str:
    """Build a SQLAlchemy MySQL URL with percent-encoded credentials.

    ``quote(..., safe="")`` is used rather than ``quote_plus``: inside
    the userinfo component a literal ``+`` means a plus sign, whereas
    ``quote_plus`` would turn a space into ``+`` and SQLAlchemy's
    ``unquote`` of the parsed password would then return the wrong
    value.
    """
    return (
        f"mysql+pymysql://"
        f"{quote(user, safe='')}:{quote(password, safe='')}"
        f"@{host}:{port}/{database}"
    )


def classify_database_error(exc: BaseException) -> str:
    """Map an exception to a coarse, non-revealing category.

    Only the exception type and its module are inspected — the message
    is never read, because driver messages are free-form text that may
    contain connection details.
    """
    name = type(exc).__name__
    module = type(exc).__module__ or ""

    if name == "NoSuchModuleError":
        return "driver_missing"

    if isinstance(exc, (ModuleNotFoundError, ImportError)):
        return "driver_missing"

    if name == "ArgumentError":
        return "invalid_url"

    if isinstance(exc, ValueError):
        # SQLAlchemy raises ValueError for e.g. a non-numeric port.
        return "invalid_url"

    if module.startswith(("sqlalchemy", "pymysql", "MySQLdb", "mysql")):
        if name in ("OperationalError", "InterfaceError"):
            return "connection_failed"

        return "database_error"

    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return "connection_failed"

    return "unknown"


def create_engine_safely(url: str, **kwargs):
    """``create_engine`` that fails with a classified, secret-free error.

    A missing driver, an unparseable DSN or any other construction
    failure is turned into :class:`DatabaseInitError`. The original
    exception is suppressed (``raise ... from None``) so the traceback
    that reaches the console contains this module's frame and the
    classified message — never the driver's message and never the URL.
    """
    try:
        return _sqlalchemy_create_engine(url, **kwargs)
    except Exception as exc:
        category = classify_database_error(exc)

        raise DatabaseInitError(
            f"database_init_failed category={category}"
        ) from None


def describe_database_failure(exc: BaseException) -> str:
    """One log-safe line for a database error raised at runtime."""
    category = classify_database_error(exc)
    # Redacted as a second line of defence: driver messages do not
    # normally contain the DSN, but they are third-party text and are
    # not held to any contract.
    detail = redact_secrets(str(exc)).replace("\n", " ")[:200]

    return f"category={category} detail={detail}"


def verify_url_credentials(url: str, password: str) -> bool:
    """True when ``url`` still carries ``password`` after parsing.

    Used by tests to prove that percent-encoding round-trips correctly
    through SQLAlchemy's URL parser. Never logs the values.
    """
    try:
        return make_url(url).password == password
    except Exception:
        return False
