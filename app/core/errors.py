"""Sanitized formatting of tool errors before they reach the LLM.

Exception text frequently carries infrastructure details: database
DSNs with credentials, internal hostnames, absolute file paths, or
third-party service messages. Relaying that to the model leaks
internal information and rarely helps it recover.

Only a small whitelist of input-error exception types keeps its
(redacted, truncated) message; every other exception is reduced to a
stable error category plus the current request id, so an operator can
find the full details in the server logs.

Used by AgentLoop, ToolRegistry, SkillRegistry and MCPToolRegistry so
every tool error follows the same rules.
"""

import re
import traceback


MAX_MESSAGE_CHARS = 300

# Exception types whose messages describe bad *input* to a tool or
# skill. That text helps the model correct its own arguments and says
# nothing about internal infrastructure. Note that
# json.JSONDecodeError and UnicodeDecodeError are ValueError
# subclasses, and KeyError / IndexError are LookupError subclasses.
SAFE_EXCEPTION_TYPES = (ValueError, TypeError, LookupError)


class ToolTimeoutError(RuntimeError):
    """A tool call exceeded its time budget.

    Carries an explicit category so the model-facing message can say
    "timeout" without exposing the underlying coroutine or transport.
    """

    error_category = "timeout_error"


_KEY_VALUE_SECRET = re.compile(
    r"(?i)\b(password|passwd|pwd|pass|secret|token|api[_-]?key"
    r"|access[_-]?token|authorization|auth)\b(\s*[:=]\s*)"
    # Never re-match an already-redacted value: the redaction markers
    # contain "]", which this rule would otherwise split off, and the
    # result would stop being idempotent.
    r"(?!\[REDACTED)"
    r"[\"']?[^\s\"',;)\]}]+"
)

_JWT = re.compile(
    r"\beyJ[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,}"
)

_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+")

# ``Authorization: token <value>`` / ``Authorization: Basic <base64>``
# — an HTTP auth scheme whose actual credential is the *next* token.
# Without this rule the key/value rule would consume the scheme word
# (``token``) and leave the credential in place.
_AUTH_SCHEME_CREDENTIAL = re.compile(
    r"(?i)\b(token|basic|digest|apikey)\s+([A-Za-z0-9._\-+/=]{8,})"
)

# Standalone API keys with no "key = ..." context around them, e.g.
# "upstream rejected sk-live-0a1b2c3d4e5f". Provider prefixes only —
# matching arbitrary long tokens would mangle normal log text.
_BARE_API_KEY = re.compile(
    r"\b(?:sk|pk|rk|ghp|glpat|xoxb|xoxa|xoxp|xoxs)"
    r"-[A-Za-z0-9_\-]{8,}\b"
)

_URL_WITH_SCHEME = re.compile(
    r"\b[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s'\"`<>,;)\]}]*"
)

_PRIVATE_IP = re.compile(
    r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3}"
    r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
    r"|127\.\d{1,3}\.\d{1,3}\.\d{1,3})\b"
)

_UNC_PATH = re.compile(r"\\\\[A-Za-z0-9_.\-]+\\[^\s'\"`<>,;)\]}]*")

_WINDOWS_PATH = re.compile(r"\b[A-Za-z]:\\[^\s'\"`<>,;)\]}]*")

# Forward-slash drive paths such as "/E:/agent-backend/..." (Git Bash
# and some Windows APIs report paths this way). The backslash rule
# above never matches them.
_DRIVE_PATH_FWD = re.compile(r"/?[A-Za-z]:/[^\s'\"`<>,;)\]}]*")

_POSIX_PATH = re.compile(
    r"(?<![\w.])/(?:home|Users|users|root|etc|var|opt|srv|tmp|usr|app"
    r"|mnt|media|data|logs)/[^\s'\"`<>,;)\]}]*"
)


def redact_secrets(value) -> str:
    """Remove credential-like content from text bound for any log.

    Best effort only — this is a set of regular expressions, not a
    secret scanner. It reliably handles the shapes that show up in
    this codebase (URL credentials, ``key=value`` pairs, ``Bearer``
    tokens, JWTs, provider-prefixed API keys, private IPs, absolute
    paths) and it is deliberately *not* claimed to catch everything:
    see :mod:`tests.test_redaction_boundaries` for the exact
    contract, including the shapes it is known to miss (for example
    ``password hunter2`` with a bare space, or a credential with no
    keyword around it). Never treat a redacted string as proof that no
    secret is present.
    """
    if value is None:
        return ""

    text = str(value)

    # Order matters. Token-shaped rules run FIRST: the key/value rule
    # below would otherwise consume only the scheme word of
    # ``Authorization: Bearer <token>`` (its value stops at the first
    # space) and leave the actual token in place.
    text = _JWT.sub("[REDACTED]", text)
    text = _BEARER.sub("[REDACTED]", text)
    text = _AUTH_SCHEME_CREDENTIAL.sub(r"\1=[REDACTED]", text)
    text = _KEY_VALUE_SECRET.sub(lambda m: f"{m.group(1)}=[REDACTED]", text)
    text = _URL_WITH_SCHEME.sub("[REDACTED_URL]", text)
    text = _BARE_API_KEY.sub("[REDACTED]", text)
    text = _PRIVATE_IP.sub("[REDACTED_HOST]", text)
    text = _UNC_PATH.sub("[REDACTED_PATH]", text)
    text = _WINDOWS_PATH.sub("[REDACTED_PATH]", text)
    text = _DRIVE_PATH_FWD.sub("[REDACTED_PATH]", text)
    text = _POSIX_PATH.sub("[REDACTED_PATH]", text)

    return text


def _truncate(message: str) -> str:
    if len(message) <= MAX_MESSAGE_CHARS:
        return message

    return message[:MAX_MESSAGE_CHARS] + "...[truncated]"


def _safe_request_id(request_id=None) -> str | None:
    """Current request id, or None when unavailable.

    Never raises: building an error message must not fail just
    because there is no active request context.
    """
    if request_id is None:
        try:
            from app.core.logging_config import request_id_var

            request_id = request_id_var.get()
        except Exception:
            return None

    text = str(request_id).strip()

    if not text or text == "-":
        return None

    return text[:64]


def _generic_category(exc) -> str:
    """A stable, non-revealing category for non-input errors."""
    category = getattr(exc, "error_category", None)

    if isinstance(category, str) and category:
        return category

    module = type(exc).__module__ or ""

    if module.startswith(("sqlalchemy", "pymysql", "mysql", "psycopg")):
        return "database_error"

    if module.startswith("redis"):
        return "cache_error"

    if module.startswith(
        ("httpx", "httpcore", "openai", "urllib3", "aiohttp", "requests")
    ):
        return "upstream_api_error"

    if isinstance(exc, TimeoutError):
        return "timeout_error"

    if isinstance(exc, OSError):
        return "io_error"

    return "internal_error"


def format_tool_error(exc, *, tool_name=None, request_id=None) -> str:
    """Build the error text that is safe to hand back to the model.

    Whitelisted input errors keep a redacted, truncated message.
    Everything else becomes a short category plus request id, with no
    trace of the original exception text.
    """
    label = f"tool '{tool_name}'" if tool_name else "the tool"
    request_suffix = ""

    safe_request_id = _safe_request_id(request_id)

    if safe_request_id:
        request_suffix = f" (request_id={safe_request_id})"

    if isinstance(exc, BaseException) and isinstance(
        exc, SAFE_EXCEPTION_TYPES
    ):
        message = _truncate(redact_secrets(str(exc)).strip())

        if message:
            return f"Error: {message}"

        return (
            f"Error: {label} rejected the input "
            f"({type(exc).__name__})"
        )

    return (
        f"Error: {label} failed due to a "
        f"{_generic_category(exc)}{request_suffix}; "
        f"retry or report the request id"
    )


def log_tool_error(logger, exc, *, tool_name=None) -> None:
    """Record the full exception server-side, credentials removed.

    The log keeps the diagnostic detail (and stack trace) that the
    model-facing message deliberately drops, but never contains API
    keys, passwords, tokens or DSN credentials.
    """
    exc_type = type(exc).__name__
    details = _truncate(redact_secrets(str(exc)).strip()) or "(no message)"

    if isinstance(exc, SAFE_EXCEPTION_TYPES):
        logger.warning(
            "tool_input_error tool=%s type=%s error=%s",
            tool_name or "-",
            exc_type,
            details,
        )
        return

    try:
        traceback_text = traceback.format_exc()
    except Exception:
        traceback_text = ""

    if traceback_text and "NoneType: None" not in traceback_text:
        details = (
            f"{details}\n"
            f"{_truncate(redact_secrets(traceback_text))}"
        )

    logger.error(
        "tool_execution_failed tool=%s type=%s error=%s",
        tool_name or "-",
        exc_type,
        details,
    )
