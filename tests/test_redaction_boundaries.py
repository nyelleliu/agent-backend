"""Documented boundaries of the secret-redaction rules (P9).

This file is the contract for :func:`app.core.errors.redact_secrets`.
It records what the rules catch and — just as importantly — what they
are known *not* to catch, so nobody has to guess. The implementation
is a handful of regular expressions; it is not a secret scanner, and
passing every test here is not evidence that a redacted string
contains no credential.
"""

from app.core.errors import format_tool_error, redact_secrets

BEARER_TOKEN = "shpat_0123456789abcdef"
OPAQUE_TOKEN = "ghp_abcdefghijklmnopqrstuvwxyz012345"
JWT = (
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NSJ9."
    "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
)


# ---------------------------------------------------------------------------
# Shapes the rules cover
# ---------------------------------------------------------------------------


def test_bearer_token_after_an_authorization_keyword_is_fully_removed():
    """Regression: the key/value rule used to eat only "Bearer".

    Its value stopped at the first space, so the token itself stayed
    in the redacted text.
    """
    result = redact_secrets(f"Authorization: Bearer {BEARER_TOKEN}")

    assert BEARER_TOKEN not in result
    assert "[REDACTED]" in result
    assert "Authorization" in result


def test_bearer_token_with_no_keyword_around_it_is_removed():
    result = redact_secrets(f"upstream rejected Bearer {OPAQUE_TOKEN}")

    assert OPAQUE_TOKEN not in result


def test_token_after_a_colon_is_removed():
    result = redact_secrets(f"authorization: token {OPAQUE_TOKEN}")

    assert OPAQUE_TOKEN not in result


def test_jwt_is_removed():
    result = redact_secrets(f"jwt={JWT} rejected")

    assert JWT not in result


def test_key_equals_value_form_is_removed():
    assert "hunter2" not in redact_secrets("password=hunter2")
    assert "hunter2" not in redact_secrets("password : hunter2")
    assert "hunter2" not in redact_secrets("api_key='hunter2'")


def test_credentials_inside_a_url_are_removed():
    result = redact_secrets(
        "connect to mysql+pymysql://root:hunter2@db.internal:3306/agent_db"
    )

    assert "hunter2" not in result
    assert "db.internal" not in result


def test_provider_prefixed_key_without_any_keyword_is_removed():
    result = redact_secrets("upstream rejected sk-live-0a1b2c3d4e5f")

    assert "sk-live-0a1b2c3d4e5f" not in result


def test_private_and_loopback_addresses_are_removed():
    assert "10.1.2.3" not in redact_secrets("timeout talking to 10.1.2.3:3306")
    assert "192.168.0.5" not in redact_secrets("peer 192.168.0.5 reset")
    assert "127.0.0.1" not in redact_secrets("dial 127.0.0.1:6379")


def test_absolute_paths_are_removed():
    assert "/var/lib/mysql" not in redact_secrets(
        "cannot open /var/lib/mysql/agent_db"
    )
    assert "C:\\Users" not in redact_secrets("path C:\\Users\\lenovo\\x")


# ---------------------------------------------------------------------------
# Shapes the rules deliberately do NOT cover — pinned down so a future
# change is a conscious decision, not an accident.
# ---------------------------------------------------------------------------


def test_password_followed_by_a_bare_space_is_not_redacted():
    """Known gap: no separator means no key/value match.

    Accepting a bare space would also mangle innocent sentences such
    as "the password policy requires 12 characters", and the redacted
    text is handed to the model as tool feedback. The safer rule for
    that case is the one in format_tool_error: drop the message.
    """
    assert "hunter2" in redact_secrets("password hunter2")


def test_credential_with_no_keyword_around_it_is_not_redacted():
    """Known gap: a bare hostname or token with no context passes through."""
    text = "host db-01.corp.internal is down"

    assert "db-01.corp.internal" in redact_secrets(text)


# ---------------------------------------------------------------------------
# How the two consumers behave around those gaps
# ---------------------------------------------------------------------------


def test_model_facing_message_drops_anything_not_whitelisted():
    """A non-input exception never relays its message at all."""
    result = format_tool_error(RuntimeError("password hunter2"))

    assert "hunter2" not in result
    assert "internal_error" in result


def test_model_facing_message_redacts_whitelisted_input_errors():
    result = format_tool_error(ValueError("unsupported api_key=hunter2"))

    assert "hunter2" not in result
    assert "unsupported" in result


def test_redaction_is_idempotent_on_already_redacted_text():
    once = redact_secrets("password=hunter2")

    assert redact_secrets(once) == once
