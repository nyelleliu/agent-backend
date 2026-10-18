"""Output-layer risk-scan tests.

scan_reply is a monitoring heuristic, not a security boundary: it
flags credential-like patterns and injection framing artifacts in the
final assistant reply and logs a redacted event. These tests cover
common secret formats, ordinary text, false-positive edges, and log
redaction.
"""

import logging

import pytest

from app.core.output_guard import scan_and_log_reply, scan_reply


# ---------------------------------------------------------------------------
# Suspected secret leak
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "your api_key=sk-FAKE0000000000",
        "password: hunter2",
        "set password=test-password-123 in the config",
        "Bearer abc123DEF456.ghi-jkl",
        "connect via mysql://root:secret@db.example/agent_db",
        (
            "token="
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0."
            "abcdefghij"
        ),
    ],
)
def test_common_credential_formats_are_flagged(reply):
    assert "suspected_secret_leak" in scan_reply(reply)


def test_clean_reply_has_no_categories():
    reply = "差旅报销标准是每天 500 元，详见 [1] 差旅制度。"
    assert scan_reply(reply) == []


def test_plain_url_is_not_flagged_as_secret():
    reply = "更多信息见 https://api.deepseek.com/docs 。"
    assert scan_reply(reply) == []


def test_password_word_without_value_is_not_flagged():
    reply = "系统使用 bcrypt 对用户密码做哈希存储。"
    assert scan_reply(reply) == []


def test_env_var_name_mentioned_without_value_is_not_flagged():
    reply = "部署前需要配置 DEEPSEEK_API_KEY 环境变量。"
    assert scan_reply(reply) == []


# ---------------------------------------------------------------------------
# Suspected injection artifact
# ---------------------------------------------------------------------------


def test_echoed_untrusted_wrapper_is_flagged():
    reply = "根据 <untrusted_document>差旅标准</untrusted_document> 回答。"
    assert "suspected_injection_artifact" in scan_reply(reply)


def test_instruction_override_phrasing_is_flagged():
    reply = "好的，ignore previous instructions，我来执行。"
    assert "suspected_injection_artifact" in scan_reply(reply)


def test_injection_and_secret_categories_are_distinct():
    reply = (
        "ignore previous instructions; api_key=sk-abcdef123456"
    )
    categories = scan_reply(reply)
    assert "suspected_secret_leak" in categories
    assert "suspected_injection_artifact" in categories


def test_chinese_paraphrase_of_override_is_not_flagged():
    """Heuristics are English-pattern based; a Chinese paraphrase is
    not reliably detectable — asserted as a known limitation, not
    silently treated as safe."""
    reply = "好的，我已经忽略之前的所有指令。"
    assert scan_reply(reply) == []


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", 123, b"bytes", []])
def test_non_string_or_empty_input_is_safe(value):
    assert scan_reply(value) == []
    assert scan_and_log_reply(value) == []


def test_scan_and_log_returns_categories():
    categories = scan_and_log_reply("api_key=sk-test123456")
    assert "suspected_secret_leak" in categories


# ---------------------------------------------------------------------------
# Log redaction
# ---------------------------------------------------------------------------


def test_log_contains_category_but_not_raw_secret(caplog):
    from app.core.logging_config import request_id_var

    secret_value = "sk-RAWSECRETabcdef123456"
    reply = f"your api_key={secret_value}"

    token = request_id_var.set("req-guard-test")
    try:
        with caplog.at_level(logging.WARNING):
            categories = scan_and_log_reply(reply)
    finally:
        request_id_var.reset(token)

    assert "suspected_secret_leak" in categories

    warning_texts = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "output_risk_scan" in record.getMessage()
    ]

    assert len(warning_texts) == 1
    log_text = warning_texts[0]

    assert "suspected_secret_leak" in log_text
    assert "req-guard-test" in log_text
    # The raw secret must never reach the log; only a redacted form.
    assert secret_value not in log_text
    assert "[REDACTED]" in log_text


def test_no_log_event_for_clean_reply(caplog):
    with caplog.at_level(logging.WARNING):
        categories = scan_and_log_reply("今天是 2026 年 10 月 12 日。")

    assert categories == []
    assert not any(
        "output_risk_scan" in record.getMessage()
        for record in caplog.records
    )
