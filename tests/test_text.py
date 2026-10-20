"""Tests for app.core.text.is_greeting.

Regression focus: English greetings used to be matched as raw
substrings, so ordinary sentences such as ``this is a test`` or
``which option`` were classified as greetings and the planner was
skipped. Greeting detection must stay token based while plain
Chinese and English greetings keep working.
"""

import pytest

from app.core.text import is_greeting


# ---------------------------------------------------------------------------
# Positive cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "你好",
        "您好",
        "嗨",
        "哈喽",
        "早上好",
        "下午好",
        "晚上好",
        "你好呀",
        "你好！",
        "hi",
        "Hi",
        "HI",
        "hi.",
        "hello",
        "Hello!",
        "hey",
        "hey there",
        "hello everyone",
        "hi all",
        "hello team",
    ],
)
def test_greetings_are_recognised(message):
    assert is_greeting(message) is True


# ---------------------------------------------------------------------------
# Negative cases — the substring regression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        # English words that merely *contain* a greeting substring.
        "this is a test",
        "which option",
        "history of china",
        "achievement unlocked",
        "chicken dinner",
        "his name is Tom",
        # Real requests must never be treated as greetings.
        "hi,帮我查一下差旅标准",
        "hello, what is the reimbursement policy",
        "你好，帮我查一下差旅标准",
        "hi 500",
        "请问 hi 是什么意思",
        # Chinese request text without any greeting substring.
        "一线城市住宿费标准是多少",
        # Content that is not a greeting at all.
        "123",
        "计算 100 + 20",
        "hello world",
    ],
)
def test_non_greetings_are_not_recognised(message):
    assert is_greeting(message) is False


@pytest.mark.parametrize("value", [None, "", "   ", 123, b"hi", ["hi"]])
def test_non_string_or_empty_input_is_not_a_greeting(value):
    assert is_greeting(value) is False


def test_regression_substring_false_positives():
    """The exact strings that used to slip through must stay False."""
    assert is_greeting("this is a test") is False
    assert is_greeting("which option") is False


def test_greeting_plus_request_is_not_a_pure_greeting():
    """A greeting followed by content must still reach the planner."""
    assert is_greeting("hi，帮我查一下差旅标准") is False
    assert is_greeting("你好，请说明报销流程") is False
